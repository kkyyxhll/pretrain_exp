"""Adapt compressed document packing to the training loader interface."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from olmo_core.data.data_loader import DataLoaderBase
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.data.pack import DEFAULT_DOCUMENT_BATCH_SIZE, PackedDataLoader, PackedState
from src.data.tokenization import hf_token_bytes

OFFICIAL_TRAIN_SHARDS = tuple(f"dolma3_mix_50B-{i:05d}.jsonl.zst" for i in range(8))
INPUT_HASH_CHUNK_BYTES = 8 << 20


class ShardIdentity(TypedDict):
    """Record an original compressed shard's name, byte size, and content hash."""

    name: str
    size: int
    sha256: str


class TokenizerIdentity(TypedDict):
    """Record the unchanged tokenizer backend and document encoding policy."""

    kind: Literal["hf-fast-original-text-v1"]
    backend_sha256: str
    eos_id: int
    pad_id: int
    bos_id: int | None
    vocabulary_size: int
    add_special_tokens: Literal[False]
    truncation: Literal[False]
    padding: Literal[False]


class OfficialInputIdentity(TypedDict):
    """Bind the official compressed shards to the fixed tokenizer."""

    official_compressed_shards: list[ShardIdentity]
    tokenizer: TokenizerIdentity


class PackingIdentity(TypedDict, total=False):
    """Describe checkpoint-compatible packing, with rank fields only for multiple ranks."""

    batch_size: int
    sequence_length: int
    loop: bool
    drop_last: bool
    doc_batch: int
    mask_separator: bool
    bos_id: int | None
    eos_id: int
    pad_id: int
    on_bad_utf8: Literal["raise"]
    on_blank: Literal["skip", "raise"]
    ordered_shards: list[str]
    rank: int
    world_size: int


class LoaderIdentity(TypedDict):
    """Describe inputs and packing before their canonical JSON hash is computed."""

    inputs: OfficialInputIdentity | dict[str, str]
    packing: PackingIdentity


class StreamingLoaderState(TypedDict):
    """Describe the established full streaming-loader checkpoint contract."""

    format: Literal["packed-stream-olmo-v2"]
    identity: str
    rank: int
    world_size: int
    epoch: int | None
    batches_processed: int
    packed: PackedState


class TensorBatch(TypedDict):
    """Expose only the two tensors accepted by the training model."""

    input_ids: torch.Tensor
    labels: torch.Tensor


class BatchAccounting(TypedDict):
    """Record scored tokens and padding without forwarding metadata to the model."""

    scored_tokens: int
    padded_positions: int


def official_compressed_input_identity(data_dir: str | Path) -> list[ShardIdentity]:
    """Hash all eight original compressed inputs without writing unpacked files."""
    data_directory = Path(data_dir)
    shard_paths = [data_directory / shard_name for shard_name in OFFICIAL_TRAIN_SHARDS]
    missing_shard_names = [
        shard_path.name for shard_path in shard_paths if not shard_path.is_file()
    ]
    if missing_shard_names:
        raise ValueError(
            f"All eight official compressed training shards are required: {missing_shard_names}"
        )
    else:
        pass
    shard_identities: list[ShardIdentity] = []
    for shard_path in shard_paths:
        initial_file_stat = shard_path.stat()
        content_hasher = hashlib.sha256()
        with shard_path.open("rb") as compressed_stream:
            while compressed_bytes := compressed_stream.read(INPUT_HASH_CHUNK_BYTES):
                content_hasher.update(compressed_bytes)
        final_file_stat = shard_path.stat()
        if (initial_file_stat.st_size, initial_file_stat.st_mtime_ns) != (
            final_file_stat.st_size,
            final_file_stat.st_mtime_ns,
        ):
            raise ValueError(f"Compressed input changed while hashing: {shard_path.name}")
        else:
            shard_identities.append(
                {
                    "name": shard_path.name,
                    "size": final_file_stat.st_size,
                    "sha256": content_hasher.hexdigest(),
                }
            )
    return shard_identities


class HFStreamingTokenizer:
    """Encode original documents; byte lengths are metadata, not the BPB denominator."""

    def __init__(self, huggingface_tokenizer: PreTrainedTokenizerFast) -> None:
        self.hf: PreTrainedTokenizerFast = huggingface_tokenizer
        tokenizer_eos_id = huggingface_tokenizer.eos_token_id
        if not isinstance(tokenizer_eos_id, int):
            raise ValueError("The HF tokenizer must define one EOS token id")
        else:
            self.eos_id: int = tokenizer_eos_id
        self.pad_id: int = (
            huggingface_tokenizer.pad_token_id
            if huggingface_tokenizer.pad_token_id is not None
            else self.eos_id
        )
        self.bos_id: int | None = huggingface_tokenizer.bos_token_id
        self.byte_lengths: NDArray[np.int64] = np.asarray(
            [len(token_bytes) for token_bytes in hf_token_bytes(huggingface_tokenizer)],
            dtype=np.int64,
        )
        # note (kkyyxhll): Literal added-token spellings retain their original source bytes.
        for token_piece, token_id in huggingface_tokenizer.get_added_vocab().items():
            self.byte_lengths[token_id] = len(token_piece.encode("utf-8"))
        self.identity: TokenizerIdentity = {
            "kind": "hf-fast-original-text-v1",
            "backend_sha256": hashlib.sha256(
                huggingface_tokenizer.backend_tokenizer.to_str().encode()
            ).hexdigest(),
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "vocabulary_size": len(huggingface_tokenizer),
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        }

    @classmethod
    def from_local(cls, directory: str | Path) -> HFStreamingTokenizer:
        """Load fixed local tokenizer assets without resolving remote code or weights."""
        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError("A serializable fast HF tokenizer is required")
        else:
            return cls(tokenizer)

    def encode_batch(self, texts: Sequence[str], threads: int | None = None) -> list[list[int]]:
        """Encode document text without truncation, padding, or inserted special tokens."""
        # note (kkyyxhll): The packer supplies threads; the fast tokenizer owns its Rayon pool.
        return self.hf(
            list(texts),
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]


def build_official_streaming_loader(
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    batch_size: int,
    sequence_length: int,
    work_dir: str | Path,
    doc_batch: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    threads: int | None = None,
    model_vocab_size: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> StreamingOLMoLoader:
    """Build an unfiltered looping compressed stream for an explicit training rank."""
    shard_identities = official_compressed_input_identity(data_dir)
    tokenizer = HFStreamingTokenizer.from_local(tokenizer_dir)
    if model_vocab_size is not None and max(tokenizer.hf.get_vocab().values()) >= model_vocab_size:
        raise ValueError("The tokenizer can emit ids outside the model vocabulary")
    else:
        pass
    packed_loader = PackedDataLoader(
        tokenizer,
        [Path(data_dir) / shard_identity["name"] for shard_identity in shard_identities],
        batch_size=batch_size,
        sequence_length=sequence_length,
        loop=True,
        rank=rank,
        world_size=world_size,
        drop_last=True,
        doc_batch=doc_batch,
        threads=threads,
        on_bad_utf8="raise",
        on_blank="raise",
        mask_separator=True,
    )
    streaming_loader = StreamingOLMoLoader(
        packed_loader,
        work_dir=work_dir,
        input_identity={
            "official_compressed_shards": shard_identities,
            "tokenizer": tokenizer.identity,
        },
    )
    streaming_loader.input_manifest = shard_identities
    return streaming_loader


class StreamingOLMoLoader(DataLoaderBase):
    """Preserve compressed-stream ownership and complete state across training resumes."""

    def __init__(
        self,
        packed: PackedDataLoader,
        *,
        work_dir: str | Path,
        input_identity: OfficialInputIdentity | dict[str, str],
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        rank = packed.reader.rank if rank is None else rank
        world_size = packed.reader.world_size if world_size is None else world_size
        if (rank, world_size) != (packed.reader.rank, packed.reader.world_size):
            raise ValueError("Adapter rank/world_size must match the packed reader")
        elif packed.reader.on_bad_utf8 != "raise":
            raise ValueError(
                "Competition streams must raise on invalid UTF-8, never filter records"
            )
        else:
            pass
        super().__init__(
            work_dir=work_dir,
            global_batch_size=packed.batch_size * packed.sequence_length * world_size,
            dp_rank=rank,
            dp_world_size=world_size,
        )
        self.packed: PackedDataLoader = packed
        self.input_manifest: list[ShardIdentity] = []
        self.identity_details: LoaderIdentity = {
            "inputs": input_identity,
            "packing": {
                "batch_size": packed.batch_size,
                "sequence_length": packed.sequence_length,
                "loop": packed.loop,
                "drop_last": packed.drop_last,
                "doc_batch": packed.doc_batch,
                "mask_separator": packed.mask_separator,
                "bos_id": packed.bos_id,
                "eos_id": packed.eos_id,
                "pad_id": packed.pad_id,
                "on_bad_utf8": "raise",
                "on_blank": packed.reader.on_blank,
                "ordered_shards": [shard_path.name for shard_path in packed.reader.paths],
            },
        }
        if world_size != 1:
            # note (kkyyxhll): Single-rank identity hashes predate rank ownership fields.
            self.identity_details["packing"].update(rank=rank, world_size=world_size)
        else:
            pass
        self.input_identity: str = hashlib.sha256(
            json.dumps(self.identity_details, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        self.latest_batch_accounting: BatchAccounting | None = None

    @property
    def total_batches(self) -> None:
        """Leave total batches unknown for an endlessly looping document stream."""
        return None

    def reshuffle(self, epoch: int | None = None) -> None:
        """Advance the trainer epoch without rewinding the compressed document cursor."""
        self._epoch = epoch if epoch is not None else (self._epoch or 0) + 1

    def state_dict(self) -> StreamingLoaderState:
        """Save the identity, ownership, trainer counters, and finite prefetch state."""
        return {
            "format": "packed-stream-olmo-v2",
            "identity": self.input_identity,
            "rank": self.dp_rank,
            "world_size": self.dp_world_size,
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
            "packed": self.packed.state_dict(),
        }

    def load_state_dict(self, state: StreamingLoaderState) -> None:
        """Reject incompatible ownership, inputs, or packing before complete restoration."""
        if (state.get("rank", 0), state.get("world_size", 1)) != (self.dp_rank, self.dp_world_size):
            raise ValueError("Full resume requires unchanged rank and world_size")
        elif state["format"] != "packed-stream-olmo-v2" or state["identity"] != self.input_identity:
            raise ValueError("Compressed inputs/tokenizer/packing identity changed")
        elif state["packed"]["batch_size"] != self.packed.batch_size:
            raise ValueError("Full resume requires unchanged batch size")
        else:
            self._epoch = state["epoch"]
            self.batches_processed = state["batches_processed"]
            self.packed.load_state_dict(state["packed"])

    def _iter_batches(self) -> Iterator[TensorBatch]:
        """Yield only tensors while retaining accounting in the loader."""
        for batch in self.packed.batches():
            self.latest_batch_accounting = {
                "scored_tokens": batch.scored_tokens,
                "padded_positions": batch.padded_positions,
            }
            yield {
                "input_ids": torch.from_numpy(batch.inputs),
                "labels": torch.from_numpy(batch.labels),
            }

    def global_num_tokens_in_batch(self, batch: TensorBatch) -> int:
        """Count global input positions for the unchanged parallel world size."""
        return batch["input_ids"].numel() * self.dp_world_size

    def get_mock_batch(self) -> TensorBatch:
        """Refuse unledgered dry-run work or consumption of the real stream."""
        raise NotImplementedError
