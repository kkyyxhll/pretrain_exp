"""OLMo adapter for the existing compressed-corpus PackedDataLoader.

No unpacked corpus or token cache is written. Packing, EOS masks and byte
accounting remain the responsibility of data.pack.PackedDataLoader.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from olmo_core.data.data_loader import DataLoaderBase

from .pack import PackedDataLoader

OFFICIAL_TRAIN_SHARDS = tuple(f"dolma3_mix_50B-{i:05d}.jsonl.zst" for i in range(8))


def official_compressed_input_identity(data_dir):
    """Fingerprint all eight compressed inputs without creating unpacked files.

    Returned identities contain names, lengths and actual content hashes, not
    machine-specific paths. Hashing checks that no file changed during the read.
    """
    root = Path(data_dir)
    paths = [root / name for name in OFFICIAL_TRAIN_SHARDS]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"All eight official compressed training shards are required: {missing}")
    identities = []
    for path in paths:
        before = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Compressed input changed while hashing: {path.name}")
        identities.append({"name": path.name, "size": after.st_size, "sha256": h.hexdigest()})
    return identities


class HFStreamingTokenizer:
    """Existing HF tokenizer adapted to PackedDataLoader, with no text edits.

    The HF encoder sees each original document with add_special_tokens=False.
    Literal special-token strings remain literal input and are encoded exactly
    as HF does; only the extra separator appended by the packer is masked.
    Per-token byte lengths are engineering metadata, not the official BPB
    denominator (which requires original-document byte accounting).
    """

    def __init__(self, hf):
        from .tokenization import hf_token_bytes

        self.hf = hf
        self.eos_id = hf.eos_token_id
        if not isinstance(self.eos_id, int):
            raise ValueError("The HF tokenizer must define one EOS token id")
        self.pad_id = hf.pad_token_id if hf.pad_token_id is not None else self.eos_id
        self.bos_id = hf.bos_token_id
        self.byte_lengths = np.asarray([len(b) for b in hf_token_bytes(hf)], dtype=np.int64)
        # An added-token spelling can occur in the original document. Unlike
        # the artificial separator, it retains its literal source bytes here.
        for piece, index in hf.get_added_vocab().items():
            self.byte_lengths[index] = len(piece.encode("utf-8"))
        backend = getattr(hf, "backend_tokenizer", None)
        if backend is None:
            raise ValueError("A serializable fast HF tokenizer is required")
        self.identity = {
            "kind": "hf-fast-original-text-v1",
            "backend_sha256": hashlib.sha256(backend.to_str().encode()).hexdigest(),
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "vocabulary_size": len(hf),
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        }

    @classmethod
    def from_local(cls, directory):
        from transformers import AutoTokenizer

        return cls(
            AutoTokenizer.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False
            )
        )

    def encode_batch(self, texts, threads=None):
        del threads  # HF fast tokenizers manage their own Rayon thread pool.
        return self.hf(
            list(texts),
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]


def build_official_streaming_loader(
    data_dir,
    tokenizer_dir,
    *,
    batch_size,
    sequence_length,
    work_dir,
    doc_batch=128,
    threads=None,
    model_vocab_size=None,
    rank=0,
    world_size=1,
):
    """Build a no-filter, endlessly looping stream for an explicit DP rank.

    Use a cumulative *step* stopping condition; OLMo's outer epoch stays 1 while
    the packer's corpus cursor records actual zero-based corpus passes. This
    helper performs no downloads and never resolves to unpacked files.
    """
    inputs = official_compressed_input_identity(data_dir)
    tokenizer = HFStreamingTokenizer.from_local(tokenizer_dir)
    if model_vocab_size is not None and max(tokenizer.hf.get_vocab().values()) >= model_vocab_size:
        raise ValueError("The tokenizer can emit ids outside the model vocabulary")
    packed = PackedDataLoader(
        tokenizer,
        [Path(data_dir) / p["name"] for p in inputs],
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
    loader = StreamingOLMoLoader(
        packed,
        work_dir=work_dir,
        input_identity={"official_compressed_shards": inputs, "tokenizer": tokenizer.identity},
    )
    loader.input_manifest = inputs
    return loader


class StreamingOLMoLoader(DataLoaderBase):
    def __init__(self, packed, *, work_dir, input_identity, rank=None, world_size=None):
        rank = packed.reader.rank if rank is None else rank
        world_size = packed.reader.world_size if world_size is None else world_size
        if (rank, world_size) != (packed.reader.rank, packed.reader.world_size):
            raise ValueError("Adapter rank/world_size must match the packed reader")
        if packed.reader.on_bad_utf8 != "raise":
            raise ValueError(
                "Competition streams must raise on invalid UTF-8, never filter records"
            )
        super().__init__(
            work_dir=work_dir,
            global_batch_size=packed.batch_size * packed.sequence_length * world_size,
            dp_rank=rank,
            dp_world_size=world_size,
        )
        self.packed = packed
        self.identity_details = {
            "inputs": input_identity,
            "packing": {
                k: getattr(packed, k)
                for k in (
                    "batch_size",
                    "sequence_length",
                    "loop",
                    "drop_last",
                    "doc_batch",
                    "mask_separator",
                    "bos_id",
                    "eos_id",
                    "pad_id",
                )
            },
        }
        self.identity_details["packing"].update(
            on_bad_utf8=packed.reader.on_bad_utf8,
            on_blank=packed.reader.on_blank,
            ordered_shards=[p.name for p in packed.reader.paths],
        )
        if world_size != 1:
            # Retain existing single-device identities, while DP ownership is
            # an essential part of every multi-rank checkpoint's identity.
            self.identity_details["packing"].update(rank=rank, world_size=world_size)
        self.input_identity = hashlib.sha256(
            json.dumps(self.identity_details, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        self.latest_batch_accounting = None

    @property
    def total_batches(self):
        return None

    def reshuffle(self, epoch=None, **kwargs):
        # The existing stream preserves document order. This hook must not
        # rewind a restored compressed stream or silently add a new sampler.
        self._epoch = epoch if epoch is not None else (self._epoch or 0) + 1

    def state_dict(self):
        return {
            "format": "packed-stream-olmo-v2",
            "identity": self.input_identity,
            "rank": self.dp_rank,
            "world_size": self.dp_world_size,
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
            "packed": self.packed.state_dict(),
        }

    def load_state_dict(self, state):
        if (state.get("rank", 0), state.get("world_size", 1)) != (self.dp_rank, self.dp_world_size):
            raise ValueError("Full resume requires unchanged rank and world_size")
        if state["format"] != "packed-stream-olmo-v2" or state["identity"] != self.input_identity:
            raise ValueError("Compressed inputs/tokenizer/packing identity changed")
        if state["packed"]["batch_size"] != self.packed.batch_size:
            raise ValueError("Full resume requires unchanged batch size")
        self._epoch = state["epoch"]
        self.batches_processed = state["batches_processed"]
        self.packed.load_state_dict(state["packed"])

    def _iter_batches(self):
        for batch in self.packed.batches():
            self.latest_batch_accounting = {
                "scored_tokens": batch.scored_tokens,
                "padded_positions": batch.padded_positions,
            }
            # Do not pass accounting metadata as Transformer.forward kwargs.
            yield {
                "input_ids": torch.from_numpy(batch.inputs),
                "labels": torch.from_numpy(batch.labels),
            }

    def global_num_tokens_in_batch(self, batch):
        return batch["input_ids"].numel() * self.dp_world_size

    def get_mock_batch(self):
        # A dry-run must not consume the real stream or create unledgered work.
        raise NotImplementedError
