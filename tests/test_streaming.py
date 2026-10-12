"""Compressed stream adapter: exact next batches, masks and RNG isolation."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.data.corpus import CorpusError, CorpusReader
from src.data.olmo_streaming import (
    OFFICIAL_TRAIN_SHARDS,
    HFStreamingTokenizer,
    StreamingLoaderState,
    StreamingOLMoLoader,
    build_official_streaming_loader,
    official_compressed_input_identity,
)
from src.data.pack import PackedDataLoader

zstandard = pytest.importorskip("zstandard")

type CheckpointValue = (
    bool
    | int
    | str
    | None
    | NDArray[np.int32 | np.bool_]
    | dict[str, CheckpointValue]
    | list[CheckpointValue]
    | tuple[CheckpointValue, ...]
)


class ByteTokenizer:
    bos_id: ClassVar[int] = 258
    eos_id: ClassVar[int] = 256
    pad_id: ClassVar[int] = 257
    byte_lengths: ClassVar[NDArray[np.int64]] = np.array([1] * 256 + [0, 0, 0])

    def encode_batch(self, texts: Sequence[str], threads: int | None = None) -> list[list[int]]:
        # note (kkyyxhll): The tokenizer interface accepts thread limits; byte encoding is serial.
        return [list(text.encode("utf-8")) for text in texts]


def make_loader(
    corpus_path: Path, work_dir: Path, input_identity: dict[str, str] | None = None
) -> StreamingOLMoLoader:
    packed = PackedDataLoader(
        ByteTokenizer(),
        [corpus_path],
        batch_size=2,
        sequence_length=16,
        doc_batch=4,
        loop=True,
        on_bad_utf8="raise",
        mask_separator=True,
    )
    return StreamingOLMoLoader(
        packed,
        work_dir=work_dir,
        input_identity=input_identity or {"fixture": "compressed-v1"},
    )


def test_compressed_resume_next_batches_and_global_rng(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl.zst"
    raw = b"\n".join(
        json.dumps({"text": f"你好 doc {i} abcdef"}, ensure_ascii=False).encode() for i in range(30)
    )
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    loader = make_loader(path, tmp_path / "first")
    loader.reshuffle(1)
    random_state = torch.get_rng_state().clone()
    batches = iter(loader)
    next(batches)
    next(batches)
    state = loader.state_dict()
    expected_batches = [next(batches) for _ in range(3)]
    restored = make_loader(path, tmp_path / "restored")
    restored.load_state_dict(state)
    restored.reshuffle(1)
    restored_batches = iter(restored)
    for expected_batch in expected_batches:
        restored_batch = next(restored_batches)
        assert torch.equal(restored_batch["input_ids"], expected_batch["input_ids"])
        assert torch.equal(restored_batch["labels"], expected_batch["labels"])
        assert not torch.any(restored_batch["labels"] == ByteTokenizer.eos_id)
    assert torch.equal(random_state, torch.get_rng_state())
    assert restored.batches_processed == loader.batches_processed
    assert not path.with_suffix("").exists()


def test_identity_change_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"text":"abcd"}\n')
    loader = make_loader(path, tmp_path / "one")
    loader.reshuffle(1)
    other = make_loader(path, tmp_path / "two", {"fixture": "other"})
    with pytest.raises(ValueError, match="identity changed"):
        other.load_state_dict(loader.state_dict())


def assert_state_equal(expected_state: CheckpointValue, actual_state: CheckpointValue) -> None:
    if isinstance(expected_state, np.ndarray):
        assert isinstance(actual_state, np.ndarray)
        assert np.array_equal(expected_state, actual_state)
    elif isinstance(expected_state, dict):
        assert isinstance(actual_state, dict)
        assert expected_state.keys() == actual_state.keys()
        for state_field in expected_state:
            assert_state_equal(expected_state[state_field], actual_state[state_field])
    elif isinstance(expected_state, (list, tuple)):
        assert isinstance(actual_state, (list, tuple))
        assert len(expected_state) == len(actual_state)
        for expected_value, actual_value in zip(expected_state, actual_state):
            assert_state_equal(expected_value, actual_value)
    else:
        assert expected_state == actual_state


def write_compressed(corpus_path: Path, texts: list[str]) -> Path:
    corpus_path.write_bytes(
        zstandard.ZstdCompressor().compress(
            b"\n".join(json.dumps({"text": text}, ensure_ascii=True).encode() for text in texts)
        )
    )
    return corpus_path


def test_all_state_resume_across_corpus_passes_and_iterator_reentry(tmp_path: Path) -> None:
    path = write_compressed(tmp_path / "train.jsonl.zst", ["ab", "", "中文"])
    continuous = make_loader(path, tmp_path / "continuous")
    continuous.reshuffle(1)
    for _ in range(7):
        next(iter(continuous))
    checkpoint = continuous.state_dict()
    restored = make_loader(path, tmp_path / "restored")
    restored.load_state_dict(checkpoint)
    assert_state_equal(checkpoint, restored.state_dict())
    restored_twice = make_loader(path, tmp_path / "restored-twice")
    restored_twice.load_state_dict(restored.state_dict())
    for _ in range(9):
        expected = next(iter(continuous))
        actual = next(iter(restored_twice))
        assert torch.equal(expected["input_ids"], actual["input_ids"])
        assert torch.equal(expected["labels"], actual["labels"])
        assert_state_equal(continuous.state_dict(), restored_twice.state_dict())
    assert continuous.packed.reader.state()["epoch"] > 1


@pytest.mark.parametrize(
    "packing_field,packing_value",
    [
        ("loop", False),
        ("doc_batch", 2),
        ("drop_last", False),
        ("mask_separator", False),
        ("sequence_length", 8),
        ("batch_size", 1),
    ],
)
def test_actual_packing_change_rejected_even_with_same_user_identity(
    tmp_path: Path,
    packing_field: Literal[
        "loop", "doc_batch", "drop_last", "mask_separator", "sequence_length", "batch_size"
    ],
    packing_value: int | bool,
) -> None:
    path = write_compressed(tmp_path / "train.jsonl.zst", ["abcd"])
    original = make_loader(path, tmp_path / "one")
    packing_options = dict(
        batch_size=2,
        sequence_length=16,
        doc_batch=4,
        loop=True,
        on_bad_utf8="raise",
        mask_separator=True,
    )
    packing_options[packing_field] = packing_value
    packed = PackedDataLoader(ByteTokenizer(), [path], **packing_options)
    changed = StreamingOLMoLoader(
        packed, work_dir=tmp_path / "two", input_identity={"fixture": "compressed-v1"}
    )
    with pytest.raises(ValueError, match="identity changed"):
        changed.load_state_dict(original.state_dict())


def make_hf_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    vocabulary = {
        character: token_id
        for token_id, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    vocabulary.update({"<eos>": 256, "<pad>": 257})
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", model_max_length=100000
    )
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer


def test_hf_original_text_and_literal_eos_labels(tmp_path: Path) -> None:
    tokenizer = make_hf_tokenizer(tmp_path / "tokenizer")
    adapter = HFStreamingTokenizer(tokenizer)
    texts = ["prefix <eos> suffix", "汉字🙂\n\t", ""]
    encoded = adapter.encode_batch(texts)
    assert encoded == tokenizer(texts, add_special_tokens=False)["input_ids"]
    literal_eos = encoded[0].index(adapter.eos_id)
    assert literal_eos > 0
    path = write_compressed(tmp_path / "train.jsonl.zst", texts)
    packed = PackedDataLoader(
        adapter, [path], batch_size=1, sequence_length=32, loop=True, on_bad_utf8="raise"
    )
    batch = next(packed.batches())
    assert batch.labels[0, literal_eos - 1] == adapter.eos_id
    assert batch.labels[0, len(encoded[0]) - 1] == -100


def test_official_all_eight_hashes_and_source_identity(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, name in enumerate(OFFICIAL_TRAIN_SHARDS):
        write_compressed(data_dir / name, [f"{i} xyz"])
    make_hf_tokenizer(tmp_path / "tokenizer")
    loader_options = dict(
        batch_size=2,
        sequence_length=17,
        work_dir=tmp_path / "work",
        doc_batch=5,
        model_vocab_size=258,
    )
    loader = build_official_streaming_loader(data_dir, tmp_path / "tokenizer", **loader_options)
    assert [shard_identity["name"] for shard_identity in loader.input_manifest] == list(
        OFFICIAL_TRAIN_SHARDS
    )
    assert all(len(shard_identity["sha256"]) == 64 for shard_identity in loader.input_manifest)
    assert loader.packed.loop and loader.packed.reader.on_blank == "raise"
    assert all(path.suffix == ".zst" for path in loader.packed.reader.paths)
    assert not list(data_dir.glob("*.jsonl"))
    loader.reshuffle(1)
    batches = iter(loader)
    next(batches)
    saved = loader.state_dict()
    expected_batches = [next(batches) for _ in range(5)]
    restored = build_official_streaming_loader(data_dir, tmp_path / "tokenizer", **loader_options)
    restored.load_state_dict(saved)
    for expected_batch in expected_batches:
        restored_batch = next(iter(restored))
        assert torch.equal(expected_batch["input_ids"], restored_batch["input_ids"])
        assert torch.equal(expected_batch["labels"], restored_batch["labels"])
    assert_state_equal(restored.state_dict(), loader.state_dict())
    write_compressed(data_dir / OFFICIAL_TRAIN_SHARDS[-1], ["different"])
    changed = build_official_streaming_loader(data_dir, tmp_path / "tokenizer", **loader_options)
    with pytest.raises(ValueError, match="identity changed"):
        changed.load_state_dict(saved)
    (data_dir / OFFICIAL_TRAIN_SHARDS[-1]).unlink()
    with pytest.raises(ValueError, match="eight official"):
        official_compressed_input_identity(data_dir)


@pytest.mark.parametrize("raw", [b"\n", b'{"text":"\\ud800"}\n', b'{"id":1}\n'])
def test_strict_no_filter_errors(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "train.jsonl"
    path.write_bytes(raw)
    packed = PackedDataLoader(
        ByteTokenizer(),
        [path],
        batch_size=1,
        sequence_length=1,
        loop=True,
        on_bad_utf8="raise",
        on_blank="raise",
    )
    with pytest.raises(CorpusError):
        next(packed.batches())


def test_empty_loop_fails_instead_of_hanging(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_bytes(b"")
    packed = PackedDataLoader(
        ByteTokenizer(),
        [path],
        batch_size=1,
        sequence_length=1,
        loop=True,
        on_bad_utf8="raise",
        on_blank="raise",
    )
    with pytest.raises(CorpusError, match="no documents"):
        next(packed.batches())


def test_dp2_document_union_and_complete_resume(tmp_path: Path) -> None:
    paths: list[Path] = []
    expected_documents: dict[tuple[str, int], str] = {}
    for shard_index, name in enumerate(OFFICIAL_TRAIN_SHARDS):
        texts = [f"shard{shard_index} document{line} 中文" for line in range(16)]
        paths.append(write_compressed(tmp_path / name, texts))
        expected_documents.update({(name, line): text for line, text in enumerate(texts)})
    observed_documents: dict[tuple[str, int], str] = {}
    saved_states: list[StreamingLoaderState] = []

    def ranked(rank: int, world_size: int = 2) -> StreamingOLMoLoader:
        packed = PackedDataLoader(
            ByteTokenizer(),
            paths,
            batch_size=2,
            sequence_length=13,
            rank=rank,
            world_size=world_size,
            doc_batch=3,
            on_bad_utf8="raise",
            on_blank="raise",
            loop=True,
        )
        return StreamingOLMoLoader(
            packed,
            work_dir=tmp_path / f"rank{rank}",
            input_identity={"fixture": "eight-shards-v1"},
            rank=rank,
            world_size=world_size,
        )

    for rank in range(2):
        reader = CorpusReader(
            paths, rank=rank, world_size=2, loop=False, on_bad_utf8="raise", on_blank="raise"
        )
        for document in reader.documents():
            document_location = (document.shard, document.index)
            assert document_location not in observed_documents
            assert document.index % 2 == rank
            observed_documents[document_location] = document.text
        loader = ranked(rank)
        loader.reshuffle(1)
        iterator = iter(loader)
        for _ in range(5):
            batch = next(iterator)
        assert loader.dp_rank == rank and loader.dp_world_size == 2
        assert loader.global_batch_size == 2 * 13 * 2
        assert loader.rank_batch_size == 2 * 13
        assert loader.global_num_tokens_in_batch(batch) == 2 * 13 * 2
        checkpoint = loader.state_dict()
        saved_states.append(checkpoint)
        restored = ranked(rank)
        restored.load_state_dict(checkpoint)
        assert_state_equal(checkpoint, restored.state_dict())
        for _ in range(100):
            expected = next(iterator)
            actual = next(iter(restored))
            assert torch.equal(expected["input_ids"], actual["input_ids"])
            assert torch.equal(expected["labels"], actual["labels"])
        assert_state_equal(loader.state_dict(), restored.state_dict())
        assert loader.packed.reader.state()["epoch"] >= 1
    assert observed_documents == expected_documents
    with pytest.raises(ValueError, match="rank and world_size"):
        ranked(1).load_state_dict(saved_states[0])
    with pytest.raises(ValueError, match="rank and world_size"):
        ranked(0, 4).load_state_dict(saved_states[0])


def test_official_builder_explicit_dp_rank(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in OFFICIAL_TRAIN_SHARDS:
        write_compressed(data_dir / name, [f"line {i} 中" for i in range(16)])
    make_hf_tokenizer(tmp_path / "tokenizer")
    loader = build_official_streaming_loader(
        data_dir,
        tmp_path / "tokenizer",
        batch_size=2,
        sequence_length=8,
        work_dir=tmp_path / "work",
        rank=1,
        world_size=2,
    )
    assert loader.packed.reader.rank == loader.dp_rank == 1
    assert loader.packed.reader.world_size == loader.dp_world_size == 2
    assert loader.global_batch_size == 32
    assert loader.identity_details["packing"]["rank"] == 1
    with pytest.raises(ValueError, match="must match"):
        StreamingOLMoLoader(
            loader.packed, work_dir=tmp_path / "other", input_identity={}, rank=0, world_size=2
        )


# note (kkyyxhll): Optional integration uses local tokenizers and never downloads model assets.
TOKENIZER_DIRECTORY_VALUE: object = json.loads(
    os.environ.get("PRETRAIN_TEST_HF_TOKENIZER_DIRS", "{}")
)
if not isinstance(TOKENIZER_DIRECTORY_VALUE, dict):
    raise ValueError(
        "PRETRAIN_TEST_HF_TOKENIZER_DIRS must map tokenizer names to local directories"
    )
else:
    REAL_TOKENIZERS: dict[str, str] = {}
    for tokenizer_name, tokenizer_dir in TOKENIZER_DIRECTORY_VALUE.items():
        if isinstance(tokenizer_name, str) and isinstance(tokenizer_dir, str):
            REAL_TOKENIZERS[tokenizer_name] = tokenizer_dir
        else:
            raise ValueError(
                "PRETRAIN_TEST_HF_TOKENIZER_DIRS names and directories must be strings"
            )


@pytest.mark.parametrize(
    "tokenizer_name,tokenizer_dir", list(REAL_TOKENIZERS.items()) or [(None, None)]
)
def test_real_local_hf_packing_and_complete_restore(
    tmp_path: Path, tokenizer_name: str | None, tokenizer_dir: str | None
) -> None:
    if tokenizer_dir is None:
        pytest.skip("set PRETRAIN_TEST_HF_TOKENIZER_DIRS to existing local tokenizer directories")
    else:
        pass
    adapter = HFStreamingTokenizer.from_local(tokenizer_dir)
    literal = adapter.hf.eos_token
    texts = ["", "hello 中文🙂\n", f"prefix {literal} suffix", "long 汉字 " * 12000]
    token_ids = adapter.hf(texts, add_special_tokens=False, truncation=False, padding=False)[
        "input_ids"
    ]
    assert adapter.encode_batch(texts) == token_ids
    assert len(token_ids[-1]) > 8192
    assert adapter.eos_id in token_ids[2]
    concatenated_tokens: list[int] = []
    text_mask: list[bool] = []
    for document_tokens in token_ids:
        concatenated_tokens.extend(document_tokens + [adapter.eos_id])
        text_mask.extend([True] * len(document_tokens) + [False])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for shard in OFFICIAL_TRAIN_SHARDS:
        write_compressed(data_dir / shard, texts)
    loader_options = dict(batch_size=2, sequence_length=64, doc_batch=2, work_dir=tmp_path / "work")
    loader = build_official_streaming_loader(data_dir, tokenizer_dir, **loader_options)
    loader.reshuffle(1)
    iterator = iter(loader)
    first = next(iterator)
    reference_inputs = torch.tensor([concatenated_tokens[:64], concatenated_tokens[64:128]])
    reference_labels = torch.tensor(
        [
            [concatenated_tokens[i] if text_mask[i] else -100 for i in range(1, 65)],
            [concatenated_tokens[i] if text_mask[i] else -100 for i in range(65, 129)],
        ]
    )
    assert torch.equal(first["input_ids"], reference_inputs), tokenizer_name
    assert torch.equal(first["labels"], reference_labels), tokenizer_name
    for _ in range(2):
        next(iterator)
    saved = loader.state_dict()
    expected = [next(iterator) for _ in range(5)]
    restored = build_official_streaming_loader(data_dir, tokenizer_dir, **loader_options)
    restored.load_state_dict(saved)
    assert_state_equal(saved, restored.state_dict())
    for expected_batch in expected:
        restored_batch = next(iter(restored))
        assert torch.equal(expected_batch["input_ids"], restored_batch["input_ids"]), tokenizer_name
        assert torch.equal(expected_batch["labels"], restored_batch["labels"]), tokenizer_name
    assert_state_equal(loader.state_dict(), restored.state_dict())
    assert not list(data_dir.glob("*.jsonl"))
