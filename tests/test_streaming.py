"""Compressed stream adapter: exact next batches, masks and RNG isolation."""

import json
import os

import numpy as np
import pytest
import torch

from data.corpus import CorpusError, CorpusReader
from data.olmo_streaming import (
    OFFICIAL_TRAIN_SHARDS,
    HFStreamingTokenizer,
    StreamingOLMoLoader,
    build_official_streaming_loader,
    official_compressed_input_identity,
)
from data.pack import PackedDataLoader


class ByteTokenizer:
    bos_id, eos_id, pad_id = 258, 256, 257
    byte_lengths = np.array([1] * 256 + [0, 0, 0])

    def encode_batch(self, texts, threads=None):
        return [list(text.encode("utf-8")) for text in texts]


def make_loader(path, work_dir, identity=None):
    packed = PackedDataLoader(
        ByteTokenizer(),
        [path],
        batch_size=2,
        sequence_length=16,
        doc_batch=4,
        loop=True,
        on_bad_utf8="raise",
        mask_separator=True,
    )
    return StreamingOLMoLoader(
        packed, work_dir=work_dir, input_identity=identity or {"fixture": "compressed-v1"}
    )


def test_compressed_resume_next_batches_and_global_rng(tmp_path):
    zstd = pytest.importorskip("zstandard")
    path = tmp_path / "train.jsonl.zst"
    raw = b"\n".join(
        json.dumps({"text": "你好 doc " + str(i) + " abcdef"}, ensure_ascii=False).encode()
        for i in range(30)
    )
    path.write_bytes(zstd.ZstdCompressor().compress(raw))
    loader = make_loader(path, tmp_path / "first")
    loader.reshuffle(1)
    rng = torch.get_rng_state().clone()
    it = iter(loader)
    next(it)
    next(it)
    state = loader.state_dict()
    expected = [next(it) for _ in range(3)]
    restored = make_loader(path, tmp_path / "restored")
    restored.load_state_dict(state)
    restored.reshuffle(1)
    ri = iter(restored)
    for want in expected:
        got = next(ri)
        assert torch.equal(got["input_ids"], want["input_ids"])
        assert torch.equal(got["labels"], want["labels"])
        assert not torch.any(got["labels"] == 256)  # EOS loss stays masked
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.batches_processed == loader.batches_processed
    assert not path.with_suffix("").exists()


def test_identity_change_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text":"abcd"}\n')
    loader = make_loader(path, tmp_path / "one")
    loader.reshuffle(1)
    other = make_loader(path, tmp_path / "two", {"fixture": "other"})
    with pytest.raises(ValueError, match="identity changed"):
        other.load_state_dict(loader.state_dict())


def assert_state_equal(a, b):
    if isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def write_compressed(path, texts):
    zstd = pytest.importorskip("zstandard")
    path.write_bytes(
        zstd.ZstdCompressor().compress(
            b"\n".join(json.dumps({"text": t}, ensure_ascii=True).encode() for t in texts)
        )
    )
    return path


def test_all_state_resume_across_corpus_passes_and_iterator_reentry(tmp_path):
    # A prefetch spans several complete corpus passes; no documents are lost
    # and counters must not grow just because the prefetch buffer is restored.
    path = write_compressed(tmp_path / "train.jsonl.zst", ["ab", "", "中文"])
    continuous = make_loader(path, tmp_path / "continuous")
    continuous.reshuffle(1)
    for _ in range(7):
        next(iter(continuous))
    checkpoint = continuous.state_dict()
    restored = make_loader(path, tmp_path / "restored")
    restored.load_state_dict(checkpoint)
    assert_state_equal(checkpoint, restored.state_dict())
    # A second serialization immediately after load must preserve everything.
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
    "field,value",
    [
        ("loop", False),
        ("doc_batch", 2),
        ("drop_last", False),
        ("mask_separator", False),
        ("sequence_length", 8),
        ("batch_size", 1),
    ],
)
def test_actual_packing_change_rejected_even_with_same_user_identity(tmp_path, field, value):
    path = write_compressed(tmp_path / "train.jsonl.zst", ["abcd"])
    original = make_loader(path, tmp_path / "one")
    kwargs = dict(
        batch_size=2,
        sequence_length=16,
        doc_batch=4,
        loop=True,
        on_bad_utf8="raise",
        mask_separator=True,
    )
    kwargs[field] = value
    packed = PackedDataLoader(ByteTokenizer(), [path], **kwargs)
    changed = StreamingOLMoLoader(
        packed, work_dir=tmp_path / "two", input_identity={"fixture": "compressed-v1"}
    )
    with pytest.raises(ValueError, match="identity changed"):
        changed.load_state_dict(original.state_dict())


def make_hf_tokenizer(directory):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({"<eos>": 256, "<pad>": 257})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", model_max_length=100000
    )
    hf.save_pretrained(directory)
    return hf


def test_hf_original_text_and_literal_eos_labels(tmp_path):
    hf = make_hf_tokenizer(tmp_path / "tokenizer")
    adapter = HFStreamingTokenizer(hf)
    texts = ["prefix <eos> suffix", "汉字🙂\n\t", ""]
    encoded = adapter.encode_batch(texts)
    assert encoded == hf(texts, add_special_tokens=False)["input_ids"]
    literal_eos = encoded[0].index(adapter.eos_id)
    assert literal_eos > 0
    path = write_compressed(tmp_path / "train.jsonl.zst", texts)
    packed = PackedDataLoader(
        adapter, [path], batch_size=1, sequence_length=32, loop=True, on_bad_utf8="raise"
    )
    batch = next(packed.batches())
    assert batch.labels[0, literal_eos - 1] == adapter.eos_id
    assert batch.labels[0, len(encoded[0]) - 1] == -100


def test_official_all_eight_hashes_and_source_identity(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for i, name in enumerate(OFFICIAL_TRAIN_SHARDS):
        write_compressed(data / name, [str(i) + " xyz"])
    make_hf_tokenizer(tmp_path / "tokenizer")
    args = dict(
        batch_size=2,
        sequence_length=17,
        work_dir=tmp_path / "work",
        doc_batch=5,
        model_vocab_size=258,
    )
    loader = build_official_streaming_loader(data, tmp_path / "tokenizer", **args)
    assert [r["name"] for r in loader.input_manifest] == list(OFFICIAL_TRAIN_SHARDS)
    assert all(len(r["sha256"]) == 64 for r in loader.input_manifest)
    assert loader.packed.loop and loader.packed.reader.on_blank == "raise"
    assert all(p.suffix == ".zst" for p in loader.packed.reader.paths)
    assert not list(data.glob("*.jsonl"))
    loader.reshuffle(1)
    it = iter(loader)
    next(it)
    saved = loader.state_dict()
    wanted = [next(it) for _ in range(5)]
    restored = build_official_streaming_loader(data, tmp_path / "tokenizer", **args)
    restored.load_state_dict(saved)
    for batch in wanted:
        got = next(iter(restored))
        assert torch.equal(batch["input_ids"], got["input_ids"])
        assert torch.equal(batch["labels"], got["labels"])
    assert_state_equal(restored.state_dict(), loader.state_dict())
    write_compressed(data / OFFICIAL_TRAIN_SHARDS[-1], ["different"])
    changed = build_official_streaming_loader(data, tmp_path / "tokenizer", **args)
    with pytest.raises(ValueError, match="identity changed"):
        changed.load_state_dict(saved)
    (data / OFFICIAL_TRAIN_SHARDS[-1]).unlink()
    with pytest.raises(ValueError, match="eight official"):
        official_compressed_input_identity(data)


@pytest.mark.parametrize("raw", [b"\n", b'{"text":"\\ud800"}\n', b'{"id":1}\n'])
def test_strict_no_filter_errors(tmp_path, raw):
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


def test_empty_loop_fails_instead_of_hanging(tmp_path):
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


def test_dp2_document_union_and_complete_resume(tmp_path):
    paths = []
    expected_documents = {}
    for shard_index, name in enumerate(OFFICIAL_TRAIN_SHARDS):
        texts = [f"shard{shard_index} document{line} 中文" for line in range(16)]
        paths.append(write_compressed(tmp_path / name, texts))
        expected_documents.update({(name, line): text for line, text in enumerate(texts)})
    observed = {}
    saved_states = []

    def ranked(rank, world_size=2):
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
            key = (document.shard, document.index)
            assert key not in observed  # No document duplicated across ranks.
            assert document.index % 2 == rank
            observed[key] = document.text
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
        for _ in range(100):  # Includes traversal into the next complete pass.
            expected = next(iterator)
            actual = next(iter(restored))
            assert torch.equal(expected["input_ids"], actual["input_ids"])
            assert torch.equal(expected["labels"], actual["labels"])
        assert_state_equal(loader.state_dict(), restored.state_dict())
        assert loader.packed.reader.state()["epoch"] >= 1
    assert observed == expected_documents
    with pytest.raises(ValueError, match="rank and world_size"):
        ranked(1).load_state_dict(saved_states[0])
    with pytest.raises(ValueError, match="rank and world_size"):
        ranked(0, 4).load_state_dict(saved_states[0])


def test_official_builder_explicit_dp_rank(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in OFFICIAL_TRAIN_SHARDS:
        write_compressed(data / name, [f"line {i} 中" for i in range(16)])
    make_hf_tokenizer(tmp_path / "tokenizer")
    loader = build_official_streaming_loader(
        data,
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


# Opt-in, offline integration: set a JSON object mapping model names to
# existing local tokenizer directories. No test downloads data or weights.
_REAL_TOKENIZERS = json.loads(os.environ.get("PRETRAIN_TEST_HF_TOKENIZER_DIRS", "{}"))


@pytest.mark.parametrize("name,directory", list(_REAL_TOKENIZERS.items()) or [(None, None)])
def test_real_local_hf_packing_and_complete_restore(tmp_path, name, directory):
    if directory is None:
        pytest.skip("set PRETRAIN_TEST_HF_TOKENIZER_DIRS to existing local tokenizer directories")
    adapter = HFStreamingTokenizer.from_local(directory)
    literal = adapter.hf.eos_token
    texts = ["", "hello 中文🙂\n", f"prefix {literal} suffix", "long 汉字 " * 12000]
    ids = adapter.hf(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
    assert adapter.encode_batch(texts) == ids
    assert len(ids[-1]) > 8192
    assert adapter.eos_id in ids[2]
    # Independent flatten/shift reference: literal EOS inside the text stays
    # trainable; the separator appended after each document is masked.
    flat, mask = [], []
    for doc in ids:
        flat.extend(doc + [adapter.eos_id])
        mask.extend([True] * len(doc) + [False])
    data = tmp_path / "data"
    data.mkdir()
    for shard in OFFICIAL_TRAIN_SHARDS:
        write_compressed(data / shard, texts)
    kwargs = dict(batch_size=2, sequence_length=64, doc_batch=2, work_dir=tmp_path / "work")
    loader = build_official_streaming_loader(data, directory, **kwargs)
    loader.reshuffle(1)
    iterator = iter(loader)
    first = next(iterator)
    ref_inputs = torch.tensor([flat[:64], flat[64:128]])
    ref_labels = torch.tensor(
        [
            [flat[i] if mask[i] else -100 for i in range(1, 65)],
            [flat[i] if mask[i] else -100 for i in range(65, 129)],
        ]
    )
    assert torch.equal(first["input_ids"], ref_inputs), name
    assert torch.equal(first["labels"], ref_labels), name
    for _ in range(2):
        next(iterator)
    saved = loader.state_dict()
    expected = [next(iterator) for _ in range(5)]
    restored = build_official_streaming_loader(data, directory, **kwargs)
    restored.load_state_dict(saved)
    assert_state_equal(saved, restored.state_dict())
    for want in expected:
        got = next(iter(restored))
        assert torch.equal(want["input_ids"], got["input_ids"]), name
        assert torch.equal(want["labels"], got["labels"]), name
    assert_state_equal(loader.state_dict(), restored.state_dict())
    assert not list(data.glob("*.jsonl"))
