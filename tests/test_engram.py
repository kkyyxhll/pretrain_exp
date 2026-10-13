"""Engram contracts against the demo hash and packed-document training boundaries."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.arch.memory.engram import Engram, EngramConfig, build_token_map, load_token_map


def tiny_config(**changes: object) -> EngramConfig:
    config = EngramConfig(
        layer_ids=(1, 4),
        table_sizes=(11, 17),
        heads=2,
        head_dim=2,
        kernel_size=3,
        dilation=2,
        pad_id=0,
        eos_id=11,
        seed=19,
    )
    return replace(config, **changes)


def tiny_memory(
    config: EngramConfig | None = None,
    *,
    layer_id: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Engram:
    return Engram(
        4,
        layer_id,
        config or tiny_config(),
        torch.tensor([0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10]),
        device=device,
        dtype=dtype,
    )


def reference_primes(config: EngramConfig) -> dict[int, np.ndarray]:
    """Allocate the demo's head tables without calling production hash helpers."""
    allocated: set[int] = set()
    layouts: dict[int, np.ndarray] = {}
    for layer_id in config.layer_ids:
        layout: list[int] = []
        for table_size in config.table_sizes:
            candidate = table_size
            for _ in range(config.heads):
                while candidate in allocated or any(
                    candidate % divisor == 0 for divisor in range(2, candidate)
                ):
                    candidate += 1
                layout.append(candidate)
                allocated.add(candidate)
                candidate += 1
        layouts[layer_id] = np.array(layout, dtype=np.int64)
    return layouts


def reference_hash(
    raw_ids: np.ndarray,
    token_map: np.ndarray,
    config: EngramConfig,
    layer_id: int,
) -> np.ndarray:
    """Literal per-position XOR reference, including left padding and raw EOS reset."""
    compressed_count = int(token_map.max()) + 1
    bound = (np.iinfo(np.int64).max // compressed_count) // 2
    rng = np.random.default_rng(config.seed + 10007 * layer_id)
    multipliers = rng.integers(0, max(1, bound), config.max_ngram_size, dtype=np.int64)
    multipliers = multipliers * 2 + 1
    primes = reference_primes(config)[layer_id]
    addresses = np.empty((*raw_ids.shape, config.num_hash_heads), dtype=np.int64)
    for row_index, row in enumerate(raw_ids):
        document_start = 0
        for position, raw_id in enumerate(row):
            if position and row[position - 1] == config.eos_id:
                document_start = position
            mixed = token_map[raw_id] * multipliers[0]
            for distance in range(1, config.max_ngram_size):
                context_position = position - distance
                context_id = (
                    token_map[row[context_position]]
                    if context_position >= document_start
                    else token_map[config.pad_id]
                )
                mixed ^= context_id * multipliers[distance]
                for head in range(config.heads):
                    head_index = (distance - 1) * config.heads + head
                    addresses[row_index, position, head_index] = mixed % primes[head_index]
    return addresses


@pytest.mark.parametrize("layer_id", [1, 4])
def test_hash_matches_numpy_demo_at_row_and_document_starts(layer_id: int) -> None:
    memory = tiny_memory(layer_id=layer_id)
    raw_ids = torch.tensor([[1, 4, 5, 11, 2, 3, 6], [9, 11, 11, 7, 8, 1, 2]])
    expected = reference_hash(raw_ids.numpy(), memory.token_map.numpy(), memory.config, layer_id)
    np.testing.assert_array_equal(memory.hash_ids(raw_ids).numpy(), expected)
    assert torch.equal(memory.hash_ids(raw_ids[:, :1]), memory.hash_ids(raw_ids)[:, :1])


def test_hash_table_layout_is_disjoint_and_reproducible_across_layers() -> None:
    first = tiny_memory()
    second = tiny_memory(layer_id=4)
    expected = reference_primes(first.config)
    np.testing.assert_array_equal(first.primes.numpy(), expected[1])
    np.testing.assert_array_equal(second.primes.numpy(), expected[4])
    all_primes = torch.cat((first.primes, second.primes))
    assert torch.unique(all_primes).numel() == all_primes.numel()
    assert first.embedding.num_embeddings == int(first.primes.sum())
    assert torch.equal(first.offsets, torch.tensor([0, 11, 24, 41]))
    clone = tiny_memory()
    assert torch.equal(first.multipliers, clone.multipliers)
    assert not torch.equal(first.multipliers, second.multipliers)


class VocabularyOnlyTokenizer:
    """Existing tokenizer vocabulary; no fit, encode, or training-text interface."""

    pieces = (
        "Hello",
        "HELLO",
        " héllo ",
        "Ｈｅｌｌｏ",
        "Straße",
        "straße",
        "  ",
        "\t",
        "\n",
        "a \t b",
        "A   B",
        "é",
        "e",
        "\ufffd",
        "\ufffd",
        "<eos>",
    )

    def __len__(self) -> int:
        return len(self.pieces)

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        assert len(ids) == 1
        return self.pieces[ids[0]]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {13: "ĠÃ", 14: "ĠÂ"}.get(token_id, self.pieces[token_id])


def test_token_map_normalizes_vocabulary_without_merging_broken_utf8_pieces() -> None:
    token_map = build_token_map(VocabularyOnlyTokenizer())
    assert token_map.dtype == torch.int64
    assert token_map.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7]
    assert token_map[13] != token_map[14]


def test_token_map_load_owns_storage_and_rejects_other_array_formats(tmp_path: Path) -> None:
    path = tmp_path / "map.npy"
    np.save(path, np.array([0, 1, 1, 2], dtype=np.int64), allow_pickle=False)
    loaded = load_token_map(path)
    loaded[0] = 2
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), [0, 1, 1, 2])
    for invalid in (np.array([[0, 1]], dtype=np.int64), np.array([0, 1], dtype=np.int32)):
        np.save(path, invalid, allow_pickle=False)
        with pytest.raises(ValueError, match="one-dimensional|int64"):
            load_token_map(path)
    np.save(path, np.array(["unsafe"], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_token_map(path)


@pytest.mark.parametrize("gate_variant", ["dot", "signed_sqrt"])
def test_suffix_tokens_and_hidden_states_cannot_change_prefix(gate_variant: str) -> None:
    torch.manual_seed(7)
    memory = tiny_memory(tiny_config(gate_variant=gate_variant))
    with torch.no_grad():
        memory.conv_weight.fill_(0.3)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    hidden = torch.randn(1, ids.shape[1], 4)
    changed_ids = ids.clone()
    changed_hidden = hidden.clone()
    changed_ids[:, 4:] = torch.tensor([9, 8, 11])
    changed_hidden[:, 4:] *= -7
    assert torch.equal(memory.hash_ids(ids)[:, :4], memory.hash_ids(changed_ids)[:, :4])
    assert torch.equal(memory(hidden, ids)[:, :4], memory(changed_hidden, changed_ids)[:, :4])


def test_raw_eos_resets_hash_and_nonzero_dilated_convolution() -> None:
    torch.manual_seed(23)
    memory = tiny_memory()
    with torch.no_grad():
        memory.conv_weight.copy_(torch.tensor([0.4, -0.3, 0.2]).expand(4, 1, 3))
    suffix_ids = torch.tensor([[2, 4, 6, 8, 1]])
    suffix_hidden = torch.randn(1, 5, 4)
    before_ids = torch.tensor([[9, 8, 7, 11]])
    before_hidden = torch.randn(1, 4, 4) * 4
    packed_ids = torch.cat((before_ids, suffix_ids), dim=1)
    packed_hidden = torch.cat((before_hidden, suffix_hidden), dim=1)
    assert torch.equal(memory.hash_ids(packed_ids)[:, 4:], memory.hash_ids(suffix_ids))
    torch.testing.assert_close(
        memory(packed_hidden, packed_ids)[:, 4:],
        memory(suffix_hidden, suffix_ids),
        atol=0,
        rtol=0,
    )
    # The first token must forget both nonzero taps at distances two and four.
    no_boundary = packed_ids.clone()
    no_boundary[:, 3] = 10
    assert not torch.equal(
        memory(packed_hidden, no_boundary)[:, 4:], memory(suffix_hidden, suffix_ids)
    )


def test_explicit_document_segments_reset_context_without_eos() -> None:
    memory = tiny_memory()
    with torch.no_grad():
        memory.conv_weight.fill_(0.2)
    ids = torch.tensor([[9, 8, 7, 2, 3, 4]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 1]])
    hidden = torch.randn(1, 6, 4)
    assert torch.equal(memory.hash_ids(ids, segments)[:, 3:], memory.hash_ids(ids[:, 3:]))
    torch.testing.assert_close(
        memory(hidden, ids, segments)[:, 3:],
        memory(hidden[:, 3:], ids[:, 3:]),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("gate_variant", ["dot", "signed_sqrt"])
def test_all_memory_branches_and_hidden_states_receive_finite_gradients(gate_variant: str) -> None:
    torch.manual_seed(11)
    memory = tiny_memory(tiny_config(gate_variant=gate_variant))
    with torch.no_grad():
        memory.conv_weight.normal_(std=0.2)
    ids = torch.tensor([[1, 2, 3, 4, 6, 7], [8, 7, 6, 5, 2, 1]])
    hidden = torch.randn(2, 6, 4, requires_grad=True)
    target = torch.randn_like(hidden)
    (memory(hidden, ids) - target).square().mean().backward()
    for name, parameter in memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad)


@pytest.mark.parametrize("gate_variant", ["dot", "signed_sqrt"])
def test_gate_hidden_gradient_matches_finite_differences_away_from_zero(gate_variant: str) -> None:
    torch.manual_seed(17)
    config = tiny_config(
        table_sizes=(5, 7), heads=1, head_dim=2, kernel_size=2, gate_variant=gate_variant
    )
    memory = Engram(2, 1, config, torch.arange(12), dtype=torch.float64)
    with torch.no_grad():
        memory.embedding.weight.normal_(std=0.5)
        memory.key_proj.weight.zero_()
        memory.key_proj.bias.copy_(torch.tensor([0.3, 0.6], dtype=torch.float64))
        memory.value_proj.weight.normal_(std=0.4)
        memory.conv_weight.fill_(0.1)
    ids = torch.tensor([[1, 2, 3]])
    hidden = torch.tensor([[[1.0, 2.0], [2.0, 1.0], [1.5, 1.2]]], dtype=torch.float64)
    hidden.requires_grad_()
    # Gate norms intentionally compute in FP32; finite differences use a larger
    # step than a pure-FP64 gradcheck. Positive aligned keys avoid sqrt's cusp.
    assert torch.autograd.gradcheck(
        lambda states: memory(states, ids),
        (hidden,),
        eps=1e-3,
        atol=1e-4,
        rtol=2e-2,
    )


def test_meta_materialization_restores_immutable_hash_buffers() -> None:
    reference = tiny_memory()
    meta = tiny_memory(device="meta")
    assert all(parameter.is_meta for parameter in meta.parameters())
    assert all(buffer.is_meta for buffer in meta.buffers())
    meta.to_empty(device="cpu")
    # OLMo initializes child modules after materializing their parent.
    for module in meta.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    for name in ("token_map", "primes", "offsets", "multipliers"):
        assert torch.equal(getattr(meta, name), getattr(reference, name)), name
    assert torch.count_nonzero(meta.conv_weight) == 0
    ids = torch.tensor([[1, 3, 11, 2]])
    assert torch.equal(meta.hash_ids(ids), reference.hash_ids(ids))
    assert torch.isfinite(meta(torch.randn(1, 4, 4), ids)).all()


def training_update(memory: Engram, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    hidden = torch.randn(2, 5, 4)
    ids = torch.randint(1, 11, (2, 5))
    target = torch.randn_like(hidden)
    optimizer.zero_grad(set_to_none=True)
    loss = (memory(hidden, ids) - target).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            assert_tree_equal(left_value, right_value)
    else:
        assert left == right


def test_state_roundtrip_restores_adamw_and_rng_for_exact_continuation() -> None:
    torch.manual_seed(31)
    continuous = tiny_memory()
    optimizer = torch.optim.AdamW(continuous.parameters(), lr=0.003)
    training_update(continuous, optimizer)
    training_update(continuous, optimizer)
    checkpoint = {
        "model": copy.deepcopy(continuous.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().clone(),
    }
    expected_loss = training_update(continuous, optimizer)
    expected_rng = torch.get_rng_state().clone()
    resumed = tiny_memory()
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.003)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    actual_loss = training_update(resumed, resumed_optimizer)
    assert torch.equal(actual_loss, expected_loss)
    assert_tree_equal(resumed.state_dict(), continuous.state_dict())
    assert_tree_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize(
    "changes",
    [
        {"layer_ids": ()},
        {"layer_ids": (1, 1)},
        {"layer_ids": (-1,)},
        {"layer_ids": [1, 4]},
        {"table_sizes": [11, 17]},
        {"table_sizes": (1, 17)},
        {"table_sizes": (11,)},
        {"max_ngram_size": 1},
        {"heads": 0},
        {"heads": True},
        {"head_dim": 0},
        {"kernel_size": 0},
        {"dilation": 0},
        {"seed": -1},
        {"eos_id": -1},
        {"pad_id": -1},
        {"norm_eps": 0},
        {"norm_eps": float("nan")},
        {"init_std": float("inf")},
        {"gate_variant": "unknown"},
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tiny_config(**changes).validate()


@pytest.mark.parametrize(
    "token_map",
    [torch.tensor([0, 2] * 6), torch.tensor([-1, 0] * 6), torch.arange(12, dtype=torch.int32)],
)
def test_invalid_vocabulary_projection_is_rejected(token_map: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="contiguous|int64"):
        Engram(4, 1, tiny_config(), token_map)


def test_shape_and_layer_contracts_fail_before_silent_broadcast() -> None:
    with pytest.raises(ValueError, match="not an Engram layer"):
        tiny_memory(layer_id=0)
    memory = tiny_memory()
    with pytest.raises(ValueError, match="int64"):
        memory.hash_ids(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="segments"):
        memory.hash_ids(torch.tensor([[1, 2]]), torch.tensor([[0]]))
    with pytest.raises(ValueError, match="hidden states"):
        memory(torch.randn(1, 2, 3), torch.tensor([[1, 2]]))
