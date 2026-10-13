"""Tiny OLMo training contracts for optional single-stream Engram memory."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from src.arch.core.spec import ArchSpec, LayerSpec, build_model, spec_to_olmo
from src.arch.memory.engram import EngramConfig
from src.arch.memory.olmo import (
    EngramIdentity,
    EngramTransformer,
    build_engram_model,
    config_from_identity,
    memory_identity,
    validate_memory_tokenizer,
)
from src.utils.engram_budget import budget_report_with_engram
from src.utils.megatron_budget import budget_report

ROOT = Path(__file__).resolve().parents[1]


def small_spec() -> ArchSpec:
    return ArchSpec(
        vocab_size=17,
        d_model=16,
        n_layers=2,
        dtype="float32",
        layers=[
            LayerSpec(
                attn_kwargs={"n_heads": 2, "n_kv_heads": 2, "backend": "torch"},
                ffn_kwargs={"hidden_mult": 2.0},
            )
            for _ in range(2)
        ],
    )


def small_backbone() -> TransformerConfig:
    return replace(spec_to_olmo(small_spec()), init_seed=71)


def small_memory() -> EngramConfig:
    return EngramConfig(
        layer_ids=(1,),
        table_sizes=(11, 17),
        heads=2,
        head_dim=4,
        kernel_size=3,
        dilation=2,
        seed=29,
        eos_id=16,
        pad_id=0,
    )


def initialized_memory_model(*, checkpointed: bool = False) -> EngramTransformer:
    model = EngramTransformer(
        small_backbone(), small_memory(), torch.arange(17), init_device="meta"
    )
    model.init_weights(max_seq_len=6, device=torch.device("cpu"))
    if checkpointed:
        model.apply_activation_checkpointing("full")
    return model


def language_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[1, 2, 16, 3, 4, 5], [6, 7, 8, 16, 9, 10]])
    labels = ids.roll(-1, dims=1)
    labels[:, -1] = -100
    return ids, labels


def summed_language_loss(
    model: Transformer, ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    logits = model(ids)
    assert isinstance(logits, torch.Tensor)
    return F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
    )


def canonical_gradients(model: Transformer) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        gradients[name.replace("_checkpoint_wrapped_module.", "")] = parameter.grad.clone()
    return gradients


def identity_assets(tmp_path: Path) -> tuple[Path, Path, EngramIdentity]:
    config_path = tmp_path / "engram.json"
    token_map_path = tmp_path / "token-map.npy"
    config_path.write_text(json.dumps(asdict(small_memory())))
    np.save(token_map_path, np.arange(17, dtype=np.int64), allow_pickle=False)
    return config_path, token_map_path, memory_identity(config_path, token_map_path)


def test_optional_memory_keeps_baseline_spec_and_initialized_backbone_unchanged() -> None:
    spec = small_spec()
    original_identity = spec.to_json(), spec.hash()
    baseline = build_model(spec, init_device="meta")
    baseline.init_seed = small_backbone().init_seed
    baseline.init_weights(max_seq_len=6, device=torch.device("cpu"))
    expanded = initialized_memory_model()
    assert type(baseline) is Transformer
    assert not hasattr(baseline, "engrams")
    assert (spec.to_json(), spec.hash()) == original_identity
    baseline_state = baseline.state_dict()
    expanded_state = expanded.state_dict()
    assert set(expanded_state) - set(baseline_state)
    assert all(name.startswith("engrams.") for name in set(expanded_state) - set(baseline_state))
    for name, tensor in baseline_state.items():
        assert torch.equal(tensor, expanded_state[name]), name
    assert expanded.embeddings.weight is expanded.lm_head.w_out.weight


def test_engram_meta_init_is_valid_and_independent_of_global_rng() -> None:
    torch.manual_seed(1)
    first = initialized_memory_model()
    torch.manual_seed(200)
    torch.randn(47)
    second = initialized_memory_model()
    for name, tensor in first.state_dict().items():
        assert not tensor.is_meta, name
        assert torch.isfinite(tensor).all(), name
        assert torch.equal(tensor, second.state_dict()[name]), name
    memory = first.engrams["1"]
    assert memory.token_map.tolist() == list(range(17))
    assert memory.primes.tolist() == [11, 13, 17, 19]
    assert memory.offsets.tolist() == [0, 11, 24, 41]
    assert torch.count_nonzero(memory.conv_weight) == 0


def test_prepare_inputs_preserves_raw_ids_and_adds_memory_kwargs_only_at_selected_layer() -> None:
    model = initialized_memory_model()
    ids, labels = language_batch()
    token_ids, targets, common, layers, _ = model._prepare_inputs(ids, labels)
    assert torch.equal(token_ids, ids)
    assert torch.equal(targets, labels)
    assert not any(name.startswith("engram_") for name in common)
    assert all(
        not any(name.startswith("engram_") for name in kwargs)
        for layer_id, kwargs in layers.items()
        if layer_id != 1
    )
    assert torch.equal(layers[1]["engram_input_ids"], ids)
    assert torch.equal(
        layers[1]["engram_segment_ids"], torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]])
    )


def test_real_olmo_forward_loss_and_memory_gradients() -> None:
    model = initialized_memory_model()
    ids, labels = language_batch()
    loss = summed_language_loss(model, ids, labels) / labels.ne(-100).sum()
    assert torch.isfinite(loss)
    loss.backward()
    for name, gradient in canonical_gradients(model).items():
        assert torch.isfinite(gradient).all(), name
    for name in ("embedding.weight", "key_proj.weight", "value_proj.weight", "conv_weight"):
        parameter = dict(model.engrams["1"].named_parameters())[name]
        assert torch.count_nonzero(parameter.grad), name


def test_full_activation_recompute_preserves_memory_inputs_outputs_and_gradients() -> None:
    plain = initialized_memory_model()
    recomputed = initialized_memory_model(checkpointed=True)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_memory_inputs(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        del module
        calls.append((inputs[1].clone(), inputs[2].clone()))

    hook = recomputed.engrams["1"].register_forward_pre_hook(capture_memory_inputs)
    ids, labels = language_batch()
    plain_logits = plain(ids)
    recomputed_logits = recomputed(ids)
    assert isinstance(plain_logits, torch.Tensor)
    assert isinstance(recomputed_logits, torch.Tensor)
    torch.testing.assert_close(recomputed_logits, plain_logits, atol=0, rtol=0)
    target_count = labels.ne(-100).sum()
    plain_loss = (
        F.cross_entropy(
            plain_logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        / target_count
    )
    recomputed_loss = (
        F.cross_entropy(
            recomputed_logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        / target_count
    )
    plain_loss.backward()
    recomputed_loss.backward()
    hook.remove()
    assert len(calls) == 2
    for raw_ids, segments in calls:
        assert torch.equal(raw_ids, ids)
        assert torch.equal(segments[:, 3:], torch.tensor([[1, 1, 1], [0, 1, 1]]))
    plain_gradients = canonical_gradients(plain)
    recomputed_gradients = canonical_gradients(recomputed)
    assert plain_gradients.keys() == recomputed_gradients.keys()
    for name, expected in plain_gradients.items():
        torch.testing.assert_close(recomputed_gradients[name], expected, atol=1e-7, rtol=1e-5)


def test_two_accumulated_microbatches_match_full_batch_gradient() -> None:
    full_batch = initialized_memory_model(checkpointed=True)
    accumulated = initialized_memory_model(checkpointed=True)
    ids, labels = language_batch()
    target_count = labels.ne(-100).sum()
    (summed_language_loss(full_batch, ids, labels) / target_count).backward()
    for micro_ids, micro_labels in zip(ids.split(1), labels.split(1), strict=True):
        (summed_language_loss(accumulated, micro_ids, micro_labels) / target_count).backward()
    full_gradients = canonical_gradients(full_batch)
    micro_gradients = canonical_gradients(accumulated)
    assert full_gradients.keys() == micro_gradients.keys()
    for name, expected in full_gradients.items():
        torch.testing.assert_close(micro_gradients[name], expected, atol=1e-6, rtol=1e-4)


def test_declared_assets_and_state_roundtrip_reject_changed_token_map(tmp_path: Path) -> None:
    config_path, map_path, identity = identity_assets(tmp_path)
    assert config_from_identity(identity) == small_memory()
    first = build_engram_model(small_backbone(), identity)
    first.init_weights(max_seq_len=6, device=torch.device("cpu"))
    ids, labels = language_batch()
    optimizer = torch.optim.AdamW(first.parameters(), lr=0.001)
    summed_language_loss(first, ids, labels).backward()
    optimizer.step()
    saved_state = copy.deepcopy(first.state_dict())
    restored = build_engram_model(small_backbone(), identity)
    restored.init_weights(max_seq_len=6, device=torch.device("cpu"))
    restored.load_state_dict(saved_state)
    torch.testing.assert_close(restored(ids), first(ids), atol=0, rtol=0)
    np.save(map_path, np.arange(16, -1, -1, dtype=np.int64), allow_pickle=False)
    assert (
        memory_identity(config_path, map_path)["token_map_sha256"] != identity["token_map_sha256"]
    )
    with pytest.raises(ValueError, match="projection changed"):
        build_engram_model(small_backbone(), identity)


def test_engram_rejects_layer_outside_backbone_and_unknown_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="layer count"):
        EngramTransformer(
            small_backbone(), replace(small_memory(), layer_ids=(2,)), torch.arange(17)
        )
    config_path, map_path, _ = identity_assets(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["unrecognized_parameter"] = 1
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown Engram"):
        memory_identity(config_path, map_path)


@pytest.mark.parametrize("budget", [3 * 10**17, 10**18, 3 * 10**18, 10**19])
def test_engram_budget_keeps_official_backbone_and_experimental_appendix_distinct(
    budget: int,
) -> None:
    backbone = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())
    memory = EngramConfig()
    baseline = budget_report(backbone, sequence_length=8192, batch_sequences=128, budget=budget)
    combined = budget_report_with_engram(
        backbone,
        engram_config=memory,
        sequence_length=8192,
        batch_sequences=128,
        budget=budget,
    )
    assert (
        combined["backbone_megatron_training_flops_per_token"]
        == baseline["training_flops_per_token"]
    )
    assert combined["engram_experimental_training_flops_per_token"] == 2 * 12607488
    assert combined["training_flops_per_token"] == baseline["training_flops_per_token"] + 25214976
    assert combined["official_engram_certification"] == "not_available"
    assert combined["megatron_args"] == baseline["megatron_args"]
    update_tokens = 128 * 8192
    update_flops = combined["training_flops_per_token"] * update_tokens
    assert combined["training_flops_per_optimizer_step"] == update_flops
    steps = combined["whole_optimizer_steps"]
    assert combined["actual_whole_step_flops"] == steps * update_flops
    assert combined["whole_step_token_capacity"] == steps * update_tokens
    assert combined["continuous_token_capacity"] == budget / combined["training_flops_per_token"]
    assert combined["relative_budget_gap"] == (budget - steps * update_flops) / budget
    assert abs(combined["actual_whole_step_flops"] - budget) <= update_flops / 2
    assert combined["within_one_percent"] == (abs(steps * update_flops - budget) <= budget // 100)


def test_engram_major_flops_depend_on_active_width_and_layers_not_table_capacity() -> None:
    backbone = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())
    memory = replace(EngramConfig(), layer_ids=(1,))
    shared_settings = {"sequence_length": 8192, "batch_sequences": 128, "budget": 3 * 10**17}
    single = budget_report_with_engram(backbone, engram_config=memory, **shared_settings)
    large_tables = budget_report_with_engram(
        backbone, engram_config=replace(memory, table_sizes=(65536, 131072)), **shared_settings
    )
    assert single["engram_experimental_training_flops_per_token"] == 12607488
    assert large_tables["training_flops_per_token"] == single["training_flops_per_token"]
    assert large_tables["whole_optimizer_steps"] == single["whole_optimizer_steps"]
    floor = budget_report_with_engram(
        backbone, engram_config=memory, rounding="floor", **shared_settings
    )
    assert floor["actual_whole_step_flops"] <= shared_settings["budget"]
    assert floor["whole_optimizer_steps"] <= single["whole_optimizer_steps"]


def fast_tokenizer(directory: Path, *, alternate_backend: bool = False) -> PreTrainedTokenizerFast:
    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update({f"word{index}": index for index in range(2, 16)})
    vocabulary["<eos>"] = 16
    if alternate_backend:
        del vocabulary["word2"]
        vocabulary["another-word"] = 2
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(directory)
    return tokenizer


def tokenizer_identity_assets(tmp_path: Path) -> tuple[Path, Path, Path, EngramIdentity]:
    config_path, map_path, _ = identity_assets(tmp_path)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = fast_tokenizer(tokenizer_dir)
    map_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "token_map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                "tokenizer_backend_sha256": hashlib.sha256(
                    tokenizer.backend_tokenizer.to_str().encode()
                ).hexdigest(),
            }
        )
    )
    return config_path, map_path, tokenizer_dir, memory_identity(config_path, map_path)


def test_memory_tokenizer_binding_accepts_matching_local_fast_tokenizer(tmp_path: Path) -> None:
    _, _, tokenizer_dir, identity = tokenizer_identity_assets(tmp_path)
    assert validate_memory_tokenizer(identity, tokenizer_dir) is None


def test_memory_tokenizer_binding_rejects_same_size_different_backend(tmp_path: Path) -> None:
    _, _, original_dir, identity = tokenizer_identity_assets(tmp_path)
    alternate_dir = tmp_path / "alternate-tokenizer"
    alternate = fast_tokenizer(alternate_dir, alternate_backend=True)
    original = PreTrainedTokenizerFast.from_pretrained(original_dir, local_files_only=True)
    assert len(original) == len(alternate) == 17
    assert original.eos_token_id == alternate.eos_token_id == 16
    assert original.pad_token_id == alternate.pad_token_id == 0
    with pytest.raises(ValueError, match="different training tokenizer"):
        validate_memory_tokenizer(identity, alternate_dir)


@pytest.mark.parametrize("special_id", ["eos_id", "pad_id"])
def test_memory_tokenizer_binding_rejects_different_special_token_ids(
    tmp_path: Path, special_id: str
) -> None:
    config_path, map_path, tokenizer_dir, _ = tokenizer_identity_assets(tmp_path)
    configuration = json.loads(config_path.read_text())
    configuration[special_id] = 15
    config_path.write_text(json.dumps(configuration))
    identity = memory_identity(config_path, map_path)
    with pytest.raises(ValueError, match="EOS/pad IDs"):
        validate_memory_tokenizer(identity, tokenizer_dir)


def test_memory_tokenizer_binding_rejects_wrong_map_length(tmp_path: Path) -> None:
    config_path, map_path, tokenizer_dir, _ = tokenizer_identity_assets(tmp_path)
    np.save(map_path, np.arange(16, dtype=np.int64), allow_pickle=False)
    metadata_path = map_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["token_map_sha256"] = hashlib.sha256(map_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    identity = memory_identity(config_path, map_path)
    with pytest.raises(ValueError, match="length differs"):
        validate_memory_tokenizer(identity, tokenizer_dir)


def test_memory_identity_rejects_sidecar_for_different_map_bytes(tmp_path: Path) -> None:
    config_path, map_path, _, _ = tokenizer_identity_assets(tmp_path)
    metadata_path = map_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["token_map_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="preparation metadata"):
        memory_identity(config_path, map_path)


def test_memory_tokenizer_binding_requires_backend_identity_metadata(tmp_path: Path) -> None:
    _, _, tokenizer_dir, identity = tokenizer_identity_assets(tmp_path)
    del identity["tokenizer_backend_sha256"]
    with pytest.raises(ValueError, match="tokenizer identity metadata"):
        validate_memory_tokenizer(identity, tokenizer_dir)


@pytest.mark.parametrize("backend_hash", [None, "too-short"])
def test_memory_identity_rejects_sidecar_without_valid_backend_hash(
    tmp_path: Path, backend_hash: str | None
) -> None:
    config_path, map_path, _, _ = tokenizer_identity_assets(tmp_path)
    metadata_path = map_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if backend_hash is None:
        del metadata["tokenizer_backend_sha256"]
    else:
        metadata["tokenizer_backend_sha256"] = backend_hash
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identify the tokenizer backend"):
        memory_identity(config_path, map_path)


def test_memory_tokenizer_binding_uses_eos_when_tokenizer_has_no_padding_token(
    tmp_path: Path,
) -> None:
    config_path, map_path, tokenizer_dir, _ = tokenizer_identity_assets(tmp_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.pad_token = None
    tokenizer.save_pretrained(tokenizer_dir)
    configuration = json.loads(config_path.read_text())
    configuration["pad_id"] = configuration["eos_id"]
    config_path.write_text(json.dumps(configuration))
    identity = memory_identity(config_path, map_path)
    assert validate_memory_tokenizer(identity, tokenizer_dir) is None
