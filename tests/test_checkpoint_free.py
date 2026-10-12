"""Keep checkpoint-free attempts and fixed-weight evaluation independently auditable."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from olmo_core.nn.transformer import Transformer

import eval.native as native_evaluator
from src.arch.core.spec import ArchSpec, LayerSpec, spec_to_olmo
from src.train.loss import ChunkedLMHeadConfig

TRAINING = importlib.import_module("recipes.qwen35-08b.train")


@pytest.mark.parametrize(
    "extra_arguments,error_message",
    [
        (["--resume", "own-checkpoint"], "cannot resume"),
        ([], "requires --eval-cache"),
    ],
)
def test_checkpoint_free_formal_cli_rejects_unrecoverable_plans_before_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_arguments: list[str],
    error_message: str,
) -> None:
    output_directory = tmp_path / "attempt"
    with pytest.raises(SystemExit) as failure:
        TRAINING.arguments(
            [
                "--budget",
                str(3 * 10**17),
                "--data-dir",
                str(tmp_path),
                "--tokenizer-dir",
                str(tmp_path),
                "--output",
                str(output_directory),
                "--no-checkpoints",
                *extra_arguments,
            ]
        )
    assert failure.value.code == 2
    assert error_message in capsys.readouterr().err
    assert not output_directory.exists()


def test_checkpoint_free_engineering_cli_accepts_bounded_training_without_evaluation(
    tmp_path: Path,
) -> None:
    options = TRAINING.arguments(
        [
            "--budget",
            str(3 * 10**17),
            "--data-dir",
            str(tmp_path),
            "--tokenizer-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "attempt"),
            "--no-checkpoints",
            "--smoke-steps",
            "2",
        ]
    )
    assert options.no_checkpoints and options.smoke_steps == 2
    assert options.eval_cache is None
    assert not options.output.exists()


@pytest.fixture
def evaluation_model() -> Transformer:
    model_config = spec_to_olmo(
        ArchSpec(
            vocab_size=23,
            d_model=8,
            n_layers=1,
            dtype="float32",
            layers=[LayerSpec(attn_kwargs={"n_heads": 2, "backend": "torch"})],
        )
    )
    model_config.lm_head = ChunkedLMHeadConfig(bias=False, chunk_tokens=1024)
    model = model_config.build(init_device="cpu")
    model.init_weights(max_seq_len=4)
    return model


@pytest.mark.parametrize("rank", [0, 1])
def test_in_memory_full_evaluation_reuses_two_ranks_without_checkpoint_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluation_model: Transformer, rank: int
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    tokens = np.asarray([1, 2, 3, 4, 1, 5, 6, 7, 1], dtype=np.int32)
    text_mask = np.asarray([False, True, True, True, False, True, True, True, False])
    tokens.tofile(cache_directory / "tokens.bin")
    text_mask.tofile(cache_directory / "is_text.bin")
    tokenizer_identity = {"fixture": "unchanged"}
    original_utf8_bytes = len("汉🙂abc".encode())
    (cache_directory / "metadata.json").write_text(
        json.dumps(
            dict(
                total_bytes=original_utf8_bytes,
                total_tokens=6,
                pad_token_id=1,
                tokenizer_identity=tokenizer_identity,
            )
        )
    )
    training_spec = tmp_path / "train.json"
    training_spec.write_text(json.dumps({"data": {"tokenizer_identity": tokenizer_identity}}))
    output_directory = tmp_path / "evaluation"
    initial_weights = {
        name: value.detach().clone() for name, value in evaluation_model.named_parameters()
    }
    initial_digest = native_evaluator.weight_digest(evaluation_model)
    initial_training_mode = evaluation_model.training
    input_array, label_array = native_evaluator.official_module().packed_batch(
        tokens, text_mask, 0, 2, 4, 1
    )
    labels = torch.from_numpy(label_array)
    reference_rank_nll: list[float] = []
    with (
        torch.inference_mode(),
        native_evaluator.evaluation_autocast("fp32-master-bf16", torch.device("cpu")),
    ):
        for index in (0, 1):
            rank_labels = labels[index : index + 1]
            logits = evaluation_model(torch.from_numpy(input_array[index : index + 1]))
            reference_losses = functional.cross_entropy(
                logits.float().flatten(0, 1),
                rank_labels.flatten(),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(rank_labels)
            reference_rank_nll.append(float(reference_losses[rank_labels != -100].double().sum()))
    peer_nll = reference_rank_nll[1 - rank]
    expected_nll = sum(reference_rank_nll)

    def sum_peer_values(value: torch.Tensor) -> None:
        if value.numel() == 2:
            value += torch.tensor([peer_nll, 3], dtype=value.dtype)
        elif value.dtype == torch.int64:
            value += 3
        else:
            value += peer_nll

    def gather_matching_digests(result: list[str | None], digest: str) -> None:
        result[:] = [digest, digest]

    group_initialization = Mock(side_effect=AssertionError("must reuse the training group"))
    group_destruction = Mock(side_effect=AssertionError("training owns its group"))
    checkpoint_loading = Mock(side_effect=AssertionError("must use final in-memory weights"))
    checkpoint_saving = Mock(side_effect=AssertionError("weights must not be saved"))
    monkeypatch.setattr(native_evaluator.distributed, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(native_evaluator.distributed, "get_rank", Mock(return_value=rank))
    monkeypatch.setattr(native_evaluator.distributed, "get_world_size", Mock(return_value=2))
    monkeypatch.setattr(native_evaluator.distributed, "all_gather_object", gather_matching_digests)
    monkeypatch.setattr(native_evaluator.distributed, "all_reduce", sum_peer_values)
    monkeypatch.setattr(native_evaluator.distributed, "barrier", Mock())
    monkeypatch.setattr(native_evaluator.distributed, "init_process_group", group_initialization)
    monkeypatch.setattr(native_evaluator.distributed, "destroy_process_group", group_destruction)
    monkeypatch.setattr(native_evaluator, "load_model", checkpoint_loading)
    monkeypatch.setattr(native_evaluator.torch, "save", checkpoint_saving)
    monkeypatch.setattr(native_evaluator, "source_commit", Mock(return_value="fixture"))

    root_forward_sync = Mock(side_effect=AssertionError("must bypass root DDP forward hooks"))
    hook = evaluation_model.register_forward_pre_hook(root_forward_sync)
    try:
        result = native_evaluator.evaluate_in_memory(
            evaluation_model,
            training_spec,
            cache_directory,
            output_directory,
            training_step=55,
            sequence_length=4,
            batch_size=1,
        )
    finally:
        hook.remove()

    assert result.full_dataset and result.scored_tokens == 6
    assert result.original_utf8_bytes == original_utf8_bytes
    assert result.nll_nats == pytest.approx(expected_nll, rel=1e-6)
    assert result.bpb == pytest.approx(expected_nll / (math.log(2) * original_utf8_bytes), rel=1e-6)
    assert result.checkpoint is None and result.weight_source == "in_memory"
    assert result.training_step == 55 and result.weights_sha256 == initial_digest
    assert evaluation_model.training == initial_training_mode
    for name, parameter in evaluation_model.named_parameters():
        torch.testing.assert_close(parameter, initial_weights[name], rtol=0, atol=0)
    for forbidden_operation in (
        group_initialization,
        group_destruction,
        checkpoint_loading,
        checkpoint_saving,
        root_forward_sync,
    ):
        forbidden_operation.assert_not_called()
    if rank == 0:
        assert {path.name for path in output_directory.iterdir()} == {
            "run_meta.json",
            "progress.json",
            "result.json",
        }
        saved_result = json.loads((output_directory / "result.json").read_text())
        assert saved_result["checkpoint"] is None
        assert saved_result["weights_sha256"] == initial_digest
    else:
        assert not output_directory.exists()


def test_in_memory_evaluation_rejects_different_rank_weights_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluation_model: Transformer
) -> None:
    monkeypatch.setattr(native_evaluator.distributed, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(native_evaluator.distributed, "get_world_size", Mock(return_value=2))

    def gather_different_digests(result: list[str | None], digest: str) -> None:
        result[:] = [digest, "different weights"]

    monkeypatch.setattr(native_evaluator.distributed, "all_gather_object", gather_different_digests)
    output_directory = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="weights differ"):
        native_evaluator.evaluate_in_memory(
            evaluation_model,
            tmp_path / "train.json",
            tmp_path / "cache",
            output_directory,
            training_step=55,
        )
    assert not output_directory.exists()
