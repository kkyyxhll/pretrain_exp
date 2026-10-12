"""Chunking must preserve loss normalization, masking and all parameter gradients."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Literal

import pytest
import torch
from olmo_core.nn.lm_head import LMHead
from torch.distributed.checkpoint.metadata import Metadata, TensorProperties, TensorStorageMetadata

from eval.checkpoint import (
    evaluation_autocast,
    prepare_model_precision,
    saved_training_spec,
    validate_checkpoint_layout,
    validate_master_checkpoint,
)
from eval.native import token_nll
from src.train.loss import ChunkedLMHead


class TestChunkedLoss(unittest.TestCase):
    def test_native_loss_and_gradients(self) -> None:
        torch.manual_seed(7)
        loss_reductions: tuple[Literal["mean", "sum", "none"], ...] = ("mean", "sum", "none")
        for loss_reduction in loss_reductions:
            for z_loss_multiplier in (None, 1e-4):
                native_head = LMHead(d_model=8, vocab_size=23, bias=True)
                chunked_head = ChunkedLMHead(d_model=8, vocab_size=23, bias=True, chunk_tokens=7)
                chunked_head.load_state_dict(native_head.state_dict())
                native_hidden_states = torch.randn(2, 11, 8, requires_grad=True)
                chunked_hidden_states = native_hidden_states.detach().clone().requires_grad_()
                labels = torch.randint(0, 23, (2, 11))
                labels[:, :7] = -100
                loss_options = dict(
                    labels=labels,
                    loss_reduction=loss_reduction,
                    loss_div_factor=9.0,
                    z_loss_multiplier=z_loss_multiplier,
                    return_logits=False,
                )
                expected = native_head(native_hidden_states, **loss_options)
                actual = chunked_head(chunked_hidden_states, **loss_options)
                # note (kkyyxhll): Unreduced auxiliary loss also excludes masked targets.
                if loss_reduction == "none" and z_loss_multiplier is not None:
                    expected = expected._replace(
                        loss=expected.loss * (labels != -100),
                        z_loss=expected.z_loss * (labels != -100),
                    )
                else:
                    pass
                torch.testing.assert_close(actual.loss, expected.loss)
                expected.loss.sum().backward()
                actual.loss.sum().backward()
                torch.testing.assert_close(native_hidden_states.grad, chunked_hidden_states.grad)
                for native_parameter, chunked_parameter in zip(
                    native_head.parameters(), chunked_head.parameters()
                ):
                    torch.testing.assert_close(native_parameter.grad, chunked_parameter.grad)
                self.assertEqual(set(native_head.state_dict()), set(chunked_head.state_dict()))

    def test_ignored_sum_remains_differentiable(self) -> None:
        chunked_head = ChunkedLMHead(d_model=8, vocab_size=23, chunk_tokens=3)
        hidden_states = torch.randn(2, 5, 8, requires_grad=True)
        loss_output = chunked_head(
            hidden_states, labels=torch.full((2, 5), -100), loss_reduction="sum"
        )
        self.assertEqual(loss_output.loss.item(), 0)
        loss_output.loss.backward()
        torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))

    def test_logits_and_selection(self) -> None:
        native_head = LMHead(d_model=8, vocab_size=23)
        chunked_head = ChunkedLMHead(d_model=8, vocab_size=23, chunk_tokens=3)
        chunked_head.load_state_dict(native_head.state_dict())
        hidden_states = torch.randn(2, 5, 8)
        for logits_to_keep in (0, 2, torch.tensor([[0, 3], [1, 4]])):
            torch.testing.assert_close(
                native_head(hidden_states, logits_to_keep=logits_to_keep),
                chunked_head(hidden_states, logits_to_keep=logits_to_keep),
            )


@pytest.mark.parametrize(
    "precision,parameter_dtype", [("bf16", torch.bfloat16), ("fp32-master-bf16", torch.float32)]
)
def test_evaluation_storage_and_compute_precision(
    precision: Literal["bf16", "fp32-master-bf16"], parameter_dtype: torch.dtype
) -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0001)
    original_weights = model.weight.detach().clone()
    resolved_device = torch.device("cpu")
    prepared_model = prepare_model_precision(model, resolved_device, precision)
    assert prepared_model is model
    assert model.weight.dtype == parameter_dtype
    assert not model.training and not model.weight.requires_grad
    torch.testing.assert_close(model.weight, original_weights.to(parameter_dtype), rtol=0, atol=0)
    with torch.inference_mode(), evaluation_autocast(precision, resolved_device):
        logits = model(torch.ones(1, 3, dtype=parameter_dtype))
    assert logits.dtype == torch.bfloat16
    assert model.weight.dtype == parameter_dtype
    with pytest.raises(ValueError, match="unsupported evaluation precision"):
        evaluation_autocast("invalid", resolved_device)
    with pytest.raises(ValueError, match="unsupported evaluation precision"):
        prepare_model_precision(model, resolved_device, "invalid")


@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_master_evaluation_rejects_rounded_checkpoint_storage(parameter_dtype: torch.dtype) -> None:
    checkpoint_metadata = Metadata(
        state_dict_metadata={
            "model.weight": TensorStorageMetadata(
                properties=TensorProperties(dtype=parameter_dtype),
                size=torch.Size((2, 3)),
                chunks=[],
            ),
            "optimizer.weight": TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.bfloat16),
                size=torch.Size((2, 3)),
                chunks=[],
            ),
        }
    )
    if parameter_dtype == torch.float32:
        validate_master_checkpoint(checkpoint_metadata)
    else:
        with pytest.raises(ValueError, match="requires FP32 checkpoint tensors"):
            validate_master_checkpoint(checkpoint_metadata)


def test_checkpoint_layout_requires_all_model_keys_and_shapes() -> None:
    model_state = {"weight": torch.zeros(2, 3)}
    stored_tensor = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32), size=torch.Size((2, 3)), chunks=[]
    )
    validate_checkpoint_layout(
        model_state, Metadata(state_dict_metadata={"model.weight": stored_tensor})
    )
    with pytest.raises(ValueError, match="model keys differ"):
        validate_checkpoint_layout(model_state, Metadata(state_dict_metadata={}))
    with pytest.raises(ValueError, match="model keys differ"):
        validate_checkpoint_layout(
            model_state,
            Metadata(
                state_dict_metadata={"model.weight": stored_tensor, "model.extra": stored_tensor}
            ),
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_checkpoint_layout(
            {"weight": torch.zeros(3, 2)},
            Metadata(state_dict_metadata={"model.weight": stored_tensor}),
        )


def test_fixed_weight_loss_excludes_masked_tokens_and_uses_original_source_bytes() -> None:
    losses = torch.tensor([[1.25, float("nan"), 2.75, 4.0]], dtype=torch.float32)
    labels = torch.tensor([[3, -100, 7, 8]])
    summed_loss_nats = token_nll(losses, labels)
    assert summed_loss_nats.dtype == torch.float64
    assert summed_loss_nats.item() == 8.0
    source_byte_count = len("汉🙂".encode())
    assert summed_loss_nats.item() / (math.log(2) * source_byte_count) == pytest.approx(
        8.0 / (math.log(2) * 7)
    )
    with pytest.raises(ValueError, match="identical shapes"):
        token_nll(losses, labels[:, :2])
    with pytest.raises(RuntimeError, match="non-finite"):
        token_nll(losses, torch.zeros_like(labels))


def test_checkpoint_requires_saved_architecture_in_supported_parent_locations(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint" / "step2"
    checkpoint_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="retain its train.json"):
        saved_training_spec(checkpoint_dir)
    training_spec = run_dir / "train.json"
    training_spec.write_text('{"arch":{}}\n')
    assert saved_training_spec(checkpoint_dir) == training_spec
    adjacent_spec = checkpoint_dir.parent / "train.json"
    adjacent_spec.write_text('{"arch":{}}\n')
    assert saved_training_spec(checkpoint_dir) == adjacent_spec
