"""Record reference compute costs and full-state recovery digests."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import numpy as np
import torch
import torch.distributed as distributed
from olmo_core.train.callbacks.callback import Callback
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule

from src.data.olmo_streaming import StreamingOLMoLoader, TensorBatch

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

INPUT_DIGEST_STEPS = 8
LEDGER_CALLBACK_PRIORITY = 100


class AttemptResult(TypedDict):
    """Keep attempt costs distinct from the cumulative restored training path."""

    status: Literal["preparing", "running", "evaluating", "completed", "smoke_completed", "failed"]
    formal_training: bool
    completed_new_updates: int
    new_update_flops: int
    checkpoint_saved: NotRequired[bool]
    training_completed_wall_seconds: NotRequired[float]
    evaluation_bpb: NotRequired[float | None]
    evaluation_output: NotRequired[str]
    evaluation_elapsed_seconds: NotRequired[float]
    attempted_step: NotRequired[int]
    cumulative_step: NotRequired[int]
    megatron_cumulative_flops: NotRequired[int]
    scored_tokens: NotRequired[int]
    wall_seconds: NotRequired[float]
    input_preparation_seconds: NotRequired[float]
    final_checkpoint: NotRequired[str]
    error: NotRequired[str]
    partial_failed_update_flops: NotRequired[str]


class LedgerState(TypedDict):
    """Preserve callback checkpoint keys across full-state resumes."""

    cumulative_flops: int
    scored_tokens: int


class StateDigests(TypedDict):
    """Identify the model, optimizer, RNG, data and trainer at one update."""

    model: str
    optimizer: str
    rng: str
    data: str
    trainer: str
    step: int
    megatron_cumulative_flops: int


def write_json(path: Path, data: JsonValue | AttemptResult | StateDigests) -> None:
    """Replace a small evidence record atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary_path.replace(path)


def digest(value: object) -> str:
    """Hash heterogeneous native checkpoint values without changing their ordering."""
    state_hash = hashlib.sha256()

    def visit(checkpoint_value: object) -> None:
        if isinstance(checkpoint_value, torch.Tensor):
            tensor = checkpoint_value.detach().cpu().contiguous()
            state_hash.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            state_hash.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(checkpoint_value, np.ndarray):
            state_hash.update(str((checkpoint_value.shape, str(checkpoint_value.dtype))).encode())
            state_hash.update(checkpoint_value.tobytes())
        elif isinstance(checkpoint_value, dict):
            for state_key in sorted(checkpoint_value, key=str):
                state_hash.update(repr(state_key).encode())
                visit(checkpoint_value[state_key])
        elif isinstance(checkpoint_value, (tuple, list)):
            state_hash.update(type(checkpoint_value).__name__.encode())
            for state_item in checkpoint_value:
                visit(state_item)
        else:
            state_hash.update(repr(checkpoint_value).encode())

    visit(value)
    return state_hash.hexdigest()


class TrainingMeter(Callback):
    """Save each completed update's cost before the checkpointer runs."""

    priority = LEDGER_CALLBACK_PRIORITY

    def __init__(
        self,
        *,
        output_directory: Path,
        result: AttemptResult,
        update_flops: int,
        loader: StreamingOLMoLoader,
        train_module: TransformerTrainModule,
        started_seconds: float,
        world_size: int,
        record_state_digests: bool = True,
    ) -> None:
        super().__init__()
        self.output_directory = output_directory
        self.result = result
        self.update_flops = update_flops
        self.loader = loader
        self.train_module = train_module
        self.started_seconds = started_seconds
        self.world_size = world_size
        self.record_state_digests = record_state_digests
        self.cumulative_flops = 0
        self.scored_tokens = 0
        self.batch_input_sha256: str | None = None
        self.step_started_seconds = started_seconds

    def state_dict(self) -> LedgerState:
        """Save cumulative costs, including updates from parent checkpoints."""
        return {"cumulative_flops": self.cumulative_flops, "scored_tokens": self.scored_tokens}

    def load_state_dict(self, state_dict: LedgerState) -> None:
        """Restore costs without resetting the retained training path."""
        self.cumulative_flops = state_dict["cumulative_flops"]
        self.scored_tokens = state_dict["scored_tokens"]

    def state_digests(self) -> StateDigests:
        """Capture all native recovery components at the current update."""
        trainer_state = self.trainer.state_dict()
        return {
            "model": digest(self.train_module.model.state_dict()),
            "optimizer": digest(self.train_module.optim.state_dict()),
            "rng": digest(trainer_state["rng"]),
            "data": digest(trainer_state["data_loader"]),
            "trainer": digest(
                {
                    state_key: trainer_state[state_key]
                    for state_key in (
                        "global_step",
                        "global_train_tokens_seen",
                        "epoch",
                        "world_size",
                    )
                }
            ),
            "step": self.trainer.global_step,
            "megatron_cumulative_flops": self.cumulative_flops,
        }

    def pre_step(self, batch: TensorBatch) -> None:
        """Record the attempted update before compute begins."""
        self.batch_input_sha256 = digest(batch) if self.step <= INPUT_DIGEST_STEPS else None
        self.step_started_seconds = time.monotonic()
        self.result["attempted_step"] = self.step

    def post_train_batch(self) -> None:
        """Append one global update exactly once to this rank's evidence."""
        torch.cuda.synchronize()
        update_flops = self.update_flops
        self.cumulative_flops += update_flops
        if self.cumulative_flops != self.step * update_flops:
            raise ValueError("The retained training path has inconsistent cumulative FLOPs")
        else:
            pass
        scored_token_count = torch.tensor(
            self.loader.latest_batch_accounting["scored_tokens"],
            dtype=torch.int64,
            device=self.train_module.device,
        )
        if self.world_size > 1:
            distributed.all_reduce(scored_token_count)
        else:
            pass
        self.scored_tokens += int(scored_token_count.item())
        completed_updates = self.result["completed_new_updates"] + 1
        self.result.update(
            status="running",
            completed_new_updates=completed_updates,
            cumulative_step=self.step,
            megatron_cumulative_flops=self.cumulative_flops,
            scored_tokens=self.scored_tokens,
            new_update_flops=completed_updates * update_flops,
            wall_seconds=time.monotonic() - self.started_seconds,
        )
        step_record: dict[str, JsonValue] = {
            "step": self.step,
            "seconds": time.monotonic() - self.step_started_seconds,
            "cumulative_flops": self.cumulative_flops,
            "scored_tokens": self.scored_tokens,
            "lr": self.train_module.optim.param_groups[0]["lr"],
            "input_sha256": self.batch_input_sha256,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        with (self.output_directory / "steps.jsonl").open("a") as evidence_file:
            evidence_file.write(json.dumps(step_record) + "\n")
        write_json(self.output_directory / "result.json", self.result)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Retain metrics and fail on nonfinite loss or gradient norm."""
        with (self.output_directory / "metrics.jsonl").open("a") as evidence_file:
            evidence_file.write(json.dumps({"step": step, "metrics": metrics}) + "\n")
        if any(
            not math.isfinite(metric_value)
            for metric_name, metric_value in metrics.items()
            if isinstance(metric_value, (int, float))
            and ("loss" in metric_name.lower() or "grad_norm" in metric_name.lower())
        ):
            raise ValueError("Nonfinite loss or gradient norm; retain attempt for review")
        else:
            pass

    def post_train(self) -> None:
        """Save the final full-state identity after training ends."""
        if self.record_state_digests:
            write_json(self.output_directory / "final-state-digests.json", self.state_digests())
