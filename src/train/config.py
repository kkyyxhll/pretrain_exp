"""Describe training configuration with stable JSON fields and identity hashes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

from src.arch.core.spec import ArchPayload, ArchSpec
from src.data.olmo_streaming import LoaderIdentity

__all__ = ["TrainSpec", "TrainSpecError"]


class TrainingDataPayload(TypedDict):
    """Bind the streaming data format to its complete input and packing identity."""

    format: Literal["compressed-stream-v1"]
    identity: LoaderIdentity


TrainPayload = dict[
    str, ArchPayload | TrainingDataPayload | str | int | float | bool | list[float] | None
]


class TrainSpecError(Exception):
    """Report unsupported training fields or invalid configuration values."""


@dataclass(kw_only=True)
class TrainSpec:
    """Keep the runner's established constructor names and complete training payload."""

    arch: ArchPayload
    data: TrainingDataPayload
    save_folder: str
    rank_microbatch_size: int = 1
    max_steps: int = 500
    max_epochs: int | None = None
    resume_from: str | None = None
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    scheduler_t_max: int | None = None
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 0
    autocast: Literal["bfloat16", "float32"] = "bfloat16"
    compile_model: bool = False
    data_parallel: Literal["fsdp", "ddp"] = "fsdp"
    loss_backend: Literal["torch", "torch_chunked", "fused_linear"] = "torch_chunked"
    loss_chunk_tokens: int = 1024
    training_bpb_manifest: str | None = None
    activation_checkpointing: Literal["full", "none"] = "full"

    def __post_init__(self) -> None:
        for setting_name, setting_value in (
            ("rank_microbatch_size", self.rank_microbatch_size),
            ("max_steps", self.max_steps),
            ("warmup_steps", self.warmup_steps),
            ("seed", self.seed),
        ):
            if not isinstance(setting_value, int) or setting_value < 0:
                raise TrainSpecError(
                    f"{setting_name} must be a nonnegative integer: {setting_value!r}"
                )
            else:
                pass
        for setting_name, setting_value in (
            ("max_epochs", self.max_epochs),
            ("scheduler_t_max", self.scheduler_t_max),
        ):
            if setting_value is not None and (type(setting_value) is not int or setting_value <= 0):
                raise TrainSpecError(f"{setting_name} must be a positive integer or null")
            else:
                pass
        if self.resume_from is not None and (
            not isinstance(self.resume_from, str) or not self.resume_from.strip()
        ):
            raise TrainSpecError("resume_from must be a nonempty checkpoint path or null")
        elif self.data_parallel not in ("fsdp", "ddp"):
            raise TrainSpecError("data_parallel must be fsdp or ddp")
        elif self.loss_backend not in ("torch", "torch_chunked", "fused_linear"):
            raise TrainSpecError("invalid loss_backend")
        elif not isinstance(self.loss_chunk_tokens, int) or self.loss_chunk_tokens <= 0:
            raise TrainSpecError("loss_chunk_tokens must be positive")
        elif self.activation_checkpointing not in ("full", "none"):
            raise TrainSpecError("activation_checkpointing must be full or none")
        elif self.rank_microbatch_size <= 0:
            raise TrainSpecError("rank_microbatch_size must be positive")
        elif self.max_steps <= 0:
            raise TrainSpecError(f"max_steps must be positive: {self.max_steps}")
        elif self.lr <= 0:
            raise TrainSpecError(f"lr must be positive: {self.lr}")
        elif self.autocast not in ("bfloat16", "float32"):
            raise TrainSpecError(f"autocast must be bfloat16 or float32: {self.autocast!r}")
        else:
            pass

    def to_dict(self) -> TrainPayload:
        """Serialize the existing fields while preserving raw architecture and data payloads."""
        training_payload = dataclasses.asdict(self)
        training_payload["betas"] = list(self.betas)
        return training_payload

    def to_json(self) -> str:
        """Encode stable JSON used by the existing training identity hash."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, serialized_spec: Mapping[str, object]) -> TrainSpec:
        """Narrow decoded fields and reject unsupported constructor settings."""
        known_fields = {spec_field.name for spec_field in dataclasses.fields(cls)}
        unknown_fields = set(serialized_spec) - known_fields
        if unknown_fields:
            raise TrainSpecError(f"Unknown TrainSpec fields: {sorted(unknown_fields)}")
        else:
            pass
        constructor_settings = dict(serialized_spec)
        serialized_betas = constructor_settings.get("betas", cls.betas)
        if not isinstance(serialized_betas, (list, tuple)):
            raise TrainSpecError("betas must be a coefficient sequence")
        else:
            constructor_settings["betas"] = tuple(serialized_betas)
        return cls(**constructor_settings)

    def hash(self) -> str:
        """Hash the unchanged sorted JSON representation."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def to_arch_spec(self) -> ArchSpec:
        """Resolve the saved architecture only when the caller builds the model."""
        return ArchSpec.from_dict(self.arch)
