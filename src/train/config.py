"""Strict JSON training specification for the Qwen3.5 OLMo runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["TrainSpec", "TrainSpecError"]


class TrainSpecError(Exception):
    pass


@dataclass
class TrainSpec:
    arch: Dict[str, Any]  # ArchSpec.to_dict()
    data: Dict[str, Any]  # DataSpec.to_dict()
    save_folder: str
    rank_microbatch_size: int = 1  # 每 rank 每条微批的序列数（实例数）
    max_steps: int = 500
    max_epochs: int | None = None  # overrides max_steps; scheduler uses loader epoch length
    resume_from: str | None = None  # require model, optimizer, trainer and RNG state
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)  # Qwen 系口径
    scheduler_t_max: int | None = None  # optional independent cosine horizon (steps)
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 0
    autocast: str = "bfloat16"  # bfloat16 | float32
    compile_model: bool = False
    data_parallel: str = "fsdp"  # fsdp | ddp; torchrun ranks must synchronize gradients
    loss_backend: str = "torch_chunked"  # torch | torch_chunked | fused_linear
    loss_chunk_tokens: int = 1024
    training_bpb_manifest: str | None = None
    activation_checkpointing: str = "full"  # full | none

    def __post_init__(self):
        for k in ("rank_microbatch_size", "max_steps", "warmup_steps", "seed"):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 0:
                raise TrainSpecError(f"{k} 必须为非负整数: {getattr(self, k)!r}")
        if self.max_epochs is not None and (
            type(self.max_epochs) is not int or self.max_epochs <= 0
        ):
            raise TrainSpecError("max_epochs must be a positive integer or null")
        if self.scheduler_t_max is not None and (
            type(self.scheduler_t_max) is not int or self.scheduler_t_max <= 0
        ):
            raise TrainSpecError("scheduler_t_max must be a positive integer or null")
        if self.resume_from is not None and (
            not isinstance(self.resume_from, str) or not self.resume_from.strip()
        ):
            raise TrainSpecError("resume_from must be a nonempty checkpoint path or null")
        if self.data_parallel not in ("fsdp", "ddp"):
            raise TrainSpecError("data_parallel must be fsdp or ddp")
        if self.loss_backend not in ("torch", "torch_chunked", "fused_linear"):
            raise TrainSpecError("invalid loss_backend")
        if not isinstance(self.loss_chunk_tokens, int) or self.loss_chunk_tokens <= 0:
            raise TrainSpecError("loss_chunk_tokens must be positive")
        if self.activation_checkpointing not in ("full", "none"):
            raise TrainSpecError("activation_checkpointing must be full or none")
        if self.rank_microbatch_size <= 0:
            raise TrainSpecError("rank_microbatch_size must be positive")
        if self.max_steps <= 0:
            raise TrainSpecError(f"max_steps 必须为正: {self.max_steps}")
        if self.lr <= 0:
            raise TrainSpecError(f"lr 必须为正: {self.lr}")
        if self.autocast not in ("bfloat16", "float32"):
            raise TrainSpecError(f"autocast 仅支持 bfloat16/float32: {self.autocast!r}")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["betas"] = list(self.betas)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise TrainSpecError(f"TrainSpec 未知字段: {sorted(unknown)}")
        d = dict(d)
        d["betas"] = tuple(d.get("betas", (0.9, 0.95)))
        return cls(**d)

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def to_arch_spec(self):
        from arch import ArchSpec

        return ArchSpec.from_dict(self.arch)
