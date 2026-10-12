"""Qwen3.5 SwiGLU adapter using OLMo-core (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from olmo_core.config import DType
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.feed_forward import ActivationFunction, FeedForward, FeedForwardConfig

__all__: list[str] = ["SwigluConfig"]


@dataclass(kw_only=True)
class SwigluConfig(ModuleConfig):
    """SwiGLU projection width and activation configuration."""

    hidden_mult: float = 4.0
    activation: ActivationFunction = ActivationFunction.silu
    bias: bool = False
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"

    def to_olmo(self, d_model: int) -> FeedForwardConfig:
        return FeedForwardConfig(
            hidden_size=int(self.hidden_mult * d_model),
            activation=self.activation,
            bias=self.bias,
            dtype=DType(self.dtype),
        )

    def num_params(self, d_model: int) -> int:
        return self.to_olmo(d_model).num_params(d_model)

    def build(self, d_model: int, *, init_device: str = "cpu") -> FeedForward:
        return self.to_olmo(d_model).build(d_model, init_device=init_device)
