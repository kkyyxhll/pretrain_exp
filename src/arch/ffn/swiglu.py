"""Qwen3.5 SwiGLU adapter using OLMo-core (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass

from olmo_core.config import DType
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig

from ..core.registry import register

__all__ = ["SwigluConfig"]


@register("ffn", "swiglu")
@dataclass
class SwigluConfig(ModuleConfig):
    """可搜索超参：hidden_mult（相对 d_model 的隐藏维倍数）、activation、bias。"""

    hidden_mult: float = 4.0  # 允许非整数倍（Qwen3.5 = 3.5）
    activation: ActivationFunction = ActivationFunction.silu
    bias: bool = False
    dtype: str = "float32"

    def to_olmo(self, d_model: int) -> FeedForwardConfig:
        return FeedForwardConfig(
            hidden_size=int(self.hidden_mult * d_model),
            activation=self.activation,
            bias=self.bias,
            dtype=DType(self.dtype),
        )

    def num_params(self, d_model: int) -> int:
        return self.to_olmo(d_model).num_params(d_model)

    def build(self, d_model: int, *, init_device: str = "cpu"):
        return self.to_olmo(d_model).build(d_model, init_device=init_device)
