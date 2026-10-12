"""Qwen RMSNorm adapters using OLMo-core contracts (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
from olmo_core.config import DType
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType, RMSNorm

from ..core.registry import register

__all__ = ["RmsNormConfig"]


class Qwen23RMSNorm(RMSNorm):
    """HF gain arithmetic, followed by OLMo's input-dtype output contract.

    In particular, BF16 Q/K with FP32 master gains must not round the gain
    before multiplication. The final cast matches the FlashAttention entry.
    """

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            h = x.float()
            h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
            if self.weight is not None:
                h = h * self.weight
            if self.bias is not None:
                h = h + self.bias
            return h.to(x.dtype)


class Qwen35RMSNorm(RMSNorm):
    """HF Qwen3.5: zero gain, FP32 normalize/affine, then cast back."""

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.zeros_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            h = x.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            if self.weight is not None:
                h = h * (1.0 + self.weight.float())
            if self.bias is not None:
                h = h + self.bias.float()
            return h.to(x.dtype)


@dataclass
class Qwen35RMSNormConfig(LayerNormConfig):
    """Use OLMo's norm contract, including meta materialization/reset."""

    norm_cls: ClassVar[type] = Qwen35RMSNorm

    def build(self, size, init_device="cpu", weight_shape=None):
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        if (dtype := kwargs.pop("dtype", None)) is not None:
            kwargs["dtype"] = dtype.as_pt()
        if weight_shape is not None:
            kwargs["weight_shape"] = weight_shape
        return self.norm_cls(size=size, init_device=init_device, **kwargs)


@dataclass
class Qwen23RMSNormConfig(Qwen35RMSNormConfig):
    norm_cls: ClassVar[type] = Qwen23RMSNorm


@register("norm", "rms")
@dataclass
class RmsNormConfig(ModuleConfig):
    """qwen selects Qwen2/3 rounding; unit_offset selects Qwen3.5."""

    qwen: bool = False  # True → Qwen2/3 x*w, weight initialized to one
    unit_offset: bool = False  # Qwen3.5 (1+w), weight initialized to zero
    eps: Optional[float] = None
    bias: Optional[bool] = None
    dtype: str = "float32"

    def to_olmo(self) -> LayerNormConfig:
        if self.qwen and self.unit_offset:
            raise ValueError("qwen and unit_offset select distinct norm semantics")
        config_cls = (
            Qwen35RMSNormConfig
            if self.unit_offset
            else Qwen23RMSNormConfig
            if self.qwen
            else LayerNormConfig
        )
        return config_cls(
            name=LayerNormType.qwen_rms if self.qwen else LayerNormType.rms,
            eps=self.eps,
            bias=False if self.bias is None else self.bias,
            dtype=DType(self.dtype),
        )

    def num_params(self, d_model: int) -> int:
        return self.to_olmo().num_params(d_model)

    def build(self, d_model: int, *, init_device: str = "cpu"):
        return self.to_olmo().build(d_model, init_device=init_device)
