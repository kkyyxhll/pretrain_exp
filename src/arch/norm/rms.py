"""Qwen RMSNorm adapters using OLMo-core contracts (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypedDict

import torch
from olmo_core.config import DType
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.layer_norm import LayerNorm, LayerNormConfig, LayerNormType, RMSNorm


class NormBuildArguments(TypedDict, total=False):
    eps: float
    elementwise_affine: bool
    bias: bool
    full_precision: bool
    dtype: torch.dtype
    weight_shape: tuple[int, ...]


class Qwen23RMSNorm(RMSNorm):
    """Normalize before applying the gain in the input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            hidden_states = x.float()
            hidden_states = (
                hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
            ).to(x.dtype)
            if self.weight is not None:
                hidden_states = hidden_states * self.weight
            else:
                pass
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
            else:
                pass
            return hidden_states.to(x.dtype)


class Qwen35RMSNorm(RMSNorm):
    """Normalize and apply the zero-offset gain in FP32."""

    def reset_parameters(self) -> None:
        if self.weight is not None:
            torch.nn.init.zeros_(self.weight)
        else:
            pass
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        else:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            hidden_states = x.float()
            hidden_states = hidden_states * torch.rsqrt(
                hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
            )
            if self.weight is not None:
                hidden_states = hidden_states * (1.0 + self.weight.float())
            else:
                pass
            if self.bias is not None:
                hidden_states = hidden_states + self.bias.float()
            else:
                pass
            return hidden_states.to(x.dtype)


@dataclass(kw_only=True)
class Qwen35RMSNormConfig(LayerNormConfig):
    """Preserve the norm initialization and meta-materialization contract."""

    norm_cls: ClassVar[type[RMSNorm]] = Qwen35RMSNorm

    def build(
        self,
        size: int,
        init_device: str = "cpu",
        weight_shape: tuple[int, ...] | None = None,
    ) -> RMSNorm:
        norm_arguments: NormBuildArguments = {}
        if self.eps is not None:
            norm_arguments["eps"] = self.eps
        else:
            pass
        if self.elementwise_affine is not None:
            norm_arguments["elementwise_affine"] = self.elementwise_affine
        else:
            pass
        if self.bias is not None:
            norm_arguments["bias"] = self.bias
        else:
            pass
        if self.full_precision is not None:
            norm_arguments["full_precision"] = self.full_precision
        else:
            pass
        if self.dtype is not None:
            norm_arguments["dtype"] = self.dtype.as_pt()
        else:
            pass
        if weight_shape is not None:
            norm_arguments["weight_shape"] = weight_shape
        else:
            pass
        return self.norm_cls(size=size, init_device=init_device, **norm_arguments)


@dataclass(kw_only=True)
class Qwen23RMSNormConfig(Qwen35RMSNormConfig):
    """Select the Qwen2/3 gain arithmetic."""

    norm_cls: ClassVar[type[RMSNorm]] = Qwen23RMSNorm


@dataclass(kw_only=True)
class RmsNormConfig(ModuleConfig):
    """Select Qwen gain arithmetic without changing serialized config keys."""

    qwen: bool = False
    unit_offset: bool = False
    eps: float | None = None
    bias: bool | None = None
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"

    def to_olmo(self) -> LayerNormConfig:
        if self.qwen and self.unit_offset:
            raise ValueError("qwen and unit_offset select distinct norm semantics")
        elif self.unit_offset:
            norm_config_class = Qwen35RMSNormConfig
        elif self.qwen:
            norm_config_class = Qwen23RMSNormConfig
        else:
            norm_config_class = LayerNormConfig
        return norm_config_class(
            name=LayerNormType.qwen_rms if self.qwen else LayerNormType.rms,
            eps=self.eps,
            bias=False if self.bias is None else self.bias,
            dtype=DType(self.dtype),
        )

    def num_params(self, d_model: int) -> int:
        return self.to_olmo().num_params(d_model)

    def build(self, d_model: int, *, init_device: str = "cpu") -> LayerNorm:
        return self.to_olmo().build(d_model, init_device=init_device)
