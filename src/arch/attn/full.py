"""Qwen3.5 full attention adapter using OLMo-core (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from olmo_core.config import DType
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.attention.base import SequenceMixer
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.rope import RoPEConfig

from src.arch.norm.rms import RmsNormConfig


@dataclass(kw_only=True)
class FullAttentionConfig(AttentionConfig):
    """Configure full attention while retaining serialized parameter names."""

    n_heads: int = 16
    n_kv_heads: int | None = None
    head_dim: int | None = None
    qk_norm: bool = False
    use_head_qk_norm: bool = False
    qk_norm_kind: Literal["rms", "qwen", "qwen35"] = "rms"
    qk_norm_eps: float | None = None
    gate: bool = False
    gate_full_precision: bool = True
    bias: bool = False
    qkv_bias: bool = False
    fused: bool = False
    backend: (
        AttentionBackendName | Literal["torch", "flash_2", "flash_3", "flash_4", "te"] | None
    ) = None
    rope_theta: float = 10000.0
    rope_full_precision: bool = False
    partial_rotary_factor: float = 1.0
    window: int | None = None

    def to_olmo(self) -> AttentionConfig:
        if self.qk_norm_kind not in ("rms", "qwen", "qwen35"):
            raise ValueError(f"unsupported qk_norm_kind: {self.qk_norm_kind}")
        else:
            query_key_norm = (
                RmsNormConfig(
                    qwen=self.qk_norm_kind == "qwen",
                    unit_offset=self.qk_norm_kind == "qwen35",
                    eps=self.qk_norm_eps,
                ).to_olmo()
                if self.qk_norm
                else None
            )
        return AttentionConfig(
            name=AttentionType.fused_v2 if self.fused else AttentionType.default,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            bias=self.bias,
            qk_norm=query_key_norm,
            use_head_qk_norm=self.use_head_qk_norm,
            gate=GateConfig(
                granularity=GateGranularity.elementwise, full_precision=self.gate_full_precision
            )
            if self.gate
            else None,
            rope=RoPEConfig(
                theta=int(self.rope_theta),
                full_precision=self.rope_full_precision,
                partial_rotary_factor=self.partial_rotary_factor,
            ),
            dtype=self.dtype if isinstance(self.dtype, DType) else DType(self.dtype),
            backend=AttentionBackendName(self.backend) if self.backend else None,
            sliding_window=(
                SlidingWindowAttentionConfig(
                    pattern=[self.window],
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=False,
                )
                if self.window is not None
                else None
            ),
        )

    def num_params(self, d_model: int) -> int:
        parameter_count = self.to_olmo().num_params(d_model)
        if self.qkv_bias and not self.bias:
            head_channels = self.head_dim if self.head_dim is not None else d_model // self.n_heads
            key_value_head_count = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
            parameter_count += (
                self.n_heads * head_channels + 2 * key_value_head_count * head_channels
            )
        else:
            pass
        return parameter_count

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: BufferCache | None = None,
    ) -> SequenceMixer:
        attention = self.to_olmo().build(
            d_model, layer_idx=layer_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        if self.qkv_bias and not self.bias:
            assert isinstance(attention, Attention)
            for projection in (attention.w_q, attention.w_k, attention.w_v):
                projection.bias = torch.nn.Parameter(
                    torch.zeros(
                        projection.out_features,
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    )
                )
        else:
            pass
        return attention
