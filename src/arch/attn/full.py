"""Qwen3.5 full attention adapter using OLMo-core (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.rope import RoPEConfig

from ..core.registry import register

__all__ = ["FullAttentionConfig"]


@register("attn", "full")
@dataclass
class FullAttentionConfig(AttentionConfig):
    """可搜索超参：n_heads / n_kv_heads / head_dim / qk_norm / fused / backend / rope_theta / window。

    GQA = ``n_kv_heads < n_heads``（需整除）；MHA = 相等或 None。
    ``window`` 由 spec.py 按层注入（None = 全局因果），实现为逐层滑动窗口。
    """

    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    qk_norm: bool = False
    use_head_qk_norm: bool = False  # True = q/k norm 逐头（Qwen3 系：norm 维度 = head_dim）
    qk_norm_kind: str = "rms"  # rms | qwen (Qwen3) | qwen35 (unit offset)
    qk_norm_eps: Optional[float] = None
    gate: bool = False  # True = 输出门（Qwen3 系 attn_output_gate，elementwise 粒度）
    gate_full_precision: bool = True
    bias: bool = False
    qkv_bias: bool = False  # True = 仅 q/k/v 投影带 bias（Qwen2 系约定，o_proj 无 bias）
    fused: bool = False  # True → OLMo fused_v2（packed QKV）
    backend: Optional[str] = None  # None → OLMo 自动选择（CPU 回退 torch）
    rope_theta: float = 10000.0
    rope_full_precision: bool = False  # True = fp32 rope（OLMo 默认），False = 跟随输入 dtype
    partial_rotary_factor: float = 1.0  # <1 只对前若干维施 rope（Qwen3.5 用 0.25）
    window: Optional[int] = None

    def to_olmo(self) -> AttentionConfig:
        from ..norm.rms import RmsNormConfig

        if self.qk_norm_kind not in ("rms", "qwen", "qwen35"):
            raise ValueError(f"unsupported qk_norm_kind: {self.qk_norm_kind}")
        qk_config = (
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
            qk_norm=qk_config,
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
        total = self.to_olmo().num_params(d_model)
        if self.qkv_bias and not self.bias:
            hd = self.head_dim if self.head_dim is not None else d_model // self.n_heads
            n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
            total += self.n_heads * hd + 2 * n_kv * hd  # 注入的 q/k/v bias
        return total

    def build(
        self, d_model: int, *, layer_idx: int, n_layers: int, init_device: str = "cpu", cache=None
    ):
        m = self.to_olmo().build(
            d_model, layer_idx=layer_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        if self.qkv_bias and not self.bias:
            # Qwen2 系约定：q/k/v 带 bias、o_proj 无。OLMo AttentionConfig 只有一个全局
            # bias 开关，因此按 bias=False 构建后向 q/k/v 的 Linear 注入 bias 参数
            # （nn.Linear 语义：bias 参数非 None 即参与前向）。
            import torch

            for name in ("w_q", "w_k", "w_v"):
                lin = getattr(m, name)
                lin.bias = torch.nn.Parameter(
                    torch.zeros(lin.out_features, device=lin.weight.device, dtype=lin.weight.dtype)
                )
        return m
