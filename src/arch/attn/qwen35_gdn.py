"""Qwen3.5 GatedDeltaNet, adapted from Transformers 5.12.1 (Apache-2.0).

Source: transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.
The torch reference and FLA CUDA backends share parameters and forward semantics.
Incremental decoding is outside this training-only adapter. FLOPs are counted
by the pinned Megatron reference at the training-module boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.olmo import SequenceMixer, SequenceMixerConfig
from ..core.registry import register

__all__ = ["Qwen35GDNConfig", "Qwen35GatedDeltaNet"]

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class _GatedRMSNorm(nn.Module):
    """transformers Qwen3_5RMSNormGated 的移植（Apache-2.0）：norm → weight → silu(gate)。"""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        h = self.weight * h.to(input_dtype)
        if gate is not None:
            h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


@register("attn", "qwen35_gdn")
@dataclass
class Qwen35GDNConfig(SequenceMixerConfig):
    """Qwen3.5 linear_attention 层的全部可搜索超参（字段名对齐 HF config）。"""

    num_k_heads: int = 16  # linear_num_key_heads
    head_k_dim: int = 128  # linear_key_head_dim
    num_v_heads: int = 16  # linear_num_value_heads
    head_v_dim: int = 128  # linear_value_head_dim
    conv_kernel_size: int = 4  # linear_conv_kernel_dim
    norm_eps: float = 1e-6
    dtype: str = "bfloat16"
    backend: str = "torch"

    def __post_init__(self, type=None):  # type 为 Registrable 基类的 InitVar
        for k in ("num_k_heads", "head_k_dim", "num_v_heads", "head_v_dim", "conv_kernel_size"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} 必须为正: {getattr(self, k)!r}")
        if self.backend not in ("torch", "fla"):
            raise ValueError(f"unsupported GDN backend: {self.backend!r}")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype 仅支持 {sorted(_DTYPES)}: {self.dtype!r}")

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    def num_params(self, d_model: int) -> int:
        V, C = self.value_dim, 2 * self.key_dim + self.value_dim
        total = (
            d_model * C  # in_proj_qkv
            + d_model * V  # in_proj_z
            + 2 * d_model * self.num_v_heads  # in_proj_b / in_proj_a
            + C * self.conv_kernel_size  # 深度卷积（groups=C，每通道 kernel 个权重）
            + V * d_model  # out_proj
            + self.head_v_dim  # 门控 RMSNorm 权重
            + 2 * self.num_v_heads  # dt_bias + A_log
        )
        return total

    def build(
        self, d_model: int, *, layer_idx: int, n_layers: int, init_device: str = "cpu", cache=None
    ):
        return Qwen35GatedDeltaNet(self, d_model, init_device=init_device)


class Qwen35GatedDeltaNet(SequenceMixer):
    """transformers Qwen3_5GatedDeltaNet 的移植（Apache-2.0），去掉 cache/打包路径。"""

    def __init__(self, cfg: Qwen35GDNConfig, d_model: int, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        from importlib import import_module

        backend = import_module(f"{__package__}.backends.gated_delta_{cfg.backend}")
        self._causal_conv1d = backend.causal_conv1d
        self._gated_delta_rule = backend.chunk_gated_delta_rule
        self.d_model = d_model
        dtype = _DTYPES[cfg.dtype]
        V, C = cfg.value_dim, 2 * cfg.key_dim + cfg.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=C,
            out_channels=C,
            bias=False,
            kernel_size=cfg.conv_kernel_size,
            groups=C,
            padding=cfg.conv_kernel_size - 1,
            device=init_device,
            dtype=dtype,
        )
        self.dt_bias = nn.Parameter(torch.ones(cfg.num_v_heads, device=init_device, dtype=dtype))
        A = torch.empty(cfg.num_v_heads, device=init_device).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A).to(dtype))
        self.norm = _GatedRMSNorm(cfg.head_v_dim, eps=cfg.norm_eps, device=init_device, dtype=dtype)
        self.out_proj = nn.Linear(V, d_model, bias=False, device=init_device, dtype=dtype)
        self.in_proj_qkv = nn.Linear(d_model, C, bias=False, device=init_device, dtype=dtype)
        self.in_proj_z = nn.Linear(d_model, V, bias=False, device=init_device, dtype=dtype)
        self.in_proj_b = nn.Linear(
            d_model, cfg.num_v_heads, bias=False, device=init_device, dtype=dtype
        )
        self.in_proj_a = nn.Linear(
            d_model, cfg.num_v_heads, bias=False, device=init_device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, **kwargs):
        if kwargs.get("cache") is not None:
            raise NotImplementedError("qwen35_gdn 增量解码 cache 未实现（训练链路不需要）")
        B, T, D = x.shape
        cfg = self.cfg

        mixed_qkv = self.in_proj_qkv(x)  # backend contract: (B, T, C)
        z = self.in_proj_z(x).reshape(B, T, cfg.num_v_heads, cfg.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if cfg.backend == "fla" and not x.is_cuda:
            raise RuntimeError("GDN backend='fla' requires CUDA; use backend='torch' on CPU")
        mixed_qkv = self._causal_conv1d(mixed_qkv, self.conv1d.weight.squeeze(1))
        q, k, v = torch.split(mixed_qkv, [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1)
        q = q.reshape(B, T, cfg.num_k_heads, cfg.head_k_dim)
        k = k.reshape(B, T, cfg.num_k_heads, cfg.head_k_dim)
        v = v.reshape(B, T, cfg.num_v_heads, cfg.head_v_dim)
        if cfg.num_v_heads != cfg.num_k_heads:
            rep = cfg.num_v_heads // cfg.num_k_heads
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        core_attn_out = self._gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        core_attn_out = core_attn_out.reshape(B * T * cfg.num_v_heads, cfg.head_v_dim)
        z = z.reshape(B * T * cfg.num_v_heads, cfg.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(B, T, cfg.value_dim)
        return self.out_proj(core_attn_out)

    def num_flops_per_token(self, seq_len: int) -> int:
        raise NotImplementedError("Use the pinned Megatron FLOPs reference in the training module")

    def apply_tp(
        self,
        tp_mesh,
        input_layout=None,
        output_layout=None,
        use_local_output=True,
        float8_enabled=False,
    ):
        raise NotImplementedError("qwen35_gdn 的 TP 未实现")

    def apply_cp(self, cp_mesh, ring=None, uly=None):
        raise NotImplementedError("qwen35_gdn 的 CP 未实现")

    @torch.no_grad()
    def init_weights(
        self, *, init_method, d_model, block_idx, num_blocks, std=0.02, generator=None
    ):
        """Initialize every parameter after meta materialization, including FSDP shards."""
        from olmo_core.nn.transformer.init import InitMethod, _apply_init, init_linear

        for layer in (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
            self.conv1d,
        ):
            layer_std = layer.weight.shape[1] ** -0.5 if init_method == InitMethod.fan_in else std
            init_linear(layer, std=layer_std, generator=generator)
        out_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            out_std /= (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            out_std /= (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.fan_in:
            out_std = self.cfg.value_dim**-0.5
        init_linear(self.out_proj, std=out_std, generator=generator)

        def init_decay(tensor):
            # Same positive decay distribution as the constructor, generated in FP32.
            values = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
            values.uniform_(0, 16, generator=generator).clamp_min_(torch.finfo(torch.float32).tiny)
            tensor.copy_(values.log())

        _apply_init(init_decay, self.A_log)
        nn.init.ones_(self.dt_bias)
        nn.init.ones_(self.norm.weight)
