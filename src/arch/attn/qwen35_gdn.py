"""Qwen3.5 GatedDeltaNet adapted from Transformers (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as functional
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.transformer.init import InitMethod, _apply_init, init_linear
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from src.arch.attn.backends.gated_delta_fla import (
    causal_conv1d as fla_causal_convolution,
)
from src.arch.attn.backends.gated_delta_fla import (
    chunk_gated_delta_rule as fla_gated_delta_rule,
)
from src.arch.attn.backends.gated_delta_torch import (
    causal_conv1d as torch_causal_convolution,
)
from src.arch.attn.backends.gated_delta_torch import (
    chunk_gated_delta_rule as torch_gated_delta_rule,
)

PARAMETER_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
DECAY_INITIALIZATION_MAXIMUM = 16


class CausalConvolution(Protocol):
    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...


class GatedDeltaRule(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        update_gate: torch.Tensor,
        /,
        *,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> torch.Tensor: ...


class GatedRMSNorm(nn.Module):
    """Normalize, apply gain, and apply the output gate."""

    def __init__(
        self,
        channel_count: int,
        *,
        eps: float,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.ones(channel_count, device=device, dtype=dtype)
        )
        self.eps: float = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = x.dtype
        hidden_states = x.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * functional.silu(gate.to(torch.float32))
        else:
            pass
        return hidden_states.to(input_dtype)


@dataclass(kw_only=True)
class Qwen35GDNConfig(SequenceMixerConfig):
    """GatedDeltaNet dimensions with stable serialized configuration keys."""

    num_k_heads: int = 16
    head_k_dim: int = 128
    num_v_heads: int = 16
    head_v_dim: int = 128
    conv_kernel_size: int = 4
    norm_eps: float = 1e-6
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    backend: Literal["torch", "fla"] = "torch"

    def __post_init__(self, type: str | None = None) -> None:
        # note (kkyyxhll): Registrable requires the unused type InitVar.
        for dimension_name, dimension_count in (
            ("num_k_heads", self.num_k_heads),
            ("head_k_dim", self.head_k_dim),
            ("num_v_heads", self.num_v_heads),
            ("head_v_dim", self.head_v_dim),
            ("conv_kernel_size", self.conv_kernel_size),
        ):
            if dimension_count <= 0:
                raise ValueError(f"{dimension_name} must be positive: {dimension_count!r}")
            else:
                pass
        if self.backend not in ("torch", "fla"):
            raise ValueError(f"unsupported GDN backend: {self.backend!r}")
        else:
            pass
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        else:
            pass
        if self.dtype not in PARAMETER_DTYPES:
            raise ValueError(f"unsupported dtype: {self.dtype!r}")
        else:
            pass

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    def num_params(self, d_model: int) -> int:
        value_channels = self.value_dim
        projection_channels = 2 * self.key_dim + self.value_dim
        return (
            d_model * projection_channels
            + d_model * value_channels
            + 2 * d_model * self.num_v_heads
            + projection_channels * self.conv_kernel_size
            + value_channels * d_model
            + self.head_v_dim
            + 2 * self.num_v_heads
        )

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: BufferCache | None = None,
    ) -> Qwen35GatedDeltaNet:
        # note (kkyyxhll): SequenceMixerConfig requires the unused layer and cache parameters.
        return Qwen35GatedDeltaNet(self, d_model, init_device=init_device)


class Qwen35GatedDeltaNet(SequenceMixer):
    """Training-only GatedDeltaNet with interchangeable Torch and FLA kernels."""

    def __init__(self, config: Qwen35GDNConfig, d_model: int, init_device: str = "cpu") -> None:
        super().__init__()
        self.config: Qwen35GDNConfig = config
        self.causal_convolution: CausalConvolution
        self.gated_delta_rule: GatedDeltaRule
        if config.backend == "fla":
            self.causal_convolution = fla_causal_convolution
            self.gated_delta_rule = fla_gated_delta_rule
        else:
            self.causal_convolution = torch_causal_convolution
            self.gated_delta_rule = torch_gated_delta_rule
        self.d_model: int = d_model
        parameter_dtype = PARAMETER_DTYPES[config.dtype]
        value_channels = config.value_dim
        projection_channels = 2 * config.key_dim + config.value_dim
        self.conv1d: nn.Conv1d = nn.Conv1d(
            in_channels=projection_channels,
            out_channels=projection_channels,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=projection_channels,
            padding=config.conv_kernel_size - 1,
            device=init_device,
            dtype=parameter_dtype,
        )
        self.dt_bias: nn.Parameter = nn.Parameter(
            torch.ones(config.num_v_heads, device=init_device, dtype=parameter_dtype)
        )
        decay_rates = torch.empty(config.num_v_heads, device=init_device).uniform_(
            0, DECAY_INITIALIZATION_MAXIMUM
        )
        self.A_log: nn.Parameter = nn.Parameter(torch.log(decay_rates).to(parameter_dtype))
        self.norm: GatedRMSNorm = GatedRMSNorm(
            config.head_v_dim,
            eps=config.norm_eps,
            device=init_device,
            dtype=parameter_dtype,
        )
        self.out_proj: nn.Linear = nn.Linear(
            value_channels, d_model, bias=False, device=init_device, dtype=parameter_dtype
        )
        self.in_proj_qkv: nn.Linear = nn.Linear(
            d_model, projection_channels, bias=False, device=init_device, dtype=parameter_dtype
        )
        self.in_proj_z: nn.Linear = nn.Linear(
            d_model, value_channels, bias=False, device=init_device, dtype=parameter_dtype
        )
        self.in_proj_b: nn.Linear = nn.Linear(
            d_model, config.num_v_heads, bias=False, device=init_device, dtype=parameter_dtype
        )
        self.in_proj_a: nn.Linear = nn.Linear(
            d_model, config.num_v_heads, bias=False, device=init_device, dtype=parameter_dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: torch.Tensor | None = None,
        cu_doc_lens_q: torch.Tensor | None = None,
        cu_doc_lens_k: torch.Tensor | None = None,
        max_doc_len: int | None = None,
        max_doc_len_q: int | None = None,
        max_doc_len_k: int | None = None,
        local_k_slice: slice | None = None,
        pos_sin: torch.Tensor | None = None,
        pos_cos: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        cache_leftpad: torch.Tensor | None = None,
        cache: BufferCache | None = None,
    ) -> torch.Tensor:
        # note (kkyyxhll): The attention interface supplies unused packing and rotary parameters.
        if cache is not None:
            raise NotImplementedError("incremental GDN decoding is not supported")
        else:
            pass
        batch_size, sequence_tokens = x.shape[:2]
        config = self.config
        mixed_projections = self.in_proj_qkv(x)
        output_gate = self.in_proj_z(x).reshape(
            batch_size, sequence_tokens, config.num_v_heads, config.head_v_dim
        )
        update_gate_logits = self.in_proj_b(x)
        decay_logits = self.in_proj_a(x)
        update_gate = update_gate_logits.sigmoid()
        decay = -self.A_log.float().exp() * functional.softplus(decay_logits.float() + self.dt_bias)

        if config.backend == "fla" and not x.is_cuda:
            raise RuntimeError("GDN backend='fla' requires CUDA; use backend='torch' on CPU")
        else:
            mixed_projections = self.causal_convolution(
                mixed_projections, self.conv1d.weight.squeeze(1)
            )
        query, key, value = torch.split(
            mixed_projections, [config.key_dim, config.key_dim, config.value_dim], dim=-1
        )
        query = query.reshape(batch_size, sequence_tokens, config.num_k_heads, config.head_k_dim)
        key = key.reshape(batch_size, sequence_tokens, config.num_k_heads, config.head_k_dim)
        value = value.reshape(batch_size, sequence_tokens, config.num_v_heads, config.head_v_dim)
        if config.num_v_heads != config.num_k_heads:
            head_repetitions = config.num_v_heads // config.num_k_heads
            query = query.repeat_interleave(head_repetitions, dim=2)
            key = key.repeat_interleave(head_repetitions, dim=2)
        else:
            pass
        attention_output = self.gated_delta_rule(
            query, key, value, decay, update_gate, use_qk_l2norm_in_kernel=True
        )
        attention_output = attention_output.reshape(
            batch_size * sequence_tokens * config.num_v_heads, config.head_v_dim
        )
        output_gate = output_gate.reshape(
            batch_size * sequence_tokens * config.num_v_heads, config.head_v_dim
        )
        attention_output = self.norm(attention_output, output_gate)
        attention_output = attention_output.reshape(batch_size, sequence_tokens, config.value_dim)
        return self.out_proj(attention_output)

    def num_flops_per_token(self, seq_len: int) -> int:
        raise NotImplementedError("Use the pinned Megatron FLOPs reference in the training module")

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        raise NotImplementedError("GDN tensor parallelism is not supported")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ) -> None:
        raise NotImplementedError("GDN context parallelism is not supported")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: InitMethod,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize materialized parameters, including sharded parameters."""
        for projection in (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
            self.conv1d,
        ):
            projection_deviation = (
                projection.weight.shape[1] ** -0.5 if init_method == InitMethod.fan_in else std
            )
            init_linear(projection, std=projection_deviation, generator=generator)
        output_deviation = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            output_deviation /= (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_deviation /= (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.fan_in:
            output_deviation = self.config.value_dim**-0.5
        else:
            pass
        init_linear(self.out_proj, std=output_deviation, generator=generator)

        def initialize_decay(decay_tensor: torch.Tensor) -> None:
            decay_rates = torch.empty(
                decay_tensor.shape, device=decay_tensor.device, dtype=torch.float32
            )
            decay_rates.uniform_(0, DECAY_INITIALIZATION_MAXIMUM, generator=generator).clamp_min_(
                torch.finfo(torch.float32).tiny
            )
            decay_tensor.copy_(decay_rates.log())

        _apply_init(initialize_decay, self.A_log)
        nn.init.ones_(self.dt_bias)
        nn.init.ones_(self.norm.weight)
