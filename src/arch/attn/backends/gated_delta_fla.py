"""FLA convolution and GatedDeltaNet adapters."""

import torch
from fla.modules.convolution import causal_conv1d as causal_convolution_kernel
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as gated_delta_kernel


def causal_conv1d(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return causal_convolution_kernel(hidden_states, weight, bias=None, activation="silu")[0]


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    update_gate: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    return gated_delta_kernel(
        query,
        key,
        value,
        decay,
        update_gate,
        output_final_state=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )[0]
