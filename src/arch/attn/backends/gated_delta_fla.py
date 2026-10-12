"""FLA adapters. Imports are lazy; torch reference has no FLA requirement."""


def causal_conv1d(x, weight):
    from fla.modules.convolution import causal_conv1d as kernel

    return kernel(x, weight, bias=None, activation="silu")[0]


def chunk_gated_delta_rule(q, k, v, g, beta, **kwargs):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as kernel

    return kernel(q, k, v, g, beta, output_final_state=False, **kwargs)[0]
