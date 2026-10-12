"""Torch GatedDeltaNet reference adapted from Transformers (Apache-2.0)."""

import torch
import torch.nn.functional as functional

L2_NORMALIZATION_EPSILON = 1e-6
DEFAULT_CHUNK_TOKENS = 64


def l2_normalize(hidden_states: torch.Tensor, dimension: int, epsilon: float) -> torch.Tensor:
    return hidden_states * torch.rsqrt(
        (hidden_states * hidden_states).sum(dim=dimension, keepdim=True) + epsilon
    )


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Apply the chunkwise gated delta recurrence."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2_normalize(query, dimension=-1, epsilon=L2_NORMALIZATION_EPSILON)
        key = l2_normalize(key, dimension=-1, epsilon=L2_NORMALIZATION_EPSILON)
    else:
        pass
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, g)
    ]
    batch_size, head_count, sequence_tokens, key_head_channels = key.shape
    value_head_channels = value.shape[-1]
    padding_tokens = (chunk_size - sequence_tokens % chunk_size) % chunk_size
    query = functional.pad(query, (0, 0, 0, padding_tokens))
    key = functional.pad(key, (0, 0, 0, padding_tokens))
    value = functional.pad(value, (0, 0, 0, padding_tokens))
    beta = functional.pad(beta, (0, padding_tokens))
    g = functional.pad(g, (0, padding_tokens))
    padded_sequence_tokens = sequence_tokens + padding_tokens
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    gated_values = value * beta.unsqueeze(-1)
    gated_keys = key * beta.unsqueeze(-1)
    query, key, value, gated_keys, gated_values = [
        tensor.reshape(tensor.shape[0], tensor.shape[1], -1, chunk_size, tensor.shape[-1])
        for tensor in (query, key, value, gated_keys, gated_values)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attention_weights = -((gated_keys @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row_weights = attention_weights[..., i, :i].clone()
        previous_weights = attention_weights[..., :i, :i].clone()
        attention_weights[..., i, :i] = row_weights + (
            row_weights.unsqueeze(-1) * previous_weights
        ).sum(-2)
    attention_weights = attention_weights + torch.eye(
        chunk_size, dtype=attention_weights.dtype, device=attention_weights.device
    )
    value = attention_weights @ gated_values
    decayed_keys = attention_weights @ (gated_keys * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(
        batch_size,
        head_count,
        key_head_channels,
        value_head_channels,
        dtype=value.dtype,
        device=value.device,
    )
    attention_output = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, padded_sequence_tokens // chunk_size):
        chunk_query, chunk_key, chunk_value = query[:, :, i], key[:, :, i], value[:, :, i]
        attention_weights = (
            chunk_query @ chunk_key.transpose(-1, -2) * decay_mask[:, :, i]
        ).masked_fill(mask, 0)
        predicted_values = (decayed_keys[:, :, i]) @ last_recurrent_state
        updated_values = chunk_value - predicted_values
        recurrent_attention = (chunk_query * g[:, :, i, :, None].exp()) @ last_recurrent_state
        attention_output[:, :, i] = recurrent_attention + attention_weights @ updated_values
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (chunk_key * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ updated_values
        )

    attention_output = attention_output.reshape(
        attention_output.shape[0], attention_output.shape[1], -1, attention_output.shape[-1]
    )
    attention_output = attention_output[:, :, :sequence_tokens]
    attention_output = attention_output.transpose(1, 2).contiguous().to(initial_dtype)
    return attention_output


def causal_conv1d(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Convolve batch-token-channel inputs with per-channel kernels."""
    channel_count = hidden_states.shape[-1]
    convolved_states = functional.conv1d(
        hidden_states.transpose(1, 2),
        weight.unsqueeze(1),
        groups=channel_count,
        padding=weight.shape[-1] - 1,
    )
    return functional.silu(convolved_states[:, :, : hidden_states.shape[1]]).transpose(1, 2)
