"""Compute checkpointed token-chunk losses without retaining full vocabulary logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHead, LMHeadConfig, LMLossImplementation, LMOutputWithLoss
from torch.utils.checkpoint import checkpoint


@dataclass
class ChunkedLMHeadConfig(LMHeadConfig):
    chunk_tokens: int = 1024

    def build(self, *, d_model: int, vocab_size: int, init_device: str = "cpu") -> ChunkedLMHead:
        return ChunkedLMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            layer_norm=self.layer_norm,
            bias=self.bias,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
            chunk_tokens=self.chunk_tokens,
        )


class ChunkedLMHead(LMHead):
    """Keep the original parameter layout while bounding projected loss memory."""

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        layer_norm: LayerNormConfig | None = None,
        dtype: torch.dtype = torch.float32,
        bias: bool = True,
        init_device: str = "cpu",
        loss_implementation: LMLossImplementation = LMLossImplementation.default,
        chunk_tokens: int = 1024,
    ) -> None:
        if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be a positive integer")
        else:
            super().__init__(
                d_model=d_model,
                vocab_size=vocab_size,
                layer_norm=layer_norm,
                dtype=dtype,
                bias=bias,
                init_device=init_device,
                loss_implementation=loss_implementation,
            )
            self.chunk_tokens: int = chunk_tokens

    def forward(
        self,
        x: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: float | None = None,
        loss_div_factor: torch.Tensor | float | None = None,
        return_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> torch.Tensor | LMOutputWithLoss:
        if labels is None or return_logits is True:
            return super().forward(
                x,
                labels=labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                logits_to_keep=logits_to_keep,
            )
        else:
            pass
        if self.tp_enabled or self.cp_enabled:
            raise NotImplementedError("torch_chunked loss currently supports data parallelism only")
        else:
            pass
        if loss_reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid loss reduction: {loss_reduction}")
        else:
            pass
        hidden_states = self.norm(x) if self.norm is not None else x
        if isinstance(logits_to_keep, int):
            if logits_to_keep:
                hidden_states, labels = (
                    hidden_states[:, -logits_to_keep:],
                    labels[:, -logits_to_keep:],
                )
            else:
                pass
        else:
            hidden_states = hidden_states.gather(
                1, logits_to_keep.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            )
            labels = labels.gather(1, logits_to_keep)
        batch_sequences = hidden_states.shape[0]
        hidden_states = get_local_tensor(hidden_states).reshape(-1, self.d_model)
        labels = get_local_tensor(labels).reshape(-1)
        weight = get_local_tensor(self.w_out.weight)
        bias = get_local_tensor(self.w_out.bias) if self.w_out.bias is not None else None

        def chunk_loss(
            hidden_states: torch.Tensor,
            target_tokens: torch.Tensor,
            vocabulary_weight: torch.Tensor,
            vocabulary_bias: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = F.linear(hidden_states, vocabulary_weight, vocabulary_bias).float()
            cross_entropy = F.cross_entropy(
                logits, target_tokens, ignore_index=ignore_index, reduction="none"
            )
            if z_loss_multiplier is None:
                z_loss = torch.zeros_like(cross_entropy)
            else:
                z_loss = logits.logsumexp(-1).square() * z_loss_multiplier
                z_loss = z_loss * (target_tokens != ignore_index)
            return cross_entropy, z_loss

        cross_entropy_chunks: list[torch.Tensor] = []
        z_loss_chunks: list[torch.Tensor] = []
        for start_token in range(0, hidden_states.shape[0], self.chunk_tokens):
            cross_entropy, z_loss = checkpoint(
                chunk_loss,
                hidden_states[start_token : start_token + self.chunk_tokens],
                labels[start_token : start_token + self.chunk_tokens],
                weight,
                bias,
                use_reentrant=False,
            )
            cross_entropy_chunks.append(cross_entropy)
            z_loss_chunks.append(z_loss)
        cross_entropy, z_loss = torch.cat(cross_entropy_chunks), torch.cat(z_loss_chunks)
        if loss_reduction != "none":
            cross_entropy, z_loss = cross_entropy.sum(), z_loss.sum()
            if loss_reduction == "mean":
                scored_tokens = (labels != ignore_index).sum()
                cross_entropy, z_loss = cross_entropy / scored_tokens, z_loss / scored_tokens
            else:
                pass
        else:
            pass

        def finalize(loss: torch.Tensor) -> torch.Tensor:
            return self._finalize_loss(
                loss,
                batch_sequences,
                loss_reduction=loss_reduction,
                loss_div_factor=loss_div_factor,
            )

        return LMOutputWithLoss(
            logits=None,
            loss=finalize(cross_entropy + z_loss),
            ce_loss=finalize(cross_entropy.detach()),
            z_loss=finalize(z_loss.detach()) if z_loss_multiplier is not None else None,
        )
