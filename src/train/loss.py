"""Training loss implementations; architecture and state-dict layout stay unchanged."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.lm_head import LMHead, LMHeadConfig, LMOutputWithLoss
from torch.utils.checkpoint import checkpoint


@dataclass
class ChunkedLMHeadConfig(LMHeadConfig):
    chunk_tokens: int = 1024

    def build(self, *, d_model, vocab_size, init_device="cpu"):
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
    """Checkpoint projection + FP32 CE per token chunk, never retaining B*T*V.

    Only the training loss is chunked. Requests for logits use the native head
    and necessarily materialize their requested output. Parameter names remain
    w_out/norm, including tied embeddings and FSDP state dicts.
    """

    def __init__(self, *, chunk_tokens=1024, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be a positive integer")
        self.chunk_tokens = chunk_tokens

    def forward(
        self,
        x,
        *,
        labels=None,
        ignore_index=-100,
        loss_reduction="mean",
        z_loss_multiplier=None,
        loss_div_factor=None,
        return_logits=None,
        logits_to_keep=0,
    ):
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
        if self.tp_enabled or self.cp_enabled:
            raise NotImplementedError("torch_chunked loss currently supports data parallelism only")
        if loss_reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid loss reduction: {loss_reduction}")
        h = self.norm(x) if self.norm is not None else x
        if isinstance(logits_to_keep, int):
            if logits_to_keep:
                h, labels = h[:, -logits_to_keep:], labels[:, -logits_to_keep:]
        else:
            h = h.gather(1, logits_to_keep.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            labels = labels.gather(1, logits_to_keep)
        batch = h.shape[0]
        h = get_local_tensor(h).reshape(-1, self.d_model)
        labels = get_local_tensor(labels).reshape(-1)
        weight = get_local_tensor(self.w_out.weight)
        bias = get_local_tensor(self.w_out.bias) if self.w_out.bias is not None else None

        def chunk_loss(hidden, target, weight, bias):
            logits = F.linear(hidden, weight, bias).float()
            ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
            if z_loss_multiplier is None:
                z = torch.zeros_like(ce)
            else:
                z = logits.logsumexp(-1).square() * z_loss_multiplier
                z = z * (target != ignore_index)
            return ce, z

        ces, zs = [], []
        for start in range(0, h.shape[0], self.chunk_tokens):
            ce, z = checkpoint(
                chunk_loss,
                h[start : start + self.chunk_tokens],
                labels[start : start + self.chunk_tokens],
                weight,
                bias,
                use_reentrant=False,
            )
            ces.append(ce)
            zs.append(z)
        ce, z = torch.cat(ces), torch.cat(zs)
        if loss_reduction != "none":
            ce, z = ce.sum(), z.sum()
            if loss_reduction == "mean":
                count = (labels != ignore_index).sum()
                ce, z = ce / count, z / count

        def finalize(value):
            return self._finalize_loss(
                value, batch, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
            )

        return LMOutputWithLoss(
            logits=None,
            loss=finalize(ce + z),
            ce_loss=finalize(ce.detach()),
            z_loss=finalize(z.detach()) if z_loss_multiplier is not None else None,
        )
