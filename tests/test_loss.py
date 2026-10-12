"""Chunking must preserve loss normalization, masking and all parameter gradients."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from olmo_core.nn.lm_head import LMHead

from train.loss import ChunkedLMHead


class TestChunkedLoss(unittest.TestCase):
    def test_native_loss_and_gradients(self):
        torch.manual_seed(7)
        for reduction in ("mean", "sum", "none"):
            for z in (None, 1e-4):
                a = LMHead(d_model=8, vocab_size=23, bias=True)
                b = ChunkedLMHead(d_model=8, vocab_size=23, bias=True, chunk_tokens=7)
                b.load_state_dict(a.state_dict())
                x = torch.randn(2, 11, 8, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                labels = torch.randint(0, 23, (2, 11))
                labels[:, :7] = -100
                opts = dict(
                    labels=labels,
                    loss_reduction=reduction,
                    loss_div_factor=9.0,
                    z_loss_multiplier=z,
                    return_logits=False,
                )
                expected, actual = a(x, **opts), b(y, **opts)
                # OLMo does not mask z-loss for reduction=none. Our token loss deliberately does.
                if reduction == "none" and z is not None:
                    expected = expected._replace(
                        loss=expected.loss * (labels != -100),
                        z_loss=expected.z_loss * (labels != -100),
                    )
                torch.testing.assert_close(actual.loss, expected.loss)
                expected.loss.sum().backward()
                actual.loss.sum().backward()
                torch.testing.assert_close(x.grad, y.grad)
                for p, q in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(p.grad, q.grad)
                self.assertEqual(set(a.state_dict()), set(b.state_dict()))

    def test_ignored_sum_remains_differentiable(self):
        b = ChunkedLMHead(d_model=8, vocab_size=23, chunk_tokens=3)
        x = torch.randn(2, 5, 8, requires_grad=True)
        out = b(x, labels=torch.full((2, 5), -100), loss_reduction="sum")
        self.assertEqual(out.loss.item(), 0)
        out.loss.backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_logits_and_selection(self):
        a = LMHead(d_model=8, vocab_size=23)
        b = ChunkedLMHead(d_model=8, vocab_size=23, chunk_tokens=3)
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 5, 8)
        for keep in (0, 2, torch.tensor([[0, 3], [1, 4]])):
            torch.testing.assert_close(a(x, logits_to_keep=keep), b(x, logits_to_keep=keep))
