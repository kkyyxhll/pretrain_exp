"""Single-stream Engram: normalized token hashes, gated memory, and causal convolution.

The paper's dot-product gate is the default. The official demo's signed-square-root
gate is an explicit alternative. Hashes match the demo inside each document; EOS
boundaries additionally reset hash and convolution context in packed training rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Regex, normalizers
from torch import nn
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class EngramConfig:
    """Memory dimensions and immutable hashing identity; layer IDs are zero-based."""

    layer_ids: tuple[int, ...] = (1, 15)
    table_sizes: tuple[int, ...] = (32768, 32768)
    max_ngram_size: int = 3
    heads: int = 8
    head_dim: int = 64
    kernel_size: int = 4
    dilation: int = 3
    seed: int = 20261011
    pad_id: int = 248044
    eos_id: int = 248044
    gate_variant: Literal["dot", "signed_sqrt"] = "dot"
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def validate(self) -> None:
        if not isinstance(self.layer_ids, tuple) or not isinstance(self.table_sizes, tuple):
            raise ValueError("Engram layer IDs and table sizes must be immutable tuples")
        integer_values = (
            *self.layer_ids,
            *self.table_sizes,
            self.max_ngram_size,
            self.heads,
            self.head_dim,
            self.kernel_size,
            self.dilation,
            self.seed,
            self.pad_id,
            self.eos_id,
        )
        if any(type(value) is not int for value in integer_values):
            raise ValueError("Engram dimensions, layer IDs, seed, and token IDs must be integers")
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("Engram layer IDs must be nonempty and unique")
        if any(layer_id < 0 for layer_id in self.layer_ids):
            raise ValueError("Engram layer IDs must be zero-based nonnegative integers")
        if self.max_ngram_size < 2:
            raise ValueError("Engram needs at least bigram memory")
        if len(self.table_sizes) != self.max_ngram_size - 1:
            raise ValueError("Provide one table size per n-gram order from 2 to N")
        if any(size < 2 for size in self.table_sizes):
            raise ValueError("Each Engram hash table needs at least two rows")
        if min(self.heads, self.head_dim, self.kernel_size, self.dilation) < 1:
            raise ValueError("Engram head, convolution, and dilation sizes must be positive")
        if min(self.seed, self.pad_id, self.eos_id) < 0:
            raise ValueError("Engram seed and special token IDs must be nonnegative")
        if self.gate_variant not in ("dot", "signed_sqrt"):
            raise ValueError("Engram gate must be dot or signed_sqrt")
        if not math.isfinite(self.norm_eps) or not math.isfinite(self.init_std):
            raise ValueError("Engram normalization epsilon and initialization std must be finite")
        if self.norm_eps <= 0 or self.init_std <= 0:
            raise ValueError("Engram normalization epsilon and initialization std must be positive")

    @property
    def num_hash_heads(self) -> int:
        return (self.max_ngram_size - 1) * self.heads

    @property
    def memory_dim(self) -> int:
        return self.num_hash_heads * self.head_dim


def build_token_map(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Project an existing tokenizer's vocabulary without reading training text."""
    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    canonical_ids: dict[str, int] = {}
    token_map: list[int] = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.convert_ids_to_tokens(token_id)
        else:
            key = normalize.normalize_str(text) or text
        if not isinstance(key, str):
            raise ValueError(f"Tokenizer ID {token_id} has no vocabulary string")
        if key not in canonical_ids:
            canonical_ids[key] = len(canonical_ids)
        token_map.append(canonical_ids[key])
    return torch.tensor(token_map, dtype=torch.long)


def load_token_map(path: Path) -> torch.Tensor:
    """Load a non-pickled vocabulary projection produced from the fixed tokenizer."""
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError("Engram token map must be a one-dimensional NumPy array")
    if values.dtype != np.int64:
        raise ValueError("Engram token map must contain int64 IDs")
    return torch.from_numpy(values.copy())


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _layer_primes(config: EngramConfig, layer_id: int) -> torch.Tensor:
    """Use the demo's successive unique primes across all configured layers."""
    used_primes: set[int] = set()
    for current_layer in config.layer_ids:
        primes: list[int] = []
        for table_size in config.table_sizes:
            candidate = table_size
            for _ in range(config.heads):
                while candidate in used_primes or not _is_prime(candidate):
                    candidate += 1
                primes.append(candidate)
                used_primes.add(candidate)
                candidate += 1
        if current_layer == layer_id:
            return torch.tensor(primes, dtype=torch.long)
    raise ValueError(f"Layer {layer_id} is not an Engram layer")


def _document_segments(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    if input_ids.shape[1] == 0:
        return torch.zeros_like(input_ids)
    starts = F.pad(input_ids[:, :-1].eq(eos_id), (1, 0))
    return starts.long().cumsum(dim=1)


def _shift_within_document(
    values: torch.Tensor,
    distance: int,
    segment_ids: torch.Tensor,
    padding_value: int,
) -> torch.Tensor:
    if distance == 0:
        return values
    if distance >= values.shape[1]:
        return torch.full_like(values, padding_value)
    padding = torch.full_like(values[:, :distance], padding_value)
    shifted = torch.cat((padding, values[:, :-distance]), dim=1)
    same_document = segment_ids[:, distance:] == segment_ids[:, :-distance]
    same_document = F.pad(same_document, (distance, 0))
    if values.ndim == 3:
        same_document = same_document.unsqueeze(-1)
    return torch.where(same_document, shifted, padding_value)


class _MemoryEmbedding(nn.Embedding):
    def __init__(
        self,
        rows: int,
        width: int,
        init_std: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        self.init_std = init_std
        super().__init__(rows, width, sparse=False, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=self.init_std)


class _MemoryProjection(nn.Linear):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        init_std: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        self.init_std = init_std
        super().__init__(input_width, output_width, bias=True, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=self.init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Engram(nn.Module):
    """GPU-resident trainable memory that returns a delta for one residual stream."""

    def __init__(
        self,
        d_model: int,
        layer_id: int,
        config: EngramConfig,
        token_map: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if d_model < 1:
            raise ValueError("Engram hidden width must be positive")
        if token_map.is_meta or token_map.ndim != 1 or token_map.dtype != torch.long:
            raise ValueError("Engram token map must be a materialized one-dimensional int64 tensor")
        token_map_cpu = token_map.detach().to(device="cpu").clone()
        if token_map_cpu.numel() <= max(config.pad_id, config.eos_id):
            raise ValueError("Engram special token IDs must be inside the original vocabulary")
        canonical_count = int(token_map_cpu.max()) + 1
        if not torch.equal(torch.unique(token_map_cpu), torch.arange(canonical_count)):
            raise ValueError("Engram canonical token IDs must be contiguous from zero")

        self.d_model = d_model
        self.layer_id = layer_id
        self.config = config
        self.compressed_pad_id = int(token_map_cpu[config.pad_id])
        self._token_map_cpu = token_map_cpu
        self._primes_cpu = _layer_primes(config, layer_id)
        self._offsets_cpu = F.pad(self._primes_cpu.cumsum(0)[:-1], (1, 0))
        multiplier_bound = max(1, (np.iinfo(np.int64).max // canonical_count) // 2)
        random = np.random.default_rng(config.seed + 10007 * layer_id)
        multipliers = random.integers(
            0, multiplier_bound, size=config.max_ngram_size, dtype=np.int64
        )
        self._multipliers_cpu = torch.from_numpy(multipliers * 2 + 1)
        self.register_buffer("token_map", self._token_map_cpu.to(device=device))
        self.register_buffer("primes", self._primes_cpu.to(device=device))
        self.register_buffer("offsets", self._offsets_cpu.to(device=device))
        self.register_buffer("multipliers", self._multipliers_cpu.to(device=device))

        table_rows = int(self._primes_cpu.sum())
        self.embedding = _MemoryEmbedding(
            table_rows, config.head_dim, config.init_std, device, dtype
        )
        self.key_proj = _MemoryProjection(
            config.memory_dim, d_model, config.init_std, device, dtype
        )
        self.value_proj = _MemoryProjection(
            config.memory_dim, d_model, config.init_std, device, dtype
        )
        self.query_norm = nn.RMSNorm(d_model, eps=config.norm_eps, device=device, dtype=dtype)
        self.key_norm = nn.RMSNorm(d_model, eps=config.norm_eps, device=device, dtype=dtype)
        self.conv_norm = nn.RMSNorm(d_model, eps=config.norm_eps, device=device, dtype=dtype)
        self.conv_weight = nn.Parameter(
            torch.empty(d_model, 1, config.kernel_size, device=device, dtype=dtype)
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Restore constant buffers after OLMo materializes a meta-built module."""
        self.token_map.copy_(self._token_map_cpu)
        self.primes.copy_(self._primes_cpu)
        self.offsets.copy_(self._offsets_cpu)
        self.multipliers.copy_(self._multipliers_cpu)
        nn.init.zeros_(self.conv_weight)

    @torch.no_grad()
    def init_weights(self, generator: torch.Generator) -> None:
        """Initialize memory independently from the backbone's random stream."""
        self.reset_parameters()
        nn.init.normal_(self.embedding.weight, std=self.config.init_std, generator=generator)
        for projection in (self.key_proj, self.value_proj):
            nn.init.normal_(projection.weight, std=self.config.init_std, generator=generator)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        for norm in (self.query_norm, self.key_norm, self.conv_norm):
            nn.init.ones_(norm.weight)

    def hash_ids(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return local addresses; supplied segments must increase at each boundary."""
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("Engram input IDs must be an int64 [batch, sequence] tensor")
        if segment_ids is None:
            segment_ids = _document_segments(input_ids, self.config.eos_id)
        if segment_ids.shape != input_ids.shape or segment_ids.dtype != torch.long:
            raise ValueError("Engram document segments must be int64 and match the input IDs")
        compressed = F.embedding(input_ids, self.token_map.unsqueeze(-1)).squeeze(-1)
        mixed = compressed * self.multipliers[0]
        hashes: list[torch.Tensor] = []
        for distance in range(1, self.config.max_ngram_size):
            context = _shift_within_document(
                compressed, distance, segment_ids, self.compressed_pad_id
            )
            mixed = torch.bitwise_xor(mixed, context * self.multipliers[distance])
            head_start = (distance - 1) * self.config.heads
            primes = self.primes[head_start : head_start + self.config.heads]
            hashes.append(mixed.unsqueeze(-1).remainder(primes))
        return torch.cat(hashes, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape != (*input_ids.shape, self.d_model):
            raise ValueError("Engram hidden states must have shape [batch, sequence, hidden]")
        if segment_ids is None:
            segment_ids = _document_segments(input_ids, self.config.eos_id)
        addresses = self.hash_ids(input_ids, segment_ids) + self.offsets
        memory = self.embedding(addresses).flatten(start_dim=-2)
        key = self.key_proj(memory)
        query = self.query_norm(hidden_states.float())
        key = self.key_norm(key.float())
        score = (query * key).sum(dim=-1) / math.sqrt(self.d_model)
        if self.config.gate_variant == "signed_sqrt":
            score = score.sign() * score.abs().clamp_min(1e-6).sqrt()
        gate = score.sigmoid().unsqueeze(-1)
        value = gate * self.value_proj(memory)
        normalized = self.conv_norm(value.float())
        convolved = torch.zeros_like(value)
        for tap in range(self.config.kernel_size):
            distance = (self.config.kernel_size - 1 - tap) * self.config.dilation
            context = _shift_within_document(normalized, distance, segment_ids, 0)
            convolved = convolved + context * self.conv_weight[:, 0, tap]
        return (value + F.silu(convolved)).to(hidden_states.dtype)
