"""Inject single-stream Engram before selected OLMo Transformer blocks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, fields
from functools import partial
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
import torch.nn.functional as F
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerConfig, TransformerType
from torch import nn
from torch.distributed import DeviceMesh
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.arch.memory.engram import Engram, EngramConfig, load_token_map


class EngramIdentity(TypedDict):
    """Bind memory configuration to the exact tokenizer projection asset."""

    config: dict[str, object]
    token_map_path: str
    token_map_sha256: str
    tokenizer_backend_sha256: NotRequired[str]


def _config(payload: Mapping[str, object]) -> EngramConfig:
    settings = dict(payload)
    unknown = settings.keys() - {entry.name for entry in fields(EngramConfig)}
    if unknown:
        raise ValueError(f"Unknown Engram configuration fields: {sorted(unknown)}")
    for name in ("layer_ids", "table_sizes"):
        if name in settings:
            value = settings[name]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Engram {name} must be an integer sequence")
            settings[name] = tuple(value)
    config = EngramConfig(**settings)
    config.validate()
    return config


def config_from_identity(identity: EngramIdentity) -> EngramConfig:
    return _config(identity["config"])


def memory_identity(config_path: Path, token_map_path: Path) -> EngramIdentity:
    """Read only a declared configuration and an existing vocabulary map."""
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Engram configuration must be a JSON object")
    config = _config(payload)
    load_token_map(token_map_path)
    identity = EngramIdentity(
        config=json.loads(json.dumps(asdict(config))),
        token_map_path=str(token_map_path.resolve()),
        token_map_sha256=hashlib.sha256(token_map_path.read_bytes()).hexdigest(),
    )
    metadata_path = token_map_path.with_suffix(".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("token_map_sha256") != identity["token_map_sha256"]:
            raise ValueError("Engram token map does not match its preparation metadata")
        backend_hash = metadata.get("tokenizer_backend_sha256")
        if not isinstance(backend_hash, str) or len(backend_hash) != 64:
            raise ValueError("Engram preparation metadata must identify the tokenizer backend")
        identity["tokenizer_backend_sha256"] = backend_hash
    return identity


def validate_memory_tokenizer(identity: EngramIdentity, tokenizer_dir: Path) -> None:
    """Bind the memory-only projection to the training tokenizer before GPU admission."""
    if "tokenizer_backend_sha256" not in identity:
        raise ValueError("Prepare the Engram token map with its tokenizer identity metadata")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Engram training requires the existing fast tokenizer")
    backend_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if backend_hash != identity["tokenizer_backend_sha256"]:
        raise ValueError("Engram token map was prepared for a different training tokenizer")
    token_map = load_token_map(Path(identity["token_map_path"]))
    if token_map.numel() != len(tokenizer):
        raise ValueError("Engram token map length differs from the tokenizer vocabulary")
    config = config_from_identity(identity)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if config.eos_id != tokenizer.eos_token_id or config.pad_id != pad_id:
        raise ValueError("Engram EOS/pad IDs differ from the streaming tokenizer")


def _inject_memory(
    memory: Engram,
    block: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, object],
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    del block
    input_ids = kwargs.pop("engram_input_ids")
    segment_ids = kwargs.pop("engram_segment_ids")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(segment_ids, torch.Tensor):
        raise TypeError("Engram block inputs must include token and segment tensors")
    hidden_states = inputs[0]
    delta = memory(hidden_states, input_ids, segment_ids)
    return (hidden_states + delta, *inputs[1:]), kwargs


class EngramTransformer(Transformer):
    """Reuse OLMo forward, initialization, DDP, loss, and checkpoint machinery."""

    def __init__(
        self,
        model_config: TransformerConfig,
        memory_config: EngramConfig,
        token_map: torch.Tensor,
        *,
        init_device: str = "cpu",
    ) -> None:
        if model_config.name != TransformerType.default:
            raise ValueError("Engram currently supports the standard single-stream backbone")
        memory_config.validate()
        if max(memory_config.layer_ids) >= model_config.n_layers:
            raise ValueError("Engram layer IDs exceed the backbone layer count")
        super().__init__(
            d_model=model_config.d_model,
            vocab_size=model_config.vocab_size,
            n_layers=model_config.n_layers,
            block=model_config.block,
            lm_head=model_config.lm_head,
            embedding_norm=model_config.embedding_norm,
            dtype=model_config.dtype.as_pt(),
            init_method=model_config.init_method,
            init_device=init_device,
            init_seed=model_config.init_seed,
            init_std=model_config.init_std,
            embedding_init_std=model_config.embedding_init_std,
            block_overrides=model_config.block_overrides,
            block_pattern=model_config.block_pattern,
            embed_scale=model_config.embed_scale,
            tie_word_embeddings=model_config.tie_word_embeddings,
        )
        self.memory_config = memory_config
        self.engrams = nn.ModuleDict()
        for layer_id in memory_config.layer_ids:
            memory = Engram(
                self.d_model,
                layer_id,
                memory_config,
                token_map,
                device=init_device,
                dtype=model_config.dtype.as_pt(),
            )
            self.engrams[str(layer_id)] = memory
            self.blocks[str(layer_id)].register_forward_pre_hook(
                partial(_inject_memory, memory), with_kwargs=True
            )

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: int | None = None,
        max_local_microbatch_size: int | None = None,
        device: torch.device | None = None,
        world_mesh: DeviceMesh | None = None,
        model_part_idx: int = 0,
    ) -> torch.Generator:
        generator = super().init_weights(
            max_seq_len=max_seq_len,
            max_local_microbatch_size=max_local_microbatch_size,
            device=device,
            world_mesh=world_mesh,
            model_part_idx=model_part_idx,
        )
        for memory in self.engrams.values():
            seed = self.memory_config.seed + 10007 * memory.layer_id
            memory.init_weights(torch.Generator(self.device).manual_seed(seed))
        return generator

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        dict[str, object],
        dict[int, dict[str, object]],
        dict[str, object],
    ]:
        if self.pp_enabled or self.tp_enabled or self._cp_load_balancer is not None:
            raise NotImplementedError("Engram supports unsharded sequences and data parallelism")
        prepared = super()._prepare_inputs(input_ids, labels, **kwargs)
        token_ids, labels, all_kwargs, layer_kwargs, head_kwargs = prepared
        document_starts = F.pad(token_ids[:, :-1].eq(self.memory_config.eos_id), (1, 0))
        segment_ids = document_starts.long().cumsum(dim=1)
        for layer_id in self.memory_config.layer_ids:
            layer_kwargs[layer_id]["engram_input_ids"] = token_ids
            layer_kwargs[layer_id]["engram_segment_ids"] = segment_ids
        return token_ids, labels, all_kwargs, layer_kwargs, head_kwargs


def build_engram_model(
    model_config: TransformerConfig,
    identity: EngramIdentity,
    *,
    init_device: str = "meta",
) -> EngramTransformer:
    """Verify a saved lookup identity before allocating trainable memory."""
    map_path = Path(identity["token_map_path"])
    if hashlib.sha256(map_path.read_bytes()).hexdigest() != identity["token_map_sha256"]:
        raise ValueError("Engram tokenizer projection changed since protocol preparation")
    return EngramTransformer(
        model_config,
        config_from_identity(identity),
        load_token_map(map_path),
        init_device=init_device,
    )
