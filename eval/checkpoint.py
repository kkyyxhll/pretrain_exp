"""Load fixed training weights and enforce precision and tokenizer identity contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import torch
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
from olmo_core.nn.transformer import Transformer
from torch.distributed.checkpoint import load
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from src.arch.core.spec import spec_to_olmo
from src.arch.memory.olmo import build_engram_model
from src.data.olmo_streaming import TokenizerIdentity
from src.train.config import TrainSpec
from src.train.loss import ChunkedLMHeadConfig

CHECKPOINT_READER_THREADS = 2
PRECISION_MODES = ("bf16", "fp32-master-bf16")
PrecisionMode = Literal["bf16", "fp32-master-bf16"]


@dataclass(kw_only=True)
class EvaluateOptions:
    """Hold the complete fixed-weight evaluation configuration."""

    spec: Path
    checkpoint: Path | None
    cache_dir: Path
    output_dir: Path
    sequence_length: int
    batch_size: int
    loss_chunk_tokens: int
    max_batches: int | None
    precision: PrecisionMode


class CacheMetadata(TypedDict):
    """Describe official packing counts and optional cache provenance fields."""

    total_bytes: int
    total_tokens: int
    pad_token_id: int
    dataset_path: NotRequired[str]
    dataset_file_bytes: NotRequired[int]
    tokenizer_path: NotRequired[str]
    tokenizer_identity: NotRequired[TokenizerIdentity]
    vocab_size: NotRequired[int]
    packing: NotRequired[str]
    preparation_seconds: NotRequired[float]


def evaluation_autocast(
    precision: PrecisionMode, device: torch.device
) -> nullcontext[None] | torch.autocast:
    """Use BF16 compute without rounding the stored FP32 training parameters."""
    if precision == "bf16":
        return nullcontext()
    elif precision == "fp32-master-bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        raise ValueError(f"unsupported evaluation precision: {precision}")


def prepare_model_precision(
    model: torch.nn.Module, device: torch.device, precision: PrecisionMode
) -> torch.nn.Module:
    """Freeze evaluation parameters while preserving the selected storage precision."""
    if precision not in PRECISION_MODES:
        raise ValueError(f"unsupported evaluation precision: {precision}")
    else:
        parameter_dtype = torch.float32 if precision == "fp32-master-bf16" else torch.bfloat16
        return model.to(device=device, dtype=parameter_dtype).eval().requires_grad_(False)


def validate_master_checkpoint(metadata: Metadata) -> None:
    """Reject rounded floating tensors when original FP32 training values are required."""
    for parameter_name, stored_tensor in metadata.state_dict_metadata.items():
        if not parameter_name.startswith("model."):
            continue
        elif not isinstance(stored_tensor, TensorStorageMetadata):
            raise ValueError(f"checkpoint model entry is not a tensor: {parameter_name}")
        else:
            parameter_dtype = stored_tensor.properties.dtype
            if parameter_dtype.is_floating_point and parameter_dtype != torch.float32:
                raise ValueError(
                    f"FP32 master evaluation requires FP32 checkpoint tensors: {parameter_name} is {parameter_dtype}"
                )
            else:
                pass


def validate_evaluation_cache(
    spec_data: Mapping[str, object], metadata: CacheMetadata, precision: PrecisionMode
) -> None:
    """Require the training and evaluation tokenizer identities to agree."""
    training_data = spec_data.get("data", {})
    if not isinstance(training_data, dict):
        raise ValueError("Training data specification must be a mapping")
    else:
        expected_identity = training_data.get("tokenizer_identity")
        stream_identity_payload = training_data.get("identity", {})
        if not isinstance(stream_identity_payload, dict):
            raise ValueError("Training stream identity must be a mapping")
        else:
            input_identity = stream_identity_payload.get("inputs", {})
            if not isinstance(input_identity, dict):
                raise ValueError("Training input identity must be a mapping")
            else:
                stream_tokenizer_identity = input_identity.get("tokenizer")
        if (
            expected_identity is not None
            and stream_tokenizer_identity is not None
            and expected_identity != stream_tokenizer_identity
        ):
            raise ValueError("Training tokenizer identities disagree")
        else:
            expected_identity = (
                stream_tokenizer_identity if expected_identity is None else expected_identity
            )
        if expected_identity is None:
            if precision == "fp32-master-bf16":
                raise ValueError(
                    "FP32 master evaluation requires a tokenizer_identity in train.json"
                )
            else:
                return
        elif not isinstance(expected_identity, dict):
            raise ValueError("Training tokenizer identity must be a mapping")
        elif metadata.get("tokenizer_identity") != expected_identity:
            raise ValueError(
                "Evaluation cache tokenizer differs from training; prepare a matching cache"
            )
        else:
            pass


def validate_checkpoint_layout(state: Mapping[str, torch.Tensor], metadata: Metadata) -> None:
    """Reject missing model entries and incompatible tensor shapes before loading."""
    stored_parameters = metadata.state_dict_metadata
    expected_names = {"model." + parameter_name for parameter_name in state}
    present_names = {
        parameter_name
        for parameter_name in stored_parameters
        if parameter_name.startswith("model.")
    }
    if expected_names != present_names:
        raise ValueError(
            f"checkpoint model keys differ: missing={expected_names - present_names}, unexpected={present_names - expected_names}"
        )
    else:
        for parameter_name, parameter_tensor in state.items():
            stored_tensor = stored_parameters["model." + parameter_name]
            if not isinstance(stored_tensor, TensorStorageMetadata):
                raise ValueError(f"checkpoint model entry is not a tensor: {parameter_name}")
            elif tuple(stored_tensor.size) != tuple(parameter_tensor.shape):
                raise ValueError(f"checkpoint shape mismatch for {parameter_name}")
            else:
                pass


def saved_training_spec(checkpoint: Path) -> Path:
    """Find the architecture record beside the checkpoint or its parent training folder."""
    for checkpoint_parent in (checkpoint.parent, checkpoint.parent.parent):
        training_spec = checkpoint_parent / "train.json"
        if training_spec.is_file():
            return training_spec
        else:
            pass
    raise ValueError("checkpoint run must retain its train.json architecture")


def load_model(options: EvaluateOptions, device: torch.device) -> tuple[Transformer, list[str]]:
    """Load only the trained model tensors while retaining FP32 master values."""
    if options.checkpoint is None:
        raise ValueError("Checkpoint evaluation requires a checkpoint path")
    spec_data: object = json.loads(options.spec.read_text())
    saved_spec_data: object = json.loads(saved_training_spec(options.checkpoint).read_text())
    if not isinstance(spec_data, dict) or not isinstance(saved_spec_data, dict):
        raise ValueError("Training specifications must be mappings")
    elif saved_spec_data["arch"] != spec_data["arch"]:
        raise ValueError("evaluation architecture differs from the saved training spec")
    elif saved_spec_data.get("engram") != spec_data.get("engram"):
        raise ValueError("evaluation Engram identity differs from the saved training spec")
    else:
        training_spec = TrainSpec.from_dict(spec_data)
    if options.precision == "fp32-master-bf16" and training_spec.autocast != "bfloat16":
        raise ValueError("FP32 master/BF16 evaluation requires BF16 training autocast")
    else:
        model_config = spec_to_olmo(training_spec.to_arch_spec())
    model_config.lm_head = ChunkedLMHeadConfig(
        layer_norm=model_config.lm_head.layer_norm,
        bias=model_config.lm_head.bias,
        dtype=model_config.lm_head.dtype,
        chunk_tokens=options.loss_chunk_tokens,
    )
    model = (
        build_engram_model(model_config, training_spec.engram, init_device="cpu")
        if training_spec.engram is not None
        else model_config.build(init_device="cpu")
    )
    if options.precision == "fp32-master-bf16":
        # note (kkyyxhll): Checkpoint loading uses destination dtype, so promotion must precede it.
        model.float()
    else:
        pass
    model_state = model.state_dict()
    checkpoint_reader = RemoteFileSystemReader(
        options.checkpoint / "model_and_optim", thread_count=CHECKPOINT_READER_THREADS
    )
    checkpoint_metadata = checkpoint_reader.read_metadata()
    validate_checkpoint_layout(model_state, checkpoint_metadata)
    if options.precision == "fp32-master-bf16":
        validate_master_checkpoint(checkpoint_metadata)
    else:
        pass
    load({"model": model_state}, storage_reader=checkpoint_reader)
    model.load_state_dict(model_state, strict=True)
    del model_state
    storage_dtypes: set[str] = set()
    for parameter_name, stored_tensor in checkpoint_metadata.state_dict_metadata.items():
        if not parameter_name.startswith("model."):
            continue
        else:
            assert isinstance(stored_tensor, TensorStorageMetadata)
            storage_dtypes.add(str(stored_tensor.properties.dtype))
    prepare_model_precision(model, device, options.precision)
    if model.tie_word_embeddings and model.embeddings.weight is not model.lm_head.w_out.weight:
        raise RuntimeError("evaluation model lost tied vocabulary parameters")
    else:
        return model, sorted(storage_dtypes)
