"""Qwen3.5-only OLMo architecture adapters."""

from .attn import full as full  # register supported attention blocks
from .attn import qwen35_gdn as qwen35_gdn
from .core.spec import (
    ArchSpec,
    LayerSpec,
    build_model,
    pattern_to_layers,
    spec_num_params,
    spec_to_olmo,
)
from .ffn import swiglu as swiglu
from .norm import rms as rms

__all__ = [
    "ArchSpec",
    "LayerSpec",
    "build_model",
    "pattern_to_layers",
    "spec_num_params",
    "spec_to_olmo",
]
