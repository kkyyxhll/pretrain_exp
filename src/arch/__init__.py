"""Qwen3.5-only OLMo architecture adapters."""

from src.arch.core.spec import (
    ArchSpec,
    LayerSpec,
    build_model,
    pattern_to_layers,
    spec_num_params,
    spec_to_olmo,
)

__all__: list[str] = [
    "ArchSpec",
    "LayerSpec",
    "build_model",
    "pattern_to_layers",
    "spec_num_params",
    "spec_to_olmo",
]
