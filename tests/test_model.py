"""Production recipe dimensions, weight tying and no accidental native FLOPs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.arch.core.spec import build_model

RECIPE = importlib.import_module("recipes.qwen35-08b.model")

ROOT = Path(__file__).resolve().parents[1]


def test_qwen35_config_and_meta_parameters() -> None:
    text_config = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())["text_config"]
    model_spec = RECIPE.build_spec()
    assert (model_spec.n_layers, model_spec.d_model, model_spec.vocab_size) == (
        text_config["num_hidden_layers"],
        text_config["hidden_size"],
        text_config["vocab_size"],
    )
    assert [layer.attn for layer in model_spec.layers] == [
        "full" if layer_type == "full_attention" else "qwen35_gdn"
        for layer_type in text_config["layer_types"]
    ]
    model = build_model(model_spec, init_device="meta")
    assert sum(parameter.numel() for parameter in model.parameters()) == 752393024
    assert model.embeddings.weight is model.lm_head.w_out.weight
    with pytest.raises(NotImplementedError, match="Megatron"):
        model.num_flops_per_token(8192)
