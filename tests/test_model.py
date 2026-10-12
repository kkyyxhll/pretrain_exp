"""Production recipe dimensions, weight tying and no accidental native FLOPs."""

import importlib
import json
from pathlib import Path

import pytest

from arch import build_model

build_spec = importlib.import_module("recipes.qwen35-08b.model").build_spec

ROOT = Path(__file__).resolve().parents[1]


def test_qwen35_config_and_meta_parameters():
    c = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())["text_config"]
    spec = build_spec()
    assert (spec.n_layers, spec.d_model, spec.vocab_size) == (
        c["num_hidden_layers"],
        c["hidden_size"],
        c["vocab_size"],
    )
    assert [layer.attn for layer in spec.layers] == [
        "full" if t == "full_attention" else "qwen35_gdn" for t in c["layer_types"]
    ]
    model = build_model(spec, init_device="meta")
    assert sum(p.numel() for p in model.parameters()) == 752393024
    assert model.embeddings.weight is model.lm_head.w_out.weight
    with pytest.raises(NotImplementedError, match="Megatron"):
        model.num_flops_per_token(8192)
