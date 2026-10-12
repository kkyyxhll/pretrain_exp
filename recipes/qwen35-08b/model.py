"""Build the random-init Qwen3.5 text backbone from its public dimensions."""

from __future__ import annotations

from src.arch.core.spec import ArchSpec, pattern_to_layers

NUM_LAYERS = 24


def build_spec() -> ArchSpec:
    """Use full attention every fourth layer and GatedDeltaNet elsewhere."""
    pattern = "attn=qwen35_gdn * 24; attn=full @ 3,7,11,15,19,23"
    layers = pattern_to_layers(pattern, NUM_LAYERS)
    for layer in layers:
        layer.norm_kwargs = {"unit_offset": True, "eps": 1e-6}
        layer.ffn_kwargs = {"hidden_mult": 3.5}
        if layer.attn == "full":
            layer.attn_kwargs.update(
                n_heads=8,
                n_kv_heads=2,
                head_dim=256,
                bias=False,
                qk_norm=True,
                use_head_qk_norm=True,
                gate=True,
                qk_norm_kind="qwen35",
                qk_norm_eps=1e-6,
                gate_full_precision=False,
                partial_rotary_factor=0.25,
            )
        else:
            layer.attn_kwargs.update(
                num_k_heads=16,
                head_k_dim=128,
                num_v_heads=16,
                head_v_dim=128,
                conv_kernel_size=4,
                norm_eps=1e-6,
            )
    return ArchSpec(
        vocab_size=248320,
        d_model=1024,
        n_layers=NUM_LAYERS,
        rope_theta=10_000_000.0,
        dtype="bfloat16",
        tie_word_embeddings=True,
        init_std=0.02,
        final_norm="rms",
        final_norm_kwargs={"unit_offset": True, "eps": 1e-6},
        layers=layers,
    )
