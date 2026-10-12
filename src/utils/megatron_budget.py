"""HF config -> unmodified competition-pinned Megatron FLOPs function.

This module maps dimensions and divides the budget; it implements no model FLOPs
formula. Only the three reviewed dense/GDN decoder families are supported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

MEGATRON_COMMIT = "b92c3ed70029005fd5769d2cf4c65bc3d061a74b"
FUNCTION_SHA256 = "a32d21cfdfc8617790a96c78c7c13a3cecdcb295f2a4cc057fd1120bf0f8318c"
REFERENCE = Path(__file__).resolve().parents[2] / "third_party" / "megatron" / "reference.py"


def upstream_function():
    source = REFERENCE.read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "num_floating_point_operations"
    )
    body = ast.get_source_segment(source, node) + "\n"
    if hashlib.sha256(body.encode()).hexdigest() != FUNCTION_SHA256:
        raise ValueError("Pinned Megatron function differs from upstream snapshot")
    # These classification helpers are verbatim upstream predicates. The three
    # decoders use the Transformer branch, including its mixed GDN/SDPA path.
    helpers = """
def is_hybrid_model(args):
    return args.hybrid_layer_pattern is not None
GDN_ATTENTION_VARIANTS = ("gdn", "gdn2")
_DEPRECATED_ATTENTION_VARIANT_ALIASES = {"gated_delta_net": "gdn"}
def is_gated_delta_net_variant(experimental_attention_variant):
    canonical = _DEPRECATED_ATTENTION_VARIANT_ALIASES.get(
        experimental_attention_variant, experimental_attention_variant
    )
    return canonical in GDN_ATTENTION_VARIANTS
def is_linear_attention_variant(experimental_attention_variant):
    return is_gated_delta_net_variant(experimental_attention_variant)
"""
    namespace = {}
    exec(compile(helpers + body, str(REFERENCE), "exec"), namespace)
    return namespace["num_floating_point_operations"]


def config_to_args(config, sequence_length):
    c = config.get("text_config", config)
    if c.get("model_type") not in ("qwen2", "qwen3", "qwen3_5_text"):
        raise ValueError("Only reviewed Qwen2/Qwen3/Qwen3.5 text configs supported")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    types = c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])
    if len(types) != c["num_hidden_layers"] or set(types) - {"full_attention", "linear_attention"}:
        raise ValueError("Unsupported layer pattern")
    defaults = dict(
        hybrid_layer_pattern=None,
        multi_latent_attention=False,
        num_experts=None,
        moe_layer_freq=1,
        moe_router_topk=0,
        moe_ffn_hidden_size=None,
        moe_latent_size=None,
        moe_shared_expert_intermediate_size=None,
        mtp_num_layers=None,
        experimental_attention_variant="gdn" if "linear_attention" in types else None,
        linear_attention_freq=[int(x == "linear_attention") for x in types],
        linear_key_head_dim=c.get("linear_key_head_dim"),
        linear_value_head_dim=c.get("linear_value_head_dim"),
        linear_num_key_heads=c.get("linear_num_key_heads"),
        linear_num_value_heads=c.get("linear_num_value_heads"),
        linear_conv_kernel_dim=c.get("linear_conv_kernel_dim"),
        hidden_size=c["hidden_size"],
        ffn_hidden_size=c["intermediate_size"],
        num_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"],
        num_query_groups=c["num_key_value_heads"],
        group_query_attention=True,
        kv_channels=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        attention_output_gate=c.get("attn_output_gate", False),
        padded_vocab_size=c["vocab_size"],
        seq_length=sequence_length,
        swiglu=c["hidden_act"] == "silu",
    )
    if not defaults["swiglu"]:
        raise ValueError("Only reviewed SwiGLU configs supported")
    return SimpleNamespace(**defaults)


def budget_report(
    config, *, sequence_length=8192, batch_sequences=1, budget=3 * 10**17, rounding="floor"
):
    if type(batch_sequences) is not int or batch_sequences <= 0 or budget <= 0:
        raise ValueError("positive integer batch and positive budget required")
    if rounding not in ("floor", "nearest"):
        raise ValueError("rounding must be 'floor' or 'nearest'")
    if rounding == "nearest" and type(budget) is not int:
        raise ValueError("nearest rounding requires an integer FLOPs budget")
    args = config_to_args(config, sequence_length)
    count = upstream_function()(args, batch_sequences)
    if count != int(count):
        raise ValueError("Megatron batch FLOPs should be integral for this configuration")
    flops_step = int(count)
    tokens_step = batch_sequences * sequence_length
    steps, remainder = divmod(budget, flops_step)
    if rounding == "nearest" and 2 * remainder >= flops_step:
        # Integer comparison keeps budgets above float64's exact integer range
        # precise. Ties choose the larger complete optimizer step.
        steps += 1
    actual = steps * flops_step
    return {
        "megatron_commit": MEGATRON_COMMIT,
        "function_sha256": FUNCTION_SHA256,
        "megatron_args": vars(args),
        "budget": budget,
        "rounding": rounding,
        "rounding_tie_break": "up" if rounding == "nearest" else None,
        "layout": "BSHD (THD degeneracy: no document mask)",
        "sequence_length": sequence_length,
        "global_batch_sequences": batch_sequences,
        "tokens_per_optimizer_step": tokens_step,
        "training_flops_per_optimizer_step": flops_step,
        "training_flops_per_token": flops_step / tokens_step,
        "continuous_token_capacity": budget * tokens_step / flops_step,
        "whole_optimizer_steps": steps,
        "whole_step_token_capacity": steps * tokens_step,
        "actual_whole_step_flops": actual,
        "relative_budget_gap": (budget - actual) / budget,
        "within_one_percent": abs(actual - budget) <= budget // 100,
        "scope": "unmodified Megatron reference; random-init text decoder, no vision/MTP; not official certification",
        "gdn_kernel_scope": "Megatron recurrent logical estimate, FLA chunk implementation overhead not certified"
        if args.experimental_attention_variant
        else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--sequence-length", type=int, default=8192)
    p.add_argument("--batch-sequences", type=int, default=1)
    p.add_argument("--budget", type=int, default=3 * 10**17)
    p.add_argument(
        "--rounding",
        choices=("floor", "nearest"),
        default="floor",
        help="Whole optimizer steps; nearest uses exact integers and rounds ties up",
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    config_bytes = a.config.read_bytes()
    result = budget_report(
        json.loads(config_bytes),
        sequence_length=a.sequence_length,
        batch_sequences=a.batch_sequences,
        budget=a.budget,
        rounding=a.rounding,
    )
    result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    body = json.dumps(result, indent=2) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(body)
    print(body)


if __name__ == "__main__":
    main()
