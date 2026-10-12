"""Map reviewed text configurations to the pinned reference training budget."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

MEGATRON_COMMIT = "b92c3ed70029005fd5769d2cf4c65bc3d061a74b"
FUNCTION_SHA256 = "a32d21cfdfc8617790a96c78c7c13a3cecdcb295f2a4cc057fd1120bf0f8318c"
REFERENCE = Path(__file__).resolve().parents[2] / "third_party" / "megatron" / "reference.py"
DEFAULT_SEQUENCE_LENGTH_TOKENS = 8192
DEFAULT_BATCH_SEQUENCES = 1
DEFAULT_BUDGET_FLOPS = 3 * 10**17

BudgetScalar = int | float | bool | str | None
BudgetPayload = dict[str, BudgetScalar | dict[str, int | bool | str | list[int] | None]]


@dataclass(kw_only=True)
class MegatronArguments:
    """Provide the fixed function's argument names for reviewed text decoders."""

    hybrid_layer_pattern: None = None
    multi_latent_attention: Literal[False] = False
    num_experts: None = None
    moe_layer_freq: Literal[1] = 1
    moe_router_topk: Literal[0] = 0
    moe_ffn_hidden_size: None = None
    moe_latent_size: None = None
    moe_shared_expert_intermediate_size: None = None
    mtp_num_layers: None = None
    experimental_attention_variant: Literal["gdn"] | None
    linear_attention_freq: list[int]
    linear_key_head_dim: int | None
    linear_value_head_dim: int | None
    linear_num_key_heads: int | None
    linear_num_value_heads: int | None
    linear_conv_kernel_dim: int | None
    hidden_size: int
    ffn_hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_query_groups: int
    group_query_attention: Literal[True] = True
    kv_channels: int
    attention_output_gate: bool
    padded_vocab_size: int
    seq_length: int
    swiglu: Literal[True] = True


class MegatronFunction(Protocol):
    """Keep the dynamically compiled reference's exact calling signature."""

    def __call__(
        self,
        args: MegatronArguments,
        batch_size: int,
        seqlen_squared_sum_in_batch: int | None = None,
        total_real_tokens_in_batch: int | None = None,
    ) -> int | float: ...


@dataclass(kw_only=True)
class BudgetReport:
    """Describe full-update capacity and its fixed reference identity."""

    megatron_commit: str = MEGATRON_COMMIT
    function_sha256: str = FUNCTION_SHA256
    megatron_args: MegatronArguments
    budget: int
    rounding: Literal["floor", "nearest"]
    rounding_tie_break: Literal["up"] | None = field(init=False)
    layout: str = "BSHD (THD degeneracy: no document mask)"
    sequence_length: int
    global_batch_sequences: int
    tokens_per_optimizer_step: int = field(init=False)
    training_flops_per_optimizer_step: int
    training_flops_per_token: float = field(init=False)
    continuous_token_capacity: float = field(init=False)
    whole_optimizer_steps: int
    whole_step_token_capacity: int = field(init=False)
    actual_whole_step_flops: int = field(init=False)
    relative_budget_gap: float = field(init=False)
    within_one_percent: bool = field(init=False)
    scope: str = (
        "unmodified Megatron reference; random-init text decoder, no vision/MTP; "
        "not official certification"
    )
    gdn_kernel_scope: str | None = field(init=False)

    def __post_init__(self) -> None:
        self.rounding_tie_break = "up" if self.rounding == "nearest" else None
        self.tokens_per_optimizer_step = self.global_batch_sequences * self.sequence_length
        self.training_flops_per_token = (
            self.training_flops_per_optimizer_step / self.tokens_per_optimizer_step
        )
        self.continuous_token_capacity = (
            self.budget * self.tokens_per_optimizer_step / self.training_flops_per_optimizer_step
        )
        self.whole_step_token_capacity = self.whole_optimizer_steps * self.tokens_per_optimizer_step
        self.actual_whole_step_flops = (
            self.whole_optimizer_steps * self.training_flops_per_optimizer_step
        )
        self.relative_budget_gap = (self.budget - self.actual_whole_step_flops) / self.budget
        self.within_one_percent = (
            abs(self.actual_whole_step_flops - self.budget) <= self.budget // 100
        )
        self.gdn_kernel_scope = (
            "Megatron recurrent logical estimate, FLA chunk implementation overhead not certified"
            if self.megatron_args.experimental_attention_variant
            else None
        )


def upstream_function() -> MegatronFunction:
    """Compile the reference after verifying its unchanged source hash."""
    reference_source = REFERENCE.read_text()
    function_node = next(
        source_node
        for source_node in ast.parse(reference_source).body
        if isinstance(source_node, ast.FunctionDef)
        and source_node.name == "num_floating_point_operations"
    )
    function_source = ast.get_source_segment(reference_source, function_node)
    assert function_source is not None
    function_source += "\n"
    if hashlib.sha256(function_source.encode()).hexdigest() != FUNCTION_SHA256:
        raise ValueError("Pinned Megatron function differs from upstream snapshot")
    # note (kkyyxhll): Fixed predicates select the same formula branch as the pinned reference.
    classification_source = """
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
    reference_namespace: dict[str, MegatronFunction] = {}
    exec(
        compile(classification_source + function_source, str(REFERENCE), "exec"),
        reference_namespace,
    )
    return reference_namespace["num_floating_point_operations"]


def config_to_args(config: Mapping[str, object], sequence_length: int) -> MegatronArguments:
    """Translate reviewed decoder dimensions without substituting a FLOPs formula."""
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise ValueError("text_config must be a mapping")
    if (
        text_config.get("model_type") not in ("qwen2", "qwen3", "qwen3_5_text")
        or sequence_length <= 0
    ):
        raise ValueError("A reviewed text decoder and positive sequence_length are required")

    layer_types = text_config.get(
        "layer_types", ["full_attention"] * text_config["num_hidden_layers"]
    )
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != text_config["num_hidden_layers"]
        or any(
            layer_type not in ("full_attention", "linear_attention") for layer_type in layer_types
        )
        or text_config.get("hidden_act") != "silu"
    ):
        raise ValueError("Unsupported decoder layer pattern or activation")
    return MegatronArguments(
        experimental_attention_variant="gdn" if "linear_attention" in layer_types else None,
        linear_attention_freq=[int(layer_type == "linear_attention") for layer_type in layer_types],
        linear_key_head_dim=text_config.get("linear_key_head_dim"),
        linear_value_head_dim=text_config.get("linear_value_head_dim"),
        linear_num_key_heads=text_config.get("linear_num_key_heads"),
        linear_num_value_heads=text_config.get("linear_num_value_heads"),
        linear_conv_kernel_dim=text_config.get("linear_conv_kernel_dim"),
        hidden_size=text_config["hidden_size"],
        ffn_hidden_size=text_config["intermediate_size"],
        num_layers=text_config["num_hidden_layers"],
        num_attention_heads=text_config["num_attention_heads"],
        num_query_groups=text_config["num_key_value_heads"],
        kv_channels=text_config.get(
            "head_dim", text_config["hidden_size"] // text_config["num_attention_heads"]
        ),
        attention_output_gate=text_config.get("attn_output_gate", False),
        padded_vocab_size=text_config["vocab_size"],
        seq_length=sequence_length,
    )


def budget_report(
    config: Mapping[str, object],
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH_TOKENS,
    batch_sequences: int = DEFAULT_BATCH_SEQUENCES,
    budget: int = DEFAULT_BUDGET_FLOPS,
    rounding: Literal["floor", "nearest"] = "floor",
) -> BudgetPayload:
    """Allocate complete optimizer updates and preserve the public dictionary report."""
    if (
        type(batch_sequences) is not int
        or batch_sequences <= 0
        or budget <= 0
        or rounding not in ("floor", "nearest")
        or (rounding == "nearest" and type(budget) is not int)
    ):
        raise ValueError(
            "Use positive batch/budget, floor or nearest rounding, and integer nearest budget"
        )
    megatron_arguments = config_to_args(config, sequence_length)
    reference_flops = upstream_function()(megatron_arguments, batch_sequences)
    if reference_flops != int(reference_flops):
        raise ValueError("Megatron batch FLOPs should be integral for this configuration")
    update_flops = int(reference_flops)
    optimizer_step_count, remaining_flops = divmod(budget, update_flops)
    # note (kkyyxhll): Exact integers keep large budgets precise and round nearest ties upward.
    optimizer_step_count += int(rounding == "nearest" and 2 * remaining_flops >= update_flops)
    report = BudgetReport(
        megatron_args=megatron_arguments,
        budget=budget,
        rounding=rounding,
        sequence_length=sequence_length,
        global_batch_sequences=batch_sequences,
        training_flops_per_optimizer_step=update_flops,
        whole_optimizer_steps=optimizer_step_count,
    )
    return asdict(report)


def main() -> None:
    """Print or save a reference budget report for a local model configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH_TOKENS)
    parser.add_argument("--batch-sequences", type=int, default=DEFAULT_BATCH_SEQUENCES)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_FLOPS)
    parser.add_argument("--rounding", choices=("floor", "nearest"), default="floor")
    parser.add_argument("--output", type=Path)
    command_options = parser.parse_args()
    config_bytes = command_options.config.read_bytes()
    decoded_config: object = json.loads(config_bytes)
    if not isinstance(decoded_config, dict):
        raise ValueError("Model configuration must be a mapping")
    result = budget_report(
        decoded_config,
        sequence_length=command_options.sequence_length,
        batch_sequences=command_options.batch_sequences,
        budget=command_options.budget,
        rounding=command_options.rounding,
    )
    result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    report_json = json.dumps(result, indent=2) + "\n"
    if command_options.output:
        command_options.output.parent.mkdir(parents=True, exist_ok=True)
        command_options.output.write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    main()
