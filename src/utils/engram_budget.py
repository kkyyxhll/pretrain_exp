"""Keep Engram's experimental major terms separate from pinned Megatron FLOPs."""

from __future__ import annotations

from typing import Literal

from src.arch.memory.engram import EngramConfig
from src.utils.megatron_budget import BudgetPayload, budget_report


def budget_report_with_engram(
    backbone_config: dict[str, object],
    *,
    engram_config: EngramConfig,
    sequence_length: int,
    batch_sequences: int,
    budget: int,
    rounding: Literal["nearest", "floor"] = "nearest",
) -> BudgetPayload:
    """Estimate trainable projections/conv with Megatron's multiply-add and ×3 rule.

    The pinned competition function has no Engram term. This appendix omits norm,
    gate, bias, activation, and lookup-gradient additions; it is not an officially
    certified actual-graph ledger and cannot admit a competition candidate.
    """
    engram_config.validate()
    baseline = budget_report(
        backbone_config,
        sequence_length=sequence_length,
        batch_sequences=batch_sequences,
        budget=budget,
        rounding=rounding,
    )
    text_config = backbone_config.get("text_config", backbone_config)
    if not isinstance(text_config, dict):
        raise ValueError("The backbone text configuration must be an object")
    hidden_size = int(text_config["hidden_size"])
    num_layers = int(text_config["num_hidden_layers"])
    if max(engram_config.layer_ids) >= num_layers:
        raise ValueError("Engram layer IDs exceed the backbone layer count")
    projections = 2 * engram_config.memory_dim * hidden_size
    convolution = hidden_size * engram_config.kernel_size
    extra_per_token = 6 * len(engram_config.layer_ids) * (projections + convolution)
    baseline_per_token = int(baseline["training_flops_per_token"])
    total_per_token = baseline_per_token + extra_per_token
    update_flops = total_per_token * batch_sequences * sequence_length
    updates, remaining_flops = divmod(budget, update_flops)
    updates += int(rounding == "nearest" and 2 * remaining_flops >= update_flops)
    if updates <= 0:
        raise ValueError("Engram budget must cover at least one complete optimizer update")
    report = dict(baseline)
    report["backbone_megatron_training_flops_per_token"] = baseline_per_token
    report["engram_experimental_training_flops_per_token"] = extra_per_token
    report["official_engram_certification"] = "not_available"
    report["engram_accounting_scope"] = (
        "Single-stream trainable Key/Value and valid-output depthwise Conv major terms; "
        "excludes hash integer work, norm/gate/bias/SiLU, lookup gradient additions, "
        "optimizer, communication, and activation recomputation."
    )
    report["training_flops_per_token"] = total_per_token
    report["training_flops_per_optimizer_step"] = update_flops
    report["whole_optimizer_steps"] = updates
    actual_flops = updates * update_flops
    report["actual_whole_step_flops"] = actual_flops
    report["continuous_token_capacity"] = budget / total_per_token
    report["whole_step_token_capacity"] = updates * batch_sequences * sequence_length
    report["relative_budget_gap"] = (budget - actual_flops) / budget
    report["within_one_percent"] = abs(actual_flops - budget) <= budget // 100
    report["scope"] = (
        "Pinned Megatron backbone plus experimental Engram major terms; not certification"
    )
    return report
