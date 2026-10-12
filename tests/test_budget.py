"""Budget reference integrity and global accounting across DP2 microbatches."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule
from olmo_core.train.trainer import Trainer

import src.utils.megatron_budget as megatron_budget
from src.data.olmo_streaming import StreamingOLMoLoader
from src.utils.megatron_budget import budget_report, upstream_function

ROOT = Path(__file__).resolve().parents[1]
TRAINING = importlib.import_module("recipes.qwen35-08b.train")
CALLBACKS = importlib.import_module("recipes.qwen35-08b.callbacks")


@pytest.mark.parametrize(
    "budget_flops,optimizer_steps,actual_flops",
    [
        (3 * 10**17, 55, 301694823746764800),
        (10**18, 182, 998335598580203520),
        (3 * 10**18, 547, 3000492156172369920),
        (10**19, 1823, 9999812067097313280),
    ],
)
def test_four_tiers_budget_and_dp2_additivity(
    budget_flops: int, optimizer_steps: int, actual_flops: int
) -> None:
    model_config = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())
    plan = budget_report(model_config, batch_sequences=128, budget=budget_flops, rounding="nearest")
    microbatch_plan = budget_report(model_config, batch_sequences=16, budget=budget_flops)
    assert plan["whole_optimizer_steps"] == optimizer_steps
    assert plan["actual_whole_step_flops"] == actual_flops
    assert plan["within_one_percent"]
    assert (
        plan["training_flops_per_optimizer_step"]
        == 2 * 4 * microbatch_plan["training_flops_per_optimizer_step"]
    )
    assert plan["tokens_per_optimizer_step"] == 1048576
    assert plan["megatron_args"]["mtp_num_layers"] is None
    assert sum(plan["megatron_args"]["linear_attention_freq"]) == 18


def test_pinned_formula_integrity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reference_path = tmp_path / "tampered.py"
    reference_path.write_text(
        megatron_budget.REFERENCE.read_text().replace(
            "forward_backward_expansion_factor = 3", "forward_backward_expansion_factor = 2"
        )
    )
    monkeypatch.setattr(megatron_budget, "REFERENCE", reference_path)
    with pytest.raises(ValueError, match="differs"):
        upstream_function()


def test_dp2_cli_plan_without_gpu_execution(tmp_path: Path) -> None:
    command_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recipes.qwen35-08b.train",
            "--budget",
            str(3 * 10**17),
            "--data-dir",
            str(tmp_path),
            "--tokenizer-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "fresh"),
            "--plan",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    plan = json.loads(command_result.stdout)
    assert (
        plan["world_size"],
        plan["local_batch_sequences"],
        plan["microbatch_sequences"],
        plan["gradient_accumulation"],
    ) == (2, 64, 16, 4)
    assert not (tmp_path / "fresh").exists()


def test_source_identity_cannot_borrow_parent_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init", str(parent)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    child = parent / "new-project"
    child.mkdir()
    monkeypatch.setattr(TRAINING, "ROOT", child)
    source_commit, source_files = TRAINING.source_identity(require_clean=False)
    assert source_commit is None
    assert source_files == {}
    with pytest.raises(ValueError, match="committed, clean"):
        TRAINING.source_identity(require_clean=True)


def test_source_identity_excludes_local_tokenizer_downloads_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = tmp_path / "recipes/qwen35-08b"
    tokenizer = recipe / "tokenizer-qwen35"
    cache = tokenizer / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (recipe / "train.py").write_text("retained training source\n")
    (recipe / "config.json").write_text('{"model_type":"qwen3_5_text"}\n')
    (tokenizer / "tokenizer.json").write_text("downloaded tokenizer identity\n")
    (cache / "tokenizer.json.metadata").write_text("mutable download metadata\n")
    monkeypatch.setattr(TRAINING, "ROOT", tmp_path)

    _, source_files = TRAINING.source_identity(require_clean=False)

    assert set(source_files) == {"recipes/qwen35-08b/train.py", "recipes/qwen35-08b/config.json"}
    (tokenizer / "tokenizer.json").write_text("a changed input must be audited as input\n")
    (cache / "tokenizer.json.metadata").write_text("a changed download timestamp\n")
    assert TRAINING.source_identity(require_clean=False)[1] == source_files


def test_gitignore_keeps_streaming_sources_and_excludes_downloads(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    sources = ("src/data/pack.py", "src/data/tokenization.py", "src/data/olmo_streaming.py")
    assets = ("data/example.jsonl.zst", "recipes/qwen35-08b/tokenizer-qwen35/tokenizer.json")
    for name in (*sources, *assets):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        status = subprocess.run(
            ["git", "-C", str(tmp_path), "check-ignore", name], capture_output=True
        )
        assert status.returncode == (1 if name in sources else 0), name


def test_training_budget_failure_retains_attempt_cost_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_directory = tmp_path / "failed-attempt"
    for environment_name, environment_value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "2"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("NCCL_P2P_DISABLE", "1"),
        ("TRITON_CACHE_DIR", str(tmp_path / "kernel-cache")),
    ):
        monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(TRAINING.torch.cuda, "device_count", Mock(return_value=2))
    monkeypatch.setattr(TRAINING.torch, "use_deterministic_algorithms", Mock())
    monkeypatch.setattr(TRAINING, "prepare_cli_environment", Mock())
    monkeypatch.setattr(TRAINING, "seed_all", Mock())
    distributed_initialization = Mock()
    monkeypatch.setattr(TRAINING, "init_distributed", distributed_initialization)
    monkeypatch.setattr(TRAINING.distributed, "is_initialized", Mock(return_value=False))
    monkeypatch.setattr(TRAINING, "source_identity", Mock(return_value=("fixture", {})))
    failed_budget = Mock(side_effect=ValueError("fixture budget failure"))
    monkeypatch.setattr(TRAINING, "training_budget", failed_budget)

    with pytest.raises(ValueError, match="fixture budget failure"):
        TRAINING.main(
            [
                "--budget",
                str(3 * 10**17),
                "--data-dir",
                str(tmp_path),
                "--tokenizer-dir",
                str(tmp_path),
                "--output",
                str(output_directory),
                "--smoke-steps",
                "1",
            ]
        )

    failed_budget.assert_called_once_with(3 * 10**17)
    distributed_initialization.assert_called_once()
    attempt_result = json.loads((output_directory / "result.json").read_text())
    assert attempt_result["status"] == "failed"
    assert not attempt_result["formal_training"]
    assert attempt_result["completed_new_updates"] == attempt_result["new_update_flops"] == 0
    assert attempt_result["wall_seconds"] >= 0
    assert "ValueError: fixture budget failure" in attempt_result["error"]
    assert "unknown" in attempt_result["partial_failed_update_flops"]
    assert not (output_directory / "checkpoint").exists()


def test_training_ledger_resume_preserves_history_and_counts_new_attempt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = Mock(spec=StreamingOLMoLoader)
    loader.latest_batch_accounting = {"scored_tokens": 3, "padded_positions": 1}
    train_module = Mock(spec=TransformerTrainModule)
    train_module.device = torch.device("cpu")
    train_module.model = torch.nn.Linear(1, 1)
    train_module.optim = torch.optim.AdamW(train_module.model.parameters(), lr=3e-4)
    trainer = Mock(spec=Trainer)
    trainer.global_step = 3
    attempt_result = dict(
        status="preparing", formal_training=False, completed_new_updates=0, new_update_flops=0
    )
    update_flops = TRAINING.training_budget(3 * 10**17)["training_flops_per_optimizer_step"]
    meter = CALLBACKS.TrainingMeter(
        output_directory=tmp_path,
        result=attempt_result,
        update_flops=update_flops,
        loader=loader,
        train_module=train_module,
        started_seconds=time.monotonic(),
        world_size=2,
    )
    meter.trainer = trainer
    meter.load_state_dict({"cumulative_flops": 2 * update_flops, "scored_tokens": 20})
    monkeypatch.setattr(CALLBACKS.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(CALLBACKS.torch.cuda, "max_memory_allocated", Mock(return_value=0))

    def sum_identical_rank_counts(scored_token_count: torch.Tensor) -> None:
        scored_token_count.mul_(2)

    monkeypatch.setattr(CALLBACKS.distributed, "all_reduce", sum_identical_rank_counts)
    meter.pre_step({"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[2, 3, 4]])})
    meter.post_train_batch()

    assert meter.state_dict() == {"cumulative_flops": 3 * update_flops, "scored_tokens": 26}
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "running"
    assert result["attempted_step"] == result["cumulative_step"] == 3
    assert result["completed_new_updates"] == 1
    assert result["new_update_flops"] == update_flops
    assert result["megatron_cumulative_flops"] == 3 * update_flops
    assert result["scored_tokens"] == 26
    step_records = [
        json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    assert len(step_records) == 1
    assert step_records[0]["step"] == 3
    assert step_records[0]["cumulative_flops"] == 3 * update_flops
    assert step_records[0]["scored_tokens"] == 26
    assert len(step_records[0]["input_sha256"]) == 64
    assert step_records[0]["lr"] == 3e-4
