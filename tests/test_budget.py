"""Budget reference integrity and global accounting across DP2 microbatches."""

import importlib
import json
from pathlib import Path

import pytest

from utils.megatron_budget import budget_report, upstream_function

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "budget,steps,actual",
    [
        (3 * 10**17, 55, 301694823746764800),
        (10**18, 182, 998335598580203520),
        (3 * 10**18, 547, 3000492156172369920),
        (10**19, 1823, 9999812067097313280),
    ],
)
def test_four_tiers_budget_and_dp2_additivity(budget, steps, actual):
    c = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())
    plan = budget_report(c, batch_sequences=128, budget=budget, rounding="nearest")
    micro = budget_report(c, batch_sequences=16, budget=budget)
    assert plan["whole_optimizer_steps"] == steps
    assert plan["actual_whole_step_flops"] == actual
    assert plan["within_one_percent"]
    assert (
        plan["training_flops_per_optimizer_step"]
        == 2 * 4 * micro["training_flops_per_optimizer_step"]
    )
    assert plan["tokens_per_optimizer_step"] == 1048576
    assert plan["megatron_args"]["mtp_num_layers"] is None
    assert sum(plan["megatron_args"]["linear_attention_freq"]) == 18


def test_pinned_formula_integrity(monkeypatch, tmp_path):
    import utils.megatron_budget as m

    p = tmp_path / "tampered.py"
    p.write_text(
        m.REFERENCE.read_text().replace(
            "forward_backward_expansion_factor = 3", "forward_backward_expansion_factor = 2"
        )
    )
    monkeypatch.setattr(m, "REFERENCE", p)
    with pytest.raises(ValueError, match="differs"):
        upstream_function()


def test_dp2_cli_plan_without_gpu_import(tmp_path):
    import subprocess
    import sys

    value = subprocess.run(
        [
            sys.executable,
            str(ROOT / "recipes/qwen35-08b/train.py"),
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
        capture_output=True,
        text=True,
    )
    plan = json.loads(value.stdout)
    assert (
        plan["world_size"],
        plan["local_batch_sequences"],
        plan["microbatch_sequences"],
        plan["gradient_accumulation"],
    ) == (2, 64, 16, 4)
    assert not (tmp_path / "fresh").exists()


def test_source_identity_cannot_borrow_parent_repository(tmp_path, monkeypatch):
    import subprocess

    training = importlib.import_module("recipes.qwen35-08b.train")
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
    monkeypatch.setattr(training, "ROOT", child)
    source, files = training.source_identity(require_clean=False)
    assert source is None
    assert files == {}
    with pytest.raises(ValueError, match="committed, clean"):
        training.source_identity(require_clean=True)


def test_source_identity_excludes_local_tokenizer_downloads_and_cache(tmp_path, monkeypatch):
    training = importlib.import_module("recipes.qwen35-08b.train")
    recipe = tmp_path / "recipes/qwen35-08b"
    tokenizer = recipe / "tokenizer-qwen35"
    cache = tokenizer / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (recipe / "train.py").write_text("retained training source\n")
    (recipe / "config.json").write_text('{"model_type":"qwen3_5_text"}\n')
    (tokenizer / "tokenizer.json").write_text("downloaded tokenizer identity\n")
    (cache / "tokenizer.json.metadata").write_text("mutable download metadata\n")
    monkeypatch.setattr(training, "ROOT", tmp_path)

    _, source_files = training.source_identity(require_clean=False)

    assert set(source_files) == {"recipes/qwen35-08b/train.py", "recipes/qwen35-08b/config.json"}
    (tokenizer / "tokenizer.json").write_text("a changed input must be audited as input\n")
    (cache / "tokenizer.json.metadata").write_text("a changed download timestamp\n")
    assert training.source_identity(require_clean=False)[1] == source_files


def test_gitignore_keeps_streaming_sources_and_excludes_downloads(tmp_path):
    import subprocess

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
