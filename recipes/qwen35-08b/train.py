"""Train the Qwen3.5 text backbone at four pinned reference budgets."""

from __future__ import annotations

import third_party as third_party  # isort: skip

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import MethodType

import numpy as np
import olmo_core
import torch
import torch.distributed as distributed
from olmo_core.config import DType
from olmo_core.distributed.utils import init_distributed
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback, CheckpointRemovalStrategy
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule
from olmo_core.utils import prepare_cli_environment, seed_all

from eval.native import evaluate_in_memory
from src.arch.core.spec import spec_to_olmo
from src.data.olmo_streaming import build_official_streaming_loader
from src.train.config import TrainSpec
from src.train.loss import ChunkedLMHeadConfig
from src.utils.megatron_budget import BudgetPayload, budget_report

from .callbacks import AttemptResult, TrainingMeter, write_json
from .model import build_spec

ROOT = Path(__file__).resolve().parents[2]
TIERS = (3 * 10**17, 10**18, 3 * 10**18, 10**19)
GLOBAL_BATCH = 128
SEQUENCE_LENGTH = 8192
SEED = 20261011
PEAK_LEARNING_RATE = 3e-4
CE_CHUNK_TOKENS = 1024


@dataclass(kw_only=True)
class TrainingArguments:
    """Keep command-line options distinct from the serialized training spec."""

    budget: int
    data_dir: Path
    tokenizer_dir: Path
    output: Path
    resume: Path | None
    dp_world_size: int
    microbatch: int
    plan: bool
    smoke_steps: int | None
    model: str = "qwen35"
    no_checkpoints: bool = False
    eval_cache: Path | None = None


def arguments(argv: Sequence[str] | None = None) -> TrainingArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(model="qwen35")
    parser.add_argument("--budget", type=int, choices=TIERS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dp-world-size", type=int, choices=(2,), default=2)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=16)
    parser.add_argument(
        "--plan", action="store_true", help="Print the budget plan without training"
    )
    parser.add_argument("--smoke-steps", type=int, help="Cumulative target 1..8; never a candidate")
    parser.add_argument("--no-checkpoints", action="store_true", help="Keep weights only in memory")
    parser.add_argument(
        "--eval-cache", type=Path, help="Prepared full A cache for post-training evaluation"
    )
    options = TrainingArguments(**vars(parser.parse_args(argv)))
    if options.output.exists() and int(os.environ.get("RANK", "0")) == 0:
        parser.error("Every attempt requires a fresh output directory")
    elif options.smoke_steps is not None and not 1 <= options.smoke_steps <= 8:
        parser.error("Engineering smoke is bounded to 1..8 cumulative steps")
    elif options.no_checkpoints and options.resume:
        parser.error("Checkpoint-free training cannot resume")
    elif options.no_checkpoints and options.smoke_steps is None and options.eval_cache is None:
        parser.error("Checkpoint-free formal training requires --eval-cache")
    else:
        return options


def source_identity(*, require_clean: bool) -> tuple[str | None, dict[str, str]]:
    """Identify this checkout while excluding local tokenizer assets and bytecode."""
    repository_top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True
    )
    own_repository = (
        repository_top.returncode == 0
        and Path(repository_top.stdout.strip()).resolve() == ROOT.resolve()
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    source = revision.stdout.strip() if own_repository and revision.returncode == 0 else None
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True
    )
    if require_clean and (source is None or dirty.returncode or dirty.stdout.strip()):
        raise ValueError("Formal training requires a committed, clean source checkout")
    else:
        pass
    source_files: dict[str, str] = {}
    tokenizer_assets = ROOT / "recipes/qwen35-08b/tokenizer-qwen35"
    for folder in ("src", "recipes", "third_party", "eval", "scripts"):
        for source_path in sorted((ROOT / folder).rglob("*")):
            if (
                source_path.is_file()
                and not source_path.is_relative_to(tokenizer_assets)
                and "__pycache__" not in source_path.parts
                and source_path.suffix != ".pyc"
            ):
                source_files[str(source_path.relative_to(ROOT))] = hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest()
            else:
                pass
    return source, source_files


def training_budget(budget_flops: int) -> BudgetPayload:
    """Use the same frozen global batch for planning and actual training."""
    return budget_report(
        json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text()),
        sequence_length=SEQUENCE_LENGTH,
        batch_sequences=GLOBAL_BATCH,
        budget=budget_flops,
        rounding="nearest",
    )


def main(argv: Sequence[str] | None = None) -> None:
    options = arguments(argv)
    config_path = ROOT / "recipes/qwen35-08b/config.json"
    local_batch_sequences = GLOBAL_BATCH // options.dp_world_size
    if options.plan:
        plan = training_budget(options.budget)
        plan.update(
            world_size=2,
            local_batch_sequences=64,
            microbatch_sequences=options.microbatch,
            gradient_accumulation=64 // options.microbatch,
            seed=SEED,
        )
        print(json.dumps(plan, indent=2))
        return
    else:
        pass
    rank = int(os.environ.get("RANK", "0"))
    output_directory = options.output if rank == 0 else options.output / "ranks" / f"rank{rank}"
    if rank == 0:
        options.output.mkdir(parents=True)
    else:
        pass
    started_seconds = time.monotonic()
    result: AttemptResult = {
        "status": "preparing",
        "formal_training": options.smoke_steps is None,
        "completed_new_updates": 0,
        "new_update_flops": 0,
        "checkpoint_saved": False,
    }
    training_finished = False
    if rank == 0:
        write_json(output_directory / "result.json", result)
    else:
        pass
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        # note (kkyyxhll): Resumes reuse measured Triton tile choices instead of a per-PID cache.
        kernel_cache = (
            options.output.parent
            / "kernel-cache"
            / f"{options.model}-dp{options.dp_world_size}"
            / f"rank{rank}"
        )
        kernel_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(kernel_cache.resolve())
        if int(os.environ.get("WORLD_SIZE", "1")) != options.dp_world_size:
            raise ValueError("WORLD_SIZE differs from the declared data parallel degree")
        elif torch.cuda.device_count() != options.dp_world_size:
            raise ValueError("Visible CUDA devices differ from the declared data parallel degree")
        else:
            pass
        source, source_files = source_identity(require_clean=options.smoke_steps is None)
        torch.use_deterministic_algorithms(True)
        prepare_cli_environment()
        init_distributed(timeout=timedelta(minutes=10))
        output_directory.mkdir(parents=True, exist_ok=True)
        seed_all(SEED)
        budget = training_budget(options.budget)
        if not budget["within_one_percent"]:
            raise ValueError("Whole optimizer steps fall outside the competition budget tolerance")
        else:
            pass
        horizon_steps = budget["whole_optimizer_steps"]
        target_steps = options.smoke_steps or horizon_steps
        warmup_steps = max(10, round(0.02 * horizon_steps))
        arch_spec = build_spec()
        arch_spec.dtype = "float32"
        for layer in arch_spec.layers:
            layer.attn_kwargs["backend"] = "flash_2" if layer.attn == "full" else "fla"
        settings = dict(
            model=options.model,
            budget=options.budget,
            batch_sequences=GLOBAL_BATCH,
            dp_world_size=options.dp_world_size,
            local_batch_sequences=local_batch_sequences,
            sequence_length=SEQUENCE_LENGTH,
            microbatch_sequences=options.microbatch,
            gradient_accumulation=local_batch_sequences // options.microbatch,
            seed=SEED,
            loss_normalization="DDP mean of per-rank CE / per-rank valid tokens",
            peak_lr=PEAK_LEARNING_RATE,
            cosine_final_fraction=0.1,
            warmup_steps=warmup_steps,
            scheduler_horizon=horizon_steps,
            betas=[0.9, 0.95],
            weight_decay=0.1,
            max_grad_norm=1.0,
            parameter_dtype="float32",
            autocast="bfloat16",
            deterministic=True,
            activation_checkpointing="full",
            ce_chunk_tokens=CE_CHUNK_TOKENS,
            nccl_p2p_disable=os.environ["NCCL_P2P_DISABLE"],
            source_commit=source,
            source_files_sha256=source_files,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            formal_training=options.smoke_steps is None,
            checkpointing="disabled" if options.no_checkpoints else "full_native",
            evaluation_cache=str(options.eval_cache) if options.eval_cache else None,
        )
        write_json(output_directory / "settings.json", settings)
        write_json(output_directory / "megatron-budget.json", budget)
        write_json(
            output_directory / "environment.json",
            dict(
                torch=torch.__version__,
                numpy=np.__version__,
                rank=rank,
                device=torch.cuda.get_device_name(),
                olmo_source=olmo_core.__file__,
                packages={
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "flash-attn",
                        "flash-linear-attention",
                        "numpy",
                    )
                },
                triton_cache_dir=os.environ["TRITON_CACHE_DIR"],
                cuda=torch.version.cuda,
                visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                nccl_p2p_disable=os.environ["NCCL_P2P_DISABLE"],
            ),
        )
        loader = build_official_streaming_loader(
            options.data_dir,
            options.tokenizer_dir,
            batch_size=local_batch_sequences,
            sequence_length=SEQUENCE_LENGTH,
            work_dir=output_directory / "loader",
            threads=4,
            model_vocab_size=arch_spec.vocab_size,
            rank=rank,
            world_size=options.dp_world_size,
        )
        write_json(output_directory / "input-identity.json", loader.identity_details)
        train_spec = TrainSpec(
            arch=arch_spec.to_dict(),
            data={"format": "compressed-stream-v1", "identity": loader.identity_details},
            save_folder=str(options.output / "checkpoint"),
            rank_microbatch_size=options.microbatch,
            max_steps=horizon_steps,
            resume_from=str(options.resume) if options.resume else None,
            lr=PEAK_LEARNING_RATE,
            warmup_steps=warmup_steps,
            scheduler_t_max=horizon_steps,
            seed=SEED,
            autocast="bfloat16",
            loss_backend="torch_chunked",
            data_parallel="ddp",
        )
        write_json(output_directory / "train.json", train_spec.to_dict())
        write_json(
            output_directory / "training-path.json",
            dict(
                parent_checkpoint=str(options.resume) if options.resume else None,
                initialization="full native checkpoint" if options.resume else "random",
                target_step=target_steps,
                scheduler_horizon=horizon_steps,
                formal_training=options.smoke_steps is None,
            ),
        )
        if options.resume:
            missing_states = [
                str(options.resume / "train" / f"rank{resume_rank}.pt")
                for resume_rank in range(options.dp_world_size)
                if not (options.resume / "train" / f"rank{resume_rank}.pt").is_file()
            ]
            if missing_states:
                raise ValueError(f"Full resume requires every rank state: {missing_states}")
            else:
                pass
            previous_directory = options.resume.parent.parent
            if rank:
                previous_directory = previous_directory / "ranks" / f"rank{rank}"
            else:
                pass
            if json.loads((previous_directory / "settings.json").read_text()) != settings:
                raise ValueError(
                    "Full resume requires identical frozen training settings and source commit"
                )
            elif (
                json.loads((previous_directory / "input-identity.json").read_text())
                != loader.identity_details
            ):
                raise ValueError("Input identity changed since the checkpoint")
            else:
                pass
        else:
            pass
        model_config = spec_to_olmo(arch_spec)
        model_config.init_seed = SEED
        model_config.lm_head = ChunkedLMHeadConfig(
            layer_norm=model_config.lm_head.layer_norm,
            bias=False,
            dtype=DType.float32,
            chunk_tokens=CE_CHUNK_TOKENS,
        )
        model = model_config.build(init_device="meta")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("Expected FP32 master parameters throughout")
        else:
            pass
        train_module = TransformerTrainModuleConfig(
            rank_microbatch_size=options.microbatch * SEQUENCE_LENGTH,
            max_sequence_length=SEQUENCE_LENGTH,
            optim=AdamWConfig(lr=PEAK_LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1),
            scheduler=CosWithWarmup(warmup=warmup_steps, t_max=horizon_steps, alpha_f=0.1),
            max_grad_norm=1.0,
            autocast_precision=DType.bfloat16,
            dp_config=TransformerDataParallelConfig(
                name="ddp", param_dtype=DType.float32, reduce_dtype=DType.float32
            ),
            ac_config=TransformerActivationCheckpointingConfig(mode="full"),
        ).build(model)
        if any(parameter.dtype != torch.float32 for parameter in train_module.model.parameters()):
            raise ValueError("Parallel wrapping changed the FP32 master parameter dtype")
        else:
            pass

        def megatron_per_token(self: TransformerTrainModule, seq_len: int) -> int:
            if seq_len != SEQUENCE_LENGTH:
                raise ValueError("Sequence length changed after protocol freeze")
            else:
                return int(budget["training_flops_per_token"])

        train_module.num_flops_per_token = MethodType(megatron_per_token, train_module)
        trainer = TrainerConfig(
            save_folder=str(options.output / "checkpoint"),
            no_checkpoints=options.no_checkpoints,
            load_strategy=LoadStrategy.never,
            max_duration=Duration.steps(target_steps),
            metrics_collect_interval=5,
            callbacks={
                "checkpointer": CheckpointerCallback(
                    save_interval=500,
                    pre_train_checkpoint=False,
                    save_async=False,
                    max_checkpoints=None,
                    remove=CheckpointRemovalStrategy.never,
                )
            },
        ).build(train_module, loader)
        meter = TrainingMeter(
            output_directory=output_directory,
            result=result,
            update_flops=budget["training_flops_per_optimizer_step"],
            loader=loader,
            train_module=train_module,
            started_seconds=started_seconds,
            world_size=options.dp_world_size,
            record_state_digests=not options.no_checkpoints,
        )
        trainer.add_callback("megatron_ledger", meter)
        if options.resume:
            trainer.load_checkpoint(
                str(options.resume), load_trainer_state=True, load_optim_state=True
            )
            write_json(output_directory / "loaded-state-digests.json", meter.state_digests())
            if (
                meter.cumulative_flops
                != trainer.global_step * budget["training_flops_per_optimizer_step"]
            ):
                raise ValueError("Restored cumulative budget does not match global_step")
            else:
                pass
        else:
            pass
        if trainer.global_step >= target_steps:
            raise ValueError("The selected checkpoint already reached the cumulative target")
        else:
            pass
        result["input_preparation_seconds"] = time.monotonic() - started_seconds
        trainer.fit()
        if trainer.global_step != target_steps:
            raise ValueError("Training ended before the declared target")
        else:
            pass
        training_finished = True
        result["training_completed_wall_seconds"] = time.monotonic() - started_seconds
        if not options.no_checkpoints:
            result["checkpoint_saved"] = True
            result["final_checkpoint"] = str(options.output / "checkpoint" / f"step{target_steps}")
        if options.eval_cache:
            result["status"] = "evaluating"
            write_json(output_directory / "result.json", result)
            train_module.optim.zero_grad(set_to_none=True)
            evaluation = evaluate_in_memory(
                train_module.model,
                options.output / "train.json",
                options.eval_cache,
                options.output / "eval-a",
                training_step=trainer.global_step,
                sequence_length=SEQUENCE_LENGTH,
                batch_size=16,
                loss_chunk_tokens=CE_CHUNK_TOKENS,
            )
            result["evaluation_bpb"] = evaluation.bpb
            result["evaluation_output"] = str(options.output / "eval-a")
            result["evaluation_elapsed_seconds"] = evaluation.elapsed_seconds
        result["status"] = "completed" if options.smoke_steps is None else "smoke_completed"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        result["partial_failed_update_flops"] = (
            "none; training finished before evaluation failure"
            if training_finished
            else "unknown; completed updates are a lower bound"
        )
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started_seconds
        if output_directory.exists():
            write_json(output_directory / "result.json", result)
        else:
            pass
        if distributed.is_initialized():
            distributed.destroy_process_group()
        else:
            pass


if __name__ == "__main__":
    main()
else:
    pass
