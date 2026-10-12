"""Single-node DP2 Qwen3.5 text training at four pinned Megatron budgets.

Real compressed official inputs, local HF tokenizer only, random initialization.
Use --smoke-steps for a separate, non-candidate engineering run. Every attempt
requires a fresh directory, including full-state resumes of an earlier attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
TIERS = (3 * 10**17, 10**18, 3 * 10**18, 10**19)
GLOBAL_BATCH = 128
SEQUENCE_LENGTH = 8192


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(model="qwen35")
    p.add_argument("--budget", type=int, choices=tuple(TIERS), required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--dp-world-size", type=int, choices=(2,), default=2)
    p.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=16)
    p.add_argument("--plan", action="store_true", help="Print frozen CPU-only budget plan and exit")
    p.add_argument("--smoke-steps", type=int, help="Cumulative target 1..8; never a candidate")
    a = p.parse_args(argv)
    if a.output.exists() and int(os.environ.get("RANK", "0")) == 0:
        p.error("Every attempt requires a fresh output directory")
    if a.smoke_steps is not None and not 1 <= a.smoke_steps <= 8:
        p.error("Engineering smoke is bounded to 1..8 cumulative steps")
    return a


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_identity(*, require_clean):
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True
    )
    own_repo = top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    source = revision.stdout.strip() if own_repo and revision.returncode == 0 else None
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True
    )
    if require_clean and (source is None or dirty.returncode or dirty.stdout.strip()):
        raise ValueError("Formal training requires a committed, clean source checkout")
    files = {}
    tokenizer_assets = ROOT / "recipes/qwen35-08b/tokenizer-qwen35"
    for folder in ("src", "recipes", "third_party", "eval", "scripts"):
        for path in sorted((ROOT / folder).rglob("*")):
            if (
                path.is_file()
                and not path.is_relative_to(tokenizer_assets)
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return source, files


def main(argv=None):
    a = arguments(argv)
    if a.plan:
        sys.path.insert(0, str(ROOT / "src"))
        from utils.megatron_budget import budget_report

        config = json.loads((ROOT / "recipes/qwen35-08b/config.json").read_text())
        plan = budget_report(
            config,
            sequence_length=SEQUENCE_LENGTH,
            batch_sequences=GLOBAL_BATCH,
            budget=a.budget,
            rounding="nearest",
        )
        plan.update(
            world_size=2,
            local_batch_sequences=64,
            microbatch_sequences=a.microbatch,
            gradient_accumulation=64 // a.microbatch,
            seed=20261011,
        )
        print(json.dumps(plan, indent=2))
        return
    rank = int(os.environ.get("RANK", "0"))
    out = a.output if rank == 0 else a.output / "ranks" / f"rank{rank}"
    if rank == 0:
        a.output.mkdir(parents=True)
    started = time.monotonic()
    result = {
        "status": "preparing",
        "formal_training": a.smoke_steps is None,
        "completed_new_updates": 0,
        "new_update_flops": 0,
    }
    if rank == 0:
        write(out / "result.json", result)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        # OLMo otherwise creates a different per-PID Triton autotune cache.
        # Keep the same measured reduction/tile choices across native resumes.
        kernel_cache = (
            a.output.parent / "kernel-cache" / f"{a.model}-dp{a.dp_world_size}" / f"rank{rank}"
        )
        kernel_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(kernel_cache.resolve())
        sys.path[:0] = [str(ROOT), str(ROOT / "src")]
        import third_party as third_party  # isort: skip
        import numpy as np
        import torch
        from olmo_core.config import DType
        from olmo_core.optim import AdamWConfig
        from olmo_core.optim.scheduler import CosWithWarmup
        from olmo_core.train import (
            Duration,
            TrainerConfig,
            init_distributed,
            prepare_cli_environment,
        )
        from olmo_core.train.callbacks import Callback, CheckpointerCallback
        from olmo_core.train.callbacks.checkpointer import CheckpointRemovalStrategy
        from olmo_core.train.train_module import TransformerTrainModuleConfig
        from olmo_core.train.train_module.transformer.config import (
            TransformerActivationCheckpointingConfig,
            TransformerDataParallelConfig,
        )
        from olmo_core.train.trainer import LoadStrategy
        from olmo_core.utils import seed_all

        from arch import spec_to_olmo
        from data.olmo_streaming import build_official_streaming_loader
        from train.config import TrainSpec
        from train.loss import ChunkedLMHeadConfig
        from utils.megatron_budget import budget_report

        if int(os.environ.get("WORLD_SIZE", "1")) != a.dp_world_size:
            raise ValueError("WORLD_SIZE differs from the declared data parallel degree")
        if torch.cuda.device_count() != a.dp_world_size:
            raise ValueError("Visible CUDA devices differ from the declared data parallel degree")
        source, source_files = source_identity(require_clean=a.smoke_steps is None)
        torch.use_deterministic_algorithms(True)
        prepare_cli_environment()
        if a.dp_world_size > 1:
            init_distributed(timeout=timedelta(minutes=10))
        else:
            torch.cuda.set_device(0)
        out.mkdir(parents=True, exist_ok=True)
        seed = 20261011
        seed_all(seed)
        batch, length = GLOBAL_BATCH, SEQUENCE_LENGTH
        local_batch = batch // a.dp_world_size
        microbatch = a.microbatch
        config_path = ROOT / "recipes/qwen35-08b/config.json"
        config = json.loads(config_path.read_text())
        budget = budget_report(
            config,
            sequence_length=length,
            batch_sequences=batch,
            budget=a.budget,
            rounding="nearest",
        )
        if not budget["within_one_percent"]:
            raise ValueError("Whole optimizer steps fall outside the competition budget tolerance")
        horizon = budget["whole_optimizer_steps"]
        target = a.smoke_steps or horizon
        warmup = max(10, round(0.02 * horizon))
        spec = importlib.import_module("recipes.qwen35-08b.model").build_spec()
        spec.dtype = "float32"
        for layer in spec.layers:
            layer.attn_kwargs["backend"] = "flash_2" if layer.attn == "full" else "fla"
        settings = dict(
            model=a.model,
            budget=a.budget,
            batch_sequences=batch,
            dp_world_size=a.dp_world_size,
            local_batch_sequences=local_batch,
            sequence_length=length,
            microbatch_sequences=microbatch,
            gradient_accumulation=local_batch // microbatch,
            seed=seed,
            loss_normalization="DDP mean of per-rank CE / per-rank valid tokens",
            peak_lr=3e-4,
            cosine_final_fraction=0.1,
            warmup_steps=warmup,
            scheduler_horizon=horizon,
            betas=[0.9, 0.95],
            weight_decay=0.1,
            max_grad_norm=1.0,
            parameter_dtype="float32",
            autocast="bfloat16",
            deterministic=True,
            activation_checkpointing="full",
            ce_chunk_tokens=1024,
            nccl_p2p_disable=os.environ["NCCL_P2P_DISABLE"],
            source_commit=source,
            source_files_sha256=source_files,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            formal_training=a.smoke_steps is None,
        )
        write(out / "settings.json", settings)
        write(out / "megatron-budget.json", budget)
        write(
            out / "environment.json",
            dict(
                torch=torch.__version__,
                numpy=np.__version__,
                rank=rank,
                device=torch.cuda.get_device_name(),
                olmo_source=importlib.import_module("olmo_core").__file__,
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
            a.data_dir,
            a.tokenizer_dir,
            batch_size=local_batch,
            sequence_length=length,
            work_dir=out / "loader",
            threads=4,
            model_vocab_size=spec.vocab_size,
            rank=rank,
            world_size=a.dp_world_size,
        )
        write(out / "input-identity.json", loader.identity_details)
        train = TrainSpec(
            arch=spec.to_dict(),
            data={"format": "compressed-stream-v1", "identity": loader.identity_details},
            save_folder=str(a.output / "checkpoint"),
            rank_microbatch_size=microbatch,
            max_steps=horizon,
            resume_from=str(a.resume) if a.resume else None,
            lr=3e-4,
            warmup_steps=warmup,
            scheduler_t_max=horizon,
            seed=seed,
            autocast="bfloat16",
            loss_backend="torch_chunked",
            data_parallel="ddp",
        )
        write(out / "train.json", train.to_dict())
        write(
            out / "training-path.json",
            dict(
                parent_checkpoint=str(a.resume) if a.resume else None,
                initialization="full native checkpoint" if a.resume else "random",
                target_step=target,
                scheduler_horizon=horizon,
                formal_training=a.smoke_steps is None,
            ),
        )
        if a.resume:
            missing_states = [
                str(a.resume / "train" / f"rank{i}.pt")
                for i in range(a.dp_world_size)
                if not (a.resume / "train" / f"rank{i}.pt").is_file()
            ]
            if missing_states:
                raise ValueError(f"Full resume requires every rank state: {missing_states}")
            previous = a.resume.parent.parent
            if rank:
                previous = previous / "ranks" / f"rank{rank}"
            if json.loads((previous / "settings.json").read_text()) != settings:
                raise ValueError(
                    "Full resume requires identical frozen training settings and source commit"
                )
            if (
                json.loads((previous / "input-identity.json").read_text())
                != loader.identity_details
            ):
                raise ValueError("Input identity changed since the checkpoint")
        mc = spec_to_olmo(spec)
        mc.init_seed = seed
        mc.lm_head = ChunkedLMHeadConfig(
            layer_norm=mc.lm_head.layer_norm, bias=False, dtype=DType.float32, chunk_tokens=1024
        )
        model = mc.build(init_device="meta")
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Expected FP32 master parameters throughout")
        module = TransformerTrainModuleConfig(
            rank_microbatch_size=microbatch * length,
            max_sequence_length=length,
            optim=AdamWConfig(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1),
            scheduler=CosWithWarmup(warmup=warmup, t_max=horizon, alpha_f=0.1),
            max_grad_norm=1.0,
            autocast_precision=DType.bfloat16,
            dp_config=(
                TransformerDataParallelConfig(
                    name="ddp", param_dtype=DType.float32, reduce_dtype=DType.float32
                )
                if a.dp_world_size > 1
                else None
            ),
            ac_config=TransformerActivationCheckpointingConfig(mode="full"),
        ).build(model)
        if any(p.dtype != torch.float32 for p in module.model.parameters()):
            raise ValueError("Parallel wrapping changed the FP32 master parameter dtype")

        def megatron_per_token(self, seq_len):
            if seq_len != length:
                raise ValueError("Sequence length changed after protocol freeze")
            return int(budget["training_flops_per_token"])

        module.num_flops_per_token = MethodType(megatron_per_token, module)
        trainer = TrainerConfig(
            save_folder=str(a.output / "checkpoint"),
            load_strategy=LoadStrategy.never,
            max_duration=Duration.steps(target),
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
        ).build(module, loader)

        def digest(value):
            h = hashlib.sha256()

            def visit(v):
                if isinstance(v, torch.Tensor):
                    t = v.detach().cpu().contiguous()
                    h.update(str((tuple(t.shape), str(t.dtype))).encode())
                    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
                elif isinstance(v, np.ndarray):
                    h.update(str((v.shape, str(v.dtype))).encode())
                    h.update(v.tobytes())
                elif isinstance(v, dict):
                    for k in sorted(v, key=str):
                        h.update(repr(k).encode())
                        visit(v[k])
                elif isinstance(v, (tuple, list)):
                    h.update(type(v).__name__.encode())
                    for item in v:
                        visit(item)
                else:
                    h.update(repr(v).encode())

            visit(value)
            return h.hexdigest()

        def state_digest():
            native = trainer.state_dict()
            return dict(
                model=digest(module.model.state_dict()),
                optimizer=digest(module.optim.state_dict()),
                rng=digest(native["rng"]),
                data=digest(native["data_loader"]),
                trainer=digest(
                    {
                        k: native[k]
                        for k in ("global_step", "global_train_tokens_seen", "epoch", "world_size")
                    }
                ),
                step=trainer.global_step,
                megatron_cumulative_flops=meter.cumulative_flops,
            )

        class Meter(Callback):
            priority = 100
            cumulative_flops = 0
            scored_tokens = 0

            def state_dict(self):
                return dict(
                    cumulative_flops=self.cumulative_flops, scored_tokens=self.scored_tokens
                )

            def load_state_dict(self, state):
                self.cumulative_flops = state["cumulative_flops"]
                self.scored_tokens = state["scored_tokens"]

            def pre_step(self, batch):
                self.batch_input_sha256 = digest(batch) if self.step <= 8 else None
                self.tick = time.monotonic()
                result["attempted_step"] = self.step

            def post_train_batch(self):
                torch.cuda.synchronize()
                self.cumulative_flops += budget["training_flops_per_optimizer_step"]
                if self.cumulative_flops != self.step * budget["training_flops_per_optimizer_step"]:
                    raise ValueError("The retained training path has inconsistent cumulative FLOPs")
                count = torch.tensor(
                    loader.latest_batch_accounting["scored_tokens"],
                    dtype=torch.int64,
                    device=module.device,
                )
                if a.dp_world_size > 1:
                    torch.distributed.all_reduce(count)
                self.scored_tokens += int(count.item())
                result.update(
                    status="running",
                    completed_new_updates=result["completed_new_updates"] + 1,
                    cumulative_step=self.step,
                    megatron_cumulative_flops=self.cumulative_flops,
                    scored_tokens=self.scored_tokens,
                    new_update_flops=(result["completed_new_updates"] + 1)
                    * budget["training_flops_per_optimizer_step"],
                    wall_seconds=time.monotonic() - started,
                )
                row = dict(
                    step=self.step,
                    seconds=time.monotonic() - self.tick,
                    cumulative_flops=self.cumulative_flops,
                    scored_tokens=self.scored_tokens,
                    lr=module.optim.param_groups[0]["lr"],
                    input_sha256=self.batch_input_sha256,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                )
                with (out / "steps.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                write(out / "result.json", result)

            def log_metrics(self, step, metrics):
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(dict(step=step, metrics=metrics)) + "\n")
                if any(
                    not math.isfinite(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                    and ("loss" in k.lower() or "grad_norm" in k.lower())
                ):
                    raise ValueError("Nonfinite loss or gradient norm; retain attempt for review")

            def post_train(self):
                write(out / "final-state-digests.json", state_digest())

        meter = Meter()
        trainer.add_callback("megatron_ledger", meter)
        if a.resume:
            trainer.load_checkpoint(str(a.resume), load_trainer_state=True, load_optim_state=True)
            write(out / "loaded-state-digests.json", state_digest())
            if (
                meter.cumulative_flops
                != trainer.global_step * budget["training_flops_per_optimizer_step"]
            ):
                raise ValueError("Restored cumulative budget does not match global_step")
        if trainer.global_step >= target:
            raise ValueError("The selected checkpoint already reached the cumulative target")
        result["input_preparation_seconds"] = time.monotonic() - started
        trainer.fit()
        if trainer.global_step != target:
            raise ValueError("Training ended before the declared target")
        result["status"] = "completed" if a.smoke_steps is None else "smoke_completed"
        result["final_checkpoint"] = str(a.output / "checkpoint" / f"step{target}")
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        result["partial_failed_update_flops"] = "unknown; completed updates are a lower bound"
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        if out.exists():
            write(out / "result.json", result)
        if "torch" in locals() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
