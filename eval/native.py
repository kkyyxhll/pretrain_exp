"""Evaluate fixed weights with official packing and the original byte denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Literal

import third_party as third_party  # isort: skip
import numpy as np
import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from eval.checkpoint import (
    PRECISION_MODES,
    CacheMetadata,
    EvaluateOptions,
    PrecisionMode,
    evaluation_autocast,
    load_model,
    prepare_model_precision,
    saved_training_spec,
    validate_checkpoint_layout,
    validate_evaluation_cache,
    validate_master_checkpoint,
)
from eval.official import eval as official_evaluator
from src.data.olmo_streaming import HFStreamingTokenizer

__all__ = [
    "evaluation_autocast",
    "prepare_model_precision",
    "validate_master_checkpoint",
    "validate_evaluation_cache",
    "official_module",
    "check_protocol",
    "source_commit",
    "path_label",
    "write_json",
    "prepare",
    "partition_batch",
    "validate_checkpoint_layout",
    "token_nll",
    "saved_training_spec",
    "load_model",
    "evaluate",
    "main",
]
ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
OFFICIAL_EVALUATOR_SHA256 = "264a869600ee3923fc56cb2683065cd708fd36f11bb072cc4479393f34b3fcfd"
MAX_SEQUENCE_LENGTH = 8192
CE_PARITY_PREFIX_TOKENS = 128
CE_PARITY_TOLERANCE = 1e-5
PROGRESS_INTERVAL_BATCHES = 10


@dataclass(kw_only=True)
class PrepareOptions:
    """Hold the paths needed to prepare a local evaluation cache."""

    dataset_path: Path
    tokenizer: Path
    cache_dir: Path


@dataclass(kw_only=True)
class EvaluationRecord:
    """Record the selected inputs, precision and fixed-weight evaluation scope."""

    spec: str
    checkpoint: str
    cache_metadata: CacheMetadata
    world_size: int
    batch_size: int
    sequence_length: int
    loss_chunk_tokens: int
    checkpoint_storage_dtypes: list[str]
    precision_mode: PrecisionMode
    parameter_dtypes: list[str]
    autocast_dtype: Literal["bfloat16"] | None
    unique_parameters: int
    full_logits_ce_parity_max_error: float
    evaluator_commit: str
    compute_dtype: Literal["bfloat16"] = "bfloat16"
    model_training: Literal[False] = False
    gradients_enabled: Literal[False] = False
    bpb_protocol: str = "official pack_jsonl / packed_batch and original UTF-8 denominator"
    timing_scope: str = "development runtime including CE, not competition EvalTime"


@dataclass(kw_only=True)
class EvaluationProgress:
    """Record cumulative loss and coverage at a completed evaluation batch."""

    completed_batches: int
    total_batches: int
    scored_tokens: int
    ce_nats_per_token: float
    elapsed_seconds: float


@dataclass(kw_only=True)
class EvaluationResult:
    """Report fixed-weight loss; complete BPB requires full dataset coverage."""

    full_dataset: bool
    checkpoint: str
    precision_mode: PrecisionMode
    nll_nats: float
    scored_tokens: int
    ce_nats_per_token: float
    original_utf8_bytes: int | None
    bpb: float | None
    elapsed_seconds: float
    peak_device_memory_bytes: int
    status: Literal["complete"] = "complete"
    official_evaltime: None = None


def official_module() -> ModuleType:
    """Return the unchanged evaluator module for its established packing API."""
    return official_evaluator


def check_protocol() -> None:
    """Verify the packing reference before using an evaluation cache."""
    evaluator_sha256 = hashlib.sha256((ROOT / "eval/official/eval.py").read_bytes()).hexdigest()
    if evaluator_sha256 != OFFICIAL_EVALUATOR_SHA256:
        raise ValueError("Official packing snapshot changed")
    else:
        pass


def source_commit() -> str:
    """Read the evaluator revision required for checkpoint evaluation evidence."""
    command_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    if command_result.returncode:
        raise ValueError("Commit the source before checkpoint evaluation")
    else:
        return command_result.stdout.strip()


def path_label(path: str | Path) -> str:
    """Prefer repository-relative paths in evaluation records."""
    resolved_path = Path(path).resolve()
    try:
        return str(resolved_path.relative_to(ROOT))
    except ValueError:
        return str(resolved_path)


def write_json(
    path: Path, value: CacheMetadata | EvaluationRecord | EvaluationProgress | EvaluationResult
) -> None:
    """Replace a JSON record atomically so interrupted writes remain distinguishable."""
    payload = value if isinstance(value, dict) else asdict(value)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary_path.replace(path)


def prepare(options: PrepareOptions) -> None:
    """Prepare official tokens and masks from unchanged evaluation documents."""
    check_protocol()
    if options.dataset_path.suffix != ".jsonl":
        raise ValueError("dataset must be a decompressed JSONL file")
    else:
        options.cache_dir.mkdir(parents=True, exist_ok=False)
    started_seconds = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer, local_files_only=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("A serializable fast HF tokenizer is required")
    else:
        tokenizer_identity = HFStreamingTokenizer(tokenizer).identity
    metadata: CacheMetadata = official_module().pack_jsonl(
        options.dataset_path, tokenizer, options.cache_dir
    )
    metadata.update(
        dataset_path=path_label(options.dataset_path),
        dataset_file_bytes=options.dataset_path.stat().st_size,
        tokenizer_path=path_label(options.tokenizer),
        tokenizer_identity=tokenizer_identity,
        vocab_size=len(tokenizer),
        packing="unaltered eval/official/eval.py:pack_jsonl",
        preparation_seconds=time.perf_counter() - started_seconds,
    )
    write_json(options.cache_dir / "metadata.json", metadata)
    print(json.dumps(metadata), flush=True)


def partition_batch(
    step: int, rank: int, world_size: int, batch_size: int, num_sequences: int
) -> tuple[int, int]:
    """Select disjoint sequence rows for one evaluation rank and batch."""
    start_sequence = step * world_size * batch_size + rank * batch_size
    return start_sequence, min(start_sequence + batch_size, num_sequences)


def token_nll(losses: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum per-token loss in FP64, excluding masked EOS and padding targets."""
    if losses.shape != labels.shape:
        raise ValueError("per-token loss and shifted labels must have identical shapes")
    else:
        selected_losses = losses[labels != -100]
        if not torch.isfinite(selected_losses).all():
            raise RuntimeError("non-finite evaluation loss")
        else:
            return selected_losses.double().sum()


def evaluate(options: EvaluateOptions) -> None:
    """Evaluate fixed weights with complete text-token coverage and original bytes."""
    if not 1 <= options.sequence_length <= MAX_SEQUENCE_LENGTH or options.batch_size < 1:
        raise ValueError("invalid sequence length or batch size")
    elif options.max_batches is not None and options.max_batches < 1:
        raise ValueError("max-batches must be positive")
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    resolved_device = torch.device("cuda", rank)
    torch.cuda.set_device(resolved_device)
    if rank == 0:
        check_protocol()
        options.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        pass
    metadata_value: object = json.loads((options.cache_dir / "metadata.json").read_text())
    training_spec_value: object = json.loads(options.spec.read_text())
    if not isinstance(metadata_value, dict) or not isinstance(training_spec_value, dict):
        raise ValueError("Evaluation metadata and training specification must be mappings")
    elif not all(
        isinstance(metadata_value.get(field_name), int)
        for field_name in ("total_bytes", "total_tokens", "pad_token_id")
    ):
        raise ValueError("Evaluation metadata requires integer byte, token and padding fields")
    else:
        metadata: CacheMetadata = metadata_value
        validate_evaluation_cache(training_spec_value, metadata, options.precision)
    tokens = np.memmap(options.cache_dir / "tokens.bin", dtype=np.int32, mode="r")
    text_mask = np.memmap(options.cache_dir / "is_text.bin", dtype=np.bool_, mode="r")
    if len(tokens) != len(text_mask) or int(text_mask.sum()) != metadata["total_tokens"]:
        raise ValueError("packed cache lengths or text-token count differ")
    else:
        model, storage_dtypes = load_model(options, resolved_device)
    distributed.init_process_group("nccl", device_id=resolved_device, timeout=timedelta(minutes=20))
    evaluator = official_module()
    num_sequences = math.ceil((len(tokens) - 1) / options.sequence_length)
    all_steps = math.ceil(num_sequences / (options.batch_size * world_size))
    steps = min(all_steps, options.max_batches) if options.max_batches else all_steps

    def batch(start: int, stop: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_array, label_array = evaluator.packed_batch(
            tokens, text_mask, start, stop, length, metadata["pad_token_id"]
        )
        return torch.from_numpy(input_array).to(resolved_device), torch.from_numpy(label_array).to(
            resolved_device
        )

    with torch.inference_mode(), evaluation_autocast(options.precision, resolved_device):
        input_ids, labels = batch(0, 1, min(CE_PARITY_PREFIX_TOKENS, options.sequence_length))
        logits = model(input_ids)
        reference_losses = functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none"
        ).reshape_as(labels)
        actual_losses = model(
            input_ids, labels=labels, loss_reduction="none", return_logits=False
        ).ce_loss
        torch.testing.assert_close(
            actual_losses, reference_losses, rtol=CE_PARITY_TOLERANCE, atol=CE_PARITY_TOLERANCE
        )
        parity_max_error = float((actual_losses - reference_losses).abs().max())
        del logits, reference_losses, actual_losses, input_ids, labels
    distributed.barrier()
    run_metadata = EvaluationRecord(
        spec=path_label(options.spec),
        checkpoint=path_label(options.checkpoint),
        cache_metadata=metadata,
        world_size=world_size,
        batch_size=options.batch_size,
        sequence_length=options.sequence_length,
        loss_chunk_tokens=options.loss_chunk_tokens,
        checkpoint_storage_dtypes=storage_dtypes,
        precision_mode=options.precision,
        parameter_dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()}),
        autocast_dtype="bfloat16" if options.precision == "fp32-master-bf16" else None,
        unique_parameters=sum(parameter.numel() for parameter in model.parameters()),
        full_logits_ce_parity_max_error=parity_max_error,
        evaluator_commit=source_commit(),
    )
    if rank == 0:
        write_json(options.output_dir / "run_meta.json", run_metadata)
        LOGGER.info(
            f"Loaded trained checkpoint; {num_sequences} sequences, {steps} batches; CE parity error={parity_max_error}"
        )
    else:
        pass
    total_nll = torch.zeros((), device=resolved_device, dtype=torch.float64)
    scored_token_count = torch.zeros((), device=resolved_device, dtype=torch.int64)
    started_seconds = time.perf_counter()
    with torch.inference_mode(), evaluation_autocast(options.precision, resolved_device):
        for step in range(steps):
            start_sequence, stop_sequence = partition_batch(
                step, rank, world_size, options.batch_size, num_sequences
            )
            if start_sequence < stop_sequence:
                input_ids, labels = batch(start_sequence, stop_sequence, options.sequence_length)
                loss_output = model(
                    input_ids, labels=labels, loss_reduction="none", return_logits=False
                )
                total_nll += token_nll(loss_output.ce_loss, labels)
                scored_token_count += (labels != -100).sum()
                del input_ids, labels, loss_output
            else:
                pass
            if (step + 1) % PROGRESS_INTERVAL_BATCHES == 0 or step + 1 == steps:
                reduced_counts = torch.stack((total_nll, scored_token_count.double()))
                distributed.all_reduce(reduced_counts)
                if rank == 0:
                    progress = EvaluationProgress(
                        completed_batches=step + 1,
                        total_batches=steps,
                        scored_tokens=int(reduced_counts[1]),
                        ce_nats_per_token=float(reduced_counts[0] / reduced_counts[1]),
                        elapsed_seconds=time.perf_counter() - started_seconds,
                    )
                    write_json(options.output_dir / "progress.json", progress)
                    LOGGER.info(f"{json.dumps(asdict(progress))}")
                else:
                    pass
            else:
                pass
    distributed.all_reduce(total_nll)
    distributed.all_reduce(scored_token_count)
    is_full_dataset = steps == all_steps
    if is_full_dataset and int(scored_token_count) != metadata["total_tokens"]:
        raise RuntimeError("scored token count differs from official packed text tokens")
    else:
        pass
    result = EvaluationResult(
        full_dataset=is_full_dataset,
        checkpoint=path_label(options.checkpoint),
        precision_mode=options.precision,
        nll_nats=float(total_nll),
        scored_tokens=int(scored_token_count),
        ce_nats_per_token=float(total_nll / scored_token_count),
        original_utf8_bytes=metadata["total_bytes"] if is_full_dataset else None,
        bpb=float(total_nll) / (math.log(2) * metadata["total_bytes"]) if is_full_dataset else None,
        elapsed_seconds=time.perf_counter() - started_seconds,
        peak_device_memory_bytes=torch.cuda.max_memory_allocated(resolved_device),
    )
    if not math.isfinite(result.ce_nats_per_token):
        raise RuntimeError("non-finite evaluation result")
    elif rank == 0:
        write_json(options.output_dir / "result.json", result)
        print(json.dumps(asdict(result)), flush=True)
    else:
        pass
    distributed.barrier()
    distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    """Run cache preparation or fixed-weight evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = modes.add_parser("prepare")
    prepare_parser.add_argument("--dataset-path", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer", type=Path, required=True)
    prepare_parser.add_argument("--cache-dir", type=Path, required=True)
    evaluate_parser = modes.add_parser("evaluate")
    for path_flag in ("spec", "checkpoint", "cache-dir", "output-dir"):
        evaluate_parser.add_argument(f"--{path_flag}", type=Path, required=True)
    evaluate_parser.add_argument("--sequence-length", type=int, default=8192)
    evaluate_parser.add_argument("--batch-size", type=int, default=4)
    evaluate_parser.add_argument("--loss-chunk-tokens", type=int, default=512)
    evaluate_parser.add_argument("--max-batches", type=int)
    evaluate_parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="bf16",
        help="bf16 evaluates BF16 weights; fp32-master-bf16 preserves FP32 checkpoint parameters with BF16 autocast",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    if arguments.mode == "prepare":
        prepare(
            PrepareOptions(
                dataset_path=arguments.dataset_path,
                tokenizer=arguments.tokenizer,
                cache_dir=arguments.cache_dir,
            )
        )
    else:
        evaluate(
            EvaluateOptions(
                spec=arguments.spec,
                checkpoint=arguments.checkpoint,
                cache_dir=arguments.cache_dir,
                output_dir=arguments.output_dir,
                sequence_length=arguments.sequence_length,
                batch_size=arguments.batch_size,
                loss_chunk_tokens=arguments.loss_chunk_tokens,
                max_batches=arguments.max_batches,
                precision=arguments.precision,
            )
        )


if __name__ == "__main__":
    main()
else:
    pass
