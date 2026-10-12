#!/usr/bin/env python3
"""Fixed-weight BPB for native OLMo checkpoints, using official packing.

This is a development evaluator. Its elapsed time includes loss computation
and is not the competition EvalTime. No optimizer or training data is loaded.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

PRECISION_MODES = ("bf16", "fp32-master-bf16")


def evaluation_autocast(precision, device):
    """Match single-device training without rounding its FP32 master gains."""
    import torch

    if precision == "bf16":
        return nullcontext()
    if precision == "fp32-master-bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"unsupported evaluation precision: {precision}")


def prepare_model_precision(model, device, precision):
    import torch

    if precision not in PRECISION_MODES:
        raise ValueError(f"unsupported evaluation precision: {precision}")
    dtype = torch.float32 if precision == "fp32-master-bf16" else torch.bfloat16
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def validate_master_checkpoint(metadata):
    """Casting rounded BF16 storage to FP32 cannot recover FP32 master values."""
    import torch

    for key, value in metadata.state_dict_metadata.items():
        if not key.startswith("model."):
            continue
        dtype = value.properties.dtype
        if dtype.is_floating_point and dtype != torch.float32:
            raise ValueError(
                f"FP32 master evaluation requires FP32 checkpoint tensors: {key} is {dtype}"
            )


def validate_evaluation_cache(spec_data, metadata, precision):
    data = spec_data.get("data", {})
    expected = data.get("tokenizer_identity")
    stream_identity = data.get("identity", {}).get("inputs", {}).get("tokenizer")
    if expected is not None and stream_identity is not None and expected != stream_identity:
        raise ValueError("Training tokenizer identities disagree")
    expected = stream_identity if expected is None else expected
    if expected is None:
        if precision == "fp32-master-bf16":
            raise ValueError("FP32 master evaluation requires a tokenizer_identity in train.json")
        return
    if metadata.get("tokenizer_identity") != expected:
        raise ValueError(
            "Evaluation cache tokenizer differs from training; prepare a matching cache"
        )


def official_module():
    spec = importlib.util.spec_from_file_location(
        "native_eval_official", ROOT / "eval/official/eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_protocol():
    import hashlib

    actual = hashlib.sha256((ROOT / "eval/official/eval.py").read_bytes()).hexdigest()
    if actual != "264a869600ee3923fc56cb2683065cd708fd36f11bb072cc4479393f34b3fcfd":
        raise ValueError("Official packing snapshot changed")


def source_commit():
    value = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    if value.returncode:
        raise ValueError("Commit the source before checkpoint evaluation")
    return value.stdout.strip()


def path_label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    from transformers import AutoTokenizer
    import third_party as third_party  # isort: skip
    from data.olmo_streaming import HFStreamingTokenizer

    check_protocol()
    if args.dataset_path.suffix != ".jsonl":
        raise ValueError("dataset must be a decompressed JSONL file")
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_identity = HFStreamingTokenizer(tokenizer).identity
    metadata = official_module().pack_jsonl(args.dataset_path, tokenizer, args.cache_dir)
    metadata.update(
        dataset_path=path_label(args.dataset_path),
        dataset_file_bytes=args.dataset_path.stat().st_size,
        tokenizer_path=path_label(args.tokenizer),
        tokenizer_identity=tokenizer_identity,
        vocab_size=len(tokenizer),
        packing="unaltered eval/official/eval.py:pack_jsonl",
        preparation_seconds=time.perf_counter() - started,
    )
    write_json(args.cache_dir / "metadata.json", metadata)
    print(json.dumps(metadata), flush=True)


def partition_batch(step, rank, world_size, batch_size, num_sequences):
    start = step * world_size * batch_size + rank * batch_size
    return start, min(start + batch_size, num_sequences)


def validate_checkpoint_layout(state, metadata):
    """Reject missing/extra model entries and shape mismatches before loading."""
    stored = metadata.state_dict_metadata
    expected = {"model." + key for key in state}
    present = {key for key in stored if key.startswith("model.")}
    if expected != present:
        raise ValueError(
            f"checkpoint model keys differ: missing={expected - present}, "
            f"unexpected={present - expected}"
        )
    for key, tensor in state.items():
        if tuple(stored["model." + key].size) != tuple(tensor.shape):
            raise ValueError(f"checkpoint shape mismatch for {key}")


def token_nll(losses, labels):
    """Sum per-token FP32 CE in FP64, excluding inserted EOS and padding."""
    import torch

    if losses.shape != labels.shape:
        raise ValueError("per-token loss and shifted labels must have identical shapes")
    selected = losses[labels != -100]
    if not torch.isfinite(selected).all():
        raise RuntimeError("non-finite evaluation loss")
    return selected.double().sum()


def saved_training_spec(checkpoint):
    # save_folder may be the run directory or its training/ child.
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        candidate = parent / "train.json"
        if candidate.is_file():
            return candidate
    raise ValueError("checkpoint run must retain its train.json architecture")


def load_model(args, device):
    import third_party as third_party  # isort: skip
    from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
    from torch.distributed.checkpoint import load

    from arch import spec_to_olmo
    from train.config import TrainSpec
    from train.loss import ChunkedLMHeadConfig

    spec_data = json.loads(args.spec.read_text())
    saved_spec = saved_training_spec(args.checkpoint)
    if json.loads(saved_spec.read_text())["arch"] != spec_data["arch"]:
        raise ValueError("evaluation architecture differs from the saved training spec")
    train = TrainSpec.from_dict(spec_data)
    precision = getattr(args, "precision", "bf16")
    if precision == "fp32-master-bf16" and train.autocast != "bfloat16":
        raise ValueError("FP32 master/BF16 evaluation requires BF16 training autocast")
    config = spec_to_olmo(train.to_arch_spec())
    config.lm_head = ChunkedLMHeadConfig(
        layer_norm=config.lm_head.layer_norm,
        bias=config.lm_head.bias,
        dtype=config.lm_head.dtype,
        chunk_tokens=args.loss_chunk_tokens,
    )
    model = config.build(init_device="cpu")
    if precision == "fp32-master-bf16":
        # DCP loads into the provided state's dtype. Promote before requesting
        # weights, so an architecture's BF16 default cannot truncate masters.
        model.float()
    state = model.state_dict()
    reader = RemoteFileSystemReader(args.checkpoint / "model_and_optim", thread_count=2)
    metadata = reader.read_metadata()
    validate_checkpoint_layout(state, metadata)
    if precision == "fp32-master-bf16":
        validate_master_checkpoint(metadata)
    # Each worker loads a complete evaluation replica before NCCL initialization.
    # Only model entries are requested, never optimizer state or pretrained weights.
    load({"model": state}, storage_reader=reader)
    model.load_state_dict(state, strict=True)
    del state
    storage_dtypes = sorted(
        {
            str(value.properties.dtype)
            for key, value in metadata.state_dict_metadata.items()
            if key.startswith("model.")
        }
    )
    # Preserve the old FSDP evaluation behavior by default. Single-device
    # FP32-master training instead keeps the master gains/decay parameters and
    # uses BF16 autocast for eligible operations in both evaluation passes.
    prepare_model_precision(model, device, precision)
    if model.tie_word_embeddings:
        if model.embeddings.weight is not model.lm_head.w_out.weight:
            raise RuntimeError("evaluation model lost tied vocabulary parameters")
    return model, storage_dtypes


def evaluate(args):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    if not 1 <= args.sequence_length <= 8192 or args.batch_size < 1:
        raise ValueError("invalid sequence length or batch size")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", rank)
    precision = getattr(args, "precision", "bf16")
    torch.cuda.set_device(device)
    if rank == 0:
        check_protocol()
        args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    validate_evaluation_cache(json.loads(args.spec.read_text()), metadata, precision)
    tokens = np.memmap(args.cache_dir / "tokens.bin", dtype=np.int32, mode="r")
    is_text = np.memmap(args.cache_dir / "is_text.bin", dtype=np.bool_, mode="r")
    if len(tokens) != len(is_text) or int(is_text.sum()) != metadata["total_tokens"]:
        raise ValueError("packed cache lengths or text-token count differ")
    model, storage_dtypes = load_model(args, device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=20))
    official = official_module()
    num_sequences = math.ceil((len(tokens) - 1) / args.sequence_length)
    all_steps = math.ceil(num_sequences / (args.batch_size * world))
    steps = min(all_steps, args.max_batches) if args.max_batches else all_steps

    def batch(start, stop, length=args.sequence_length):
        x, y = official.packed_batch(tokens, is_text, start, stop, length, metadata["pad_token_id"])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # Verify shifted labels and the memory-bounded head against full logits/CE
    # on a small real evaluation prefix. This is inference only.
    with torch.inference_mode(), evaluation_autocast(precision, device):
        x, y = batch(0, 1, min(128, args.sequence_length))
        logits = model(x)
        reference = F.cross_entropy(
            logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none"
        ).reshape_as(y)
        actual = model(x, labels=y, loss_reduction="none", return_logits=False).ce_loss
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
        parity_max_error = float((actual - reference).abs().max())
        del logits, reference, actual, x, y
    dist.barrier()
    run_meta = dict(
        spec=path_label(args.spec),
        checkpoint=path_label(args.checkpoint),
        cache_metadata=metadata,
        world_size=world,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        loss_chunk_tokens=args.loss_chunk_tokens,
        checkpoint_storage_dtypes=storage_dtypes,
        compute_dtype="bfloat16",
        precision_mode=precision,
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        autocast_dtype="bfloat16" if precision == "fp32-master-bf16" else None,
        unique_parameters=sum(p.numel() for p in model.parameters()),
        full_logits_ce_parity_max_error=parity_max_error,
        evaluator_commit=source_commit(),
        model_training=False,
        gradients_enabled=False,
        bpb_protocol="official pack_jsonl / packed_batch and original UTF-8 denominator",
        timing_scope="development runtime including CE, not competition EvalTime",
    )
    if rank == 0:
        write_json(args.output_dir / "run_meta.json", run_meta)
        print(
            f"Loaded trained checkpoint; {num_sequences} sequences, {steps} batches; "
            f"CE parity error={parity_max_error}",
            flush=True,
        )
    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.int64)
    started = time.perf_counter()
    with torch.inference_mode(), evaluation_autocast(precision, device):
        for step in range(steps):
            start, stop = partition_batch(step, rank, world, args.batch_size, num_sequences)
            if start < stop:
                x, y = batch(start, stop)
                out = model(x, labels=y, loss_reduction="none", return_logits=False)
                total_nll += token_nll(out.ce_loss, y)
                count += (y != -100).sum()
                del x, y, out
            if (step + 1) % 10 == 0 or step + 1 == steps:
                values = torch.stack((total_nll, count.double()))
                dist.all_reduce(values)
                if rank == 0:
                    progress = dict(
                        completed_batches=step + 1,
                        total_batches=steps,
                        scored_tokens=int(values[1]),
                        ce_nats_per_token=float(values[0] / values[1]),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                    write_json(args.output_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    dist.all_reduce(total_nll)
    dist.all_reduce(count)
    full = steps == all_steps
    if full and int(count) != metadata["total_tokens"]:
        raise RuntimeError("scored token count differs from official packed text tokens")
    result = dict(
        status="complete",
        full_dataset=full,
        checkpoint=path_label(args.checkpoint),
        precision_mode=precision,
        nll_nats=float(total_nll),
        scored_tokens=int(count),
        ce_nats_per_token=float(total_nll / count),
        original_utf8_bytes=metadata["total_bytes"] if full else None,
        bpb=float(total_nll) / (math.log(2) * metadata["total_bytes"]) if full else None,
        elapsed_seconds=time.perf_counter() - started,
        peak_device_memory_bytes=torch.cuda.max_memory_allocated(device),
        official_evaltime=None,
    )
    if not math.isfinite(result["ce_nats_per_token"]):
        raise RuntimeError("non-finite evaluation result")
    if rank == 0:
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--dataset-path", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--cache-dir", type=Path, required=True)
    run = modes.add_parser("evaluate")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--sequence-length", type=int, default=8192)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--loss-chunk-tokens", type=int, default=512)
    run.add_argument("--max-batches", type=int)
    run.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="bf16",
        help="bf16 preserves legacy FSDP evaluation; fp32-master-bf16 "
        "preserves FP32 checkpoint parameters with BF16 autocast",
    )
    args = parser.parse_args(argv)
    prepare(args) if args.mode == "prepare" else evaluate(args)


if __name__ == "__main__":
    main()
