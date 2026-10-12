#!/usr/bin/env python3
"""Single-node, 8-GPU Transformers evaluation: JSONL -> packing -> bpb/EvalTime.

BF16 and SDPA are example choices; adapt them for the submitted model.
Warms each batch shape five times before one complete evaluation.
EvalTime counts only forward (including the vocabulary projection), in
s / (1B non-vocabulary active parameters * 1M scored tokens).
"""

import argparse
import json
import math
import os
import tempfile
import time
from itertools import islice
from pathlib import Path

import numpy as np


def pack_jsonl(dataset_path, tokenizer, cache_dir):
    """Write a flat token stream and a mask distinguishing text from inserted EOS."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("The tokenizer must define eos_token_id for document separation")
    total_bytes = total_tokens = 0
    with (
        dataset_path.open(encoding="utf-8") as source,
        (cache_dir / "tokens.bin").open("wb") as token_file,
        (cache_dir / "is_text.bin").open("wb") as mask_file,
    ):
        records = (json.loads(line)["text"] for line in source if line.strip())
        while texts := list(islice(records, 128)):
            total_bytes += sum(len(text.encode("utf-8")) for text in texts)
            encoded = tokenizer(
                texts, add_special_tokens=False, truncation=False,
                return_attention_mask=False, return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
            for text, ids in zip(texts, encoded):
                if text and not ids:
                    raise ValueError("Tokenizer returned no tokens for nonempty text")
                np.asarray([eos, *ids], dtype=np.int32).tofile(token_file)
                mask = np.ones(len(ids) + 1, dtype=np.bool_)
                mask[0] = False
                mask.tofile(mask_file)
                total_tokens += len(ids)
    if not total_bytes or not total_tokens:
        raise ValueError("The dataset must contain nonempty text")
    return dict(
        total_bytes=total_bytes, total_tokens=total_tokens,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
    )


def packed_batch(tokens, is_text, start, stop, sequence_length, pad_token_id):
    """Return input/target arrays; the first target is the first real text token."""
    inputs = np.full((stop - start, sequence_length), pad_token_id, dtype=np.int64)
    labels = np.full_like(inputs, -100)
    for row, sequence_index in enumerate(range(start, stop)):
        offset = sequence_index * sequence_length
        length = min(sequence_length, len(tokens) - 1 - offset)
        inputs[row, :length] = tokens[offset:offset + length]
        target_slice = slice(offset + 1, offset + length + 1)
        labels[row, :length] = np.where(is_text[target_slice], tokens[target_slice], -100)
    return inputs, labels


def active_parameter_count(model):
    """Count dense N_a, excluding input/output vocabulary parameters.

    For MoE or dynamic depth, adapt the evaluator to collect actual routing
    statistics outside timed forward calls. Average over scored evaluation tokens,
    counting each active non-vocabulary parameter at most once per token.
    """
    if any(getattr(model.config, key, 0) > 1 for key in (
        "num_experts", "num_local_experts", "n_routed_experts"
    ) if getattr(model.config, key, None) is not None):
        raise NotImplementedError("Implement active-parameter counting for this model's routing")
    excluded = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            excluded.update(id(p) for p in module.parameters())
    return sum(p.numel() for p in model.parameters() if id(p) not in excluded)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True, help="JSONL with a text field")
    parser.add_argument("--model-path", required=True, help="Local Transformers checkpoint")
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences per GPU")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    if not 1 <= args.sequence_length <= 8192:
        parser.error("--sequence-length must be between 1 and 8192")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output path")
    return args


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    set_seed(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Only rank 0 tokenizes. Other ranks open the same temporary files read-only.
    temporary = None
    shared = [None]
    if rank == 0:
        temporary = tempfile.TemporaryDirectory(prefix=".eval-", dir=args.output.parent)
        cache_dir = Path(temporary.name).resolve()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        metadata = pack_jsonl(args.dataset_path, tokenizer, cache_dir)
        shared[0] = (str(cache_dir), metadata)
        print(f"Packed {metadata['total_tokens']} text tokens", flush=True)
    dist.broadcast_object_list(shared, src=0, device=device)
    cache_path, metadata = shared[0]
    tokens = np.memmap(Path(cache_path) / "tokens.bin", dtype=np.int32, mode="r")
    is_text = np.memmap(Path(cache_path) / "is_text.bin", dtype=np.bool_, mode="r")
    num_sequences = math.ceil((len(tokens) - 1) / args.sequence_length)
    global_batch_size = args.batch_size * 8
    num_steps = math.ceil(num_sequences / global_batch_size)

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    config = config.get_text_config()
    # Example precision and attention implementation; teams may change both.
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint did not fully initialize the language model: {loading_info}")
    model.to(device).eval()
    active_params = active_parameter_count(model)
    if active_params <= 0:
        raise ValueError("Non-vocabulary active parameter count must be positive")

    def prepare_batch(step):
        start = step * global_batch_size + rank * args.batch_size
        stop = min(start + args.batch_size, num_sequences)
        if start >= stop:
            return None
        inputs, labels = packed_batch(
            tokens, is_text, start, stop, args.sequence_length, metadata["pad_token_id"]
        )
        return (torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device),
                int(np.count_nonzero(labels != -100)))

    def forward_only(inputs):
        # Right padding follows every valid target, so it cannot affect causal predictions.
        # Fresh state on every call; no labels means model.forward does not compute loss.
        return model(input_ids=inputs, use_cache=False, return_dict=True).logits

    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    scored_tokens = 0
    forward_seconds = 0.0
    with torch.inference_mode():
        # Warm both the full batch and any smaller final batch on this rank.
        last_step = (num_sequences - 1 - rank * args.batch_size) // global_batch_size
        warmed_shapes = set()
        for step in (0, last_step):
            if step < 0:
                continue
            batch = prepare_batch(step)
            if batch is None or tuple(batch[0].shape) in warmed_shapes:
                continue
            warmed_shapes.add(tuple(batch[0].shape))
            for _ in range(5):
                logits = forward_only(batch[0])
                del logits
            torch.cuda.synchronize(device)
        dist.barrier()

        set_seed(0)
        for step in range(num_steps):
            batch = prepare_batch(step)
            elapsed = 0.0
            if batch is not None:
                inputs, labels, count = batch
                torch.cuda.synchronize(device)  # H2D transfer and previous work are excluded.
                start_time = time.perf_counter()
                logits = forward_only(inputs)
                torch.cuda.synchronize(device)  # Wait for all kernels, including the LM head.
                elapsed = time.perf_counter() - start_time

            step_time = torch.tensor(elapsed, device=device, dtype=torch.float64)
            # Replicas work concurrently: sum the per-batch maximum, not per-GPU time.
            dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
            forward_seconds += step_time.item()

            if batch is not None:
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_labels = labels.reshape(-1)
                # Chunk FP32 cross-entropy to limit memory use, outside the timer.
                for offset in range(0, flat_labels.numel(), 512):
                    loss = F.cross_entropy(
                        flat_logits[offset:offset + 512].float(),
                        flat_labels[offset:offset + 512],
                        ignore_index=-100, reduction="none",
                    )
                    total_nll += loss.double().sum()
                scored_tokens += count
                del flat_logits, flat_labels, loss, logits
        if rank == 0:
            print(f"Forward time: {forward_seconds:.3f}s", flush=True)

    count_tensor = torch.tensor(scored_tokens, device=device, dtype=torch.int64)
    dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    if count_tensor.item() != metadata["total_tokens"]:
        raise RuntimeError("Scored token count does not match the original text token count")
    bpb = total_nll.item() / (math.log(2) * metadata["total_bytes"])
    evaltime = forward_seconds / ((active_params / 1e9) * (metadata["total_tokens"] / 1e6))
    if not math.isfinite(bpb) or not math.isfinite(evaltime) or evaltime <= 0:
        raise RuntimeError("Evaluation produced a non-finite loss or invalid forward time")
    if rank == 0:
        result = dict(model_path=args.model_path, bpb=bpb, evaltime=evaltime)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    del tokens, is_text
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
