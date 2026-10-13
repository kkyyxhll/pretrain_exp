"""Build an Engram vocabulary projection from the pinned local Qwen tokenizer."""

from __future__ import annotations

import third_party as third_party  # isort: skip

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.arch.memory import build_token_map

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    options = parser.parse_args()
    runtime_directory = options.runtime_dir.resolve()
    if runtime_directory.is_relative_to(ROOT):
        parser.error("Engram vocabulary assets belong in RUNTIME_DIR outside the repository")
    elif not options.tokenizer_dir.is_dir():
        parser.error("Provide the existing pinned local tokenizer directory")
    output_path = runtime_directory / "token-map.npy"
    identity_path = runtime_directory / "token-map.json"
    if output_path.exists() or identity_path.exists():
        parser.error("Vocabulary preparation requires fresh output files")

    started_seconds = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        str(options.tokenizer_dir),
        revision=TOKENIZER_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Engram preparation requires the existing fast tokenizer")
    token_map = build_token_map(tokenizer)
    runtime_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output_file:
        np.save(output_file, token_map.numpy(), allow_pickle=False)
    tokenizer_hashes = {
        str(asset.relative_to(options.tokenizer_dir)): hashlib.sha256(
            asset.read_bytes()
        ).hexdigest()
        for asset in sorted(options.tokenizer_dir.rglob("*"))
        if asset.is_file() and ".cache" not in asset.parts
    }
    identity = {
        "token_map_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "original_vocabulary_size": token_map.numel(),
        "canonical_vocabulary_size": int(token_map.max()) + 1,
        "tokenizer_backend_sha256": hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizer_revision_requested": TOKENIZER_REVISION,
        "tokenizer_files_sha256": tokenizer_hashes,
        "tokenizer_identity_basis": "local asset hashes; owner supplies pinned revision provenance",
        "normalizer": "official Engram demo vocabulary normalization",
        "training_corpus_read": False,
        "model_weights_loaded": False,
        "preparation_seconds": time.monotonic() - started_seconds,
    }
    with identity_path.open("x") as identity_file:
        json.dump(identity, identity_file, ensure_ascii=False, indent=2)
        identity_file.write("\n")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
