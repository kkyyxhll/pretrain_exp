#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
: "${DATA_DIR:?Set DATA_DIR to the directory containing all eight compressed training shards}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR to the pinned local Qwen3.5 tokenizer}"
: "${RUN_DIR:?Set RUN_DIR to a fresh output directory}"
budget=${1:?Usage: bash recipes/qwen35-08b/train_qwen35_dp2.sh BUDGET [extra training args]}
shift
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Calibrated single-node container route; override only after a passing P2P check.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
exec python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  recipes/qwen35-08b/train.py \
  --dp-world-size 2 --budget "$budget" \
  --data-dir "$DATA_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --output "$RUN_DIR" "$@"
