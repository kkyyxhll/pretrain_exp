#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$recipe_dir/../.." && pwd)
: "${ENGRAM_TOKEN_MAP:?Set ENGRAM_TOKEN_MAP to the prepared runtime token-map.npy}"
budget=${1:?Usage: bash recipes/qwen35-engram/train_qwen35_engram_dp2.sh BUDGET [extra args]}
shift
exec bash "$repo_dir/recipes/qwen35-08b/train_qwen35_dp2.sh" "$budget" \
  --engram-config "$recipe_dir/config.json" --engram-token-map "$ENGRAM_TOKEN_MAP" "$@"
