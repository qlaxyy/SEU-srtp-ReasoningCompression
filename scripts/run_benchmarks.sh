#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-1p5b}"
DATASET="${DATASET:-math500}"
METHODS="${METHODS:-unsteered rebalance dynamic32 full}"
SEED="${SEED:-42}"
ACTION="${ACTION:---dry-run}"
if [[ "$ACTION" != '--dry-run' && "$ACTION" != '--execute' ]]; then
  echo 'ACTION must be --dry-run or --execute' >&2; exit 2
fi
for method in $METHODS; do
  args=(--model "$MODEL" --dataset "$DATASET" --method "$method" --seed "$SEED" "$ACTION")
  if [[ "$ACTION" == '--execute' ]]; then
    : "${MODEL_PATH:?Set MODEL_PATH to a local DeepSeek model snapshot}"
    args+=(--input "data/$DATASET/questions.json" --model-path "$MODEL_PATH" --output "outputs/${MODEL}_${DATASET}_${method}_s${SEED}")
  fi
  python -m reasoning_compression.cli "${args[@]}"
done
