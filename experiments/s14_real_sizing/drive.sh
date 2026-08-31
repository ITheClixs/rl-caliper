#!/bin/bash
# Six recipes on one corpus, at a matched rollout budget of 64 per update, so the comparison is
# between splits and step sizes rather than between spending levels. Each is one anchor run, which
# produces the forecast, and then the validation seeds.
set -u
cd "$(dirname "$0")/../.."
CORPUS=${CORPUS:-runs/s12_support_sweep/band-0.85.json}
SEEDS=${SEEDS:-16}
STEPS=${STEPS:-12}
STORE=${STORE:-runs/s14_real_sizing}
for split in "16 4" "8 8"; do
  set -- $split
  P=$1; G=$2
  for LR in 2.5e-4 5e-4 1e-3; do
    LABEL="p${P}g${G}-lr${LR}"
    if ls "$STORE"/*.json >/dev/null 2>&1 && grep -lq "\"label\": \"$LABEL\"" "$STORE"/*.json; then
      echo "skip $LABEL, already present"
      continue
    fi
    echo "=== $LABEL ==="
    uv run python experiments/s11_prospective/run.py \
      --label "$LABEL" --prompts-file "$CORPUS" --store "$STORE" \
      --prompts "$P" --group-size "$G" --learning-rate "$LR" \
      --seeds "$SEEDS" --steps "$STEPS" --threshold 8.0 || echo "FAILED $LABEL"
  done
done
