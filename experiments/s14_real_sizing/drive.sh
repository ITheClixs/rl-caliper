#!/bin/bash
# Six recipes on one corpus at a matched rollout budget of 64 per update, so what is being compared
# is how the budget is split and how far each step goes, not how much is spent. Each recipe is one
# anchor run, which produces the forecast, followed by the validation seeds.
#
# Records land in runs/s11_prospective, which is where that experiment writes; the store argument
# only redirects the frozen artefacts. Recipes already present are skipped, so the script resumes.
set -u
cd "$(dirname "$0")/../.."
CORPUS=${CORPUS:-runs/s12_support_sweep/band-0.85.json}
SEEDS=${SEEDS:-16}
STEPS=${STEPS:-12}
FROZEN=${FROZEN:-runs/s14_real_sizing}
mkdir -p "$FROZEN"
for split in "16 4" "8 8"; do
  set -- $split
  P=$1; G=$2
  for LR in 2.5e-4 5e-4 1e-3; do
    LABEL="sizing-p${P}g${G}-lr${LR}"
    if grep -lq "\"label\": \"$LABEL\"" runs/s11_prospective/*.json 2>/dev/null; then
      echo "skip $LABEL, already present"
      continue
    fi
    echo "=== $LABEL  $(date +%H:%M) ==="
    uv run python experiments/s11_prospective/run.py \
      --label "$LABEL" --prompts-file "$CORPUS" --store "$FROZEN" \
      --prompts "$P" --group-size "$G" --learning-rate "$LR" \
      --seeds "$SEEDS" --steps "$STEPS" --threshold 8.0 || echo "FAILED $LABEL"
  done
done
echo "=== all recipes done $(date +%H:%M) ==="
