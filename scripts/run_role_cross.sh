#!/usr/bin/env bash
# =============================================================================
# 4x4 Buyer x Vendor role-cross matrix — {opus, gpt, qwen, deepseek} (Sonnet is the held-out
# design model, so it is not in the matrix). Question: does the forecast-bias effect track the
# BUYER (row) rather than the Vendor (column)? Backs fig:rolecross (paper appendix, via
# make_role_cross.py).
#
#   - Bias-ladder scope (--conditions bias): GPU-FREE (perturbed forecaster is analytic).
#   - All models via the Bedrock Converse API on the long-lived `default` profile (non-expiring).
#   - seed=1 -> same baskets/targets/forecasts as everything (paired).
#   - Job pool of MAXJOBS concurrent cells. Host memory is usually the limiter, not quota (each
#     cell takes ~3-4GB). Opus-involving cells run at lower in-flight to respect Opus's 3M TPM.
#   - Exits non-zero if any cell is still incomplete after its retries.
#   - Per-cell resume-retry (valid-episode-aware) backfills any throttle losses.
#
#   MAXJOBS=6 nohup bash scripts/run_role_cross.sh > log/run_role_cross.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export AWS_PROFILE="${AWS_PROFILE:-default}"
mkdir -p log

SETS=16; EP=25; PCT=80; SEED=1; PANEL=1500; MINWK=5
MAXJOBS=${MAXJOBS:-6}
MODELS=(opus gpt qwen deepseek)
mid() { case "$1" in
  opus) echo us.anthropic.claude-opus-4-6-v1;;
  gpt) echo us.openai.gpt-5.6-sol;;
  qwen) echo qwen.qwen3-235b-a22b-2507-v1:0;;
  deepseek) echo deepseek.v3.2;; esac; }

run_cell() {
  local b="$1" v="$2" buyer vendor out conc
  buyer="$(mid "$b")"; vendor="$(mid "$v")"
  out="experiments/role_cross/b_${b}__v_${v}"
  conc=8; { [ "$b" = opus ] || [ "$v" = opus ]; } && conc=4   # throttle Opus cells (3M TPM)
  for attempt in 1 2 3 4 5 6; do
    if uv run python scripts/check_complete.py "$out/experiment" $SETS $EP bias >/dev/null 2>&1; then
      echo "$(date -u +%FT%TZ)  DONE  b=$b v=$v"; return 0
    fi
    uv run python scripts/run_experiment.py --model bedrock \
      --model-id "$buyer" --vendor-model-id "$vendor" --conditions bias \
      --sets $SETS --episodes $EP --target-percentile $PCT --seed $SEED \
      --concurrency $conc --cell-concurrency 10 --min-weekly $MINWK --panel-size $PANEL \
      --out "$out/experiment" >> "log/rc_b_${b}__v_${v}.log" 2>&1 || true
    sleep 60
  done
  if uv run python scripts/check_complete.py "$out/experiment" $SETS $EP bias >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ)  DONE  b=$b v=$v"; return 0
  fi
  echo "$(date -u +%FT%TZ)  INCOMPLETE  b=$b v=$v (gave up after 6 attempts)"
  return 1
}

echo "matrix start $(date -u +%FT%TZ)  MAXJOBS=$MAXJOBS"
pids=()
for b in "${MODELS[@]}"; do
  for v in "${MODELS[@]}"; do
    run_cell "$b" "$v" &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
  done
done
incomplete=0
for pid in "${pids[@]}"; do wait "$pid" || incomplete=$((incomplete + 1)); done
echo "matrix done $(date -u +%FT%TZ)"
if [ "$incomplete" -gt 0 ]; then echo "$incomplete cell(s) INCOMPLETE (re-run to resume)"; exit 1; fi
