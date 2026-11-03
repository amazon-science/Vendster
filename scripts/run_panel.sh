#!/usr/bin/env bash
# =============================================================================
# The cross-model self-play panel: one full per-model bundle for each of the four panel models
# over the same seed=1 grid (26 conditions x K=16 x m=25), so every model negotiates the same
# baskets/targets/forecasts and only the negotiator differs. make_panel.py then assembles the
# bundles into the headline tables.
#
# This is a thin loop over build_model_bundle.sh (the single per-model body). Models run
# sequentially: the neural forecasters are a GPU singleton, so two live runs would contend/OOM.
# The first model builds experiments/shared/ (model-independent forecast quality); the rest reuse
# it. Already-complete bundles resume as a no-op (valid-episode-aware).
#
#   nohup bash scripts/run_panel.sh > log/run_panel.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export AWS_PROFILE="${AWS_PROFILE:-default}"

# name|model_id|concurrency|cell-concurrency   ('|' separator: model ids contain ':').
# Concurrency is an ops setting for each model's Bedrock TPM quota (see build_model_bundle.sh);
# it does not affect results. Opus 4.6 has a tight quota and makes 2x calls/turn in self-play.
MODELS=(
  "opus_4_6|us.anthropic.claude-opus-4-6-v1|8|8"
  "gpt_5_6_sol|us.openai.gpt-5.6-sol|15|16"
  "qwen3_235b|qwen.qwen3-235b-a22b-2507-v1:0|15|16"
  "deepseek_v3_2|deepseek.v3.2|15|16"
)
failed=()
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name mid conc cell <<< "$spec"
  echo "==== $(date -u +%FT%TZ)  $name  ($mid, ${conc}x${cell} in-flight) ===="
  MODEL_ID="$mid" OUT="experiments/panel/$name" CONC="$conc" CELLCONC="$cell" MAX_ATTEMPTS=8 \
    bash scripts/build_model_bundle.sh || { echo "[$name] bundle FAILED — continuing with the next model"; failed+=("$name"); }
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "==== panel finished with incomplete bundles: ${failed[*]} (re-run to resume) ===="; exit 1
fi
echo "==== panel complete $(date -u +%FT%TZ) ===="
