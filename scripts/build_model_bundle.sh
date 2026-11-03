#!/usr/bin/env bash
# =============================================================================
# Vendster — reproducible end-to-end build of ONE model's evidence bundle.
#
# Builds the full per-model bundle for one model (Opus 4.6 by default; override MODEL_ID for
# any other). run_panel.sh calls this once per panel model; make_panel.py then assembles the
# bundles into the headline tables. This is the ONLY per-model body — there is no second copy.
#
# One command builds everything (the locked design of the paper):
#   1. forecaster quality census on the 140-pool             (offline, deterministic; skipped if present)
#   2. per-basket-5 forecast quality, same seed/sets          (offline, deterministic; skipped if present)
#   3. primary paired negotiation experiment, 26 conditions   (Bedrock, RESUMABLE)
#   4. assemble tables + figures + robustness panel + examples (offline)
#   5. optional provenance comparison against a prior bundle  (offline; only if INC is set)
#
# Frozen params below ARE the experiment identity and match the LOCKED pre-registration
# (P80, K=16, m=25, seed=1). Unattended-safe: step 3 runs in a RESUME-RETRY loop gated by
# check_complete.py, so a transient credential/network blip resumes instead of leaving a
# silently partial bundle; the offline steps retry 3x. LLM episodes are not bit-reproducible
# (Bedrock has no seed); the archived transcripts under experiment/set_*/transcripts/ are the
# provenance record.
#
#   bash scripts/build_model_bundle.sh                               # Opus 4.6, foreground
#   mkdir -p log && nohup bash scripts/build_model_bundle.sh > log/bundle.log 2>&1 &  # detached
#   MODEL_ID=us.anthropic.claude-sonnet-4-6 OUT=experiments/sonnet_holdout \
#       bash scripts/build_model_bundle.sh                           # another model / slot
#   INC=experiments/panel/opus_4_6 OUT=/tmp/opus_rebuild \
#       bash scripts/build_model_bundle.sh                           # + provenance compare
#   CONC=8 CELLCONC=8 bash scripts/build_model_bundle.sh             # throttle for a tight TPM quota
#   MODEL=mock bash scripts/build_model_bundle.sh                    # zero-spend run -> experiments/mock_bundle
#   SHARED_ONLY=1 bash scripts/build_model_bundle.sh                 # only steps 1-2 (no LLM)
#
# Steps 1-2 are model-independent and deterministic from the seed, so they are skipped when
# both outputs already exist under SHARED (REBUILD_SHARED=1 forces them). A bundle whose episodes are
# already complete skips step 3 without calling the model, and keeps its existing MANIFEST.md.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."                                   # repo root (no interactive cd)

# --- auth + model ---------------------------------------------------------------
# Any AWS profile with bedrock:InvokeModel on the target model works. We default to a
# long-lived profile so a long run can't die on token expiry (see README auth notes).
# Default model = Opus 4.6. MODEL=mock gives a zero-spend dry run of the pipeline.
export AWS_PROFILE="${AWS_PROFILE:-default}"
MODEL="${MODEL:-bedrock}"
MODEL_ID="${MODEL_ID:-us.anthropic.claude-opus-4-6-v1}"

# --- frozen parameters: the LOCKED operating point (paper, Environment Setup appendix) ---
# seed=1 is HELD OUT from the seed-0 operating-point sweep (anti-p-hacking); P80 = the
# challenging threshold (no_forecast ~= 0.30); K=16, m=25 from the power analysis.
SEED=1; SETS=16; EPISODES=25; PANEL=1500; MINWK=5
PCT=80
# Concurrency is an OPS setting (Bedrock per-model TPM quota), not a scientific parameter: episodes
# are seeded independently of execution order. 15x16 = 240 in-flight suits roomy quotas; Opus 4.6's
# tight TPM wants ~8x8 (self-play doubles its calls/turn). Throttling only slows the resume loop.
CONC="${CONC:-15}"; CELLCONC="${CELLCONC:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"                        # resume-retry budget for step 3
# Model-INDEPENDENT inputs (pool-level census + per-basket quality) live in shared/ and are reused by
# every model; this model's own episodes + assembled bundle live in OUT (default = its panel slot).
SHARED="${SHARED:-experiments/shared}"
# A mock run never defaults into a real panel slot.
if [ "$MODEL" = mock ]; then OUT="${OUT:-experiments/mock_bundle}"; else OUT="${OUT:-experiments/panel/opus_4_6}"; fi
INC="${INC:-}"                                            # optional INCumbent bundle: if set, step 5 checks this run replicates it
mkdir -p "$OUT" "$SHARED"
SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "== build @ $SHA (model=$MODEL_ID -> $OUT, shared=$SHARED, AWS_PROFILE=$AWS_PROFILE)${INC:+ compare vs $INC} =="

retry() { local i; for i in 1 2 3; do "$@" && return 0; echo "[retry $i] $*"; sleep 30; done; return 1; }

PQ="$SHARED/forecaster_quality/report.json"; BQ="$SHARED/basket_quality/quality.json"
if [ -z "${REBUILD_SHARED:-}" ] && [ -f "$PQ" ] && [ -f "$BQ" ]; then
    echo "[1/5] [2/5] shared forecast quality already present under $SHARED — reusing (REBUILD_SHARED=1 to force)."
else
    echo "[1/5] forecaster quality census on the pool (offline, shared)..."
    retry uv run python scripts/eval_quality_pool.py --min-weekly $MINWK --train-panel $PANEL \
        --out "$SHARED/forecaster_quality" || { echo "step 1 failed"; exit 1; }

    echo "[2/5] per-basket-5 forecast quality (offline, shared)..."
    retry uv run python scripts/eval_quality_baskets.py --sets $SETS --seed $SEED \
        --min-weekly $MINWK --panel-size $PANEL --out "$SHARED/basket_quality" || { echo "step 2 failed"; exit 1; }
fi
if [ -n "${SHARED_ONLY:-}" ]; then echo "== SHARED_ONLY: shared forecast quality ready under $SHARED =="; exit 0; fi

RAN_EPISODES=0
if uv run python scripts/check_complete.py "$OUT/experiment" $SETS $EPISODES >/dev/null 2>&1; then
    echo "[3/5] episodes already complete under $OUT/experiment — skipping the LLM step."
else
echo "[3/5] paired negotiation experiment (Bedrock, resume-retry loop)..."
RAN_EPISODES=1
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    uv run python scripts/run_experiment.py --model "$MODEL" --model-id "$MODEL_ID" \
        --sets $SETS --episodes $EPISODES \
        --seed $SEED --target-percentile $PCT --concurrency $CONC \
        --cell-concurrency $CELLCONC --min-weekly $MINWK --panel-size $PANEL \
        --out "$OUT/experiment" || echo "[episode attempt $attempt] non-zero exit"
    if uv run python scripts/check_complete.py "$OUT/experiment" $SETS $EPISODES; then
        echo "episodes complete after attempt $attempt"; break
    fi
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] && { echo "[episode attempt $attempt] incomplete — resuming in 120s"; sleep 120; }
done
fi
uv run python scripts/check_complete.py "$OUT/experiment" $SETS $EPISODES \
    || { echo "step 3 STILL INCOMPLETE after $MAX_ATTEMPTS attempts — aborting before analysis (re-run to resume)"; exit 1; }

echo "[4/5] assemble bundle (tables + figures + robustness panel + examples)..."
retry uv run python scripts/make_one_model.py --run "$OUT/experiment" \
    --pool-quality "$PQ" --basket-quality "$BQ" \
    --out "$OUT/paper" || { echo "step 4 failed"; exit 1; }

if [ -n "$INC" ] && [ -d "$INC/experiment" ]; then
    echo "[5/5] provenance comparison vs $INC..."
    uv run python scripts/compare_bundles.py --a "$INC/experiment" --b "$OUT/experiment" \
        --out "$OUT/paper/provenance_check.md"
else
    echo "[5/5] no INC bundle given — skipping provenance comparison."
fi

# MANIFEST is written last so its presence means the bundle completed. It records the run that
# produced the episodes, so an offline rebuild of an already-complete bundle leaves it alone.
if [ "$RAN_EPISODES" = 1 ] || [ ! -f "$OUT/MANIFEST.md" ]; then
cat > "$OUT/MANIFEST.md" <<EOF
# Vendster — evidence bundle ($MODEL_ID)

- git_sha: \`$SHA\`
- built_utc: $(date -u +%FT%TZ)
- model: $MODEL_ID (Bedrock $MODEL, temperature 0.0 where the model accepts it)  [self-play: Buyer = Vendor = this model]
- concurrency: ${CONC}x${CELLCONC} in-flight (ops setting; does not affect results)
- frozen params: seed=$SEED  sets(K)=$SETS  episodes/condition(m)=$EPISODES  target=analytic P$PCT
- pool: stable sell-price CV<5% AND min weekly units >= $MINWK (=140 products); panel=$PANEL
- conditions: 26 = 10 real forecasters + no_forecast (classical: seasonal_naive/auto_ets/auto_theta; GBM:
  lightgbm_global; foundation: chronos2/chronos_bolt/timesfm25; DL: nf_nhits/nf_patchtst/nf_deepar)
  + 15-cell perturbation grid (8-level bias +-{.05,.10,.15,.20} + unbiased + 6-level width ladder);
  ~$((SETS*26*EPISODES)) episodes
- primary confirmatory: H1c signed-bias dose-response + H2c interval-width dose-response
  (basket-clustered trend t-tests); H0 gate (every real forecaster + oracle > no_forecast);
  observational H1o/H2o/H3o (real forecasters' measured bias/width/accuracy -> attainment) =
  supporting (accuracy null, robust across the 10 real forecasters — see paper/robustness_metrics.md).
- provenance:${INC:+ compared against $INC in paper/provenance_check.md (expected: equivalent within Monte-Carlo noise);} completeness gated by check_complete.py ($SETS x 26 x $EPISODES).
- reproducibility: pool census, per-basket quality, and analytic P$PCT targets are deterministic
  from (git_sha, seed); LLM negotiation episodes are seeded for the DGP but Bedrock is not
  seedable -> the archived turn-by-turn transcripts are the record. Re-run this script to resume.
EOF
fi
echo "== DONE -> $OUT (paper/tables.md, paper/figures/, paper/examples.md, paper/robustness_metrics.md, MANIFEST.md${INC:+, paper/provenance_check.md}) =="
