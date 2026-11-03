#!/usr/bin/env bash
# =============================================================================
# End-to-end reproduction of the Vendster paper, in the order the study was actually run:
# a-priori power analysis, then the panel and the role-cross (Bedrock), then all offline
# analysis, statistics, transcripts, paper figures, and the final verify against main.tex.
#
#   bash scripts/reproduce.sh
#
# The two Bedrock steps ([2],[3]) are the long ones and skip completed
# episodes on resume; [0], [1] and [4]-[6] are offline. Locked params
# (seed=1, K=16, m=25, P80) live in the drivers those steps call. Any failing
# step stops the run.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
say() { echo -e "\n=== $* ==="; }

PANELS="opus:experiments/panel/opus_4_6 gpt:experiments/panel/gpt_5_6_sol \
        qwen:experiments/panel/qwen3_235b deepseek:experiments/panel/deepseek_v3_2"

say "[0/6] shared forecast-quality census + per-basket quality (offline; skipped if present)"
SHARED_ONLY=1 bash scripts/build_model_bundle.sh

# A-priori: checks K=16 / m=25 against the shared per-basket quality + the seed-0 sweep baselines only.
# It reads nothing from the panel, and chronologically it was fixed before the panel was run.
say "[1/6] a-priori power / MDE against the 0.10-attainment target (pre-data; no panel inputs)"
uv run python scripts/power_analysis.py --out experiments/power
say "[2/6] cross-model self-play panel (Bedrock; resumable)"
bash scripts/run_panel.sh

say "[3/6] 4x4 Buyer x Vendor role-cross (Bedrock; resumable)"
bash scripts/run_role_cross.sh

# The reactivity fingerprint (behavioral slopes, tab:reactivity) is computed here into panel.json.
say "[4/6] cross-model tables + reactivity fingerprint + figures; role-cross heatmap"
uv run python scripts/make_panel.py --models $PANELS
uv run python scripts/make_role_cross.py --root experiments/role_cross --out experiments/role_cross

say "[5/6] hypothesis tests + Holm + MDE (tab:crossmodel / tab:robust / tab:reactivity)"
uv run python scripts/paper_stats.py --out experiments/audit


say "[6/6] curated transcripts (+ appendix_transcripts.tex), refresh the paper's figures, then verify the paper"
uv run python scripts/make_transcripts.py --tex overleaf/appendix_transcripts.tex
# The three figures main.tex includes (\graphicspath{{images/}}), copied under the names it uses.
cp experiments/panel/figures/panel_perturbation_grid.png     overleaf/images/perturbation_grid.png
cp experiments/role_cross/figures/role_cross_heatmap.png     overleaf/images/role_cross_heatmap.png
cp experiments/arena_sweep/figures/operating_point.png       overleaf/images/arena_operating_point.png
# Single gate: every table cell in the paper must be reproduced by the scripts.
uv run python scripts/paper_stats.py --out experiments/audit --verify overleaf/main.tex
echo -e "\nDONE. Stats: experiments/audit/audit.json | power: experiments/power/power.json"
echo    "     paper figures: overleaf/images/ | appendix transcripts: overleaf/appendix_transcripts.tex"
