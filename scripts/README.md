# `scripts/` — entrypoints & tools

Runnable scripts for the Vendster benchmark (18 `.py`, 4 `.sh`). Each script's module docstring or
header has full usage; run with `uv run python scripts/<name>.py --help` where applicable. Every
script `cd`s / `sys.path`s to the repo root, imports from `vendster/`, and writes under
`experiments/` (tracked) unless told otherwise.

**Naming:** `run_*` spends LLM calls to produce negotiation episodes (needs Bedrock unless
`--model mock`); `make_*` reads archived episodes offline and assembles tables, figures, or
documents from them. Everything else is named by function (`eval_quality_*` scores forecasters
on M5 with no episodes involved; `check_`/`compare_`/`show_` are utilities).

**Reproduce the whole paper in one command:** `bash scripts/reproduce.sh`. **Confirm the paper's
numbers without a rerun:** `uv run python scripts/paper_stats.py --verify overleaf/main.tex`
recomputes every cell of the five paper tables and diffs them against the manuscript at displayed
precision, exiting non-zero on any mismatch (currently **359/359 cells match**).

## How the pieces fit

```
reproduce.sh                                            
 ├─ [1] power_analysis.py         offline         a-priori sizing -> experiments/power/power.json
 ├─ [2] run_panel.sh              Bedrock + GPU   4 panel bundles -> experiments/panel/<model>/
 │       └─ per model: build_model_bundle.sh  (the paper's four models, with their quota settings)
 ├─ [3] run_role_cross.sh         Bedrock         16 Buyer×Vendor cells -> experiments/role_cross/
 │       └─ per cell:  run_experiment.py --conditions bias ⟲ check_complete.py
 ├─ [4] make_panel.py, make_role_cross.py
 ├─ [5] paper_stats.py            -> experiments/audit/audit.json
 └─ [6] make_transcripts.py --tex, cp 3 figures -> overleaf/images/, paper_stats.py --verify

build_model_bundle.sh             Bedrock + GPU   ONE model's bundle — the general per-model body
 └─ [eval_quality_pool.py -> eval_quality_baskets.py, skipped if experiments/shared/ exists]
    -> run_experiment.py ⟲ check_complete.py -> make_one_model.py
    -> compare_bundles.py  (only when INC=<prior bundle> is given: a replication check against it)
```

**General tool vs. paper instance.** `build_model_bundle.sh` is the general per-model pipeline:
point it at any Bedrock model id and output slot and it produces a complete, comparable bundle at
the locked design. `run_panel.sh` is the paper's specific instantiation of it — the four evaluated
models, their output slots under `experiments/panel/`, and the per-model concurrency each one's
quota allows. To evaluate a fifth model, call `build_model_bundle.sh` directly (see below) and add
it to `make_panel.py --models`; nothing else changes.

`reproduce.sh` **reuses** two archived inputs rather than regenerating them: the Stage-1
operating-point sweep under
`experiments/arena_sweep/` (`run_percentile_sweep.py` + `make_operating_point.py`, Sonnet 4.6,
seed 0), and the held-out Sonnet bundle under `experiments/sonnet_holdout/`. Everything else it
produces from scratch; in a fresh checkout the first panel model builds the model-independent
`experiments/shared/` quality files and the rest reuse them.

**Layout rule under `experiments/`:** one directory per study; its runs live in subdirectories
and its aggregate analysis sits at the top level of the same directory.

```
panel/        <model>/{experiment/, paper/, MANIFEST.md}   + panel.{md,json}, figures/panel_*.png  (+ example_transcripts.md from your own runs)
role_cross/   b_<buyer>__v_<vendor>/experiment/           + role_cross.{md,json}, figures/role_cross_heatmap.png
arena_sweep/  P<xx>/<cond>/set_*/                          + operating_point.{md,json}, figures/operating_point.png
```

`shared/` (model-independent forecast quality), `power/`, and `audit/` are single-purpose outputs
with no runs underneath. The paper's figures live where `main.tex` reads them, `overleaf/images/`;
`reproduce.sh` refreshes them from the three study directories.

## Orchestrators (`.sh`)

| script | what it does | needs |
|---|---|---|
| `reproduce.sh` | **Paper front door**, in the order the study was run. A-priori power → panel → role-cross → cross-model tables + reactivity fingerprint → hypothesis stats + Holm + MDE → curated transcripts (+ `overleaf/appendix_transcripts.tex`) → refresh the three paper figures in `overleaf/images/` → `paper_stats --verify`. Steps 2–3 call Bedrock and are resumable (a complete bundle is a no-op); the rest are offline. | `AWS_PROFILE`, `MAXJOBS` (via children) |
| `run_panel.sh` | **The paper's panel**: a thin loop that calls `build_model_bundle.sh` once per evaluated model (Opus 4.6 / GPT-5.6 / Qwen3-235B / DeepSeek-V3.2) into `experiments/panel/<name>/`, passing each model's concurrency (an ops setting for its Bedrock quota; Opus runs 8×8, the others 15×16). Sequential, because the neural forecasters are a GPU singleton. A complete bundle is a no-op; exits non-zero if any model is still incomplete. | `AWS_PROFILE` |
| `run_role_cross.sh` | The **4×4 Buyer×Vendor role-cross** over the same four models, bias-ladder conditions only (analytic forecaster, so GPU-free). Cells run as a `MAXJOBS`-wide job pool with per-cell resume-retry; Opus-involving cells throttle to respect its TPM quota. Backs `fig:rolecross`. | `AWS_PROFILE`, `MAXJOBS` (default 6) |
| `build_model_bundle.sh` | **The general per-model pipeline** (used by `run_panel.sh`; also how to evaluate any model of your own). At the locked design: shared pool-level + per-basket forecast quality (→ `experiments/shared/`, skipped if already present), the paired experiment in a resume-retry loop with a hard completeness gate, `make_one_model`, and, only when `INC=<path to a prior bundle>` is set, a `compare_bundles.py` replication check against that bundle (`paper/provenance_check.md`). `MANIFEST.md` is written last, so its presence means the bundle completed. Defaults to Opus 4.6 into `experiments/panel/opus_4_6`; `MODEL_ID`/`OUT` pick another model or slot (e.g. the held-out Sonnet bundle), `CONC`/`CELLCONC` throttle for a tight quota, `MODEL=mock` dry-runs with zero AWS spend. | `AWS_PROFILE`, `MODEL`, `MODEL_ID`, `OUT`, `SHARED`, `INC`, `CONC`, `CELLCONC`, `MAX_ATTEMPTS`, `REBUILD_SHARED` |

The locked design — K=16 baskets, m=25 episodes per cell, P80 target, seed=1, panel 1500, min 5
units/week — is fixed as shell constants in `build_model_bundle.sh` and `run_role_cross.sh`
(`run_panel.sh` inherits them; see the Environment Setup appendix of `overleaf/main.tex`).

## Experiment stages (`.py`, called by the orchestrators; runnable standalone)

| script | what it does | needs |
|---|---|---|
| `run_experiment.py` | **The primary paired experiment driver.** Samples K baskets, sets each basket's analytic P-percentile target, runs every condition paired per basket with episode + cell concurrency, archives per-episode transcripts. `--conditions {all,bias,grid}` selects the roster subset; `--vendor-model-id` enables role-cross; `--model mock` runs the whole pipeline with zero AWS spend. | Bedrock (unless `mock`); GPU for `all` |
| `eval_quality_pool.py` | Forecaster-quality census over the 140-product pool (single-origin Chronos/M5 protocol, full metric suite vs seasonal naive; `tab:quality`) → `experiments/shared/forecaster_quality/report.json`. Deterministic. | offline (GPU helps) |
| `eval_quality_baskets.py` | Per-basket-5 forecast quality: re-derives the same seeded baskets a paired run used and scores each basket's actual products → `experiments/shared/basket_quality/quality.json`. Feeds `tab:robust` and the basket-level quality↔outcome join. | offline (GPU helps) |
| `check_complete.py` | `run_dir n_sets m [all\|bias\|grid]` — exit 0 iff every (condition, basket) cell has ≥ m valid episodes. The completeness gate every resume-retry loop uses. | offline |

## Analysis & assembly (`.py`, offline, deterministic)

| script | what it does |
|---|---|
| `make_one_model.py` | Assemble **one** model's bundle from a run dir → `<bundle>/paper/`: outcome tables (`tables.md`), causal + observational trend tests (`aggregate.json`), the quality↔outcome link (`link.json`), figures, example transcripts, and the **robustness panel** (`robustness_metrics.{md,json}`): does a real forecaster's per-basket error property predict its attainment, for six accuracy metrics (relWQL/MASE/MAPE/sMAPE/RMSE/CRPS) and three calibration properties (signed bias, interval width, coverage), each under the design-correct basket-clustered test, a naive pooled OLS (the clustering trap), and a forecaster-level Pearson. |
| `make_panel.py` | The **multi-model headline** from several per-model bundles: Table A (H0/H1c/H2c/H1o/H3o per model), B (the reactivity fingerprint: bias-ladder behavioral slopes, `tab:reactivity`), C (real-forecaster attainment by model), D (accuracy slope by metric × model), welfare stats, and five figures (bias ladder, quality→outcome scatter, perturbation grid, agreed-only mechanism, welfare) → `experiments/panel/panel.{md,json}` + `figures/panel_*.png`. Only the perturbation grid is in the paper (`fig:grid`). |
| `make_role_cross.py` | Role-cross analysis: per-cell bias slope, first-offer slope, and agreement rate rendered as a Buyer(row)×Vendor(col) heatmap (row structure ⇒ buyer-driven) → `experiments/role_cross/role_cross.{md,json}` + `figures/role_cross_heatmap.png` (`fig:rolecross`). |
| `paper_stats.py` | **The paper's hypothesis table**: per-hypothesis effect, raw + Holm-across-4-LLMs p, and realized-precision MDE for H0/H1c/H2c/H1o/H2o/H3o → `experiments/audit/audit.json`, cross-checked against `panel.json`. With `--verify overleaf/main.tex` it recomputes every cell of all five paper tables and diffs them against the manuscript. |
| `power_analysis.py` | Size (K, m) by simulation **and** report a-priori power / MDE for every hypothesis against a single 0.10-attainment MDE → `experiments/power/power.json` (`tab:power`). |
| `make_transcripts.py` | The **curated, model-diverse** appendix transcripts (one representative episode per model × situation) → `experiments/panel/example_transcripts.md`; `--tex` also regenerates `overleaf/appendix_transcripts.tex` verbatim. |
| `compare_bundles.py` | Condition-by-condition attainment comparison between two runs with a 2·SE Monte-Carlo band — the "is this re-run a replication?" check `build_model_bundle.sh` emits when `INC` is set. Pure stdlib. |

## Inspection tools (`.py`, offline)

| script | what it does |
|---|---|
| `make_reader.py` | Static HTML reader over the transcripts of your own panel runs (the release ships none) → `experiments/reader.html` (per-episode index embedded; each full log fetched on click; filter by model/condition/outcome). Must be **served** (`uv run python -m http.server -d experiments 8000`) or hosted on GitHub Pages, not opened as `file://`. |
| `show_episode.py` | Render one archived episode (or one per condition) from a run dir as readable markdown. |

## Design-stage scripts (one-off; each produced or justifies a locked decision)

These are not on the `reproduce.sh` path. Their outputs are tracked under `experiments/` and reused.

| script | decision it supports | needs |
|---|---|---|
| `run_percentile_sweep.py` | **The effect-blind difficulty sweep** that fixed the operating point (`no_forecast` vs `unbiased` at P50…P90, Sonnet 4.6, seed 0) that selected P80 → `experiments/arena_sweep/P<xx>/`. | Bedrock |
| `make_operating_point.py` | Aggregates the sweep into the P80-justification table + figure → `experiments/arena_sweep/operating_point.{md,json}` + `figures/operating_point.png` (`fig:operating`). | offline |
| `run_condition.py` | Run **one** condition on K baskets × m episodes at one target percentile; pooled attainment + Wilson 95% CI. A quick single-cell probe; the sweep above is its multi-percentile form. | Bedrock |
| `validate_target_calibration.py` | Confirm the analytic per-basket target sits the `no_forecast` Buyer at the intended operating point (pooled attainment + Wilson CI + binomial test) → `calibration_report.md`. Diagnostic; `run_experiment.py --validate-xx N` runs the same check inline. | Bedrock |
| `probe_throughput.py` | Measure the Bedrock throughput ceiling on realistic episode prompts to size `--concurrency`. Ops sizing only. | Bedrock |

See the repo `README.md` for the quick start and authentication notes, and `EXTENDING.md` for adding
forecasters / scenarios / data sources.
