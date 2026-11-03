# Vendster

Vendster is a reusable buyer–vendor negotiation-arena benchmark for measuring how
**demand-forecast quality and bias affect LLM agent decisions**. In each episode a
Buyer (who has a sales target and privately sees a demand forecast) negotiates
promotional discounts against a Vendor (who maximizes portfolio profit and privately
knows unit margins) over a basket of products. By varying only the forecast the Buyer
sees — real time-series models spanning the SOTA quality/bias landscape, plus a
controlled bias/interval-width perturbation grid — and holding the negotiation arena,
demand data-generating process (DGP), and action space fixed, Vendster isolates the
causal effect of forecast information on negotiation outcomes. The reference
instance (**Vendster-M5**) is grounded in real M5 retail demand data.

The benchmark is run across a **co-equal panel of frontier LLMs** — Claude Opus 4.6,
OpenAI GPT-5.6, Qwen3-235B, and DeepSeek-V3.2 — in self-play. The cross-model comparison is a
first-class result: whether an agent translates forecast information into negotiation behavior
turns out to be model-dependent. A full 4×4 Buyer×Vendor **role-cross** (a preliminary
exploration in the paper's appendix) localizes the effect to the **Buyer**, the only role shown
the forecast. Claude Sonnet 4.6 was used only to *design* the benchmark (operating point, power analysis)
and is then deliberately **held out** of the evaluated panel so no model is scored on a task tuned to it.

For more information, refer to our [FMTS](https://fmts-workshop.github.io/) @ [NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/workshop/137514) paper: [Vendster: An Evaluation Arena to Investigate How Forecasting Tool Quality Affects LLM Agent Behaviors in Buyer-Vendor Negotiation](https://arxiv.org/abs/2610.38292).

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## Install / setup

Requires **Python >= 3.10** and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

The dependency set is heavy: `torch`, `chronos-forecasting`, `neuralforecast`, and
`timesfm[torch]` pull in the full GPU stack (a CUDA GPU is used automatically if present;
otherwise these run on CPU, slowly). The M5 dataset is **auto-downloaded on first run**
from a single public mirror (`github.com/Nixtla/m5-forecasts`, no Kaggle credentials) and
cached locally under `data/m5_cache/` — subsequent runs read from cache.

On macOS, LightGBM needs the OpenMP runtime (`brew install libomp`). Without it every
`lightgbm_global` episode fails with a `libomp` load error and the cell reports no episodes.

## The benchmark: Vendster-M5

`vendster/reference.py` composes the frozen, versioned reference `BenchmarkConfig`
(`version = vendster-m5-v1`): M5 data source, elasticity demand model, the `vendster`
buyer–vendor scenario, and **26 experimental conditions** —

- **10 real forecasters** across four families: classical (`seasonal_naive`, `auto_ets`,
  `auto_theta`), gradient-boosted (`lightgbm_global`), foundation (`chronos2`,
  `chronos_bolt`, `timesfm25`), and deep learning (`nf_nhits`, `nf_patchtst`, `nf_deepar`),
  plus a `no_forecast` control; and
- a **15-cell perturbation grid**: an 8-level signed-bias dose-response ladder
  (±{0.05,0.10,0.15,0.20}, spanning the empirically-realistic range real forecasters exhibit —
  |bias_log| ≲ 0.20) at calibrated width, the clean `unbiased` oracle, and a 6-level interval-width
  ladder (variance ×{0.25,0.5,0.75,1.5,2,3}) at zero bias.

Each episode's Buyer sales **target** is the **P80** of the basket's total-sales
distribution under a uniform-random reference policy — the "challenging threshold"
operating point where a forecast becomes decision-relevant. The percentile was picked so the
held-out Sonnet 4.6 no-forecast agent lands nearest 0.30; the evaluated panel's own no-forecast
attainment ranges 0.11–0.25, so uplifts are not comparable across models. The hypotheses, the
effect-blind selection rule, and the locked design (P80, K=16 baskets, m=25 episodes per cell,
seed held out from the operating-point sweep) are stated in the paper: see the Hypotheses and
Pre-experiment Sizing paragraphs and the Environment Setup appendix of
[`overleaf/main.tex`](overleaf/main.tex).

Changing the reference instance's components or params requires bumping
`task_core.REFERENCE_VERSION` so published numbers stay comparable. Deriving your own
instance (different data, forecasters, or scenario) needs **no core edits** — see
[`EXTENDING.md`](EXTENDING.md).

## Quick start

### Verify the install (mock LLM, zero Bedrock)

Runs the full paired pipeline end-to-end against a deterministic mock LLM — no AWS
credentials, tiny panel and episode counts:

```bash
uv run python scripts/run_experiment.py \
    --model mock --sets 2 --episodes 2 --panel-size 200 --out /tmp/x
```

### Evaluate one model

Needs an AWS profile with `bedrock:InvokeModel` on the target model id (see
[Authentication](#authentication)). `build_model_bundle.sh` is the general per-model pipeline,
and the way to **evaluate a model of your own** on Vendster-M5: it builds one model's entire
evidence bundle (pool-level forecast quality, per-basket quality, the 26-condition paired
negotiation experiment, and the assembled tables/figures/robustness panel) at the locked
design, so the result is directly comparable with the paper's four. Defaults to Opus 4.6; any
Bedrock model id via `MODEL_ID`:

```bash
bash scripts/build_model_bundle.sh                         # Opus 4.6 (default), foreground
nohup bash scripts/build_model_bundle.sh &                 # detached
MODEL_ID=us.anthropic.claude-sonnet-4-6 OUT=experiments/sonnet_holdout \
    bash scripts/build_model_bundle.sh                     # any other model / output slot
INC=experiments/panel/opus_4_6 OUT=/tmp/opus_rebuild \
    bash scripts/build_model_bundle.sh                     # + replication check vs a prior bundle
CONC=8 CELLCONC=8 MODEL_ID=... bash scripts/build_model_bundle.sh   # throttle for a tight TPM quota
```

It is **unattended-safe**: the Bedrock step runs in a resume-retry loop gated on completeness
(completed episodes are skipped, so re-running resumes), offline steps are deterministic, and
`MANIFEST.md` (git SHA, model, frozen parameters) is written only once the bundle is complete.
To put a new model alongside the paper's panel, build its bundle into `experiments/panel/<name>`
and add `<label>:experiments/panel/<name>` to the `scripts/make_panel.py --models` list.

### The archived experiments

The paper's results (per-episode outcome records, tables, and figures) are committed under
`experiments/`. Every offline script and every command below defaults to that path. The
experiments can also be rerun from scratch.

The archive does not include the turn-by-turn negotiation **transcripts**. The per-episode
outcome records (`episodes/*.jsonl`) are complete, so every table in the paper recomputes
offline. To read negotiations, rerun any cell with the codebase; every run writes its
transcripts under `transcripts/`.

### Reproduce the paper

```bash
bash scripts/reproduce.sh
```

runs the study in the order it was actually carried out: the a-priori power analysis; the
four-model self-play panel (`run_panel.sh`, which calls `build_model_bundle.sh` once per model);
the 4×4 Buyer×Vendor role-cross (`run_role_cross.sh`); the cross-model tables, reactivity
fingerprint, and role-cross heatmap; the per-hypothesis statistics with Holm correction and
MDE; and finally the curated appendix transcripts (from your own runs; the paper's appendix is
left unchanged when they are absent), a refresh of the three figures
`overleaf/main.tex` includes, and a verification pass. The two Bedrock steps are resumable
(a complete bundle is a no-op); everything else is offline.

**To confirm the paper's numbers without rerunning anything on Bedrock:**

```bash
uv run python scripts/paper_stats.py --verify overleaf/main.tex
```

recomputes every cell of the five paper tables from the archived episodes and diffs them
against the manuscript at displayed precision, exiting non-zero on any mismatch (currently
**359/359 cells match**).

The reproducibility contract:

- **Offline artifacts are deterministic** from `(git_sha, seed)`: pool-level forecast quality,
  per-basket quality, and the analytic P-percentile targets.
- **The DGP is seeded** with `blake2b`-derived, process-stable episode seeds (Python's
  salted `hash()` is deliberately avoided), so demand realizations reproduce exactly.
- **LLM negotiation episodes are NOT bit-reproducible** — Bedrock has no seed. The released
  record is therefore the **archived per-episode outcome records**
  (`experiments/*/.../episodes/*.jsonl`). A run also writes each episode's turn-by-turn
  transcript (`transcripts/*.jsonl`, append-only); the paper's transcripts are not released.
- Every run records a **manifest** (git SHA, UTC timestamp, Python version, model, and seed)
  in the `manifest` field of its `summary.json`. Each completed bundle also gets a
  human-readable `MANIFEST.md`.
- **This repository was published fresh** from the development repository, so its history starts
  at the public release. The git SHAs recorded in the archived manifests refer to commits in that
  pre-release history and do not resolve here. `paper_stats.py --verify` recomputes the paper's
  tables from the archived episodes with the code in this repository.

### Browse transcripts from your own runs

After running panel bundles, build an HTML reader over their negotiations (indexes every episode
with a transcript; loads each turn-by-turn log on demand), then serve it. It fetches transcript
files, so it must be served over http, not opened as a bare `file://`:

```bash
uv run python scripts/make_reader.py                 # -> experiments/reader.html
uv run python -m http.server -d experiments 8000     # open http://localhost:8000/reader.html
```

Filter by model / condition / outcome and click an episode to read it. For GitHub Pages, publish
`experiments/reader.html` alongside the `panel/*/experiment/*/transcripts/` tree.

## Scripts

Everything under `scripts/` is a CLI entrypoint (`uv run python scripts/<name>.py --help`, or
`bash scripts/<name>.sh`). `run_*` scripts spend LLM calls to produce negotiation episodes;
`make_*` scripts assemble tables, figures, and documents offline from archived episodes; the
rest are the quality census, the paper statistics, and small utilities.
[`scripts/README.md`](scripts/README.md) documents every script, the call graph between them,
and the layout rule under `experiments/`.

## Authentication

The negotiation experiments call Amazon Bedrock. Any AWS profile (`AWS_PROFILE`, resolved
by boto3) works as long as its principal has **`bedrock:InvokeModel`** on the target model
IDs. All models (Claude, GPT-5.6, Qwen3, DeepSeek) go through the single `bedrock` client,
which uses the vendor-neutral Converse API so the panel shares one code path. The region
defaults to `us-west-2`.

Two things matter for long runs:

- **Prefer long-lived (non-expiring) credentials.** These jobs run for hours; a short-lived
  STS snapshot will die mid-run with `ExpiredTokenException`. Use a profile whose credentials
  auto-refresh (e.g. a `credential_process`) or a long-lived IAM user. The run scripts default
  to a profile named `default` — override with `AWS_PROFILE=<yours>`.
- **Don't gate reachability on `ListFoundationModels`.** Some least-privilege profiles deny it
  while `InvokeModel` works fine; probe with a 1-token `invoke-model` (or `list-inference-profiles`).

## License

This repository is licensed under CC BY-NC 4.0. The released experiments do not include LLM-generated negotiation transcripts; sessions can be rerun using the codebase.

## Citation

If you use Vendster in your research, please cite our [FMTS](https://fmts-workshop.github.io/) @ [NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/workshop/137514) [paper](https://arxiv.org/abs/2610.38292):

```
@misc{polpanumas2026vendsterevaluationarenainvestigate,
      title={Vendster: An Evaluation Arena to Investigate How Forecasting Tool Quality Affects LLM Agent Behaviors in Buyer-Vendor Negotiation}, 
      author={Charin Polpanumas},
      year={2026},
      eprint={2610.38292},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2610.38292}, 
}
```