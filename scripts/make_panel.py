"""
Multi-model headline assembler. Reads several per-model bundles (each already built by
make_one_model) and produces the paper's cross-model story:

  panel.md    Table A — results per model (H0 gate, H1c/H2c bias/width causal, H1o/H3o
                              observational) + verdict; Table B — BEHAVIORAL fingerprint:
                              bias-ladder slope of each negotiation behavior (how the model
                              translates the forecast into action).
  panel.json  machine-readable.
  panel.md    ... also Table C — real-forecaster attainment by model; Table D — accuracy→
                    attainment slope by metric × model (the accuracy-null robustness, whole panel).
  figures/panel_bias_ladder.png        attainment vs bias, one line per model.
  figures/panel_quality_outcome.png    4-panel accuracy→outcome scatter (shared color legend).
  figures/panel_perturbation_grid.png  2×4 causal grid: bias (H1c) vs width (H2c) per model.
  figures/panel_mechanism.png          4-panel belief→behavior→outcome (H4) per model;
                                             agreed-only pathway + agreement-rate (deal-closure) channel.
  figures/panel_welfare.png            4-panel buyer–vendor surplus guard (H5) per model.

The behavioral slopes are the main-body "why models differ" evidence: a model that USES the
forecast to negotiate harder has steep behavioral slopes; a model decoupled from it has flat ones.

    uv run python scripts/make_panel.py \
        --models opus:experiments/panel/opus_4_6 gpt:experiments/panel/gpt_5_6_sol \
                 qwen:experiments/panel/qwen3_235b deepseek:experiments/panel/deepseek_v3_2 \
        --out experiments/panel
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, ttest_1samp

from vendster.analysis import aggregate as agg_mod
from vendster.core.config import REAL_FORECASTERS as REAL

BIAS = agg_mod.BIAS_LADDER
WIDTH = agg_mod.WIDTH_LADDER
XS = [x for _, x in BIAS]
# (record-field, display) — every behavioral angle: opening stance, concession, deal depth,
# persistence, deal-closing.
BEHAV = [("first_offer_aggressiveness", "1st-offer aggr"), ("concession_rate", "concession"),
         ("mean_discount", "agreed discount"), ("turns", "turns"), ("agree", "agreement rate")]


def _cellmean(run, cid, si, key):
    f = f"{run}/experiment/set_{si}/episodes/{cid}.jsonl"
    if not os.path.exists(f):
        return np.nan
    vals = []
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "error":
            continue
        if key == "agree":
            vals.append(1.0 if r.get("agreement") else 0.0)
        elif key == "mean_discount":
            d = r.get("agreed_discounts") or {}
            if r.get("agreement") and d:
                vals.append(float(np.mean(list(d.values()))))
        elif key == "turns":
            if r.get("turns") is not None:
                vals.append(float(r["turns"]))
        else:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def _behav_slope(run, key):
    """Per-basket OLS slope of a behavioral metric on the bias ladder -> one-sample t over baskets."""
    slopes, si = [], 0
    while os.path.isdir(f"{run}/experiment/set_{si}"):
        ys = [_cellmean(run, cid, si, key) for cid, _ in BIAS]
        if all(not np.isnan(y) for y in ys) and np.ptp(XS) > 0:
            slopes.append(float(np.polyfit(XS, ys, 1)[0]))
        si += 1
    if len(slopes) < 2:
        return {"slope": float("nan"), "p": float("nan"), "n": len(slopes)}
    _, p = ttest_1samp(slopes, 0.0)
    return {"slope": float(np.mean(slopes)), "p": float(p), "n": len(slopes)}


def _behav_uplift(run, key):
    """Per-basket PRESENCE uplift of a behavior: mean over the ten real forecasters minus no_forecast,
    one-sample t over baskets. Same contrast as the H0 attainment uplift (Forecast Uplift), but on each
    behavior---so it decomposes what having a forecast (regardless of its content) does to negotiation."""
    ups, si = [], 0
    while os.path.isdir(f"{run}/experiment/set_{si}"):
        nf = _cellmean(run, "no_forecast", si, key)
        rf = [x for x in (_cellmean(run, fc, si, key) for fc in REAL) if not np.isnan(x)]
        if rf and not np.isnan(nf):
            ups.append(float(np.mean(rf)) - nf)
        si += 1
    if len(ups) < 2:
        return {"delta": float("nan"), "p": float("nan"), "n": len(ups)}
    _, p = ttest_1samp(ups, 0.0)
    return {"delta": float(np.mean(ups)), "p": float(p), "n": len(ups)}




def _agree(cell):
    a = cell.get("agree")
    return a.get("mean") if isinstance(a, dict) else a


def _welfare_stats(cond):
    """Welfare guard (H5) numbers used in the Results welfare paragraph, computed across the 25 non-\
    no_forecast conditions: the attainment-vs-Vendor-profit correlation (do bigger pies accompany higher
    attainment, or is it a transfer?), the profit-vs-agreement-rate correlation (deal-closure channel),
    and the min/max agreement rate along the bias ladder (the closure swing)."""
    noNF = [c for c in cond if c != "no_forecast" and "profit" in cond[c] and _agree(cond[c]) is not None]
    at = [cond[c]["attain_pooled"] for c in noNF]
    pr = [cond[c]["profit"]["mean"] for c in noNF]
    ar = [_agree(cond[c]) for c in noNF]
    ags = [_agree(cond[c]) for c, _ in BIAS if c in cond and _agree(cond[c]) is not None]
    ap_r, ap_p = pearsonr(at, pr)
    return {"attain_profit_r": float(ap_r), "attain_profit_p": float(ap_p),
            "profit_agree_r": float(pearsonr(ar, pr)[0]),
            "agree_min": float(min(ags)), "agree_max": float(max(ags)), "n": len(noNF)}


def _tr(t):
    return f"{t['slope']:+.3f} (p={t['p_value']:.3f})" if t.get("n_baskets", 0) > 1 else "n/a"


def _mech_series(run):
    """Per bias-ladder condition, separating the two channels a forecast acts through:
      - opening stance (first_offer_aggressiveness) over ALL episodes — the pure belief→behavior signal;
      - agreed discount + attainment conditioned on AGREED deals — the pathway among closed deals;
      - agreement rate over all episodes — the deal-closure channel.
    Conditioning the outcome/discount on agreed deals removes the deal-failure confound (aggressive
    under-forecast negotiation fails to close more often, esp. GPT), so the pathway reads cleanly."""
    xs, first, disc, att_agr, agree = [], [], [], [], []
    for cid, x in BIAS:
        recs = []
        for f in glob.glob(f"{run}/experiment/set_*/episodes/{cid}.jsonl"):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") != "error":
                        recs.append(r)
        if not recs:
            continue
        agr = [r for r in recs if r.get("agreement")]
        fo = [r["first_offer_aggressiveness"] for r in recs if r.get("first_offer_aggressiveness") is not None]
        dv = [float(np.mean(list(r["agreed_discounts"].values()))) for r in agr if r.get("agreed_discounts")]
        xs.append(x)
        first.append(float(np.mean(fo)) if fo else np.nan)
        disc.append(float(np.mean(dv)) if dv else np.nan)
        att_agr.append(float(np.mean([1.0 if r.get("target_met") else 0.0 for r in agr])) if agr else np.nan)
        agree.append(len(agr) / len(recs))
    return xs, first, disc, att_agr, agree


def _mechanism_panel(data, figdir):
    """4-panel (one per model) of the belief→behavior→outcome mechanism along the bias ladder.
    Outcome (attainment) and agreed discount are conditioned on AGREED deals, and agreement rate is
    drawn separately (right axis) — so the belief→behavior pathway is isolated from the deal-closure
    channel. Translators' curves slope (Opus/GPT); non-translators' are flat (Qwen/DeepSeek)."""
    n = len(data)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), sharex=True, squeeze=False)
    axf = axes.flat
    h1 = h2 = lab1 = lab2 = None
    for ax, (m, r) in zip(axf, data.items()):
        xs, first, disc, att_agr, agree = _mech_series(r["dir"])
        ax.plot(xs, att_agr, "o-", color="C0", label="attainment (agreed)")
        ax.plot(xs, first, "s--", color="C1", label="1st-offer aggr")
        ax.plot(xs, disc, "^--", color="C2", label="agreed discount")
        ax.axvline(0.0, color="gray", lw=0.7, ls=":")
        ax2 = ax.twinx()
        ax2.plot(xs, agree, "d:", color="C3", label="agreement rate")
        ax2.set_ylim(0, 1.05)
        ax.set_title(m, fontsize=10, fontweight="bold")
        ax.set_ylabel("attain(agr) / aggr / disc", fontsize=8)
        ax2.set_ylabel("agreement rate", fontsize=8)
        if h1 is None:
            h1, lab1 = ax.get_legend_handles_labels()
            h2, lab2 = ax2.get_legend_handles_labels()
    for ax in list(axf)[n:]:                              # hide unused cells
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("forecast bias (log; − under, + over)", fontsize=8)
    fig.legend(h1 + h2, lab1 + lab2, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Mechanism by model (agreed-only pathway + deal-closure channel)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png",):
        fig.savefig(figdir / f"panel_mechanism.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _welfare_panel(data, figdir):
    """4-panel buyer–vendor welfare (one per model): each condition's attainment vs mean vendor
    profit. Guards H5 — buyer attainment gains aren't merely surplus transferred from the vendor."""
    n = len(data)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    axf = axes.flat
    for ax, (m, r) in zip(axf, data.items()):
        cond = r["cond"]
        xs = [cond[c]["attain_pooled"] for c in cond]
        ys = [cond[c]["profit"]["mean"] for c in cond]
        ax.scatter(xs, ys, s=16, color="C0", alpha=0.7)
        if "no_forecast" in cond:                         # anchor the baseline
            ax.scatter([cond["no_forecast"]["attain_pooled"]], [cond["no_forecast"]["profit"]["mean"]],
                       s=45, color="C3", marker="X", label="no_forecast", zorder=3)
            ax.legend(fontsize=7, loc="best")
        ax.set_title(m, fontsize=10, fontweight="bold")
        ax.set_xlabel("buyer attainment", fontsize=8)
        ax.set_ylabel("mean vendor profit", fontsize=8)
    for ax in list(axf)[n:]:
        ax.set_visible(False)
    fig.suptitle("Buyer–vendor welfare by condition (surplus-transfer guard)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png",):
        fig.savefig(figdir / f"panel_welfare.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _quality_outcome_panel(data, figdir):
    """4-panel accuracy→outcome scatter, one per model: x = each real forecaster's relWQL (lower =
    better, model-independent), y = that forecaster's attainment under this model; no_forecast baseline
    line. Forecasters are COLOR-coded with a single shared legend (no per-point text) so the panels stay
    readable; the flat clouds (vs GPT's mild downslope) show accuracy rarely predicts outcome."""
    fcs = [fc for fc in REAL if any(fc in r["cond"] for r in data.values())]
    cmap = plt.get_cmap("tab10")
    colors = {fc: cmap(i % 10) for i, fc in enumerate(fcs)}
    # relWQL per forecaster is model-independent → take it from any model's robustness_metrics fc_means.
    any_rob = next(iter(data.values()))["rob"]["accuracy_metrics"]
    relwql = next(x for x in any_rob if x["metric"] == "relative_wql")["fc_means"]
    n = len(data)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.0 * nrow), squeeze=False)
    axf = axes.flat
    for ax, (m, r) in zip(axf, data.items()):
        for fc in fcs:
            c = r["cond"].get(fc)
            if c is None or fc not in relwql:
                continue
            ax.scatter(relwql[fc], c["attain_pooled"], s=45, color=colors[fc], zorder=3)
        nf = r["cond"].get("no_forecast")
        if nf:
            ax.axhline(nf["attain_pooled"], ls="--", color="grey", lw=1)
        ax.set_title(m, fontsize=10, fontweight="bold")
        ax.set_xlabel("forecast quality — relWQL (lower = better)", fontsize=8)
        ax.set_ylabel("target attainment", fontsize=8)
    for ax in list(axf)[n:]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[fc], label=fc) for fc in fcs]
    handles.append(plt.Line2D([], [], ls="--", color="grey", label="no_forecast baseline"))
    fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Forecast quality → negotiation outcome (accuracy rarely predicts attainment)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("png",):
        fig.savefig(figdir / f"panel_quality_outcome.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _perturbation_panel(data, figdir):
    """Causal dose-response grid — 2 rows (bias, interval width) × one column per model. Pooled
    attainment ± Wilson CI along each *dialed* ladder + the trend slope/p. The whole causal argument
    in one figure: bias moves attainment (columns slope) while width is flat."""
    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.0), sharey=True, squeeze=False)
    for j, (m, r) in enumerate(data.items()):
        cond = r["cond"]
        for i, (ladder, tkey, xlab, lab) in enumerate([
                (BIAS, "H1c", "forecast bias (log; − under, + over)", "bias"),
                (WIDTH, "H2c", "interval width × (0.25 over → 3.0 under)", "width")]):
            ax = axes[i][j]
            xs, ys, lo, hi = [], [], [], []
            for cid, x in ladder:
                c = cond.get(cid)
                if c is None:
                    continue
                xs.append(x); ys.append(c["attain_pooled"])
                lo.append(c["attain_ci95"][0]); hi.append(c["attain_ci95"][1])
            ys = np.array(ys)
            err = [np.clip(ys - np.array(lo), 0, None), np.clip(np.array(hi) - ys, 0, None)]
            ax.errorbar(xs, ys, yerr=err, fmt="o-", capsize=2, color="#333399")
            if ladder is WIDTH:
                ax.set_xscale("log", base=2)
            else:
                ax.axvline(0.0, color="gray", lw=0.7, ls=":")
            t = r[tkey]
            sub = f"\nslope={t['slope']:+.2f}, p={t['p_value']:.3f}" if t.get("n_baskets", 0) > 1 else ""
            ax.set_title(f"{m} — {lab}{sub}", fontsize=8, fontweight="bold" if i == 0 else "normal")
            ax.set_ylim(0, 0.6)
            if i == 1:
                ax.set_xlabel(xlab, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{lab}\ntarget attainment", fontsize=8)
    fig.suptitle("Causal dose-response: bias moves attainment, width does not", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png",):
        fig.savefig(figdir / f"panel_perturbation_grid.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label:bundle_dir ...")
    ap.add_argument("--out", default="experiments/panel")
    args = ap.parse_args()
    out = Path(args.out); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    models = [m.split(":", 1) for m in args.models]
    data = {}
    for label, d in models:
        agg = json.loads((Path(d) / "paper" / "aggregate.json").read_text())
        cond = agg["aggregate"]["conditions"]; tr = agg["trends"]
        nf = cond["no_forecast"]["attain_pooled"]
        reals = [cond[fc]["attain_pooled"] for fc in REAL if fc in cond]
        ladder = [cond[cid]["attain_pooled"] if cid in cond else float("nan") for cid, _ in BIAS]
        data[label] = {
            "dir": d, "no_fc": nf, "gate_min": min(reals) - nf, "gate_max": max(reals) - nf,
            "H1c": tr["H1c_bias_causal"], "H2c": tr["H2c_width_causal"],
            "H1o": tr["H1o_bias_obs"], "H3o": tr["H3o_accuracy_obs"], "ladder": ladder,
            "behav": {k: _behav_slope(d, k) for k, _ in BEHAV},
            "behav_uplift": {k: _behav_uplift(d, k) for k, _ in BEHAV},   # presence effect per behavior
            "welfare": _welfare_stats(cond),             # H5 welfare-paragraph correlations (attain~profit, closure)
            "cond": cond,                                # full per-condition means (for the 4-panel figs)
            "rob": json.loads((Path(d) / "paper" / "robustness_metrics.json").read_text()),
        }

    # ---- Table A: cross-model results ----
    L = ["# Vendster — cross-model results", "",
         "### Table A — forecast effect by model",
         "| model | no_fc | beats no_fc (Δ range) | bias slope (causal grid) | width slope (causal grid) | bias slope (real fc) | accuracy slope (real fc) | forecast-reactive? |",
         "|---|--:|---|---|---|---|---|---|"]
    for m, r in data.items():
        shows = "**yes**" if r["H1c"]["p_value"] < 0.05 and r["H1c"]["slope"] < 0 else "no"
        L.append(f"| {m} | {r['no_fc']:.2f} | +{r['gate_min']:.2f}..+{r['gate_max']:.2f} | "
                 f"{_tr(r['H1c'])} | {_tr(r['H2c'])} | {_tr(r['H1o'])} | {_tr(r['H3o'])} | {shows} |")
    L += ["", "_Causal-grid slopes: basket-clustered dose-response on the dialed perturbation grid. "
          "Real-fc slopes: per-basket slope across the real forecasters (observational). Negative bias "
          "slope = under-forecasting helps / over-forecasting hurts (complacency)._", ""]

    # ---- Table B: behavioral fingerprint ----
    L += ["### Table B — behavioral fingerprint: bias-ladder slope of negotiation behavior",
          "How strongly each model translates the forecast into negotiation behavior (steeper |slope| "
          "= more forecast-reactive). Under→over-forecast direction.", "",
          "| model | " + " | ".join(disp for _, disp in BEHAV) + " |",
          "|---|" + "--:|" * len(BEHAV)]
    for m, r in data.items():
        cells = []
        for k, _ in BEHAV:
            b = r["behav"][k]
            star = "*" if b["p"] < 0.05 else ""
            cells.append(f"{b['slope']:+.3f}{star}" if b["n"] > 1 else "n/a")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "_* p<0.05 (one-sample t over baskets). first-offer aggressiveness = opening stance; "
          "concession = give-rate; agreed discount = deal depth; turns = persistence; agreement = "
          "deal-closing._", ""]

    # ---- Table C: real-forecaster attainment across the panel (full roster) ----
    L += ["### Table C — real-forecaster target attainment by model (Δ vs no_forecast)",
          "Every real forecaster vs the no_forecast baseline, per model. Cell = pooled attainment "
          "(Δ vs that model's no_forecast).", "",
          "| forecaster | " + " | ".join(m for m in data) + " |",
          "|---|" + "--:|" * len(data)]
    for fc in ["no_forecast"] + REAL:
        cells = []
        for m, r in data.items():
            c = r["cond"].get(fc)
            if c is None:
                cells.append("n/a"); continue
            a = c["attain_pooled"]
            cells.append(f"{a:.2f}" if fc == "no_forecast" else f"{a:.2f} ({a - r['no_fc']:+.2f})")
        L.append(f"| {fc} | " + " | ".join(cells) + " |")
    L += ["", "_Δ in parentheses = attainment gain over that model's no_forecast baseline. Every "
          "forecaster clears the baseline for every model._", ""]

    # ---- Table D: accuracy → attainment across metrics × models (the accuracy-null robustness) ----
    METRICS = ["relative_wql", "mase", "mape", "smape", "rmse", "crps"]
    L += ["### Table D — forecast accuracy → attainment: design-correct slope by metric × model",
          "Per-basket slope of attainment on each accuracy metric across the real forecasters "
          "(negative = better accuracy helps). Bold = p<0.05. Null for almost every (metric, model) — "
          "the exception is the most forecast-reactive model on the primary metric.", "",
          "| metric | " + " | ".join(m for m in data) + " |",
          "|---|" + "--:|" * len(data)]
    for met in METRICS:
        cells = []
        for m, r in data.items():
            row = next((x for x in r["rob"]["accuracy_metrics"] if x["metric"] == met), None)
            if row is None:
                cells.append("n/a"); continue
            s, p = row["design_slope"], row["design_p"]
            cells.append(f"**{s:+.3f} (p={p:.3f})**" if p < 0.05 else f"{s:+.3f} (p={p:.3f})")
        L.append(f"| {met} | " + " | ".join(cells) + " |")
    L += ["", "_Design-correct test (basket = unit of analysis, one-sample t over baskets), one row per "
          "accuracy metric. See each model's bundle (paper/robustness_metrics.md) for the naive-pooled "
          "and forecaster-level lenses._", ""]

    (out / "panel.md").write_text("\n".join(L) + "\n")
    slim = {m: {k: v for k, v in r.items() if k not in ("cond", "rob")} for m, r in data.items()}  # drop bulky blobs
    (out / "panel.json").write_text(json.dumps(slim, indent=2, default=str))

    # ---- Figure: bias-ladder attainment, one line per model ----
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m, r in data.items():
        ax.plot(XS, r["ladder"], "o-", label=m)
    ax.axvline(0.0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("forecast bias (log; − under, + over)")
    ax.set_ylabel("target attainment"); ax.set_ylim(0, 0.6)
    ax.set_title("Bias dose-response by model (down-slope = complacency)")
    ax.legend(fontsize=8)
    for ext in ("png",):
        fig.savefig(figdir / f"panel_bias_ladder.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- 4-panel main-body figures: quality→outcome + perturbation grid + mechanism + welfare ----
    _quality_outcome_panel(data, figdir)
    _perturbation_panel(data, figdir)
    _mechanism_panel(data, figdir)
    _welfare_panel(data, figdir)

    print("\n".join(L))
    print(f"\nwrote {out}/panel.md + .json + figures/{{bias_ladder, quality_outcome, "
          "perturbation_grid, mechanism, welfare}}.png (panel_*)")


if __name__ == "__main__":
    main()
