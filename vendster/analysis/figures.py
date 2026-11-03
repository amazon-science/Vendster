"""
Paper figures (matplotlib, Agg backend).

Descriptive, order-free names: quality->attainment, perturbation grid,
mechanism, target calibration, cross-basket spread, buyer-vendor welfare.
Each saves a .png into ``out_dir``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png",):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(str(p))
    plt.close(fig)
    return paths


def quality_outcome(link_fc: dict, baseline_attain: float, out_dir: str,
                    link_basket: dict | None = None):
    """x = forecast quality (relWQL, lower=better); y = attainment. Forecaster means
    (+ basket scatter & OLS fit if provided); no_forecast as a horizontal baseline."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if link_basket and link_basket.get("points"):
        bx = [p["x"] for p in link_basket["points"]]
        by = [p["y"] for p in link_basket["points"]]
        ax.scatter(bx, by, s=14, alpha=0.25, color="#8888cc", label="per basket×forecaster")
        if not np.isnan(link_basket.get("ols_slope", float("nan"))):
            xs = np.linspace(min(bx), max(bx), 50)
            ax.plot(xs, link_basket["ols_intercept"] + link_basket["ols_slope"] * xs,
                    "-", color="#333399", lw=1.5,
                    label=f"fit ρ={link_basket['spearman']:.2f}")
    for p in link_fc["points"]:
        ax.scatter(p["x"], p["y"], s=70, zorder=5)
        ax.annotate(p["forecaster"], (p["x"], p["y"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    if baseline_attain == baseline_attain:                      # not NaN
        ax.axhline(baseline_attain, ls="--", color="grey", lw=1,
                   label=f"no_forecast baseline ({baseline_attain:.0%})")
    ax.set_xlabel("forecast quality — relative WQL (lower = better)")
    ax.set_ylabel("target attainment")
    ax.set_title("Forecast quality → negotiation outcome")
    ax.legend(fontsize=7, loc="best")
    return _save(fig, out_dir, "quality_outcome")


def _ladder_xy(agg, ladder):
    """(xs, ys, err) of pooled attainment ± Wilson CI along a ladder of conditions."""
    xs, ys, lo, hi = [], [], [], []
    for cid, x in ladder:
        c = agg["conditions"].get(cid)
        if c is None:
            continue
        xs.append(x); ys.append(c["attain_pooled"])
        lo.append(c["attain_ci95"][0]); hi.append(c["attain_ci95"][1])
    ys = np.array(ys)
    err = [np.clip(ys - np.array(lo), 0, None), np.clip(np.array(hi) - ys, 0, None)]
    return np.array(xs), ys, err


def perturbation_grid(agg: dict, out_dir: str, bias_trend: dict | None = None,
                      width_trend: dict | None = None):
    """Two dose-response panels: (A) attainment vs bias @ calibrated width; (B) attainment
    vs interval width @ zero bias. Points = pooled attainment ± Wilson CI; faint lines =
    per-basket trajectories (the paired structure); trend slope + p annotated if provided."""
    from .aggregate import BIAS_LADDER, WIDTH_LADDER, ladder_matrix
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, ladder, title, xlab, trend in (
        (axes[0], BIAS_LADDER, "Bias dose-response (calibrated width)",
         "forecast bias (log)", bias_trend),
        (axes[1], WIDTH_LADDER, "Interval calibration (unbiased)",
         "interval-width × (0.5 over → 2.0 under)", width_trend)):
        xm, mat = ladder_matrix(agg, ladder)
        for row in mat:                                   # faint per-basket trajectories
            ax.plot(xm, row, "-", color="#bbbbbb", lw=0.6, alpha=0.5)
        xs, ys, err = _ladder_xy(agg, ladder)
        ax.errorbar(xs, ys, yerr=err, fmt="o-", capsize=3, color="#333399", zorder=5)
        ax.axhline(0.5, ls=":", color="grey", lw=1)
        if ladder is WIDTH_LADDER:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xlab); ax.set_ylabel("target attainment"); ax.set_ylim(0, 1)
        t = trend
        sub = f"  (slope={t['slope']:+.2f}, p={t['p_value']:.3f})" if t and t["n_baskets"] > 1 else ""
        ax.set_title(title + sub, fontsize=9)
    fig.tight_layout()
    return _save(fig, out_dir, "perturbation_grid")


def mechanism(agg: dict, out_dir: str):
    """Belief -> behavior -> outcome along the (realistic) bias ladder — the mechanism. Overlays
    attainment, first-offer aggressiveness, and mean agreed discount (left axis) with realized
    sales/target (right axis). The aligned shape shows over-forecasting breeds COMPLACENCY (the buyer
    believes demand is high -> under-negotiates -> lower discount -> under-shoots the target) while
    under-forecasting spurs harder negotiation -> higher attainment."""
    from .aggregate import BIAS_LADDER
    present = [(cid, x) for cid, x in BIAS_LADDER if cid in agg["conditions"]]
    xs = [x for _, x in present]

    def ser(field):                                  # mean of a per-episode FIELD along the ladder
        return [agg["conditions"][cid][field]["mean"] for cid, _ in present]

    attain = [agg["conditions"][cid]["attain_pooled"] for cid, _ in present]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xs, attain, "o-", color="C0", label="attainment")
    ax.plot(xs, ser("first_offer"), "s--", color="C1", label="1st-offer aggressiveness")
    ax.plot(xs, ser("mean_discount"), "^--", color="C2", label="mean agreed discount")
    ax.axvline(0.0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("forecast bias (log; − under, + over)")
    ax.set_ylabel("attainment / aggressiveness / discount")
    ax2 = ax.twinx()
    ax2.plot(xs, ser("sales_target"), "d:", color="C3", label="realized sales / target")
    ax2.set_ylabel("realized sales / target")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="lower center", ncol=2)
    ax.set_title("Mechanism: over-forecasting → complacency → under-shoot")
    return _save(fig, out_dir, "mechanism")


def cross_basket_spread(agg: dict, conds: list, out_dir: str):
    """Box of per-basket attainment across baskets, per condition."""
    data, labels = [], []
    for c in conds:
        vals = [r["attain"] for r in agg["per_basket"].get(c, []) if not np.isnan(r["attain"])]
        if vals:
            data.append(vals); labels.append(c)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(labels)), 4.0))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0.5, ls=":", color="grey", lw=1)
    ax.set_ylabel("attainment (per basket)"); ax.set_ylim(0, 1)
    ax.set_title("Cross-basket attainment spread")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
    return _save(fig, out_dir, "cross_basket_spread")


def welfare(agg: dict, conds: list, out_dir: str):
    """Buyer–vendor welfare: attainment vs mean vendor profit by condition."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for c in conds:
        cc = agg["conditions"][c]
        x, y = cc["profit"]["mean"], cc["attain_pooled"]
        if not (np.isnan(x) or np.isnan(y)):
            ax.scatter(x, y, s=60)
            ax.annotate(c, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("mean vendor profit"); ax.set_ylabel("Buyer target attainment")
    ax.set_title("Buyer–vendor welfare by condition")
    return _save(fig, out_dir, "welfare")
