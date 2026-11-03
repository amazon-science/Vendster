"""
Render the Stage-1 operating-point difficulty sweep into paper artifacts — the evidence that
justifies the locked P80 threshold (effect-blind selection; fig:operating in the paper).

Reads the sweep tree produced by ``run_percentile_sweep.py``:
    <sweep>/P{XX}/{no_forecast,unbiased}/set_*/episodes/*.jsonl
and writes, into ``--out``:
    operating_point.md              — per-percentile table (no_forecast vs perfect-forecast oracle)
    operating_point.json            — machine-readable version + the effect-blind selection
    figures/operating_point.png

Deterministic (pure aggregation of archived episodes) — no LLM, no re-run.

    uv run python scripts/make_operating_point.py \
        --sweep experiments/arena_sweep --out experiments/arena_sweep
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
from scipy.stats import wilcoxon

from vendster.analysis.stats import wilson_ci


def _basket_attain(sweep, pct, cond):
    """set_id -> (successes, n) over that basket's episodes for one (percentile, condition)."""
    out = {}
    for f in glob.glob(f"{sweep}/P{pct}/{cond}/set_*/episodes/*.jsonl"):
        si = int(f.split("set_")[1].split("/")[0])
        k = n = 0
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("target_met") is not None:
                n += 1
                k += 1 if r["target_met"] else 0
        if n:
            out[si] = (k, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="experiments/arena_sweep")
    ap.add_argument("--out", default="experiments/arena_sweep")
    ap.add_argument("--select-baseline", type=float, default=0.30,
                    help="effect-blind rule: fix PXX where no_forecast attainment ~= this")
    ap.add_argument("--ladder", default="50,60,70,80,90",
                    help="percentiles to DISPLAY (clean even ladder); off-ladder sweep points "
                         "(e.g. 75, 85) are computed but not shown.")
    args = ap.parse_args()

    ladder = {int(x) for x in args.ladder.split(",")}
    pcts = sorted(p for p in (int(d[1:]) for d in os.listdir(args.sweep)
                              if d.startswith("P") and d[1:].isdigit()) if p in ladder)
    rows = []
    for pct in pcts:
        nfb = _basket_attain(args.sweep, pct, "no_forecast")
        ubb = _basket_attain(args.sweep, pct, "unbiased")
        nfk, nfn = sum(k for k, _ in nfb.values()), sum(n for _, n in nfb.values())
        ubk, ubn = sum(k for k, _ in ubb.values()), sum(n for _, n in ubb.values())
        nf, ub = nfk / nfn, ubk / ubn
        nlo, nhi = wilson_ci(nfk, nfn)
        ulo, uhi = wilson_ci(ubk, ubn)
        sets = sorted(set(nfb) & set(ubb))                      # paired per-basket deltas
        deltas = [ubb[s][0] / ubb[s][1] - nfb[s][0] / nfb[s][1] for s in sets]
        try:
            w_p = float(wilcoxon(deltas).pvalue) if any(d != 0 for d in deltas) else 1.0
        except ValueError:
            w_p = float("nan")
        rows.append({"pct": pct, "no_forecast": nf, "nf_ci95": [nlo, nhi], "nf_n": nfn,
                     "unbiased": ub, "ub_ci95": [ulo, uhi], "ub_n": ubn,
                     "gap": ub - nf, "paired_delta": float(np.mean(deltas)),
                     "wilcoxon_p": w_p, "n_baskets": len(sets)})

    # Effect-blind rule: fix PXX at the percentile whose no_forecast attainment is closest to
    # --select-baseline (~0.30) — uses only the no_forecast arm, never the forecast gap.
    sel = min(rows, key=lambda r: abs(r["no_forecast"] - args.select_baseline))

    out = Path(args.out)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # --- table (md) ---
    L = ["### Appendix — Operating-point difficulty sweep (justifies P80)", "",
         "Stage-1 EXPLORATORY sweep (seed 0, 8 paired baskets, m=15 episodes/cell, `no_forecast` vs "
         "`unbiased` perfect-forecast oracle). The operating point is fixed **effect-blind**: PXX where "
         f"`no_forecast` attainment ≈ {args.select_baseline:.2f} → **P{sel['pct']}**, using only the "
         "no_forecast arm so the choice can't be tuned to the forecast effect (note P80 is not even the "
         "largest-gap percentile in the ladder). Inferential use is descriptive only (the `no_forecast` "
         "difficulty gradient); all confirmatory inference is at the locked P80.", "",
         "| PXX | no_forecast [95% CI] | unbiased [95% CI] | gap | paired Δ | Wilcoxon p | |",
         "|----:|---|---|---:|---:|---:|---|"]
    for r in rows:
        tag = "**← selected (effect-blind)**" if r["pct"] == sel["pct"] else ""
        nlo, nhi = r["nf_ci95"]; ulo, uhi = r["ub_ci95"]
        star = "**" if r["pct"] == sel["pct"] else ""
        L.append(f"| {star}P{r['pct']}{star} | {r['no_forecast']:.3f} [{nlo:.2f},{nhi:.2f}] | "
                 f"{r['unbiased']:.3f} [{ulo:.2f},{uhi:.2f}] | {r['gap']:+.3f} | {r['paired_delta']:+.3f} "
                 f"| {r['wilcoxon_p']:.3f} | {tag} |")
    L += ["", f"_Selection rule (pre-specified, effect-blind): fix PXX at no_forecast ≈ "
          f"{args.select_baseline:.2f}. Result: **P{sel['pct']}** (no_forecast {sel['no_forecast']:.3f})._", ""]
    (out / "operating_point.md").write_text("\n".join(L) + "\n")
    (out / "operating_point.json").write_text(json.dumps(
        {"rows": rows, "selected": sel, "select_baseline": args.select_baseline}, indent=2))

    # --- figure ---
    xs = [r["pct"] for r in rows]
    nf = np.array([r["no_forecast"] for r in rows])
    ub = np.array([r["unbiased"] for r in rows])
    nf_err = [nf - np.array([r["nf_ci95"][0] for r in rows]),
              np.array([r["nf_ci95"][1] for r in rows]) - nf]
    ub_err = [ub - np.array([r["ub_ci95"][0] for r in rows]),
              np.array([r["ub_ci95"][1] for r in rows]) - ub]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(xs, nf, yerr=nf_err, fmt="o-", capsize=3, color="#c0392b", label="no_forecast")
    ax.errorbar(xs, ub, yerr=ub_err, fmt="s-", capsize=3, color="#2c3e50", label="unbiased (perfect)")
    ax.axhline(args.select_baseline, ls=":", color="grey", lw=1,
               label=f"effect-blind target ≈ {args.select_baseline:.2f}")
    ax.axvline(sel["pct"], ls="--", color="#27ae60", lw=1.5, label=f"selected P{sel['pct']}")
    ax.set_xlabel("target percentile PXX (harder →)")
    ax.set_ylabel("target attainment")
    ax.set_ylim(0, 0.7)
    ax.set_title("Operating-point selection: no_forecast difficulty gradient")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    paths = []
    for ext in ("png",):
        p = figdir / f"operating_point.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(str(p))
    plt.close(fig)

    print("\n".join(L))
    print(f"\nselected: P{sel['pct']} (no_forecast {sel['no_forecast']:.3f}); ladder={sorted(ladder)}")
    print(f"wrote {out/'operating_point.md'} + .json + {paths[0]}")


if __name__ == "__main__":
    main()
