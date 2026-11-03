"""
Role-cross (Buyer x Vendor) analysis. For each cell in a 4x4 matrix, compute the bias effect
(H1c: bias-ladder attainment slope) and the buyer's behavioral response (first-offer-aggressiveness
slope) + agreement rate. Renders heatmaps rows=Buyer, cols=Vendor.

Headline claim: if the bias effect tracks the ROW (buyer) not the COLUMN (vendor), it is buyer-driven
(the buyer is the only role shown the forecast). The heatmap is the paper's fig:rolecross.

Reads cells written by run_role_cross.sh:  <root>/b_<buyer>__v_<vendor>/experiment/

    uv run python scripts/make_role_cross.py --root experiments/role_cross --out experiments/role_cross
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
from scipy.stats import ttest_1samp

from vendster.analysis import aggregate as agg_mod

BIAS = agg_mod.BIAS_LADDER
XS = [x for _, x in BIAS]


def _cellmean(run, cid, si, key):
    f = f"{run}/set_{si}/episodes/{cid}.jsonl"
    if not os.path.exists(f):
        return np.nan
    vals = []
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "error":
            continue
        if key == "attain":
            if r.get("target_met") is not None:
                vals.append(1.0 if r["target_met"] else 0.0)
        elif key == "agree":
            vals.append(1.0 if r.get("agreement") else 0.0)
        elif key == "first_offer":
            v = r.get("first_offer_aggressiveness")
            if v is not None:
                vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def _slope(run, key):
    slopes, si = [], 0
    while os.path.isdir(f"{run}/set_{si}"):
        ys = [_cellmean(run, cid, si, key) for cid, _ in BIAS]
        if all(not np.isnan(y) for y in ys) and np.ptp(XS) > 0:
            slopes.append(float(np.polyfit(XS, ys, 1)[0]))
        si += 1
    if len(slopes) < 2:
        return float("nan"), float("nan")
    _, p = ttest_1samp(slopes, 0.0)
    return float(np.mean(slopes)), float(p)


def _agree(run):
    vals = []
    for f in glob.glob(f"{run}/set_*/episodes/*.jsonl"):
        for line in open(f):
            if line.strip() and json.loads(line).get("status") != "error":
                vals.append(1.0 if json.loads(line).get("agreement") else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/role_cross")
    ap.add_argument("--out", default="experiments/role_cross")
    args = ap.parse_args()
    out = Path(args.out); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    cells = {}
    buyers, vendors = set(), set()
    for d in sorted(glob.glob(f"{args.root}/b_*__v_*")):
        name = os.path.basename(d)
        b = name.split("b_")[1].split("__v_")[0]; v = name.split("__v_")[1]
        run = f"{d}/experiment"
        if not os.path.isdir(run):
            continue
        h1c_s, h1c_p = _slope(run, "attain")
        fo_s, fo_p = _slope(run, "first_offer")
        cells[(b, v)] = {"H1c_slope": h1c_s, "H1c_p": h1c_p, "first_offer_slope": fo_s,
                         "first_offer_p": fo_p, "agree": _agree(run)}
        buyers.add(b); vendors.add(v)

    order = ["opus", "gpt", "qwen", "deepseek"]
    buyers = [m for m in order if m in buyers]; vendors = [m for m in order if m in vendors]
    (out / "role_cross.json").write_text(json.dumps(
        {f"b_{b}__v_{v}": cells[(b, v)] for (b, v) in cells}, indent=2))

    # markdown table
    L = ["# Role-cross (Buyer × Vendor) — is the bias effect buyer-driven?", "",
         "### bias-attainment slope per cell (negative = bias effect present)",
         "| Buyer ↓ / Vendor → | " + " | ".join(vendors) + " |", "|---|" + "--:|" * len(vendors)]
    for b in buyers:
        row = [f"**{b}**"]
        for v in vendors:
            c = cells.get((b, v))
            row.append(f"{c['H1c_slope']:+.3f}{'*' if c['H1c_p'] < 0.05 else ''}" if c else "—")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "_If cells are colored by ROW (buyer), the effect is buyer-driven. * p<0.05._", "",
          "### First-offer-aggressiveness slope per cell (buyer's forecast-driven opening stance)",
          "| Buyer ↓ / Vendor → | " + " | ".join(vendors) + " |", "|---|" + "--:|" * len(vendors)]
    for b in buyers:
        row = [f"**{b}**"]
        for v in vendors:
            c = cells.get((b, v))
            row.append(f"{c['first_offer_slope']:+.3f}{'*' if c['first_offer_p'] < 0.05 else ''}" if c else "—")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "### agreement rate per cell (watch for deal-failure confound)",
          "| Buyer ↓ / Vendor → | " + " | ".join(vendors) + " |", "|---|" + "--:|" * len(vendors)]
    for b in buyers:
        L.append("| " + " | ".join([f"**{b}**"] + [f"{cells[(b,v)]['agree']:.2f}" if (b, v) in cells else "—" for v in vendors]) + " |")
    (out / "role_cross.md").write_text("\n".join(L) + "\n")

    # heatmap of H1c slope (rows=buyer, cols=vendor)
    M = np.array([[cells[(b, v)]["H1c_slope"] if (b, v) in cells else np.nan for v in vendors] for b in buyers])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(vendors), 1.4 + 0.9 * len(buyers)))
    im = ax.imshow(M, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(vendors))); ax.set_xticklabels(vendors, rotation=30, ha="right")
    ax.set_yticks(range(len(buyers))); ax.set_yticklabels(buyers)
    ax.set_xlabel("Vendor"); ax.set_ylabel("Buyer")
    for i in range(len(buyers)):
        for j in range(len(vendors)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Bias-attainment slope by Buyer×Vendor\n(row-structure ⇒ buyer-driven)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="bias→attainment slope")
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(figdir / f"role_cross_heatmap.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("\n".join(L))
    print(f"\nwrote {out}/role_cross.md + .json + figures/role_cross_heatmap.png")


if __name__ == "__main__":
    main()
