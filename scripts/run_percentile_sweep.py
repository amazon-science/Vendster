"""
Sweep the target percentile to pick the operating point PXX where a PERFECT forecast (unbiased)
maximally separates from no_forecast — i.e. where forecast INFORMATION is most valuable.

At P50 the gap is ~0 (median target: over-discounting clears it regardless of forecast). At P80
we measured +0.16. This maps the whole curve to find the peak-separation percentile, then that
PXX becomes the primary experiment's operating point.

Paired design: SAME 8 baskets (seed) across conditions & percentiles. Reports per percentile:
pooled no_forecast vs unbiased (Wilson CI), the paired per-basket delta, and the Wilcoxon
signed-rank p (robust to the odd high-variance basket at small K).

Runs the full grid through ONE flat thread pool at high concurrency (each cell = a
(percentile, cond, basket) of m episodes; the global _SETUP_LOCK keeps forecaster inference safe).

    AWS_PROFILE=<your-bedrock-profile> uv run python scripts/run_percentile_sweep.py \
        --percentiles 50 60 70 75 80 85 90 --baskets 8 --episodes 15 \
        --cell-concurrency 15 --outer 16 --out experiments/arena_sweep
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import vendster.forecasters  # noqa: F401
from vendster.analysis.stats import paired_ttest_delta, wilson_ci
from vendster.core.config import reference_conditions
from vendster.llm import add_llm_args, make_llm
from vendster.data.m5 import M5DataSource
from vendster.dgp.demand import ElasticityDemandModel
from vendster.dgp.stretch_pool import build_stretch_pool
from vendster.experiment.runner import _run_condition, analytic_target

CONDS = ["no_forecast", "unbiased"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentiles", type=float, nargs="+", default=[50, 60, 70, 75, 80, 85, 90])
    add_llm_args(ap, default="bedrock")
    ap.add_argument("--baskets", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-weekly", type=int, default=5)
    ap.add_argument("--panel-size", type=int, default=1500)
    ap.add_argument("--cell-concurrency", type=int, default=15,
                    help="parallel episodes within one (pct,cond,basket) cell")
    ap.add_argument("--outer", type=int, default=16,
                    help="cells run concurrently; total in-flight ~= outer * cell-concurrency")
    ap.add_argument("--out", default="experiments/arena_sweep")
    args = ap.parse_args()

    t0 = time.time()
    src = M5DataSource(min_weekly=args.min_weekly, panel_size=args.panel_size)
    llm = make_llm(args.model, args.model_id)
    dm = ElasticityDemandModel()
    conds = {c.id: c for c in reference_conditions()}

    # Per-basket dataset + pool built once (percentile-independent); targets per (basket, pct).
    print(f"building {args.baskets} baskets ...", flush=True)
    ds_by_si, pool_by_si = {}, {}
    for si in range(args.baskets):
        basket = src.sample_basket(np.random.default_rng(args.seed * 7919 + si))
        ds_by_si[si] = src.build_dataset(basket, cost_seed=si)
        pool_by_si[si] = build_stretch_pool(ds_by_si[si], min_history=60)
    tgt = {(si, pct): analytic_target(ds_by_si[si], seed=args.seed * 100 + si, percentile=pct)["target"]
           for si in range(args.baskets) for pct in args.percentiles}

    # Flat job list: one cell per (pct, cond, basket).
    jobs = [(pct, cid, si) for pct in args.percentiles for cid in CONDS for si in range(args.baskets)]
    print(f"{len(jobs)} cells x {args.episodes} eps  (outer={args.outer} x cell={args.cell_concurrency} "
          f"= ~{args.outer * args.cell_concurrency} in-flight)", flush=True)

    def run_cell(job):
        pct, cid, si = job
        out = Path(args.out) / f"P{int(pct)}" / cid / f"set_{si}"
        recs = _run_condition(conds[cid], ds_by_si[si], pool_by_si[si], tgt[(si, pct)], llm,
                              args.episodes, args.seed * 100 + si, out, {}, True, dm,
                              args.cell_concurrency)
        ok = [r for r in recs if r.get("target_met") is not None]
        succ = sum(1 for r in ok if r["target_met"])
        return (pct, cid, si, succ, len(ok))

    with ThreadPoolExecutor(max_workers=args.outer) as ex:
        results = list(ex.map(run_cell, jobs))

    # Aggregate per (pct, cond): pooled + per-basket attainment vectors (ordered by si).
    def cell(pct, cid, si):
        for (p, c, s, k, n) in results:
            if p == pct and c == cid and s == si:
                return k, n
        return 0, 0

    print(f"\n{'PXX':>5} | {'no_forecast':>22} | {'unbiased(perfect)':>22} | {'gap':>6} | "
          f"{'paired Δ':>8} | {'t p':>6} | {'wilcox p':>8}", flush=True)
    print("-" * 92, flush=True)
    rows = []
    for pct in args.percentiles:
        nf_k = sum(cell(pct, "no_forecast", si)[0] for si in range(args.baskets))
        nf_n = sum(cell(pct, "no_forecast", si)[1] for si in range(args.baskets))
        ub_k = sum(cell(pct, "unbiased", si)[0] for si in range(args.baskets))
        ub_n = sum(cell(pct, "unbiased", si)[1] for si in range(args.baskets))
        nf_v = [cell(pct, "no_forecast", si)[0] / max(1, cell(pct, "no_forecast", si)[1])
                for si in range(args.baskets)]
        ub_v = [cell(pct, "unbiased", si)[0] / max(1, cell(pct, "unbiased", si)[1])
                for si in range(args.baskets)]
        nf_a, ub_a = nf_k / max(1, nf_n), ub_k / max(1, ub_n)
        nlo, nhi = wilson_ci(nf_k, max(1, nf_n))
        ulo, uhi = wilson_ci(ub_k, max(1, ub_n))
        st = paired_ttest_delta(nf_v, ub_v)
        gap = ub_a - nf_a
        sep = "  <-- CIs disjoint" if nhi < ulo else ""
        rows.append({"pct": pct, "nf": nf_a, "ub": ub_a, "gap": gap,
                     "delta": st["delta"], "t_p": st["p_value"], "wilcox_p": st["p_wilcoxon"],
                     "ci_disjoint": nhi < ulo})
        print(f"P{int(pct):>4} | {nf_a:>6.3f} [{nlo:.2f},{nhi:.2f}] | {ub_a:>6.3f} [{ulo:.2f},{uhi:.2f}] | "
              f"{gap:>+6.3f} | {st['delta']:>+8.3f} | {st['p_value']:>6.3f} | {st['p_wilcoxon']:>8.3f}{sep}",
              flush=True)

    usable = [r for r in rows if 0.05 <= r["nf"] and r["ub"] <= 0.95]
    best = max(usable or rows, key=lambda r: r["gap"])
    print(f"\n==> PEAK SEPARATION at P{int(best['pct'])}: gap={best['gap']:+.3f} "
          f"(no_forecast {best['nf']:.3f} -> unbiased {best['ub']:.3f}), "
          f"paired Δ={best['delta']:+.3f}, wilcoxon p={best['wilcox_p']:.3f}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
