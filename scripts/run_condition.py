"""
Run ONE condition on K baskets x m episodes at a given target percentile; report pooled
attainment + Wilson 95% CI. Used to (a) tightly measure a single condition (e.g. unbiased),
and (b) sweep the target percentile to see if forecast value grows with target difficulty.

    AWS_PROFILE=<your-bedrock-profile> uv run python scripts/run_condition.py \
        --cond unbiased --baskets 8 --episodes 20 --percentile 80 --out experiments/spot_unbiased
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
from vendster.analysis.stats import wilson_ci
from vendster.core.config import reference_conditions
from vendster.llm import add_llm_args, make_llm
from vendster.data.m5 import M5DataSource
from vendster.dgp.demand import ElasticityDemandModel
from vendster.dgp.stretch_pool import build_stretch_pool
from vendster.experiment.runner import _run_condition, analytic_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    add_llm_args(ap, default="bedrock")
    ap.add_argument("--baskets", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--percentile", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-weekly", type=int, default=5)
    ap.add_argument("--panel-size", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=24,
                    help="parallel episodes per basket")
    ap.add_argument("--set-concurrency", type=int, default=8,
                    help="parallel baskets; total in-flight ~= set-concurrency * concurrency "
                         "(measured safe up to ~192 on TPM=6M/RPM=10K)")
    ap.add_argument("--out", default="experiments/cond_check")
    args = ap.parse_args()

    t0 = time.time()
    src = M5DataSource(min_weekly=args.min_weekly, panel_size=args.panel_size)
    cond = next(c for c in reference_conditions() if c.id == args.cond)
    llm = make_llm(args.model, args.model_id)
    dm = ElasticityDemandModel()
    print(f"cond={args.cond}  P{args.percentile:.0f}  {args.baskets} baskets x {args.episodes} eps",
          flush=True)

    def one_basket(si):
        basket = src.sample_basket(np.random.default_rng(args.seed * 7919 + si))
        dataset = src.build_dataset(basket, cost_seed=si)
        target = analytic_target(dataset, seed=args.seed * 100 + si,
                                 percentile=args.percentile)["target"]
        pool = build_stretch_pool(dataset, min_history=60)
        recs = _run_condition(cond, dataset, pool, target, llm, args.episodes,
                              args.seed * 100 + si, Path(args.out) / f"set_{si}", {}, True, dm,
                              args.concurrency)
        ok = [r for r in recs if r.get("target_met") is not None]
        bk = sum(1 for r in ok if r["target_met"])
        attain = bk / len(ok) if ok else float("nan")
        print(f"  set_{si}: target=${target:,.0f}  attain={attain:.2f}", flush=True)
        return si, bk, len(ok), attain

    idx = list(range(args.baskets))
    sc = max(1, min(args.set_concurrency, args.baskets))
    if sc > 1 and args.baskets > 1:
        with ThreadPoolExecutor(max_workers=sc) as ex:
            out_rows = list(ex.map(one_basket, idx))
    else:
        out_rows = [one_basket(si) for si in idx]
    out_rows.sort(key=lambda r: r[0])
    per = [r[3] for r in out_rows]
    k = sum(r[1] for r in out_rows)
    n = sum(r[2] for r in out_rows)

    lo, hi = wilson_ci(k, n)
    print(f"\nPOOLED {args.cond} @ P{args.percentile:.0f}: {k/n:.3f} ({k}/{n})  "
          f"Wilson95 [{lo:.3f}, {hi:.3f}]", flush=True)
    print(f"per-basket: {[round(p,2) for p in per]}  mean±sd {np.mean(per):.3f}±{np.std(per):.3f}",
          flush=True)
    print(f"DONE in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
