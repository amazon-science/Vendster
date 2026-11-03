"""
Per-basket-5 forecast quality — feeds the basket-level quality->outcome join and tab:robust.

Re-derives the SAME K baskets the paired run used (baskets are seeded:
``sample_basket(default_rng(seed*7919 + i))``) and scores each basket's *actual 5 products*
with REAL_FORECASTERS at the single 1-step origin (same Chronos/M5 protocol as the pool census),
normalized to seasonal naive. Output: {set_id: {forecaster: {relative_wql, mase, ...}}}.

    uv run python scripts/eval_quality_baskets.py --sets 8 --seed 0 \
        --out experiments/shared/basket_quality
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from vendster.core.config import REAL_FORECASTERS
from vendster.core.registry import FORECASTERS  # noqa: F401  (registers forecasters on import)
from vendster.data.m5 import M5DataSource
from vendster.experiment.runner import make_forecaster
from vendster.forecasters.metrics import pooled_quality, relative_from_pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=8, help="must match the paired run's --sets")
    ap.add_argument("--seed", type=int, default=0, help="must match the paired run's --seed")
    ap.add_argument("--min-weekly", type=int, default=5)
    ap.add_argument("--panel-size", type=int, default=1500)
    ap.add_argument("--out", default="experiments/shared/basket_quality")
    args = ap.parse_args()

    t0 = time.time()
    ds = M5DataSource(min_weekly=args.min_weekly, panel_size=args.panel_size)
    ds._ensure_loaded()
    origin = ds._windows.train_end
    panel_keys = ds._panel[:args.panel_size]
    panel = {ds._uid(k): ds._series(k) for k in panel_keys}
    print(f"pool={len(ds.pool())}; origin={origin}; scoring {args.sets} baskets x 5 products", flush=True)

    # Capability-driven construction (no name special-casing): panel-trained models get the panel
    # context via their REQUIRES; per-series/zero-shot models get nothing. nf_<model> -> neural_global.
    def make(name, exog):
        ctx = {"panel": panel, "exog": exog, "train_cutoff": origin}
        reg, params = ("neural_global", {"model": name[3:]}) if name.startswith("nf_") else (name, {})
        return make_forecaster(reg, params, ctx)

    quality = {}
    for si in range(args.sets):
        basket = ds.sample_basket(np.random.default_rng(args.seed * 7919 + si))
        eval_series = {ds._uid(k): ds._series(k) for k in basket}
        exog = ds._build_exog(ds._calendar, ds._prices, ds._sales, list(ds._weekly.columns),
                              list(dict.fromkeys(list(basket) + list(panel_keys))))
        naive = pooled_quality(make("seasonal_naive", exog), eval_series, origin)
        row = {}
        for name in REAL_FORECASTERS:
            pq = pooled_quality(make(name, exog), eval_series, origin)
            row[name] = relative_from_pooled(pq, naive)
        quality[f"set_{si}"] = row
        best = min(row, key=lambda k: row[k]["relative_wql"])
        print(f"  set_{si}: best={best} relWQL={row[best]['relative_wql']:.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "quality.json").write_text(json.dumps(quality, indent=2, default=str))
    print(f"\nwrote {out/'quality.json'} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
