"""
Offline forecaster-quality census over the ENVIRONMENT POOL (tab:quality).

Chronos/M5 protocol: single 1-step origin, evaluate across the pool (stable-price
CV<5% + min weekly units >= floor = the population episodes sample 5-baskets from),
full metric suite normalized to seasonal naive. Fast (~hundreds of series).

    uv run python scripts/eval_quality_pool.py --min-weekly 5 --train-panel 1500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendster.core.config import REAL_FORECASTERS
from vendster.core.registry import FORECASTERS  # noqa: F401  (registers forecasters on import)
from vendster.data.m5 import M5DataSource
from vendster.experiment.runner import make_forecaster
from vendster.forecasters.metrics import pooled_quality, relative_from_pooled


def build_inputs(min_weekly: int, train_panel: int):
    ds = M5DataSource(min_weekly=min_weekly, panel_size=train_panel)
    ds._ensure_loaded()
    origin = ds._windows.train_end
    pool_keys = ds._pool
    eval_series = {ds._uid(k): ds._series(k) for k in pool_keys}
    panel_keys = ds._panel[:train_panel]
    panel = {ds._uid(k): ds._series(k) for k in panel_keys}
    exog = ds._build_exog(ds._calendar, ds._prices, ds._sales, list(ds._weekly.columns),
                          list(dict.fromkeys(list(pool_keys) + list(panel_keys))))
    return dict(eval_series=eval_series, panel=panel, exog=exog, origin=origin,
                n_total=ds._weekly.shape[0], n_pool=len(pool_keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-weekly", type=int, default=5)
    ap.add_argument("--train-panel", type=int, default=1500)
    ap.add_argument("--out", default="experiments/shared/forecaster_quality")
    args = ap.parse_args()

    t0 = time.time()
    inp = build_inputs(args.min_weekly, args.train_panel)
    origin, eval_series = inp["origin"], inp["eval_series"]
    print(f"pool (stable-price + min-week>={args.min_weekly}): {inp['n_pool']} series; "
          f"origin={origin} (1-step)")

    # Capability-driven (no name special-casing): nf_<model> is neural_global(model=<model>);
    # every forecaster is fed only the context keys its REQUIRES declares.
    ctx = {"panel": inp["panel"], "exog": inp["exog"], "train_cutoff": origin}

    def make(name):
        reg, params = ("neural_global", {"model": name[3:]}) if name.startswith("nf_") else (name, {})
        return make_forecaster(reg, params, ctx)

    naive = pooled_quality(make("seasonal_naive"), eval_series, origin)
    rows = {}
    cols = ["relative_wql", "mase", "scaled_crps", "mape", "smape", "wql",
            "rmse", "crps", "coverage_80", "sharpness_norm"]
    print("\n" + f"{'forecaster':16}" + "".join(f"{c[:9]:>10}" for c in cols) + f"{'sec':>7}")
    for name in REAL_FORECASTERS:
        t = time.time()
        pq = pooled_quality(make(name), eval_series, origin)
        r = relative_from_pooled(pq, naive)
        r["n_scored"] = pq["n"]
        rows[name] = r
        print(f"{name:16}" + "".join(f"{r[c]:10.3f}" for c in cols) + f"{time.time()-t:7.1f}")

    sn = rows["seasonal_naive"]
    models = {k: v for k, v in rows.items() if k != "seasonal_naive"}
    checks = {
        "naive_relwql_is_1": abs(sn["relative_wql"] - 1.0) < 1e-6,
        "all_models_beat_naive_relwql": all(v["relative_wql"] < 1.0 for v in models.values()),
        "all_models_beat_naive_mase": all(v["mase"] < sn["mase"] for v in models.values()),
        "chronos_best_relwql": rows["chronos2"]["relative_wql"] == min(v["relative_wql"] for v in rows.values()),
    }
    print("\nsanity:", checks)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(
        {"protocol": "chronos_m5_single_origin_1step", "origin": origin,
         "pool": {"n": inp["n_pool"], "min_weekly": args.min_weekly},
         "metrics": rows, "sanity": checks, "elapsed_sec": time.time() - t0},
        indent=2, default=str))
    print(f"\nwrote {out/'report.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
