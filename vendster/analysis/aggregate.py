"""
Paired-aware aggregation of episode logs

Reads a run directory in either layout:
  * paired  ->  ``<run>/set_*/episodes/*.jsonl``   PRIMARY. K sampled baskets, one target each;
                written by ``scripts/run_experiment.py`` (``runner.run_paired_experiment``) —
                every archived bundle under experiments/ has this layout.
  * single  ->  ``<run>/episodes/*.jsonl``          one dataset, one target; written by
                ``runner.run_single_experiment``, the bring-your-own-DataSource path behind
                ``reference.run_benchmark`` (a BYO source only promises ``load()``, no pool).

and produces, per condition, the full outcome+behavioral suite aggregated **across
baskets**, retaining the per-basket matrix so downstream code can run the paired
within-basket comparison (stats.py) and the basket-level quality<->outcome scatter (link.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.config import reference_conditions
from .stats import wilson_ci

# Outcome/behavioral fields to aggregate: (display, jsonl_key, is_rate)
FIELDS = [
    ("attain", "target_met", True),
    ("agree", "agreement", True),
    ("turns", "turns", False),
    ("sales", "total_sales", False),
    ("profit", "vendor_profit", False),
    ("eff", "sales_efficiency", False),
    ("regret", "decision_regret", False),
    ("first_offer", "first_offer_aggressiveness", False),
    ("concession", "concession_rate", False),
    ("mean_discount", "mean_discount", False),
    ("sales_target", "sales_target", False),
]


# Analysis view of the two perturbation ladders. Each is an ordered set of
# conditions sharing the clean-oracle center ``unbiased``; x = the manipulated scalar.
BIAS_LADDER = [("bias_m20", -0.20), ("bias_m15", -0.15), ("bias_m10", -0.10), ("bias_m05", -0.05),
               ("unbiased", 0.0), ("bias_p05", 0.05), ("bias_p10", 0.10), ("bias_p15", 0.15),
               ("bias_p20", 0.20)]   # realistic range (|bias_log| <= 0.20, real-forecaster p99)
WIDTH_LADDER = [("width_v025", 0.25), ("width_v050", 0.5), ("width_v075", 0.75), ("unbiased", 1.0),
                ("width_v150", 1.5), ("width_v200", 2.0), ("width_v300", 3.0)]   # interval-width multiplier


def ladder_matrix(agg: dict, ladder: list, metric: str = "attain"):
    """Return (xs, K×L matrix) of a metric along an ordered ladder, paired by basket.
    Rows are baskets (aligned by index across conditions), columns the ladder levels."""
    conds = [c for c, _ in ladder]
    xs = [x for _, x in ladder]
    rows = {c: agg["per_basket"].get(c, []) for c in conds}
    K = max((len(r) for r in rows.values()), default=0)
    mat = []
    for i in range(K):
        mat.append([(rows[c][i][metric] if i < len(rows[c]) else float("nan")) for c in conds])
    return xs, mat


def condition_groups(conditions=None):
    """Partition conditions into real-forecaster vs perturbation-grid. Defaults to
    ``reference_conditions()``; pass a custom roster to report a custom benchmark instance."""
    real, grid = [], []
    for c in (conditions if conditions is not None else reference_conditions()):
        fc = getattr(c, "forecaster", None)
        if fc is not None and getattr(fc, "name", None) == "perturbed":
            grid.append(c.id)
        else:
            real.append(c.id)          # no_forecast + real forecasters
    return real, grid


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in recs:                                   # inject computed mechanism traces
        ad = r.get("agreed_discounts")
        r["mean_discount"] = float(np.mean(list(ad.values()))) if ad else None
        t = r.get("target")
        r["sales_target"] = (r.get("total_sales", 0.0) / t) if t else None
    return recs


def load_run(run_dir: str) -> dict:
    """Return {'layout', 'baskets': [{'set_id','target','episodes': {cond: [recs]}}]}."""
    run = Path(run_dir)
    set_dirs = sorted(run.glob("set_*"), key=lambda p: int(p.name.split("_")[1]))
    if set_dirs:
        layout, dirs = "paired", set_dirs
    elif (run / "episodes").exists():
        layout, dirs = "single", [run]
    else:
        raise FileNotFoundError(f"no episodes/ or set_*/ under {run_dir}")
    real, grid = condition_groups()
    conds = real + grid
    baskets = []
    for d in dirs:
        target = None
        for tf in (d / "calibration" / "analytic_target.json",
                   d / "calibration" / "calibration.json"):
            if tf.exists():
                target = json.loads(tf.read_text()).get("target")
                break
        eps = {c: _read_jsonl(d / "episodes" / f"{c}.jsonl") for c in conds}
        baskets.append({"set_id": d.name, "target": target, "episodes": eps})
    return {"layout": layout, "baskets": baskets}


def _rate(recs, key):
    vv = [1.0 if r.get(key) else 0.0 for r in recs if r.get(key) is not None]
    return (float(np.mean(vv)), len(vv), int(sum(vv))) if vv else (float("nan"), 0, 0)


def _mean(recs, key):
    vv = [r[key] for r in recs
          if r.get(key) is not None and not isinstance(r.get(key), bool)]
    return float(np.mean(vv)) if vv else float("nan")


def _ok(recs):
    return [r for r in recs if r.get("status") != "error"]


def per_basket(run: dict, cond: str) -> list:
    """Per-basket dict of every field for one condition (paired rows, basket-ordered)."""
    rows = []
    for b in run["baskets"]:
        recs = _ok(b["episodes"].get(cond, []))
        row = {"set_id": b["set_id"], "n": len(recs)}
        for disp, key, is_rate in FIELDS:
            row[disp] = _rate(recs, key)[0] if is_rate else _mean(recs, key)
        rows.append(row)
    return rows


def aggregate(run_dir: str) -> dict:
    """Full aggregation: per condition, cross-basket means, pooled attainment Wilson CI,
    by-basket mean/SD, and the per-basket matrix (for paired tests + scatter)."""
    run = load_run(run_dir)
    real, grid = condition_groups()
    out = {"layout": run["layout"], "n_baskets": len(run["baskets"]),
           "real": real, "grid": grid, "conditions": {}, "per_basket": {}}
    for cond in real + grid:
        # pooled attainment (episode-level) for a Wilson CI
        pooled = [r for b in run["baskets"] for r in _ok(b["episodes"].get(cond, []))]
        k = sum(1 for r in pooled if r.get("target_met"))
        n = sum(1 for r in pooled if r.get("target_met") is not None)
        lo, hi = wilson_ci(k, n)
        rows = per_basket(run, cond)
        summary = {"n_episodes": n, "n_baskets": sum(1 for r in rows if r["n"] > 0),
                   "attain_pooled": (k / n) if n else float("nan"),
                   "attain_ci95": [lo, hi]}
        for disp, _key, _r in FIELDS:                      # by-basket mean +/- SD
            vals = [r[disp] for r in rows if not np.isnan(r[disp])]
            summary[disp] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                             "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        out["conditions"][cond] = summary
        out["per_basket"][cond] = rows
    return out
