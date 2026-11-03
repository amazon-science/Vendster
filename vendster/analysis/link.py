"""
Forecast-quality <-> negotiation-outcome link

Two forms:
  * forecaster-level : one point per real forecaster, x = pooled quality (relWQL), y = mean attainment.
  * basket-level     : K x (real forecasters) points, x = that basket's-5 forecast quality,
                       y = that basket's attainment for that forecaster (needs per-basket-5 quality).
``no_forecast`` has no forecast, so it is the baseline (a horizontal reference), not a point.
"""

from __future__ import annotations

import numpy as np


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return {"n": int(len(x)), "pearson": float("nan"), "spearman": float("nan"),
                "ols_slope": float("nan"), "ols_intercept": float("nan")}
    try:
        from scipy.stats import pearsonr, spearmanr
        pear = float(pearsonr(x, y)[0]); spear = float(spearmanr(x, y)[0])
    except Exception:
        pear = float(np.corrcoef(x, y)[0, 1]); spear = float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return {"n": int(len(x)), "pearson": pear, "spearman": spear,
            "ols_slope": float(slope), "ols_intercept": float(intercept)}


def forecaster_level(quality: dict, attain: dict, *, metric: str = "relative_wql") -> dict:
    """quality: {forecaster: {metric: value}}; attain: {forecaster: attainment}."""
    pts = []
    for fc, a in attain.items():
        q = quality.get(fc, {}).get(metric)
        if q is not None and a is not None and not np.isnan(a):
            pts.append({"forecaster": fc, "x": float(q), "y": float(a)})
    stat = _corr([p["x"] for p in pts], [p["y"] for p in pts])
    return {"metric": metric, "points": pts, **stat}


def basket_level(points: list, *, metric: str = "relative_wql") -> dict:
    """points: list of {'forecaster','set_id','x'(quality),'y'(attainment)}."""
    stat = _corr([p["x"] for p in points], [p["y"] for p in points])
    return {"metric": metric, "points": points, **stat}


def build_basket_points(agg: dict, quality_by_basket: dict, forecasters: list,
                        metric: str = "relative_wql") -> list:
    """Join per-basket attainment (agg['per_basket'][fc][i]) with per-basket-5 quality
    (quality_by_basket[set_id][fc][metric]) into scatter points."""
    pts = []
    for fc in forecasters:
        rows = agg["per_basket"].get(fc, [])
        for row in rows:
            q = quality_by_basket.get(row["set_id"], {}).get(fc, {}).get(metric)
            if q is not None and not np.isnan(row.get("attain", float("nan"))):
                pts.append({"forecaster": fc, "set_id": row["set_id"],
                            "x": float(q), "y": float(row["attain"])})
    return pts
