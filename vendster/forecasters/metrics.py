"""
Forecast-evaluation metric suite (computed on the held-out eval window).

Each metric is registered as a ``Metric`` so it is pluggable, and ``evaluate_suite``
computes the seasonal-naive scale and runs the standard suite over quantile
forecasts. Point metrics use the 0.5 quantile; probabilistic metrics (CRPS/WQL)
use the pinball loss across all provided quantile levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.registry import METRICS

_EPS = 1e-8


def _pinball(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    e = y - pred
    return np.maximum(q * e, (q - 1.0) * e)


def _crps(y: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    """Mean per-point CRPS approximation from quantile levels (2/K * sum pinball)."""
    levels = sorted(quantiles)
    total = np.zeros_like(y, dtype=float)
    for q in levels:
        total += _pinball(y, quantiles[q], q)
    return float((2.0 / len(levels) * total).mean())


def _point(quantiles: dict[float, np.ndarray]) -> np.ndarray:
    return quantiles[0.5]


# --- Metric implementations ---------------------------------------------------

def _register(name):
    def deco(fn):
        @METRICS.register(name)
        class _M:
            pass
        _M.name = name
        _M.compute = staticmethod(lambda y, q, scale=None, _fn=fn: _fn(y, q, scale or {}))
        _M.__name__ = f"Metric_{name}"
        return fn
    return deco


@_register("rmse")
def _rmse(y, q, scale):
    p = _point(q)
    return float(np.sqrt(np.mean((y - p) ** 2)))


@_register("mae")
def _mae(y, q, scale):
    return float(np.mean(np.abs(y - _point(q))))


@_register("mape")
def _mape(y, q, scale):
    p = _point(q)
    mask = np.abs(y) > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])))


@_register("smape")
def _smape(y, q, scale):
    p = _point(q)
    denom = np.abs(y) + np.abs(p)
    mask = denom > _EPS
    return float(np.mean(2.0 * np.abs(y[mask] - p[mask]) / denom[mask])) if mask.any() else float("nan")


@_register("mase")
def _mase(y, q, scale):
    naive_mae = (scale or {}).get("naive_mae")
    if not naive_mae or naive_mae < _EPS:
        return float("nan")
    return float(np.mean(np.abs(y - _point(q))) / naive_mae)


@_register("crps")
def _crps_metric(y, q, scale):
    return _crps(y, q)


@_register("scaled_crps")
def _scaled_crps(y, q, scale):
    naive = (scale or {}).get("naive_mae")   # seasonal-naive CRPS == its MAE (point fc)
    if not naive or naive < _EPS:
        return float("nan")
    return _crps(y, q) / naive


@_register("wql")
def _wql(y, q, scale):
    denom = np.mean(np.abs(y))
    return _crps(y, q) / denom if denom > _EPS else float("nan")


@_register("coverage_80")
def _coverage(y, q, scale):
    lo, hi = q[min(q)], q[max(q)]
    return float(np.mean((y >= lo) & (y <= hi)))


@_register("sharpness")
def _sharpness(y, q, scale):
    return float(np.mean(q[max(q)] - q[min(q)]))


DEFAULT_SUITE = [
    "mase", "mape", "smape", "rmse", "crps", "scaled_crps", "wql",
    "coverage_80", "sharpness",
]


def in_sample_naive_mae(series: pd.Series, up_to: int, season_length: int = 52) -> float:
    """Hyndman MASE denominator: in-sample seasonal-naive MAE over the TRAINING
    history ``[season_length, up_to)`` — a stable per-series scale (NOT the eval
    points). Used for MASE and scaled CRPS.
    """
    a = series.to_numpy(dtype=float)[:up_to]
    if len(a) <= season_length:
        return float("nan")
    diffs = np.abs(a[season_length:] - a[:-season_length])
    return float(np.mean(diffs)) if diffs.size else float("nan")


def evaluate_suite(
    y_true: np.ndarray,
    quantiles: dict[float, np.ndarray],
    naive_scale: float,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Run the metric suite for one series' forecasts. ``naive_scale`` is the
    in-sample seasonal-naive MAE (Hyndman) used for MASE / scaled CRPS.
    """
    names = metrics or DEFAULT_SUITE
    scale = {"naive_mae": naive_scale}
    out = {}
    for name in names:
        metric = METRICS.create(name)
        out[name] = metric.compute(np.asarray(y_true, dtype=float), quantiles, scale)
    return out


def _forecast_at(forecaster, series: pd.Series, origin: int):
    item = series.name if series.name is not None else "series"
    bf = forecaster.predict(item, series.iloc[:origin], [origin])[0]
    return bf


def pooled_quality(
    forecaster,
    series_dict: dict,
    origin: int,
    season_length: int = 52,
    quantile_levels=(0.1, 0.5, 0.9),
) -> dict[str, float]:
    """Chronos/M5 protocol aggregated across MANY series at a single 1-step origin.

    Returns pooled quantities and the summary metrics:
      - ``crps_sum`` / ``absy_sum`` (for pooled WQL and cross-model relative WQL)
      - ``wql`` = crps_sum / absy_sum   (pooled weighted quantile loss)
      - ``mase`` = mean over series of |err| / in-sample naive scale
      - ``coverage_80`` = fraction of series with y in [P10, P90]
      - ``sharpness_norm`` = mean over series of (P90-P10)/scale
      - ``n`` = number of scored series
    Report WQL and CRPS *relative to seasonal naive* via :func:`relative_from_pooled`.
    """
    crps_sum = absy_sum = err_sq = 0.0
    mape, smape, mase, scaled_crps, sharp_norm, sharp_raw, crps_pt, bias_log = \
        [], [], [], [], [], [], [], []
    hits, n = 0, 0
    for s in series_dict.values():
        y = float(s.iloc[origin])
        bf = _forecast_at(forecaster, s, origin)
        p = bf.point
        e = y - p
        if y > _EPS and p > _EPS:
            bias_log.append(float(np.log(p / y)))          # signed log-bias (>0 = over-estimate)
        yq = {0.1: np.array([bf.lower]), 0.5: np.array([p]), 0.9: np.array([bf.upper])}
        c = _crps(np.array([y]), yq)
        crps_sum += c
        crps_pt.append(c)
        absy_sum += abs(y)
        err_sq += e * e
        if abs(y) > _EPS:
            mape.append(abs(e) / abs(y))
        den = abs(y) + abs(p)
        if den > _EPS:
            smape.append(2 * abs(e) / den)
        scale = in_sample_naive_mae(s, origin, season_length)
        if scale and not np.isnan(scale) and scale > 0:
            mase.append(abs(e) / scale)
            scaled_crps.append(c / scale)
            sharp_norm.append((bf.upper - bf.lower) / scale)
        sharp_raw.append(bf.upper - bf.lower)
        hits += int(bf.lower <= y <= bf.upper)
        n += 1
    def m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "mase": m(mase), "mape": m(mape), "smape": m(smape),
        "rmse": float(np.sqrt(err_sq / n)) if n else float("nan"),
        "crps": m(crps_pt), "scaled_crps": m(scaled_crps),
        "wql": crps_sum / absy_sum if absy_sum > 0 else float("nan"),
        "coverage_80": hits / n if n else float("nan"),
        "sharpness": m(sharp_raw), "sharpness_norm": m(sharp_norm),
        "bias_log": m(bias_log),          # mean signed log-bias -> locates fc on the bias ladder
        "crps_sum": crps_sum, "absy_sum": absy_sum, "n": n,
    }


def relative_from_pooled(model: dict, naive: dict) -> dict[str, float]:
    """Full census metric suite, plus relative-WQL vs seasonal naive (<1 beats naive).

    Scale-free / headline: relative_wql, mase, scaled_crps, mape, smape, coverage_80,
    sharpness_norm. Scale-dependent (report but interpret with care across heterogeneous
    series): rmse, crps, wql, sharpness.
    """
    rel_wql = model["crps_sum"] / naive["crps_sum"] if naive["crps_sum"] > 0 else float("nan")
    return {
        "relative_wql": rel_wql,
        "mase": model["mase"], "mape": model["mape"], "smape": model["smape"],
        "rmse": model["rmse"], "crps": model["crps"], "scaled_crps": model["scaled_crps"],
        "wql": model["wql"], "coverage_80": model["coverage_80"],
        "sharpness": model["sharpness"], "sharpness_norm": model["sharpness_norm"],
        "bias_log": model["bias_log"],
        "n": model["n"],
    }


def relative_geomean(model_scores, naive_scores) -> float:
    """Geometric mean of per-group (model / naive) — for aggregating across
    multiple datasets/strata. A failed/undefined pair contributes 1.0 (parity)."""
    ratios = []
    for m, nv in zip(model_scores, naive_scores):
        if m is None or nv is None or np.isnan(m) or np.isnan(nv) or nv <= 0:
            ratios.append(1.0)
        else:
            ratios.append(max(m / nv, 1e-9))
    if not ratios:
        return float("nan")
    return float(np.exp(np.mean(np.log(ratios))))


