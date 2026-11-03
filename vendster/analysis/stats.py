"""
Statistical inference for the paired design.

Conditions share the same K baskets, so a condition is compared to ``no_forecast`` via
**within-basket paired attainment deltas with a cluster (basket) bootstrap** — not a naive
pooled two-proportion test, which would ignore basket correlation and understate uncertainty.
"""

from __future__ import annotations

import math

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    """Wilson score 95% CI for a binomial proportion (no scipy dependency)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_bootstrap_delta(baseline: list, cond: list, *, B: int = 10000,
                           seed: int = 0) -> dict:
    """Cluster-bootstrap the mean within-basket attainment delta (cond - baseline).

    ``baseline`` and ``cond`` are per-basket attainment rates, paired by basket index
    (same order). Resamples baskets with replacement; reports the observed mean delta, a
    percentile 95% CI, and a two-sided bootstrap p-value against 0.
    """
    base = np.asarray([b for b in baseline], dtype=float)
    cnd = np.asarray([c for c in cond], dtype=float)
    mask = ~(np.isnan(base) | np.isnan(cnd))
    base, cnd = base[mask], cnd[mask]
    m = len(base)
    if m == 0:
        return {"delta": float("nan"), "ci95": [float("nan"), float("nan")],
                "p_value": float("nan"), "n_baskets": 0}
    deltas = cnd - base
    obs = float(np.mean(deltas))
    if m == 1:
        return {"delta": obs, "ci95": [float("nan"), float("nan")],
                "p_value": float("nan"), "n_baskets": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(B, m))
    boot = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return {"delta": obs, "ci95": [float(lo), float(hi)],
            "p_value": min(1.0, p), "n_baskets": m}


def paired_ttest_delta(baseline: list, cond: list) -> dict:
    """PRIMARY test — basket is the unit of analysis. Paired t-test on the K per-basket
    attainment deltas (cond - baseline) against 0, with a Student-t CI. Well-calibrated at
    the small cluster counts (K) this design uses, where the percentile bootstrap is
    anti-conservative. Also returns Wilcoxon signed-rank p as a nonparametric sensitivity."""
    base = np.asarray(baseline, float); cnd = np.asarray(cond, float)
    mask = ~(np.isnan(base) | np.isnan(cnd))
    d = cnd[mask] - base[mask]
    m = len(d)
    if m == 0:
        return {"delta": float("nan"), "ci95": [float("nan"), float("nan")],
                "p_value": float("nan"), "p_wilcoxon": float("nan"), "n_baskets": 0}
    obs = float(np.mean(d))
    if m == 1 or np.allclose(d, d[0]):
        return {"delta": obs, "ci95": [float("nan"), float("nan")],
                "p_value": float("nan"), "p_wilcoxon": float("nan"), "n_baskets": m}
    from scipy.stats import t as _t
    from scipy.stats import ttest_1samp, wilcoxon
    se = float(np.std(d, ddof=1) / math.sqrt(m))
    tcrit = float(_t.ppf(0.975, m - 1))
    p = float(ttest_1samp(d, 0.0).pvalue)
    try:
        pw = float(wilcoxon(d).pvalue)
    except Exception:
        pw = float("nan")
    return {"delta": obs, "ci95": [obs - tcrit * se, obs + tcrit * se],
            "p_value": p, "p_wilcoxon": pw, "n_baskets": m}


def trend_test(xs: list, mat: list) -> dict:
    """PRIMARY dose-response test (basket-clustered). Fit a per-basket slope of the metric
    on the ordered level ``xs`` (least squares over the L levels), then a one-sample t-test
    on the K basket slopes against 0. This is the confirmatory test for the perturbation
    ladders (bias, width) and mirrors the paired t-test (basket = unit of analysis)."""
    x = np.asarray(xs, float)
    slopes = []
    for row in mat:
        y = np.asarray(row, float)
        ok = ~np.isnan(y)
        if ok.sum() >= 2 and np.ptp(x[ok]) > 0:
            slopes.append(float(np.polyfit(x[ok], y[ok], 1)[0]))
    slopes = np.asarray(slopes, float)
    K = len(slopes)
    if K == 0:
        return {"slope": float("nan"), "ci95": [float("nan"), float("nan")],
                "p_value": float("nan"), "n_baskets": 0}
    mean = float(np.mean(slopes))
    if K == 1 or np.allclose(slopes, slopes[0]):
        return {"slope": mean, "ci95": [float("nan"), float("nan")],
                "p_value": float("nan"), "n_baskets": K}
    from scipy.stats import t as _t
    from scipy.stats import ttest_1samp
    se = float(np.std(slopes, ddof=1) / math.sqrt(K))
    tcrit = float(_t.ppf(0.975, K - 1))
    return {"slope": mean, "ci95": [mean - tcrit * se, mean + tcrit * se],
            "p_value": float(ttest_1samp(slopes, 0.0).pvalue), "n_baskets": K}


def compare_to_baseline(agg: dict, baseline: str = "no_forecast", *, metric: str = "attain",
                        method: str = "ttest", B: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of every condition vs ``baseline`` on ``metric``.

    ``method='ttest'`` (default, PRIMARY) = paired t-test on per-basket deltas (calibrated at
    small K). ``method='bootstrap'`` = cluster bootstrap (robustness; anti-conservative at
    small K). Attaches the bootstrap CI/p under ``bootstrap`` for reference either way.
    """
    pb = agg["per_basket"]
    base_vals = [r[metric] for r in pb.get(baseline, [])]
    result = {}
    for cond, rows in pb.items():
        if cond == baseline:
            continue
        vals = [r[metric] for r in rows]
        primary = (paired_ttest_delta(base_vals, vals) if method == "ttest"
                   else paired_bootstrap_delta(base_vals, vals, B=B, seed=seed))
        primary["bootstrap"] = paired_bootstrap_delta(base_vals, vals, B=B, seed=seed)
        result[cond] = primary
    return result


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down adjustment within a family. Input {key: raw p}; returns
    {key: adjusted p} controlling family-wise error (uniformly more powerful than Bonferroni,
    same guarantee). NaN p-values pass through as NaN and don't count toward the family size."""
    items = sorted(((k, v) for k, v in pvalues.items() if v == v), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)          # enforce monotonic non-decreasing
        adj[k] = min(1.0, running)
    for k, v in pvalues.items():
        if v != v:
            adj[k] = float("nan")
    return adj


def stars(p: float) -> str:
    """Significance markers for tables."""
    if p != p:                       # NaN
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
