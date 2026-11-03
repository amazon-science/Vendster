"""
Leakage-safe stretch pool for per-episode event resampling.

A *stretch* is a 4-month window of event weeks ``[m1, m1+g, m1+2g, m4]`` (g =
event gap) applied to ALL products for one episode. The pool is the set of
candidate ``m4`` positions that satisfy:

  1. **enough pre-history** — ``m1 >= min_history`` (forecasters need >= a season),
  2. **fixed seasonal phase** — ``m4``'s calendar month is in ``phase_months``
     (so stretches are seasonally comparable — the main distributional confounder),
  3. **strict homogeneity band** — for EVERY product the stretch-window local mean
     is within ``mean_tol`` of the product's global mean AND local CV <= ``cv_ratio``
     × global CV (so all stretches are drawn from one demand regime).

Each stretch carries ``train_cutoff = m1``: forecasters for that stretch train on
weeks ``< m1`` only, so episode target weeks are never in any training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import Dataset


@dataclass
class StretchWindow:
    """One 4-month stretch: event weeks + the leakage-safe training cutoff."""

    m4: int
    event_weeks: list[int]     # [m1, m2, m3, m4]
    train_cutoff: int          # = m1; forecasters train on weeks < m1


def _phase_of_last_week(dataset: Dataset) -> set[int]:
    """Default seasonal phase = the 3 calendar months of the quarter containing the
    final week (so the pool draws one same-season stretch per year)."""
    ex = next(iter(dataset.exog.values()), None)
    n = len(next(iter(dataset.weekly.values())))
    if ex is None or "month" not in ex:
        return set(range(1, 13))
    m = int(ex["month"].to_numpy()[n - 1])
    q_start = ((m - 1) // 3) * 3 + 1
    return {q_start, q_start + 1, q_start + 2}


def build_stretch_pool(
    dataset: Dataset,
    event_gap: int = 4,
    min_history: int = 60,
    phase_months: set[int] | None = None,
    mean_tol: float | None = None,
    cv_ratio: float | None = None,
) -> list[StretchWindow]:
    """Enumerate candidate stretches. Primary control is **fixed seasonal phase**
    (the real confounder); this alone yields a usable pool of same-season windows.

    The per-product-vs-global band (``mean_tol``/``cv_ratio``) is OFF by default:
    with diverse-by-design products (a seasonal one, an erratic one), no window
    keeps all products near their global means at once, so a strict band empties
    the pool — and "local vs global" is the wrong comparison for seasonal series
    anyway (same-season windows should differ from the annual mean). Homogeneity
    is instead reported stretch-to-stretch via :func:`pool_homogeneity`. The band
    remains available as an optional extra filter.
    """
    weekly = dataset.weekly
    products = dataset.products
    n = len(next(iter(weekly.values())))
    if phase_months is None:
        phase_months = _phase_of_last_week(dataset)

    ex = next(iter(dataset.exog.values()), None)
    months = ex["month"].to_numpy() if (ex is not None and "month" in ex) else None

    # Global per-product stats over the full series (regime reference).
    gstats = {}
    for p in products:
        s = weekly[p.item_id].to_numpy(dtype=float)
        gmean = float(s.mean())
        gstats[p.item_id] = (gmean, float(s.std()) / gmean if gmean > 0 else np.inf)

    pool: list[StretchWindow] = []
    for m4 in range(min_history + 3 * event_gap, n):
        m1 = m4 - 3 * event_gap
        if m1 < min_history:
            continue
        if months is not None and int(months[m4]) not in phase_months:
            continue
        ok = True
        for p in products:
            s = weekly[p.item_id].to_numpy(dtype=float)
            span = s[m1:m4 + 1]
            gmean, gcv = gstats[p.item_id]
            lmean = float(span.mean())
            lcv = float(span.std()) / lmean if lmean > 0 else np.inf
            if gmean <= 0:
                ok = False
                break
            if mean_tol is not None and abs(lmean - gmean) / gmean > mean_tol:
                ok = False
                break
            if cv_ratio is not None and lcv > cv_ratio * gcv:
                ok = False
                break
        if ok:
            pool.append(StretchWindow(
                m4=m4,
                event_weeks=[m1, m1 + event_gap, m1 + 2 * event_gap, m4],
                train_cutoff=m1,
            ))
    return pool


def pool_homogeneity(dataset: Dataset, pool: list[StretchWindow]) -> dict:
    """Report evidence the pool is one distribution: per-product dispersion of the
    stretch-window means (low CV across stretches => homogeneous)."""
    report = {"n_stretches": len(pool), "m4_weeks": [w.m4 for w in pool], "per_product": {}}
    for p in dataset.products:
        s = dataset.weekly[p.item_id].to_numpy(dtype=float)
        means = [float(s[w.event_weeks[0]:w.m4 + 1].mean()) for w in pool]
        if means:
            mu = float(np.mean(means))
            report["per_product"][p.item_id] = {
                "stretch_mean_mean": mu,
                "stretch_mean_cv": float(np.std(means) / mu) if mu > 0 else float("nan"),
            }
    return report


def sample_stretch(pool: list[StretchWindow], rng: np.random.Generator) -> StretchWindow:
    """Uniformly sample a stretch from the pool (seeded per episode)."""
    if not pool:
        raise ValueError("empty stretch pool — relax phase_months / band / min_history")
    return pool[int(rng.integers(len(pool)))]
