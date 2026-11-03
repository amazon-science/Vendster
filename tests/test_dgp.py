"""Tests for the elasticity demand model and event realization."""

import math

import numpy as np
import pandas as pd
import pytest

from vendster.core.types import Dataset, ProductSpec, TemporalWindows
from vendster.dgp.demand import (
    ElasticityDemandModel,
    cv_to_sigma,
    generate_past_events,
    realize_event,
)
from vendster.dgp.stretch_pool import build_stretch_pool, pool_homogeneity, sample_stretch


def _product(cv=0.3, beta=-2.30, price=10.0, ucr=0.4):
    return ProductSpec(
        item_id="P", department="FOODS_3", price=price, base_demand_mean=100,
        base_demand_cv=cv, elasticity=beta, unit_cost_ratio=ucr,
    )


def test_cv_to_sigma():
    assert cv_to_sigma(0.0) == 0.0
    assert cv_to_sigma(0.5) == pytest.approx(math.sqrt(math.log(1.25)))


def test_expected_units_monotonic_in_discount():
    m = ElasticityDemandModel()
    p = _product()
    vals = [m.expected_units(p, 100.0, d) for d in (0.05, 0.10, 0.20, 0.30)]
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(100 * (1 - 0.05) ** (-2.30))   # constant-elasticity


def test_sample_mean_and_cv_recovered():
    m = ElasticityDemandModel()
    p = _product(cv=0.3)
    rng = np.random.default_rng(0)
    draws = np.array([m.sample_units(p, 100.0, 0.10, rng) for _ in range(20000)])
    expected = m.expected_units(p, 100.0, 0.10)
    assert draws.mean() == pytest.approx(expected, rel=0.03)   # mean-preserving noise
    assert draws.std() / draws.mean() == pytest.approx(0.3, rel=0.06)


def test_single_draw_deterministic_under_seed():
    m = ElasticityDemandModel()
    p = _product()
    a = m.sample_units(p, 100.0, 0.10, np.random.default_rng(42))
    b = m.sample_units(p, 100.0, 0.10, np.random.default_rng(42))
    assert a == b


def _dataset():
    products = [_product()]
    weekly = {"P": pd.Series([100.0] * 30)}
    windows = TemporalWindows(train_end=10, eval_weeks=list(range(10, 20)),
                              event_weeks=[22, 24, 26, 28])
    return Dataset(products=products, weekly=weekly, windows=windows)


def test_realize_event_shapes():
    m = ElasticityDemandModel()
    d = _dataset()
    ev = realize_event(m, d, {"P": 0.10}, event_week=28, rng=np.random.default_rng(1))
    pr = ev["products"]["P"]
    assert pr["units"] >= 0
    # sales = units * discounted price ; profit uses unit_cost = 0.4*10 = 4
    assert pr["sales"] == pytest.approx(pr["units"] * 10.0 * 0.9)
    assert pr["profit"] == pytest.approx(pr["units"] * (10.0 * 0.9 - 4.0))
    assert ev["total_sales"] == pytest.approx(pr["sales"])


def _pool_dataset(n=1000, spike=False):
    a = np.full(n, 100.0)
    if spike:
        a[500:513] = 300.0                        # localized anomaly (diluted in a long series)
    weekly = {"P": pd.Series(a)}
    exog = {"P": pd.DataFrame({"month": [((t // 4) % 12) + 1 for t in range(n)]})}
    windows = TemporalWindows(train_end=60, eval_weeks=list(range(60, 76)), event_weeks=[80, 84, 88, 92])
    return Dataset(products=[_product()], weekly=weekly, windows=windows, exog=exog)


def test_stretch_pool_homogeneous_all_pass():
    ds = _pool_dataset()
    pool = build_stretch_pool(ds, event_gap=4, min_history=60, phase_months=set(range(1, 13)))
    assert len(pool) > 0
    for w in pool:
        assert w.train_cutoff == w.event_weeks[0] == w.m4 - 12          # leakage cutoff = m1
        assert w.event_weeks == [w.m4 - 12, w.m4 - 8, w.m4 - 4, w.m4]   # spaced by gap


def test_stretch_pool_band_excludes_spike():
    ds = _pool_dataset(spike=True)          # anomaly at weeks [500,512], global ~ stable
    m4s = {w.m4 for w in build_stretch_pool(ds, event_gap=4, min_history=60,
                                            phase_months=set(range(1, 13)),
                                            mean_tol=0.10, cv_ratio=1.25)}  # opt-in band
    assert 512 not in m4s                   # span [500,512] fully anomalous -> excluded
    assert 200 in m4s                       # span [188,200] fully normal -> included


def test_stretch_pool_fixed_phase():
    ds = _pool_dataset()
    months = ds.exog["P"]["month"].to_numpy()
    pool = build_stretch_pool(ds, event_gap=4, min_history=60, phase_months={3})
    assert pool and all(int(months[w.m4]) == 3 for w in pool)


def test_sample_stretch_deterministic_and_homogeneity_report():
    ds = _pool_dataset()
    pool = build_stretch_pool(ds, event_gap=4, min_history=60, phase_months=set(range(1, 13)))
    a = sample_stretch(pool, np.random.default_rng(7))
    b = sample_stretch(pool, np.random.default_rng(7))
    assert a.m4 == b.m4
    rep = pool_homogeneity(ds, pool)
    assert rep["n_stretches"] == len(pool)
    assert "P" in rep["per_product"]


def test_generate_past_events_uses_months_1_to_3():
    m = ElasticityDemandModel()
    d = _dataset()
    events = generate_past_events(m, d, np.random.default_rng(2),
                                  discount_levels=[0.05, 0.10, 0.20], n_events=3)
    assert [e["month"] for e in events] == [1, 2, 3]
    assert [e["week"] for e in events] == [22, 24, 26]   # first 3 event weeks
    for e in events:
        assert e["products"]["P"]["discount"] in (0.05, 0.10, 0.20)
