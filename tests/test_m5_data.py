"""Unit tests for M5 data-source pure helpers (no download required)."""

import numpy as np
import pandas as pd
import pytest

from vendster.data import m5


def test_assign_elasticity():
    # single overall Bijmolt mean (-2.6) applied to every category now
    assert m5.assign_elasticity("FOODS") == -2.6
    assert m5.assign_elasticity("HOUSEHOLD") == -2.6
    assert m5.assign_elasticity("HOBBIES") == -2.6
    assert m5.assign_elasticity("UNKNOWN") == m5.OVERALL_ELASTICITY
    assert m5.assign_elasticity("FOODS", override=-4.0) == -4.0


def test_assign_cost_mix_is_2_2_1_and_seeded():
    a = m5.assign_cost_mix(np.random.default_rng(0))
    b = m5.assign_cost_mix(np.random.default_rng(0))
    assert a == b                                  # deterministic under seed
    assert sorted(a) == sorted(m5.COST_MIX)        # a permutation of the 2/2/1 mix
    assert a.count(0.20) == 2 and a.count(0.45) == 2 and a.count(0.70) == 1


def test_price_cv_and_stable_filter():
    prices = pd.DataFrame({
        "store_id": ["CA_1"] * 4 + ["CA_1"] * 4,
        "item_id": ["A"] * 4 + ["B"] * 4,
        "wm_yr_wk": [1, 2, 3, 4] * 2,
        "sell_price": [10.0, 10.0, 10.0, 10.0] + [10.0, 12.0, 8.0, 14.0],
    })
    cv = m5.price_cv(prices)
    assert cv.loc[("CA_1", "A")] == pytest.approx(0.0)
    assert cv.loc[("CA_1", "B")] > 0.05
    stable = m5.stable_price_items(prices, threshold=0.05)
    assert ("CA_1", "A") in stable and ("CA_1", "B") not in stable


def test_series_stats_mean_cv_min():
    weekly = pd.DataFrame(
        [[100, 100, 100, 100], [0, 10, 20, 10]],
        index=["X", "Y"], columns=[1, 2, 3, 4],
    )
    s = m5.series_stats(weekly)
    assert s.loc["X", "mean"] == pytest.approx(100)
    assert s.loc["X", "cv"] == pytest.approx(0.0)
    assert s.loc["X", "min"] == 100
    assert s.loc["Y", "min"] == 0          # min-week floor would exclude Y at any X>=1


def test_min_week_pool_logic():
    # pool = min weekly units >= floor (consistently sold every week)
    weekly = pd.DataFrame(
        [[10, 8, 12, 6], [3, 0, 5, 4], [50, 40, 60, 55]],
        index=["keep", "drop_zero_week", "keep_high"], columns=[1, 2, 3, 4],
    )
    mins = m5.series_stats(weekly)["min"]
    pool = set(mins.index[mins >= 5])
    assert pool == {"keep", "keep_high"}   # drop_zero_week has a 0 week -> excluded


def test_temporal_windows():
    w = m5.temporal_windows(n_weeks=100, eval_len=16, event_gap=4)
    assert w.event_weeks == [87, 91, 95, 99]
    assert len(w.eval_weeks) == 16 and w.train_end == 71
    w.validate()
    with pytest.raises(ValueError):
        m5.temporal_windows(n_weeks=20, eval_len=16)


def test_m5_registered():
    from vendster.core.registry import DATA_SOURCES
    assert "m5" in DATA_SOURCES
