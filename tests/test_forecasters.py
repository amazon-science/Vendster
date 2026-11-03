"""Unit tests for metrics, the perturbation grid, and elasticity-scaled presentation."""
import math

import numpy as np
import pandas as pd
import pytest

from vendster.core.types import BaseForecast, Dataset, ProductSpec, TemporalWindows
from vendster.forecasters import metrics as M
from vendster.forecasters.perturbed import PerturbedForecaster
from vendster.forecasters.presentation import build_event_forecast


def test_perturbation_bias_direction():
    true = pd.Series([100.0] * 30)
    tw = {"P": true}
    sig = {"P": 0.3}
    over = PerturbedForecaster(1, 0.20, 0.25, tw, sig).predict("P", true, [10])[0]
    none = PerturbedForecaster(0, 0.20, 0.25, tw, sig).predict("P", true, [10])[0]
    under = PerturbedForecaster(-1, 0.20, 0.25, tw, sig).predict("P", true, [10])[0]
    assert over.point > none.point > under.point
    assert none.point == pytest.approx(100.0)
    assert over.point == pytest.approx(100.0 * math.exp(0.20))    # log-bias: 122.14
    assert under.point == pytest.approx(100.0 * math.exp(-0.20))  # reciprocal: 81.87


def test_perturbation_variance_widens_interval():
    true = pd.Series([100.0] * 30)
    tw, sig = {"P": true}, {"P": 0.3}
    lo = PerturbedForecaster(0, 0.20, 0.25, tw, sig).predict("P", true, [10])[0]
    hi = PerturbedForecaster(0, 0.20, 1.00, tw, sig).predict("P", true, [10])[0]
    assert (hi.upper - hi.lower) > (lo.upper - lo.lower)


def test_identical_elasticity_scaling_across_products():
    # two products, same beta -> same discount factor applied to each base forecast
    p1 = ProductSpec("A", "FOODS_3", 10.0, 100, 0.2, -2.30, 0.2)
    p2 = ProductSpec("B", "FOODS_3", 20.0, 200, 0.2, -2.30, 0.2)
    true = {"A": pd.Series([100.0] * 30), "B": pd.Series([200.0] * 30)}
    ds = Dataset([p1, p2], true, TemporalWindows(5, list(range(5, 15)), [16, 18, 20, 22]))
    f = PerturbedForecaster(0, 0.0, 0.25, true, {"A": 0.2, "B": 0.2})
    table = build_event_forecast(f, ds, event_week=22, discount_levels=[0.10, 0.20])
    factor10 = (1 - 0.10) ** (-2.30)          # constant-elasticity
    assert table["A"][0.10].point == pytest.approx(100 * factor10)
    assert table["B"][0.10].point == pytest.approx(200 * factor10)
    # same relative bump for both products
    assert (table["A"][0.20].point / table["A"][0.10].point) == pytest.approx(
        table["B"][0.20].point / table["B"][0.10].point)


# --- metrics ------------------------------------------------------------------

def _quantiles(point, lo, hi):
    return {0.1: np.array(lo, float), 0.5: np.array(point, float), 0.9: np.array(hi, float)}


def test_point_metrics():
    y = np.array([100.0, 100.0, 100.0])
    q = _quantiles([90, 110, 100], [80, 90, 90], [100, 130, 110])
    assert M._rmse(y, q, {}) == pytest.approx(np.sqrt((100 + 100 + 0) / 3))
    assert M._mae(y, q, {}) == pytest.approx((10 + 10 + 0) / 3)
    assert M._mape(y, q, {}) == pytest.approx((0.10 + 0.10 + 0.0) / 3)


def test_coverage_and_sharpness():
    y = np.array([95.0, 200.0])         # first inside [80,100], second outside [90,110]
    q = _quantiles([90, 100], [80, 90], [100, 110])
    assert M._coverage(y, q, {}) == pytest.approx(0.5)
    assert M._sharpness(y, q, {}) == pytest.approx(((100 - 80) + (110 - 90)) / 2)


def test_mase_uses_scale():
    y = np.array([100.0, 100.0])
    q = _quantiles([90, 110], [80, 90], [100, 130])
    assert M._mase(y, q, {"naive_mae": 20.0}) == pytest.approx(10.0 / 20.0)
    assert np.isnan(M._mase(y, q, {"naive_mae": 0.0}))


def test_crps_positive_and_scaled():
    y = np.array([100.0, 120.0, 80.0])
    q = _quantiles([100, 110, 90], [80, 90, 70], [120, 130, 110])
    crps = M._crps_metric(y, q, {})
    assert crps > 0
    assert M._scaled_crps(y, q, {"naive_mae": crps}) == pytest.approx(1.0)


def test_in_sample_naive_mae():
    # ramp 0..59: seasonal (52) diffs are all 52; in-sample MAE over training = 52
    actual = pd.Series([float(i) for i in range(60)])
    assert M.in_sample_naive_mae(actual, up_to=60, season_length=52) == pytest.approx(52.0)
    # too little history for a full season -> nan
    assert np.isnan(M.in_sample_naive_mae(actual, up_to=40, season_length=52))


def test_relative_geomean():
    # ratios 0.5 and 2.0 -> geometric mean 1.0; failed pair -> 1.0
    assert M.relative_geomean([1.0, 4.0], [2.0, 2.0]) == pytest.approx(1.0)
    assert M.relative_geomean([1.0], [2.0]) == pytest.approx(0.5)
    assert M.relative_geomean([float("nan")], [2.0]) == pytest.approx(1.0)   # failed -> parity
    assert M.relative_geomean([1.0], [0.0]) == pytest.approx(1.0)            # bad scale -> parity


def test_metric_definitions_hand_calc():
    # pinball (standard quantile loss): y=10, yhat=8, q=0.9 -> 0.9*2 = 1.8
    assert M._pinball(np.array([10.0]), np.array([8.0]), 0.9)[0] == pytest.approx(1.8)
    # CRPS approx = (2/K) * sum_q pinball_q. y=10, q={0.1:6, 0.5:9, 0.9:12}
    #   rho_.1=0.1*4=0.4 ; rho_.5=0.5*1=0.5 ; rho_.9=(0.9-1)*(-2)=0.2 -> sum 1.1 -> (2/3)*1.1
    q = {0.1: np.array([6.0]), 0.5: np.array([9.0]), 0.9: np.array([12.0])}
    assert M._crps(np.array([10.0]), q) == pytest.approx((2 / 3) * 1.1)
    # WQL = sum(crps)/sum(|y|) (GluonTS mean_wQuantileLoss). One point above:
    assert M._wql(np.array([10.0]), q, {}) == pytest.approx((2 / 3) * 1.1 / 10.0)
    # MASE = mean|err| / in-sample seasonal-naive MAE (Hyndman). err |10-9|=1, scale=2 -> 0.5
    assert M._mase(np.array([10.0]), q, {"naive_mae": 2.0}) == pytest.approx(0.5)


def test_pooled_full_suite_keys_and_relwql():
    # tiny 2-series pooled check: relative_wql = model crps_sum / naive crps_sum
    import pandas as pd

    class ConstFC:  # forecaster returning a fixed point/interval regardless of history
        def __init__(self, pt, lo, hi): self.pt, self.lo, self.hi = pt, lo, hi
        def predict(self, item, history, tw): return [BaseForecast(self.pt, self.lo, self.hi)]

    series = {}
    for name, vals in {"a": [10.0] * 60 + [10.0], "b": [20.0] * 60 + [20.0]}.items():
        s = pd.Series(vals); s.name = name; series[name] = s
    origin = 60
    model = M.pooled_quality(ConstFC(10, 8, 12), {"a": series["a"]}, origin, season_length=52)
    r = M.relative_from_pooled(model, model)   # vs itself -> relative_wql == 1
    for k in ["relative_wql", "mase", "mape", "smape", "rmse", "crps", "scaled_crps",
              "wql", "coverage_80", "sharpness", "sharpness_norm"]:
        assert k in r
    assert r["relative_wql"] == pytest.approx(1.0)


def test_metrics_registered():
    from vendster.core.registry import METRICS
    for name in M.DEFAULT_SUITE:
        assert name in METRICS
