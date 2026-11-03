"""Unit tests for the analysis & presentation layer (no Bedrock; synthetic data)."""

import json

import numpy as np
import pytest

from vendster.analysis import aggregate as agg_mod
from vendster.analysis import link as link_mod
from vendster.analysis import stats as stats_mod
from vendster.analysis.transcript import render_markdown


def test_wilson_ci_known():
    lo, hi = stats_mod.wilson_ci(5, 10)
    assert lo == pytest.approx(0.237, abs=0.02) and hi == pytest.approx(0.763, abs=0.02)
    assert stats_mod.wilson_ci(0, 0) == (pytest.approx(float("nan"), nan_ok=True),) * 2 \
        or np.isnan(stats_mod.wilson_ci(0, 0)[0])


def test_paired_bootstrap_positive_delta():
    base = [0.4, 0.5, 0.3, 0.45]
    cond = [0.7, 0.8, 0.6, 0.75]                 # uniformly higher -> positive, significant
    r = stats_mod.paired_bootstrap_delta(base, cond, B=2000, seed=1)
    assert r["delta"] == pytest.approx(0.30, abs=0.02)
    assert r["ci95"][0] > 0 and r["p_value"] < 0.05
    assert r["n_baskets"] == 4


def test_paired_bootstrap_handles_single_and_empty():
    assert stats_mod.paired_bootstrap_delta([], [])["n_baskets"] == 0
    assert stats_mod.paired_bootstrap_delta([0.5], [0.7])["delta"] == pytest.approx(0.2)


def test_holm_stepdown():
    # classic Holm example: raw p sorted, adjusted = max running of (m-i)*p, capped at 1
    adj = stats_mod.holm({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04})
    assert adj["a"] == pytest.approx(0.04)      # 4*0.01
    assert adj["b"] == pytest.approx(0.06)      # 3*0.02
    assert adj["c"] == pytest.approx(0.06)      # 2*0.03=0.06, monotone
    assert adj["d"] == pytest.approx(0.06)      # 1*0.04=0.04 -> held at running max 0.06
    assert adj == {k: v for k, v in adj.items() if v <= 1.0}   # all capped
    # NaN passes through, doesn't count toward family size
    a2 = stats_mod.holm({"x": 0.05, "y": float("nan")})
    assert a2["x"] == pytest.approx(0.05) and a2["y"] != a2["y"]


def test_stars():
    assert stats_mod.stars(0.0005) == "***" and stats_mod.stars(0.03) == "*"
    assert stats_mod.stars(0.2) == "" and stats_mod.stars(float("nan")) == ""


def _write_run(tmp_path, layout):
    """Two baskets, 2 conditions (no_forecast, chronos2), attain differs by basket."""
    data = {
        "set_0": {"no_forecast": [1, 0], "chronos2": [1, 1]},
        "set_1": {"no_forecast": [0, 0], "chronos2": [1, 0]},
    }
    for si, conds in data.items():
        d = tmp_path / si / "episodes"
        d.mkdir(parents=True)
        (tmp_path / si / "calibration").mkdir(parents=True)
        (tmp_path / si / "calibration" / "analytic_target.json").write_text(
            json.dumps({"target": 1000.0}))
        for cond, hits in conds.items():
            lines = [json.dumps({"episode_id": f"{cond}_{i}", "status": "ok",
                                 "target_met": bool(h), "agreement": True, "turns": 3,
                                 "total_sales": 1000.0 + 50 * h, "vendor_profit": 300.0,
                                 "sales_efficiency": 0.9, "decision_regret": 50.0,
                                 "first_offer_aggressiveness": 0.2, "concession_rate": 0.1})
                     for i, h in enumerate(hits)]
            (d / f"{cond}.jsonl").write_text("\n".join(lines) + "\n")


def test_aggregate_paired(tmp_path):
    _write_run(tmp_path, "paired")
    agg = agg_mod.aggregate(str(tmp_path))
    assert agg["layout"] == "paired" and agg["n_baskets"] == 2
    nf = agg["conditions"]["no_forecast"]
    assert nf["n_episodes"] == 4 and nf["attain_pooled"] == pytest.approx(0.25)   # 1/4
    ch = agg["conditions"]["chronos2"]
    assert ch["attain_pooled"] == pytest.approx(0.75)                             # 3/4
    # per-basket matrix retained, paired by set
    pb = agg["per_basket"]["no_forecast"]
    assert [r["set_id"] for r in pb] == ["set_0", "set_1"]
    assert pb[0]["attain"] == pytest.approx(0.5) and pb[1]["attain"] == pytest.approx(0.0)


def test_compare_to_baseline_paired(tmp_path):
    _write_run(tmp_path, "paired")
    agg = agg_mod.aggregate(str(tmp_path))
    tests = stats_mod.compare_to_baseline(agg, baseline="no_forecast", B=1000, seed=0)
    assert tests["chronos2"]["delta"] == pytest.approx(0.5)     # (1.0-0.5 + 0.5-0.0)/2


def test_condition_groups_partition():
    real, grid = agg_mod.condition_groups()
    assert "no_forecast" in real and "chronos2" in real
    # 15-cell grid: 8 bias-ladder (|b|<=0.20) + unbiased center + 6-level width ladder
    assert len(grid) == 15 and "no_forecast" not in grid
    assert {"bias_m20", "bias_p20", "unbiased", "width_v050", "width_v200", "width_v025"} <= set(grid)


def test_trend_test_detects_positive_slope():
    # 3 baskets, 3 ordered levels, attainment rises with x in every basket
    xs = [-1.0, 0.0, 1.0]
    mat = [[0.2, 0.5, 0.8], [0.3, 0.5, 0.7], [0.1, 0.4, 0.9]]
    r = stats_mod.trend_test(xs, mat)
    assert r["n_baskets"] == 3 and r["slope"] > 0 and r["p_value"] < 0.05
    # flat → no slope
    flat = stats_mod.trend_test(xs, [[0.5, 0.5, 0.5]] * 3)
    assert abs(flat["slope"]) < 1e-9


def test_ladder_matrix_shapes(tmp_path):
    _write_run(tmp_path, "paired")
    agg = agg_mod.aggregate(str(tmp_path))
    # a 2-level ladder over the two conditions present in the synthetic run
    xs, mat = agg_mod.ladder_matrix(agg, [("no_forecast", 0.0), ("chronos2", 1.0)])
    assert xs == [0.0, 1.0]
    assert len(mat) == 2 and all(len(row) == 2 for row in mat)   # 2 baskets × 2 levels


def test_link_forecaster_level_monotonic():
    quality = {"seasonal_naive": {"relative_wql": 1.0}, "chronos2": {"relative_wql": 0.5},
               "auto_theta": {"relative_wql": 0.8}, "lightgbm_global": {"relative_wql": 0.7}}
    attain = {"seasonal_naive": 0.50, "auto_theta": 0.60,
              "lightgbm_global": 0.63, "chronos2": 0.72}       # better quality -> higher attain
    r = link_mod.forecaster_level(quality, attain)
    assert r["n"] == 4 and r["pearson"] < -0.9 and r["ols_slope"] < 0


def test_render_markdown_shows_turns():
    rec = {"episode_id": "e1", "condition_id": "no_forecast", "set_id": "set_0",
           "target": 1000.0, "target_met": True, "agreement": True, "turns": 2,
           "status": "agreement", "total_sales": 1100.0, "agreed_discounts": {"A": 0.1},
           "communication_log": [
               {"agent": "Buyer", "turn": 1, "type": "proposal", "content": "hi",
                "proposal": {"A": 0.2}},
               {"agent": "Vendor", "turn": 2, "type": "decision", "decision": "accept",
                "content": "ok"}]}
    md = render_markdown(rec)
    assert "PROPOSE A 20%" in md and "DECISION: ACCEPT" in md and "target met: **True**" in md
