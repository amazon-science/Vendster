"""Framework-seam tests: registration, process-stable seeds, the leakage guard, capability-driven
forecaster construction, and the swappable negotiation Scenario."""

import hashlib

import numpy as np
import pandas as pd
import pytest

import vendster.forecasters  # noqa: F401  (register forecasters)
from vendster.core.registry import FORECASTERS, LLM_CLIENTS, SCENARIOS
from vendster.core.types import Dataset, ProductSpec, TemporalWindows
from vendster.dgp.demand import ElasticityDemandModel
from vendster.dgp.stretch_pool import StretchWindow
from vendster.experiment.runner import _SEED_BASE, _episode_seed, make_forecaster
from vendster.negotiation import Episode, VendsterScenario


def _dataset(n=120):
    prods = [ProductSpec("A", "FOODS_3", 12.0, 400, 0.15, -2.30, 0.20, profile="hv"),
             ProductSpec("B", "FOODS_3", 18.0, 200, 0.25, -2.60, 0.30, profile="int")]
    rng = np.random.default_rng(0)
    weekly = {p.item_id: pd.Series(np.abs(rng.normal(p.base_demand_mean, p.base_demand_mean * 0.1, n)))
              for p in prods}
    for p in prods:
        weekly[p.item_id].name = p.item_id
    windows = TemporalWindows(train_end=80, eval_weeks=list(range(64, 80)), event_weeks=[80, 84, 88, 92])
    return Dataset(products=prods, weekly=weekly, windows=windows)


# --- registration ---------------------------------------------------------------------
def test_all_reference_forecasters_registered():
    for name in ["seasonal_naive", "auto_ets", "auto_theta", "lightgbm_global",
                 "chronos2", "chronos_bolt", "timesfm25", "neural_global", "perturbed"]:
        assert name in FORECASTERS, name
    assert "vendster" in SCENARIOS


# --- reproducibility: the episode seed must be process-stable (not builtin salted hash) -----
def test_episode_seed_is_process_stable():
    tag = "calib"
    th = int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=4).digest(), "big")
    assert _episode_seed(7, tag, 3) == (7 * _SEED_BASE + th % 100000 + 3) % (2 ** 32)
    # identical inputs -> identical seed within a process too
    assert _episode_seed(2, "chronos2", 5) == _episode_seed(2, "chronos2", 5)


# --- leakage guard: must RAISE (not a stripped assert) when the cutoff follows an event ------
def test_leakage_guard_raises_on_bad_cutoff():
    ds = _dataset()
    bad = StretchWindow(m4=92, event_weeks=[80, 84, 88, 92], train_cutoff=85)  # 85 > min event 80
    with pytest.raises(ValueError, match="leakage"):
        Episode(demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
                dataset=ds, stretch=bad, condition_id="x", forecaster=None, target=1.0,
                rng=np.random.default_rng(0))


# --- capability seam: REQUIRES-driven injection, no name special-casing ----------------------
def test_make_forecaster_injects_only_declared_context():
    ctx = {"panel": {}, "exog": {}, "train_cutoff": 80, "true_weekly": {}, "sigmas": {}}
    # per-series model declares nothing -> gets no context kwargs
    assert make_forecaster("seasonal_naive", {}, ctx).name == "seasonal_naive"
    # perturbed declares (true_weekly, sigmas) -> injected
    p = make_forecaster("perturbed", {"bias_direction": -1, "bias_magnitude": 0.30}, ctx)
    assert p.s == -1 and p.bias_magnitude == 0.30


# --- scenario seam: default reproduces Vendster; a custom scenario swaps the identity ---------
def test_vendster_scenario_defaults():
    sc = VendsterScenario()
    assert sc.roles == ("Buyer", "Vendor") and sc.next_role("Buyer") == "Vendor"
    assert "SALES target" in sc.system_prompt("Buyer", 5000.0)
    assert sc.target_met(6000.0, 5000.0) is True and sc.target_met(10.0, None) is None


def test_custom_scenario_is_pluggable():
    @SCENARIOS.register("test_flip")
    class FlipScenario(VendsterScenario):
        name = "test_flip"
        def target_met(self, total_sales, target):     # invert the success criterion
            return None if target is None else total_sales < target

    sc = SCENARIOS.create("test_flip")
    assert sc.target_met(100.0, 5000.0) is True         # "success" = below target now
    ds = _dataset()
    ep = Episode(demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
                 dataset=ds, stretch=StretchWindow(m4=92, event_weeks=[80, 84, 88, 92], train_cutoff=80),
                 condition_id="x", forecaster=None, target=5000.0, rng=np.random.default_rng(1),
                 scenario=sc)
    r = ep.run()
    assert r.status in ("agreement", "max_turns")       # runs under the swapped scenario
