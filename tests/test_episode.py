"""End-to-end mock-LLM episode tests (zero Bedrock spend)."""

import numpy as np
import pandas as pd
import pytest

from vendster.core.registry import LLM_CLIENTS
from vendster.core.types import Dataset, ProductSpec, TemporalWindows
from vendster.dgp.demand import ElasticityDemandModel
from vendster.dgp.stretch_pool import StretchWindow
from vendster.forecasters.perturbed import PerturbedForecaster
from vendster.negotiation import Episode


def _dataset(n=120):
    prods = [
        ProductSpec("A", "FOODS_3", 12.0, 400, 0.15, -2.30, 0.20, profile="hv"),
        ProductSpec("B", "HOUSEHOLD_1", 18.0, 200, 0.25, -2.97, 0.45, profile="int"),
        ProductSpec("C", "HOBBIES_1", 35.0, 60, 0.30, -2.92, 0.70, profile="dist"),
    ]
    rng = np.random.default_rng(0)
    weekly = {p.item_id: pd.Series(np.abs(rng.normal(p.base_demand_mean, p.base_demand_mean * 0.1, n)))
              for p in prods}
    for p in prods:
        weekly[p.item_id].name = p.item_id
    exog = {p.item_id: pd.DataFrame({"month": [((t // 4) % 12) + 1 for t in range(n)]}) for p in prods}
    windows = TemporalWindows(train_end=80, eval_weeks=list(range(64, 80)), event_weeks=[80, 84, 88, 92])
    return Dataset(products=prods, weekly=weekly, windows=windows, exog=exog)


def _stretch():
    return StretchWindow(m4=92, event_weeks=[80, 84, 88, 92], train_cutoff=80)


def test_mock_episode_no_forecast_reaches_agreement():
    ds = _dataset()
    ep = Episode(
        demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
        dataset=ds, stretch=_stretch(), condition_id="no_forecast", forecaster=None,
        target=5000.0, rng=np.random.default_rng(1), episode_id="t1",
    )
    r = ep.run()
    assert r.agreement and r.status == "agreement"
    assert set(r.agreed_discounts) == {"A", "B", "C"}
    assert all(d == 0.10 for d in r.agreed_discounts.values())     # mock converges to vendor 0.10
    assert r.total_sales > 0 and r.vendor_profit != 0
    assert r.first_offer_aggressiveness == pytest.approx(0.20)     # buyer opened at 0.20
    # concession_rate is None when no agent makes >=2 proposals (mock converges fast)
    assert r.concession_rate is None or r.concession_rate >= 0
    assert r.decision_regret >= 0
    assert 0.0 <= r.sales_efficiency <= 1.0000001
    assert len(r.communication_log) >= 3


def test_mock_episode_with_forecast_builds_buyer_view():
    ds = _dataset()
    fc = PerturbedForecaster(bias_direction=1, bias_magnitude=0.2, variance_mult=0.25,
                             true_weekly=ds.weekly, sigmas={p.item_id: 0.3 for p in ds.products})
    ep = Episode(
        demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
        dataset=ds, stretch=_stretch(), condition_id="over_est_low_var", forecaster=fc,
        target=5000.0, rng=np.random.default_rng(2), show_track_record=True, episode_id="t2",
    )
    r = ep.run()
    assert r.agreement
    # Buyer's turn prompt must contain the forecast + track record; Vendor's must not.
    assert ep.forecast_table is not None and ep.track_record is not None
    buyer_prompt = ep._turn_prompt("Buyer", 1, None, [])
    vendor_prompt = ep._turn_prompt("Vendor", 2, None, [])
    assert "FORECAST (Buyer only)" in buyer_prompt and "TRACK RECORD" in buyer_prompt
    assert "FORECAST (Buyer only)" not in vendor_prompt
    assert "COSTS/MARGINS" in vendor_prompt and "COSTS/MARGINS" not in buyer_prompt


def test_context_snapshot_is_self_contained():
    """The archived context must capture exactly what each side privately saw, so a transcript
    renders in full without re-running the seeded DGP."""
    from vendster.analysis.transcript import render_markdown

    ds = _dataset()
    fc = PerturbedForecaster(bias_direction=-1, bias_magnitude=0.2, variance_mult=1.0,
                             true_weekly=ds.weekly, sigmas={p.item_id: 0.3 for p in ds.products})
    r = Episode(
        demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
        dataset=ds, stretch=_stretch(), condition_id="under_est", forecaster=fc,
        target=5000.0, rng=np.random.default_rng(3), show_track_record=True, episode_id="t3",
    ).run()
    ctx = r.context
    assert ctx["buyer_forecast"] and "FORECAST (Buyer only)" in ctx["buyer_forecast"]
    assert ctx["buyer_track_record"] and "TRACK RECORD" in ctx["buyer_track_record"]
    assert ctx["vendor_margins"] and "COSTS/MARGINS" in ctx["vendor_margins"]
    assert "profit" in ctx["past_events_vendor"] and "profit" not in ctx["past_events_buyer"]
    # rendered transcript embeds the private context, grouped by visibility
    md = render_markdown({**r.__dict__, "communication_log": r.communication_log, "context": ctx})
    assert "CONTEXT (what each side saw)" in md and "BUYER private" in md and "VENDOR private" in md

    # no_forecast episode: no forecast/track-record blocks captured
    r0 = Episode(
        demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
        dataset=ds, stretch=_stretch(), condition_id="no_forecast", forecaster=None,
        target=5000.0, rng=np.random.default_rng(4), episode_id="t4",
    ).run()
    assert r0.context["buyer_forecast"] is None and r0.context["buyer_track_record"] is None


def test_single_draw_realization_varies_with_seed():
    ds = _dataset()
    def sales(seed):
        return Episode(
            demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
            dataset=ds, stretch=_stretch(), condition_id="no_forecast", forecaster=None,
            target=5000.0, rng=np.random.default_rng(seed),
        ).run().total_sales
    assert sales(10) != sales(11)          # single ε draw -> distinct realized outcomes


class _ScriptedLLM:
    """Replays a fixed action sequence, one per turn, regardless of role."""

    def __init__(self, actions):
        self.actions = list(actions)

    def generate(self, system_prompt, messages):
        import json
        return "```json\n" + json.dumps(self.actions.pop(0)) + "\n```"


def test_proposer_cannot_accept_its_own_rejected_proposal():
    ds = _dataset()
    buyer_offer = {"A": 0.30, "B": 0.30, "C": 0.30}
    vendor_offer = {"A": 0.05, "B": 0.05, "C": 0.05}
    script = [
        {"type": "proposal", "content": "open", "proposal": buyer_offer},     # 1 Buyer
        {"type": "decision", "content": "no", "decision": "reject"},          # 2 Vendor
        {"type": "decision", "content": "take it", "decision": "accept"},     # 3 Buyer: nothing standing
        {"type": "proposal", "content": "counter", "proposal": vendor_offer}, # 4 Vendor
        {"type": "decision", "content": "deal", "decision": "accept"},        # 5 Buyer
    ]
    r = Episode(
        demand_model=ElasticityDemandModel(), llm_client=_ScriptedLLM(script),
        dataset=ds, stretch=_stretch(), condition_id="no_forecast", forecaster=None,
        target=5000.0, rng=np.random.default_rng(5), episode_id="t5",
    ).run()
    assert r.status == "agreement" and r.turns == 5
    assert r.agreed_discounts == vendor_offer
