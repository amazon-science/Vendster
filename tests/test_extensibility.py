"""Bring-your-own-context: a custom DataSource driven purely via registry + config."""

import numpy as np
import pandas as pd

from vendster import dgp, forecasters, llm  # noqa: F401  register components
from vendster.core.config import BenchmarkConfig, ComponentSpec, reference_conditions
from vendster.core.factory import build_benchmark
from vendster.core.registry import DATA_SOURCES
from vendster.core.types import Dataset, ProductSpec, TemporalWindows
from vendster.experiment import analytic_target, run_single_experiment


@DATA_SOURCES.register("toy_test")
class ToyDataSource:
    """A non-M5 context defined entirely through the public interface."""

    name = "toy_test"

    def __init__(self, n_weeks: int = 200):
        self.n_weeks = n_weeks

    def load(self) -> Dataset:
        n = self.n_weeks
        rng = np.random.default_rng(0)
        prods = [
            ProductSpec("X", "FOODS_3", 10.0, 300, 0.2, -2.30, 0.20),
            ProductSpec("Y", "HOUSEHOLD_1", 20.0, 150, 0.3, -2.97, 0.50),
        ]
        weekly = {}
        for p in prods:
            s = pd.Series(np.abs(rng.normal(p.base_demand_mean, p.base_demand_mean * 0.1, n)))
            s.name = p.item_id
            weekly[p.item_id] = s
        exog = {p.item_id: pd.DataFrame({"month": [((t // 4) % 12) + 1 for t in range(n)]}) for p in prods}
        windows = TemporalWindows(80, list(range(64, 80)), [80, 84, 88, 92])
        return Dataset(products=prods, weekly=weekly, windows=windows, exog=exog, train_panel=weekly)


def test_byo_datasource_runs_via_config_and_registry(tmp_path):
    # Build an instance from a config that references the custom source BY NAME.
    cfg = BenchmarkConfig(
        version="toy-v1",
        data_source=ComponentSpec("toy_test"),
        demand_model=ComponentSpec("elasticity"),
        llm_client=ComponentSpec("mock"),
        conditions=reference_conditions(),
    )
    bench = build_benchmark(cfg)                       # resolves all components via registries
    assert bench.data_source.name == "toy_test"
    dataset = bench.data_source.load()

    # Analytic target through the resolved (custom) components — no core edits, no LLM.
    calib = analytic_target(dataset, n=50, seed=0, demand_model=bench.demand_model)
    assert calib["n"] == 50 and calib["target"] >= 0.0
    assert calib["D0_summary"]["min"] <= calib["target"] <= calib["D0_summary"]["max"]

    # Experiment halts after Wave 1 (min_spread unreachable) → exercises the harness
    # on a BYO context without needing panel/GPU forecasters.
    summ = run_single_experiment(dataset, bench.llm_client, calib["target"], episodes_per_condition=1,
                          seed=0, demand_model=bench.demand_model, out_dir=str(tmp_path / "r"),
                          min_spread=99.0)
    assert summ["checkpoint"]["passed"] is False
    assert "no_forecast" in summ["conditions"]


def test_run_benchmark_honors_negotiation_params_and_fixed_target(tmp_path):
    import json

    from vendster.core.config import ConditionSpec, NegotiationParams
    from vendster.reference import run_benchmark

    cfg = BenchmarkConfig(
        version="toy-v1",
        data_source=ComponentSpec("toy_test"),
        demand_model=ComponentSpec("elasticity"),
        llm_client=ComponentSpec("mock"),
        conditions=[ConditionSpec("no_forecast", forecaster=None, is_bracketing=True)],
        negotiation=NegotiationParams(max_turns=1),
        sales_target=1234.0,
        n_episodes=2,
    )
    out = run_benchmark(cfg, out_dir=str(tmp_path / "b"), checkpoint_enabled=False)
    assert out["target"] == 1234.0
    recs = [json.loads(l) for l in (tmp_path / "b/run/episodes/no_forecast.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    assert all(r["turns"] == 1 and r["status"] == "max_turns" and r["target"] == 1234.0 for r in recs)
