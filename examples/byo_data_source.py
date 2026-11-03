"""
Bring-your-own-context example: run Vendster on a NON-M5 dataset by implementing
one interface (`DataSource`), registering it, and building an `Episode` on its data —
without touching any core module. To run it through a `BenchmarkConfig` instead, reference
the registered name (see EXTENDING.md).

    uv run python examples/byo_data_source.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendster import dgp, forecasters, llm  # noqa: F401  register built-ins
from vendster.core.registry import DATA_SOURCES, LLM_CLIENTS
from vendster.core.types import Dataset, ProductSpec, TemporalWindows
from vendster.dgp.demand import ElasticityDemandModel
from vendster.dgp.stretch_pool import build_stretch_pool, sample_stretch
from vendster.negotiation import Episode


# 1) Implement + register your own DataSource (here: a tiny synthetic electronics shop).
@DATA_SOURCES.register("my_shop")
class MyShopDataSource:
    name = "my_shop"

    def load(self) -> Dataset:
        n, rng = 160, np.random.default_rng(7)
        products = [
            ProductSpec("fan", "APPLIANCE", 40.0, 220, 0.25, -2.5, 0.35),
            ProductSpec("tv", "ELECTRONICS", 300.0, 60, 0.40, -3.0, 0.55),
            ProductSpec("cable", "ACCESSORY", 8.0, 500, 0.20, -2.0, 0.25),
        ]
        weekly = {}
        for p in products:
            s = pd.Series(np.abs(rng.normal(p.base_demand_mean, p.base_demand_mean * 0.15, n)))
            s.name = p.item_id
            weekly[p.item_id] = s
        exog = {p.item_id: pd.DataFrame({"month": [((t // 4) % 12) + 1 for t in range(n)]})
                for p in products}
        windows = TemporalWindows(80, list(range(64, 80)), [80, 84, 88, 92])
        return Dataset(products=products, weekly=weekly, windows=windows, exog=exog, train_panel=weekly)


def main():
    ds = DATA_SOURCES.create("my_shop").load()          # resolve by name
    stretch = sample_stretch(build_stretch_pool(ds, min_history=52), np.random.default_rng(0))
    ep = Episode(
        demand_model=ElasticityDemandModel(),
        llm_client=LLM_CLIENTS.create("mock"),          # swap to "bedrock" for a real model
        dataset=ds, stretch=stretch, condition_id="no_forecast", forecaster=None,
        target=8000.0, rng=np.random.default_rng(0), episode_id="byo",
    )
    r = ep.run()
    print(f"BYO episode on 'my_shop': status={r.status} turns={r.turns} "
          f"agreed={ {k: round(v, 2) for k, v in r.agreed_discounts.items()} }")
    print(f"  realized sales=${r.total_sales:,.0f}  target=${r.target:,.0f}  met={r.target_met}")
    print("Adapted Vendster to a new context via one DataSource + the registry — zero core edits.")


if __name__ == "__main__":
    main()
