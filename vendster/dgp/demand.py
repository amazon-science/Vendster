"""
Data-generating process for Vendster.

The reference demand model is constant-elasticity in discount and log-normal in noise:

    Units(d) = base_demand * (1 - d)^(-|beta|) * eps,   eps ~ LogNormal(-s^2/2, s^2)

Constant-elasticity (textbook) demand: a 1% price cut raises quantity by |beta|%. For
elastic goods (|beta|>1) revenue is monotone-increasing in the discount, so the discount
is a real lever on sales (a linear ``1+|beta|d`` approximation instead plants a spurious
revenue peak mid-grid). The ``-s^2/2`` drift keeps
``E[Units(d)] == base_demand * (1 - d)^(-|beta|)`` so the elasticity term is the *expected*
response and the noise is mean-preserving. A single ``eps`` is drawn per product per episode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..core.registry import DEMAND_MODELS
from ..core.types import Dataset, ProductSpec


def cv_to_sigma(cv: float) -> float:
    """Log-normal sigma that reproduces a target coefficient of variation."""
    return math.sqrt(math.log(1.0 + cv * cv))


@DEMAND_MODELS.register("elasticity")
class ElasticityDemandModel:
    """Reference multiplicative-elasticity, log-normal-noise demand model."""

    name = "elasticity"

    def __init__(self, sigma_cap: float | None = None):
        # Optional cap on log-normal sigma (extreme-CV series like intermittent
        # items produce very wide draws). Default None = fully faithful to CV.
        self.sigma_cap = sigma_cap

    def _sigma(self, product: ProductSpec) -> float:
        s = product.noise_sigma
        if s is None:
            s = cv_to_sigma(product.base_demand_cv)
        if self.sigma_cap is not None:
            s = min(s, self.sigma_cap)
        return s

    def expected_units(self, product: ProductSpec, base_demand: float, discount: float) -> float:
        # constant-elasticity: a 1% price cut raises quantity by |beta|% (price = list*(1-d))
        return base_demand * (1.0 - discount) ** (-abs(product.elasticity))

    def sample_units(
        self,
        product: ProductSpec,
        base_demand: float,
        discount: float,
        rng: np.random.Generator,
    ) -> float:
        mu = self.expected_units(product, base_demand, discount)
        s = self._sigma(product)
        z = rng.standard_normal()
        return max(0.0, mu * math.exp(s * z - 0.5 * s * s))


# --- Event realization helpers -----------------------------------------------

def realize_event(
    demand_model,
    dataset: Dataset,
    discounts: dict[str, float],
    event_week: int,
    rng: np.random.Generator,
) -> dict:
    """Realize one event: single-draw units, sales, and profit per product.

    ``discounts`` maps item_id -> agreed discount. ``event_week`` indexes into
    each product's aligned weekly series to obtain the true base demand.
    """
    per_product = {}
    total_sales = total_profit = 0.0
    for p in dataset.products:
        base = float(dataset.weekly[p.item_id].iloc[event_week])
        d = float(discounts[p.item_id])
        units = demand_model.sample_units(p, base, d, rng)
        price = p.price * (1.0 - d)
        sales = units * price
        profit = units * (price - p.unit_cost)
        per_product[p.item_id] = {
            "discount": d, "base_demand": base, "units": units,
            "sales": sales, "profit": profit,
        }
        total_sales += sales
        total_profit += profit
    return {"week": event_week, "products": per_product,
            "total_sales": total_sales, "total_profit": total_profit}


def generate_past_events(
    demand_model,
    dataset: Dataset,
    rng: np.random.Generator,
    discount_levels: Sequence[float],
    n_events: int = 3,
) -> list[dict]:
    """Generate months 1..n_events past events (open history shown to both agents).

    Each event assigns a random past discount per product from ``discount_levels``
    and realizes units via the same DGP as the negotiation target.
    """
    weeks = dataset.windows.event_weeks[:n_events]
    events = []
    for i, wk in enumerate(weeks):
        discounts = {p.item_id: float(rng.choice(discount_levels)) for p in dataset.products}
        ev = realize_event(demand_model, dataset, discounts, wk, rng)
        ev["month"] = i + 1
        events.append(ev)
    return events
