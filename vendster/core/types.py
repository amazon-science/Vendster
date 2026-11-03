"""
Core data containers shared across Vendster components.

These are deliberately lightweight, dependency-light dataclasses so that
extension authors can construct them without importing heavy modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ProductSpec:
    """Static product metadata plus derived negotiation parameters.

    A ``DataSource`` produces one ``ProductSpec`` per product. Demand history
    lives separately in ``Dataset.weekly`` keyed by ``item_id``.
    """

    item_id: str
    department: str
    price: float
    base_demand_mean: float
    base_demand_cv: float
    elasticity: float            # beta; stored as a signed value, DGP uses |beta|
    unit_cost_ratio: float       # unit_cost / price -> negotiation dynamic
    profile: str = ""            # demand-profile label (e.g. "high_vol_low_var")
    category: str | None = None
    noise_sigma: float | None = None  # log-normal sigma; derived from CV if None

    @property
    def unit_cost(self) -> float:
        """Per-unit cost implied by ``unit_cost_ratio``."""
        return self.unit_cost_ratio * self.price


@dataclass
class TemporalWindows:
    """Disjoint week-index partitions of a product's weekly series.

    Indices are positions into each product's weekly series (all products share
    the same aligned weekly index). ``event_weeks`` holds exactly 4 entries:
    months 1-3 (open history) followed by month 4 (the negotiation target).
    """

    train_end: int              # exclusive end index of the training region
    eval_weeks: list[int]       # held-out evaluation window (reported metrics only)
    event_weeks: list[int]      # months 1-4 event weeks, len == 4

    def validate(self) -> None:
        """Raise ``ValueError`` if windows overlap or are mis-ordered."""
        if len(self.event_weeks) != 4:
            raise ValueError(f"event_weeks must have length 4, got {len(self.event_weeks)}")
        eval_set = set(self.eval_weeks)
        event_set = set(self.event_weeks)
        if eval_set & event_set:
            raise ValueError("eval_weeks and event_weeks overlap")
        if any(w < self.train_end for w in self.eval_weeks):
            raise ValueError("eval_weeks must lie after train_end")
        if self.eval_weeks and min(self.event_weeks) <= max(self.eval_weeks):
            raise ValueError("event_weeks must lie after the eval window")


@dataclass
class Dataset:
    """A fully resolved benchmark dataset: products, weekly history, windows.

    ``train_panel`` (optional) holds a larger sample of aligned weekly series used
    to fit *global* forecasters (e.g. the M5-style global LightGBM). It includes
    the selected products plus additional same-distribution series.
    """

    products: list[ProductSpec]
    weekly: dict[str, pd.Series]   # item_id -> weekly units (aligned integer index)
    windows: TemporalWindows
    meta: dict = field(default_factory=dict)
    train_panel: dict[str, pd.Series] = field(default_factory=dict)
    # Optional known-covariate frames per item_id (aligned weekly index): columns
    # such as log_price, snap, event, month, woy, dept, cat, state. Consumed by
    # global forecasters; known for all weeks incl. the target (M5 covariates).
    exog: dict[str, pd.DataFrame] = field(default_factory=dict)

    def product(self, item_id: str) -> ProductSpec:
        for p in self.products:
            if p.item_id == item_id:
                return p
        raise KeyError(f"Unknown item_id: {item_id}")


@dataclass
class BaseForecast:
    """A base-demand forecast for a single event week (pre elasticity scaling).

    ``lower``/``upper`` are the P10/P90 bounds of the 80% prediction interval.
    """

    point: float
    lower: float
    upper: float

    def scaled(self, abs_beta: float, discount: float) -> BaseForecast:
        """Return the forecast at a discount level via the shared elasticity term.

        The identical constant-elasticity factor ``(1 - discount)^(-|beta|)`` is applied to
        point and both interval bounds, so forecasters differ only in base-demand accuracy.
        MUST match the DGP response in :meth:`ElasticityDemandModel.expected_units`.
        """
        factor = (1.0 - discount) ** (-abs_beta)
        return BaseForecast(self.point * factor, self.lower * factor, self.upper * factor)
