"""
Stable extension interfaces for Vendster.

These are the seams that make Vendster an adaptable benchmark: implement one of
these Protocols, register it (see ``vendster.core.registry``), and reference it
by name from a ``BenchmarkConfig`` — no changes to negotiation or harness code.

The negotiation *identity* — roles, agent objectives, asymmetric information, and
success criterion — is a ``Scenario`` (see below): register your own and reference it
by name from ``BenchmarkConfig.scenario``. The turn engine, discount action space, and
DGP realization are the shared *arena* (see ``vendster.negotiation.episode`` /
``vendster.core.task_core``) that all scenarios run on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .types import BaseForecast, Dataset, ProductSpec


@runtime_checkable
class DataSource(Protocol):
    """Produces the products, weekly demand history, and temporal windows."""

    name: str

    def load(self) -> Dataset:
        """Return a fully resolved :class:`Dataset`."""
        ...


@runtime_checkable
class DemandModel(Protocol):
    """Maps a product + discount to expected and realized units."""

    name: str

    def expected_units(self, product: ProductSpec, base_demand: float, discount: float) -> float:
        """Deterministic expected units at ``discount`` (no noise)."""
        ...

    def sample_units(
        self,
        product: ProductSpec,
        base_demand: float,
        discount: float,
        rng: np.random.Generator,
    ) -> float:
        """A single stochastic realization of units at ``discount``."""
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Predicts base demand for one or more target weeks.

    ``history`` is the weekly series strictly *before* the first target week
    (real forecasters must not see the target weeks). Perturbation forecasters
    may ignore ``history`` and derive forecasts from the true series supplied at
    construction. Returns one :class:`BaseForecast` per requested target week.
    """

    name: str

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Generates an agent response given a system prompt and message list."""

    name: str

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        ...


@runtime_checkable
class Scenario(Protocol):
    """The negotiation *identity* — swappable via ``BenchmarkConfig.scenario``.

    A Scenario defines WHO negotiates and toward WHAT, plus who sees what private
    information; the shared arena (``Episode``) drives the turn loop, retries, transcript,
    and DGP realization. Implement this Protocol, register under a name, and reference it by
    name — no changes to the arena or harness. ``VendsterScenario`` is the reference
    buyer–vendor discount negotiation.
    """

    name: str
    roles: tuple[str, ...]
    opening_role: str

    def next_role(self, role: str) -> str:
        """The role that acts after ``role`` (turn alternation)."""
        ...

    def system_prompt(self, role: str, target: float | None) -> str:
        """The role's objective/system prompt (``target`` may be None for calibration)."""
        ...

    def role_blocks(self, role: str, episode) -> list[str]:
        """Role-specific (asymmetric) context blocks injected into the turn prompt —
        e.g. the forecast for the Buyer, private margins for the Vendor. Uses the
        ``episode``'s shared block renderers."""
        ...

    def target_met(self, total_sales: float, target: float | None) -> bool | None:
        """Success criterion for the outcome (None when there is no target)."""
        ...


@runtime_checkable
class Metric(Protocol):
    """A forecast-evaluation metric over aligned actuals and quantile forecasts.

    ``quantiles`` maps quantile level (e.g. 0.1, 0.5, 0.9) to an array aligned
    with ``y_true``. Point metrics use the 0.5 quantile; probabilistic metrics
    use the full set. ``scale`` carries normalizers computed by the evaluator
    (e.g. ``{"naive_mae": ...}`` for MASE / scaled CRPS); metrics that do not need
    it ignore it. Lower is better for all built-in metrics except coverage, which
    targets its nominal level.
    """

    name: str

    def compute(
        self,
        y_true: np.ndarray,
        quantiles: dict[float, np.ndarray],
        scale: dict | None = None,
    ) -> float:
        ...
