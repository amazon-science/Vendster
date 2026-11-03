"""
Name -> implementation registries for the pluggable component interfaces.

Built-in implementations register themselves under stable names when their package is
imported (``vendster.data``, ``vendster.dgp``, ``vendster.forecasters``, ``vendster.llm``,
``vendster.negotiation``).
Extension authors register their own under new names and reference them from a
``BenchmarkConfig`` without editing core modules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple name -> class registry with decorator registration."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: register ``cls`` under ``name`` (must be unique)."""

        def _decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} name already registered: {name!r}")
            self._items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name`` or raise ``KeyError``."""
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind}: {name!r}. Available: {sorted(self._items)}"
            )
        return self._items[name]

    def create(self, name: str, **params) -> T:
        """Instantiate the class registered under ``name`` with ``params``."""
        return self.get(name)(**params)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


# One registry per interface. Import these and use the ``register`` decorator.
DATA_SOURCES: Registry = Registry("DataSource")
DEMAND_MODELS: Registry = Registry("DemandModel")
FORECASTERS: Registry = Registry("Forecaster")
LLM_CLIENTS: Registry = Registry("LLMClient")
METRICS: Registry = Registry("Metric")
SCENARIOS: Registry = Registry("Scenario")
