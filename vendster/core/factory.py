"""
Resolve a :class:`BenchmarkConfig` into concrete, instantiated components.

The factory is the single place that turns declarative names into live objects
via the registries. It performs no I/O beyond what the components themselves do.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import registry
from .config import BenchmarkConfig
from .interfaces import DataSource, DemandModel, LLMClient


@dataclass
class Benchmark:
    """A resolved benchmark: live components plus the originating config.

    Forecasters are NOT built here — the experiment harness builds them per stretch via the
    capability-driven ``runner.make_forecaster`` (which injects the panel/true-series context a
    forecaster declares in ``REQUIRES``). Keeping a single construction path avoids drift.
    """

    config: BenchmarkConfig
    data_source: DataSource
    demand_model: DemandModel
    llm_client: LLMClient


def build_benchmark(config: BenchmarkConfig) -> Benchmark:
    """Resolve every component referenced by ``config`` into a :class:`Benchmark`."""
    data_source = registry.DATA_SOURCES.create(
        config.data_source.name, **config.data_source.params
    )
    demand_model = registry.DEMAND_MODELS.create(
        config.demand_model.name, **config.demand_model.params
    )
    llm_client = registry.LLM_CLIENTS.create(
        config.llm_client.name, **config.llm_client.params
    )
    return Benchmark(
        config=config,
        data_source=data_source,
        demand_model=demand_model,
        llm_client=llm_client,
    )
