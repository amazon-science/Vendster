"""Vendster core: extension interfaces, registries, config, and factory."""

from . import task_core
from .config import (
    BenchmarkConfig,
    ComponentSpec,
    ConditionSpec,
    NegotiationParams,
    reference_conditions,
)
from .factory import Benchmark, build_benchmark
from .interfaces import DataSource, DemandModel, Forecaster, LLMClient, Metric
from .registry import (
    DATA_SOURCES,
    DEMAND_MODELS,
    FORECASTERS,
    LLM_CLIENTS,
    METRICS,
    Registry,
)
from .types import BaseForecast, Dataset, ProductSpec, TemporalWindows

__all__ = [
    "task_core",
    "BenchmarkConfig",
    "ComponentSpec",
    "ConditionSpec",
    "NegotiationParams",
    "reference_conditions",
    "Benchmark",
    "build_benchmark",
    "DataSource",
    "DemandModel",
    "Forecaster",
    "LLMClient",
    "Metric",
    "DATA_SOURCES",
    "DEMAND_MODELS",
    "FORECASTERS",
    "LLM_CLIENTS",
    "METRICS",
    "Registry",
    "BaseForecast",
    "Dataset",
    "ProductSpec",
    "TemporalWindows",
]
