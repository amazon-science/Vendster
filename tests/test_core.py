"""Tests for the extension architecture: registry, config, factory, task core."""

import pytest

from vendster import core
from vendster.core import task_core
from vendster.core.config import BenchmarkConfig, ComponentSpec, reference_conditions
from vendster.core.factory import build_benchmark
from vendster.core.registry import Registry
from vendster.core.types import BaseForecast, ProductSpec, TemporalWindows

# --- Registry -----------------------------------------------------------------

def test_register_and_resolve_by_name():
    reg: Registry = Registry("Widget")

    @reg.register("gadget")
    class Gadget:
        def __init__(self, size=1):
            self.size = size

    assert "gadget" in reg
    assert reg.get("gadget") is Gadget
    inst = reg.create("gadget", size=3)
    assert inst.size == 3
    assert reg.names() == ["gadget"]


def test_duplicate_registration_raises():
    reg: Registry = Registry("Widget")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(ValueError):
        @reg.register("dup")
        class B:
            pass


def test_unknown_name_raises():
    reg: Registry = Registry("Widget")
    with pytest.raises(KeyError):
        reg.get("missing")
    with pytest.raises(KeyError):
        reg.create("missing")


# --- Factory: config -> instance ----------------------------------------------

def _minimal_config(monkeypatch) -> BenchmarkConfig:
    """Register throwaway stub components and build a config referencing them.

    Uses ``monkeypatch`` to swap in fresh registries so the stubs never leak into
    (or collide with) the real built-in registrations from other modules.
    """
    ds = Registry("DataSource")
    dm = Registry("DemandModel")
    llm = Registry("LLMClient")

    # Temporarily replace the module-level registries the factory reads.
    monkeypatch.setattr(core.registry, "DATA_SOURCES", ds)
    monkeypatch.setattr(core.registry, "DEMAND_MODELS", dm)
    monkeypatch.setattr(core.registry, "LLM_CLIENTS", llm)
    monkeypatch.setattr(core.registry, "FORECASTERS", Registry("Forecaster"))
    monkeypatch.setattr(core.registry, "METRICS", Registry("Metric"))

    @ds.register("stub_data")
    class StubData:
        name = "stub_data"

        def __init__(self, note="x"):
            self.note = note

    @dm.register("stub_demand")
    class StubDemand:
        name = "stub_demand"

    @llm.register("stub_llm")
    class StubLLM:
        name = "stub_llm"

    @core.registry.FORECASTERS.register("perturbed")
    class StubPerturbed:
        name = "perturbed"

        def __init__(self, bias_direction=0, bias_magnitude=0.2, variance_mult=0.25):
            self.bias_direction = bias_direction
            self.bias_magnitude = bias_magnitude
            self.variance_mult = variance_mult

    return BenchmarkConfig(
        version="test-v1",
        data_source=ComponentSpec("stub_data", {"note": "hi"}),
        demand_model=ComponentSpec("stub_demand"),
        llm_client=ComponentSpec("stub_llm"),
        conditions=reference_conditions(),
    )


def test_build_benchmark_resolves_components(monkeypatch):
    cfg = _minimal_config(monkeypatch)
    bench = build_benchmark(cfg)
    assert bench.data_source.note == "hi"
    assert bench.demand_model.name == "stub_demand"
    assert bench.llm_client.name == "stub_llm"


def test_make_forecaster_capability_injection(monkeypatch):
    """Forecasters are built via the capability-driven runner.make_forecaster, which injects only
    the context keys a forecaster CLASS declares in REQUIRES — no name special-casing."""
    from vendster.experiment.runner import make_forecaster
    cfg = _minimal_config(monkeypatch)
    ctx = {"true_weekly": {}, "sigmas": {}}          # perturbed's declared REQUIRES
    assert cfg.condition("no_forecast").forecaster is None
    over_spec = cfg.condition("bias_p20").forecaster
    over = make_forecaster(over_spec.name, dict(over_spec.params), ctx)
    assert over.s == 1 and over.bias_magnitude == 0.20
    width_spec = cfg.condition("width_v050").forecaster
    width050 = make_forecaster(width_spec.name, dict(width_spec.params), ctx)
    assert width050.s == 0 and width050.variance_mult == 0.5


def test_make_forecaster_unknown_name_raises():
    from vendster.experiment.runner import make_forecaster
    with pytest.raises(KeyError):
        make_forecaster("does_not_exist", {}, {})


# --- Reference conditions -----------------------------------------------------

def test_reference_conditions_shape():
    conds = reference_conditions()
    ids = [c.id for c in conds]
    assert len(conds) == 26          # 11 episode real + 15-cell perturbation grid
    assert {"no_forecast", "seasonal_naive", "auto_ets", "auto_theta", "lightgbm_global",
            "chronos2", "chronos_bolt", "timesfm25",
            "nf_nhits", "nf_patchtst", "nf_deepar"} <= set(ids)
    # bracketing = spread anchors: no_forecast + strongest under/over bias + clean oracle
    bracketing = [c.id for c in conds if c.is_bracketing]
    assert set(bracketing) == {"no_forecast", "bias_m20", "bias_p20", "unbiased"}
    # perturbation grid = 15 cells (8 bias-ladder |b|<=0.20 + unbiased + 6-level width ladder)
    perturbed = [c for c in conds if c.forecaster and c.forecaster.name == "perturbed"]
    assert len(perturbed) == 15


# --- Task core & types --------------------------------------------------------

def test_task_core_frozen_constants():
    assert task_core.DISCOUNT_LEVELS == (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    assert task_core.OPENING_ROLE == "Buyer"
    assert task_core.REFERENCE_VERSION == "vendster-m5-v1"


def test_temporal_windows_validation():
    ok = TemporalWindows(train_end=100, eval_weeks=[100, 101, 102], event_weeks=[110, 114, 118, 122])
    ok.validate()
    with pytest.raises(ValueError):
        TemporalWindows(train_end=100, eval_weeks=[100, 110], event_weeks=[110, 114, 118, 122]).validate()
    with pytest.raises(ValueError):
        TemporalWindows(train_end=100, eval_weeks=[100], event_weeks=[110, 114, 118]).validate()


def test_product_unit_cost_and_forecast_scaling():
    p = ProductSpec(
        item_id="P1", department="FOODS", price=12.0, base_demand_mean=420,
        base_demand_cv=0.15, elasticity=-2.30, unit_cost_ratio=0.20,
    )
    assert p.unit_cost == pytest.approx(2.40)
    fc = BaseForecast(point=100, lower=80, upper=120)
    scaled = fc.scaled(abs_beta=2.30, discount=0.10)
    factor = (1.0 - 0.10) ** (-2.30)          # constant-elasticity
    assert scaled.point == pytest.approx(100 * factor)
    assert scaled.lower == pytest.approx(80 * factor)
    assert scaled.upper == pytest.approx(120 * factor)
