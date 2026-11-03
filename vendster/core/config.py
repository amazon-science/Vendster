"""
Declarative benchmark configuration.

A ``BenchmarkConfig`` composes a runnable benchmark instance from component
*names* (resolved via the registries) plus the fixed-core negotiation params and
experiment settings. The frozen reference instance ``Vendster-M5`` is just a
specific ``BenchmarkConfig`` (assembled in the reference-instance module).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import task_core


@dataclass
class ComponentSpec:
    """A registry name plus constructor params for one pluggable component."""

    name: str
    params: dict = field(default_factory=dict)


@dataclass
class NegotiationParams:
    """Negotiation protocol settings passed to every ``Episode`` (defaults come from ``task_core``).
    The opening role belongs to the ``Scenario``. The number of past events is fixed by the
    stretch layout (months 1-3 before the month-4 event), so it is not a parameter."""

    discount_levels: tuple[float, ...] = task_core.DISCOUNT_LEVELS
    max_turns: int = task_core.MAX_TURNS
    show_track_record: bool = True
    max_action_retries: int = 3

    def episode_kwargs(self) -> dict:
        return {"discount_levels": self.discount_levels, "max_turns": self.max_turns,
                "show_track_record": self.show_track_record, "max_retries": self.max_action_retries}


@dataclass
class ConditionSpec:
    """One experimental condition: an id plus its forecaster (None = no-forecast)."""

    id: str
    forecaster: ComponentSpec | None = None
    is_bracketing: bool = False


@dataclass
class BenchmarkConfig:
    """Everything needed to construct and run a benchmark instance."""

    version: str
    data_source: ComponentSpec
    demand_model: ComponentSpec
    llm_client: ComponentSpec
    conditions: list[ConditionSpec]
    # the negotiation identity (roles/objectives/asymmetric info/success); default = the reference
    # buyer–vendor discount negotiation. Swap by registering a Scenario and naming it here.
    scenario: ComponentSpec = field(default_factory=lambda: ComponentSpec("vendster"))
    negotiation: NegotiationParams = field(default_factory=NegotiationParams)
    n_episodes: int = 200
    sales_target: float | None = None      # fixed target; None = analytic P{target_percentile}
    target_percentile: float = 0.80      # P80 operating point
    seed: int = 0

    def condition(self, condition_id: str) -> ConditionSpec:
        for c in self.conditions:
            if c.id == condition_id:
                return c
        raise KeyError(f"Unknown condition: {condition_id}")


# --- Reference condition set (the 26 conditions of Vendster-M5) ----------------

# Perturbation grid. Two orthogonal MAIN-EFFECT dose-response ladders, both centered
# on the CLEAN oracle (unbiased point, calibrated interval), crossed only at that center (no bias x
# width interaction cells — powered for the two main effects, not the difference-of-differences):
#   * BIAS ladder (point-level error) @ calibrated width: signed log-bias, 8 nonzero levels
#     +/-{0.05,0.10,0.15,0.20}. This spans the EMPIRICALLY-REALISTIC range: real forecasters' signed
#     log-bias reaches |bias_log| ~= 0.20 at the p99 (max 0.26; mean +0.02, per-basket quality census), so every
#     rung is a bias a real forecaster actually exhibits on some baskets. `point = true*exp(s*mag)`
#     (log-symmetric). (Out-of-distribution biases |b|>=0.30 — which no real forecaster reaches —
#     and the associated distrust/self-correction dynamics are deferred to future work.)
#   * WIDTH/calibration ladder (interval error) @ zero bias: variance_mult over a 12x range
#     {0.25,0.5,0.75,1.0,1.5,2.0,3.0} = interval half-width multiplier -> overconfident ...
#     CALIBRATED (1.0, matches DGP noise) ... underconfident.
# Within this realistic range the bias dose-response is monotone, so the basket-clustered linear
# TREND is the primary test (confirmatory at K=16); per-condition contrasts vs no_forecast support it.
_BIAS_LADDER = (-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)   # log-bias @ calibrated width
_CAL_VAR = 1.0                                           # calibrated (~80% coverage, matches DGP)
# interval-width (variance) ladder @ zero bias; 1.0 = the unbiased/calibrated center (not repeated).
_WIDTH_LADDER = (0.25, 0.5, 0.75, 1.5, 2.0, 3.0)


def _perturbed(bias_value: float, variance_mult: float) -> ComponentSpec:
    """A grid cell from a signed log-bias value and an interval-width multiplier."""
    direction = 0 if bias_value == 0 else (1 if bias_value > 0 else -1)
    return ComponentSpec(
        name="perturbed",
        params={
            "bias_direction": direction,           # -1 under, 0 none, +1 over
            "bias_magnitude": abs(bias_value),
            "variance_mult": variance_mult,
        },
    )


def _bias_id(bias_value: float) -> str:
    return "bias_" + ("m" if bias_value < 0 else "p") + f"{int(round(abs(bias_value) * 100)):02d}"


def _width_id(variance_mult: float) -> str:
    return f"width_v{int(round(variance_mult * 100)):03d}"    # 0.5->width_v050, 2.0->width_v200, …


# The real-forecaster roster: every model below is both scored offline in the quality census and
# run as a negotiation condition. AutoTheta is the classical slot (fast; AutoARIMA was dropped — ~9.5s/series at seasonal m=52, too slow for
# a broad census, and AutoTheta reaches ~= quality).
# neuralforecast deep-learning models added for forecaster DIVERSITY (they span accuracy & bias
# far more than the near-unbiased classical/foundation set) — all are Chronos/Chronos-2 baselines
# (arXiv 2403.07815 / 2510.15821). Trained global on the broad panel per leakage-safe cutoff.
# dlinear dropped: on weekly grocery it fit degenerately (relWQL 1.16 > naive, ~15% coverage —
# broken intervals) in the placement scan; the other three land in the real-forecaster cluster.
NEURAL_MODELS: tuple[str, ...] = ("nhits", "patchtst", "deepar")


def _neural_conditions() -> list[ConditionSpec]:
    return [ConditionSpec(f"nf_{m}", ComponentSpec("neural_global", {"model": m, "max_steps": 200}))
            for m in NEURAL_MODELS]


# AutoARIMA excluded everywhere (not the census, not episodes): it clusters with AutoTheta/AutoETS
# (both fully run), and refits ~20x/episode under the global lock (~2.9h). Redundant + pathological.
REAL_FORECASTERS: list[str] = [
    "seasonal_naive", "auto_ets", "auto_theta", "lightgbm_global",
    "chronos2", "chronos_bolt", "timesfm25",
] + [f"nf_{m}" for m in NEURAL_MODELS]

# The evaluated LLM panel (paper): short key -> (display label, experiments/panel/<dir>). Single source
# for the analysis scripts that iterate the panel (make_transcripts, make_panel, paper_stats).
EVALUATED_PANEL: dict[str, tuple[str, str]] = {
    "opus":     ("Opus",     "opus_4_6"),
    "gpt":      ("GPT",      "gpt_5_6_sol"),
    "qwen":     ("Qwen",     "qwen3_235b"),
    "deepseek": ("DeepSeek", "deepseek_v3_2"),
}


def reference_conditions() -> list[ConditionSpec]:
    """The 26 EPISODE conditions: 10 real forecasters + no_forecast + 15-cell perturbation grid (bias +-{.05,.10,.15,.20} + width ladder).

    Real forecasters span four families (classical: seasonal_naive/auto_ets/auto_theta; GBM:
    lightgbm_global; foundation: chronos2/chronos_bolt/timesfm25; deep learning: nf_nhits/
    nf_patchtst/nf_deepar) — a deliberately broad roster so the quality/bias->outcome null is
    shown to be robust across the SOTA landscape, not an artifact of one model class.

    To add a forecaster: register it (``@FORECASTERS.register("name")``) and add a ConditionSpec
    here (or compose a custom ``BenchmarkConfig.conditions`` list — no core edits, see EXTENDING.md).

    Real-forecaster quality is relative WQL vs seasonal naive under the Chronos/M5 protocol
    (single origin, many stable-price series, in-sample MASE scale); the measured ladder is
    ``experiments/shared/forecaster_quality/report.json``. AutoTheta is the classical slot
    (fast, 0.05s/series vs AutoARIMA's ~9.5s at seasonal m=52).

    Perturbation grid = an 8-level BIAS dose-response ladder (±{.05,.10,.15,.20}, calibrated
    width) + a 6-level WIDTH/calibration ladder (variance ×{.25,.5,.75,1.5,2,3}, zero bias),
    crossed only at the clean-oracle center ``unbiased`` (bias 0, calibrated). 8 + 6 + 1 = 15
    cells. Bracketing
    conditions (order first; gate the aggregate collapse checkpoint) are the spread
    anchors: no_forecast, the strongest under/over bias, and the clean oracle.
    """
    # auto_arima is excluded everywhere: as an episode condition it refits per predict
    # (~20/episode) under the global setup lock, stalling the run; it clusters with theta/ets anyway.
    real = [
        ConditionSpec("no_forecast", forecaster=None, is_bracketing=True),
        ConditionSpec("seasonal_naive", ComponentSpec("seasonal_naive")),
        ConditionSpec("auto_ets", ComponentSpec("auto_ets")),
        ConditionSpec("auto_theta", ComponentSpec("auto_theta")),
        ConditionSpec("lightgbm_global", ComponentSpec("lightgbm_global")),
        ConditionSpec("chronos2", ComponentSpec("chronos2")),
        ConditionSpec("chronos_bolt", ComponentSpec("chronos_bolt")),
        ConditionSpec("timesfm25", ComponentSpec("timesfm25")),
    ] + _neural_conditions()
    # BIAS ladder @ calibrated width (nonzero bias); strongest ends bracket the spread.
    _bias_max = max(abs(b) for b in _BIAS_LADDER)
    grid = [ConditionSpec(_bias_id(b), _perturbed(b, _CAL_VAR),
                          is_bracketing=(abs(b) == _bias_max)) for b in _BIAS_LADDER]
    # Clean-oracle center (shared by both ladders) + WIDTH ladder @ zero bias.
    grid += [ConditionSpec("unbiased", _perturbed(0.0, _CAL_VAR), is_bracketing=True)]
    grid += [ConditionSpec(_width_id(v), _perturbed(0.0, v)) for v in _WIDTH_LADDER]
    return real + grid
