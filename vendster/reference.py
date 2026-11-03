"""
The frozen, versioned reference benchmark instance: **Vendster-M5**.

This is just a specific ``BenchmarkConfig`` composed from registered components.
Changing its components/params requires bumping ``task_core.REFERENCE_VERSION``
so published results stay comparable. Deriving a new instance (e.g. your own
data) means supplying a different ``DataSource`` — no core changes (see
``examples/byo_data_source.py``).
"""

from __future__ import annotations

from . import data as _data  # noqa: F401  register m5 DataSource
from . import dgp as _dgp  # noqa: F401  register elasticity DemandModel
from . import forecasters as _fc  # noqa: F401  register forecasters + metrics
from . import llm as _llm  # noqa: F401  register llm clients
from .core import task_core
from .core.config import BenchmarkConfig, ComponentSpec, NegotiationParams, reference_conditions
from .core.factory import build_benchmark


def reference_config(model: str = "bedrock", n_episodes: int = 25, seed: int = 1,
                     panel_size: int = 1500) -> BenchmarkConfig:
    """The frozen Vendster-M5 reference instance: its components, the 26 conditions, and the
    paper's locked per-condition settings (m=25 episodes, seed 1, 1,500-product panel, P80).

    ``run_benchmark`` runs it as ONE basket with a collapse checkpoint. The paper's K=16 paired
    design is ``run_paired_experiment`` (``scripts/run_experiment.py`` /
    ``scripts/build_model_bundle.sh``)."""
    return BenchmarkConfig(
        version=task_core.REFERENCE_VERSION,
        data_source=ComponentSpec("m5", {"panel_size": panel_size}),
        demand_model=ComponentSpec("elasticity"),
        llm_client=ComponentSpec(model),
        conditions=reference_conditions(),
        negotiation=NegotiationParams(),
        n_episodes=n_episodes, target_percentile=0.80, seed=seed,
    )


def run_benchmark(config: BenchmarkConfig, *, ref_draws: int = 2000,
                  out_dir: str = "experiments/vendster_m5", **experiment_kwargs) -> dict:
    """Resolve a ``BenchmarkConfig`` and run target-setting -> experiment end-to-end.

    The target is ``config.sales_target`` when set, else the analytic reference-policy
    P{target_percentile} of the dataset's total-sales distribution (``ref_draws`` seeded DGP
    draws, no LLM). ``config.negotiation`` sets every episode's protocol. Extra keyword arguments
    go to ``run_single_experiment`` (e.g. ``checkpoint_enabled=False``). Works for ANY config
    (reference or a user's own instance) — components are resolved purely through the registries.
    """
    import json
    from pathlib import Path

    from .core.registry import SCENARIOS
    from .experiment import analytic_target, run_single_experiment

    bench = build_benchmark(config)
    dataset = bench.data_source.load()
    scenario = SCENARIOS.create(config.scenario.name, **config.scenario.params)  # honor config's scenario
    episode_kwargs = config.negotiation.episode_kwargs()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if config.sales_target is not None:
        calib = {"target": float(config.sales_target), "source": "BenchmarkConfig.sales_target"}
    else:
        calib = analytic_target(dataset, n=ref_draws, percentile=config.target_percentile * 100,
                                seed=config.seed, demand_model=bench.demand_model,
                                discount_levels=config.negotiation.discount_levels)
    Path(f"{out_dir}/analytic_target.json").write_text(json.dumps(calib, indent=2, default=str))
    summary = run_single_experiment(dataset, bench.llm_client, calib["target"],
                             episodes_per_condition=config.n_episodes, seed=config.seed,
                             demand_model=bench.demand_model, model_name=config.llm_client.name,
                             conditions=config.conditions,        # honor the config's roster
                             scenario=scenario,                   # honor the config's scenario
                             episode_kwargs=episode_kwargs,       # honor the config's protocol
                             out_dir=f"{out_dir}/run", **experiment_kwargs)
    return {"version": config.version, "target": calib["target"], "summary": summary}
