# Extending Vendster

Vendster is a **layered** benchmark:

- **`Vendster-M5`** — a frozen, versioned reference instance (`vendster/reference.py`,
  version `vendster-m5-v1`) that produces the paper's citeable numbers. Changing its
  components/params requires bumping `task_core.REFERENCE_VERSION`.
- **The framework** — clean seams (six `Protocol`s in `vendster/core/interfaces.py`, one
  registry each in `vendster/core/registry.py`) so you can build your *own* instance
  without editing any core module.

The pattern is always the same: **implement a Protocol → register it under a name →
reference the name from a `BenchmarkConfig`.** Nothing in the negotiation arena or the
experiment harness needs to change.

The shared **arena** — the turn engine, the discrete discount action space, the DGP
realization, retries, and transcript archiving (`vendster/negotiation/episode.py`,
`vendster/core/task_core.py`) — is common to every scenario and instance. This is what
makes results comparable.

---

## Add a forecaster (time-series model)

Implement the `Forecaster` protocol and register it:

```python
import pandas as pd
from vendster.core.registry import FORECASTERS
from vendster.core.types import BaseForecast

@FORECASTERS.register("my_model")
class MyForecaster:
    name = "my_model"

    def predict(self, item_id: str, history: pd.Series,
                target_weeks) -> list[BaseForecast]:
        # `history` is the weekly series strictly BEFORE the first target week
        # (real forecasters must not see the target weeks). Return one
        # BaseForecast (point + 80% interval) per requested target week.
        ...
```

### Capability-driven context injection (`REQUIRES`)

Zero-shot / per-series models need only `history` and require **no** declaration. A
panel-trained or global model instead declares the context keys it needs via a `REQUIRES`
class attribute; the harness's `make_forecaster` / `forecaster_context`
(`vendster/experiment/runner.py`) inject **exactly** those keys at construction time —
**no harness edits** (this generic seam replaced the old name-special-casing):

```python
@FORECASTERS.register("my_global")
class MyGlobalForecaster:
    name = "my_global"
    REQUIRES = ("panel", "exog", "train_cutoff")   # injected as constructor kwargs
    def __init__(self, panel, exog, train_cutoff, **params): ...
    def predict(self, item_id, history, target_weeks): ...
```

Available context keys are: `panel`, `exog`, `train_cutoff`, `true_weekly`, `sigmas`.
Perturbation-style forecasters (which derive from the true series rather than learning
from history) declare `REQUIRES = ("true_weekly", "sigmas")`. The harness caches a
constructed forecaster by `(name, params)` — and additionally by `train_cutoff` for
cutoff-dependent models, so panel models refit once per cutoff.

### Registered forecaster names (built-ins)

| Name | Family | `REQUIRES` |
|------|--------|-----------|
| `seasonal_naive` | classical | — (per-series) |
| `auto_ets` | classical | — |
| `auto_theta` | classical | — |
| `lightgbm_global` | gradient-boosted, global | `("panel","exog","train_cutoff")` |
| `chronos2` | foundation | — (zero-shot) |
| `chronos_bolt` | foundation | — |
| `timesfm25` | foundation | — |
| `neural_global` | deep learning (neuralforecast) | `("panel","exog","train_cutoff")` |
| `perturbed` | controlled bias/width perturbation | `("true_weekly","sigmas")` |

`neural_global` takes a `model` param selecting the architecture (`nhits`, `patchtst`,
`deepar`); the reference conditions expose these as `nf_nhits` / `nf_patchtst` /
`nf_deepar`. `perturbed` takes `bias_direction`, `bias_magnitude`, `variance_mult`.

> Note: `auto_arima` and `lightgbm` are **not** registered (deliberately dropped — see
> `vendster/core/config.py`). Use `auto_theta` (classical slot) and `lightgbm_global`.

Once registered, add the forecaster as a `ConditionSpec` (below) — no core edits needed.

---

## Add a negotiation scenario

The negotiation **identity** — who negotiates, toward what objective, who sees what
private information, and the success criterion — is a swappable `Scenario`. Implement the
protocol, register it, and name it in `BenchmarkConfig.scenario`:

```python
from typing import Optional
from vendster.core.registry import SCENARIOS

@SCENARIOS.register("my_scenario")
class MyScenario:
    name = "my_scenario"
    roles = ("Buyer", "Vendor")
    opening_role = "Buyer"

    def next_role(self, role: str) -> str: ...
    def system_prompt(self, role: str, target: Optional[float]) -> str: ...
    def role_blocks(self, role: str, episode) -> list[str]: ...   # asymmetric per-role info
    def target_met(self, total_sales: float, target: Optional[float]) -> Optional[bool]: ...
```

`role_blocks` uses the episode's shared block renderers (e.g. `episode._forecast_block()`,
`episode._margins_block()`) to inject each role's private context. The reference
implementation is **`VendsterScenario`** (`vendster/negotiation/scenario.py`): Buyer has a
threshold sales target and privately sees the demand forecast; Vendor maximizes total
portfolio profit and privately knows unit margins. The turn engine, discount action space,
and DGP realization are the **shared arena** every scenario runs on — a new scenario is
"implement + register", nothing else.

Then set it on the config:

```python
config.scenario = ComponentSpec("my_scenario")
```

The built-in scenario name is `vendster` (the config default).

---

## Add a data source / demand model / metric / LLM client

The other four registries follow the identical implement-register-reference pattern:

| Interface | Method(s) | Built-in names |
|-----------|-----------|----------------|
| `DataSource`  | `load() -> Dataset` | `m5` |
| `DemandModel` | `expected_units(...)`, `sample_units(...)` | `elasticity` |
| `Metric`      | `compute(y_true, quantiles, scale) -> float` | `mae`, `mape`, `smape`, `mase`, `rmse`, `crps`, `scaled_crps`, `wql`, `coverage_80`, `sharpness` |
| `LLMClient`   | `generate(system_prompt, messages) -> str` | `mock`, `bedrock` (any Bedrock model via the Converse API; `model_id` param) |

**Evaluating a new LLM** needs no code at all: the `bedrock` client takes any Bedrock model id, so
`MODEL_ID=<id> OUT=experiments/panel/<name> bash scripts/build_model_bundle.sh` produces a full
bundle at the paper's locked design, comparable with the published panel (add it to
`scripts/make_panel.py --models` to put it in the cross-model tables). Every negotiation script
takes `--model bedrock --model-id <id>`, and `--model-id` is required with `bedrock`.

A non-Bedrock provider is one `LLMClient` implementation registered under a new name. Reference
it from a `BenchmarkConfig` (`llm_client=ComponentSpec("<name>")`) and run it with
`run_benchmark`. The command-line scripts offer only the clients registered when `vendster.llm`
is imported, so to use yours from a script, register it in a module that `vendster/llm/__init__.py`
imports.

For a complete, runnable **bring-your-own-data** example, see
[`examples/byo_data_source.py`](examples/byo_data_source.py) — it registers a synthetic
`DataSource` and runs an episode with zero core edits (`uv run python
examples/byo_data_source.py`). Each `ProductSpec` carries its own **elasticity** (β), so a
BYO source can mix elastic and inelastic goods freely (the example uses −2.0/−2.5/−3.0).
The M5 reference uses one literature value, β = −2.6, for every product
(`vendster/data/m5.py`).

---

## Composing and running a custom `BenchmarkConfig`

A `BenchmarkConfig` is just component *names* plus the experiment settings. Build your own
`conditions` list (a `ConditionSpec` is an id + an optional forecaster `ComponentSpec`;
`forecaster=None` is the no-forecast control) and a `scenario`, then run target-setting +
experiment via `reference.run_benchmark`. The target is the analytic reference-policy
P`target_percentile` of total sales (seeded DGP draws, no LLM), so it is deterministic and
identical across models; set `sales_target` to fix it instead. `conditions`, `scenario`, and
`negotiation` (a `NegotiationParams`: discount levels, max turns, track record, action retries)
are all honored.

`run_benchmark` runs one basket in two waves. It runs the bracketing conditions first, then
stops with `"halted": "COLLAPSE…"` in the summary if their attainment is outside the viable band
(no-forecast attainment in 0.15–0.70 and a bracket spread of at least 0.15). Only then does it run
the rest of the roster. The mock client always lands at 0 or 1 attainment, so it always stops
there; pass `checkpoint_enabled=False` to run the whole roster anyway:

```python
from vendster.core.config import BenchmarkConfig, ComponentSpec, ConditionSpec
from vendster.reference import run_benchmark

cfg = BenchmarkConfig(
    version="my-instance-v1",
    data_source=ComponentSpec("my_shop"),            # your DataSource
    demand_model=ComponentSpec("elasticity"),
    llm_client=ComponentSpec("mock"),                # or "bedrock"
    scenario=ComponentSpec("my_scenario"),           # or the default "vendster"
    conditions=[
        ConditionSpec("no_forecast", forecaster=None, is_bracketing=True),
        ConditionSpec("my_model", ComponentSpec("my_model")),
        ConditionSpec("clean", ComponentSpec("perturbed",
                      {"bias_direction": 0, "bias_magnitude": 0.0, "variance_mult": 1.0}),
                      is_bracketing=True),
    ],
    n_episodes=20,
)
run_benchmark(cfg, ref_draws=2000, out_dir="experiments/my_instance",
              checkpoint_enabled=False)          # mock: skip the collapse checkpoint
```

To reuse the reference roster instead, import `reference_conditions()` from
`vendster.core.config` and pass it as `conditions`.

---

## Reproducibility & leakage (inherited by every instance)

- The forecaster for a stretch trains strictly before it (`train_cutoff = m1`); episode
  target weeks are never in any training set. The leakage invariant is enforced at
  `Episode` construction (a raised `ValueError`, not an `assert`).
- Offline artifacts (forecast quality, analytic targets) are deterministic from
  `(git_sha, seed)`; the DGP is seeded with process-stable `blake2b` seeds.
- LLM episodes are **not** bit-reproducible (Bedrock has no seed). The reproducibility
  contract is: the **frozen target** + **seeded DGP draws** + **archived per-episode records** +
  a per-run **manifest** (git SHA / model / seed).
</content>
