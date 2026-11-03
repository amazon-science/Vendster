# Vendster — cross-model results

### Table A — forecast effect by model
| model | no_fc | beats no_fc (Δ range) | bias slope (causal grid) | width slope (causal grid) | bias slope (real fc) | accuracy slope (real fc) | forecast-reactive? |
|---|--:|---|---|---|---|---|---|
| opus | 0.24 | +0.09..+0.18 | -0.387 (p=0.000) | +0.016 (p=0.225) | -0.466 (p=0.009) | -0.021 (p=0.703) | **yes** |
| gpt | 0.25 | +0.08..+0.23 | -0.416 (p=0.000) | +0.002 (p=0.882) | -0.408 (p=0.057) | -0.126 (p=0.016) | **yes** |
| qwen | 0.11 | +0.08..+0.15 | +0.001 (p=0.988) | -0.003 (p=0.796) | -0.195 (p=0.266) | +0.009 (p=0.835) | no |
| deepseek | 0.22 | +0.01..+0.10 | +0.012 (p=0.855) | -0.002 (p=0.867) | -0.182 (p=0.255) | +0.044 (p=0.362) | no |

_Causal-grid slopes: basket-clustered dose-response on the dialed perturbation grid. Real-fc slopes: per-basket slope across the real forecasters (observational). Negative bias slope = under-forecasting helps / over-forecasting hurts (complacency)._

### Table B — behavioral fingerprint: bias-ladder slope of negotiation behavior
How strongly each model translates the forecast into negotiation behavior (steeper |slope| = more forecast-reactive). Under→over-forecast direction.

| model | 1st-offer aggr | concession | agreed discount | turns | agreement rate |
|---|--:|--:|--:|--:|--:|
| opus | -0.101* | +0.002 | -0.082* | -3.093* | +0.002 |
| gpt | -0.139* | +0.047* | -0.195* | -21.936* | +0.926* |
| qwen | +0.026* | -0.001 | -0.004 | +0.082 | +0.038 |
| deepseek | +0.003 | +0.001 | -0.002 | -1.104* | +0.033 |

_* p<0.05 (one-sample t over baskets). first-offer aggressiveness = opening stance; concession = give-rate; agreed discount = deal depth; turns = persistence; agreement = deal-closing._

### Table C — real-forecaster target attainment by model (Δ vs no_forecast)
Every real forecaster vs the no_forecast baseline, per model. Cell = pooled attainment (Δ vs that model's no_forecast).

| forecaster | opus | gpt | qwen | deepseek |
|---|--:|--:|--:|--:|
| no_forecast | 0.24 | 0.25 | 0.11 | 0.22 |
| seasonal_naive | 0.35 (+0.11) | 0.33 (+0.08) | 0.24 (+0.13) | 0.31 (+0.10) |
| auto_ets | 0.36 (+0.12) | 0.42 (+0.17) | 0.23 (+0.12) | 0.31 (+0.09) |
| auto_theta | 0.35 (+0.11) | 0.43 (+0.18) | 0.26 (+0.15) | 0.29 (+0.08) |
| lightgbm_global | 0.37 (+0.13) | 0.41 (+0.16) | 0.21 (+0.10) | 0.29 (+0.08) |
| chronos2 | 0.33 (+0.09) | 0.43 (+0.18) | 0.19 (+0.08) | 0.23 (+0.01) |
| chronos_bolt | 0.33 (+0.09) | 0.39 (+0.14) | 0.21 (+0.10) | 0.29 (+0.08) |
| timesfm25 | 0.34 (+0.10) | 0.39 (+0.14) | 0.20 (+0.09) | 0.28 (+0.06) |
| nf_nhits | 0.41 (+0.17) | 0.45 (+0.20) | 0.23 (+0.12) | 0.32 (+0.10) |
| nf_patchtst | 0.39 (+0.14) | 0.43 (+0.18) | 0.23 (+0.12) | 0.32 (+0.10) |
| nf_deepar | 0.42 (+0.18) | 0.48 (+0.23) | 0.20 (+0.09) | 0.29 (+0.07) |

_Δ in parentheses = attainment gain over that model's no_forecast baseline. Every forecaster clears the baseline for every model._

### Table D — forecast accuracy → attainment: design-correct slope by metric × model
Per-basket slope of attainment on each accuracy metric across the real forecasters (negative = better accuracy helps). Bold = p<0.05. Null for almost every (metric, model) — the exception is the most forecast-reactive model on the primary metric.

| metric | opus | gpt | qwen | deepseek |
|---|--:|--:|--:|--:|
| relative_wql | -0.021 (p=0.703) | **-0.126 (p=0.016)** | +0.009 (p=0.835) | +0.044 (p=0.362) |
| mase | -0.095 (p=0.362) | -0.194 (p=0.104) | +0.005 (p=0.961) | +0.055 (p=0.510) |
| mape | -0.156 (p=0.535) | -0.353 (p=0.261) | +0.015 (p=0.953) | +0.308 (p=0.154) |
| smape | -0.195 (p=0.598) | -0.408 (p=0.424) | +0.098 (p=0.765) | +0.387 (p=0.147) |
| rmse | -0.003 (p=0.453) | -0.007 (p=0.122) | +0.001 (p=0.854) | +0.003 (p=0.488) |
| crps | -0.004 (p=0.477) | -0.011 (p=0.068) | +0.003 (p=0.625) | +0.006 (p=0.363) |

_Design-correct test (basket = unit of analysis, one-sample t over baskets), one row per accuracy metric. See each model's bundle (paper/robustness_metrics.md) for the naive-pooled and forecaster-level lenses._

