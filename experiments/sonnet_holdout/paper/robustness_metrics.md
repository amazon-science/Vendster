# Robustness — forecast accuracy → attainment across metrics

Lower metric = better forecast; a NEGATIVE slope / correlation means *better accuracy → higher attainment* ("quality helps"). The **design-correct** column (basket = unit of analysis) is the one to report; the **pooled-OLS** column ignores clustering and is anti-conservative (shown to flag the trap).

| metric | design-correct slope (p) | naive pooled slope (p) | forecaster-level r (p, n=10) |
|---|---|---|---|
| relative_wql | -0.0515 (p=0.370) | +0.0720 (p=0.035) | -0.253 (p=0.480) |
| mase | -0.1434 (p=0.267) | -0.0023 (p=0.957) | -0.369 (p=0.294) |
| mape | -0.2063 (p=0.422) | -0.0649 (p=0.541) | -0.409 (p=0.241) |
| smape | -0.3094 (p=0.497) | -0.0507 (p=0.692) | -0.358 (p=0.309) |
| rmse | -0.0024 (p=0.620) | -0.0012 (p=0.283) | -0.268 (p=0.453) |
| crps | -0.0044 (p=0.491) | -0.0018 (p=0.482) | -0.244 (p=0.497) |

**Finding:** under the design-correct test, *no* accuracy metric significantly predicts attainment — the primary null (relWQL) is robust to metric choice, not a relWQL artifact. Direction is uniformly "quality helps" (all forecaster-level r < 0), and the H0 gate confirms forecasts beat no_forecast; the observational accuracy gradient (H3o) is simply unresolved because the real forecasters cluster tightly in both accuracy and outcome (all near-unbiased). Causal leverage comes from the bias-perturbation grid (H1c). The naive pooled column shows how ignoring basket clustering would have manufactured spurious significance for MASE/MAPE/sMAPE/RMSE/CRPS.

## Observational analog of the perturbation grid — do real forecasters' bias/width move outcomes?

The dialed fake grid shows signed bias moves attainment (H1c, p<0.001) and width does not (H2c). Do the *real* forecasters' measured properties reproduce this? (design-correct = basket-clustered; these are the observational H1o/H2o tests.)

| property | real-forecaster range | design-correct slope (p) | forecaster-level r (p) |
|---|---|---|---|
| signed bias | [-0.191, +0.255] | -0.4971 (p=0.016) | -0.566 (p=0.088) |
| interval width | [+1.226, +3.203] | -0.0013 (p=0.946) | +0.011 (p=0.976) |
| coverage@80 | [+0.400, +1.000] | +0.0591 (p=0.307) | +0.376 (p=0.285) |

**Finding (H1o/H2o):** the *observational* signed-bias slope (H1o, -0.4971, p=0.016) carries the SAME negative sign as the causal grid (H1c) and is **significant** across baskets — the observational arm reproduces the causal result, while width (H2o) is null as predicted. (Forecaster-level correlations are weak/confounded — interval width proxies quality, e.g. seasonal_naive is both widest and worst — so the basket-clustered slope is the design-correct test.) The diverse roster (Chronos-2/Bolt + neuralforecast DL, arXiv 2510.15821 / 2403.07815) is what gives the observational arm enough quality range to test this at all; the controlled bias grid (H1c) remains the primary lever.
