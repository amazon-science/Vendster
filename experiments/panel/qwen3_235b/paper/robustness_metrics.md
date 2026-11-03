# Robustness — do real forecasters' error properties predict attainment?

Per-(forecaster, basket) points: x = the forecaster's offline error property on that basket (from basket_quality), y = its attainment on that basket. **Design-correct** = per-basket OLS slope across forecasters, one-sample t over baskets (basket = unit of analysis; the reported test). **Pooled** ignores clustering and is anti-conservative (shown only to flag the trap). **Forecaster-level** = Pearson over per-forecaster means (n=10).

## Accuracy metrics (lower = better forecast; negative slope = better accuracy → higher attainment)

| metric | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|
| relative_wql | +0.0093 (p=0.835) | +0.0132 (p=0.619) | +0.2254 (p=0.531) |
| mase | +0.0047 (p=0.961) | +0.0353 (p=0.281) | +0.2906 (p=0.415) |
| mape | +0.0148 (p=0.953) | +0.0502 (p=0.541) | +0.3055 (p=0.391) |
| smape | +0.0979 (p=0.765) | +0.0956 (p=0.332) | +0.3082 (p=0.386) |
| rmse | +0.0009 (p=0.854) | -0.0000 (p=0.968) | +0.2060 (p=0.568) |
| crps | +0.0028 (p=0.625) | +0.0008 (p=0.674) | +0.2099 (p=0.561) |

## Calibration properties (observational analog of the perturbation grid: H1o bias, H2o width)

| property | real-forecaster range | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|---|
| signed bias | [-0.191, +0.255] | -0.1952 (p=0.266) | -0.0746 (p=0.338) | -0.0463 (p=0.899) |
| interval width | [+1.226, +3.203] | +0.0099 (p=0.480) | +0.0072 (p=0.614) | +0.2652 (p=0.459) |
| coverage@80 | [+0.400, +1.000] | +0.0048 (p=0.927) | +0.0143 (p=0.750) | +0.0598 (p=0.870) |
