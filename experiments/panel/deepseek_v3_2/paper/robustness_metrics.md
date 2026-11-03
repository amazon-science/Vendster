# Robustness — do real forecasters' error properties predict attainment?

Per-(forecaster, basket) points: x = the forecaster's offline error property on that basket (from basket_quality), y = its attainment on that basket. **Design-correct** = per-basket OLS slope across forecasters, one-sample t over baskets (basket = unit of analysis; the reported test). **Pooled** ignores clustering and is anti-conservative (shown only to flag the trap). **Forecaster-level** = Pearson over per-forecaster means (n=10).

## Accuracy metrics (lower = better forecast; negative slope = better accuracy → higher attainment)

| metric | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|
| relative_wql | +0.0440 (p=0.362) | +0.0577 (p=0.078) | +0.2554 (p=0.476) |
| mase | +0.0547 (p=0.510) | +0.0301 (p=0.458) | +0.2424 (p=0.500) |
| mape | +0.3081 (p=0.154) | +0.0830 (p=0.414) | +0.2415 (p=0.502) |
| smape | +0.3874 (p=0.147) | +0.0969 (p=0.428) | +0.2543 (p=0.478) |
| rmse | +0.0030 (p=0.488) | +0.0009 (p=0.397) | +0.2460 (p=0.493) |
| crps | +0.0057 (p=0.363) | +0.0029 (p=0.245) | +0.2645 (p=0.460) |

## Calibration properties (observational analog of the perturbation grid: H1o bias, H2o width)

| property | real-forecaster range | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|---|
| signed bias | [-0.191, +0.255] | -0.1817 (p=0.255) | +0.0532 (p=0.581) | -0.0982 (p=0.787) |
| interval width | [+1.226, +3.203] | +0.0127 (p=0.475) | +0.0051 (p=0.774) | +0.2512 (p=0.484) |
| coverage@80 | [+0.400, +1.000] | -0.0260 (p=0.706) | -0.0528 (p=0.340) | -0.0698 (p=0.848) |
