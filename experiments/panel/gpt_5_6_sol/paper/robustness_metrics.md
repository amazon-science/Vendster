# Robustness — do real forecasters' error properties predict attainment?

Per-(forecaster, basket) points: x = the forecaster's offline error property on that basket (from basket_quality), y = its attainment on that basket. **Design-correct** = per-basket OLS slope across forecasters, one-sample t over baskets (basket = unit of analysis; the reported test). **Pooled** ignores clustering and is anti-conservative (shown only to flag the trap). **Forecaster-level** = Pearson over per-forecaster means (n=10).

## Accuracy metrics (lower = better forecast; negative slope = better accuracy → higher attainment)

| metric | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|
| relative_wql | -0.1259 (p=0.016) | -0.0320 (p=0.385) | -0.6631 (p=0.037) |
| mase | -0.1938 (p=0.104) | -0.0979 (p=0.030) | -0.7641 (p=0.010) |
| mape | -0.3525 (p=0.261) | -0.2495 (p=0.027) | -0.7822 (p=0.007) |
| smape | -0.4075 (p=0.424) | -0.2723 (p=0.046) | -0.7464 (p=0.013) |
| rmse | -0.0075 (p=0.122) | -0.0028 (p=0.020) | -0.6771 (p=0.032) |
| crps | -0.0106 (p=0.068) | -0.0050 (p=0.075) | -0.6538 (p=0.040) |

## Calibration properties (observational analog of the perturbation grid: H1o bias, H2o width)

| property | real-forecaster range | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|---|
| signed bias | [-0.191, +0.255] | -0.4082 (p=0.057) | -0.1429 (p=0.185) | -0.3631 (p=0.302) |
| interval width | [+1.226, +3.203] | -0.0390 (p=0.026) | -0.0379 (p=0.053) | -0.4908 (p=0.150) |
| coverage@80 | [+0.400, +1.000] | +0.0198 (p=0.784) | +0.0498 (p=0.423) | +0.1733 (p=0.632) |
