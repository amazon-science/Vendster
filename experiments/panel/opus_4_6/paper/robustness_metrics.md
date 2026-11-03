# Robustness — do real forecasters' error properties predict attainment?

Per-(forecaster, basket) points: x = the forecaster's offline error property on that basket (from basket_quality), y = its attainment on that basket. **Design-correct** = per-basket OLS slope across forecasters, one-sample t over baskets (basket = unit of analysis; the reported test). **Pooled** ignores clustering and is anti-conservative (shown only to flag the trap). **Forecaster-level** = Pearson over per-forecaster means (n=10).

## Accuracy metrics (lower = better forecast; negative slope = better accuracy → higher attainment)

| metric | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|
| relative_wql | -0.0211 (p=0.703) | +0.0348 (p=0.270) | -0.0119 (p=0.974) |
| mase | -0.0951 (p=0.362) | -0.0404 (p=0.299) | -0.1384 (p=0.703) |
| mape | -0.1556 (p=0.535) | -0.1797 (p=0.064) | -0.1648 (p=0.649) |
| smape | -0.1948 (p=0.598) | -0.1767 (p=0.131) | -0.1112 (p=0.760) |
| rmse | -0.0028 (p=0.453) | -0.0003 (p=0.766) | -0.0142 (p=0.969) |
| crps | -0.0042 (p=0.477) | +0.0002 (p=0.928) | +0.0159 (p=0.965) |

## Calibration properties (observational analog of the perturbation grid: H1o bias, H2o width)

| property | real-forecaster range | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |
|---|---|---|---|---|
| signed bias | [-0.191, +0.255] | -0.4661 (p=0.009) | -0.2148 (p=0.019) | -0.5193 (p=0.124) |
| interval width | [+1.226, +3.203] | +0.0052 (p=0.747) | +0.0039 (p=0.816) | +0.1157 (p=0.750) |
| coverage@80 | [+0.400, +1.000] | -0.0326 (p=0.600) | +0.0399 (p=0.454) | +0.0843 (p=0.817) |
