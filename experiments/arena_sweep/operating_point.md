### Appendix — Operating-point difficulty sweep (justifies P80)

Stage-1 EXPLORATORY sweep (seed 0, 8 paired baskets, m=15 episodes/cell, `no_forecast` vs `unbiased` perfect-forecast oracle). The operating point is fixed **effect-blind**: PXX where `no_forecast` attainment ≈ 0.30 → **P80**, using only the no_forecast arm so the choice can't be tuned to the forecast effect (note P80 is not even the largest-gap percentile in the ladder). Inferential use is descriptive only (the `no_forecast` difficulty gradient); all confirmatory inference is at the locked P80.

| PXX | no_forecast [95% CI] | unbiased [95% CI] | gap | paired Δ | Wilcoxon p | |
|----:|---|---|---:|---:|---:|---|
| P50 | 0.467 [0.38,0.56] | 0.533 [0.44,0.62] | +0.067 | +0.067 | 0.375 |  |
| P60 | 0.392 [0.31,0.48] | 0.483 [0.40,0.57] | +0.092 | +0.092 | 0.125 |  |
| P70 | 0.333 [0.26,0.42] | 0.450 [0.36,0.54] | +0.117 | +0.117 | 0.125 |  |
| **P80** | 0.308 [0.23,0.40] | 0.383 [0.30,0.47] | +0.075 | +0.075 | 0.188 | **← selected (effect-blind)** |
| P90 | 0.183 [0.12,0.26] | 0.283 [0.21,0.37] | +0.100 | +0.100 | 0.234 |  |

_Selection rule (pre-specified, effect-blind): fix PXX at no_forecast ≈ 0.30. Result: **P80** (no_forecast 0.308)._

