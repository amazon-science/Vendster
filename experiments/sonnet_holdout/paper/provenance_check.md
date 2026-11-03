# Provenance check — incremental vs fresh re-run

A = `experiments/paper_v1_incremental/experiment`  ·  B = `experiments/paper_v1/experiment`

Bedrock is non-deterministic + finite m, so exact match is NOT expected; each condition should agree within ~2*SE of the proportion difference. A flag means B shifted beyond the Monte-Carlo band vs A (investigate).

| condition | A attain (n) | B attain (n) | Δ (B−A) | 2·SE | within noise? |
|---|---|---|---|---|---|
| auto_ets | 0.420 (400) | 0.432 (400) | +0.013 | 0.070 | ✓ |
| auto_theta | 0.390 (400) | 0.398 (400) | +0.008 | 0.069 | ✓ |
| bias_m05 | 0.448 (400) | 0.463 (400) | +0.015 | 0.070 | ✓ |
| bias_m10 | 0.458 (400) | 0.450 (400) | -0.008 | 0.070 | ✓ |
| bias_m15 | 0.425 (400) | 0.448 (400) | +0.023 | 0.070 | ✓ |
| bias_m20 | 0.425 (400) | 0.475 (400) | +0.050 | 0.070 | ✓ |
| bias_p05 | 0.355 (400) | 0.350 (400) | -0.005 | 0.068 | ✓ |
| bias_p10 | 0.325 (400) | 0.300 (400) | -0.025 | 0.066 | ✓ |
| bias_p15 | 0.290 (400) | 0.312 (400) | +0.023 | 0.065 | ✓ |
| bias_p20 | 0.325 (400) | 0.328 (400) | +0.003 | 0.066 | ✓ |
| chronos2 | 0.393 (400) | 0.380 (400) | -0.013 | 0.069 | ✓ |
| chronos_bolt | 0.378 (400) | 0.380 (400) | +0.003 | 0.069 | ✓ |
| lightgbm_global | 0.400 (400) | 0.393 (400) | -0.008 | 0.069 | ✓ |
| narrow | 0.352 (400) | 0.375 (400) | +0.023 | 0.068 | ✓ |
| nf_deepar | 0.487 (400) | 0.495 (400) | +0.008 | 0.071 | ✓ |
| nf_nhits | 0.440 (400) | 0.422 (400) | -0.018 | 0.070 | ✓ |
| nf_patchtst | 0.435 (400) | 0.407 (400) | -0.028 | 0.070 | ✓ |
| no_forecast | 0.230 (400) | 0.235 (400) | +0.005 | 0.060 | ✓ |
| seasonal_naive | 0.352 (400) | 0.362 (400) | +0.010 | 0.068 | ✓ |
| timesfm25 | 0.367 (400) | 0.375 (400) | +0.008 | 0.068 | ✓ |
| unbiased | 0.385 (400) | 0.395 (400) | +0.010 | 0.069 | ✓ |
| wide | 0.347 (400) | 0.398 (400) | +0.050 | 0.068 | ✓ |
| width_v025 | 0.383 (400) | 0.393 (400) | +0.010 | 0.069 | ✓ |
| width_v075 | 0.315 (400) | 0.290 (400) | -0.025 | 0.065 | ✓ |
| width_v150 | 0.388 (400) | 0.378 (400) | -0.010 | 0.069 | ✓ |
| width_v300 | 0.378 (400) | 0.393 (400) | +0.015 | 0.069 | ✓ |

**mean |Δ| across conditions = 0.016**  ·  flags = 0

**Verdict:** REPLICATION CONFIRMED — all conditions within the Monte-Carlo band; the incremental bundle shows no science-impacting artifact.
