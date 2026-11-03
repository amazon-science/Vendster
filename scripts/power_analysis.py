"""
Power analysis to SIZE the primary paired experiment (Methodology rationale).

The design is paired/clustered: K baskets (clusters), each seen by every condition, with m
episodes per condition per basket. The headline test is a condition's within-basket
attainment gain over no_forecast, tested with the SAME calibrated test we report — the paired
t-test on the K per-basket deltas, basket = unit of analysis (the percentile cluster bootstrap
is anti-conservative at these small K: FP≈0.10 vs t-test FP≈0.05, verified below).

We ground the simulation in REAL data: the per-basket no_forecast attainments measured in the
calibration run (empirical baseline distribution, incl. its between-basket heterogeneity). For
a candidate (K, m) and a true additive effect Δ (a forecaster raises attainment by Δ points),
we simulate S experiments — draw K basket baselines from the empirical set, draw Binomial(m)
outcomes for no_forecast and the condition (independent episodes), run the paired t-test —
and report POWER = P(detect a positive effect at α=0.05) and the false-positive rate at Δ=0.

    uv run python scripts/power_analysis.py --out experiments/power
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from vendster.analysis.aggregate import BIAS_LADDER, WIDTH_LADDER  # canonical name->level maps
from vendster.core.config import REAL_FORECASTERS, reference_conditions  # canonical 10-forecaster roster

# Per-basket no_forecast attainment baselines at the P80 OPERATING POINT (constant-elasticity env,
# 8 baskets x 15 eps; experiments/arena_sweep, seed 0). Pooled 30.8% — the "challenging threshold"
# selected by the effect-blind rule (no_forecast ~= 0.30; see the paper's Environment Setup appendix). These size the
# CONFIRMATORY run, which is on FRESH product-sets (seed != 0).
EMPIRICAL_BASELINES = [0.20, 0.133, 0.267, 0.467, 0.40, 0.267, 0.40, 0.333]


def _ttest_reject(deltas, alpha):
    """Vectorized two-sided paired t-test over axis=1 (S sims x K baskets).
    Returns (reject_two_sided, mean_delta) arrays of length S."""
    from scipy.stats import t as _t
    S, K = deltas.shape
    mean = deltas.mean(axis=1)
    sd = deltas.std(axis=1, ddof=1)
    se = sd / math.sqrt(K)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, mean / se, 0.0)
    p = 2.0 * _t.sf(np.abs(tstat), K - 1)
    return (p < alpha), mean


def simulate_power(K, m, delta, *, baselines, S=5000, alpha=0.05, seed=0):
    """PRIMARY test = paired t-test on per-basket deltas (calibrated at small K).
    Fully vectorized across S simulations."""
    rng = np.random.default_rng(seed)
    emp = np.asarray(baselines, float)
    p0 = emp[rng.integers(0, len(emp), size=(S, K))]        # (S,K) heterogeneous baselines
    pc = np.clip(p0 + delta, 0.0, 1.0)
    base = rng.binomial(m, p0) / m
    cond = rng.binomial(m, pc) / m
    rej, mean = _ttest_reject(cond - base, alpha)
    power = float(np.mean(rej & (mean > 0)))                 # detect a POSITIVE effect
    # false-positive under the null: two independent no_forecast draws
    base0 = rng.binomial(m, p0) / m
    cond0 = rng.binomial(m, p0) / m
    rej0, _ = _ttest_reject(cond0 - base0, alpha)
    return power, float(np.mean(rej0))


def _slope_weights(x):
    """OLS-slope linear weights w s.t. slope = w . y for fixed design points x."""
    x = np.asarray(x, float)
    sxx = ((x - x.mean()) ** 2).sum()
    return (x - x.mean()) / sxx


def simulate_slope_power(K, m, beta, *, xlevels, baselines, S=20000, alpha=0.05, seed=0):
    """A-PRIORI power for a CAUSAL-ladder trend test (H1c bias, H2c width): per basket, fit an OLS
    slope of attainment on the DIALED ladder level, then a one-sample t over K baskets. Uses only the
    fixed ladder + the pre-specified baseline attainment (no observed effects) -> genuine a-priori sizing.
    Attainment at level x is Binomial(m, clip(baseline + beta*x))."""
    rng = np.random.default_rng(seed)
    x = np.asarray(xlevels, float)
    xc = x - x.mean()                                             # centered: slope test is location-invariant
    w = _slope_weights(x)
    emp = np.asarray(baselines, float)
    p0 = emp[rng.integers(0, len(emp), size=(S, K))]              # (S,K) heterogeneous baselines
    p = np.clip(p0[..., None] + beta * xc[None, None, :], 0.0, 1.0)  # (S,K,L)
    y = rng.binomial(m, p) / m
    slope = np.einsum("l,skl->sk", w, y)                          # per-basket slope
    rej, mean = _ttest_reject(slope, alpha)
    power = float(np.mean(rej & (np.sign(mean) == np.sign(beta)))) if beta != 0 else float(np.mean(rej))
    y0 = rng.binomial(m, np.broadcast_to(p0[..., None], (S, K, len(x)))) / m
    rej0, _ = _ttest_reject(np.einsum("l,skl->sk", w, y0), alpha)
    return power, float(np.mean(rej0))


def mde_attain(K, m, *, xlevels, baselines, S=20000, alpha=0.05, power=0.80, seed=0):
    """Smallest END-TO-END attainment effect detectable at the target power (a-priori). Bisect the
    attainment effect E in [0,1] (a sane, common scale for every predictor) and map to a slope E/range;
    this avoids the slope-scale blow-up when the predictor has large raw units (e.g. sharpness)."""
    x = np.asarray(xlevels, float)
    R = float(x.max() - x.min())
    lo, hi = 0.0, 1.0
    for _ in range(26):
        E = (lo + hi) / 2
        pw, _ = simulate_slope_power(K, m, E / R, xlevels=xlevels, baselines=baselines, S=S,
                                     alpha=alpha, seed=seed)
        lo, hi = (E, hi) if pw < power else (lo, E)
    return hi


# Dialed causal ladders (fixed by design, known a priori): the canonical analysis-view ladders (each
# includes the unbiased center), from which we take the level values.
BIAS_LEVELS = [lvl for _, lvl in BIAS_LADDER]
WIDTH_LEVELS = [lvl for _, lvl in WIDTH_LADDER]
REAL = REAL_FORECASTERS
OBS_METRIC = {"H1o_bias_obs": "bias_log", "H2o_width_obs": "sharpness", "H3o_quality_obs": "relative_wql"}


def load_forecaster_spreads(quality_path):
    """Per-forecaster mean of each observational predictor across baskets, from the basket-quality
    census. The OBSERVATIONAL slope tests (H1o/H2o/H3o) regress attainment on these measured values.
    Power for a fixed end-to-end attainment effect is ~range-independent, so the exact spread barely
    matters; grounding in the realized spread just makes the number faithful."""
    p = Path(quality_path)
    if not p.exists():
        raise SystemExit(f"{p} not found: build the shared per-basket quality first "
                         "(SHARED_ONLY=1 bash scripts/build_model_bundle.sh)")
    q = json.loads(p.read_text())
    out = {}
    for name, metric in OBS_METRIC.items():
        vals = []
        for fc in REAL:
            v = [q[s][fc][metric] for s in q if fc in q[s] and metric in q[s][fc]]
            if v:
                vals.append(float(np.mean(v)))
        if len(vals) >= 3 and (max(vals) - min(vals)) > 0:
            out[name] = vals
    return out


def slope_sizing(K, m, tests, *, baselines, S, alpha=0.05, target_power=0.80, target_attain=0.10, seed=1):
    """For each slope test, the pre-specified target is TARGET_ATTAIN attainment end-to-end across the
    predictor's range; translate to a slope (target_attain / range) and report a-priori power at it,
    plus the detectable-slope floor (MDE). ``tests`` maps name -> predictor x-levels."""
    out = {}
    for name, xl in tests.items():
        rng = float(max(xl) - min(xl))
        target_slope = target_attain / rng
        pw, fp = simulate_slope_power(K, m, target_slope, xlevels=xl, baselines=baselines, S=S,
                                      alpha=alpha, seed=seed)
        mde_a = mde_attain(K, m, xlevels=xl, baselines=baselines, S=S, alpha=alpha,
                           power=target_power, seed=seed)
        out[name] = {"range": rng, "target_attain": target_attain, "target_slope": target_slope,
                     "power_at_target": pw, "fp": fp, "mde_attain": mde_a,
                     "mde_slope": mde_a / rng, "n_levels": len(xl)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[6, 8, 10, 12, 16])
    ap.add_argument("--ms", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.075, 0.10, 0.125, 0.15])
    ap.add_argument("--S", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1, help="Monte-Carlo seed (matches the experiment seed)")
    ap.add_argument("--target-power", type=float, default=0.80)
    ap.add_argument("--target-attain", type=float, default=0.10,
                    help="pre-specified MDE in attainment units (end-to-end across each ladder)")
    ap.add_argument("--basket-quality", default="experiments/shared/basket_quality/quality.json",
                    help="census used for the observational predictors' realized spread (H1o/H2o/H3o)")
    ap.add_argument("--out", default="experiments/power")
    args = ap.parse_args()
    n_conds = len(reference_conditions())                  # episode cost = K x conditions x m

    print(f"baselines (empirical, n={len(EMPIRICAL_BASELINES)}): {EMPIRICAL_BASELINES}")
    print(f"S={args.S} sims, alpha=0.05, target power={args.target_power}\n")
    results = {}
    for delta in args.deltas:
        print(f"=== true effect Δ = {delta:+.2f} attainment ===")
        corner = "K\\m"                                    # kept out of the f-string: 3.10/3.11 reject
        print(f"{corner:>5}" + "".join(f"{m:>9}" for m in args.ms) + "   (cost: episodes)")
        for K in args.Ks:
            row = []
            for m in args.ms:
                pw, fp = simulate_power(K, m, delta, baselines=EMPIRICAL_BASELINES,
                                        S=args.S, seed=1000 * K + m)
                results[f"K{K}_m{m}_d{delta}"] = {"power": pw, "fp": fp,
                                                  "episodes": K * n_conds * m}
                row.append(f"{pw:>8.2f}")
            eps = [K * n_conds * m for m in args.ms]
            print(f"{K:>5}" + "".join(row) + f"   {eps}")
        print()

    # false-positive rate (should be ~alpha) reported once from the delta=0-equivalent checks
    fps = [v["fp"] for v in results.values()]
    print(f"mean false-positive rate under null across cells: {np.mean(fps):.3f} (target ≈ 0.05)")

    # smallest-cost cell meeting target power, per delta
    print("\ncheapest (K,m) meeting target power:")
    for delta in args.deltas:
        cells = [(k, v) for k, v in results.items() if k.endswith(f"d{delta}")
                 and v["power"] >= args.target_power]
        if cells:
            k, v = min(cells, key=lambda kv: kv[1]["episodes"])
            # parallel wall estimate: ~1.2 episodes/s at ~300 in-flight Bedrock requests
            print(f"  Δ={delta:+.2f}: {k.split('_d')[0]}  power={v['power']:.2f}  "
                  f"episodes={v['episodes']} (~{v['episodes']/1.2/60:.0f} min parallel)")
        else:
            print(f"  Δ={delta:+.2f}: none in grid reach {args.target_power}")

    # A-priori sizing for all slope trend tests at the locked design (K=16, m=25), against a single
    # pre-specified MDE of --target-attain attainment end-to-end. Causal ladders use their dialed
    # levels; observational tests (H1o/H2o/H3o) use the realized per-forecaster predictor spread.
    Klock, mlock = 16, 25
    tests = {"H1c_bias": BIAS_LEVELS, "H2c_width": WIDTH_LEVELS}
    tests.update(load_forecaster_spreads(args.basket_quality))
    print(f"\n=== a-priori sizing at locked K={Klock}, m={mlock}, "
          f"MDE={args.target_attain:.2f} attain (baseline ~= {np.mean(EMPIRICAL_BASELINES):.2f}) ===")
    h0_pw, h0_fp = simulate_power(Klock, mlock, args.target_attain, baselines=EMPIRICAL_BASELINES,
                                  S=args.S, seed=args.seed)
    print(f"  {'H0 (single cond.)':16} target {args.target_attain:.2f} attain gain vs no_forecast "
          f"-> power {h0_pw:.2f}; FP={h0_fp:.3f}")
    trend = slope_sizing(Klock, mlock, tests, baselines=EMPIRICAL_BASELINES, S=args.S,
                         target_power=args.target_power, target_attain=args.target_attain, seed=args.seed)
    trend["H0_single"] = {"power_at_target": h0_pw, "fp": h0_fp, "target_attain": args.target_attain}
    for name, v in trend.items():
        if name == "H0_single":
            continue
        print(f"  {name:16} target {v['target_attain']:.2f} attain (slope {v['target_slope']:.3f}) "
              f"-> power {v['power_at_target']:.2f}; floor {v['mde_attain']:.3f} attain; FP={v['fp']:.3f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "power.json").write_text(json.dumps(
        {"baselines": EMPIRICAL_BASELINES, "grid": results, "trend_sizing": trend,
         "params": {"S": args.S, "test": "paired_ttest_perbasket", "target_power": args.target_power,
                    "trend_test": "per_basket_OLS_slope_then_one_sample_t", "locked": {"K": Klock, "m": mlock}}},
        indent=2))
    print(f"\nwrote {out/'power.json'}")


if __name__ == "__main__":
    main()
