"""
Build one model's tables, figures, and robustness panel from a paired experiment run.

Outputs (under --out): tables.md, aggregate.json, link.json, figures/*.png, examples.md, and
robustness_metrics.{md,json} — the observational "does a real forecaster's error property predict
attainment?" panel over six accuracy metrics (relWQL/MASE/MAPE/sMAPE/RMSE/CRPS) and three
calibration properties (signed bias, interval width, coverage), each under three lenses:
the design-correct basket-clustered slope (the reported test), a naive pooled OLS (shown only to
flag the clustering trap), and a forecaster-level Pearson over per-forecaster means.

    uv run python scripts/make_one_model.py --run experiments/panel/opus_4_6/experiment \
        --pool-quality experiments/shared/forecaster_quality/report.json \
        --basket-quality experiments/shared/basket_quality/quality.json \
        --out experiments/panel/opus_4_6/paper

All inputs except --run are optional; the relevant figures/columns are simply omitted if
absent, so this works on partial runs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import pearsonr, ttest_1samp

from vendster.analysis import aggregate as agg_mod
from vendster.analysis import figures as figs
from vendster.analysis import link as link_mod
from vendster.analysis import stats as stats_mod
from vendster.analysis.transcript import render_run_examples
from vendster.core.config import REAL_FORECASTERS

COLS = [("attain", "attain"), ("agree", "agree"), ("turns", "turns"), ("sales", "sales"),
        ("profit", "profit"), ("eff", "eff"), ("regret", "regret"),
        ("first_offer", "1stOff"), ("concession", "conc"),
        ("mean_discount", "disc"), ("sales_target", "s/tgt")]


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


ACCURACY_METRICS = ["relative_wql", "mase", "mape", "smape", "rmse", "crps"]
PROPERTIES = [("bias_log", "signed bias"), ("sharpness_norm", "interval width"),
              ("coverage_80", "coverage@80")]


def _obs_slopes(agg, quality, key, forecasters):
    """Observational test: does a real forecaster's per-basket ``key`` (from basket_quality)
    predict its attainment on that basket? Three lenses over the (forecaster, basket) points:

    - design-correct: per basket, OLS slope of attainment on ``key`` across the forecasters,
      then a one-sample t over baskets (basket = unit of analysis). This is the reported test,
      matching the causal grid's basket-clustered trend tests (H1o/H2o/H3o).
    - pooled: one OLS over all points, ignoring clustering — anti-conservative; kept only to
      show the trap.
    - forecaster-level: Pearson over per-forecaster means (n = #forecasters).
    """
    attain = {c: {r["set_id"]: r["attain"] for r in agg["per_basket"].get(c, [])} for c in forecasters}
    slopes, X, Y = [], [], []
    for s in quality:
        xs, ys = [], []
        for c in forecasters:
            m = quality[s].get(c, {}).get(key)
            y = attain.get(c, {}).get(s)
            if m is not None and y is not None and not np.isnan(y):
                xs.append(m); ys.append(y)
        X += xs; Y += ys
        if len(xs) >= 3 and np.ptp(xs) > 0:
            slopes.append(float(np.polyfit(xs, ys, 1)[0]))
    nan = float("nan")
    out = {"design_slope": nan, "design_p": nan, "n_baskets": len(slopes),
           "pooled_slope": nan, "pooled_p": nan, "fc_pearson": nan, "fc_p": nan,
           "fc_means": {}, "spread": [nan, nan]}
    if len(slopes) >= 2:
        out["design_slope"] = float(np.mean(slopes)); out["design_p"] = float(ttest_1samp(slopes, 0.0)[1])
    if len(X) >= 3 and np.ptp(X) > 0:
        out["pooled_slope"] = float(np.polyfit(X, Y, 1)[0]); out["pooled_p"] = float(pearsonr(X, Y)[1])
        out["spread"] = [float(min(X)), float(max(X))]
    fx, fy = [], []
    for c in forecasters:
        ms = [quality[s][c][key] for s in quality if key in quality[s].get(c, {})]
        ys = [v for v in attain.get(c, {}).values() if not np.isnan(v)]
        if ms and ys:
            fx.append(float(np.mean(ms))); fy.append(float(np.mean(ys))); out["fc_means"][c] = fx[-1]
    if len(fx) >= 3 and np.ptp(fx) > 0:
        r, p = pearsonr(fx, fy); out["fc_pearson"] = float(r); out["fc_p"] = float(p)
    return out


def _robustness_md(acc_rows, prop_rows, n_fc):
    """Plain tables, no generated prose — the manuscript draws its own conclusions."""
    def cell(r, lens):
        s, p = {"design": ("design_slope", "design_p"), "pooled": ("pooled_slope", "pooled_p"),
                "fc": ("fc_pearson", "fc_p")}[lens]
        return "—" if np.isnan(r[s]) else f"{r[s]:+.4f} (p={r[p]:.3f})"
    L = ["# Robustness — do real forecasters' error properties predict attainment?", "",
         "Per-(forecaster, basket) points: x = the forecaster's offline error property on that basket "
         "(from basket_quality), y = its attainment on that basket. **Design-correct** = per-basket OLS "
         "slope across forecasters, one-sample t over baskets (basket = unit of analysis; the reported "
         "test). **Pooled** ignores clustering and is anti-conservative (shown only to flag the trap). "
         f"**Forecaster-level** = Pearson over per-forecaster means (n={n_fc}).", "",
         "## Accuracy metrics (lower = better forecast; negative slope = better accuracy → higher attainment)", "",
         "| metric | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |", "|---|---|---|---|"]
    for r in acc_rows:
        L.append(f"| {r['metric']} | {cell(r, 'design')} | {cell(r, 'pooled')} | {cell(r, 'fc')} |")
    L += ["", "## Calibration properties (observational analog of the perturbation grid: H1o bias, H2o width)", "",
          "| property | real-forecaster range | design-correct slope (p) | pooled slope (p) | forecaster-level r (p) |",
          "|---|---|---|---|---|"]
    for r in prop_rows:
        lo, hi = r["spread"]
        rng = "—" if np.isnan(lo) else f"[{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| {r['property']} | {rng} | {cell(r, 'design')} | {cell(r, 'pooled')} | {cell(r, 'fc')} |")
    return "\n".join(L) + "\n"


def _table(title, conds, agg, tests):
    # Holm-Bonferroni WITHIN this table's contrast family (conditions tested vs no_forecast).
    fam = {c: tests[c]["p_value"] for c in conds if c in tests and tests[c]["n_baskets"]}
    padj = stats_mod.holm(fam)
    L = [f"### {title}", "",
         "| condition | n | attain (95% CI) | Δ vs no_fc | p (Holm) | " +
         " | ".join(h for _, h in COLS[1:]) + " |",
         "|" + "---|" * (5 + len(COLS[1:]))]
    for c in conds:
        cc = agg["conditions"][c]
        lo, hi = cc["attain_ci95"]
        ci = f"{_f(cc['attain_pooled'])} [{_f(lo)},{_f(hi)}]"
        if c in padj:
            delta = f"{tests[c]['delta']:+.2f}{stats_mod.stars(padj[c])}"
            padj_s = _f(padj[c], 3)
        else:
            delta, padj_s = "— (ref)", "—"
        cells = [_f(cc[k]["mean"], 0 if k in ("sales", "profit", "regret", "turns") else 2)
                 for k, _ in COLS[1:]]
        L.append(f"| {c} | {cc['n_episodes']} | {ci} | {delta} | {padj_s} | "
                 + " | ".join(cells) + " |")
    L += ["", "_Δ = paired within-basket attainment gain vs no_forecast (paired t-test on "
          "per-basket deltas); p = Holm-adjusted within this table's family; "
          "* p<.05, ** p<.01, *** p<.001._", ""]
    return "\n".join(L)


def _forecaster_properties(args):
    """Accuracy, SIGNED bias (log), and calibration per real forecaster — pool-level census and
    per-basket-5 (mean±sd over baskets). Locates real forecasters on the synthetic grid's
    (bias, coverage) axes: they should sit near the center (near-unbiased, ~calibrated), so
    the grid's bias/width extremes are stress-tests beyond the real range."""
    if not (args.pool_quality and Path(args.pool_quality).exists()):
        return ""
    pool = json.loads(Path(args.pool_quality).read_text()).get("metrics", {})
    qb = (json.loads(Path(args.basket_quality).read_text())
          if args.basket_quality and Path(args.basket_quality).exists() else {})

    def per_basket(fc, key):
        vals = [qb[s][fc][key] for s in qb if fc in qb.get(s, {}) and key in qb[s][fc]]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    L = ["### Forecaster accuracy, signed bias & calibration (bridge to the grid)", "",
         "| forecaster | relWQL | bias_log (pool) | bias_log per-basket (mean±sd) | "
         "cov@80 (pool) | cov@80 per-basket (mean±sd) |", "|" + "---|" * 6]
    for fc in REAL_FORECASTERS:
        if fc not in pool:
            continue
        bm, bs = per_basket(fc, "bias_log")
        cm, cs = per_basket(fc, "coverage_80")
        L.append(f"| {fc} | {_f(pool[fc].get('relative_wql'), 3)} | "
                 f"{_f(pool[fc].get('bias_log'), 3)} | {_f(bm, 3)}±{_f(bs, 3)} | "
                 f"{_f(pool[fc].get('coverage_80'), 2)} | {_f(cm, 2)}±{_f(cs, 2)} |")
    L += ["", "_bias_log = mean signed log-forecast-error (>0 over-estimate); real forecasters "
          "are near-unbiased & near-calibrated, so they sit near the grid center — the signed-bias "
          "(±{.05,.10,.15,.20}) and interval-width (×{.25,.5,.75,1.5,2,3}) rungs probe failure modes "
          "beyond the real range._", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool-quality", default=None,
                    help="experiments/shared/forecaster_quality/report.json (pool-level quality census)")
    ap.add_argument("--basket-quality", default=None)
    ap.add_argument("--metric", default="relative_wql", choices=ACCURACY_METRICS,
                    help="accuracy axis for the quality->outcome figure and the H3o summary slope")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"

    agg = agg_mod.aggregate(args.run)
    tests = stats_mod.compare_to_baseline(agg, baseline="no_forecast")
    # PRIMARY causal tests: basket-clustered dose-response slopes along each ladder. The bias ladder
    # spans the empirically-realistic range (|bias_log| <= 0.20), where the dose-response is monotone,
    # so a single linear slope is the right test (H2: under-forecasting helps, over hurts).
    bias_trend = stats_mod.trend_test(*agg_mod.ladder_matrix(agg, agg_mod.BIAS_LADDER))
    width_trend = stats_mod.trend_test(*agg_mod.ladder_matrix(agg, agg_mod.WIDTH_LADDER))
    # CONTROL (agreement confound): attainment must move via discounting, NOT deal-failure.
    # If agreement rate itself trends with the manipulation, the attainment trend is confounded.
    bias_agree = stats_mod.trend_test(*agg_mod.ladder_matrix(agg, agg_mod.BIAS_LADDER, metric="agree"))
    width_agree = stats_mod.trend_test(*agg_mod.ladder_matrix(agg, agg_mod.WIDTH_LADDER, metric="agree"))
    # OBSERVATIONAL trends (real forecasters) — same three error dimensions as the causal grid
    # (H1o signed bias, H2o width, H3o accuracy), so causal and observational are directly
    # comparable. The full panel (all six accuracy metrics + coverage, three lenses) goes to
    # robustness_metrics.{json,md}; the three headline slopes also go into aggregate.json.
    h1o = h2o = h3o = {"n_baskets": 0}
    acc_rows, prop_rows = [], []
    if args.basket_quality and Path(args.basket_quality).exists():
        _bq = json.loads(Path(args.basket_quality).read_text())
        reals = [c for c in REAL_FORECASTERS if c in agg["conditions"]]
        acc_rows = [{"metric": m, **_obs_slopes(agg, _bq, m, reals)} for m in ACCURACY_METRICS]
        prop_rows = [{"property": lbl, "key": k, **_obs_slopes(agg, _bq, k, reals)} for k, lbl in PROPERTIES]
        def _slim(r):   # aggregate.json trend shape (matches the causal trend_test records)
            return {"slope": r["design_slope"], "p_value": r["design_p"], "n_baskets": r["n_baskets"]}
        h1o = _slim(next(r for r in prop_rows if r["key"] == "bias_log"))
        h2o = _slim(next(r for r in prop_rows if r["key"] == "sharpness_norm"))
        h3o = _slim(next(r for r in acc_rows if r["metric"] == args.metric))
        (out / "robustness_metrics.json").write_text(json.dumps(
            {"accuracy_metrics": acc_rows, "properties": prop_rows,
             "n_forecasters": len(reals), "n_baskets": agg["n_baskets"]}, indent=2, default=str))
        (out / "robustness_metrics.md").write_text(_robustness_md(acc_rows, prop_rows, len(reals)))
    (out / "aggregate.json").write_text(json.dumps(
        {"aggregate": agg, "paired_tests": tests,
         "trends": {"H1c_bias_causal": bias_trend, "H2c_width_causal": width_trend,
                    "H1o_bias_obs": h1o, "H2o_width_obs": h2o, "H3o_accuracy_obs": h3o},
         "agreement_confound": {"bias_agree_trend": bias_agree, "width_agree_trend": width_agree}},
        indent=2, default=str))

    # Outcome tables (functional names only — the manuscript numbers its own floats)
    md = [f"# Vendster — results ({agg['n_baskets']} baskets, {agg['layout']})", ""]
    md.append(_table("Outcomes by real-forecaster condition", agg["real"], agg, tests))
    md.append(_table("Outcomes on the perturbation grid", agg["grid"], agg, tests))
    md.append(_forecaster_properties(args))          # accuracy, signed bias, calibration
    (out / "tables.md").write_text("\n".join(md))

    made = []
    baseline_attain = agg["conditions"]["no_forecast"]["attain_pooled"]

    # quality -> outcome scatter (needs the pool quality census)
    if args.pool_quality and Path(args.pool_quality).exists():
        q = json.loads(Path(args.pool_quality).read_text()).get("metrics", {})
        attain = {fc: agg["conditions"][fc]["attain_pooled"] for fc in REAL_FORECASTERS
                  if fc in agg["conditions"]}
        lfc = link_mod.forecaster_level(q, attain, metric=args.metric)
        lbasket = None
        if args.basket_quality and Path(args.basket_quality).exists():
            qb = json.loads(Path(args.basket_quality).read_text())
            pts = link_mod.build_basket_points(agg, qb, REAL_FORECASTERS, metric=args.metric)
            lbasket = link_mod.basket_level(pts, metric=args.metric)
        made += figs.quality_outcome(lfc, baseline_attain, figdir, link_basket=lbasket)
        (out / "link.json").write_text(json.dumps(
            {"forecaster_level": lfc, "basket_level": lbasket}, indent=2, default=str))

    # perturbation grid, mechanism traces, cross-basket spread, welfare
    made += figs.perturbation_grid(agg, figdir, bias_trend=bias_trend, width_trend=width_trend)
    made += figs.mechanism(agg, figdir)
    made += figs.cross_basket_spread(agg, agg["real"] + agg["grid"], figdir)
    made += figs.welfare(agg, agg["real"], figdir)

    # example transcripts
    (out / "examples.md").write_text(render_run_examples(args.run, agg["real"] + agg["grid"]))

    def _tr(t):
        return (f"slope={t['slope']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] "
                f"p={t['p_value']:.3f} (K={t['n_baskets']})") if t["n_baskets"] > 1 else "n/a"
    def _ot(t):  # observational trend printer (per-basket OLS slope -> one-sample t)
        return (f"slope={t['slope']:+.3f} p={t['p_value']:.3f} (K={t['n_baskets']})"
                if t.get("n_baskets", 0) > 1 else "n/a")
    print("CAUSAL trends — perturbation grid (attainment):")
    print(f"  H1c bias dose-response (realistic |bias|<=0.20): {_tr(bias_trend)}")
    print(f"  H2c interval-width dose-response:                {_tr(width_trend)}")
    print("OBSERVATIONAL trends — real forecasters (per-basket slope, attainment vs error property):")
    print(f"  H1o signed bias  (bias_log -> attain):          {_ot(h1o)}")
    print(f"  H2o interval width (sharpness_norm -> attain):  {_ot(h2o)}")
    print(f"  H3o accuracy ({args.metric} -> attain; neg=better fc): {_ot(h3o)}")
    if acc_rows:
        print("  all accuracy metrics (design-correct slope, p):")
        for r in acc_rows:
            print(f"    {r['metric']:13} {_ot(_slim(r))}")
    print("CONTROL — agreement-rate trend (want n.s.; else attainment moves via deal-failure):")
    print(f"  bias  : {_tr(bias_agree)}")
    print(f"  width : {_tr(width_agree)}")
    print(f"tables  -> {out/'tables.md'}" + (f" | robustness -> {out/'robustness_metrics.md'}" if acc_rows else ""))
    print(f"examples-> {out/'examples.md'}")
    print(f"figures -> {figdir}/ ({len([m for m in made if m.endswith('.png')])} figs)")
    for c in agg["real"]:
        cc = agg["conditions"][c]
        t = tests.get(c, {})
        print(f"  {c:16} attain {_f(cc['attain_pooled'])} "
              f"[{_f(cc['attain_ci95'][0])},{_f(cc['attain_ci95'][1])}]  "
              f"Δ={_f(t.get('delta'))}{stats_mod.stars(t.get('p_value', float('nan')))}")


if __name__ == "__main__":
    main()
