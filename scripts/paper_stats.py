"""
Single-source audit + power for every Vendster hypothesis.

Each hypothesis reduces to a ONE-SAMPLE t over K=16 per-basket statistics theta_b:
  H0   (baseline)      theta_b = mean_{10 model forecasters} attain[fc,b] - attain[no_forecast,b]
  H1c  (bias, causal)  theta_b = OLS slope of attain on signed bias over the 9-point bias grid
  H2c  (width, causal) theta_b = OLS slope of attain on the width multiplier over the 7-point grid
  H1o  (bias, obs)     theta_b = OLS slope of attain on measured log-bias across the 10 forecasters
  H2o  (width, obs)    theta_b = OLS slope of attain on measured interval width across the forecasters
  H3o  (quality, obs)  theta_b = OLS slope of attain on relWQL across the 10 forecasters

For each (hypothesis, LLM) we report the observed effect (mean theta_b), the paired/one-sample
t p-value, the family-wise Holm adjustment ACROSS THE FOUR LLMs, and power grounded in the observed
per-basket dispersion: power to detect the observed effect, and the minimum detectable effect (MDE) at
80% power. Power uses the noncentral-t (the observed SD(theta_b) already captures episode sampling +
basket heterogeneity, so no DGP re-simulation is needed). This reproduces the numbers in tab:quality / the
Results; the causal/observational slopes are cross-checked against experiments/panel/panel.json.

With --verify overleaf/main.tex, it recomputes EVERY numeric cell of the paper tables
(cross-model, robustness, the reactivity fingerprint, power/sizing, and the forecaster-quality
census) -- the behavioral slopes from panel.json, the power/MDE from power.json,
the census from forecaster_quality/report.json -- diffs each against the paper at its displayed precision,
and exits non-zero on any mismatch. This is the reviewer one-liner: confirm every reported number without
rerunning anything on Bedrock.

    uv run python scripts/paper_stats.py --out experiments/audit
    uv run python scripts/paper_stats.py --verify overleaf/main.tex   # check the paper matches the scripts
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats

from vendster.analysis.aggregate import BIAS_LADDER, WIDTH_LADDER  # canonical name->level maps
from vendster.analysis.stats import holm  # canonical Holm-Bonferroni
from vendster.core.config import EVALUATED_PANEL, REAL_FORECASTERS  # canonical panel + 10-forecaster roster

PANELS = {label: d for (label, d) in EVALUATED_PANEL.values()}       # display label -> experiment dir
REAL = REAL_FORECASTERS
BIAS_GRID = dict(BIAS_LADDER)
WIDTH_GRID = dict(WIDTH_LADDER)
ALPHA = 0.05
TARGET_POWER = 0.80
# Behavioral fingerprint: bias-ladder slope of each negotiation behavior (tab:reactivity, bias-slope rows), keyed by the
# panel.json field names. Different dependent variable from attainment, so each is sized in its
# own units (not the 0.10-attainment MDE).
BEHAV = ["first_offer_aggressiveness", "concession_rate", "mean_discount", "turns", "agree"]
# Robustness grid: attainment-on-measured-metric slope across the forecasters (tab:robust), relWQL primary.
METRICS = ["relative_wql", "mase", "mape", "smape", "rmse", "crps"]


def _attain(fp: Path):
    if not fp.exists():
        return np.nan
    ys = []
    for line in fp.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "error":
            continue
        y = r.get("target_met")
        if y is not None:
            ys.append(1.0 if y else 0.0)
    return float(np.mean(ys)) if ys else np.nan


def _slope(x, y):
    """OLS slope of y on x (ignores NaN); None if <2 valid points or zero x-variance."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _ttest(theta):
    theta = np.asarray([t for t in theta if t is not None and not np.isnan(t)], float)
    if len(theta) < 2:
        return dict(k=len(theta), effect=float("nan"), sd=float("nan"), p=float("nan"))
    t, p = stats.ttest_1samp(theta, 0.0)
    return dict(k=len(theta), effect=float(theta.mean()), sd=float(theta.std(ddof=1)), p=float(p))


def _power_at(effect, sd, k, alpha=ALPHA):
    """Two-sided one-sample t power to detect a true mean `effect` given SD and n=k."""
    if not np.isfinite(effect) or not np.isfinite(sd) or sd == 0 or k < 2:
        return float("nan")
    df = k - 1
    ncp = effect / (sd / np.sqrt(k))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    # P(|T| > tcrit) under noncentral t. For a large |ncp| scipy's nct underflows the far tail
    # (the one on the opposite side of the effect) to nan instead of ~0; that tail is negligible
    # there by construction, so treat a non-finite term as 0 rather than poisoning the sum.
    upper, lower = stats.nct.sf(tcrit, df, ncp), stats.nct.cdf(-tcrit, df, ncp)
    upper = upper if np.isfinite(upper) else 0.0
    lower = lower if np.isfinite(lower) else 0.0
    return float(min(1.0, upper + lower))


def _mde(sd, k, alpha=ALPHA, power=TARGET_POWER):
    """Minimum detectable effect (two-sided) at the given power, via bisection on |effect|."""
    if not np.isfinite(sd) or sd == 0 or k < 2:
        return float("nan")
    lo, hi = 0.0, 10 * sd + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _power_at(mid, sd, k, alpha) < power:
            lo = mid
        else:
            hi = mid
    return float(hi)


def per_basket_stats(run: Path, quality: dict):
    """Return dict hyp -> list of per-basket theta_b for one LLM run."""
    sets = sorted(run.glob("set_*"), key=lambda p: int(p.name.split("_")[1]))
    out = {h: [] for h in ["H0", "H1c", "H2c", "H1o", "H2o", "H3o", "_nf"]}
    for s in sets:
        si = s.name
        ep = s / "episodes"
        A = {c: _attain(ep / f"{c}.jsonl") for c in
             set(REAL) | set(BIAS_GRID) | set(WIDTH_GRID) | {"no_forecast"}}
        # raw no_forecast attainment level (tab:crossmodel column 1)
        if not np.isnan(A["no_forecast"]):
            out["_nf"].append(A["no_forecast"])
        # H0: mean real forecaster - no_forecast
        fcs = [A[c] for c in REAL if not np.isnan(A[c])]
        if fcs and not np.isnan(A["no_forecast"]):
            out["H0"].append(np.mean(fcs) - A["no_forecast"])
        # H1c / H2c: causal grids
        out["H1c"].append(_slope([BIAS_GRID[c] for c in BIAS_GRID], [A[c] for c in BIAS_GRID]))
        out["H2c"].append(_slope([WIDTH_GRID[c] for c in WIDTH_GRID], [A[c] for c in WIDTH_GRID]))
        # H1o / H2o / H3o: observational slopes across the 10 forecasters (measured metric per basket)
        q = quality.get(si, {})

        def mvec(metric):
            # `q` is read here, not captured for later: every mvec() call below happens in this
            # same iteration, before the next one rebinds it.
            return [q.get(c, {}).get(metric, np.nan) for c in REAL]  # noqa: B023
        yvec = [A[c] for c in REAL]
        out["H1o"].append(_slope(mvec("bias_log"), yvec))
        out["H2o"].append(_slope(mvec("sharpness"), yvec))
        out["H3o"].append(_slope(mvec("relative_wql"), yvec))
    return out


def _mde_from_stat(slope, p, n, power=TARGET_POWER, alpha=ALPHA):
    """Realized-precision MDE backed out of a reported (slope, two-sided p, n): SE = |slope| / t_obs,
    then MDE = (t_{1-a/2} + t_{power}) * SE. Used for the behavioral slopes, which are sourced from the
    canonical panel.json (slope + p) rather than recomputed (their agreed-only aggregation lives
    in make_panel)."""
    if slope is None or p is None or not (0 < p < 1) or slope == 0 or n < 2:
        return float("nan")
    t_obs = stats.t.ppf(1 - p / 2, n - 1)
    se = abs(slope) / t_obs if t_obs else float("nan")
    return float((stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(power, n - 1)) * se)


def robust_stats(run: Path, quality: dict):
    """Per metric: list of per-basket slopes of attainment on the measured metric across forecasters."""
    out = {mt: [] for mt in METRICS}
    for s in sorted(run.glob("set_*"), key=lambda p: int(p.name.split("_")[1])):
        ep, q = s / "episodes", quality.get(s.name, {})
        yvec = [_attain(ep / f"{c}.jsonl") for c in REAL]
        for mt in METRICS:
            out[mt].append(_slope([q.get(c, {}).get(mt, np.nan) for c in REAL], yvec))
    return out


def _family(raw_stats: dict):
    """Given {group: {LLM: stat}}, add Holm-adjusted p across the LLMs within each group."""
    for g in raw_stats:
        adj = holm({m: raw_stats[g][m]["p"] for m in PANELS})
        for m in PANELS:
            raw_stats[g][m]["p_holm"] = adj[m]
    return raw_stats


# ----------------------------------------------------------------------------------------------
# Paper verification: recompute every table cell and diff it against the paper (reviewer one-liner).
# ----------------------------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def _split_top(s, sep):
    """Split on `sep` only at brace-depth 0 (so `\\\\` inside \\makecell{...} is not a row break)."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1; cur.append(c); i += 1
        elif c == "}":
            depth -= 1; cur.append(c); i += 1
        elif depth == 0 and s.startswith(sep, i):
            out.append("".join(cur)); cur = []; i += len(sep)
        else:
            cur.append(c); i += 1
    out.append("".join(cur))
    return out


def _table_rows(tex, label):
    """Return {row_label: [cell_str, ...]} for the tabular carrying \\label{label} (body rows only)."""
    i = tex.find("\\label{" + label + "}")
    seg = tex[i:tex.find("\\end{table}", i)]
    body = seg[seg.find("\\midrule") + 8: seg.find("\\bottomrule")]
    rows = {}
    for raw in _split_top(body, "\\\\"):
        cells = [c.strip() for c in _split_top(raw, "&")]
        if len(cells) < 2 or not cells[0].strip():
            continue
        lbl = re.sub(r"\\texttt\{|\}|\\|\s", "", cells[0]).replace("_", "").lower()  # normalize label
        rows[lbl] = cells
    return rows


def _cell_nums(cell):
    """Ordered numeric tokens shown in a cell (strips *, bold/underline/scriptsize wrappers)."""
    t = cell.replace("^{*}", "").replace("\\,", " ")
    for w in ("\\textbf", "\\underline", "\\texttt", "\\scriptsize", "\\makecell"):
        t = t.replace(w, "")
    return _NUM.findall(t)


def _chk(records, table, row, col, shown, computed):
    """Compare a shown token to a computed value at the token's displayed precision (~0.6 ulp)."""
    d = len(shown.split(".")[1]) if "." in shown else 0
    ok = (computed is not None and np.isfinite(computed)
          and abs(computed - float(shown)) <= 0.6 * 10 ** (-d) + 1e-12)
    records.append((ok, table, row, col, shown, "nan" if computed is None or not np.isfinite(computed)
                    else f"{computed:.{d}f}"))


def verify_paper(tex, results, nf_level, behav, robust, quality, power, uplift):
    """Recompute every numeric table cell and diff against the paper. Returns list of check records."""
    rec = []
    label = {m: m.lower() for m in PANELS}

    # tab:crossmodel — no_forecast level + 6 (effect, raw p, Holm p) triples per LLM
    rows = _table_rows(tex, "tab:crossmodel")
    cm_cols = ["H0", "H1c", "H2c", "H1o", "H2o", "H3o"]
    for m in PANELS:
        cells = rows[label[m]]
        _chk(rec, "crossmodel", m, "no_forecast", _cell_nums(cells[1])[0], nf_level[m])
        for j, h in enumerate(cm_cols):
            n = _cell_nums(cells[2 + j]); r = results[h][m]
            for k, key in enumerate(["effect", "p", "p_holm"]):
                _chk(rec, "crossmodel", m, f"{h}.{key}", n[k], r[key])

    # tab:robust — (effect, raw p, Holm p) per metric x LLM
    rows = _table_rows(tex, "tab:robust")
    mlab = {"relwql": "relative_wql", "mase": "mase", "mape": "mape",
            "smape": "smape", "rmse": "rmse", "crps": "crps"}
    for texlab, mt in mlab.items():
        cells = rows[texlab]
        for c, m in enumerate(PANELS):
            n = _cell_nums(cells[1 + c]); r = robust[mt][m]
            for k, key in enumerate(["effect", "p", "p_holm"]):
                _chk(rec, "robust", mt, f"{m}.{key}", n[k], r[key])

    # tab:reactivity — combined fingerprint: two rows per model (presence uplift, then bias slope) over
    # the five behaviors. Rows come in model order; the behavior cells are the LAST five of each row
    # (after the multirow model column and the measure column), so multirow/blank prefixes don't matter.
    beh = ["first_offer_aggressiveness", "mean_discount", "concession_rate", "turns", "agree"]
    i = tex.find("\\label{tab:reactivity}")
    seg = tex[i:tex.find("\\end{table}", i)]
    body = re.sub(r"\\cmidrule[^\n]*", "", seg[seg.find("\\midrule") + 8: seg.find("\\bottomrule")])
    rowtexts = [r for r in _split_top(body, "\\\\") if r.strip()]
    order = [(m, src) for m in PANELS for src in ("presence", "slope")]     # Opus-pres, Opus-slope, GPT-...
    for rowtext, (m, src) in zip(rowtexts, order):
        bcells = _split_top(rowtext, "&")[-5:]
        stats = uplift if src == "presence" else behav
        for c, b in enumerate(beh):
            n = _cell_nums(bcells[c]); r = stats[b][m]
            for k, key in enumerate(["effect", "p", "p_holm"]):
                _chk(rec, f"react-{src}", m, f"{b}.{key}", n[k], r[key])

    # tab:quality — forecaster census (8 metrics)
    rows = _table_rows(tex, "tab:quality")
    qcols = ["relative_wql", "mase", "mape", "smape", "rmse", "crps", "coverage_80", "bias_log"]
    for fc, mvals in quality.items():
        key = fc.replace("_", "").lower()
        if key not in rows:
            continue
        cells = rows[key]
        for c, met in enumerate(qcols):
            _chk(rec, "quality", fc, met, _cell_nums(cells[1 + c])[0], mvals.get(met))

    # tab:power — a-priori power + detectable-effect floor at the locked design (from power.json)
    rows = _table_rows(tex, "tab:power")
    ts = power.get("trend_sizing", {})
    pmap = [("h0", "H0_single", False), ("h1c", "H1c_bias", True), ("h2c", "H2c_width", True),
            ("h1o", "H1o_bias_obs", True), ("h2o", "H2o_width_obs", True), ("h3o", "H3o_quality_obs", True)]
    for rlab, key, has_floor in pmap:
        if rlab not in rows or key not in ts:
            rec.append((False, "power", rlab, key, "?", "missing")); continue
        cells = rows[rlab]
        _chk(rec, "power", rlab, "power", _cell_nums(cells[2])[0], ts[key].get("power_at_target"))
        if has_floor:
            fn = _cell_nums(cells[3])
            if fn:
                _chk(rec, "power", rlab, "floor", fn[0], ts[key].get("mde_attain"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", default="experiments/panel")
    ap.add_argument("--basket-quality", default="experiments/shared/basket_quality/quality.json")
    ap.add_argument("--panel-json", default="experiments/panel/panel.json")
    ap.add_argument("--forecaster-quality", default="experiments/shared/forecaster_quality/report.json",
                    help="held-out quality census behind tab:quality (only read for --verify)")
    ap.add_argument("--power", default="experiments/power/power.json",
                    help="a-priori power/MDE behind tab:power (only read for --verify)")
    ap.add_argument("--verify", metavar="MAIN_TEX", default=None,
                    help="recompute every table cell and diff against this main.tex (use overleaf/main.tex); "
                         "exit 1 on mismatch")
    ap.add_argument("--out", default="experiments/audit")
    args = ap.parse_args()

    quality = json.loads(Path(args.basket_quality).read_text())
    xm = json.loads(Path(args.panel_json).read_text()) if Path(args.panel_json).exists() else {}
    xkey = {"Opus": "opus", "GPT": "gpt", "Qwen": "qwen", "DeepSeek": "deepseek"}

    results = {h: {} for h in ["H0", "H1c", "H2c", "H1o", "H2o", "H3o"]}
    nf_level = {}
    for m, d in PANELS.items():
        pb = per_basket_stats(Path(args.panel_root) / d / "experiment", quality)
        nf_level[m] = float(np.mean(pb["_nf"])) if pb["_nf"] else float("nan")
        for h in results:
            st = _ttest(pb[h])
            st["power_at_effect"] = _power_at(st["effect"], st["sd"], st["k"])
            st["mde_80"] = _mde(st["sd"], st["k"])
            results[h][m] = st

    # Holm across the four LLMs, per hypothesis
    for h in results:
        adj = holm({m: results[h][m]["p"] for m in PANELS})
        for m in PANELS:
            results[h][m]["p_holm"] = adj[m]

    # Behavioral fingerprint (tab:reactivity, bias-slope rows) and robustness grid (tab:robust): per-basket slopes,
    # then Holm across the four LLMs within each behavior / metric family.
    behav = {b: {} for b in BEHAV}
    robust = {mt: {} for mt in METRICS}
    for m, d in PANELS.items():
        # Behavioral slopes come from the canonical panel.json (correct agreed-only aggregation);
        # we add the family-wise Holm and a realized-precision MDE.
        bd = xm.get(xkey[m], {}).get("behav", {})
        for b in BEHAV:
            s = bd.get(b, {})
            behav[b][m] = {"effect": s.get("slope"), "p": s.get("p"), "n": s.get("n", 16),
                           "mde_80": _mde_from_stat(s.get("slope"), s.get("p"), s.get("n", 16))}
        rs = robust_stats(Path(args.panel_root) / d / "experiment", quality)
        for mt in METRICS:
            st = _ttest(rs[mt]); st["mde_80"] = _mde(st["sd"], st["k"]); robust[mt][m] = st
    _family(behav); _family(robust)

    # Presence uplift on each behavior (tab:reactivity, presence rows): per-basket (mean real forecaster - no_forecast),
    # from panel.json, Holm across the four models within each behavior (parallel to the slopes).
    uplift = {b: {} for b in BEHAV}
    for m in PANELS:
        bu = xm.get(xkey[m], {}).get("behav_uplift", {})
        for b in BEHAV:
            s = bu.get(b, {})
            uplift[b][m] = {"effect": s.get("delta"), "p": s.get("p"), "n": s.get("n", 16)}
    _family(uplift)

    # cross-check causal/observational slopes against panel.json
    check = []
    for h, key in [("H1c", "H1c"), ("H2c", "H2c"), ("H1o", "H1o"), ("H3o", "H3o")]:
        for m in PANELS:
            ours = results[h][m]["effect"]
            ref = xm.get(xkey[m], {}).get(key, {}).get("slope")
            if ref is not None and np.isfinite(ours):
                check.append((h, m, ours, ref, abs(ours - ref)))

    # print
    print(f"{'hyp':4} {'LLM':9} {'effect':>9} {'raw p':>10} {'Holm p':>10} "
          f"{'pow@eff':>8} {'MDE@80%':>9} K")
    for h in ["H0", "H1c", "H2c", "H1o", "H2o", "H3o"]:
        for m in PANELS:
            r = results[h][m]
            print(f"{h:4} {m:9} {r['effect']:+9.3f} {r['p']:>10.2e} {r['p_holm']:>10.2e} "
                  f"{r['power_at_effect']:>8.2f} {r['mde_80']:>9.3f} {r['k']}")
        print()
    if check:
        worst = max(c[4] for c in check)
        print(f"cross-check vs panel.json: max |slope diff| = {worst:.4f} "
              f"({'OK' if worst < 0.02 else 'MISMATCH'})")

    print("\n== behavioral fingerprint (tab:reactivity, bias-slope rows): slope on bias ladder, Holm across LLMs, MDE (own units) ==")
    print(f"{'behavior':13} {'LLM':9} {'slope':>9} {'raw p':>10} {'Holm p':>9} {'MDE@80%':>9}")
    for b in BEHAV:
        for m in PANELS:
            r = behav[b][m]
            print(f"{b:13} {m:9} {r['effect']:+9.3f} {r['p']:>10.2e} {r['p_holm']:>9.3f} {r['mde_80']:>9.3f}")
    print("\n== robustness grid (tab:robust): attainment-on-metric slope, Holm per metric across LLMs ==")
    print(f"{'metric':13} " + "".join(f"{m:>22}" for m in PANELS) + "   (slope (raw p, Holm p))")
    for mt in METRICS:
        print(f"{mt:13} " + "".join(
            f"{robust[mt][m]['effect']:+.3f} ({robust[mt][m]['p']:.3f},{robust[mt][m]['p_holm']:.3f})".rjust(22)
            for m in PANELS))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(
        {"hypotheses": results, "behavioral": behav, "presence_uplift": uplift,
         "robust": robust, "no_forecast": nf_level},
        indent=2, default=str))
    print(f"\nwrote {out/'audit.json'}")

    if args.verify:
        tex = Path(args.verify).read_text()
        qual = json.loads(Path(args.forecaster_quality).read_text())["metrics"]
        pwr = json.loads(Path(args.power).read_text())
        rec = verify_paper(tex, results, nf_level, behav, robust, qual, pwr, uplift)
        fails = [r for r in rec if not r[0]]
        print(f"\n== verify against {args.verify}: {len(rec)-len(fails)}/{len(rec)} cells match ==")
        if fails:
            print(f"{'table':11} {'row':10} {'field':28} {'paper':>9} {'computed':>10}")
            for _, table, row, col, shown, comp in fails:
                print(f"{table:11} {row:10} {col:28} {shown:>9} {comp:>10}")
            print(f"\nFAIL: {len(fails)} cell(s) disagree with the scripts.")
            sys.exit(1)
        print("OK: every table cell in the paper is reproduced by the scripts.")


if __name__ == "__main__":
    main()
