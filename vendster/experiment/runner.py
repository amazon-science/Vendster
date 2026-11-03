"""
Target-setting and experiment harness.

Target: each basket's sales target = P80 of its reference-policy total-sales distribution
(analytic, no LLM; ``analytic_target``) — the challenging operating point (see the
paper's Environment Setup appendix). It is deterministic from (seed, basket), so every model is scored against
the same targets.

Experiment: the reference conditions (26 by default, or a caller-supplied list) x
episodes_per_condition, run paired within each basket. Per-episode structured logs (JSONL)
support resume (completed episodes are skipped). Forecasters are built per stretch with the
leakage-safe cutoff. Episodes parallelize (episode + cell concurrency) with a global lock
serializing non-thread-safe forecaster inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core.config import reference_conditions
from ..core.registry import FORECASTERS
from ..core.task_core import DISCOUNT_LEVELS
from ..core.types import Dataset
from ..dgp.demand import ElasticityDemandModel, cv_to_sigma, realize_event
from ..dgp.stretch_pool import StretchWindow, build_stretch_pool, sample_stretch
from ..negotiation import Episode

_SEED_BASE = 1_000_003

# Global lock serializing non-thread-safe forecaster inference (the shared Chronos GPU pipeline,
# statsforecast/LightGBM fits) and transcript appends. It is MODULE-LEVEL — not per-call — so it
# holds even when multiple baskets/sets run concurrently (outer parallelism). The slow Bedrock
# negotiation runs OUTSIDE it, so this does not bottleneck the parallelism we care about.
_SETUP_LOCK = threading.Lock()


def _episode_seed(seed: int, tag: str, i: int) -> int:
    # Stable across processes: Python's built-in hash() is salted per-process (PYTHONHASHSEED),
    # which would break the seeded-DGP reproducibility contract. blake2b is deterministic.
    tag_hash = int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=4).digest(), "big")
    return (seed * _SEED_BASE + tag_hash % 100000 + i) % (2**32)


def forecaster_context(dataset: Dataset, train_cutoff) -> dict:
    """All context a forecaster might need, keyed by capability name. A forecaster class declares
    which keys it needs via its ``REQUIRES`` attribute; the harness injects only those. This is the
    generic seam that replaces name-matching — a new panel-trained model just sets
    ``REQUIRES = ("panel", "exog", "train_cutoff")`` and needs no harness edits."""
    return {
        "panel": dataset.train_panel,
        "exog": dataset.exog,
        "train_cutoff": train_cutoff,
        "true_weekly": dataset.weekly,
        "sigmas": {p.item_id: cv_to_sigma(p.base_demand_cv) for p in dataset.products},
    }


def make_forecaster(name: str, params: dict, context: dict, cache: dict | None = None):
    """Capability-driven construction: inject the context keys the forecaster CLASS declares in
    ``REQUIRES`` (empty for zero-shot/per-series models), plus its config ``params``. Cache key
    includes ``train_cutoff`` only for cutoff-dependent (panel-trained) models so they refit once
    per cutoff; others cache by (name, params)."""
    cls = FORECASTERS.get(name)
    requires = tuple(getattr(cls, "REQUIRES", ()))
    kwargs = {k: context[k] for k in requires}
    cutoff = context.get("train_cutoff") if "train_cutoff" in requires else None
    key = (name, tuple(sorted(params.items())), cutoff)
    if cache is not None and key in cache:
        return cache[key]
    fc = FORECASTERS.create(name, **kwargs, **params)
    if cache is not None:
        cache[key] = fc
    return fc


def build_forecaster_for(cond, stretch: StretchWindow, dataset: Dataset, cache: dict):
    """Build the forecaster for a condition at a stretch (leakage-safe cutoff). Fully generic —
    no per-forecaster name-matching; see ``make_forecaster`` / ``forecaster_context``."""
    if cond.forecaster is None:
        return None
    ctx = forecaster_context(dataset, stretch.train_cutoff)
    return make_forecaster(cond.forecaster.name, dict(cond.forecaster.params), ctx, cache)


def _manifest(model_name: str, extra: dict) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    return {
        "git_sha": sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "model": model_name,
        "note": "LLM episodes are NOT bit-reproducible (Bedrock has no seed); the "
                "reproducibility contract is the frozen target + seeded DGP + archived transcripts.",
        **extra,
    }


def _episode_record(r) -> dict:
    """Compact summary (no transcript) for fast aggregation/resume — the full turn-by-turn
    negotiation is archived separately by ``_write_transcript``."""
    d = asdict(r)
    d.pop("communication_log", None)
    d.pop("context", None)
    return d


def _write_transcript(out: Path, cond_id: str, r) -> None:
    """Archive the FULL turn-by-turn negotiation (communication_log) so every episode is
    auditable and can be rendered as an appendix example. One JSONL line per episode; kept
    separate from the compact summary so aggregation stays cheap. Bedrock has no seed, so a
    lost transcript is unrecoverable — this is the provenance record."""
    tpath = out / "transcripts" / f"{cond_id}.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    rec = {"episode_id": r.episode_id, "condition_id": r.condition_id, "status": r.status,
           "target": r.target, "target_met": r.target_met, "agreement": r.agreement,
           "turns": r.turns, "agreed_discounts": r.agreed_discounts,
           "total_sales": r.total_sales, "vendor_profit": r.vendor_profit,
           "context": getattr(r, "context", None),
           "communication_log": r.communication_log}
    with tpath.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def analytic_target(dataset: Dataset, *, n: int = 2000, seed: int = 0, percentile: float = 80.0,
                    demand_model=None, min_history: int = 60,
                    discount_levels=DISCOUNT_LEVELS) -> dict:
    """Cheap per-basket sales target (no LLM) -> enables large K.

    Target = P{percentile} of the basket's total-sales distribution under a REFERENCE
    POLICY (a uniform-random allowed discount per product — the environment's own naive
    baseline, matching ``generate_past_events``), realized over the SAME DGP pathway
    (``realize_event`` at the stretch's month-4, single eps draw) as real episodes. Because
    the target is the P{percentile} of this distribution, reference attainment is ~(100-XX)%
    by construction. ``percentile`` defaults to P80 — the "challenging threshold" operating
    point where a forecast becomes decision-relevant; it was selected
    by an effect-blind difficulty sweep (no_forecast attainment ~= 0.30), superseding the earlier
    P50 "maximum-sensitivity" rationale, which the sweep empirically falsified.
    """
    pool = build_stretch_pool(dataset, min_history=min_history)
    dm = demand_model or ElasticityDemandModel()
    sales = []
    for i in range(n):
        rng = np.random.default_rng(_episode_seed(seed, "refpol", i))
        stretch = sample_stretch(pool, rng)
        discounts = {p.item_id: float(rng.choice(discount_levels)) for p in dataset.products}
        ev = realize_event(dm, dataset, discounts, stretch.m4, rng)
        sales.append(float(ev["total_sales"]))
    D0 = sorted(sales)
    return {
        "target": float(np.percentile(D0, percentile)), "percentile": percentile, "n": n,
        "policy": "uniform_random_discount", "pool_size": len(pool),
        "D0_summary": {"min": D0[0], "p25": float(np.percentile(D0, 25)),
                       "p50": float(np.percentile(D0, 50)), "p75": float(np.percentile(D0, 75)),
                       "max": D0[-1], "mean": float(np.mean(D0))},
    }


def _recenter_hint(pooled_attain: float, percentile: float) -> dict:
    """Offset of the LLM no-forecast baseline from the by-construction reference-policy attainment
    (100-XX)%. Informational only — the operating point is chosen effect-blind (see the paper),
    NOT by centring the baseline (the falsified P50 max-sensitivity rationale)."""
    ref = 1.0 - percentile / 100.0
    if np.isnan(pooled_attain):
        return {"reference_attainment": round(ref, 3), "note": "no valid episodes"}
    return {"reference_attainment": round(ref, 3),
            "note": f"LLM no-forecast baseline={pooled_attain:.0%} vs reference-policy {ref:.0%} "
                    f"(offset {pooled_attain-ref:+.0%})"}


def _wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    """Wilson score 95% CI for a binomial proportion (no scipy needed)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _binom_p_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test p-value against p0 (scipy if available)."""
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p0, alternative="two-sided").pvalue)
    except Exception:                       # normal approximation fallback
        se = math.sqrt(p0 * (1 - p0) / n)
        z = abs(k / n - p0) / se if se > 0 else 0.0
        return float(math.erfc(z / math.sqrt(2)))


def validate_xx(source, llm_client, *, n_baskets: int = 3, episodes: int = 10,
                percentile: float = 80.0, seed: int = 0, ref_draws: int = 2000,
                demand_model=None, min_history: int = 60,
                out_dir: str = "experiments/validate_xx", model_name: str = "mock",
                concurrency: int = 1, set_concurrency: int = 1) -> dict:
    """Empirically check that analytic P{percentile} targets place the LLM no-forecast baseline
    near the by-construction reference-policy attainment (100-XX)%. Runs no-forecast episodes on
    ``n_baskets`` and reports POOLED attainment with a Wilson 95% CI and a two-sided binomial test
    against that reference expectation, the per-basket breakdown (basket as unit: mean +/- SD), and
    the LLM-vs-reference offset. Cheap: only no-forecast episodes. (The P80 operating point is chosen
    effect-blind (see the paper); this validator confirms target CALIBRATION, it does
    not select the percentile.)"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dm = demand_model or ElasticityDemandModel()
    nf = next(c for c in reference_conditions() if c.id == "no_forecast")

    def _one_basket(si):
        basket = source.sample_basket(np.random.default_rng(seed * 7919 + si))
        dataset = source.build_dataset(basket, cost_seed=si)
        tinfo = analytic_target(dataset, n=ref_draws, seed=seed * 100 + si,
                                percentile=percentile, demand_model=dm, min_history=min_history)
        set_dir = out / f"set_{si}"
        pool = build_stretch_pool(dataset, min_history=min_history)
        recs = _run_condition(nf, dataset, pool, tinfo["target"], llm_client, episodes,
                              seed * 100 + si, set_dir, {}, True, dm, concurrency)
        got = [r for r in recs if r.get("target_met") is not None]
        succ = sum(1 for r in got if r["target_met"])
        lo, hi = _wilson_ci(succ, len(got))
        return si, {"target": tinfo["target"], "successes": succ, "n": len(got),
                    "attainment": (succ / len(got)) if got else float("nan"),
                    "ci95": [lo, hi]}

    res = {}
    sc = max(1, set_concurrency)
    if sc > 1 and n_baskets > 1:
        with ThreadPoolExecutor(max_workers=sc) as ex:
            for si, entry in ex.map(_one_basket, range(n_baskets)):
                res[si] = entry
    else:
        for si in range(n_baskets):
            si_, entry = _one_basket(si)
            res[si_] = entry
    per = [res[si] for si in range(n_baskets)]
    # Pooled (episode-level) test against the by-construction reference-policy attainment
    # ref = (100-XX)%, plus basket-level mean +/- SD.
    tot_succ = sum(p["successes"] for p in per)
    tot_n = sum(p["n"] for p in per)
    pooled = (tot_succ / tot_n) if tot_n else float("nan")
    plo, phi = _wilson_ci(tot_succ, tot_n)
    ref = 1.0 - percentile / 100.0
    bvals = [p["attainment"] for p in per if not np.isnan(p["attainment"])]
    result = {
        "percentile": percentile, "reference_attainment": ref,
        "pooled": {"successes": tot_succ, "n": tot_n, "attainment": pooled,
                   "wilson_ci95": [plo, phi],
                   "binom_p_vs_ref": _binom_p_two_sided(tot_succ, tot_n, ref),
                   "consistent_with_ref": bool(plo <= ref <= phi) if tot_n else None},
        "by_basket": {"mean": float(np.mean(bvals)) if bvals else float("nan"),
                      "sd": float(np.std(bvals, ddof=1)) if len(bvals) > 1 else 0.0,
                      "n_baskets": len(bvals)},
        # read by run_single_experiment.py / validate_target_calibration.py and vendster.analysis
        "mean_baseline_attainment": pooled,
        "recentre": _recenter_hint(pooled, percentile),
        "per_basket": per,
        "manifest": _manifest(model_name, {"kind": "validate_xx", "seed": seed,
                                           "n_baskets": n_baskets, "episodes": episodes,
                                           "ref_draws": ref_draws}),
    }
    (out / "validation.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _attainment(records: list) -> float:
    got = [r for r in records if r.get("target_met") is not None]
    return float(np.mean([1.0 if r["target_met"] else 0.0 for r in got])) if got else float("nan")


def _run_condition(cond, dataset, pool, target, llm_client, n, seed, out, cache, resume, dm,
                   concurrency: int = 1, scenario=None, vendor_llm_client=None,
                   episode_kwargs: dict | None = None):
    """Run episodes for one condition. With ``concurrency>1``, episodes run on a thread pool
    (each Episode is internally sequential; the LLM calls parallelize) — Bedrock has ample RPM/TPM
    headroom. A lock guards the shared forecaster cache and the file writes; the boto3 client is
    thread-local (see BedrockLLMClient). ``episode_kwargs`` are protocol settings passed to every
    Episode (``NegotiationParams.episode_kwargs()``; default = the reference protocol)."""
    path = out / "episodes" / f"{cond.id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resume is VALID-episode-aware: keep only successful records and re-run exactly the MISSING
    # episode indices. Counting raw lines would treat a throttled/errored episode ("status":"error")
    # as "done" and never backfill it — so a transient ThrottlingException would permanently shrink
    # the cell. Parse the index from episode_id ("{cond}_{i}") to know which indices are still needed.
    good: dict = {}                                       # episode index -> successful record
    if resume and path.exists():
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") != "error":
                try:
                    good[int(r["episode_id"].rsplit("_", 1)[1])] = r
                except (KeyError, ValueError, IndexError):
                    pass
    todo = [i for i in range(n) if i not in good]
    if not todo:                                          # complete — rewrite clean (drops any error lines) + return
        recs = [good[i] for i in sorted(good)]
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        return recs

    lock = _SETUP_LOCK                                    # module-level: safe across parallel baskets

    def _one(i):
        rng = np.random.default_rng(_episode_seed(seed, cond.id, i))
        stretch = sample_stretch(pool, rng)
        try:
            # Build the forecaster AND the Episode (which computes the forecast table via
            # forecaster.predict) under the lock: statsforecast/LightGBM fits and the shared
            # Chronos GPU pipeline are NOT thread-safe. This is fast + deterministic + pre-LLM;
            # the slow Bedrock calls live in .run(), which stays outside the lock (the parallelism
            # we actually want).
            with lock:
                fc = build_forecaster_for(cond, stretch, dataset, cache)
                ep = Episode(demand_model=dm, llm_client=llm_client, dataset=dataset, stretch=stretch,
                             condition_id=cond.id, forecaster=fc, target=target, rng=rng,
                             episode_id=f"{cond.id}_{i}", scenario=scenario,
                             vendor_llm_client=vendor_llm_client, **(episode_kwargs or {}))
                ep._setup()                              # forecaster inference under the lock
            r = ep.run()                                 # slow Bedrock calls run in parallel
            rec = _episode_record(r)
            with lock:
                _write_transcript(out, cond.id, r)
        except Exception as e:                           # failure isolation: log + continue
            print(f"[episode-error {cond.id}_{i}] {type(e).__name__}: {str(e)[:160]}", flush=True)
            rec = {"episode_id": f"{cond.id}_{i}", "condition_id": cond.id,
                   "status": "error", "error": str(e)[:300], "target_met": None}
        return i, rec

    idx = todo
    new = {}
    if concurrency > 1 and len(idx) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for i, rec in ex.map(_one, idx):
                new[i] = rec
    else:
        for i in idx:
            _, rec = _one(i); new[i] = rec

    for i, rec in new.items():                           # merge freshly-run SUCCESSES; errors stay
        if rec.get("status") != "error":                 # absent so a later resume retries them
            good[i] = rec
    recs = [good[i] for i in sorted(good)]               # rewrite the whole file, clean + index-ordered
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return recs


def run_single_experiment(dataset: Dataset, llm_client, target: float, *, episodes_per_condition: int = 200,
                   seed: int = 0, out_dir: str = "experiments/run", model_name: str = "mock",
                   min_history: int = 60, resume: bool = True,
                   band=(0.15, 0.70), min_spread: float = 0.15, checkpoint_enabled: bool = True,
                   demand_model=None, concurrency: int = 1, conditions=None, scenario=None,
                   vendor_llm_client=None, episode_kwargs: dict | None = None) -> dict:
    """Bracketing-first experiment with a collapse checkpoint, logging, and resume.

    ``conditions``: the ConditionSpec list to run (default ``reference_conditions()``). Passing a
    custom list is how a user runs their own roster without editing core (see BenchmarkConfig).
    ``scenario``: the negotiation Scenario (default VendsterScenario via Episode)."""
    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    dm = demand_model or ElasticityDemandModel()
    pool = build_stretch_pool(dataset, min_history=min_history)
    conds = conditions if conditions is not None else reference_conditions()
    bracket = [c for c in conds if c.is_bracketing]
    rest = [c for c in conds if not c.is_bracketing]
    cache: dict = {}
    summary = {"target": target, "conditions": {}, "checkpoint": None,
               "manifest": _manifest(model_name, {"kind": "experiment", "seed": seed,
                                                   "episodes_per_condition": episodes_per_condition})}

    # WAVE 1: bracketing conditions
    for cond in bracket:
        recs = _run_condition(cond, dataset, pool, target, llm_client, episodes_per_condition,
                              seed, out, cache, resume, dm, concurrency, scenario, vendor_llm_client,
                              episode_kwargs)
        summary["conditions"][cond.id] = {"attainment": _attainment(recs), "n": len(recs)}

    # CHECKPOINT (collapse stop-rule)
    attain = {c.id: summary["conditions"][c.id]["attainment"] for c in bracket}
    baseline = attain.get("no_forecast", float("nan"))
    vals = [v for v in attain.values() if not np.isnan(v)]
    spread = (max(vals) - min(vals)) if vals else 0.0
    pinned = any(v in (0.0, 1.0) for v in vals)
    passed = (band[0] <= baseline <= band[1]) and spread >= min_spread and not pinned
    if not checkpoint_enabled:
        passed = True                        # bypass (pilot/verification only)
    summary["checkpoint"] = {"baseline_attainment": baseline, "bracket_spread": spread,
                             "pinned": pinned, "passed": bool(passed), "band": band,
                             "min_spread": min_spread, "checkpoint_enabled": checkpoint_enabled}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not passed:
        summary["halted"] = ("COLLAPSE: baseline/spread out of viable range — recalibrate "
                             "(adjust target_percentile) and re-run bracketing.")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    # WAVE 2: remaining conditions
    for cond in rest:
        recs = _run_condition(cond, dataset, pool, target, llm_client, episodes_per_condition,
                              seed, out, cache, resume, dm, concurrency, scenario, vendor_llm_client,
                              episode_kwargs)
        summary["conditions"][cond.id] = {"attainment": _attainment(recs), "n": len(recs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_paired_experiment(source, llm_client, *, n_sets: int = 3,
                          episodes_per_condition: int = 20, seed: int = 0,
                          out_dir: str = "experiments/paired", model_name: str = "mock",
                          demand_model=None, min_history: int = 60, resume: bool = True,
                          band=(0.15, 0.70), min_spread: float = 0.15,
                          target_percentile: float = 80.0,
                          ref_draws: int = 2000, concurrency: int = 1,
                          set_concurrency: int = 1, cell_concurrency: int = 0,
                          conditions=None, scenario=None, vendor_llm_client=None,
                          episode_kwargs: dict | None = None) -> dict:
    """Paired product-set design (PRIMARY): sample ``n_sets`` 5-baskets from the pool;
    per set, set its own target then run ALL conditions on that basket (paired — same
    products across conditions, controlling product variation); aggregate per condition
    across sets (mean of within-set attainment). ``source`` is an M5DataSource-like with
    ``sample_basket`` + ``build_dataset``. Each basket's target is the analytic reference-policy
    P{target_percentile} (``analytic_target``; no LLM, so large K is cheap).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dm = demand_model or ElasticityDemandModel()
    conds = conditions if conditions is not None else reference_conditions()

    # Per-set context (dataset / stretch pool / target) built once. Targets are LLM-free (analytic)
    # so this is cheap and sequential; the LLM cost is all in the episode cells below.
    ctx = {}
    for si in range(n_sets):
        basket = source.sample_basket(np.random.default_rng(seed * 7919 + si))
        dataset = source.build_dataset(basket, cost_seed=si)
        set_dir = out / f"set_{si}"
        calib = analytic_target(dataset, n=ref_draws, seed=seed * 100 + si,
                                percentile=target_percentile, demand_model=dm,
                                min_history=min_history,
                                discount_levels=(episode_kwargs or {}).get("discount_levels",
                                                                           DISCOUNT_LEVELS))
        (set_dir / "calibration").mkdir(parents=True, exist_ok=True)
        (set_dir / "calibration" / "analytic_target.json").write_text(
            json.dumps(calib, indent=2, default=str))
        ctx[si] = {"dataset": dataset, "pool": build_stretch_pool(dataset, min_history=min_history),
                   "target": calib["target"], "set_dir": set_dir,
                   "basket": dataset.meta.get("basket")}

    # Flatten to (set, condition) cells and run the WHOLE grid concurrently (not just sets) so the
    # full K x conditions grid parallelizes. Total in-flight Bedrock requests ~= cell_concurrency *
    # concurrency; the global _SETUP_LOCK serializes forecaster inference (shared Chronos GPU).
    cc = max(1, cell_concurrency or set_concurrency)
    cells = [(si, cond) for si in range(n_sets) for cond in conds]

    def _one_cell(job):
        si, cond = job
        c = ctx[si]
        recs = _run_condition(cond, c["dataset"], c["pool"], c["target"], llm_client,
                              episodes_per_condition, seed * 100 + si, c["set_dir"], {}, resume,
                              dm, concurrency, scenario, vendor_llm_client, episode_kwargs)
        return si, cond.id, {"attainment": _attainment(recs), "n": len(recs)}

    cond_res = {si: {} for si in range(n_sets)}
    if cc > 1 and len(cells) > 1:
        with ThreadPoolExecutor(max_workers=cc) as ex:
            for si, cid, entry in ex.map(_one_cell, cells):
                cond_res[si][cid] = entry
    else:
        for job in cells:
            si, cid, entry = _one_cell(job)
            cond_res[si][cid] = entry
    for si in range(n_sets):
        (ctx[si]["set_dir"] / "summary.json").write_text(json.dumps(
            {"target": ctx[si]["target"], "conditions": cond_res[si]}, indent=2, default=str))
    per_set = [{"basket": ctx[si]["basket"], "target": ctx[si]["target"],
                "conditions": cond_res[si]} for si in range(n_sets)]

    # Aggregate across sets: per-condition mean of within-set attainment (paired).
    agg = {}
    for cond in conds:
        vals = [s["conditions"][cond.id]["attainment"] for s in per_set
                if not np.isnan(s["conditions"][cond.id]["attainment"])]
        agg[cond.id] = {"mean_attainment": float(np.mean(vals)) if vals else float("nan"),
                        "sd_across_sets": float(np.std(vals)) if len(vals) > 1 else 0.0,
                        "n_sets": len(vals)}
    baseline = agg.get("no_forecast", {}).get("mean_attainment", float("nan"))
    bvals = [agg[c.id]["mean_attainment"] for c in conds
             if c.is_bracketing and not np.isnan(agg[c.id]["mean_attainment"])]
    spread = (max(bvals) - min(bvals)) if bvals else 0.0
    passed = (band[0] <= baseline <= band[1]) and spread >= min_spread \
        and not any(v in (0.0, 1.0) for v in bvals)
    summary = {"n_sets": n_sets, "aggregate": agg, "per_set": per_set,
               "checkpoint": {"baseline_attainment": baseline, "bracket_spread": spread,
                              "passed": bool(passed), "band": band, "min_spread": min_spread},
               "manifest": _manifest(model_name, {"kind": "paired_experiment", "seed": seed,
                                                  "n_sets": n_sets,
                                                  "episodes_per_condition": episodes_per_condition,
                                                  "target_percentile": target_percentile,
                                                  "ref_draws": ref_draws})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
