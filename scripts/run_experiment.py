"""
Run Vendster experiments (paired product-set design).

Sample K 5-baskets from the pool; per set, define its own target (P80 of reference-policy sales by
default), then run all reference conditions on that basket; aggregate per condition across sets.

    # verify (mock, zero Bedrock, tiny):
    uv run python scripts/run_experiment.py --model mock --sets 2 \
        --episodes 2 --panel-size 200 --out experiments/pilot_paired

    # the locked run for one panel model (Bedrock; P80 targets, K=16, m=25, held-out seed):
    uv run python scripts/run_experiment.py --model bedrock --model-id us.anthropic.claude-opus-4-6-v1 \
        --sets 16 --episodes 25 --target-percentile 80 --seed 1 \
        --concurrency 15 --cell-concurrency 16 --out experiments/panel/opus_4_6/experiment
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vendster.forecasters  # noqa: F401  registers forecasters
from vendster.llm import add_llm_args, make_llm  # importing registers the clients
from vendster.data.m5 import M5DataSource
from vendster.experiment import run_paired_experiment, validate_xx


def main():
    ap = argparse.ArgumentParser()
    add_llm_args(ap)
    ap.add_argument("--conditions", default="all", choices=["all", "bias", "grid"],
                    help="condition subset: all=26; bias=no_forecast+8 bias cells+unbiased "
                         "grid=no_forecast+full 15-cell perturbation grid (bias and interval width)")
    ap.add_argument("--vendor-model-id", default=None,
                    help="if set, the VENDOR role uses this model id (same client as --model) while the "
                         "Buyer uses --model-id — a Buyer×Vendor role-cross. Default: self-play (both roles "
                         "= --model-id).")
    ap.add_argument("--sets", type=int, default=20, help="number of sampled product-sets (K)")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per condition per set")
    ap.add_argument("--target-percentile", type=float, default=80.0,
                    help="each basket's target = P{XX} of its total sales under a uniform-random "
                         "discount reference policy (analytic, no LLM); 80 = the locked challenging "
                         "operating point (see the paper's Environment Setup appendix)")
    ap.add_argument("--ref-draws", type=int, default=2000,
                    help="seeded reference-policy DGP draws per basket for the target")
    ap.add_argument("--validate-xx", type=int, default=0, metavar="N",
                    help="if >0: run the XX-centering check on N baskets (LLM no-forecast) and exit")
    ap.add_argument("--min-weekly", type=int, default=5, help="minimum units per week sales to include a product")
    ap.add_argument("--panel-size", type=int, default=1500, help="number of products to sample products of interest from")
    ap.add_argument("--concurrency", type=int, default=15,
                    help="parallel episodes per (set,condition) cell")
    ap.add_argument("--cell-concurrency", type=int, default=20,
                    help="parallel (set,condition) cells for the paired grid; total in-flight "
                         "~= cell-concurrency * concurrency (measured safe up to ~384)")
    ap.add_argument("--set-concurrency", type=int, default=16,
                    help="parallel baskets for --validate-xx (no_forecast only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/paired", help="output directory")
    ap.add_argument("--band-lo", type=float, default=0.30)
    ap.add_argument("--band-hi", type=float, default=0.70)
    ap.add_argument("--min-spread", type=float, default=0.25)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] M5 pool (min-week>={args.min_weekly})...", flush=True)
    source = M5DataSource(min_weekly=args.min_weekly, panel_size=args.panel_size)
    print(f"  pool size: {len(source.pool())} products", flush=True)
    llm = make_llm(args.model, args.model_id)                                   # Buyer (sees the forecast)
    vendor_llm = make_llm(args.model, args.vendor_model_id) if args.vendor_model_id else None
    model_name = (f"buyer={args.model_id or args.model};vendor={args.vendor_model_id}"
                  if vendor_llm else (args.model_id or args.model))
    print(f"  LLM client: {args.model}"
          + (f" buyer={args.model_id}" if args.model_id else "")
          + (f" vendor={args.vendor_model_id}" if vendor_llm else " (self-play)"), flush=True)

    if args.validate_xx > 0:
        print(f"[validate-xx] P{args.target_percentile:.0f} targets on {args.validate_xx} baskets, "
              f"{args.episodes} no-forecast episodes each ...", flush=True)
        v = validate_xx(source, llm, n_baskets=args.validate_xx, episodes=args.episodes,
                        percentile=args.target_percentile, seed=args.seed, ref_draws=args.ref_draws,
                        out_dir=args.out, model_name=model_name, concurrency=args.concurrency,
                        set_concurrency=args.set_concurrency)
        print(f"\n  LLM no-forecast baseline = {v['mean_baseline_attainment']:.2%} "
              f"(reference-policy attainment {v['reference_attainment']:.0%})", flush=True)
        print(f"  {v['recentre']['note']}", flush=True)
        for i, p in enumerate(v["per_basket"]):
            print(f"    set {i}: target=${p['target']:,.0f}  attainment={p['attainment']:.2f} "
                  f"(n={p['n']})", flush=True)
        print(f"\nDONE in {time.time()-t0:.0f}s -> {args.out}/validation.json", flush=True)
        return

    print(f"  sampling {args.sets} baskets of 5; target=analytic P{args.target_percentile:.0f} "
          f"({args.ref_draws} reference-policy draws/basket)", flush=True)
    from vendster.core.config import reference_conditions
    conds = reference_conditions()
    if args.conditions == "bias":
        conds = [c for c in conds if c.id in ("no_forecast", "unbiased") or c.id.startswith("bias_")]
    elif args.conditions == "grid":
        conds = [c for c in conds if c.forecaster is None or c.forecaster.name == "perturbed"]
    if args.conditions != "all":
        print(f"  conditions subset '{args.conditions}': {len(conds)} -> {[c.id for c in conds]}", flush=True)

    summary = run_paired_experiment(
        source, llm, n_sets=args.sets, conditions=conds,
        episodes_per_condition=args.episodes, seed=args.seed, out_dir=args.out,
        model_name=model_name, band=(args.band_lo, args.band_hi), min_spread=args.min_spread,
        target_percentile=args.target_percentile,
        ref_draws=args.ref_draws, concurrency=args.concurrency,
        set_concurrency=args.set_concurrency, cell_concurrency=args.cell_concurrency,
        vendor_llm_client=vendor_llm)

    ck = summary["checkpoint"]
    print(f"\n[checkpoint] baseline={ck['baseline_attainment']:.2f} "
          f"spread={ck['bracket_spread']:.2f} passed={ck['passed']}", flush=True)
    print(f"[aggregate attainment across {summary['n_sets']} sets]", flush=True)
    for cid, a in summary["aggregate"].items():
        print(f"   {cid:20} mean={a['mean_attainment']:.3f}  sd={a['sd_across_sets']:.3f}  "
              f"sets={a['n_sets']}", flush=True)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
