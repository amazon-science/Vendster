"""
Compare two Vendster experiment bundles condition-by-condition to CONFIRM a re-run is a
uniform-provenance replication (equivalent within sampling noise), not a science-impacting change.

For each condition: pooled attainment in bundle A vs B, the difference, and whether it is within the
expected Monte-Carlo band (Bedrock is non-deterministic + finite m, so exact equality is NOT expected;
we test |Δ| against ~2*SE of the difference of two proportions). Flags any condition whose shift
exceeds the noise band — those would warrant investigation.

    uv run python scripts/compare_bundles.py \
        --a experiments/sonnet_holdout/experiment --b experiments/panel/opus_4_6/experiment \
        --out /tmp/provenance_check.md
"""
import argparse
import glob
import json
import math
from pathlib import Path


def attain(run_dir):
    out = {}
    for cond_dir in sorted({f.split("/")[-1].replace(".jsonl", "")
                            for f in glob.glob(f"{run_dir}/set_*/episodes/*.jsonl")}):
        k = n = 0
        for f in glob.glob(f"{run_dir}/set_*/episodes/{cond_dir}.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("target_met") is not None:
                    n += 1
                    k += 1 if r["target_met"] else 0
        out[cond_dir] = (k, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="bundle A experiment dir (e.g. incremental)")
    ap.add_argument("--b", required=True, help="bundle B experiment dir (e.g. fresh)")
    ap.add_argument("--out", default="provenance_check.md")
    args = ap.parse_args()

    A, B = attain(args.a), attain(args.b)
    conds = sorted(set(A) | set(B))
    rows, flags = [], 0
    for c in conds:
        ka, na = A.get(c, (0, 0)); kb, nb = B.get(c, (0, 0))
        pa = ka / na if na else float("nan")
        pb = kb / nb if nb else float("nan")
        d = pb - pa
        # SE of the difference of two independent proportions
        se = math.sqrt((pa * (1 - pa) / na if na else 0) + (pb * (1 - pb) / nb if nb else 0)) if na and nb else float("nan")
        within = (abs(d) <= 2 * se) if se == se and se > 0 else None
        if within is False:
            flags += 1
        rows.append((c, pa, na, pb, nb, d, se, within))

    lines = ["# Provenance check — incremental vs fresh re-run\n",
             f"A = `{args.a}`  ·  B = `{args.b}`\n",
             "Bedrock is non-deterministic + finite m, so exact match is NOT expected; each condition "
             "should agree within ~2*SE of the proportion difference. A flag means B shifted beyond the "
             "Monte-Carlo band vs A (investigate).\n",
             "| condition | A attain (n) | B attain (n) | Δ (B−A) | 2·SE | within noise? |",
             "|---|---|---|---|---|---|"]
    for c, pa, na, pb, nb, d, se, within in rows:
        mark = "✓" if within else ("**FLAG**" if within is False else "—")
        lines.append(f"| {c} | {pa:.3f} ({na}) | {pb:.3f} ({nb}) | {d:+.3f} | {2*se:.3f} | {mark} |"
                     if se == se else f"| {c} | {pa:.3f} ({na}) | {pb:.3f} ({nb}) | {d:+.3f} | — | — |")
    mean_abs = sum(abs(r[5]) for r in rows if r[5] == r[5]) / max(1, sum(1 for r in rows if r[5] == r[5]))
    verdict = ("REPLICATION CONFIRMED — all conditions within the Monte-Carlo band; the incremental "
               "bundle shows no science-impacting artifact." if flags == 0 else
               f"{flags} condition(s) exceeded the ~2*SE band — investigate before trusting equivalence.")
    lines += [f"\n**mean |Δ| across conditions = {mean_abs:.3f}**  ·  flags = {flags}\n", f"**Verdict:** {verdict}"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
