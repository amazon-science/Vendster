"""Exit 0 iff every (condition, basket) cell in an experiment dir has >= m VALID episode records
(status != "error"). Used by the retry loop to decide whether a resumable Bedrock run is actually
finished — counting raw lines would let throttled/errored episodes masquerade as complete.

    uv run python scripts/check_complete.py experiments/panel/opus_4_6/experiment 16 25
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

run_dir, n_sets, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
subset = sys.argv[4] if len(sys.argv) > 4 else "all"     # all | bias | grid (match run_experiment)
import vendster.forecasters  # noqa: F401,E402
from vendster.core.config import reference_conditions  # noqa: E402

conds = reference_conditions()
if subset == "bias":
    conds = [c for c in conds if c.id in ("no_forecast", "unbiased") or c.id.startswith("bias_")]
elif subset == "grid":
    conds = [c for c in conds if c.forecaster is None or c.forecaster.name == "perturbed"]
want = {c.id for c in conds}
counts = defaultdict(dict)
for f in glob.glob(f"{run_dir}/set_*/episodes/*.jsonl"):
    si = int(f.split("set_")[1].split("/")[0])
    cid = f.split("/")[-1].replace(".jsonl", "")
    counts[cid][si] = sum(1 for l in open(f) if l.strip() and json.loads(l).get("status") != "error")

missing = []
for cid in want:
    for si in range(n_sets):
        if counts.get(cid, {}).get(si, 0) < m:
            missing.append(f"{cid}/set_{si}={counts.get(cid, {}).get(si, 0)}")
if missing:
    print(f"INCOMPLETE: {len(missing)} cells short (e.g. {missing[:5]})")
    sys.exit(1)
print(f"COMPLETE: {len(want)} conditions x {n_sets} sets x >= {m} eps")
sys.exit(0)
