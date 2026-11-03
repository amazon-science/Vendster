"""
Measure the Bedrock throughput ceiling of any Bedrock model for Vendster episodes, to size
--concurrency from first principles (NOT a magic number).

Builds a REALISTIC Buyer-turn prompt from an actual Episode (products + forecast + history),
fires it at increasing concurrency, and captures per-request latency + token usage + throttle
rate. Then computes the sustainable requests/sec under the account quota (TPM/RPM) and the
concurrency that reaches a target utilisation.

    AWS_PROFILE=<your-bedrock-profile> uv run python scripts/probe_throughput.py \
        --model-id us.anthropic.claude-opus-4-6-v1 \
        --levels 32 64 128 192 --per-level 64 --tpm 6000000 --rpm 10000
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import vendster.forecasters  # noqa: F401
import vendster.llm  # noqa: F401
from vendster.core.config import reference_conditions
from vendster.core.registry import LLM_CLIENTS
from vendster.data.m5 import M5DataSource
from vendster.dgp.demand import ElasticityDemandModel
from vendster.dgp.stretch_pool import build_stretch_pool, sample_stretch
from vendster.experiment.runner import analytic_target, build_forecaster_for
from vendster.negotiation.episode import Episode


def build_realistic_prompt():
    """A true Buyer turn-1 prompt+system from a real episode (unbiased forecast condition)."""
    src = M5DataSource(min_weekly=5, panel_size=200)
    basket = src.sample_basket(np.random.default_rng(0))
    ds = src.build_dataset(basket, cost_seed=0)
    target = analytic_target(ds, seed=0)["target"]
    pool = build_stretch_pool(ds, min_history=60)
    cond = next(c for c in reference_conditions() if c.id == "unbiased")
    st = sample_stretch(pool, np.random.default_rng(1))
    fc = build_forecaster_for(cond, st, ds, {})
    ep = Episode(demand_model=ElasticityDemandModel(), llm_client=LLM_CLIENTS.create("mock"),
                 dataset=ds, stretch=st, condition_id="unbiased", forecaster=fc, target=target,
                 rng=np.random.default_rng(2), episode_id="probe")
    ep._setup()
    system = ep._system("Buyer")
    messages = ep._messages("Buyer", 1, None, [], "")
    return system, messages


def one_call(client, system, messages):
    """One Converse call through the client's own boto3 session; returns latency + token usage."""
    conv = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
    infcfg = {"maxTokens": client.max_tokens}
    if client._use_temp:
        infcfg["temperature"] = client.temperature
    kwargs = {"modelId": client.model_id, "messages": conv,
              "system": [{"text": system}], "inferenceConfig": infcfg}
    t0 = time.time()
    try:
        u = client._client_lazy().converse(**kwargs).get("usage", {})
        return {"ok": True, "lat": time.time() - t0,
                "in": u.get("inputTokens", 0), "out": u.get("outputTokens", 0)}
    except Exception as e:
        if "temperature" in infcfg and "temperature" in str(e).lower():
            client._use_temp = False             # same fallback as BedrockLLMClient.generate
            return one_call(client, system, messages)
        name = type(e).__name__
        throttled = "Throttl" in name or "TooManyRequests" in str(e) or "ThrottlingException" in str(e)
        return {"ok": False, "lat": time.time() - t0, "throttled": throttled, "err": name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True,
                    help="Bedrock model or inference-profile id to probe, e.g. us.anthropic.claude-opus-4-6-v1")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128, 192])
    ap.add_argument("--per-level", type=int, default=64, help="requests fired per concurrency level")
    ap.add_argument("--tpm", type=float, default=6_000_000)
    ap.add_argument("--rpm", type=float, default=10_000)
    ap.add_argument("--target-util", type=float, default=0.70,
                    help="fraction of the binding quota to size concurrency for")
    ap.add_argument("--reqs-per-episode", type=float, default=5.0)
    args = ap.parse_args()

    # Disable client-side retries for the probe so throttles are VISIBLE (not hidden as latency).
    client = LLM_CLIENTS.create("bedrock", model_id=args.model_id, region_name=args.region)
    client._cfg["max_attempts"] = 1
    system, messages = build_realistic_prompt()
    print(f"realistic prompt built; system={len(system)} chars, "
          f"user={len(messages[-1]['content'])} chars\n", flush=True)

    rows = []
    for c in args.levels:
        n = max(args.per_level, c)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(lambda _: one_call(client, system, messages), range(n)))
        wall = time.time() - t0
        ok = [r for r in res if r["ok"]]
        thr = sum(1 for r in res if not r["ok"] and r.get("throttled"))
        errs = sum(1 for r in res if not r["ok"] and not r.get("throttled"))
        lats = sorted(r["lat"] for r in ok)
        p50 = lats[len(lats) // 2] if lats else float("nan")
        p95 = lats[int(len(lats) * 0.95)] if lats else float("nan")
        in_tok = np.mean([r["in"] for r in ok]) if ok else 0
        out_tok = np.mean([r["out"] for r in ok]) if ok else 0
        rps = len(ok) / wall if wall else 0
        tps = (in_tok + out_tok) * rps
        rows.append({"conc": c, "n": n, "ok": len(ok), "throttled": thr, "errors": errs,
                     "wall": wall, "rps": rps, "tok_per_req": in_tok + out_tok, "tps": tps,
                     "p50": p50, "p95": p95, "in": in_tok, "out": out_tok})
        print(f"conc={c:>4}  ok={len(ok):>3}/{n}  throttled={thr:>3}  err={errs:>2}  "
              f"rps={rps:6.1f}  tok/req={in_tok+out_tok:6.0f} (in {in_tok:.0f}/out {out_tok:.0f})  "
              f"tok/s={tps:8.0f}  lat p50={p50:4.1f}s p95={p95:4.1f}s", flush=True)

    # Size concurrency from the binding quota.
    good = [r for r in rows if r["throttled"] == 0 and r["ok"] > 0]
    ref = max(good, key=lambda r: r["conc"]) if good else max(rows, key=lambda r: r["ok"])
    tok_per_req = ref["tok_per_req"] or 3000
    lat = ref["p50"] if ref["p50"] == ref["p50"] else 5.0
    tpm_rps = args.tpm / 60.0 / tok_per_req          # req/s ceiling from tokens
    rpm_rps = args.rpm / 60.0                          # req/s ceiling from request count
    ceil_rps = min(tpm_rps, rpm_rps)
    bind = "TPM" if tpm_rps < rpm_rps else "RPM"
    target_rps = args.target_util * ceil_rps
    # concurrency (threads) to sustain target_rps at observed latency: little's law N = rps * lat
    conc = int(round(target_rps * lat))
    print("\n==== CEILING (from measured tokens + latency) ====", flush=True)
    print(f"  tok/req≈{tok_per_req:.0f}  latency p50≈{lat:.1f}s", flush=True)
    print(f"  TPM ceiling: {tpm_rps:6.1f} req/s   RPM ceiling: {rpm_rps:6.1f} req/s  -> binds on {bind}",
          flush=True)
    print(f"  sustainable ceiling ≈ {ceil_rps:.1f} req/s  (~{ceil_rps*60:.0f} req/min, "
          f"~{ceil_rps/args.reqs_per_episode:.1f} episodes/s)", flush=True)
    print(f"  target {args.target_util:.0%} util -> {target_rps:.1f} req/s", flush=True)
    print(f"  ==> recommended --concurrency ≈ {conc}  (Little's law: {target_rps:.1f} rps x {lat:.1f}s)",
          flush=True)
    print("      (boto3 adaptive retry absorbs bursts above this)", flush=True)


if __name__ == "__main__":
    main()
