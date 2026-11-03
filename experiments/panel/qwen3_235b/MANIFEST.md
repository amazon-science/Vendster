# Vendster — evidence bundle (qwen.qwen3-235b-a22b-2507-v1:0)

- role: evaluated panel member; self-play (Buyer = Vendor = this model)
- git_sha: `dd84010f98` — a commit in the pre-release development history; the public repository was published fresh, so this SHA does not resolve in it (see README)
- run_utc: 2026-09-01T23:31:37Z  (python 3.12.13)
- model: `qwen.qwen3-235b-a22b-2507-v1:0` (Amazon Bedrock Converse API, temperature 0.0 where the model accepts it)
- frozen params: seed=1  sets(K)=16  episodes/condition(m)=25  target=analytic P80  panel=1500  min weekly units>=5
- conditions: 26 = 10 real forecasters + no_forecast + 15-cell perturbation grid (8-level bias ±{.05,.10,.15,.20} + unbiased + 6-level width ladder)
- integrity: 10,400 episodes; status = 8,767 agreement + 1,633 max_turns
- transcripts: turn-by-turn transcripts are not released; per-episode outcome records in `experiment/set_*/episodes/` are complete, so every table recomputes offline. Rerun the bundle with `scripts/build_model_bundle.sh` to generate transcripts.
- shared inputs: pool-level and per-basket forecast quality from `experiments/shared/` (model-independent)
- reproducibility: pool census, per-basket quality, and analytic targets are deterministic from (code, seed); LLM episodes are not bit-reproducible (Bedrock is not seedable), so the archived per-episode outcome records are the record. Rerun `scripts/build_model_bundle.sh` to resume or reproduce.
