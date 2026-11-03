"""
LLMClient implementations for Vendster.

- ``BedrockLLMClient``: any Bedrock model (Claude, OpenAI GPT-5.6, DeepSeek-V3.2, Qwen3) via the
  vendor-neutral Converse API (SigV4); strips reasoning blocks, keeps the answer.
- ``MockLLMClient``: deterministic, zero-cost negotiator for tests / pipeline
  verification. It reads a machine-readable ``[STATE] ...`` line embedded in the
  turn prompt (built by ``negotiation.episode._state_line``) so it never needs to parse free text.

Both implement the ``LLMClient`` protocol: ``generate(system_prompt, messages) -> str``.
"""

from __future__ import annotations

import json
import re
import threading

from ..core.registry import LLM_CLIENTS


def _parse_state(messages: list[dict]) -> dict:
    """Extract the ``[STATE] k=v ...`` line from the latest user message."""
    text = messages[-1]["content"] if messages else ""
    m = re.search(r"\[STATE\]\s*(.*)", text)
    state: dict = {}
    if not m:
        return state
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            state[k] = v
    return state


@LLM_CLIENTS.register("mock")
class MockLLMClient:
    """Deterministic negotiator: Buyer opens high, Vendor counters low, Buyer
    accepts — converges in ~3 turns so episode plumbing can be exercised offline.
    """

    name = "mock"

    def __init__(self, buyer_open: float = 0.20, vendor_counter: float = 0.10):
        self.buyer_open = buyer_open
        self.vendor_counter = vendor_counter

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        st = _parse_state(messages)
        role = st.get("role", "Buyer")
        products = [p for p in st.get("products", "").split(",") if p]
        has_active = st.get("active_proposal", "none") != "none"

        if role == "Vendor" and has_active:
            # Vendor counters low once, then accepts to terminate.
            turn = int(st.get("turn", "2"))
            if turn <= 2:
                proposal = {p: self.vendor_counter for p in products}
                return _json_action({"type": "proposal", "content": "Counter: protecting margin.",
                                     "proposal": proposal})
            return _json_action({"type": "decision", "content": "Acceptable.", "decision": "accept"})

        if role == "Buyer" and has_active:
            return _json_action({"type": "decision", "content": "Deal.", "decision": "accept"})

        # opening (or no active proposal): propose
        disc = self.buyer_open if role == "Buyer" else self.vendor_counter
        proposal = {p: disc for p in products}
        return _json_action({"type": "proposal", "content": f"{role} proposal.", "proposal": proposal})


def _json_action(obj: dict) -> str:
    return "```json\n" + json.dumps(obj) + "\n```"


@LLM_CLIENTS.register("bedrock")
class BedrockLLMClient:
    """Any Bedrock model (Claude, OpenAI GPT-5.6, DeepSeek-V3.2, Qwen3, ...) via the vendor-neutral
    **Converse** API, so every panel model goes through one code path and the numbers are
    cross-model comparable. Auth is a SigV4 profile (``AWS_PROFILE``) resolved by boto3; the only
    requirement is a principal granted ``bedrock:InvokeModel`` on the target model IDs.

    Reasoning models (DeepSeek-V3.2, GPT-5.6) return their chain-of-thought in a separate ``reasoningContent``
    block; the *answer* is the ``text`` block(s), which is all we join. ``max_tokens`` is generous so
    reasoning tokens don't crowd out the final action JSON. ``temperature`` is dropped (once, cached)
    if a model rejects it — some reasoning models forbid a custom temperature.
    """

    name = "bedrock"

    def __init__(self, model_id: str = "us.anthropic.claude-sonnet-4-6",
                 region_name: str = "us-west-2", max_tokens: int = 8000,
                 temperature: float = 0.0, connect_timeout: int = 20,
                 read_timeout: int = 300, max_attempts: int = 4):
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._use_temp = True                     # flipped off if the model rejects temperature
        self._cfg = dict(connect_timeout=connect_timeout, read_timeout=read_timeout,
                         max_attempts=max_attempts)
        self._local = threading.local()

    def _client_lazy(self):
        c = getattr(self._local, "client", None)
        if c is None:
            import boto3
            from botocore.config import Config
            c = boto3.client(
                "bedrock-runtime", region_name=self.region_name,
                config=Config(connect_timeout=self._cfg["connect_timeout"],
                              read_timeout=self._cfg["read_timeout"],
                              retries={"max_attempts": self._cfg["max_attempts"], "mode": "adaptive"}),
            )
            self._local.client = c
        return c

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        conv = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
        infcfg = {"maxTokens": self.max_tokens}
        if self._use_temp:
            infcfg["temperature"] = self.temperature
        kwargs = {"modelId": self.model_id, "messages": conv, "inferenceConfig": infcfg}
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
        from botocore.exceptions import ClientError
        try:
            resp = self._client_lazy().converse(**kwargs)
        except ClientError as e:
            # Some models (e.g. GPT-5.6) reject a custom temperature. Retry WITHOUT it whenever the
            # error says so — NOT gated on self._use_temp, which is racy: under concurrency another
            # thread may flip the flag between our kwargs-build and this except, which would otherwise
            # make us re-raise instead of retry (dropping the episode). Retry this call directly.
            if "temperature" in str(e).lower():
                self._use_temp = False            # cache for subsequent calls
                infcfg.pop("temperature", None)
                resp = self._client_lazy().converse(**kwargs)
            else:
                raise
        blocks = resp["output"]["message"]["content"]
        return "".join(b["text"] for b in blocks if "text" in b)     # answer only; skip reasoningContent


def add_llm_args(ap, default: str = "mock") -> None:
    """The ``--model`` / ``--model-id`` pair every negotiation script takes."""
    ap.add_argument("--model", default=default, choices=LLM_CLIENTS.names(),
                    help="registered LLM client: `mock` (deterministic, zero AWS spend), `bedrock` "
                         "(any Bedrock model via the Converse API; auth = AWS_PROFILE), or a client "
                         "you register under LLM_CLIENTS (see EXTENDING.md)")
    ap.add_argument("--model-id", default=None,
                    help="model id passed to the client; required with --model bedrock (any Bedrock "
                         "model or inference-profile id, e.g. us.anthropic.claude-opus-4-6-v1). "
                         "Ignored for --model mock.")


def make_llm(model: str, model_id: str | None):
    """Build a registered client; ``bedrock`` must be told which model to call."""
    if model == "bedrock" and not model_id:
        raise SystemExit("--model-id is required with --model bedrock "
                         "(e.g. us.anthropic.claude-opus-4-6-v1)")
    return LLM_CLIENTS.create(model, **({"model_id": model_id} if model_id and model != "mock" else {}))
