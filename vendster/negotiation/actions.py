"""Parse and validate agent JSON actions (proposal / decision)."""

from __future__ import annotations

import json
from collections.abc import Sequence


def extract_json(raw: str) -> str:
    """Pull the JSON object out of a raw LLM response (```json block or {"type"...})."""
    if "```json" in raw:
        start = raw.find("```json") + 7
        end = raw.find("```", start)
        return raw[start:end if end != -1 else len(raw)].strip()
    if '{"type"' in raw:
        start = raw.find('{"type"')
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return raw[start:i + 1]
        return raw[start:]
    return raw.strip()


class InvalidActionError(ValueError):
    """Raised when an agent action fails validation (triggers bounded retry)."""


def validate_action(raw: str, product_ids: Sequence[str], discount_levels: Sequence[float]) -> dict:
    """Validate/normalize an action. Returns a dict with type and payload, or raises."""
    try:
        action = json.loads(extract_json(raw))
    except json.JSONDecodeError as e:
        raise InvalidActionError(f"not valid JSON: {e}") from e
    if not isinstance(action, dict) or "type" not in action:
        raise InvalidActionError("missing 'type'")
    if action["type"] not in ("proposal", "decision"):
        raise InvalidActionError(f"invalid type {action['type']!r}")

    if action["type"] == "proposal":
        prop = action.get("proposal")
        if not isinstance(prop, dict):
            raise InvalidActionError("proposal must be an object")
        norm = {}
        allowed = set(round(float(d), 4) for d in discount_levels)
        for pid in product_ids:
            if str(pid) not in {str(k) for k in prop}:
                raise InvalidActionError(f"proposal missing product {pid}")
        for k, v in prop.items():
            try:
                d = round(float(v), 4)
            except (TypeError, ValueError) as e:
                raise InvalidActionError(f"non-numeric discount for {k}") from e
            if d not in allowed:
                raise InvalidActionError(f"discount {d} not in {sorted(allowed)}")
            norm[str(k)] = d
        # keep only the known products (ignore extras)
        action["proposal"] = {str(pid): norm[str(pid)] for pid in product_ids}
    else:  # decision
        if action.get("decision") not in ("accept", "reject"):
            raise InvalidActionError("decision must be accept/reject")
    action.setdefault("content", "")
    return action
