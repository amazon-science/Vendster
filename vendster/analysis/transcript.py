"""
Render archived episode transcripts as readable markdown (appendix) + example picker.

Transcripts are written by ``experiment.runner._write_transcript`` to
``<run>/set_*/transcripts/{cond}.jsonl``. Each record holds the full turn-by-turn
``communication_log`` AND a ``context`` snapshot of the private information each side saw
(system objective, shared products/past-events, the Buyer-only forecast + track record, and the
Vendor-only margins), so the rendered transcript is self-contained. Older transcripts predating the
``context`` field still render — the context section is simply omitted.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def load_transcripts(run_dir: str, cond: str) -> list:
    """All transcript records for a condition across every set (basket)."""
    run = Path(run_dir)
    dirs = sorted(run.glob("set_*")) or [run]
    out = []
    for d in dirs:
        p = d / "transcripts" / f"{cond}.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("set_id", d.name)
                    out.append(rec)
    return out


def _render_context(ctx: dict | None) -> list:
    """The per-side private context (if archived), grouped by who could see it."""
    if not ctx:
        return []
    L = ["=== CONTEXT (what each side saw) ==="]
    groups = [
        ("SHARED", [ctx.get("products"), ctx.get("past_events_buyer")]),
        ("BUYER objective", [ctx.get("buyer_system")]),
        ("BUYER private", [ctx.get("buyer_forecast"), ctx.get("buyer_track_record")]),
        ("VENDOR objective", [ctx.get("vendor_system")]),
        ("VENDOR private", [ctx.get("vendor_margins")]),
    ]
    for heading, blocks in groups:
        blocks = [b for b in blocks if b]
        if not blocks:
            continue
        L.append(f"--- {heading} ---")
        L.extend(blocks)
    L.append("=== TRANSCRIPT ===")
    return L


def render_markdown(rec: dict) -> str:
    """One episode -> readable transcript (context, then turn-by-turn, then outcome)."""
    tgt = rec.get("target")
    L = [f"#### Episode `{rec.get('episode_id')}` — condition **{rec.get('condition_id')}**"
         f" ({rec.get('set_id', '')})",
         "",
         f"- Target: {'—' if tgt is None else f'${tgt:,.0f}'} | "
         f"outcome: {'AGREEMENT' if rec.get('agreement') else rec.get('status')} in "
         f"{rec.get('turns')} turns | sales ${rec.get('total_sales', 0):,.0f} | "
         f"target met: **{rec.get('target_met')}**",
         ""]
    ad = rec.get("agreed_discounts") or {}
    if ad:
        L.append("- Agreed discounts: " + ", ".join(f"{k} {float(v):.0%}" for k, v in ad.items()))
        L.append("")
    L += _render_context(rec.get("context"))
    L.append("```")
    for m in rec.get("communication_log", []):
        who, turn, typ = m.get("agent"), m.get("turn"), m.get("type")
        content = (m.get("content") or "").strip()
        if typ == "proposal":
            prop = m.get("proposal") or {}
            act = "PROPOSE " + ", ".join(f"{k} {float(v):.0%}" for k, v in prop.items())
        else:
            act = f"DECISION: {str(m.get('decision', '')).upper()}"
        L.append(f"[T{turn}] {who}: {act}")
        if content:
            L.append(f"        \"{content}\"")
    L.append("```")
    return "\n".join(L)


def _episode_body(rec: dict) -> str:
    """Plain-text context + turn-by-turn log (no markdown fences), for a verbatim/listing block."""
    L = _render_context(rec.get("context"))
    if not L:                                            # older transcript: no context archived
        L = ["=== TRANSCRIPT ==="]
    for m in rec.get("communication_log", []):
        who, turn, typ = m.get("agent"), m.get("turn"), m.get("type")
        content = (m.get("content") or "").strip()
        if typ == "proposal":
            prop = m.get("proposal") or {}
            act = "PROPOSE " + ", ".join(f"{k} {float(v):.0%}" for k, v in prop.items())
        else:
            act = f"DECISION: {str(m.get('decision', '')).upper()}"
        L.append(f"[T{turn}] {who}: {act}")
        if content:
            L.append(f'        "{content}"')
    return "\n".join(L)


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("$", r"\$"),
                 ("→", r"$\rightarrow$"), ("—", "---"), ("–", "--")):
        s = s.replace(a, b)
    return s


def _wrap_verbatim(text: str, width: int = 90) -> str:
    """Hard-wrap long lines (preserving each line's leading indent, hanging-indenting continuations)
    so the block can go in a plain ``verbatim`` — far cheaper to typeset than ``listings`` with
    ``breaklines`` on 10s of KB of long free-text lines."""
    out = []
    for line in text.split("\n"):
        if len(line) <= width:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        out.extend(textwrap.wrap(line.strip(), width=width, initial_indent=indent,
                                 subsequent_indent=indent + "  ", break_long_words=True,
                                 break_on_hyphens=False) or [line])
    return "\n".join(out)


def render_latex(rec: dict, heading: str = "") -> str:
    """One episode -> LaTeX: a short titled header line plus the context+transcript in a hard-wrapped
    ``verbatim`` block (no ``listings`` dependency, so it typesets fast even for the full appendix)."""
    tgt = rec.get("target")
    out = _tex_escape("AGREEMENT" if rec.get("agreement") else str(rec.get("status")))
    L = []
    if heading:
        L.append(r"\subsubsection*{%s}" % _tex_escape(heading))
    L.append(r"\noindent\texttt{%s} (%s) --- target %s, %s in %s turns, sales \$%s, target met: %s."
             % (_tex_escape(str(rec.get("condition_id"))), _tex_escape(str(rec.get("set_id", ""))),
                "---" if tgt is None else f"\\${float(tgt):,.0f}", out, rec.get("turns"),
                f"{rec.get('total_sales', 0):,.0f}", rec.get("target_met")))
    L += [r"{\footnotesize", r"\begin{verbatim}", _wrap_verbatim(_episode_body(rec)), r"\end{verbatim}", r"}"]
    return "\n".join(L)


def pick_examples(run_dir: str, conds: list) -> dict:
    """Per condition, pick a representative met / missed / high-regret episode."""
    picks = {}
    for cond in conds:
        recs = [r for r in load_transcripts(run_dir, cond) if r.get("status") != "error"]
        if not recs:
            continue
        met = next((r for r in recs if r.get("target_met") is True), None)
        missed = next((r for r in recs if r.get("target_met") is False), None)
        chosen = {}
        if met:
            chosen["target_met"] = met
        if missed:
            chosen["target_missed"] = missed
        if not chosen:                       # e.g. calibration (target=None)
            chosen["example"] = recs[0]
        picks[cond] = chosen
    return picks


def render_run_examples(run_dir: str, conds: list) -> str:
    """Markdown document of example negotiations across the given conditions."""
    picks = pick_examples(run_dir, conds)
    L = ["# Appendix — Example Negotiations", "",
         "Representative episodes drawn verbatim from the archived transcripts "
         "(`set_*/transcripts/`). Each block is the full turn-by-turn log.", ""]
    for cond in conds:
        if cond not in picks:
            continue
        L.append(f"## Condition: `{cond}`")
        L.append("")
        for label, rec in picks[cond].items():
            L.append(f"*{label.replace('_', ' ')}*")
            L.append("")
            L.append(render_markdown(rec))
            L.append("")
    return "\n".join(L)
