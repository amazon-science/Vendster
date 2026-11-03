"""
Assemble a curated, model-diverse set of example negotiation transcripts for the appendix — one
representative episode per (model, condition, situation), NOT the full cartesian (that would be
26×4 = 104). Each model appears in a situation that illustrates its character; the full archive is
in a run's transcripts (`experiments/panel/<model>/experiment/set_*/transcripts/`; the committed
archive ships none, so rerun bundles first).

    uv run python scripts/make_transcripts.py --out experiments/panel/example_transcripts.md
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendster.analysis.transcript import load_transcripts, render_latex, render_markdown
from vendster.core.config import EVALUATED_PANEL

PANEL = {k: d for k, (_, d) in EVALUATED_PANEL.items()}   # short key -> experiment dir

# (model, condition, situation heading, selector). Diverse models × diverse situations; non-cartesian.
CURATED = [
    ("deepseek", "no_forecast", "No forecast — baseline negotiation (no demand signal)", "any"),
    ("opus", "chronos2", "Foundation-model forecast (Chronos-2) — buyer uses the forecast", "met"),
    ("opus", "bias_p20", "Over-forecast → complacency → undershoots the target", "missed"),
    ("gpt", "bias_m20", "Under-forecast → aggressive negotiation → hits the target", "met"),
    ("gpt", "bias_m20", "Aggressive negotiation → NO deal (surplus destroyed, vendor gets nothing)", "failed"),
    ("qwen", "unbiased", "Perfect (oracle) forecast on a weak-reactive model — behavior barely shifts", "any"),
]


def _pick(recs, sel):
    recs = [r for r in recs if r.get("status") != "error"]
    if not recs:
        return None
    if sel == "met":
        r = next((x for x in recs if x.get("target_met") is True and x.get("agreement")), None)
    elif sel == "missed":
        r = next((x for x in recs if x.get("agreement") and x.get("target_met") is False), None)
    elif sel == "failed":
        r = next((x for x in recs if not x.get("agreement")), None)
    else:
        r = None
    return r or recs[0]                                   # fall back to first if the target situation is absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", default="experiments/panel")
    ap.add_argument("--out", default="experiments/panel/example_transcripts.md")
    ap.add_argument("--tex", default=None,
                    help="also emit an \\input-able LaTeX fragment (listings) for the paper appendix")
    args = ap.parse_args()

    # Wording matches the paper verbatim, so `--tex` regeneration is lossless (no hand edits to
    # re-apply). The \label is what main.tex's \ref{app:transcripts} resolves against.
    lead = ("Representative episodes drawn verbatim from the archived transcripts, chosen to span "
            "models and situations. This is not the full 26x4 grid; the code regenerates any "
            "episode's transcript. Each block shows the private context both sides saw followed by the "
            "turn-by-turn negotiation.")
    L = ["# Appendix — Example Negotiations", "", lead, ""]
    T = [r"\subsection{Example Negotiations}", r"\label{app:transcripts}", "", lead, ""]
    missing = []
    for model, cond, heading, sel in CURATED:
        run = f"{args.panel_root}/{PANEL[model]}/experiment"
        rec = _pick(load_transcripts(run, cond), sel)
        if rec is None:
            missing.append((model, cond, sel)); continue
        L += [f"## {model} — {heading}", "", render_markdown(rec), ""]
        T += [render_latex(rec, f"{model}: {heading}"), ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(CURATED) - len(missing)}/{len(CURATED)} transcripts)")
    if args.tex and missing:
        # The paper's appendix is only rewritten from a complete curated set; a release with
        # withheld transcripts must not silently drop examples from the manuscript.
        print(f"left {args.tex} unchanged: {len(missing)} curated transcript(s) not in this archive")
    elif args.tex:
        Path(args.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.tex).write_text("\n".join(T) + "\n")
        print(f"wrote {args.tex}")
    for m, c, s in missing:
        print(f"  MISSING: {m} {c} ({s})")


if __name__ == "__main__":
    main()
