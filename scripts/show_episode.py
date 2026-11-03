"""
Render archived episode transcripts as readable markdown (ad-hoc inspection).

    # all example negotiations (per-condition met/missed) for a run:
    uv run python scripts/show_episode.py --run experiments/panel/opus_4_6/experiment

    # a specific condition, or a specific episode id:
    uv run python scripts/show_episode.py --run experiments/panel/opus_4_6/experiment \
        --cond chronos2 --episode chronos2_3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendster.analysis.aggregate import condition_groups
from vendster.analysis.transcript import load_transcripts, render_markdown, render_run_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", default=None, help="single condition (default: all, met/missed picks)")
    ap.add_argument("--episode", default=None, help="exact episode_id within --cond")
    ap.add_argument("--out", default=None, help="write markdown here instead of stdout")
    args = ap.parse_args()

    if args.cond:
        recs = load_transcripts(args.run, args.cond)
        if args.episode:
            recs = [r for r in recs if r.get("episode_id") == args.episode]
        md = "\n\n".join(render_markdown(r) for r in recs) or "(no matching transcripts)"
    else:
        real, grid = condition_groups()
        md = render_run_examples(args.run, real + grid)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
