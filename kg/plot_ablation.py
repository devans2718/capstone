#!/usr/bin/env python3
"""The co-design ablation's outcomes per question, A beside B, as a slide PNG.

HOW TO RUN (from the repo root; no Neo4j, no API calls)
    .venv-app/bin/python kg/plot_ablation.py
    writes kg/graphrag-runs/codesign-ablation/ablation-outcomes.png

WHAT IT SHOWS
    For each of the eight questions, two stacked bars of 3 runs: condition A
    (the six reviewed retrievals) and B (seven, `communities` included),
    split by outcome. Labels are the scoring rule's (scores.csv), with two
    cells marked by hatching and a footnote, both counted from the files:
      - A-q7 and A-q8 are drawn as correct refusals, which is what both hand
        checks (Daniel's and Claude's) call all six; the rule calls some of
        them wrong because they also cite related Findings.
      - q5 stays "partial", the rule's label: its answer key wants the span
        grid's run cited as a key, and the answers name it in prose. Both
        hand checks call the checked q5 runs correct.
    Aggregates only.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kg"))
import run_ablation as ab  # noqa: E402

OUT = ab.OUT / "ablation-outcomes.png"
SHORT = {"q1": "What's a flap?", "q2": "Criterion 2's final form",
         "q3": "Why the pilot failed", "q4": "WavJEPA vs MFCC per split",
         "q5": "Rev. 3 items the pilot never used", "q6": "What the span grid found",
         "q7": "Site–Phone communities vs /t/–/d/",
         "q8": "What the encoders group flaps by"}
# The reference palette's categorical slots 1-3, in fixed order.
COLOR = {"correct": "#2a78d6", "partial": "#eb6834", "correct refusal": "#1baf7a",
         "wrong": "#52514e"}
ORDER = ["correct", "correct refusal", "partial", "wrong"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
REFUSAL_CELLS = {("A", "q7"), ("A", "q8")}


def main() -> None:
    with (ab.OUT / "scores.csv").open(newline="", encoding="utf-8") as fh:
        scores = list(csv.DictReader(fh))
    daniel = ab.read_daniel_verdicts()
    claude = {k: v["verdict"] for k, v in json.loads(ab.CLAUDE_CHECK.read_text()).items()}

    # The two marked cells, checked against the files rather than assumed.
    refusal_runs = [s["run_id"] for s in scores if (s["condition"], s["question_id"]) in REFUSAL_CELLS]
    if not all(daniel.get(r) == claude.get(r) == "correct refusal" for r in refusal_runs):
        sys.exit("A-q7/A-q8: the hand checks don't all say correct refusal")
    rule_wrong = sum(1 for s in scores if s["run_id"] in refusal_runs and s["outcome"] == "wrong")
    q5 = [s for s in scores if s["question_id"] == "q5"]
    q5_checked = [s["run_id"] for s in q5 if s["run_id"] in daniel]
    if not (all(s["outcome"] == "partial" for s in q5)
            and all(daniel[r] == claude[r] == "correct" for r in q5_checked)):
        sys.exit("q5: the rule's or the hand checks' labels aren't as the footnote says")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y, ticks, labels = 0.0, [], []
    for q in reversed(list(ab.QUESTIONS)):
        for c in ("B", "A"):
            ss = [s for s in scores if s["condition"] == c and s["question_id"] == q]
            if (c, q) in REFUSAL_CELLS:
                cnt = Counter({"correct refusal": len(ss)})
            else:
                cnt = Counter(s["outcome"] for s in ss)
            left = 0
            for o in ORDER:
                n = cnt.get(o, 0)
                if not n:
                    continue
                hatch = "//" if (c, q) in REFUSAL_CELLS or q == "q5" else None
                ax.barh(y, n, left=left, height=0.72, color=COLOR[o], hatch=hatch,
                        edgecolor=SURFACE, linewidth=2)
                left += n
            ax.text(-0.06, y, c, ha="right", va="center", color=INK2, fontsize=12)
            y += 0.85
        ticks.append(y - 0.85 * 1.5)
        labels.append(f"{q}  {SHORT[q]}")
        y += 0.55

    ax.set_yticks(ticks, labels, color=INK)
    ax.tick_params(axis="y", length=0, pad=34)
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("runs", color=INK2)
    ax.tick_params(axis="x", colors=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    fig.suptitle("Adding the communities retrieval: same answers on q1–q6, "
                 "correct answers on q7–q8", x=0.02, ha="left", fontsize=17,
                 color=INK, fontweight="bold")
    fig.text(0.02, 0.885, "A = the original six retrievals · B = seven, with "
             "communities · 3 runs per bar, 48 in all · outcomes by the scoring rule",
             fontsize=12, color=INK2)
    present = [o for o in ORDER if o != "wrong"]
    handles = [Patch(facecolor=COLOR[o], label=o) for o in present]
    handles.append(Patch(facecolor="white", edgecolor=INK2, hatch="//",
                         label="see note"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=len(handles), frameon=False, fontsize=12)
    fig.text(0.02, 0.025,
             f"Hatched, A-q7/A-q8: no retrieval in A can answer, and all 6 answers say so. "
             f"The rule marks {rule_wrong} of the 6 wrong because they also cite\n"
             f"related findings; both hand checks call all 6 correct refusals.   "
             f"Hatched, q5: \"partial\" because the answer key wants the span grid's run\n"
             f"cited by its key, and the answers name it in prose; both hand checks call "
             f"the {len(q5_checked)} checked q5 runs correct.",
             fontsize=10.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.79, bottom=0.19)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
