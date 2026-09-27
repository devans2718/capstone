#!/usr/bin/env python3
"""Layer-wise curves from the pilot's results CSV (the RQ3 deliverable).

HOW TO RUN
    .venv-app/bin/python scripts/plot_layer_curves.py

    Reads results/flapping/probe-results-20260926.csv (written by
    scripts/run_probes.py assemble) and writes PNGs to
    results/flapping/figures/. It computes nothing new: every point and
    band on a figure is a row of the CSV.

WHAT EACH FIGURE SHOWS
    layer-curves-mean.png   One panel per condition, headline run, mean
                            pooling. WavJEPA's balanced accuracy by layer with
                            its word-type bootstrap 95% band, the MFCC
                            baseline as a line with its band, and chance. Every
                            panel shares one y axis, so the drop from random
                            to joint reads across panels.
    condition-drop.png      WavJEPA mean-pooled curves for all four conditions
                            on one axis, plus the subsampled secondary run
                            dashed: the "drop from random to disjoint splits"
                            plot the feedback asked for.
    controls.png            Per condition: the best WavJEPA layer, MFCC, the
                            control task at that same layer, and the
                            word-identity lookup, side by side. Under random and speaker_disjoint the lookup
                            and control are expected high, and under the
                            type-disjoint conditions near chance (§4).
    pooling.png             WavJEPA under joint for mean, max and midpoint
                            (the registered secondary comparison, §2).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "flapping" / "probe-results-20260926.csv"
FIG = REPO / "results" / "flapping" / "figures"

CONDITIONS = ("random", "speaker_disjoint", "word_disjoint", "joint")
# One hue per condition, kept across figures. The palette is colour-blind safe
# (Okabe-Ito), and the condition name is always printed beside its line too.
COLOUR = {"random": "#0072B2", "speaker_disjoint": "#009E73",
          "word_disjoint": "#E69F00", "joint": "#D55E00"}
MFCC_COLOUR = "#555555"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(CSV)
    FIG.mkdir(parents=True, exist_ok=True)
    head = df[~df.subsampled]
    wj = head[(head.encoder == "wavjepa")]
    mf = head[(head.encoder == "mfcc")]

    lo = min(df.ci_lo.min(), 0.5) - 0.02
    hi = df.ci_hi.max() + 0.02

    # ---------------------------------------------------------------- panels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        w = wj[(wj.condition == cond) & (wj.pooling == "mean")].sort_values("layer")
        m = mf[(mf.condition == cond) & (mf.pooling == "mean")].iloc[0]
        ax.fill_between(w.layer, w.ci_lo, w.ci_hi, color=COLOUR[cond], alpha=0.2, lw=0)
        ax.plot(w.layer, w.balanced_acc, "o-", color=COLOUR[cond], ms=4, label="WavJEPA")
        ax.axhspan(m.ci_lo, m.ci_hi, color=MFCC_COLOUR, alpha=0.12, lw=0)
        ax.axhline(m.balanced_acc, color=MFCC_COLOUR, ls="--", lw=1.2, label="MFCC")
        ax.axhline(0.5, color="black", lw=0.6, ls=":")
        ax.set_title(cond)
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, 13, 2))
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("balanced accuracy (pooled, 95% word-type CI)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("/t/ vs /d/ under a flap: WavJEPA layer by layer, mean pooling, "
                 "10,389 sites. Dotted line: chance.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "layer-curves-mean.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ the drop
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in CONDITIONS:
        w = wj[(wj.condition == cond) & (wj.pooling == "mean")].sort_values("layer")
        ax.plot(w.layer, w.balanced_acc, "o-", color=COLOUR[cond], ms=4)
        ax.annotate(cond, (12.2, w.balanced_acc.iloc[-1]), color=COLOUR[cond],
                    va="center", fontsize=9)
        s = df[(df.encoder == "wavjepa") & (df.condition == cond) & df.subsampled
               & (df.pooling == "mean")].sort_values("layer")
        if len(s):
            ax.plot(s.layer, s.balanced_acc, "--", color=COLOUR[cond], lw=1)
    ax.axhline(0.5, color="black", lw=0.6, ls=":")
    ax.set_xlim(-0.5, 14.5)
    ax.set_xticks(range(13))
    ax.set_xlabel("layer")
    ax.set_ylabel("balanced accuracy (pooled)")
    ax.set_title("What survives as the lexical and speaker routes close\n"
                 "solid: headline run; dashed: training subsampled to joint's size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "condition-drop.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ controls
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    bars = []
    for i, cond in enumerate(CONDITIONS):
        w = wj[(wj.condition == cond) & (wj.pooling == "mean")]
        best = w.loc[w.balanced_acc.idxmax()]
        m = mf[(mf.condition == cond) & (mf.pooling == "mean")].iloc[0]
        vals = [(best.balanced_acc, f"WavJEPA L{int(best.layer)}"),
                (m.balanced_acc, "MFCC"),
                (best.control_balanced_acc, "control task"),
                (best.lookup_balanced_acc, "word lookup")]
        for j, (v, _) in enumerate(vals):
            bars.append(ax.bar(i + (j - 1.5) * width, v, width,
                               color=["#0072B2", MFCC_COLOUR, "#CC79A7", "#F0E442"][j]))
            ax.text(i + (j - 1.5) * width, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
    ax.legend(bars[:4], ["WavJEPA, best layer", "MFCC", "control task, same layer",
                         "word-identity lookup"], fontsize=8, frameon=False,
              loc="upper right")
    ax.axhline(0.5, color="black", lw=0.6, ls=":")
    ax.set_xticks(range(4))
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("balanced accuracy (pooled)")
    ax.set_title("Controls by condition, mean pooling. "
                 "The lookup's collapse is the split doing its job.")
    fig.tight_layout()
    fig.savefig(FIG / "controls.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- pooling
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"mean": "o-", "max": "s--", "midpoint": "^:"}
    for pooling, st in styles.items():
        w = wj[(wj.condition == "joint") & (wj.pooling == pooling)].sort_values("layer")
        ax.plot(w.layer, w.balanced_acc, st, color=COLOUR["joint"], ms=4, label=f"WavJEPA {pooling}")
        m = mf[(mf.condition == "joint") & (mf.pooling == pooling)].iloc[0]
        ax.axhline(m.balanced_acc, color=MFCC_COLOUR, ls=st[1:], lw=1, label=f"MFCC {pooling}")
    ax.axhline(0.5, color="black", lw=0.6, ls=":")
    ax.set_xticks(range(13))
    ax.set_xlabel("layer")
    ax.set_ylabel("balanced accuracy (pooled)")
    ax.set_title("Pooling under joint (registered secondary; mean is the gated rule)")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "pooling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for p in sorted(FIG.glob("*.png")):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
