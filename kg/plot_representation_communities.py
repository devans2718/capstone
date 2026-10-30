#!/usr/bin/env python3
"""Draw the representation communities: the comparison table, and /t/-/d/ per community.

EXPLORATORY, NOT REGISTERED. Reads kg/communities/representation/summary.json
(written by kg/representation_communities.py) and nothing else: no Neo4j, no
per-site data.

HOW TO RUN (from the repo root, in WSL)
    .venv-app/bin/python kg/plot_representation_communities.py OUT_DIR
    Writes OUT_DIR/representation-comparison.png and
    OUT_DIR/representation-crosstab.png.

WHAT THEY SHOW
    representation-comparison  left: AMI of each representation's communities
                               (and the 10-28 Site-Phone partition, for
                               reference) with the environments, speaker, word,
                               lemma and the Site-Phone partition; the seeded
                               value, with its range over seeds 1-20 beneath.
                               Right: AMI with /t/-/d/ on its own axis, since
                               a binary label's AMI is an order of magnitude
                               smaller, with the vowel-following pair's own AMI
                               with /t/-/d/ as a reference line.
    representation-crosstab    per encoder and MFCC, each community's /d/
                               share with two 95% intervals: Wilson (thin; sites
                               as independent, what plot_communities.py draws)
                               and resampled by word (thick). Coloured by
                               whether the word-resampled interval excludes the
                               base rate.

THE BUCKEYE RULE
    Aggregates only. Labels are community numbers, representation names and
    counts; no words, lemmas, site IDs or speaker codes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_communities import BLUE, GRID, INK, INK_2, MID, ORANGE, SURFACE, style  # noqa: E402

SUMMARY = Path(__file__).resolve().parent / "communities" / "representation" / "summary.json"

# The dataviz skill's reference sequential ramp (blue, steps 100-700).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

NAMES = {"wav2vec2": "wav2vec2-base", "wavjepa": "WavJEPA", "mfcc": "MFCC"}
COLUMNS = [("vowel", "preceding\nvowel"), ("following", "following\nsegment"),
           ("pair", "vowel–\nfollowing"), ("speaker", "speaker"), ("word", "word"),
           ("lemma", "lemma"), ("site_phone", "Site–Phone\npartition")]
CAPTION = ("Exploratory, not registered: no probe trained. ɾ span, 10,334 sites; "
           "kNN graph (z-scored, cosine, k = 15), Louvain, seed 20261030.")


def row_label(rep: str, r: dict) -> str:
    return f"{NAMES[rep]} L{r['layer']}\n{r['run']['n_communities']} communities"


def comparison_figure(s: dict, out: Path) -> None:
    reps = s["representations"]
    ref = s["site_phone_reference"]
    rows = [(row_label(k, v), v["comparison"], v["stability"]["ami_range"])
            for k, v in reps.items()]
    rows.append(("Site–Phone (10-28)\n20 communities", ref["comparison"], None))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.08})
    M = np.array([[r[1][c] for c, _ in COLUMNS] for r in rows])
    mask = np.zeros_like(M, dtype=bool)
    mask[-1, -1] = True                                   # the partition against itself
    vmax = float(M[~mask].max())
    ax.imshow(np.ma.masked_array(M, mask), cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
    for i, (_, comp, rng) in enumerate(rows):
        for j, (c, _) in enumerate(COLUMNS):
            if mask[i, j]:
                ax.text(j, i, "—", ha="center", va="center", color=INK_2, fontsize=10)
                continue
            light = M[i, j] > 0.55 * vmax
            colour = SURFACE if light else INK
            ax.text(j, i - 0.12, f"{comp[c]:.3f}", ha="center", va="center",
                    color=colour, fontsize=10.5, weight="bold")
            if rng:
                ax.text(j, i + 0.22, f"{rng[c][0]:.3f}–{rng[c][1]:.3f}", ha="center",
                        va="center", color=colour, fontsize=7.5)
    ax.set_xticks(range(len(COLUMNS)), [l for _, l in COLUMNS], fontsize=9, color=INK_2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(COLUMNS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    fig.suptitle("What each representation's communities share with each grouping "
                 "(adjusted mutual information)", color=INK, fontsize=12.5, x=0.01,
                 ha="left", y=1.06)

    style(bx)
    y = np.arange(len(rows))
    for i, (_, comp, rng) in enumerate(rows):
        if rng:
            bx.plot(rng["label"], [i, i], color=INK_2, lw=2, solid_capstyle="round", zorder=2)
        bx.scatter(comp["label"], i, s=60, color=BLUE if rng else MID, edgecolors=SURFACE,
                   linewidths=2, zorder=3)
        bx.annotate(f"{comp['label']:.4f}   V {comp['label_cramers_v_corrected']:.3f}",
                    (comp["label"], i), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    pair = s["label_ami_of_the_labels_themselves"]["pair"]
    bx.axvline(pair, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    bx.text(pair, len(rows) - 0.55, " vowel–following\n pair itself", color=INK_2,
            fontsize=8, va="bottom")
    bx.set_ylim(len(rows) - 0.5, -0.5)
    bx.set_yticks(y, [""] * len(rows))
    bx.set_xlim(0, pair * 1.25)
    bx.grid(axis="x", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.set_xlabel("AMI with underlying /t/–/d/\n(V: bias-corrected Cramér's V)",
                  color=INK_2, fontsize=9)
    bx.set_title("/t/–/d/", color=INK, fontsize=11, loc="left", pad=34)
    fig.text(0.01, -0.04, CAPTION + "\nSmall figures: the range over stability seeds 1–20, "
             "which don't include the seeded run, so a seeded value can sit outside it.",
             color=INK_2, fontsize=8.5)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def crosstab_figure(s: dict, out: Path) -> None:
    base = s["sites"]["base_share_d"]
    reps = s["representations"]
    n_max = max(len(v["communities"]) for v in reps.values())
    fig, axes = plt.subplots(1, len(reps), figsize=(16, 0.42 * n_max + 2.2),
                             facecolor=SURFACE, sharex=True)
    for ax, (rep, v) in zip(axes, reps.items()):
        style(ax)
        cs = v["communities"]
        for yi, c in enumerate(cs):
            lo, hi = c["word_resampled_95"]
            colour = ORANGE if lo > base else BLUE if hi < base else MID
            ax.plot([lo, hi], [yi, yi], color=colour, lw=5, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot(c["wilson_95"], [yi, yi], color=INK, lw=1.2, solid_capstyle="round", zorder=3)
            ax.scatter(c["share_d"], yi, s=50, color=colour, edgecolors=SURFACE,
                       linewidths=2, zorder=4)
        ax.axvline(base, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(range(len(cs)), [
            f"C{c['number']}  {c['n_sites']:,} sites · {c['n_words']} words · "
            f"top 5 {100 * c['top5_word_share']:.0f}%" for c in cs], fontsize=8)
        ax.set_ylim(n_max - 0.5, -0.8)
        ax.set_xlim(0, 0.65)
        ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        iv = v["intervals"]
        ax.set_title(f"{NAMES[rep]} L{v['layer']}: {iv['exclude_base_word_resampled']} of "
                     f"{iv['n_communities']} exclude the base rate by word "
                     f"({iv['exclude_base_wilson']} by Wilson)",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("share of the community's sites with underlying /d/", color=INK_2,
                      fontsize=9)
    for ax in axes:
        ax.text(base, n_max - 0.6, f" base rate {100 * base:.1f}%", color=INK_2, fontsize=8)
    handles = [plt.Line2D([], [], color=INK, lw=1.2, label="95% Wilson (sites independent)"),
               plt.Line2D([], [], color=MID, lw=5, label="95% resampled by word")] + [
               plt.Line2D([], [], color=c, marker="o", lw=0, label=l)
               for c, l in ((BLUE, "word interval below base rate"),
                            (MID, "word interval covers it"),
                            (ORANGE, "word interval above base rate"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               labelcolor=INK_2, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Community × underlying /t/–/d/, by representation", color=INK,
                 fontsize=13, x=0.01, ha="left")
    fig.text(0.01, -0.1, CAPTION + " Word = type_key; 10,000 resamples. "
             "'Top 5': the five most frequent words' share of the community's tokens.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads(SUMMARY.read_text())
    comparison_figure(s, out / "representation-comparison.png")
    crosstab_figure(s, out / "representation-crosstab.png")
    print(f"wrote {out / 'representation-comparison.png'}, {out / 'representation-crosstab.png'}")


if __name__ == "__main__":
    main()
