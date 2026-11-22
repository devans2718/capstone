#!/usr/bin/env python3
"""How frequent are the words behind the flaps at each word position?

HOW TO RUN (in .venv-app; the corpus and the span grid's extraction mounted)
    .venv-app/bin/python scripts/count_position_frequency.py

    Read-only. Prints aggregates and writes nothing. Loads the span grid's
    sites through run_probes.SpanData and no prediction of any kind.

WHY THIS EXISTS
    The position re-score (span-grid §8, 2026-11-21; scripts/score_position.py)
    named frequency as a confound of word position but did not measure it.
    This measures it, after the fact and unregistered, for the paper's
    Discussion: frequent words are reduced more in conversation, so if
    frequency follows the label within a position group, a probe could
    separate /t/ from /d/ there by how reduced the flap is, without the
    contrast.

WHAT IT COUNTS
    Position groups as in score_position.py: word-initial when the flap is
    the citation form's first segment, word-final when it is the last, medial
    otherwise. A word's frequency is its token count over every Buckeye
    .words file, any pronunciation. Medians are per flap site (a word counts
    once per flap) and per word type; "top 100" is the corpus's hundred most
    frequent words.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
from count_flap_sites import DEFAULT_ROOT, read_words  # noqa: E402

GROUPS = ("initial", "medial", "final")
TOP_N = 100


def position(idx: int, n_canon: int) -> str:
    """The flap's position group, from its index in the citation form."""
    if idx == 0:
        return "initial"
    return "final" if idx == n_canon - 1 else "medial"


def main() -> int:
    import numpy as np

    d = rp.SpanData()
    inv = d.inv
    g = np.flatnonzero(d.in_grid)

    counts = Counter()
    for wf in DEFAULT_ROOT.rglob("*.words"):
        for _, _, word, *_ in read_words(wf):
            counts[word] += 1
    rank = {w: r for r, (w, _) in enumerate(counts.most_common(), 1)}

    by_group = {k: [] for k in GROUPS}
    for i in g:
        canon = inv.canonical_phones[i]
        n = len(canon.split()) if isinstance(canon, str) else len(canon)
        by_group[position(int(inv.phone_idx_in_word[i]), n)].append(i)
    if sum(map(len, by_group.values())) != len(g):
        raise SystemExit("position groups do not partition the grid's sites")

    print("Corpus frequency of the words behind each position group's flaps")
    print(f"corpus {DEFAULT_ROOT}: {sum(counts.values()):,} word tokens, "
          f"{len(counts):,} types; {len(g):,} span-grid flap sites")
    for name in GROUPS:
        idxs = by_group[name]
        words = [inv.word[i] for i in idxs]
        types = Counter(words)
        is_d = np.array([bool(d.is_d[i]) for i in idxs])
        per_site = np.array([counts[w] for w in words])
        per_type = np.array([counts[w] for w in types])
        top = np.array([rank[w] <= TOP_N for w in words])
        top10 = sum(n for _, n in types.most_common(10)) / len(words)
        print(f"\n== word-{name}: {len(idxs):,} flaps, {len(types)} types, /d/ {is_d.mean():.1%}")
        print(f"  corpus count per flap: median {int(np.median(per_site)):,}; "
              f"per type: median {int(np.median(per_type)):,}")
        print(f"  flaps in a top-{TOP_N} word: {top.mean():.1%}; "
              f"the group's top 10 types hold {top10:.1%}")
        for label, m in (("/t/", ~is_d), ("/d/", is_d)):
            print(f"    {label}: median corpus count per flap {int(np.median(per_site[m])):,}, "
                  f"in a top-{TOP_N} word {top[m].mean():.1%}")
        print("  top types: " + ", ".join(f"{w} {n}" for w, n in types.most_common(10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
