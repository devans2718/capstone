#!/usr/bin/env python3
"""Transfer-probe candidates: unflapped /t/ and /d/ where Buckeye flaps.

HOW TO RUN (in .venv-app; the corpus and the span grid's extraction mounted)
    .venv-app/bin/python scripts/count_transfer_candidates.py

    Read-only. Prints aggregates and writes nothing. Loads the span grid's
    sites through run_probes.SpanData and no prediction of any kind.

WHY THIS EXISTS
    The transfer probe (planned 2026-11-20) trains /t/ vs /d/ on stops where
    the contrast is pronounced and tests on the flaps of held-out words. Its
    training set is the stops that occur where the flaps occur but were not
    flapped. This script sizes that set. It decides whether the probe is
    viable, not whether it succeeds. The counts below were read on 2026-11-21
    and the definition was chosen from them, before any probe exists.

THE DEFINITION (decided 2026-11-21)
    A token is a canonical /t/ or /d/ surfacing as the same stop ([t] or [d]),
    under count_flap_sites' token rules: canonical and actual strings of equal
    length, tiers that agree, no '*' annotator mark. It counts when:
      - its neighbours on the phone tier, within or across words, are in the
        span grid's §2 classes: a vowel (nasalized included), a sonorant coda
        or a syllabic consonant before it, and a vowel, a syllabic consonant
        or 'hh' after it;
      - its position is ATTESTED: the same word type at the same position in
        its citation form is among the span grid's 10,334 flap sites, and that
        position flaps in at least MIN_FLAP_SHARE (0.40) of its tokens in this
        context;
      - the position is not word-initial.

    Buckeye's citation forms carry no stress, so stress is handled by the
    corpus: a position Buckeye does not flap (attack, before a stressed vowel)
    is out. Matching the flaps' phone contexts alone would let those
    aspirated stops in; that count is printed for comparison. With "at least
    one flap" instead of a share, rare flaps admit whole positions (time's
    onset). Word-initial positions are mostly reduced 'to'. They go back in if
    a larger corpus is put in front of this pipeline.

    No threshold brings the set's /d/ share near the flaps' (about 23%): the
    table below shows it falling only slowly. 0.40 is where the set stops
    being cheap to shrink. The probe's balanced class weights absorb most of
    the mismatch.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
from count_flap_sites import DEFAULT_ROOT, phones_in, read_phones, read_words  # noqa: E402

MIN_FLAP_SHARE = 0.40
THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70)

VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy",
          "uh", "uw"}
SYLLABIC = {"el", "em", "en", "eng"}
CODA = {"r", "l", "n", "m", "ng", "nx"}


def is_vowel(p: str) -> bool:
    """A vowel label, or its nasalized form (Buckeye appends 'n': 'aan')."""
    return p in VOWELS or (p.endswith("n") and p[:-1] in VOWELS)


def prev_ok(p: str) -> bool:
    return is_vowel(p) or p in SYLLABIC or p in CODA


def next_ok(p: str) -> bool:
    return is_vowel(p) or p in SYLLABIC or p == "hh"


def collect(grid_key):
    """Every usable canonical /t/ or /d/ token, with its phone-tier neighbours.

    Returns tuples (speaker, word, idx, canon, surface, prev, next, is_grid_flap,
    track, phone_k): the last two locate the token in its .phones tier, for
    export_transfer_stops.py; nothing here reads them.
    """
    tokens = []
    for wf in sorted(DEFAULT_ROOT.rglob("*.words")):
        pf = wf.with_suffix(".phones")
        if not pf.exists():
            continue
        phones = read_phones(pf)
        pos_of = {id(p): k for k, p in enumerate(phones)}
        for w_start, w_end, word, canon, actual, _ in read_words(wf):
            if not any(t in canon for t in ("t", "d")) or len(canon) != len(actual):
                continue
            wp = phones_in(phones, w_start, w_end)
            if len(wp) != len(actual):
                continue
            for idx, (c, a) in enumerate(zip(canon, actual)):
                if c not in ("t", "d"):
                    continue
                ph = wp[idx]
                if ph.label != a or ph.mark:
                    continue
                k = pos_of[id(ph)]
                prev = phones[k - 1].label if k > 0 else ""
                nxt = phones[k + 1].label if k + 1 < len(phones) else ""
                is_flap = (wf.stem, round(float(ph.start), 4)) in grid_key
                tokens.append((wf.parent.name, word, idx, c, a, prev, nxt, is_flap,
                               wf.stem, k))
    return tokens


def load_context():
    """The grid's flaps and every in-context token, as main() and the export use them.

    Returns (n_grid, attested, ctx, share, n_flap): attested is the set of
    (word, position) pairs among the grid's flap sites; ctx the tokens inside
    the §2 class context; share and n_flap the flap share and flap count of
    each (word, position) in that context.
    """
    import numpy as np

    d = rp.SpanData()
    inv = d.inv
    g = np.flatnonzero(d.in_grid)
    grid_key = {(inv.track[i], round(float(inv.start[i]), 4)) for i in g}
    assert len(grid_key) == len(g)
    attested = {(inv.word[i], int(inv.phone_idx_in_word[i])) for i in g}

    tokens = collect(grid_key)
    if sum(t[7] for t in tokens) != len(g):
        raise SystemExit("not every grid flap site was found on the phone tier")
    ctx = [t for t in tokens if prev_ok(t[5]) and next_ok(t[6])]

    n_pos, n_flap = Counter(), Counter()
    for t in ctx:
        n_pos[t[1], t[2]] += 1
        n_flap[t[1], t[2]] += t[4] == "dx"
    share = {k: n_flap[k] / n_pos[k] for k in n_pos}
    return len(g), attested, ctx, share, n_flap


def kept_positions(share, n_flap, attested, thr):
    """The definition's positions at a minimum flap share: attested, not
    word-initial, flapped at least once, flapping in >= thr of their tokens."""
    return {k for k, s in share.items()
            if k in attested and k[1] != 0 and n_flap[k] > 0 and s >= thr}


def stops_at(ctx, keep):
    """Unflapped tokens, surfacing as their canonical stop, at the kept positions."""
    return [t for t in ctx if not t[7] and t[4] == t[3] and (t[1], t[2]) in keep]


def summarize(stops):
    nd = sum(t[3] == "d" for t in stops)
    types = Counter(t[1] for t in stops)
    return {
        "n": len(stops), "n_d": nd, "d_share": nd / max(len(stops), 1),
        "types": len(types), "d_types": len({t[1] for t in stops if t[3] == "d"}),
        "speakers_d": len({t[0] for t in stops if t[3] == "d"}),
        "top10": sum(n for _, n in types.most_common(10)) / max(len(stops), 1),
        "top_types": types.most_common(12),
    }


def main() -> int:
    n_grid, attested, ctx, share, n_flap = load_context()
    flaps = [t for t in ctx if t[7]]

    print("Transfer-probe candidates: unflapped /t/ and /d/ at attested flap positions")
    print(f"corpus {DEFAULT_ROOT}; {n_grid:,} span-grid flap sites, "
          f"{len(flaps):,} of them inside the class context")
    print()

    ctx_only = summarize([t for t in ctx if not t[7] and t[4] == t[3]])
    print(f"For comparison, phone context alone (no attestation): {ctx_only['n']:,} stops, "
          f"{ctx_only['d_share']:.1%} /d/, {ctx_only['types']} types; mostly positions "
          "Buckeye does not flap")
    print()

    print("Attested, non-initial positions, by the position's minimum flap share")
    print(f"{'min share':>9}  {'stops':>6}  {'/d/':>5}  {'/d/ %':>5}  {'types':>5}  "
          f"{'d types':>7}  {'spk d':>5}  {'top10':>5}  {'flaps covered':>13}  {'their /d/':>9}")
    for thr in THRESHOLDS:
        keep = kept_positions(share, n_flap, attested, thr)
        s = summarize(stops_at(ctx, keep))
        cov = [t for t in flaps if (t[1], t[2]) in keep]
        cd = sum(t[3] == "d" for t in cov) / max(len(cov), 1)
        mark = "  <- decided" if thr == MIN_FLAP_SHARE else ""
        print(f"{thr:>9.2f}  {s['n']:>6,}  {s['n_d']:>5,}  {s['d_share']:>5.1%}  {s['types']:>5}  "
              f"{s['d_types']:>7}  {s['speakers_d']:>5}  {s['top10']:>5.1%}  {len(cov):>13,}  "
              f"{cd:>9.1%}{mark}")
    print()

    keep = kept_positions(share, n_flap, attested, MIN_FLAP_SHARE)
    s = summarize(stops_at(ctx, keep))
    print(f"Decided set (>= {MIN_FLAP_SHARE:.2f}, word-initial out): {s['n']:,} stops, "
          f"{s['n_d']:,} /d/ ({s['d_share']:.1%}), {s['types']} types, "
          f"{s['speakers_d']}/40 speakers with a /d/")
    print("  top types: " + ", ".join(f"{w} {n}" for w, n in s["top_types"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
