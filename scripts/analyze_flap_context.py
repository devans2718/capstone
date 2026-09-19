#!/usr/bin/env python3
"""Is word-final flapping environmentally determined rather than lexical?

WHY THIS EXISTS
    The flapping pilot's blocking finding is that the task is ~99.5% lexical:
    a word-identity lookup scores 0.9951 with no audio, because only 25 of
    1,022 word types carry both underlying classes.

    That should be a property of word-MEDIAL flaps specifically. In 'water',
    'better', 'city', the flap's environment is internal to the word and
    therefore fixed, so the word always flaps and word identity predicts the
    label perfectly.

    Word-FINAL flaps cannot work that way. 'not' flaps before 'a' and does not
    before 'fair' -- same type, both realizations, decided by the following
    word. For those tokens word identity does not determine the label, so the
    memorization route is unavailable by construction rather than by splitting.

    This script measures whether that population exists and how big it is.

WHAT IT REPORTS
    1. Realization of /t/ and /d/ by position (word-final vs word-medial).
    2. Whether word-final realization tracks the following segment, which is
       what "environmentally conditioned" has to mean if it is true.
    3. Word types attested with BOTH [t] and [dx] word-finally: how many types,
       how many tokens, how balanced within type, across how many speakers.
       A type that is 95% flapped is still nearly deterministic, so the balance
       matters as much as the count.

    Read-only. It sizes a candidate probe population; it does not build one.

ADJACENCY
    Flapping needs the following vowel to be adjacent. .words entries include
    silences and noise, which read_words() skips, so two consecutive yielded
    words are not necessarily adjacent in time. Adjacency is tested explicitly
    against the timestamps rather than assumed from iteration order.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_flap_sites import (  # noqa: E402
    DEFAULT_ROOT,
    phones_in,
    read_phones,
    read_words,
)

# Syllable nuclei in Buckeye's label set: ARPABET vowels, their nasalized
# variants (…n), and the syllabic consonants. Flapping is licensed before a
# nucleus, and 'little' -> [l ih dx el] shows syllabics count.
BASE_VOWELS = {
    "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er",
    "ey", "ih", "iy", "ow", "oy", "uh", "uw",
}
NUCLEI = BASE_VOWELS | {v + "n" for v in BASE_VOWELS} | {"el", "em", "en", "eng"}

TARGETS = ("t", "d")
ADJACENCY_TOL = 0.02  # seconds; annotation boundaries disagree by a frame or two


def classify_following(nxt) -> str:
    """'vowel' | 'consonant' | 'break' for the segment after a word-final stop."""
    if nxt is None:
        return "break"
    first = nxt[0]
    return "vowel" if first in NUCLEI else "consonant"


def collect(root: Path, keep_marked: bool = False):
    # by_position[(underlying, 'final'|'medial')][surface] = count
    by_position = defaultdict(Counter)
    # final_context[(underlying, surface)][following_class] = count
    final_context = defaultdict(Counter)
    # final_types[underlying][word][surface] = count
    final_types = defaultdict(lambda: defaultdict(Counter))
    medial_types = defaultdict(lambda: defaultdict(Counter))
    # speakers_of_type[underlying][word] = set of speakers
    speakers_of_type = defaultdict(lambda: defaultdict(set))
    stats = Counter()

    word_files = sorted(root.rglob("*.words"))
    if not word_files:
        raise SystemExit(f"no .words files under {root} — is the corpus mounted?")

    for wf in word_files:
        pf = wf.with_suffix(".phones")
        if not pf.exists():
            continue
        speaker = wf.parent.name
        phones = read_phones(pf)
        entries = list(read_words(wf))

        for i, (w_start, w_end, word, canon, actual, pos) in enumerate(entries):
            if not any(t in canon for t in TARGETS):
                continue
            if len(canon) != len(actual):
                stats["words_unaligned"] += 1
                continue
            word_phones = phones_in(phones, w_start, w_end)
            if len(word_phones) != len(actual):
                stats["tier_mismatch"] += 1
                continue

            # The following word, only if it actually abuts this one.
            following = None
            if i + 1 < len(entries):
                n_start, _, _, _, n_actual, _ = entries[i + 1]
                if n_start - w_end <= ADJACENCY_TOL and n_actual:
                    following = n_actual
            stats["adjacent" if following else "not_adjacent"] += 1

            for idx, (c, a) in enumerate(zip(canon, actual)):
                if c not in TARGETS:
                    continue
                phone = word_phones[idx]
                if phone.label != a:
                    stats["tier_label_disagreement"] += 1
                    continue
                if phone.mark and not keep_marked:
                    stats["excluded_annotator_mark"] += 1
                    continue

                is_final = idx == len(canon) - 1
                slot = "final" if is_final else "medial"
                by_position[(c, slot)][a] += 1
                if is_final:
                    final_context[(c, a)][classify_following(following)] += 1
                    final_types[c][word][a] += 1
                    speakers_of_type[c][word].add(speaker)
                else:
                    medial_types[c][word][a] += 1

    return by_position, final_context, final_types, medial_types, speakers_of_type, stats


def both_realization_report(types_map, speakers_of_type, underlying, label):
    """Types attested with both [t]/[d] and [dx]; balance is what matters."""
    stop = underlying  # the faithful surface for this underlying segment
    both = {}
    for word, counts in types_map[underlying].items():
        if counts[stop] and counts["dx"]:
            both[word] = counts

    tokens = sum(c[stop] + c["dx"] for c in both.values())
    all_tokens = sum(
        c[stop] + c["dx"] for c in types_map[underlying].values()
    )
    print(f"-- /{underlying}/ {label}: types attested with BOTH [{stop}] and [dx] --")
    print(f"  types            : {len(both):,} of {len(types_map[underlying]):,}")
    print(f"  tokens in them   : {tokens:,} of {all_tokens:,} "
          f"({tokens / all_tokens:.1%})" if all_tokens else "  (none)")
    if not both:
        print()
        return

    # Balance: a type that is 95% one way is still nearly deterministic.
    balanced = {w: c for w, c in both.items()
                if min(c[stop], c["dx"]) / (c[stop] + c["dx"]) >= 0.20}
    bal_tokens = sum(c[stop] + c["dx"] for c in balanced.values())
    print(f"  of those, minority share >= 20%: {len(balanced):,} types, "
          f"{bal_tokens:,} tokens")

    ranked = sorted(both.items(), key=lambda kv: -(kv[1][stop] + kv[1]["dx"]))
    print(f"  top 20 by token count (word: [{stop}] / [dx], speakers):")
    for word, c in ranked[:20]:
        n_spk = len(speakers_of_type[underlying][word])
        share = c["dx"] / (c[stop] + c["dx"])
        print(f"    {word:<14} {c[stop]:>5} / {c['dx']:<5}  "
              f"flap {share:5.1%}   {n_spk:>2} speakers")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--keep-marked", action="store_true")
    args = ap.parse_args()

    (by_position, final_context, final_types,
     medial_types, speakers_of_type, stats) = collect(
        args.root, keep_marked=args.keep_marked)

    print("Word-final vs word-medial realization of /t/ and /d/ in Buckeye")
    print(f"corpus: {args.root}\n")

    print("-- realization by position --")
    for u in TARGETS:
        for slot in ("medial", "final"):
            counts = by_position[(u, slot)]
            total = sum(counts.values())
            if not total:
                continue
            top = "  ".join(f"[{k}] {v:,} ({v/total:.1%})"
                            for k, v in counts.most_common(4))
            print(f"  /{u}/ {slot:<6} n={total:>6,}   {top}")
    print()

    print("-- word-final: does realization track the following segment? --")
    for u in TARGETS:
        for surf in (u, "dx"):
            ctx = final_context[(u, surf)]
            total = sum(ctx.values())
            if not total:
                continue
            print(f"  /{u}/ -> [{surf:<2}] n={total:>6,}   "
                  f"next=vowel {ctx['vowel']/total:6.1%}   "
                  f"next=consonant {ctx['consonant']/total:6.1%}   "
                  f"break {ctx['break']/total:6.1%}")
    print()

    both_realization_report(final_types, speakers_of_type, "t", "word-final")
    both_realization_report(final_types, speakers_of_type, "d", "word-final")

    # The contrast the hypothesis predicts: medial types should be far more
    # deterministic than final ones.
    print("-- control: same measure on word-MEDIAL types --")
    for u in TARGETS:
        both = {w: c for w, c in medial_types[u].items() if c[u] and c["dx"]}
        tokens = sum(c[u] + c["dx"] for c in both.values())
        all_tok = sum(c[u] + c["dx"] for c in medial_types[u].values())
        if all_tok:
            print(f"  /{u}/ medial: {len(both):,} of {len(medial_types[u]):,} types, "
                  f"{tokens:,} of {all_tok:,} tokens ({tokens/all_tok:.1%})")
    print()

    print("-- parsing --")
    for k in ("words_unaligned", "tier_mismatch", "tier_label_disagreement",
              "excluded_annotator_mark", "adjacent", "not_adjacent"):
        print(f"  {k:<28} {stats[k]:>8,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
