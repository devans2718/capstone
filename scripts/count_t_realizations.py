#!/usr/bin/env python3
"""How does underlying /t/ surface in Buckeye — [t], [dx], or [tq]?

WHY THIS EXISTS
    The flapping inventory (count_flap_sites.py) asks the reverse question:
    given a flap, which underlying segment produced it. That framing sees only
    one of /t/'s surface realizations.

    Buckeye's phone tier distinguishes at least three: plain 't', the flap 'dx',
    and 'tq' (glottalized/glottal-stop t). One underlying segment, three
    surfaces, each environmentally conditioned — and English orthography writes
    all three <t>. That is the "information present in speech but not on the
    text side" the advisor asked the study to cover (2026-09-07), measurable
    from annotation that is already aligned and already on disk.

    This script sizes it: how many tokens of each realization, spread across how
    many speakers. Read-only; it decides whether the condition is worth
    registering, not whether it succeeds.

WHAT COUNTS AS A TOKEN
    A position in a word's canonical string holding 't' (or 'd', reported
    alongside for symmetry), where:
      - canonical and actual strings have equal length, so position maps to an
        underlying segment. Buckeye ships no alignment between the tiers, so
        unequal lengths are unsafe and are reported, not dropped silently.
      - the phones inside the word's interval correspond one-to-one with the
        actual string, i.e. the two tiers agree about this word.
      - the phone carries no '*' annotator mark. Buckeye uses it for
        transcription uncertainty; a hedged label is not a clean one.

    These are exactly count_flap_sites.py's rules, reused rather than restated —
    this script imports its parsing so the two inventories cannot drift.

CONSISTENCY CHECK THAT CAN ACTUALLY FAIL
    Selecting canonical 't' and asking how it surfaces must yield the same
    t-to-flap count as selecting actual 'dx' and asking what underlies it:
    7,863 (results/corpus-verification/, 2026-08-21). The parsing helpers are
    shared, but the selection logic here is the other direction, so a mismatch
    is informative. It is asserted loudly rather than printed quietly.

    Noted because the 2026-08-03 corpus-figure error came from a check that
    could not fail. This one can.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_flap_sites import (  # noqa: E402
    DEFAULT_ROOT,
    phones_in,
    read_phones,
    read_words,
)

# The flap count this script must reproduce from the other direction.
EXPECTED_T_FLAPS = 7863

TARGETS = ("t", "d")


def collect(root: Path, keep_marked: bool = False):
    """Walk the corpus; return per-underlying realization counts and detail."""
    # realizations[underlying][surface_label] = count
    realizations: dict[str, Counter] = {u: Counter() for u in TARGETS}
    # per_speaker[underlying][speaker][surface_label] = count
    per_speaker: dict[str, dict[str, Counter]] = {
        u: defaultdict(Counter) for u in TARGETS
    }
    durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    rows: list[tuple] = []
    stats = Counter()

    word_files = sorted(root.rglob("*.words"))
    if not word_files:
        raise SystemExit(f"no .words files under {root} — is the corpus mounted?")

    for wf in word_files:
        pf = wf.with_suffix(".phones")
        if not pf.exists():
            stats["tracks_missing_phones"] += 1
            continue
        stats["tracks"] += 1

        speaker = wf.parent.name
        track = wf.stem
        phones = read_phones(pf)

        for w_start, w_end, word, canon, actual, pos in read_words(wf):
            stats["words"] += 1
            if not any(t in canon for t in TARGETS):
                continue
            stats["words_with_target"] += 1

            if len(canon) != len(actual):
                stats["words_unaligned"] += 1
                continue

            word_phones = phones_in(phones, w_start, w_end)
            if len(word_phones) != len(actual):
                stats["tier_mismatch"] += 1
                continue
            stats["words_usable"] += 1

            for idx, (c, a) in enumerate(zip(canon, actual)):
                if c not in TARGETS:
                    continue
                phone = word_phones[idx]
                # The .words 'actual' string and the phone tier should agree on
                # the label. Where they do not, this token is not trustworthy.
                if phone.label != a:
                    stats["tier_label_disagreement"] += 1
                    continue
                if phone.mark and not keep_marked:
                    stats["excluded_annotator_mark"] += 1
                    continue

                realizations[c][a] += 1
                per_speaker[c][speaker][a] += 1
                durations[(c, a)].append(phone.end - phone.start)
                rows.append(
                    (speaker, track, f"{phone.start:.6f}", f"{phone.end:.6f}",
                     word, c, a, pos)
                )

    return realizations, per_speaker, durations, rows, stats


def report(realizations, per_speaker, durations, stats, root: Path,
           keep_marked: bool = False) -> bool:
    """Print the realization report; return whether the flap cross-check agrees.

    The return value exists so main() can exit non-zero. EXPECTED_T_FLAPS is
    the one number in this file that ties it to the other inventory, and a
    cross-check that only prints cannot be depended on by a hook, a CI step, or
    an acceptance gate.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Buckeye /t/ and /d/ surface realizations — {stamp}")
    print(f"corpus: {root}")
    print()

    print("-- parsing --")
    print(f"  tracks read                      {stats['tracks']:>8,}")
    print(f"  words seen                       {stats['words']:>8,}")
    print(f"  words containing /t/ or /d/      {stats['words_with_target']:>8,}")
    print(f"  dropped: canon/actual length     {stats['words_unaligned']:>8,}")
    print(f"  dropped: tier mismatch           {stats['tier_mismatch']:>8,}")
    print(f"  words usable                     {stats['words_usable']:>8,}")
    print(f"  tokens dropped: tier label       {stats['tier_label_disagreement']:>8,}")
    print(f"  tokens dropped: annotator mark   {stats['excluded_annotator_mark']:>8,}")
    print()

    for u in TARGETS:
        counts = realizations[u]
        total = sum(counts.values())
        print(f"-- underlying /{u}/ : {total:,} tokens --")
        for label, n in counts.most_common():
            durs = durations[(u, label)]
            med_ms = statistics.median(durs) * 1000 if durs else float("nan")
            speakers = sum(1 for s in per_speaker[u] if per_speaker[u][s][label])
            print(f"  [{label:<12}] {n:>7,}  {n / total:6.1%}   "
                  f"median {med_ms:5.1f} ms   {speakers:>2}/40 speakers")
        print()

    # Speaker coverage for the three-way /t/ contrast: the analogue of "all 40
    # speakers carry both classes", which is what left the flapping split
    # unconstrained.
    print("-- speaker coverage for the /t/ three-way contrast --")
    three = ("t", "dx", "tq")
    counts_per_speaker = []
    complete = 0
    for spk, c in sorted(per_speaker["t"].items()):
        present = [lbl for lbl in three if c[lbl]]
        if len(present) == 3:
            complete += 1
        counts_per_speaker.append(min(c[lbl] for lbl in three))
    print(f"  speakers carrying all of t/dx/tq: {complete}/{len(per_speaker['t'])}")
    if counts_per_speaker:
        print(f"  per-speaker minimum cell: min {min(counts_per_speaker)}, "
              f"median {statistics.median(counts_per_speaker):.0f}, "
              f"max {max(counts_per_speaker)}")
    print()

    flaps_from_t = realizations["t"]["dx"]
    print("-- consistency check against the 2026-08-21 flap inventory --")
    print(f"  /t/ -> [dx] here:            {flaps_from_t:,}")
    if keep_marked:
        # The expected figure is the DEFAULT selection's. --keep-marked admits
        # the 754 hedged flaps, so the two numbers are supposed to differ and
        # comparing them would report a failure that is not one.
        print(f"  expected (count_flap_sites): n/a under --keep-marked "
              f"({EXPECTED_T_FLAPS:,} is the default selection's figure)")
        return True
    print(f"  expected (count_flap_sites): {EXPECTED_T_FLAPS:,}")
    if flaps_from_t != EXPECTED_T_FLAPS:
        print("  ** MISMATCH — the two inventories disagree. Do not cite either")
        print("     figure until this is resolved; one of the two selections is")
        print("     wrong, and a plausible number is the failure mode here.")
        return False
    print("  OK — the reverse selection reproduces it exactly.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Buckeye corpus root (default {DEFAULT_ROOT})")
    ap.add_argument("--keep-marked", action="store_true",
                    help="include phones carrying the '*' annotator mark")
    ap.add_argument("--dump", type=Path, help="write the token inventory to this TSV")
    args = ap.parse_args()

    realizations, per_speaker, durations, rows, stats = collect(
        args.root, keep_marked=args.keep_marked
    )
    agrees = report(realizations, per_speaker, durations, stats, root=args.root,
                    keep_marked=args.keep_marked)

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as fh:
            fh.write("speaker\ttrack\tstart\tend\tword\tunderlying\tsurface\tpos\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(f"\nwrote {len(rows):,} tokens to {args.dump}")

    # Non-zero on disagreement. The printed warning above is the diagnosis; the
    # exit code is what lets anything else depend on it — a pre-commit hook, a
    # CI step, or the acceptance gate for the Challenge 2 slice, none of which
    # can read prose. A cross-check that cannot fail a caller is a comment.
    return 0 if agrees else 1


if __name__ == "__main__":
    sys.exit(main())
