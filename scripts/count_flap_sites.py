#!/usr/bin/env python3
"""Count usable English flapping sites in Buckeye — the advisor's step 2.

WHY THIS EXISTS
    RQ1's English arm asks whether the encoder retains the UNDERLYING category
    (/t/ vs /d/) of a segment that surfaces as a flap [dx]. That probe is only
    worth building if both underlying classes are populated well enough to
    survive a speaker-disjoint split. This script answers "how many, and spread
    across how many speakers" before any GPU time is spent.

    It also sizes the frame-cache decision: sites x frames-per-site is what
    converts "extract embeddings" into a number of gigabytes.

WHAT COUNTS AS A SITE
    A phone labelled 'dx' in the .phones tier, inside a word whose .words entry
    gives a canonical and an actual phone string, where the position of the flap
    in the actual string maps to a canonical 't' or 'd'.

    Two exclusions, both deliberate:
      - annotator mark '*' on the flap. Buckeye uses it for transcription
        uncertainty, and a site the annotators hedged is not a clean label.
      - words whose canonical and actual strings differ in length. Positional
        alignment is unsafe there and Buckeye ships no alignment between the
        tiers. ~21% of flap-bearing words, reported separately as a recoverable
        ceiling rather than silently dropped.

PARSING GOTCHA THAT ALREADY BIT ONCE
    .phones data lines carry an optional annotator mark after a semicolon, and
    the semicolon is not reliably whitespace-separated:

        '   2.059495  122 ih ; *'   -> split()[2] == 'ih'
        '   0.931090  122 ey; *'    -> split()[2] == 'ey;'   <- missed

    Counting with split()[2] undercounts every label by ~7%. Two independent
    passes made this mistake and agreed with each other. Always partition on ';'
    first; that is what label_and_mark does.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path("/data/corpora/buckeye")

# Non-speech .words entries use these bracketed forms in the word field.
NONSPEECH_PREFIXES = ("<", "{")


@dataclass(frozen=True)
class Phone:
    start: float
    end: float
    label: str
    mark: str


@dataclass(frozen=True)
class Site:
    """One flapping site, with everything a probe or an extractor needs."""
    speaker: str
    track: str          # e.g. s0101a — the recording, not the speaker
    start: float
    end: float
    word: str
    underlying: str     # 't' or 'd'
    pos: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def label_and_mark(line: str) -> tuple[str | None, str]:
    """Split a .phones data line into (label, annotator_mark).

    Partitioning on ';' before splitting on whitespace is the whole point — see
    the module docstring. Returns (None, '') for lines that are not phone data.
    """
    body, sep, mark = line.partition(";")
    parts = body.split()
    if len(parts) < 3:
        return None, ""
    return parts[2], (mark.strip() if sep else "")


def read_phones(path: Path) -> list[Phone]:
    """Parse a .phones file into phones with explicit start times.

    The xlabel format stores only the END time of each segment, so a phone's
    start is its predecessor's end. The header ends at a lone '#'.
    """
    lines = path.read_text(errors="replace").splitlines()
    try:
        start_idx = lines.index("#") + 1
    except ValueError:
        return []

    phones: list[Phone] = []
    prev_end = 0.0
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        label, mark = label_and_mark(line)
        if label is None:
            continue
        try:
            end = float(line.split()[0])
        except (ValueError, IndexError):
            continue
        phones.append(Phone(prev_end, end, label, mark))
        prev_end = end
    return phones


def read_words(path: Path):
    """Yield (start, end, word, canonical, actual, pos) for real word entries.

    .words format after the '#' header:
        end_time  colour  word; canonical; actual; POS
    Non-speech entries carry S/U/B codes rather than phones and are skipped.
    """
    lines = path.read_text(errors="replace").splitlines()
    try:
        start_idx = lines.index("#") + 1
    except ValueError:
        return

    prev_end = 0.0
    for line in lines[start_idx:]:
        if ";" not in line:
            continue
        head, _, rest = line.partition(";")
        parts = head.split()
        if len(parts) < 3:
            continue
        try:
            end = float(parts[0])
        except ValueError:
            continue
        word = parts[2]
        fields = [f.strip() for f in rest.split(";")]
        if len(fields) < 3:
            prev_end = end
            continue
        if word.startswith(NONSPEECH_PREFIXES):
            prev_end = end
            continue
        yield prev_end, end, word, fields[0].split(), fields[1].split(), fields[2]
        prev_end = end


def phones_in(phones: list[Phone], start: float, end: float) -> list[Phone]:
    """Phones whose midpoint falls inside [start, end).

    Midpoint rather than overlap: word and phone boundaries are annotated
    independently and can disagree by a frame or two at the edges, which would
    make an overlap test claim a neighbouring word's phone.
    """
    out = []
    for p in phones:
        mid = (p.start + p.end) / 2
        if start <= mid < end:
            out.append(p)
        elif mid >= end:
            break
    return out


def collect_sites(root: Path, keep_marked: bool = False):
    """Walk the corpus and return (sites, stats)."""
    sites: list[Site] = []
    stats = Counter()
    underlying_all = Counter()
    per_speaker = defaultdict(Counter)

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
            if "dx" not in actual:
                continue
            stats["flap_words"] += 1

            if len(canon) != len(actual):
                stats["flap_words_unaligned"] += 1
                continue
            stats["flap_words_aligned"] += 1

            # The .words 'actual' string should correspond one-to-one with the
            # phones inside the word's interval. When it does not, the two tiers
            # disagree about this word and neither position nor timing can be
            # trusted for it.
            word_phones = phones_in(phones, w_start, w_end)
            if len(word_phones) != len(actual):
                stats["tier_mismatch"] += 1
                continue

            for idx, (c, a) in enumerate(zip(canon, actual)):
                if a != "dx":
                    continue
                underlying_all[c] += 1
                if c not in ("t", "d"):
                    stats["flap_not_from_t_or_d"] += 1
                    continue
                phone = word_phones[idx]
                if phone.label != "dx":
                    stats["tier_label_disagreement"] += 1
                    continue
                if phone.mark and not keep_marked:
                    stats["excluded_annotator_mark"] += 1
                    continue
                sites.append(
                    Site(speaker, track, phone.start, phone.end, word, c, pos)
                )
                per_speaker[speaker][c] += 1

    return sites, stats, underlying_all, per_speaker


def report(sites, stats, underlying_all, per_speaker, root=DEFAULT_ROOT,
           geometry_note=True) -> None:
    print("=" * 66)
    print("Buckeye flapping sites — RQ1 English arm")
    print("=" * 66)
    # Header mirrors results/corpus-verification/common-voice-*.txt: a report
    # that does not say which tree and when it was measured is not citable.
    print(f"  root: {root}")
    print(f"  date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")

    print("\n-- corpus --")
    print(f"  tracks parsed              {stats['tracks']:>8,}")
    print(f"  word tokens                {stats['words']:>8,}")

    print("\n-- flap-bearing words --")
    fw = stats["flap_words"]
    print(f"  words containing [dx]      {fw:>8,}")
    print(f"    canonical/actual aligned {stats['flap_words_aligned']:>8,}"
          f"  ({stats['flap_words_aligned']/fw:.1%})")
    print(f"    length mismatch, skipped {stats['flap_words_unaligned']:>8,}"
          f"  ({stats['flap_words_unaligned']/fw:.1%})")

    print("\n-- exclusions within aligned words --")
    print(f"  .words/.phones tier mismatch     {stats['tier_mismatch']:>8,}")
    print(f"  flap label disagreement          {stats['tier_label_disagreement']:>8,}")
    print(f"  flap not from /t/ or /d/         {stats['flap_not_from_t_or_d']:>8,}")
    print(f"  annotator mark '*' on the flap   {stats['excluded_annotator_mark']:>8,}")

    print("\n-- underlying category of every aligned flap --")
    for phone, count in underlying_all.most_common(8):
        print(f"  {phone:<6} {count:>7,}")

    print("\n== USABLE SITES ==")
    by_cat = Counter(s.underlying for s in sites)
    total = len(sites)
    print(f"  total                      {total:>8,}")
    for cat in ("t", "d"):
        n = by_cat[cat]
        print(f"  underlying /{cat}/            {n:>8,}  ({n/total:.1%})")
    minority = min(by_cat["t"], by_cat["d"])
    print(f"  balanced-subsample ceiling {2*minority:>8,}  "
          f"({minority:,} per class)")

    print("\n-- speaker coverage (a speaker-disjoint split needs both classes) --")
    speakers = sorted(per_speaker)
    both = [s for s in speakers if per_speaker[s]["t"] and per_speaker[s]["d"]]
    print(f"  speakers with any site     {len(speakers):>8,}")
    print(f"  speakers with both classes {len(both):>8,}")
    counts = sorted(sum(per_speaker[s].values()) for s in speakers)
    if counts:
        print(f"  sites per speaker          min {counts[0]:,} / "
              f"median {counts[len(counts)//2]:,} / max {counts[-1]:,}")

    print("\n-- duration --")
    durs = sorted(s.duration for s in sites)
    if durs:
        n = len(durs)
        print(f"  flap duration ms           min {durs[0]*1000:.1f} / "
              f"median {durs[n//2]*1000:.1f} / max {durs[-1]*1000:.1f}")
        print(f"  total flap audio           {sum(durs):.1f} s")

    if geometry_note:
        # A flap is short: at a 10 ms hop with a 15 ms receptive field, a
        # 30 ms flap has ~3 frame onsets inside it. Reported here because it
        # bounds what per-site pooling can average over, and because it is the
        # multiplier in the frame-cache sizing.
        print("\n-- frames per site (WavJEPA geometry, 10 ms hop) --")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from capstone.encoders.wavjepa import FrameGeometry

            g = FrameGeometry(
                sample_rate=16_000, target_length=32_159,
                output_steps=200, stride=160, receptive_field=240,
            )
            per_site = [len(g.frames_for_interval(s.start * 1000, s.end * 1000))
                        for s in sites]
            per_site.sort()
            m = len(per_site)
            print(f"  onsets inside the flap     min {per_site[0]} / "
                  f"median {per_site[m//2]} / max {per_site[-1]}")
            print(f"  total frames over all sites{sum(per_site):>8,}")
        except Exception as exc:  # pragma: no cover - reporting convenience
            print(f"  (skipped: {exc})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Buckeye corpus root (default {DEFAULT_ROOT})")
    ap.add_argument("--keep-marked", action="store_true",
                    help="include flaps carrying the '*' annotator mark")
    ap.add_argument("--dump", type=Path,
                    help="write the site inventory to this TSV")
    args = ap.parse_args()

    sites, stats, underlying_all, per_speaker = collect_sites(
        args.root, keep_marked=args.keep_marked
    )
    report(sites, stats, underlying_all, per_speaker, root=args.root)

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as fh:
            fh.write("speaker\ttrack\tstart\tend\tword\tunderlying\tpos\n")
            for s in sites:
                fh.write(f"{s.speaker}\t{s.track}\t{s.start:.6f}\t{s.end:.6f}\t"
                         f"{s.word}\t{s.underlying}\t{s.pos}\n")
        print(f"\nwrote {len(sites):,} sites to {args.dump}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
