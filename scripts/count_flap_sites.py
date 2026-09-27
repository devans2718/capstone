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

WHAT THE SITE ROW CARRIES, AND WHY
    Beyond the seven columns of the 2026-08-21 inventory, each site now carries
    a stable identifier, an integer audio anchor, its position inside its word,
    its segmental neighbours, and the frame/excerpt geometry the extractor will
    impose on it. Three reasons, in order of how much they cost to get wrong:

      - site_id is the join key for everything downstream. `start` is a float
        accumulated by summing durations through a text round-trip, so it is
        the one column that must never be keyed on.
      - start_sample is the audio anchor. Integers, because no float should
        participate in a join or in sample arithmetic.
      - phone_idx_in_word settles an open question in the pre-registration
        (experiments/pilot-flapping/README.md §1a): whether the 25 word types
        attested with both underlying classes are words carrying two flap sites
        (dated, hated, dedicated) or types that are genuinely ambiguous. That
        was recorded as an inference because the inventory had no
        position-within-word column. It has one now, and report() answers it.

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
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capstone.encoders.extraction import plan_excerpt  # noqa: E402
from capstone.encoders.wavjepa import FrameGeometry  # noqa: E402

DEFAULT_ROOT = Path("/data/corpora/buckeye")

# Non-speech .words entries use these bracketed forms in the word field.
NONSPEECH_PREFIXES = ("<", "{")

SAMPLE_RATE = 16_000

# The excerpt width every site is cut at, fixed by the extraction findings and
# registered in the pre-registration §2 — NOT a knob this script may vary. The
# positional encoding is re-added from position 0 each segment, so a frame's
# representation depends on where it sits inside its segment; holding context
# constant is what keeps that a constant rather than a label-correlated
# confound. See plan_excerpt's docstring for the measured magnitudes.
CONTEXT_MS = 200.0

# read_words skips silences and noise, so two consecutive yielded entries are
# not necessarily adjacent in time. Same tolerance and same reason as
# scripts/analyze_flap_context.py, which established the word-final result.
ADJACENCY_TOL = 0.02  # seconds; annotation boundaries disagree by a frame or two

# Hard-coded rather than read from the checkpoint so this script stays runnable
# without a GPU, a 785 MB download, or torch. The five values are WavJEPA's and
# are covered by tests/test_frame_geometry_model.py, which is the tripwire if
# upstream moves them.
WAVJEPA_GEOMETRY = FrameGeometry(
    sample_rate=SAMPLE_RATE, target_length=32_159,
    output_steps=200, stride=160, receptive_field=240,
)


@dataclass(frozen=True)
class Phone:
    start: float
    end: float
    label: str
    mark: str
    # Ordinal in the track's .phones tier. Defaulted because phones_in returns
    # Phone objects rather than indices, so this field is the only route by
    # which a caller can recover where a phone sat in the track — and because
    # count_t_realizations.py and analyze_flap_context.py construct nothing but
    # read attributes, which a trailing default leaves untouched.
    index: int = -1


@dataclass(frozen=True)
class Site:
    """One flapping site, with everything a probe or an extractor needs.

    The first seven fields are the 2026-08-21 inventory and the order they are
    written to TSV in. Everything after them is additive and defaulted, so the
    committed TSV stays byte-identical and no existing caller has to change.
    """
    speaker: str
    track: str          # e.g. s0101a — the recording, not the speaker
    start: float
    end: float
    word: str
    underlying: str     # 't' or 'd'
    pos: str

    # -- identity and audio anchor ----------------------------------------
    # Keyed on the phone's tier ordinal, not on `start`: `start` is a float
    # accumulated by summing durations through a text round-trip, and two
    # phones in different tracks can share one. Samples are integers for the
    # same reason — nothing that indexes audio should be a float.
    site_id: str = ""
    start_sample: int = -1
    end_sample: int = -1
    duration_ms: float = 0.0

    # -- position within the word -----------------------------------------
    # n_flaps_in_word is what disambiguates two sites inside one word, and the
    # pair is what settles pre-registration §1a's inference.
    phone_idx_in_word: int = -1
    n_flaps_in_word: int = 0
    word_start_sample: int = -1
    word_end_sample: int = -1

    # Kept as sequences. Joining them into strings destroys the label, which is
    # defined positionally between the two tiers and not recoverable from the
    # concatenation.
    canonical_phones: tuple[str, ...] = ()
    actual_phones: tuple[str, ...] = ()

    # -- context ----------------------------------------------------------
    # Neighbours come from the track-level phone tier, so they cross word
    # boundaries and can be Buckeye's silence and noise labels. That is
    # deliberate: flapping is conditioned by what actually follows, which at a
    # word edge is the next word's first segment or a pause, not a padding
    # token. next_word is empty when the following entry does not abut.
    prev_phone: str = ""
    next_phone: str = ""
    prev_word: str = ""
    prev_word_last_phone: str = ""
    next_word: str = ""
    next_word_first_phone: str = ""

    # Empty for all 10,392 retained sites; populated only under --keep-marked,
    # which admits 754 further sites the annotators hedged.
    annotator_mark: str = ""

    # -- geometry the extractor will impose -------------------------------
    # n_frame_onsets is excerpt-local, which is what encode_excerpt indexes.
    # n_frame_onsets_abs is the recording's own timebase, which is what the
    # 2026-08-21 report published. Both, because they can disagree — see §4.1
    # of docs/challenges/challenge-2-slice-brief.md and report() below.
    n_frame_onsets: int = 0
    n_frame_onsets_abs: int = 0
    excerpt_clamped: bool = False

    # -- neutralisation (decision D5) --------------------------------------
    # True when this site's (actual phone string, position) cell is attested
    # with BOTH underlying classes somewhere in the inventory: got ~ god,
    # write ~ ride. Filled in by mark_surface_ambiguous once every site is known,
    # because it is a property of the whole inventory, not of one site.
    # Ambiguous in Buckeye's transcription, NOT necessarily in the audio: the
    # labels hide the vowel-length difference before /t/ vs /d/, which is
    # plausibly the very cue that separates the pair.
    surface_ambiguous: bool = False

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

    Each Phone carries its own list index. Callers get Phone objects back from
    phones_in with no way to recover where they sat in the track otherwise, and
    that ordinal is what site_id is built from and what prev/next lookups
    index — so it is recorded at construction rather than recomputed later from
    a float timestamp.
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
        phones.append(Phone(prev_end, end, label, mark, len(phones)))
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


def recording_duration_ms(path: Path) -> float:
    """Length of a .wav from its header alone.

    The standard-library `wave` module reads the header without decoding the
    audio, so this adds no dependency and costs nothing at 255 tracks.

    Both failure paths raise rather than degrade. A missing or unexpected-rate
    recording would otherwise cost the sites in that track, and losing sites
    moves the inventory off 10,392 — which invalidates every corpus figure
    cited in the proposal, in a way that a smaller total does not announce.
    """
    if not path.exists():
        raise SystemExit(
            f"no audio for track {path.stem} at {path}. Excerpt planning needs "
            "the recording duration, and skipping the track would silently "
            "change the site count."
        )
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        if rate != SAMPLE_RATE:
            raise SystemExit(
                f"{path.stem} is {rate} Hz, expected {SAMPLE_RATE}. Sample "
                "offsets are computed at a fixed rate, so a mismatch would "
                "produce plausible-looking integers that point at the wrong "
                "audio."
            )
        return w.getnframes() / rate * 1000


def frame_onsets(start: float, end: float, recording_ms: float) -> tuple[int, int, bool]:
    """(excerpt-local onsets, absolute-timebase onsets, clamped) for one site.

    TWO TIMEBASES, AND THEY CAN DISAGREE
        The 2026-08-21 report counted onsets from the site's position in the
        recording's own timebase. What the extraction pipeline indexes is the
        EXCERPT-local span, because encode_excerpt lays the segment grid from
        the excerpt's first sample — so the absolute figure answers a question
        nobody downstream asks.

        They differ systematically, and the reason is arithmetic rather than
        accidental. context_ms = 200 is exactly 3,200 samples, which is exactly
        20 frames at a 160-sample stride, so a centred target's onset always
        lands ON a frame boundary in excerpt-local coordinates. That onset is
        inside [start, end), so the count is ceil(duration / stride) and never
        zero. In the recording's own timebase the start falls at an arbitrary
        phase, and the frame it lands inside began before it and does not count.
        Absolute counts are therefore <= excerpt-local ones.

        (Before 2026-09-24 this was the other way round. frames_for_interval
        dropped the onset at start, so excerpt counts were floor(duration /
        stride) and 39 sites had none. That was D1 in
        docs/challenges/challenge-2-slice-decisions.md, resolved by fixing the
        implementation.)

        Two smaller mechanisms also apply: the staircase leaves a 79-sample
        interval at the end of each 2.01 s segment that no frame reads, and
        sample_to_frame clamps the within-segment index to output_steps - 1, so
        a timestamp landing in that slack maps to the segment's last frame
        rather than to nothing.

        Both are returned. A difference is a result to record, not a
        discrepancy to reconcile away.
    """
    excerpt = plan_excerpt(start * 1000, end * 1000, recording_ms,
                           context_ms=CONTEXT_MS)
    local = WAVJEPA_GEOMETRY.frames_for_interval(
        excerpt.target_start_local_ms, excerpt.target_end_local_ms
    )
    absolute = WAVJEPA_GEOMETRY.frames_for_interval(start * 1000, end * 1000)
    return len(local), len(absolute), excerpt.clamped


def collect_sites(root: Path, keep_marked: bool = False):
    """Walk the corpus and return (sites, stats, underlying_all, per_speaker)."""
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
        track_ms = recording_duration_ms(wf.with_suffix(".wav"))

        # Materialized for one entry of lookahead. read_words keeps its
        # generator signature — the cost of buffering belongs to the caller
        # that needs the lookahead, not to the two scripts that do not.
        entries = list(read_words(wf))

        for i, (w_start, w_end, word, canon, actual, pos) in enumerate(entries):
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

            # The following word, but only if it actually abuts this one.
            # Reporting a non-adjacent word as the following context would be
            # worse than reporting none: it reads as a conditioning environment
            # and is not one.
            next_word, next_word_first_phone = "", ""
            if i + 1 < len(entries):
                n_start, _, n_word, _, n_actual, _ = entries[i + 1]
                if n_start - w_end <= ADJACENCY_TOL:
                    next_word = n_word
                    next_word_first_phone = n_actual[0] if n_actual else ""

            # The preceding word, same rule. Both neighbours are needed to
            # render a fragment of transcript around the site, which is what a
            # human listening to a clip actually needs: 'ladies' alone does not
            # tell you what you are about to hear, and a reviewer asked to judge
            # a label by ear without the surrounding words is guessing.
            # Empty means a pause intervened, and the renderer must show that
            # rather than closing the gap — 826 sites have no following word.
            prev_word, prev_word_last_phone = "", ""
            if i > 0:
                _, p_end, p_word, _, p_actual, _ = entries[i - 1]
                if w_start - p_end <= ADJACENCY_TOL:
                    prev_word = p_word
                    prev_word_last_phone = p_actual[-1] if p_actual else ""

            n_flaps = actual.count("dx")

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

                n_local, n_abs, clamped = frame_onsets(
                    phone.start, phone.end, track_ms
                )
                sites.append(Site(
                    speaker, track, phone.start, phone.end, word, c, pos,
                    site_id=f"{track}:{phone.index:05d}",
                    start_sample=round(phone.start * SAMPLE_RATE),
                    end_sample=round(phone.end * SAMPLE_RATE),
                    duration_ms=(phone.end - phone.start) * 1000,
                    phone_idx_in_word=idx,
                    n_flaps_in_word=n_flaps,
                    word_start_sample=round(w_start * SAMPLE_RATE),
                    word_end_sample=round(w_end * SAMPLE_RATE),
                    canonical_phones=tuple(canon),
                    actual_phones=tuple(actual),
                    prev_phone=(phones[phone.index - 1].label
                                if phone.index > 0 else ""),
                    next_phone=(phones[phone.index + 1].label
                                if phone.index + 1 < len(phones) else ""),
                    prev_word=prev_word,
                    prev_word_last_phone=prev_word_last_phone,
                    next_word=next_word,
                    next_word_first_phone=next_word_first_phone,
                    annotator_mark=phone.mark,
                    n_frame_onsets=n_local,
                    n_frame_onsets_abs=n_abs,
                    excerpt_clamped=clamped,
                ))
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

    if geometry_note and sites:
        # A flap is short: at a 10 ms hop with a 15 ms receptive field, a
        # 30 ms flap has ~3 frame onsets inside it. Reported here because it
        # bounds what per-site pooling can average over, and because it is the
        # multiplier in the frame-cache sizing.
        #
        # No try/except around this any more. It was a reporting convenience
        # when the geometry was computed here; now the figures are acceptance
        # criteria, and a swallowed import error would let the gate pass with
        # the numbers quietly missing.
        print("\n-- frames per site (WavJEPA geometry, 10 ms hop) --")
        print("  Two timebases. 'absolute' is the site's position in the")
        print("  recording, which is what the 2026-08-21 report published.")
        print("  'excerpt' is the span the extractor actually indexes, since")
        print("  encode_excerpt lays the segment grid from the excerpt's first")
        print("  sample. They need not agree; both are reported.")
        local = sorted(s.n_frame_onsets for s in sites)
        absol = sorted(s.n_frame_onsets_abs for s in sites)
        m = len(local)
        print(f"\n  {'':<26}{'excerpt':>10}{'absolute':>12}")
        print(f"  {'onsets per site, min':<26}{local[0]:>10,}{absol[0]:>12,}")
        print(f"  {'median':<26}{local[m//2]:>10,}{absol[m//2]:>12,}")
        print(f"  {'max':<26}{local[-1]:>10,}{absol[-1]:>12,}")
        print(f"  {'total onsets':<26}{sum(local):>10,}{sum(absol):>12,}")
        zero_local = sum(1 for v in local if v == 0)
        zero_abs = sum(1 for v in absol if v == 0)
        print(f"  {'sites with zero onsets':<26}{zero_local:>10,}{zero_abs:>12,}")
        n_diff = sum(1 for s in sites
                     if s.n_frame_onsets != s.n_frame_onsets_abs)
        if n_diff:
            print(f"\n  the two timebases disagree on {n_diff:,} of {m:,} sites "
                  f"({n_diff/m:.2%}).")
            print("  This is arithmetic, not noise. context_ms = 200 is exactly")
            print("  3,200 samples = exactly 20 frames at a 160-sample stride, so")
            print("  a centred target's start lands ON a frame onset, and that")
            print("  onset counts: the excerpt count is ceil(duration / stride),")
            print("  never zero. In the recording's timebase the start falls at")
            print("  an arbitrary phase, which often costs that leading frame.")
            print("  Recorded, not reconciled: the excerpt-local column is the")
            print("  one that describes what a probe will be trained on, and the")
            print("  absolute column under-counts it.")
        else:
            print("\n  the two timebases agree on every site.")

        print(f"\n-- excerpt planning at context_ms = {CONTEXT_MS:.0f} --")
        clamped = sum(1 for s in sites if s.excerpt_clamped)
        print(f"  sites clamped at a recording edge {clamped:>8,}")
        print("  A clamped site cannot be centred, so it carries a different")
        print("  positional contribution from every other site. Counted here")
        print("  so the imbalance is known before results exist, not after.")

    if sites:
        both_class_report(sites)


def both_class_report(sites) -> None:
    """Are both-class word types two-flap words, or genuinely ambiguous?

    Pre-registration §1a recorded this as an inference: the 25 types attested
    with both underlying /t/ and /d/ "appear to be words with two flap sites
    (dated, hated, dedicated) rather than genuinely ambiguous types," and said
    so explicitly because the inventory had no position-within-word column.

    With phone_idx_in_word the question is decidable. For each such type, look
    at what each position within the word is attested as. If every position
    maps to exactly one underlying class, the type's two classes come from two
    distinct sites in the same word and the type is not ambiguous at all. If
    some single position is attested as both /t/ and /d/, it genuinely is.

    The distinction matters for the split design: a positionally determined
    type is as lexically predictable as any other, so it offers a
    word-identity lookup no resistance. Only genuinely ambiguous types do.
    """
    by_type = defaultdict(Counter)              # word -> {underlying: n}
    positions = defaultdict(lambda: defaultdict(set))  # word -> idx -> classes
    for s in sites:
        by_type[s.word][s.underlying] += 1
        positions[s.word][s.phone_idx_in_word].add(s.underlying)

    both = {w: c for w, c in by_type.items() if c["t"] and c["d"]}
    n_sites_in_both = sum(sum(c.values()) for c in both.values())

    print("\n-- word types carrying BOTH underlying classes "
          "(pre-registration §1a) --")
    print(f"  types with both /t/ and /d/{len(both):>8,} of {len(by_type):,}")
    print(f"  sites in them              {n_sites_in_both:>8,}  "
          f"({n_sites_in_both/len(sites):.2%} of all sites)")
    if not both:
        return

    ambiguous = {w: {i: cs for i, cs in positions[w].items() if len(cs) > 1}
                 for w in both}
    ambiguous = {w: d for w, d in ambiguous.items() if d}
    determined = {w: c for w, c in both.items() if w not in ambiguous}

    print(f"  of those types:")
    print(f"    positionally determined  {len(determined):>8,}  "
          "(each position is one class; two flap sites, not one ambiguous one)")
    print(f"    genuinely ambiguous      {len(ambiguous):>8,}  "
          "(some single position attested as both)")

    ranked = sorted(determined.items(), key=lambda kv: -sum(kv[1].values()))
    if ranked:
        shown = ", ".join(w for w, _ in ranked[:12])
        print(f"    positionally determined, by site count: {shown}")
    if ambiguous:
        for w in sorted(ambiguous, key=lambda w: -sum(both[w].values()))[:12]:
            idxs = ", ".join(
                f"position {i} = {'/'.join(sorted(cs))}"
                for i, cs in sorted(ambiguous[w].items())
            )
            print(f"    {w}: {idxs}")


def mark_surface_ambiguous(sites) -> list:
    """Return the sites with surface_ambiguous set where the surface cell carries both classes.

    A cell is (actual phone string, phone_idx_in_word). This is D5's
    across-spelling case, distinct from §1a's within-spelling one: §1a asks
    whether one written word carries both classes at one position (it never
    does), this asks whether one transcribed pronunciation does (59 cells,
    3,173 sites, 30.53%). A column rather than a subset, so a named
    evaluation subset stays possible later without recomputing, and without
    deciding the research question now.
    """
    classes = defaultdict(set)
    for s in sites:
        classes[(s.actual_phones, s.phone_idx_in_word)].add(s.underlying)
    # Site is frozen, so this builds new records rather than mutating.
    return [replace(s, surface_ambiguous=len(classes[(s.actual_phones,
                                                      s.phone_idx_in_word)]) > 1)
            for s in sites]


def write_parquet(sites, path: Path) -> None:
    """Write the inventory as parquet with an explicitly declared schema.

    Declared, not inferred. "No float participates in a join" is only
    enforceable if site_id is a string and every *_sample column is an
    integer, and an inferred schema will happily widen an int64 column to a
    double on one null without saying it did.

    pyarrow is imported here rather than at module scope on purpose: this
    module is imported by count_t_realizations.py and analyze_flap_context.py,
    neither of which writes parquet, and neither should acquire a dependency
    to keep running.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("site_id", pa.string()),
        ("speaker", pa.string()),
        ("track", pa.string()),
        ("start", pa.float64()),
        ("end", pa.float64()),
        ("word", pa.string()),
        ("underlying", pa.string()),
        ("pos", pa.string()),
        ("start_sample", pa.int64()),
        ("end_sample", pa.int64()),
        ("duration_ms", pa.float64()),
        ("phone_idx_in_word", pa.int32()),
        ("n_flaps_in_word", pa.int32()),
        ("word_start_sample", pa.int64()),
        ("word_end_sample", pa.int64()),
        ("canonical_phones", pa.list_(pa.string())),
        ("actual_phones", pa.list_(pa.string())),
        ("prev_phone", pa.string()),
        ("next_phone", pa.string()),
        ("prev_word", pa.string()),
        ("prev_word_last_phone", pa.string()),
        ("next_word", pa.string()),
        ("next_word_first_phone", pa.string()),
        ("annotator_mark", pa.string()),
        ("n_frame_onsets", pa.int32()),
        ("n_frame_onsets_abs", pa.int32()),
        ("excerpt_clamped", pa.bool_()),
        ("surface_ambiguous", pa.bool_()),
    ])

    rows = [asdict(s) for s in sites]
    seen = {r["site_id"] for r in rows}
    if len(seen) != len(rows):
        raise SystemExit(
            f"site_id is not unique: {len(rows):,} rows, {len(seen):,} distinct. "
            "It is the join key for every downstream table; a duplicate would "
            "fan out silently on the join rather than fail."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def write_stats(sites, stats, path: Path) -> None:
    """Write the label-and-population exclusion counts as JSON.

    WHY THIS EXISTS SEPARATELY FROM THE PARQUET
        The application has to show what the retained set excludes, and three of
        those figures are not columns and never can be: 2,986 words dropped for
        canonical/actual length mismatch, 754 flaps carrying the annotator's
        hedge mark, 198 flaps from neither /t/ nor /d/. They are counts of
        things that did NOT become rows, so the only place they exist is the
        parse that rejected them.

        Without this file the application would have to carry those numbers as
        literals in its source, transcribed by hand from a text report. A
        transcribed number does not change when the measurement does, which is
        the failure mode where an interface confidently displays last month's
        figure. Writing them out at the moment they are measured is the whole
        point.

    The zero-onset and clamped counts are recomputed here from the sites rather
    than read from anywhere, and both timebases are reported for the same reason
    report() gives both.
    """
    import json

    by_cat = Counter(s.underlying for s in sites)
    payload = {
        "flap_bearing_words": stats["flap_words"],
        "excluded_length_mismatch": stats["flap_words_unaligned"],
        "excluded_tier_mismatch": stats["tier_mismatch"],
        "excluded_not_t_or_d": stats["flap_not_from_t_or_d"],
        "excluded_annotator_mark": stats["excluded_annotator_mark"],
        "n_sites": len(sites),
        "n_underlying_t": by_cat["t"],
        "n_underlying_d": by_cat["d"],
        "n_zero_onset_excerpt_local": sum(1 for s in sites if s.n_frame_onsets == 0),
        "n_zero_onset_absolute": sum(1 for s in sites if s.n_frame_onsets_abs == 0),
        "n_surface_ambiguous": sum(1 for s in sites if s.surface_ambiguous),
        "n_excerpt_clamped": sum(1 for s in sites if s.excerpt_clamped),
        "context_ms": 200,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Buckeye corpus root (default {DEFAULT_ROOT})")
    ap.add_argument("--keep-marked", action="store_true",
                    help="include flaps carrying the '*' annotator mark")
    ap.add_argument("--dump", type=Path,
                    help="write the site inventory to this TSV "
                         "(the seven 2026-08-21 columns, unchanged)")
    ap.add_argument("--parquet", type=Path,
                    help="write the full site inventory to this parquet file")
    args = ap.parse_args()

    sites, stats, underlying_all, per_speaker = collect_sites(
        args.root, keep_marked=args.keep_marked
    )
    sites = mark_surface_ambiguous(sites)
    report(sites, stats, underlying_all, per_speaker, root=args.root)

    if args.dump:
        # These seven columns in this order, at this float precision, are the
        # published 2026-08-21 inventory. New fields go to parquet; this writer
        # does not move, so the committed TSV stays byte-identical and the
        # figures cited against it stay checkable.
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as fh:
            fh.write("speaker\ttrack\tstart\tend\tword\tunderlying\tpos\n")
            for s in sites:
                fh.write(f"{s.speaker}\t{s.track}\t{s.start:.6f}\t{s.end:.6f}\t"
                         f"{s.word}\t{s.underlying}\t{s.pos}\n")
        print(f"\nwrote {len(sites):,} sites to {args.dump}")

    if args.parquet:
        write_parquet(sites, args.parquet)
        print(f"wrote {len(sites):,} sites to {args.parquet}")
        stats_path = args.parquet.with_suffix(".stats.json")
        write_stats(sites, stats, stats_path)
        print(f"wrote exclusion counts to {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
