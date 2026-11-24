#!/usr/bin/env python3
"""The transfer probe's training stops, exported for extraction (span-grid §8, 2026-11-21).

HOW TO RUN (in .venv-app; the corpus and the span grid's extraction mounted)
    .venv-app/bin/python scripts/export_transfer_stops.py

    Loads the span grid's sites through run_probes.SpanData, as
    count_transfer_candidates.py does, and no prediction or representation.

WHAT IT DOES
    Applies the registered definition, through count_transfer_candidates' own
    functions rather than a copy of them, at the LOWEST registered threshold
    (0.25, a sensitivity set). The positions kept at a higher share are a subset
    of those kept at a lower one, so the 0.25 set holds the registered 0.40 set
    and the 0.50 sensitivity set, and one extraction serves all three. Each row
    says which sets it belongs to.

    Then it locates each stop's §2 spans with the stop in the flap's place, by
    measure_span_grid's rule: the preceding stretch walks back through sonorant
    codas to a vowel (V), and the following segment is the next phone. A stop
    whose walk does not end on V gets no V+stop or V+stop+V span. The class
    context guarantees the phone just before it is a vowel, coda or syllabic
    consonant, but not that a coda run ends on a vowel. Their count is printed.

    The excerpt is the stop ± 200 ms, planned by plan_excerpt against the
    recording's length read from its header, as the flap inventory was. A
    clamped stop is kept in the file and flagged; the registration does not
    exclude any, so the count is printed for the scoring step to deal with.

WHAT IT CHECKS, AND STOPS ON
    The three sets' stop, /d/ and type counts against the registration's
    table. A mismatch means the definition or the corpus is not the
    registered one.

WHAT IT WRITES
    /data/span-grid/transfer-stops.tsv: one row per stop, sorted by site_id
    (track:ordinal in the .phones tier, the flap inventory's form). Times in
    seconds, written with repr() so plan_excerpt sees the exact float. The
    header records the grid inventory's sha256 and the sha256 of the sorted
    site IDs at each threshold.

    Buckeye-derived, so /data only, never the repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capstone.encoders.extraction import plan_excerpt  # noqa: E402
from count_flap_sites import DEFAULT_ROOT, read_phones, recording_duration_ms  # noqa: E402
from count_transfer_candidates import (MIN_FLAP_SHARE, kept_positions,  # noqa: E402
                                       load_context, stops_at)
from measure_span_grid import CODA, CONTEXT_MS, V  # noqa: E402

OUT = Path("/data/span-grid/transfer-stops.tsv")

# The registration's table (span-grid README §8, 2026-11-21, f22cad8):
# threshold -> (stops, /d/, word types).
REGISTERED = {
    0.25: (2_433, 1_162, 303),
    0.40: (2_247, 1_011, 252),
    0.50: (1_749, 746, 220),
}
assert MIN_FLAP_SHARE in REGISTERED
LOWEST = min(REGISTERED)

COLUMNS = ("site_id", "speaker", "track", "word", "phone_idx_in_word", "label",
           "start", "end", "position_flap_share", "in_025", "in_040", "in_050",
           "prev_label", "vowel_start", "following_label", "following_end",
           "clipped_pre", "clipped_post", "excerpt_clamped")


def digest(ids) -> str:
    """sha256 of the sorted IDs, one per line, newline-terminated (as exclusion_digest)."""
    return hashlib.sha256("".join(f"{s}\n" for s in sorted(ids)).encode()).hexdigest()


def main() -> int:
    _, attested, ctx, share, n_flap = load_context()

    sets = {thr: stops_at(ctx, kept_positions(share, n_flap, attested, thr))
            for thr in REGISTERED}
    for thr, stops in sets.items():
        got = (len(stops), sum(t[3] == "d" for t in stops), len({t[1] for t in stops}))
        if got != REGISTERED[thr]:
            raise SystemExit(f"at {thr:.2f}: {got} (stops, /d/, types), registered "
                             f"{REGISTERED[thr]}")
    key = lambda t: (t[8], t[9])  # noqa: E731  track, ordinal: unique per token
    members = {thr: {key(t) for t in stops} for thr, stops in sets.items()}
    for lo in REGISTERED:
        for hi in REGISTERED:
            if lo < hi and not members[hi] <= members[lo]:
                raise SystemExit(f"the {hi:.2f} set is not inside the {lo:.2f} set")

    paths = {p.stem: p for p in DEFAULT_ROOT.rglob("*.phones")}
    tiers, rec_ms = {}, {}
    ctx_s = CONTEXT_MS / 1000
    rows = []
    for t in sets[LOWEST]:
        speaker, word, idx, canon, _, prev, nxt, _, track, k = t
        if track not in tiers:
            tiers[track] = read_phones(paths[track])
            rec_ms[track] = recording_duration_ms(paths[track].with_suffix(".wav"))
        tier = tiers[track]
        ph, fol = tier[k], tier[k + 1]
        assert ph.label == canon and prev == tier[k - 1].label and nxt == fol.label

        j = k - 1
        while j >= 0 and tier[j].label in CODA:
            j -= 1
        vowel = tier[j] if j >= 0 and tier[j].label in V else None

        ex = plan_excerpt(ph.start * 1000, ph.end * 1000, rec_ms[track],
                          context_ms=CONTEXT_MS)
        rows.append({
            "site_id": f"{track}:{k:05d}", "speaker": speaker, "track": track,
            "word": word, "phone_idx_in_word": idx, "label": canon,
            "start": repr(float(ph.start)), "end": repr(float(ph.end)),
            "position_flap_share": repr(share[word, idx]),
            **{f"in_{int(round(thr * 100)):03d}": int(key(t) in members[thr])
               for thr in REGISTERED},
            "prev_label": prev,
            "vowel_start": repr(float(vowel.start)) if vowel else "",
            "following_label": fol.label, "following_end": repr(float(fol.end)),
            # §2's comparison, in seconds, as measure_span_grid.clipped makes it.
            "clipped_pre": int(vowel is not None and (ph.start - ctx_s) > vowel.start),
            "clipped_post": int(fol.end > (ph.end + ctx_s)),
            "excerpt_clamped": int(ex.clamped),
        })
    rows.sort(key=lambda r: r["site_id"])
    if len({r["site_id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate site IDs")

    # The grid inventory the definition was read against.
    from extract_site_features import INVENTORY, sha256
    digests = {f"{thr:.2f}": digest(r["site_id"] for r in rows
                                    if r[f"in_{int(round(thr * 100)):03d}"])
               for thr in REGISTERED}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        fh.write(f"# inventory_sha256={sha256(INVENTORY)}\n")
        fh.write(f"# site_id_sha256={json.dumps(digests, sort_keys=True)}\n")
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT}: {len(rows):,} stops (the {LOWEST:.2f} set)")
    for thr in REGISTERED:
        col = f"in_{int(round(thr * 100)):03d}"
        sub = [r for r in rows if r[col]]
        print(f"  {thr:.2f}{' (registered)' if thr == MIN_FLAP_SHARE else '':<13} "
              f"{len(sub):>6,} stops  "
              f"no V before {sum(not r['vowel_start'] for r in sub):>4}  "
              f"clipped pre {sum(r['clipped_pre'] for r in sub):>4}  "
              f"post {sum(r['clipped_post'] for r in sub):>4}  "
              f"clamped {sum(r['excerpt_clamped'] for r in sub):>3}  "
              f"sha256 {digests[f'{thr:.2f}'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
