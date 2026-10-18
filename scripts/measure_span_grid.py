#!/usr/bin/env python3
"""The span grid's sites and spans, measured from the corpus before any run.

Read-only with respect to the repository: no probe, features or audio are
involved. The site, span and clipping figures in experiments/span-grid/README.md
§1–§3 come from here.

WHY THIS EXISTS
    The span grid (experiments/span-grid/README.md) excludes 55 sites whose
    environment is not the one under study, and defines four spans per site from
    the .phones tier. Site IDs are Buckeye-derived and stay out of this
    repository, so the exclusion cannot be committed as a list. What is
    committed instead is this script and the sha256 of the sorted list it
    produces. A registered reader regenerates the list; the script stops if the
    result is not the registered one.

METHOD
    Sites: the inventory's eligible sites (excerpt not clamped), each located in
    its track's .phones tier by the ordinal in its site_id and checked against
    the inventory's start time and the label `dx`.

    Preceding stretch: walk back from the flap through sonorant codas (CODA) to
    a vowel (V). V is Buckeye's vowel labels, their nasalized forms, `er`, and
    the syllabic consonants. A site is EXCLUDED if the walk does not end on a
    vowel: a preceding flap, an obstruent, or a label outside V such as `ae+1`.
    Following segment: the next phone, whatever it is.

    Spans are clipped to the excerpt, flap ± CONTEXT_MS. A site is CLIPPED on a
    side if its stretch starts, or its following segment ends, outside it.

RUNNING IT
    .venv-app/bin/python scripts/measure_span_grid.py [--write PATH]
    --write saves the excluded site IDs and reasons as a TSV, outside the
    repository (default /data/span-grid/excluded-sites.tsv).
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402

from count_flap_sites import DEFAULT_ROOT, read_phones  # noqa: E402

INVENTORY = (Path(__file__).resolve().parent.parent
             / "results/flapping/sites-20260924.parquet")
CONTEXT_MS = 200.0   # the pilot's excerpt: flap ± 200 ms (pilot §2)

BASE_V = {"iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
          "ow", "uh", "uw", "er"}
# Syllabic consonants count as V (decided 2026-10-17): little's `el` follows an
# intervocalic flap.
V = BASE_V | {v + "n" for v in BASE_V} | {"el", "em", "en", "eng"}
# Sonorant codas the stretch runs back through (decided 2026-10-17): hardy
# lengthens like heeded.
CODA = {"r", "l", "n", "m", "ng", "nx"}

# sha256 of the sorted excluded site IDs, one per line, newline-terminated.
# Registered in experiments/span-grid/README.md §1.
EXPECTED_SHA256 = "964e7ab99882340560b6da10fffb7a54f6efb92e519b8247f95f0602b480db21"


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--write", type=Path, nargs="?",
                    const=Path("/data/span-grid/excluded-sites.tsv"))
    args = ap.parse_args()

    rows = [r for r in pq.read_table(INVENTORY).to_pylist()
            if not r["excerpt_clamped"]]
    paths = {p.stem: p for p in args.root.rglob("*.phones")}
    tiers: dict[str, list] = {}

    excluded: list[tuple[str, str]] = []
    kept = []   # (row, vowel, codas, following)
    for r in rows:
        tier = tiers.setdefault(r["track"], read_phones(paths[r["track"]]))
        i = int(r["site_id"].split(":")[1])
        flap = tier[i]
        if flap.label != "dx" or abs(flap.start - r["start"]) > 1e-4:
            raise SystemExit(f"{r['site_id']}: tier phone {i} is {flap.label} at "
                             f"{flap.start}, inventory says dx at {r['start']}")
        j, codas = i - 1, []
        while j >= 0 and tier[j].label in CODA:
            codas.append(tier[j].label)
            j -= 1
        if j < 0 or tier[j].label not in V:
            reason = tier[j].label if j >= 0 else "track start"
            excluded.append((r["site_id"], f"preceded by {reason}"
                             + (f" before coda {' '.join(reversed(codas))}" if codas else "")))
            continue
        kept.append((r, tier[j], codas, tier[i + 1]))

    ids = sorted(s for s, _ in excluded)
    digest = hashlib.sha256("".join(f"{s}\n" for s in ids).encode()).hexdigest()

    print(f"eligible sites            {len(rows):>7,}")
    print(f"excluded                  {len(excluded):>7,}")
    for reason, n in Counter(r for _, r in excluded).most_common():
        print(f"    {reason:<34} {n:>4}")
    print(f"span-grid sites           {len(kept):>7,}")
    print(f"exclusion list sha256     {digest}")

    print("\ncoda before the flap:")
    for c, n in Counter(" ".join(reversed(c)) or "(none)"
                        for _, _, c, _ in kept).most_common():
        print(f"    {c:<10} {n:>6,}")

    ctx = CONTEXT_MS / 1000
    pre = [(r["start"] - ctx) > v.start for r, v, _, _ in kept]
    post = [f.end > (r["end"] + ctx) for r, _, _, f in kept]
    over_pre = [(r["start"] - ctx - v.start) * 1000 for (r, v, _, _), c in zip(kept, pre) if c]
    over_post = [(f.end - r["end"] - ctx) * 1000 for (r, _, _, f), c in zip(kept, post) if c]
    either = sum(a or b for a, b in zip(pre, post))
    print(f"\nclipped, preceding side   {sum(pre):>7,}  overshoot median {median(over_pre):.1f} ms")
    print(f"clipped, following side   {sum(post):>7,}  overshoot median {median(over_post):.1f} ms")
    print(f"clipped, either side      {either:>7,}  ({100 * either / len(kept):.2f}%)")
    print(f"preceding stretch median  {median([(r['start'] - v.start) * 1000 for r, v, _, _ in kept]):.1f} ms")
    print(f"flap median               {median([r['duration_ms'] for r in rows]):.1f} ms")
    L = max(r["duration_ms"] for r in rows) + 2 * CONTEXT_MS
    print(f"L, longest excerpt        {L:.1f} ms")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("site_id\treason\n"
                              + "".join(f"{s}\t{r}\n" for s, r in sorted(excluded)))
        print(f"\nwrote {args.write}")

    if EXPECTED_SHA256 is None:
        print("\nEXPECTED_SHA256 is not set: nothing to check against.")
        return 0
    if digest != EXPECTED_SHA256:
        print(f"\nexclusion list differs from the registered one ({EXPECTED_SHA256})")
        return 1
    print("\nexclusion list matches the registered one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
