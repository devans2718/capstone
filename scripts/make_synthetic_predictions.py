#!/usr/bin/env python3
"""Generate the synthetic prediction cell for the Challenge 2 slice.

WHAT THIS PRODUCES, AND WHAT IT DOES NOT
    Three conditions over 40 deliberately chosen sites, with probabilities drawn
    from a seeded generator. NO PROBE HAS BEEN TRAINED. Every p_t here is noise,
    every row carries synthetic=true, and the manifest records it a fourth time.

    That costs nothing that matters for what the slice demonstrates. The claim
    under test is that a number reaches a screen with its provenance intact and
    a denominator attached — and retrieval is exactly as testable against noise
    as against a real result, because retrieval does not care whether a
    probability means anything.

    The reason it is synthetic is not convenience. The pre-registration at
    experiments/pilot-flapping/README.md finalizes three provisional thresholds
    in a revision that has not been written, and states that running before that
    revision is pushed voids the pre-registration. Training a probe now to make
    a demo look better would cost the thing the demo is for.

WHERE THE OUTPUT GOES
    predictions.parquet lands under /data/embeddings/, never in the repository.
    The manifest is written twice: beside the parquet, and into
    results/flapping/ where it IS committed. The manifest is provenance rather
    than data — a run that leaves no seed, no inventory hash and no commit in
    git produces figures nobody can check afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capstone.predictions import (  # noqa: E402
    CONDITIONS,
    build_manifest,
    synthesize,
    write_predictions,
)

DEFAULT_INVENTORY = Path("results/flapping/sites-20260924.parquet")
DEFAULT_OUT = Path("/data/embeddings/predictions.parquet")
DEFAULT_MANIFEST_COPY = Path("results/flapping/run-manifest.json")

# Recorded in the manifest rather than described in prose, so the split the
# predictions claim to come from is a field a reader can compare against.
SPLIT_SCHEME = ("combined: speaker x word-type 5x5 grid, all 25 cells scored; "
                "clamped sites excluded before scoring in every condition")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"predictions parquet (default {DEFAULT_OUT})")
    ap.add_argument("--manifest-copy", type=Path, default=DEFAULT_MANIFEST_COPY,
                    help="committed copy of the manifest")
    ap.add_argument("--n-sites", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--layer", type=int, default=8)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    if not args.inventory.exists():
        raise SystemExit(
            f"no site inventory at {args.inventory}. Run:\n"
            f"  python scripts/count_flap_sites.py --parquet {args.inventory}"
        )

    inventory = pq.read_table(args.inventory).to_pandas()
    sites, rows = synthesize(inventory, n_sites=args.n_sites, seed=args.seed,
                             layer=args.layer)

    write_predictions(rows, args.out)
    manifest = build_manifest(
        args.inventory, split_scheme=SPLIT_SCHEME, random_seed=args.seed,
        n_sites_scored=len(sites),
    )
    manifest.write(args.out.with_name("run-manifest.json"))
    manifest.write(args.manifest_copy)

    scored = sum(1 for r in rows if r["scored"])
    ineligible = sum(1 for r in rows if not r["eligible"])
    print(f"inventory        {len(inventory):,} sites  ({args.inventory})")
    print(f"selected         {len(sites)} sites, seed {args.seed}")
    print(f"  zero-onset     {int((sites.n_frame_onsets == 0).sum())} excerpt-local / "
          f"{int((sites.n_frame_onsets_abs == 0).sum())} absolute")
    print(f"  clamped        {int(sites.excerpt_clamped.sum())}")
    print(f"  underlying     t={int((sites.underlying == 't').sum())} "
          f"d={int((sites.underlying == 'd').sum())}")
    print(f"  two-flap words {int((sites.n_flaps_in_word >= 2).sum())}")
    print(f"\nrows             {len(rows):,} over {len(CONDITIONS)} conditions")
    print(f"  scored         {scored:,}")
    print(f"  ineligible     {ineligible:,}  (p_t null, by construction)")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.out.with_name('run-manifest.json')}")
    print(f"wrote {args.manifest_copy}  (committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
