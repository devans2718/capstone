"""Draw the listening audit's sample and write it where the app reads it.

    .venv-app/bin/python scripts/draw_audit_sample.py

Writes results/flapping/audit-sample-20260926.tsv: review order, site_id,
underlying. 40 /t/ + 40 /d/ eligible sites, seed capstone.seeds.AUDIT. Site
identifiers only, no audio, so it is committed. The draw sees the inventory
and nothing else; see capstone.audit for why.

Refuses to overwrite an existing sample. Redrawing after listening would let
what was heard choose the sample, so a new draw needs a new file and a §7 entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402

from capstone.audit import draw_sample  # noqa: E402

INVENTORY = ROOT / "results" / "flapping" / "sites-20260924.parquet"
OUT = ROOT / "results" / "flapping" / "audit-sample-20260926.tsv"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the sample is drawn once. See the docstring.")
    sample = draw_sample(pq.read_table(INVENTORY).to_pandas())
    sample.to_csv(OUT, sep="\t", index=False)
    counts = sample.underlying.value_counts().to_dict()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(sample)} sites, {counts}")


if __name__ == "__main__":
    main()
