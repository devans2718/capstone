"""Summarize the listening audit into a committed report.

    .venv-app/bin/python scripts/summarize_audit.py

Reads /data/reviews/reviewed.jsonl and results/flapping/audit-sample-20260926.tsv
and writes results/flapping/audit-<today>.txt. Counting is capstone.audit's,
as registered in pre-registration §7 (2026-09-26): last verdict wins; only
audit-view lines for pilot-20260926 on sampled sites; per class; yes / (yes +
no) with a Wilson 95% interval; can't tell beside the rate, never in it.

The report holds counts and site identifiers only, no audio, so it can be
committed. An unfinished audit is reported as unfinished: every row says how
many of its 40 sites were reviewed.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from capstone import seeds  # noqa: E402
from capstone.audit import AUDIT_TARGETS, counted_verdicts, summarize  # noqa: E402

RUN_ID = "pilot-20260926"
REVIEWS = Path("/data/reviews/reviewed.jsonl")
SAMPLE = ROOT / "results" / "flapping" / "audit-sample-20260926.tsv"


def main() -> None:
    lines = [json.loads(x) for x in REVIEWS.read_text().splitlines() if x.strip()]
    sample = pd.read_csv(SAMPLE, sep="\t")
    table = summarize(lines, sample, RUN_ID)
    got = counted_verdicts(lines, sample, RUN_ID)
    done = sum(all((s, t) in got for t in AUDIT_TARGETS) for s in sample.site_id)
    not_counted = sum(1 for x in lines if x.get("run_id") == RUN_ID) - sum(
        1 for x in lines if x.get("run_id") == RUN_ID and x.get("audit") is True
        and x.get("site_id") in set(sample.site_id) and x.get("target") in AUDIT_TARGETS)

    def fmt(r: dict) -> str:
        scored = r["yes"] + r["no"]
        rate = "  —  " if scored == 0 else f"{r['rate']:.3f}"
        ci = "" if scored == 0 else f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        return (f"  {r['question']:<5} {r['class']:<4} reviewed "
                f"{r['reviewed']:>2}/{r['of']:<2}  yes {r['yes']:>2}  no {r['no']:>2}  "
                f"can't tell {r['cant_tell']:>2}  rate {rate}{ci}")

    out = [
        "Listening audit of the flap sites — pre-registration §7 (2026-09-26)",
        f"Run {RUN_ID}. Sample {SAMPLE.relative_to(ROOT)} "
        f"({len(sample)} sites, seed {seeds.AUDIT}). Written {date.today().isoformat()}.",
        "One listener (the author, untrained), blind to the probe. It measures the "
        "inventory, not the model.",
        "",
        f"Sites fully reviewed: {done} of {len(sample)}"
        + ("" if done == len(sample) else "  — AUDIT NOT FINISHED"),
        "",
        "Per question and class: rate = yes / (yes + no), Wilson 95% interval. "
        "Can't tell is not in the rate.",
    ]
    out += [fmt(r) for r in table.to_dict("records")]
    out += ["",
            "No pooled rate: the sample is 40/40 and the inventory 75.7/24.3.",
            f"Lines for {RUN_ID} not counted (browsing, or outside the audit): "
            f"{not_counted}."]
    dest = ROOT / "results" / "flapping" / f"audit-{date.today():%Y%m%d}.txt"
    dest.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
