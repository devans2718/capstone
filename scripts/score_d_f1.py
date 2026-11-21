#!/usr/bin/env python3
"""Minority-class (/d/) F1 and layer spread for the span grid, after the fact.

HOW TO RUN (in .venv-app, after score_span_grid.py)
    .venv-app/bin/python scripts/score_d_f1.py

    NOT REGISTERED. Computed 2026-11-21, after the span grid's results were
    read, from its cached out-of-fold blocks; nothing is refit. The pilot's
    criterion 4 (/d/ F1 at the best layer >= 0.45; trivial floors 0.391 for
    always /d/, 0.327 for a coin flip) is quoted for reference, not applied:
    the span grid registered no F1 criterion.

    Per span x representation: /d/ F1 at every layer and at the span grid's
    best layer, and the layer spread (best minus worst balanced accuracy),
    the pilot's criterion 3 statistic.

WHAT IT WRITES (aggregates only)
    results/span-grid/d-f1-<date>.csv       representation x span x layer
    results/span-grid/d-f1-summary-<date>.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402

OUT = rp.REPO / "results" / "span-grid"
DATE = "20261121"


def main() -> int:
    import numpy as np
    import pandas as pd

    from capstone.probes.scoring import balanced_accuracy, f1_d

    d = rp.SpanData()
    g = np.flatnonzero(d.in_grid)
    y = d.is_d[g]
    rows, lines = [], [
        "Span grid, /d/ F1 and layer spread — after the fact, not registered",
        f"Run {DATE}, from the span grid's run {rp.SPAN_DATE}; {len(g):,} sites "
        f"({int(y.sum()):,} /d/, {y.mean():.1%}).",
        "Reference only (pilot criterion 4, not a span-grid criterion): F1 >= 0.45; "
        "always /d/ gives 0.391 on the pilot's set.",
        f"Always /d/ on these sites: F1 {f1_d(y, np.ones_like(y)):.3f}.",
        "",
        f"{'block':<14}{'best':>5}  {'BA':>6}  {'F1 /d/':>6}  {'max F1 (layer)':>14}  "
        f"{'BA spread':>9}",
    ]
    for rep in rp.SPAN_REPRESENTATIONS:
        for span in rp.SPANS:
            z = np.load(rp.GRID_DIR / rp.block_name("probe", rep, span, "joint", False))
            pred = z["real"][:, g] > 0.5
            ba = np.array([balanced_accuracy(y, p) for p in pred])
            f1 = np.array([f1_d(y, p) for p in pred])
            best = int(np.argmax(ba))
            for L in range(len(pred)):
                rows.append({"representation": rep, "span": span, "layer": L,
                             "best_layer": L == best, "balanced_acc": ba[L], "f1_d": f1[L]})
            spread = f"{ba.max() - ba.min():.4f}" if len(ba) > 1 else "—"
            lines.append(f"{rep + '_' + span:<14}{best:>5}  {ba[best]:.4f}  {f1[best]:.4f}  "
                         f"{f1.max():.4f} ({int(np.argmax(f1)):>2})    {spread:>9}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"d-f1-{DATE}.csv", index=False, float_format="%.6f")
    text = "\n".join(lines) + "\n"
    (OUT / f"d-f1-summary-{DATE}.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
