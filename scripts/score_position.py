#!/usr/bin/env python3
"""The position re-score (span grid §8, 2026-11-21): word-medial vs word-final flaps.

HOW TO RUN (in .venv-app, after score_span_grid.py)
    .venv-app/bin/python scripts/score_position.py

    Reads the span grid's cached out-of-fold blocks under
    /data/embeddings/span-run-<date>/grid/ and fits nothing. Registered in
    experiments/span-grid/README.md §8 (2026-11-21, `b9c6d59`) before any
    position split was read.

WHAT IT SCORES
    Groups, from the inventory's phone_idx_in_word against the citation form:
      word-final    the flap is the citation form's last phone
      word-medial   neither first nor last
      word-initial  the first phone
    Every block is read at its span-grid best layer, fixed in REGISTERED_LAYER
    and checked against the block's own argmax; position never picks a layer.

    Registered calls, §6's rule made two-sided (|gap| >= 0.05 and the paired
    word-type bootstrap 95% interval excludes zero; pilot index, seed 20260928):
      P1  wav2vec2-base, ɾ, layer 9: BA(word-medial) - BA(word-final)
      P2  P1's gap minus MFCC's ɾ gap
    Reported, not called: P1 and P2 for WavJEPA and the other spans; each
    group's BA with every word type weighted equally; /t/ and /d/ recall and
    type counts per group; the word-initial group; per-layer curves per group.

WHAT IT WRITES (aggregates only, nothing Buckeye-derived)
    results/span-grid/position-<date>.csv              block x layer x group
    results/span-grid/position-comparisons-<date>.csv  P1 and P2 per block
    results/span-grid/position-summary-<date>.txt      read first
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
from capstone import seeds  # noqa: E402

OUT = rp.REPO / "results" / "span-grid"
DATE = "20261121"
REGISTRATION = "b9c6d59"

# §8 (2026-11-21): each block's best layer in span-grid-summary-20261029.txt.
REGISTERED_LAYER = {
    ("wav2vec2", "r"): 9, ("wav2vec2", "vr"): 9, ("wav2vec2", "rv"): 9, ("wav2vec2", "vrv"): 9,
    ("wavjepa", "r"): 3, ("wavjepa", "vr"): 4, ("wavjepa", "rv"): 4, ("wavjepa", "vrv"): 6,
    ("mfcc", "r"): 0, ("mfcc", "vr"): 0, ("mfcc", "rv"): 0, ("mfcc", "vrv"): 0,
}
GROUPS = ("word-medial", "word-final", "word-initial")
GAP = 0.05


def boot_weighted_ba(W, type_idx, is_d, correct, site_w):
    """(B, R) balanced accuracy per resample, each site also weighted by site_w.

    scoring.boot_balanced_accuracy with a per-site weight folded into the
    per-type counts; site_w = 1 gives that function's numbers exactly (checked
    in main). Used for the type-weighted score, where site_w is one over the
    site's type's count in its group, so each type counts once.
    """
    import numpy as np

    is_d = np.asarray(is_d, bool)
    correct = np.atleast_2d(np.asarray(correct, bool))
    T = W.shape[1]

    def per_type(mask):
        m = np.atleast_2d(mask).astype(np.float64) * site_w
        return np.stack([np.bincount(type_idx, weights=r, minlength=T) for r in m])

    n_d, n_t = per_type(is_d)[0], per_type(~is_d)[0]
    c_d, c_t = per_type(correct & is_d), per_type(correct & ~is_d)
    W64 = W.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec_d = (W64 @ c_d.T) / (W64 @ n_d)[:, None]
        rec_t = (W64 @ c_t.T) / (W64 @ n_t)[:, None]
    return 0.5 * (rec_d + rec_t)


def weighted_ba(is_d, correct, w):
    """Point balanced accuracy with per-site weights."""
    return 0.5 * ((w * correct)[is_d].sum() / w[is_d].sum()
                  + (w * correct)[~is_d].sum() / w[~is_d].sum())


def call(gap, lo, hi):
    return "effect" if abs(gap) >= GAP and (lo > 0 or hi < 0) else "no effect"


def main() -> int:
    import numpy as np
    import pandas as pd

    from capstone.probes.scoring import (balanced_accuracy, boot_balanced_accuracy,
                                         interval, resample_weights)

    d = rp.SpanData()
    inv = d.inv
    g = np.flatnonzero(d.in_grid)
    y = d.is_d[g]
    tix = d.type_idx[g]
    keys = np.array([d.keys[i] for i in g])
    W = resample_weights(len(d.types), seeds.BOOTSTRAP)

    idx = inv.phone_idx_in_word.to_numpy()[g]
    ncanon = np.array([len(c) for c in inv.canonical_phones])[g]
    group = np.where(idx == ncanon - 1, "word-final", np.where(idx == 0, "word-initial", "word-medial"))
    masks = {k: group == k for k in GROUPS}
    # The §8 counts, taken before registration from the inventory alone.
    expect = {"word-final": (6428, 1462, 387), "word-medial": (2987, 867, 610),
              "word-initial": (919, 184, 31)}
    for k, (n, nd, nt) in expect.items():
        m = masks[k]
        got = (int(m.sum()), int(y[m].sum()), len(set(keys[m])))
        if got != (n, nd, nt):
            raise SystemExit(f"{k}: {got}, §8 registered {(n, nd, nt)}")

    # Type weights within each group: one over the type's site count there.
    tw = np.zeros(len(g))
    for k, m in masks.items():
        u, inv_k, cnt = np.unique(keys[m], return_inverse=True, return_counts=True)
        tw[m] = 1.0 / cnt[inv_k]

    rows, boots, boots_tw = [], {}, {}
    for (rep, span), L in REGISTERED_LAYER.items():
        z = np.load(rp.GRID_DIR / rp.block_name("probe", rep, span, "joint", False))
        if not ((z["fold"] >= 0) == d.in_grid).all():
            raise SystemExit(f"{rep}_{span}: scored sites are not the grid's")
        pred = z["real"][:, g] > 0.5
        cor = pred == y
        ba_all = np.array([balanced_accuracy(y, p) for p in pred])
        if int(np.argmax(ba_all)) != L:
            raise SystemExit(f"{rep}_{span}: best layer {int(np.argmax(ba_all))}, §8 fixed {L}")
        for k, m in masks.items():
            b = boot_balanced_accuracy(W, tix[m], y[m], cor[:, m])          # (B, layers)
            if k == "word-medial" and rep == "wav2vec2" and span == "r":
                same = boot_weighted_ba(W, tix[m], y[m], cor[:, m], np.ones(m.sum()))
                if not np.allclose(same, b, equal_nan=True):
                    raise SystemExit("boot_weighted_ba(site_w=1) disagrees with scoring")
            btw = boot_weighted_ba(W, tix[m], y[m], cor[L:L + 1, m], tw[m])[:, 0]
            boots[rep, span, k] = b[:, L]
            boots_tw[rep, span, k] = btw
            lo, hi = interval(b)
            tw_lo, tw_hi = interval(btw)
            n_types = len(set(keys[m]))
            n_types_d = len(set(keys[m & y]))
            for layer in range(len(pred)):
                p = pred[layer][m]
                rows.append({
                    "representation": rep, "span": span, "group": k, "layer": layer,
                    "registered_layer": layer == L,
                    "n": int(m.sum()), "n_d": int(y[m].sum()),
                    "n_types": n_types, "n_types_with_d": n_types_d,
                    "balanced_acc": balanced_accuracy(y[m], p),
                    "ci_lo": lo[layer], "ci_hi": hi[layer],
                    "recall_d": float(p[y[m]].mean()), "recall_t": float((~p[~y[m]]).mean()),
                    "type_weighted_balanced_acc": (weighted_ba(y[m], cor[layer][m], tw[m])
                                                   if layer == L else np.nan),
                    "type_weighted_ci_lo": tw_lo if layer == L else np.nan,
                    "type_weighted_ci_hi": tw_hi if layer == L else np.nan,
                })

    res = pd.DataFrame(rows)

    def point(rep, span, k, col="balanced_acc"):
        r = res[(res.representation == rep) & (res.span == span) & (res.group == k)
                & res.registered_layer]
        return float(r[col].iloc[0])

    comps = []
    for rep in ("wav2vec2", "wavjepa"):
        for span in rp.SPANS:
            registered = rep == "wav2vec2" and span == "r"
            for weighting, B, col in (("site", boots, "balanced_acc"),
                                      ("type", boots_tw, "type_weighted_balanced_acc")):
                p1 = point(rep, span, "word-medial", col) - point(rep, span, "word-final", col)
                m1 = point("mfcc", span, "word-medial", col) - point("mfcc", span, "word-final", col)
                d1 = B[rep, span, "word-medial"] - B[rep, span, "word-final"]
                dm = B["mfcc", span, "word-medial"] - B["mfcc", span, "word-final"]
                for cid, gap, samples in (("P1", p1, d1), ("P2", p1 - m1, d1 - dm)):
                    lo, hi = interval(samples)
                    comps.append({
                        "comparison": cid, "representation": rep, "span": span,
                        "weighting": weighting, "layer": REGISTERED_LAYER[rep, span],
                        "gap": gap, "ci_lo": float(lo), "ci_hi": float(hi),
                        "call": (call(gap, lo, hi) if registered and weighting == "site"
                                 else "reported, not called"),
                    })
    comp = pd.DataFrame(comps)

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"position-{DATE}.csv", index=False, float_format="%.6f")
    comp.to_csv(OUT / f"position-comparisons-{DATE}.csv", index=False, float_format="%.6f")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rp.REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    lines = [
        "Span grid, position re-score (§8, 2026-11-21) — word-medial vs word-final",
        f"Run {DATE}, registered at {REGISTRATION}; run on top of {commit[:7]}, "
        "this script committed with these results.",
        f"`joint` out-of-fold predictions from the span grid's run {rp.SPAN_DATE}; nothing refit.",
        f"Bootstrap: 10,000 resamples of the pilot's 991-type index, seed {seeds.BOOTSTRAP}.",
        "",
        "Groups (sites, /d/, word types, types with a /d/ site)",
    ]
    for k in GROUPS:
        r = res[(res.group == k)].iloc[0]
        lines.append(f"    {k:<13} {r.n:>6,}  {r.n_d:>5,}  {r.n_types:>4}  {r.n_types_with_d:>4}")
    lines += ["", "At the registered layer: balanced accuracy [95% CI]; recall /t/, /d/; "
              "type-weighted BA [95% CI]"]
    for (rep, span), L in REGISTERED_LAYER.items():
        lines.append(f"  {rep}_{span} (layer {L})")
        for k in GROUPS:
            r = res[(res.representation == rep) & (res.span == span) & (res.group == k)
                    & res.registered_layer].iloc[0]
            lines.append(
                f"    {k:<13} {r.balanced_acc:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]  "
                f"t {r.recall_t:.3f} d {r.recall_d:.3f}  "
                f"tw {r.type_weighted_balanced_acc:.4f} "
                f"[{r.type_weighted_ci_lo:.4f}, {r.type_weighted_ci_hi:.4f}]")
    lines += ["", "Comparisons (word-medial minus word-final; P2 less MFCC's gap). "
              "Called: P1 and P2 for wav2vec2_r, site-weighted, only"]
    for c in comps:
        lines.append(f"  {c['comparison']}  {c['representation'] + '_' + c['span']:<13} "
                     f"{c['weighting']:<5} {c['gap']:+.4f}  [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
                     f"{c['call']}")
    text = "\n".join(lines) + "\n"
    (OUT / f"position-summary-{DATE}.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
