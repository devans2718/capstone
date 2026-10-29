#!/usr/bin/env python3
"""The span grid's pooled design, scored: per-span curves, checks, C1 C4 C5 C6 C7.

HOW TO RUN (in .venv-app, after `run_probes.py span-repro` and `span-grid`)
    .venv-app/bin/python scripts/score_span_grid.py

    Reads the cached blocks under /data/embeddings/span-run-<date>/grid/ and
    fits nothing. Everything it computes is the pilot's code (capstone.probes),
    restricted to the grid's 10,334 sites as §8 (2026-10-29) fixes; the one
    new piece is paired_best_gap, checked against criterion_1 on the pilot's
    own blocks before it is used.

WHAT IT SCORES (experiments/span-grid/README.md §5-§7; pooled design only)
    For every span x representation, and every duration baseline:
      - balanced accuracy per layer, with the word-type bootstrap's 95%
        intervals (10,000 resamples, seed 20260928, the pilot's 991-type index)
      - the control task's randomization test, max over layers (the pilot's
        criterion_2, its mapping and its 10,000 null mappings)
      - the lemma-unseen re-score: the pilot's 7,586 sites intersected with the
        10,334, on the pilot's 805-type index
      - §7: the same predictions on the 9,657 unclipped sites, and each cell's
        score at the best layer
    The comparisons, each called by §6's rule: best-layer gap >= 0.05 and
    paired word-type bootstrap 95% lower bound > 0, the best layer re-selected
    on both sides in every resample. C6 is the best layer, with the 2.5-97.5
    percentiles of its index over the resamples, and no threshold. No
    multiple-comparison correction is registered, and none is applied.

    Also the 616 WavJEPA spans that lost a padded frame (§8, 2026-10-18),
    recomputed from spans.tsv's bounds and the pinned geometry, by span and by
    whether the site is clipped on the following side.

WHAT IT WRITES
    results/span-grid/  aggregates only, nothing Buckeye-derived:
        span-grid-results-<date>.csv      representation x span x layer
        span-grid-cells-<date>.csv        each cell at each block's best layer
        span-grid-comparisons-<date>.csv  every called gap, and C6
        span-grid-summary-<date>.txt      the counts, checks and calls, read first
        run-manifest-span-grid.json
    /data/embeddings/span-run-<date>/grid/predictions.parquet, per site.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
from capstone import seeds  # noqa: E402

OUT = rp.REPO / "results" / "span-grid"
DATE = rp.SPAN_DATE

ENCODERS = ("wavjepa", "wav2vec2")
BASELINES = [("duration", k) for k in rp.DURATIONS] + \
            [("mfcc_duration", s) for s in rp.MFCC_DURATION_SPANS]
BLOCKS = [(rep, span) for rep in rp.SPAN_REPRESENTATIONS for span in rp.SPANS] + BASELINES

TRIMMED = 616   # extract-manifest.json, wavjepa_spans_trimmed_to_clean_frames

SPLIT_SCHEME = ("capstone.splits, `joint` only (span grid §5): the pilot's 5x5 grid, folds "
                "assigned over its 10,389 eligible sites keyed by type_key; the 55 §1 "
                "exclusions then leave every cell's train and test sets (§8, 2026-10-29). "
                "Out-of-fold predictions pooled over the 25 cells, 10,334 sites.")


# ------------------------------------------------------------------ statistics


def paired_best_gap(W, type_idx, is_d, correct_a, correct_b):
    """Best layer of a minus best layer of b, with a paired bootstrap interval.

    correct_a, correct_b: (La, n), (Lb, n) bool. One W for both, so the
    resamples are paired; the best layer is re-selected on both sides in every
    resample (§6, the pilot's criterion 1). Returns gap, lo, hi.
    """
    import numpy as np

    from capstone.probes.scoring import (balanced_accuracy_from_correct,
                                         boot_balanced_accuracy, interval)

    ba_a = max(balanced_accuracy_from_correct(is_d, c) for c in correct_a)
    ba_b = max(balanced_accuracy_from_correct(is_d, c) for c in correct_b)
    boot = boot_balanced_accuracy(W, type_idx, is_d, np.vstack([correct_a, correct_b]))
    La = len(correct_a)
    gaps = np.nanmax(boot[:, :La], axis=1) - np.nanmax(boot[:, La:], axis=1)
    lo, hi = interval(gaps)
    return float(ba_a - ba_b), float(lo), float(hi)


def check_against_criterion_1():
    """paired_best_gap must give criterion_1's numbers on the pilot's blocks."""
    import numpy as np

    from capstone.probes import criteria as C
    from capstone.probes.scoring import resample_weights

    d = rp.Data()
    e = d.elig_idx
    y = d.is_d[e]
    W = resample_weights(len(d.types), seeds.BOOTSTRAP)
    w = np.load(rp.RUN_DIR / rp.block_name("probe", "wavjepa", "mean", "joint", False))
    m = np.load(rp.RUN_DIR / rp.block_name("probe", "mfcc", "mean", "joint", False))
    cw, cm = (w["real"][:, e] > 0.5) == y, (m["real"][:, e] > 0.5) == y
    _, n1 = C.criterion_1(W, d.type_idx[e], y, cw, cm[0])
    gap, lo, hi = paired_best_gap(W, d.type_idx[e], y, cw, cm)
    if (gap, lo, hi) != (n1["gap"], n1["gap_ci_lo"], n1["gap_ci_hi"]):
        raise SystemExit(f"paired_best_gap {gap, lo, hi} != criterion_1 "
                         f"{n1['gap'], n1['gap_ci_lo'], n1['gap_ci_hi']} on the pilot")


def call(gap, lo):
    from capstone.probes.criteria import GAP

    return "effect" if gap >= GAP and lo > 0 else "no effect"


# ------------------------------------------------------------------ the 616


def trimmed_breakdown(d):
    """The WavJEPA spans that lost a padded frame, by span x clipped-following.

    Recomputed from spans.tsv's local bounds (written with repr, so exact) and
    the pinned geometry, by the extraction's own rule; checked against the
    stored frame counts and the manifest's total before it is reported.
    """
    from count_flap_sites import WAVJEPA_GEOMETRY as G
    from extract_site_features import CONTEXT_MS

    from capstone.encoders.extraction import excerpt_samples, plan_excerpt

    inv = d.inv
    out = {(span, post): 0 for span in rp.SPANS for post in (False, True)}
    for i, r in zip(d.elig_idx, d.spans_rows):
        ex = plan_excerpt(inv.start[i] * 1000, inv.end[i] * 1000, float("inf"),
                          context_ms=CONTEXT_MS)
        a, b = excerpt_samples(ex, G.sample_rate)
        n_out, n_clean = G.n_output_frames(b - a), G.n_clean_frames(b - a)
        for span in rp.SPANS:
            if not r[f"{span}_start_local_ms"]:
                continue
            rng = G.frames_for_interval(float(r[f"{span}_start_local_ms"]),
                                        float(r[f"{span}_end_local_ms"]))
            start, stop = min(rng.start, n_out), min(rng.stop, n_out)
            trimmed = stop > n_clean
            if trimmed:
                stop = max(start, n_clean)
                out[span, r["clipped_post"] == "1"] += 1
            if stop - start != int(r[f"n_wavjepa_{span}"]):
                raise SystemExit("the trimmed-span recount disagrees with spans.tsv's "
                                 f"n_wavjepa_{span}")
    if sum(out.values()) != TRIMMED:
        raise SystemExit(f"{sum(out.values())} trimmed spans recounted, the manifest says {TRIMMED}")
    if d.spans_manifest["wavjepa_spans_trimmed_to_clean_frames"] != TRIMMED:
        raise SystemExit("the manifest's trimmed count moved")
    return out


# ------------------------------------------------------------------ scoring


def main() -> int:
    import numpy as np
    import pandas as pd

    from capstone.predictions import build_manifest, write_prediction_columns
    from capstone.probes import criteria as C
    from capstone.probes.controls import null_mappings, type_index
    from capstone.probes.scoring import (balanced_accuracy, boot_balanced_accuracy,
                                         interval, resample_weights)

    check_against_criterion_1()
    d = rp.SpanData()
    # Built first: it records the commit, and this script writes into the tree.
    manifest = build_manifest(
        rp.INVENTORY, split_scheme=SPLIT_SCHEME, seeds=seeds.REGISTERED,
        n_sites_scored=int(d.in_grid.sum()), synthetic=False,
        encoders=tuple(f"{r}_{s}" for r, s in BLOCKS), run_id=f"span-grid-{DATE}")

    g = np.flatnonzero(d.in_grid)
    y = d.is_d[g]
    tix = d.type_idx[g]                                   # the pilot's 991-type index
    W = resample_weights(len(d.types), seeds.BOOTSTRAP)
    nulls = null_mappings(len(d.types), d.share, seeds.CONTROL_NULL)
    cy = d.control_is_d[g]

    # The lemma-unseen subset: the pilot's 7,586, its 805-type index, then the grid.
    lem = pd.read_csv(rp.LEMMA_TSV, sep="\t")
    unseen = set(lem.site_id[lem.lemma_in_train.astype(str).str.lower() == "false"])
    e = d.elig_idx
    pilot_sub = np.array([sid in unseen for sid in d.inv.site_id.to_numpy()[e]])
    assert pilot_sub.sum() == 7_586 and int(d.is_d[e][pilot_sub].sum()) == 1_735
    sub_types, sub_tix_pilot = type_index([d.keys[i] for i in e[pilot_sub]])
    W_sub = resample_weights(len(sub_types), seeds.BOOTSTRAP)
    sub_type_of = np.full(d.n, -1)
    sub_type_of[e[pilot_sub]] = sub_tix_pilot
    lsub = sub_type_of[g] >= 0                            # within g
    assert lsub.sum() == 7_546 and int(y[lsub].sum()) == 1_727, "§8's lemma counts moved"
    assert len(set(sub_type_of[g][lsub])) == 797
    unclipped = ~d.clipped[g]
    assert unclipped.sum() == 9_657, "§8's unclipped count moved"
    assert len(set(tix)) == 983 and int(y.sum()) == 2_513

    rows, cell_rows, summary, correct, pred_cols = [], [], {}, {}, []
    for rep, span in BLOCKS:
        z = np.load(rp.GRID_DIR / rp.block_name("probe", rep, span, "joint", False))
        if not ((z["fold"] >= 0) == d.in_grid).all():
            raise SystemExit(f"{rep}_{span}: scored sites are not the grid's")
        real, ctrl = z["real"][:, g], z["control"][:, g]
        if np.isnan(real).any() or np.isnan(ctrl).any():
            raise SystemExit(f"{rep}_{span}: a grid site has no prediction")
        pred = real > 0.5
        cor = pred == y
        correct[rep, span] = cor
        ba = np.array([balanced_accuracy(y, p) for p in pred])
        boot = boot_balanced_accuracy(W, tix, y, cor)
        lo, hi = interval(boot)
        best = int(np.argmax(ba))
        best_idx = np.nanargmax(boot, axis=1)
        b_lo, b_hi = np.percentile(best_idx, [2.5, 97.5])
        _, c2 = C.criterion_2(ctrl > 0.5, tix, d.control_mapping, nulls)
        lem_ba = np.array([balanced_accuracy(y[lsub], p[lsub]) for p in pred])
        lem_boot = boot_balanced_accuracy(W_sub, sub_type_of[g][lsub], y[lsub], cor[:, lsub])
        lem_lo, lem_hi = interval(lem_boot)
        unc_ba = np.array([balanced_accuracy(y[unclipped], p[unclipped]) for p in pred])
        cba = np.array([balanced_accuracy(cy, p) for p in ctrl > 0.5])
        for L in range(len(ba)):
            rows.append({
                "representation": rep, "span": span, "layer": L,
                "n": len(g), "n_d": int(y.sum()),
                "balanced_acc": ba[L], "ci_lo": lo[L], "ci_hi": hi[L],
                "control_balanced_acc": cba[L],
                "lemma_unseen_balanced_acc": lem_ba[L],
                "lemma_unseen_ci_lo": lem_lo[L], "lemma_unseen_ci_hi": lem_hi[L],
                "unclipped_balanced_acc": unc_ba[L],
                "n_nonconverged": int((~z["real_converged"][L]).sum()),
                "n_nonconverged_control": int((~z["control_converged"][L]).sum()),
            })
        fold = z["fold"][g]
        cells = []
        for k in np.unique(fold):
            m = fold == k
            b = balanced_accuracy(y[m], pred[best][m]) if 0 < y[m].sum() < m.sum() else np.nan
            cells.append(b)
            cell_rows.append({"representation": rep, "span": span, "layer": best,
                              "cell": int(k), "n": int(m.sum()), "n_d": int(y[m].sum()),
                              "balanced_acc": b})
        summary[rep, span] = {
            "best_layer": best, "ba": float(ba[best]), "lo": float(lo[best]),
            "hi": float(hi[best]), "best_lo": float(b_lo), "best_hi": float(b_hi),
            "control_max": c2["control_max"], "control_p": c2["p_value"],
            "control_q95": c2["null_q95"],
            "lemma_ba": float(lem_ba[best]), "lemma_best": float(lem_ba.max()),
            "unclipped_ba": float(unc_ba[best]),
            "cells": (float(np.nanmin(cells)), float(np.nanmedian(cells)),
                      float(np.nanmax(cells))),
            "nonconverged": int((~z["real_converged"]).sum()),
            "nonconverged_control": int((~z["control_converged"]).sum()),
            "fits": int(z["real_converged"].size + z["control_converged"].size),
        }
        pred_cols.append((rep, span, "joint", False, z["fold"], z["real"]))

    # ---------------------------------------------------------- comparisons
    comps = []

    def compare(cid, a, b, per):
        gap, lo, hi = paired_best_gap(W, tix, y, correct[a], correct[b])
        comps.append({"comparison": cid, "per": per, "a": "_".join(a), "b": "_".join(b),
                      "gap": gap, "ci_lo": lo, "ci_hi": hi, "call": call(gap, lo)})

    for rep in rp.SPAN_REPRESENTATIONS:
        for span in ("vr", "rv", "vrv"):
            compare("C1", (rep, span), (rep, "r"), rep)
    for enc in ENCODERS:
        for span in ("vr", "vrv"):
            compare("C4", (enc, span), ("duration", "clipped"), enc)
    for enc in ENCODERS:
        for span in rp.SPANS:
            compare("C5", (enc, span), ("mfcc", span), enc)
        for span in rp.MFCC_DURATION_SPANS:
            compare("C5", (enc, span), ("mfcc_duration", span), enc)
    for span in rp.SPANS:
        compare("C7", ("wavjepa", span), ("wav2vec2", span), span)
    for enc in ENCODERS:
        for span in rp.SPANS:
            s = summary[enc, span]
            comps.append({"comparison": "C6", "per": enc, "a": f"{enc}_{span}", "b": "",
                          "gap": float(s["best_layer"]), "ci_lo": s["best_lo"],
                          "ci_hi": s["best_hi"], "call": "best layer; no threshold"})
    assert sum(c["comparison"] != "C6" for c in comps) == 29

    trimmed = trimmed_breakdown(d)

    # ---------------------------------------------------------- write
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"span-grid-results-{DATE}.csv", index=False,
                              float_format="%.6f")
    pd.DataFrame(cell_rows).to_csv(OUT / f"span-grid-cells-{DATE}.csv", index=False,
                                   float_format="%.6f")
    pd.DataFrame(comps).to_csv(OUT / f"span-grid-comparisons-{DATE}.csv", index=False,
                               float_format="%.6f")
    write_prediction_columns(pred_cols, d.inv.site_id.to_numpy(), d.in_grid, d.is_d,
                             rp.GRID_DIR / "predictions.parquet", synthetic=False)
    manifest.write(OUT / "run-manifest-span-grid.json")

    timing = json.loads((rp.GRID_DIR / "timing.json").read_text())
    nc = sum(s["nonconverged"] + s["nonconverged_control"] for s in summary.values())
    fits = sum(s["fits"] for s in summary.values())
    lines = [
        "Span grid, pooled design — per-span scores and C1, C4-C7",
        f"Run {DATE}. `joint`, out-of-fold over 25 cells, {len(g):,} sites "
        f"({int(y.sum()):,} /d/), 983 of the pilot's 991 type keys. Chance is 0.50.",
        "",
        f"code commit       {manifest.code_commit}",
        f"features          {rp.SPANS_DIR} (extracted at {d.spans_manifest['code_commit']})",
        f"inventory sha256  {d.spans_manifest['inventory_sha256']}",
        f"exclusion sha256  {d.spans_manifest['exclusion_sha256']}",
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"fits              {fits:,}, {nc} did not converge (none refit); "
        f"{timing['total_seconds'] / 60:.1f} min",
        f"lemma-unseen      {int(lsub.sum()):,} sites ({int(y[lsub].sum()):,} /d/), "
        "797 of 805 type keys",
        f"unclipped         {int(unclipped.sum()):,} sites",
        "",
        "Per span x representation, at the best layer (95% word-type bootstrap interval)",
        "-------------------------------------------------------------------------------",
        f"{'block':<20}{'best':>5}  {'bal. acc.':>9}  {'95% CI':>17}  {'best-layer CI':>13}  "
        f"{'ctrl p':>6}  {'lemma':>6}  {'unclip':>6}  cells min/med/max",
    ]
    for (rep, span), s in summary.items():
        best_ci = f"[{s['best_lo']:.0f}, {s['best_hi']:.0f}]"
        lines.append(
            f"{rep + '_' + span:<20}{s['best_layer']:>5}  {s['ba']:>9.4f}  "
            f"[{s['lo']:.4f}, {s['hi']:.4f}]  {best_ci:>13}  "
            f"{s['control_p']:>6.4f}  {s['lemma_ba']:>6.4f}  {s['unclipped_ba']:>6.4f}  "
            + "/".join(f"{c:.3f}" for c in s["cells"]))
    lines += ["", "Comparisons (§6: gap >= 0.05 and paired 95% lower bound > 0)",
              "-------------------------------------------------------------"]
    for c in comps:
        if c["comparison"] == "C6":
            continue
        lines.append(f"{c['comparison']}  {c['a']:<16} - {c['b']:<22} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "C6, best layer (2.5-97.5 percentile of its index over the resamples)"]
    for c in comps:
        if c["comparison"] == "C6":
            lines.append(f"    {c['a']:<16} layer {int(c['gap'])}  [{c['ci_lo']:.0f}, {c['ci_hi']:.0f}]")
    lines += ["", "No multiple-comparison correction is registered; 29 gaps are called.",
              "", "WavJEPA spans that lost a padded frame (616), by span and clipped-following:"]
    for span in rp.SPANS:
        lines.append(f"    {span:<4} clipped-following {trimmed[span, True]:>4}   "
                     f"not {trimmed[span, False]:>4}")
    text = "\n".join(lines) + "\n"
    (OUT / f"span-grid-summary-{DATE}.txt").write_text(text)
    print(text)
    print(f"wrote {OUT}/ (results, cells, comparisons, summary, manifest) and "
          f"{rp.GRID_DIR / 'predictions.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
