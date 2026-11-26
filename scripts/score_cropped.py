#!/usr/bin/env python3
"""The span grid's cropped design, scored: C2, C3, and C4 C5 C6 C7's cropped halves,
with the padding-only control and the GroupNorm measure (§3, §6, §7; §8 of
2026-10-18, 2026-10-29 and 2026-11-26).

HOW TO RUN (in .venv-app, after scripts/extract_cropped.py and
`run_probes.py span-grid-cropped`)
    .venv-app/bin/python scripts/score_cropped.py

    Reads the cached blocks in /data/embeddings/span-run-20261029/grid/ and fits
    nothing. Each block is scored by score_span_grid.score_block: per-layer
    balanced accuracy with the pilot's word-type bootstrap, the control task,
    the lemma-unseen re-score, unclipped sites, per-cell spread.

    Before anything new is read, every pooled ɾ block is re-scored and checked
    against its committed results at six decimals (score_context's tripwire).

WHAT IT CALLS (§6's rule: best-layer gap >= 0.05 and paired word-type
bootstrap 95% lower bound > 0, best layer re-selected in every resample;
one-sided, §8 2026-10-29), for WavJEPA and wav2vec2-base, and for MFCC where
§6 asks for "each representation" (C2, as C1 calls it):
    C2  cropped design: crop_Vɾ - crop_ɾ, crop_ɾV - crop_ɾ, crop_VɾV - crop_ɾ
    C3  each span, pooled - cropped
    C4  crop_Vɾ and crop_VɾV - the clipped-duration baseline
    C5  each crop - MFCC's same crop; crop_Vɾ and crop_VɾV - MFCC + duration
        on the same crop
    C7  each span, WavJEPA's crop - wav2vec2-base's crop
    C6  each encoder's best layer per crop, with its resampling interval; no
        threshold

WHAT IT REPORTS, NOT CALLED
    C2, C3, C4, C5 and C6 for HuBERT and WavLM (§8, 2026-11-26), beside their
    pooled-design best layers. §7's padding-only control: each span and ctx of
    the padded full excerpt against the same span of the pooled design, for
    every representation. The GroupNorm measure (median and 95th percentile
    over sites of the per-site medians over channels) per input and encoder,
    and WavJEPA's segment normalization alongside. Every layer's curve, the
    checks, unclipped and per-cell, for every new block.

WHAT IT WRITES
    results/span-grid/  aggregates only, nothing Buckeye-derived:
        cropped-results-<date>.csv      representation x pool x layer
        cropped-cells-<date>.csv        each cell at each block's best layer
        cropped-comparisons-<date>.csv  every gap, called or reported, and C6
        cropped-groupnorm-<date>.csv    the GroupNorm and segment measures
        cropped-summary-<date>.txt      read first
        run-manifest-cropped.json
    /data/embeddings/span-run-20261029/grid/predictions-cropped.parquet, per site.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
import score_context as sc  # noqa: E402
import score_span_grid as sg  # noqa: E402
from capstone import seeds  # noqa: E402

DATE = datetime.date.today().strftime("%Y%m%d")
SPANS = rp.SPANS
CALLED_ENCODERS = ("wavjepa", "wav2vec2")
REPORTED_ENCODERS = ("hubert", "wavlm")
ENCODERS = (*CALLED_ENCODERS, *REPORTED_ENCODERS)
REPS = (*ENCODERS, "mfcc")
# mask_ctx is score_masked.py's.
NEW_BLOCKS = [(rep, pool) for rep in rp.CROPPED_REPRESENTATIONS for pool in rp.CROPPED_POOLS
              if pool != "mask_ctx"] \
    + [("mfcc_duration", f"crop_{s}") for s in rp.MFCC_DURATION_SPANS]
POOLED_BLOCKS = [(rep, span) for rep in REPS for span in (*SPANS, "ctx")] \
    + [("duration", "clipped")] + [("mfcc_duration", s) for s in rp.MFCC_DURATION_SPANS]
# 9 C2 + 8 C3 + 4 C4 + 12 C5 + 4 C7
N_CALLED = 37
INPUTS = ("crop_r", "crop_vr", "crop_rv", "crop_vrv", "pad", "mask")


def score_blocks(ctx, blocks, summary, correct, rows, cell_rows, pred_cols, keep_new):
    for rep, pool in blocks:
        if (rep, pool) in summary:
            continue
        r, c, s, cor, col = sg.score_block(ctx, rep, pool)
        rows += r
        cell_rows += c
        summary[rep, pool] = s
        correct[rep, pool] = cor
        if keep_new and (rep, pool) in NEW_BLOCKS:
            pred_cols.append(col)


def groupnorm_table(cropped_dir, g):
    """Per input and encoder: the GroupNorm measure over the grid's sites, and
    WavJEPA's segment normalization against the full excerpt's."""
    import numpy as np

    gn = np.load(cropped_dir / "groupnorm.npy")[g]          # (n, 4, 6, 2)
    seg = np.load(cropped_dir / "segment_norm.npy")[g]      # (n, 7, 2): full, then INPUTS
    out = []
    for e_i, rep in enumerate(rp.CROPPED_REPRESENTATIONS[:4]):
        for j, inp in enumerate(INPUTS):
            v = gn[:, e_i, j]
            out.append({"measure": "groupnorm", "representation": rep, "input": inp,
                        "log_sd_median": float(np.median(v[:, 0])),
                        "log_sd_p95": float(np.percentile(v[:, 0], 95)),
                        "dmu_median": float(np.median(v[:, 1])),
                        "dmu_p95": float(np.percentile(v[:, 1], 95))})
    mu0, sd0 = seg[:, 0, 0], seg[:, 0, 1]
    for j, inp in enumerate(INPUTS):
        lsd = np.abs(np.log(seg[:, j + 1, 1] / sd0))
        dmu = np.abs(seg[:, j + 1, 0] - mu0) / sd0
        out.append({"measure": "wavjepa_segment", "representation": "wavjepa", "input": inp,
                    "log_sd_median": float(np.median(lsd)),
                    "log_sd_p95": float(np.percentile(lsd, 95)),
                    "dmu_median": float(np.median(dmu)),
                    "dmu_p95": float(np.percentile(dmu, 95))})
    return out


def main() -> int:
    import numpy as np
    import pandas as pd

    from capstone.predictions import build_manifest, write_prediction_columns

    sg.check_against_criterion_1()
    d = rp.SpanData()
    rp.check_conv_features(d)
    rp.check_context_features(d)
    cm, sites = rp.check_cropped_features(d)
    if (sites != d.in_grid).any():
        raise SystemExit("the cropped features do not cover exactly the grid's sites")
    manifest = build_manifest(
        rp.INVENTORY, split_scheme=sg.SPLIT_SCHEME, seeds=seeds.REGISTERED,
        n_sites_scored=int(d.in_grid.sum()), synthetic=False,
        encoders=tuple(f"{r}_{p}" for r, p in NEW_BLOCKS), run_id=f"cropped-{DATE}")
    ctx = sg.setup(d)
    W, g, tix, y = ctx["W"], ctx["g"], ctx["tix"], ctx["y"]

    rows, cell_rows, summary, correct, pred_cols = [], [], {}, {}, []
    score_blocks(ctx, sc.CHECK_BLOCKS, summary, correct, rows, cell_rows, pred_cols, False)
    n_checked = sc.check_r_scores(rows)
    print(f"re-scored ɾ blocks: {n_checked} rows identical to the committed results", flush=True)
    score_blocks(ctx, POOLED_BLOCKS, summary, correct, rows, cell_rows, pred_cols, False)
    score_blocks(ctx, NEW_BLOCKS, summary, correct, rows, cell_rows, pred_cols, True)

    comps = []

    def compare(cid, a, b, per, called):
        gap, lo, hi = sg.paired_best_gap(W, tix, y, correct[a], correct[b])
        comps.append({"comparison": cid, "per": per, "a": "_".join(a), "b": "_".join(b),
                      "gap": gap, "ci_lo": lo, "ci_hi": hi,
                      "call": sg.call(gap, lo) if called else "reported, not called"})

    # Called (§6), in a fixed order.
    for rep in (*CALLED_ENCODERS, "mfcc"):
        for span in ("vr", "rv", "vrv"):
            compare("C2", (rep, f"crop_{span}"), (rep, "crop_r"), rep, True)
    for enc in CALLED_ENCODERS:
        for span in SPANS:
            compare("C3", (enc, span), (enc, f"crop_{span}"), enc, True)
    for enc in CALLED_ENCODERS:
        for span in rp.MFCC_DURATION_SPANS:
            compare("C4", (enc, f"crop_{span}"), ("duration", "clipped"), enc, True)
    for enc in CALLED_ENCODERS:
        for span in SPANS:
            compare("C5", (enc, f"crop_{span}"), ("mfcc", f"crop_{span}"), enc, True)
        for span in rp.MFCC_DURATION_SPANS:
            compare("C5", (enc, f"crop_{span}"), ("mfcc_duration", f"crop_{span}"), enc, True)
    for span in SPANS:
        compare("C7", ("wavjepa", f"crop_{span}"), ("wav2vec2", f"crop_{span}"), span, True)
    assert sum(c["call"] != "reported, not called" for c in comps) == N_CALLED

    # Reported: HuBERT and WavLM (§8, 2026-11-26).
    for enc in REPORTED_ENCODERS:
        for span in ("vr", "rv", "vrv"):
            compare("C2", (enc, f"crop_{span}"), (enc, "crop_r"), enc, False)
        for span in SPANS:
            compare("C3", (enc, span), (enc, f"crop_{span}"), enc, False)
        for span in rp.MFCC_DURATION_SPANS:
            compare("C4", (enc, f"crop_{span}"), ("duration", "clipped"), enc, False)
        for span in SPANS:
            compare("C5", (enc, f"crop_{span}"), ("mfcc", f"crop_{span}"), enc, False)
        for span in rp.MFCC_DURATION_SPANS:
            compare("C5", (enc, f"crop_{span}"), ("mfcc_duration", f"crop_{span}"), enc, False)
    # Reported: §7's padding-only control, padded - pooled, every span and ctx.
    for rep in REPS:
        for span in (*SPANS, "ctx"):
            compare("pad", (rep, f"pad_{span}"), (rep, span), rep, False)
    # C6: best layer per crop, and the pooled design's beside it.
    for enc in ENCODERS:
        for span in SPANS:
            for pool in (f"crop_{span}", span):
                s = summary[enc, pool]
                comps.append({"comparison": "C6", "per": enc, "a": f"{enc}_{pool}", "b": "",
                              "gap": float(s["best_layer"]), "ci_lo": s["best_lo"],
                              "ci_hi": s["best_hi"],
                              "call": "best layer; no threshold" if pool != span
                              else "pooled design, for comparison"})

    gn_rows = groupnorm_table(rp.CROPPED_DIR, g)

    # ---------------------------------------------------------- write
    new_rows = [r for r in rows if (r["representation"], r["span"]) in NEW_BLOCKS]
    new_cells = [r for r in cell_rows if (r["representation"], r["span"]) in NEW_BLOCKS]
    sg.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).to_csv(sg.OUT / f"cropped-results-{DATE}.csv", index=False,
                                  float_format="%.6f")
    pd.DataFrame(new_cells).to_csv(sg.OUT / f"cropped-cells-{DATE}.csv", index=False,
                                   float_format="%.6f")
    pd.DataFrame(comps).to_csv(sg.OUT / f"cropped-comparisons-{DATE}.csv", index=False,
                               float_format="%.6f")
    pd.DataFrame(gn_rows).to_csv(sg.OUT / f"cropped-groupnorm-{DATE}.csv", index=False,
                                 float_format="%.6f")
    write_prediction_columns(pred_cols, d.inv.site_id.to_numpy(), d.in_grid, d.is_d,
                             rp.GRID_DIR / "predictions-cropped.parquet", synthetic=False)
    manifest.write(sg.OUT / "run-manifest-cropped.json")

    new = [b for b in summary if b in NEW_BLOCKS]
    nc = sum(summary[b]["nonconverged"] + summary[b]["nonconverged_control"] for b in new)
    fits = sum(summary[b]["fits"] for b in new)
    fb = cm.get("fallback_tracks", [])
    lines = [
        "Span grid, cropped design — C2, C3, and C4-C7's cropped halves; the "
        "padding-only control (§3, §6, §7; §8, 2026-11-26)",
        f"Run {DATE}. `joint`, out-of-fold over 25 cells, {len(g):,} sites "
        f"({int(y.sum()):,} /d/), the grid's sites, folds and probes. Chance is 0.50.",
        "",
        f"code commit       {manifest.code_commit}",
        f"features          {rp.CROPPED_DIR} (extracted at {cm['code_commit']})",
        f"L                 {cm['L_samples']:,} samples, {cm['L_ms']:.3f} ms; ramps "
        f"{cm['ramp_samples']} samples; seed base {cm['seed_base']}",
        f"noise fallback    {len(fb)} tracks from their speaker's tracks: {', '.join(fb)}",
        "full-excerpt check  every representation's ɾ equals the stored pooled design "
        "on every site",
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"new fits          {fits:,}, {nc} did not converge (none refit)",
        f"tripwire          every pooled ɾ re-scored: {n_checked} rows equal the committed results",
        "",
        "Per block, at the best layer (95% word-type bootstrap interval)",
        "---------------------------------------------------------------",
        f"{'block':<24}{'best':>5}  {'bal. acc.':>9}  {'95% CI':>17}  {'best-layer CI':>13}  "
        f"{'ctrl p':>6}  {'lemma':>6}  {'unclip':>6}  cells min/med/max",
    ]
    order = [(rep, p) for rep in REPS for p in (*(f"crop_{s}" for s in SPANS),
                                                *(f"pad_{s}" for s in SPANS), "pad_ctx")]
    order += [("mfcc_duration", f"crop_{s}") for s in rp.MFCC_DURATION_SPANS]
    for rep, pool in order:
        s = summary[rep, pool]
        best_ci = f"[{s['best_lo']:.0f}, {s['best_hi']:.0f}]"
        lines.append(
            f"{rep + '_' + pool:<24}{s['best_layer']:>5}  {s['ba']:>9.4f}  "
            f"[{s['lo']:.4f}, {s['hi']:.4f}]  {best_ci:>13}  "
            f"{s['control_p']:>6.4f}  {s['lemma_ba']:>6.4f}  {s['unclipped_ba']:>6.4f}  "
            + "/".join(f"{c:.3f}" for c in s["cells"]))
    lines += ["", f"Registered: C2, C3, C4, C5, C7 cropped ({N_CALLED} gaps; §6: gap >= 0.05 "
              "and paired 95% lower bound > 0)",
              "-" * 78]
    for c in comps[:N_CALLED]:
        lines.append(f"{c['comparison']}  {c['a']:<22} - {c['b']:<24} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "Reported, not called: HuBERT and WavLM; the padding-only control "
              "(padded - pooled)", "-" * 78]
    for c in comps[N_CALLED:]:
        if c["comparison"] == "C6":
            continue
        lines.append(f"{c['comparison']:<4}{c['a']:<22} - {c['b']:<24} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")
    lines += ["", "C6, best layer (2.5-97.5 percentile of its index over the resamples)"]
    for c in comps:
        if c["comparison"] == "C6":
            tag = "" if c["call"].startswith("best") else "   (pooled)"
            lines.append(f"    {c['a']:<18} layer {int(c['gap']):>2}  "
                         f"[{c['ci_lo']:.0f}, {c['ci_hi']:.0f}]{tag}")
    lines += ["", "GroupNorm at conv layer 0, input against the full excerpt: median over "
              "sites (95th percentile) of the per-site median over channels "
              "(§8, 2026-10-18). WavJEPA's segment z-score likewise.",
              f"    {'representation':<16}{'input':<10}{'|log σ ratio|':>22}{'|Δμ| / σ':>22}"]
    for r in gn_rows:
        name = r["representation"] + ("" if r["measure"] == "groupnorm" else " (segment)")
        lines.append(f"    {name:<16}{r['input']:<10}"
                     f"{r['log_sd_median']:>12.3f} ({r['log_sd_p95']:.3f})"
                     f"{r['dmu_median']:>12.3f} ({r['dmu_p95']:.3f})")
    lines += ["", f"No multiple-comparison correction is registered; {N_CALLED} gaps are called.",
              "C2 is called for MFCC as well as the two encoders: §6 registers it for each "
              "representation, as C1 was called, and the 2026-11-26 entry changes nothing "
              "else (decided by the author 2026-11-26, before any cropped score).",
              "C3 pairs a span's pooled block with its crop; C2-C7's pooled halves are in "
              "span-grid-summary-20261029.txt."]
    text = "\n".join(lines) + "\n"
    (sg.OUT / f"cropped-summary-{DATE}.txt").write_text(text)
    print(text)
    print(f"wrote {sg.OUT}/cropped-*-{DATE}.* and {rp.GRID_DIR / 'predictions-cropped.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
