#!/usr/bin/env python3
"""HuBERT and WavLM on the span grid's pooled design, scored: E1, E2 and C6 (§8, 2026-11-24).

HOW TO RUN (in .venv-app, after `run_probes.py span-grid` and `span-grid-conv`)
    .venv-app/bin/python scripts/score_conv_encoders.py

    Reads the cached blocks in /data/embeddings/span-run-20261029/grid/ and fits
    nothing. Each block is scored by score_span_grid.score_block, so the
    machinery is the grid's own: per-layer balanced accuracy with the pilot's
    word-type bootstrap, the control task, the lemma-unseen re-score, unclipped
    sites, per-cell spread.

    Before anything new is read, WavJEPA, wav2vec2-base and MFCC are re-scored
    and checked against the committed span-grid-results-20261029.csv, value for
    value at its six decimals. A mismatch stops the script.

WHAT IT CALLS (§6's rule: best-layer gap >= 0.05 and paired word-type
bootstrap 95% lower bound > 0, best layer re-selected in every resample)
    E1  each span, HuBERT - WavJEPA and WavLM - WavJEPA
    E2  each span, HuBERT - MFCC and WavLM - MFCC
    C6  best layer per span for HuBERT and WavLM, with its interval; no threshold

WHAT IT REPORTS, NOT CALLED
    The three LibriSpeech encoders against each other (one direction is enough:
    with the best layer re-picked on both sides, b - a is exactly -(a - b) in
    every resample). C1, C4 and C5's second half for HuBERT and WavLM. The
    raw-input variant of all three, its best layers, and raw - normalized.

WHAT IT WRITES
    results/span-grid/  aggregates only, nothing Buckeye-derived:
        conv-encoders-results-<date>.csv      representation x span x layer
        conv-encoders-cells-<date>.csv        each cell at each block's best layer
        conv-encoders-comparisons-<date>.csv  every gap, called or reported, and C6
        conv-encoders-summary-<date>.txt      read first
        run-manifest-conv-encoders.json
    /data/embeddings/span-run-20261029/grid/predictions-conv.parquet, per site.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
import score_span_grid as sg  # noqa: E402
from capstone import seeds  # noqa: E402

DATE = "20261124"
OLD_DATE = rp.SPAN_DATE
NEW = ("hubert", "wavlm")
RAW = {"wav2vec2_raw": "wav2vec2", "hubert_raw": "hubert", "wavlm_raw": "wavlm"}
OLD = rp.SPAN_REPRESENTATIONS                       # wavjepa, wav2vec2, mfcc
BLOCKS = ([(rep, span) for rep in (*OLD, *rp.CONV_REPRESENTATIONS) for span in rp.SPANS]
          + sg.BASELINES)
N_CALLED = 16

SPLIT_SCHEME = sg.SPLIT_SCHEME


def check_old_scores(rows):
    """The re-scored WavJEPA, wav2vec2-base and MFCC rows must equal the
    committed 10-29 results, as written (six decimals)."""
    import pandas as pd

    old = pd.read_csv(sg.OUT / f"span-grid-results-{OLD_DATE}.csv", dtype=str)
    new = pd.DataFrame([r for r in rows if r["representation"] in OLD])
    new = new.astype({"layer": int})
    cols = [c for c in old.columns if c not in ("representation", "span", "layer")]
    old = old[old.representation.isin(OLD)].astype({"layer": int})
    merged = old.merge(new, on=["representation", "span", "layer"], suffixes=("_old", ""))
    # 13 layers for each encoder, one for MFCC, per span.
    if len(merged) != len(old) or len(old) != (13 + 13 + 1) * len(rp.SPANS):
        raise SystemExit(f"re-scored {len(merged)} rows of the 10-29 results' {len(old)}")
    bad = []
    for c in cols:
        fmt = merged[c].map(lambda v: f"{v:.6f}" if isinstance(v, float) else str(v))
        if (fmt != merged[f"{c}_old"]).any():
            bad.append(c)
    if bad:
        raise SystemExit(f"re-scoring the 10-29 blocks moved {bad}; nothing new is read")
    return len(merged)


def main() -> int:
    import pandas as pd

    from capstone.predictions import build_manifest, write_prediction_columns

    sg.check_against_criterion_1()
    d = rp.SpanData()
    conv_manifest = rp.check_conv_features(d)
    manifest = build_manifest(
        rp.INVENTORY, split_scheme=SPLIT_SCHEME, seeds=seeds.REGISTERED,
        n_sites_scored=int(d.in_grid.sum()), synthetic=False,
        encoders=tuple(f"{r}_{s}" for r, s in BLOCKS), run_id=f"conv-encoders-{DATE}")
    ctx = sg.setup(d)
    W, tix, y = ctx["W"], ctx["tix"], ctx["y"]

    rows, cell_rows, summary, correct, pred_cols = [], [], {}, {}, []
    for rep, span in BLOCKS:
        r, c, s, cor, col = sg.score_block(ctx, rep, span)
        rows += r
        cell_rows += c
        summary[rep, span] = s
        correct[rep, span] = cor
        if rep in rp.CONV_REPRESENTATIONS:
            pred_cols.append(col)
    n_checked = check_old_scores(rows)
    print(f"re-scored 10-29 blocks: {n_checked} rows identical to the committed results",
          flush=True)

    comps = []

    def compare(cid, a, b, per, called):
        gap, lo, hi = sg.paired_best_gap(W, tix, y, correct[a], correct[b])
        comps.append({"comparison": cid, "per": per, "a": "_".join(a), "b": "_".join(b),
                      "gap": gap, "ci_lo": lo, "ci_hi": hi,
                      "call": sg.call(gap, lo) if called else "reported"})

    def c6(cid, rep):
        for span in rp.SPANS:
            s = summary[rep, span]
            comps.append({"comparison": cid, "per": rep, "a": f"{rep}_{span}", "b": "",
                          "gap": float(s["best_layer"]), "ci_lo": s["best_lo"],
                          "ci_hi": s["best_hi"], "call": "best layer; no threshold"})

    for enc in NEW:                                        # registered
        for span in rp.SPANS:
            compare("E1", (enc, span), ("wavjepa", span), enc, True)
    for enc in NEW:
        for span in rp.SPANS:
            compare("E2", (enc, span), ("mfcc", span), enc, True)
    assert sum(c["call"] != "reported" for c in comps) == N_CALLED
    for enc in NEW:
        c6("C6", enc)

    for a, b in (("hubert", "wav2vec2"), ("wavlm", "wav2vec2"), ("hubert", "wavlm")):
        for span in rp.SPANS:                              # reported from here on
            compare("LS", (a, span), (b, span), span, False)
    for enc in NEW:
        for span in ("vr", "rv", "vrv"):
            compare("C1", (enc, span), (enc, "r"), enc, False)
        for span in ("vr", "vrv"):
            compare("C4", (enc, span), ("duration", "clipped"), enc, False)
            compare("C5b", (enc, span), ("mfcc_duration", span), enc, False)
    for raw, norm in RAW.items():
        for span in rp.SPANS:
            compare("RAW", (raw, span), (norm, span), norm, False)
        c6("C6-raw", raw)

    # ---------------------------------------------------------- write
    new_rows = [r for r in rows if r["representation"] in rp.CONV_REPRESENTATIONS]
    new_cells = [r for r in cell_rows if r["representation"] in rp.CONV_REPRESENTATIONS]
    sg.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).to_csv(sg.OUT / f"conv-encoders-results-{DATE}.csv", index=False,
                                  float_format="%.6f")
    pd.DataFrame(new_cells).to_csv(sg.OUT / f"conv-encoders-cells-{DATE}.csv", index=False,
                                   float_format="%.6f")
    pd.DataFrame(comps).to_csv(sg.OUT / f"conv-encoders-comparisons-{DATE}.csv", index=False,
                               float_format="%.6f")
    write_prediction_columns(pred_cols, d.inv.site_id.to_numpy(), d.in_grid, d.is_d,
                             rp.GRID_DIR / "predictions-conv.parquet", synthetic=False)
    manifest.write(sg.OUT / "run-manifest-conv-encoders.json")

    nc = sum(summary[b]["nonconverged"] + summary[b]["nonconverged_control"]
             for b in summary if b[0] in rp.CONV_REPRESENTATIONS)
    fits = sum(summary[b]["fits"] for b in summary if b[0] in rp.CONV_REPRESENTATIONS)
    lines = [
        "Span grid, pooled design — HuBERT, WavLM and the raw-input variants (§8, 2026-11-24)",
        f"Run {DATE}. `joint`, out-of-fold over 25 cells, {len(ctx['g']):,} sites "
        f"({int(y.sum()):,} /d/), the grid's sites, folds and probes. Chance is 0.50.",
        "",
        f"code commit       {manifest.code_commit}",
        f"features          {rp.CONV_SPANS_DIR} (extracted at "
        f"{conv_manifest['code_commit']}); revisions {conv_manifest['checkpoint_revisions']}",
        "normalized        hubert, wavlm (and the stored wav2vec2); raw: "
        + ", ".join(RAW),
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"new fits          {fits:,}, {nc} did not converge (none refit)",
        f"tripwire          WavJEPA, wav2vec2, MFCC re-scored: {n_checked} rows equal "
        f"span-grid-results-{OLD_DATE}.csv",
        "",
        "Per span x representation, at the best layer (95% word-type bootstrap interval)",
        "-------------------------------------------------------------------------------",
        f"{'block':<20}{'best':>5}  {'bal. acc.':>9}  {'95% CI':>17}  {'best-layer CI':>13}  "
        f"{'ctrl p':>6}  {'lemma':>6}  {'unclip':>6}  cells min/med/max",
    ]
    for (rep, span), s in summary.items():
        if rep in ("duration", "mfcc_duration"):
            continue
        best_ci = f"[{s['best_lo']:.0f}, {s['best_hi']:.0f}]"
        lines.append(
            f"{rep + '_' + span:<20}{s['best_layer']:>5}  {s['ba']:>9.4f}  "
            f"[{s['lo']:.4f}, {s['hi']:.4f}]  {best_ci:>13}  "
            f"{s['control_p']:>6.4f}  {s['lemma_ba']:>6.4f}  {s['unclipped_ba']:>6.4f}  "
            + "/".join(f"{c:.3f}" for c in s["cells"]))
    heads = {"E1": "Registered: E1 and E2 (§6: gap >= 0.05 and paired 95% lower bound > 0)",
             "LS": "Reported, not called: the LibriSpeech encoders against each other",
             "C1": "Reported, not called: C1, C4, C5's second half for HuBERT and WavLM",
             "RAW": "Reported, not called: raw input minus normalized"}
    for c in comps:
        if c["comparison"].startswith("C6"):
            continue
        if c["comparison"] in heads:
            if not any(x["comparison"] == c["comparison"] for x in comps[:comps.index(c)]):
                lines += ["", heads[c["comparison"]], "-" * len(heads[c["comparison"]])]
        lines.append(f"{c['comparison']:<4} {c['a']:<18} - {c['b']:<22} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "C6, best layer (2.5-97.5 percentile of its index over the resamples); "
              "wavjepa and wav2vec2 from 10-29 for comparison"]
    for rep in ("wavjepa", "wav2vec2"):
        for span in rp.SPANS:
            s = summary[rep, span]
            lines.append(f"    {rep + '_' + span:<18} layer {s['best_layer']}  "
                         f"[{s['best_lo']:.0f}, {s['best_hi']:.0f}]")
    for c in comps:
        if c["comparison"].startswith("C6"):
            tag = "" if c["comparison"] == "C6" else "   (raw, reported)"
            lines.append(f"    {c['a']:<18} layer {int(c['gap'])}  "
                         f"[{c['ci_lo']:.0f}, {c['ci_hi']:.0f}]{tag}")
    lines += ["", f"No multiple-comparison correction is registered; {N_CALLED} gaps are called."]
    text = "\n".join(lines) + "\n"
    (sg.OUT / f"conv-encoders-summary-{DATE}.txt").write_text(text)
    print(text)
    print(f"wrote {sg.OUT}/conv-encoders-*-{DATE}.* and "
          f"{rp.GRID_DIR / 'predictions-conv.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
