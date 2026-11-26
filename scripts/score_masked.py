#!/usr/bin/env python3
"""The flap-masked variant, scored: M1 and M2 (span grid §8, 2026-11-26).

HOW TO RUN (in .venv-app, after scripts/extract_cropped.py and
`run_probes.py span-grid-cropped`)
    .venv-app/bin/python scripts/score_masked.py

    Reads the cached blocks in /data/embeddings/span-run-20261029/grid/ and fits
    nothing. Each block is scored by score_span_grid.score_block, and the
    pooled ɾ blocks are re-scored against their committed results first
    (score_context's tripwire).

THE INPUTS
    mask_ctx  the excerpt padded to L with the flap's samples replaced by the
              site's pause-matched noise, pooled over the context span (every
              frame of the excerpt outside that encoder's ɾ frames)
    pad_ctx   the unmasked counterpart: the padding-only control, pooled over
              the same frames. It differs from mask_ctx only in the flap's samples

WHAT IT CALLS (§6's rule: best-layer gap >= 0.05 and paired word-type
bootstrap 95% lower bound > 0, best layer re-selected in every resample)
    M1  wav2vec2-base, mask_ctx: best-layer balanced accuracy minus chance (0.5)
    M2  wav2vec2-base, pad_ctx minus mask_ctx, paired. One-sided, as §6's rule
        is (§8, 2026-10-29); the gap's sign and interval are reported either way

WHAT IT REPORTS, NOT CALLED
    M1 and M2 for WavJEPA, HuBERT, WavLM and MFCC (MFCC's context frames barely
    hear the flap, so its M2 is close to a check of the masking itself). The
    masked context against the pooled design's context (K1's span), which mixes
    the padding with the masking. Every layer's curve and the checks, for every
    mask_ctx block.

WHAT IT WRITES
    results/span-grid/  aggregates only, nothing Buckeye-derived:
        masked-results-<date>.csv      representation x layer
        masked-cells-<date>.csv        each cell at each block's best layer
        masked-comparisons-<date>.csv  every gap, called or reported
        masked-summary-<date>.txt      read first
        run-manifest-masked.json
    /data/embeddings/span-run-20261029/grid/predictions-masked.parquet, per site.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
import score_context as sc  # noqa: E402
import score_span_grid as sg  # noqa: E402
from capstone import seeds  # noqa: E402

DATE = datetime.date.today().strftime("%Y%m%d")
REPS = rp.CROPPED_REPRESENTATIONS            # wavjepa, wav2vec2, hubert, wavlm, mfcc
NEW_BLOCKS = [(rep, "mask_ctx") for rep in REPS]
OTHER_BLOCKS = [(rep, p) for rep in REPS for p in ("pad_ctx", "ctx")]
N_CALLED = 2


def main() -> int:
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
        encoders=tuple(f"{r}_{p}" for r, p in NEW_BLOCKS), run_id=f"masked-{DATE}")
    ctx = sg.setup(d)
    W, g, tix, y = ctx["W"], ctx["g"], ctx["tix"], ctx["y"]

    rows, cell_rows, summary, correct, pred_cols = [], [], {}, {}, []
    for blocks in (sc.CHECK_BLOCKS, OTHER_BLOCKS, NEW_BLOCKS):
        for rep, pool in blocks:
            if (rep, pool) in summary:
                continue
            r, c, s, cor, col = sg.score_block(ctx, rep, pool)
            rows += r
            cell_rows += c
            summary[rep, pool] = s
            correct[rep, pool] = cor
            if (rep, pool) in NEW_BLOCKS:
                pred_cols.append(col)
        if blocks is sc.CHECK_BLOCKS:
            n_checked = sc.check_r_scores(rows)
            print(f"re-scored ɾ blocks: {n_checked} rows identical to the committed results",
                  flush=True)

    comps = []

    def add(cid, rep, a, b, gap, lo, hi, called):
        comps.append({"comparison": cid, "representation": rep, "a": a, "b": b, "n": len(g),
                      "gap": gap, "ci_lo": lo, "ci_hi": hi,
                      "call": sg.call(gap, lo) if called else "reported, not called"})

    def m1(rep, called):
        gap, lo, hi = sc.chance_gap(W, tix, y, correct[rep, "mask_ctx"])
        add("M1", rep, f"{rep}_mask_ctx", "0.5", gap, lo, hi, called)

    def paired(cid, rep, a, b, called):
        gap, lo, hi = sg.paired_best_gap(W, tix, y, correct[rep, a], correct[rep, b])
        add(cid, rep, f"{rep}_{a}", f"{rep}_{b}", gap, lo, hi, called)

    m1("wav2vec2", True)                                   # registered
    paired("M2", "wav2vec2", "pad_ctx", "mask_ctx", True)
    assert sum(c["call"] != "reported, not called" for c in comps) == N_CALLED
    for rep in REPS:                                        # reported from here on
        if rep != "wav2vec2":
            m1(rep, False)
    for rep in REPS:
        if rep != "wav2vec2":
            paired("M2", rep, "pad_ctx", "mask_ctx", False)
    for rep in REPS:
        paired("vsK1", rep, "ctx", "mask_ctx", False)      # pooled ctx - masked ctx

    # ---------------------------------------------------------- write
    new_rows = [r for r in rows if (r["representation"], r["span"]) in NEW_BLOCKS]
    new_cells = [r for r in cell_rows if (r["representation"], r["span"]) in NEW_BLOCKS]
    sg.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).to_csv(sg.OUT / f"masked-results-{DATE}.csv", index=False,
                                  float_format="%.6f")
    pd.DataFrame(new_cells).to_csv(sg.OUT / f"masked-cells-{DATE}.csv", index=False,
                                   float_format="%.6f")
    pd.DataFrame(comps).to_csv(sg.OUT / f"masked-comparisons-{DATE}.csv", index=False,
                               float_format="%.6f")
    write_prediction_columns(pred_cols, d.inv.site_id.to_numpy(), d.in_grid, d.is_d,
                             rp.GRID_DIR / "predictions-masked.parquet", synthetic=False)
    manifest.write(sg.OUT / "run-manifest-masked.json")

    nc = sum(summary[b]["nonconverged"] + summary[b]["nonconverged_control"] for b in NEW_BLOCKS)
    fits = sum(summary[b]["fits"] for b in NEW_BLOCKS)
    lines = [
        "Span grid — the flap-masked variant: M1, M2 (§8, 2026-11-26)",
        f"Run {DATE}. `joint`, out-of-fold over 25 cells, {len(g):,} sites "
        f"({int(y.sum()):,} /d/), the grid's sites, folds and probes. Chance is 0.50.",
        "",
        f"code commit       {manifest.code_commit}",
        f"features          {rp.CROPPED_DIR} (extracted at {cm['code_commit']})",
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"new fits          {fits:,}, {nc} did not converge (none refit)",
        f"tripwire          every pooled ɾ re-scored: {n_checked} rows equal the committed results",
        "",
        "Per block, at the best layer (95% word-type bootstrap interval)",
        "---------------------------------------------------------------",
        f"{'block':<20}{'best':>5}  {'bal. acc.':>9}  {'95% CI':>17}  {'best-layer CI':>13}  "
        f"{'ctrl p':>6}  {'lemma':>6}  {'unclip':>6}  cells min/med/max",
    ]
    for rep in REPS:
        for pool in ("mask_ctx", "pad_ctx", "ctx"):
            s = summary[rep, pool]
            best_ci = f"[{s['best_lo']:.0f}, {s['best_hi']:.0f}]"
            tag = "" if pool == "mask_ctx" else ("   (padding-only control)" if pool == "pad_ctx"
                                                 else "   (pooled design, K1's span)")
            lines.append(
                f"{rep + '_' + pool:<20}{s['best_layer']:>5}  {s['ba']:>9.4f}  "
                f"[{s['lo']:.4f}, {s['hi']:.4f}]  {best_ci:>13}  "
                f"{s['control_p']:>6.4f}  {s['lemma_ba']:>6.4f}  {s['unclipped_ba']:>6.4f}  "
                + "/".join(f"{c:.3f}" for c in s["cells"]) + tag)
    lines += ["", "Registered: M1, M2 (§6: gap >= 0.05 and paired 95% lower bound > 0)",
              "--------------------------------------------------------------------"]
    for c in comps[:N_CALLED]:
        lines.append(f"{c['comparison']:<5}{c['a']:<20} - {c['b']:<20} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "Reported, not called (vsK1: the pooled design's context - the masked "
              "context, which mixes the padding with the masking)", "-" * 78]
    for c in comps[N_CALLED:]:
        lines.append(f"{c['comparison']:<5}{c['a']:<20} - {c['b']:<20} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")
    lines += ["", "M1-M2 reading (§8, 2026-11-26): effect/effect the context carries the "
              "label on its own and hearing the flap adds to it; effect/no effect the context "
              "carries it on its own, hearing the flap adds nothing measurable; no "
              "effect/effect K1's context score came from having heard the flap; no "
              "effect/no effect unsettled. No effect on M2 is not equality.",
              f"No multiple-comparison correction is registered; {N_CALLED} gaps are called."]
    text = "\n".join(lines) + "\n"
    (sg.OUT / f"masked-summary-{DATE}.txt").write_text(text)
    print(text)
    print(f"wrote {sg.OUT}/masked-*-{DATE}.* and {rp.GRID_DIR / 'predictions-masked.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
