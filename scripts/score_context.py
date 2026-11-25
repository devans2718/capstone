#!/usr/bin/env python3
"""The context-only span and the preceding stretch without the flap, scored: K1-K3
(span grid §8, 2026-11-23; HuBERT and WavLM's numbers, 2026-11-24).

HOW TO RUN (in .venv-app, after `run_probes.py span-grid`, `span-grid-conv` and
`span-grid-context`)
    .venv-app/bin/python scripts/score_context.py

    Reads the cached blocks in /data/embeddings/span-run-20261029/grid/ and fits
    nothing. Each block is scored by score_span_grid.score_block, so the
    machinery is the grid's own: per-layer balanced accuracy with the pilot's
    word-type bootstrap, the control task, the lemma-unseen re-score, unclipped
    sites, per-cell spread.

    Before anything new is read, the ɾ blocks of every representation are
    re-scored and checked against the committed results (10-29 for WavJEPA,
    wav2vec2-base and MFCC; 11-24 for HuBERT, WavLM and the raw variants), value
    for value at their six decimals. A mismatch stops the script.

WHAT IT CALLS (§6's rule: best-layer gap >= 0.05 and paired word-type
bootstrap 95% lower bound > 0, best layer re-selected in every resample)
    K1  wav2vec2-base, context: best-layer balanced accuracy minus chance (0.5)
    K2  wav2vec2-base, ɾ minus context, paired. One-sided, as §6's rule is; the
        gap's sign and interval are reported either way
    K3  MFCC, context: balanced accuracy minus chance

WHAT IT REPORTS, NOT CALLED
    K1 and K3's numbers for WavJEPA, HuBERT and WavLM (context minus chance), and
    K2's for WavJEPA, MFCC, HuBERT and WavLM. The preceding stretch without the
    flap (v) for every representation, and v minus Vɾ. Every layer's curve, the
    control task, lemma-unseen and unclipped re-scores, per-cell spread, for
    every block (the raw-input variants included, as curves only). The
    frequency re-score's stratified score (score_frequency.py's bands) applied
    to each representation's context predictions, at that block's best layer.

    WavJEPA's preceding stretch is empty at 6 grid sites (the context
    extraction's v_empty). run_probes.py left them out of that block alone, so
    it is scored on 10,328 sites, and its v - Vɾ on those same sites.

WHAT IT WRITES
    results/span-grid/  aggregates only, nothing Buckeye-derived:
        context-results-<date>.csv      representation x span x layer
        context-cells-<date>.csv        each cell at each block's best layer
        context-comparisons-<date>.csv  every gap, called or reported
        context-summary-<date>.txt      read first
        run-manifest-context.json
    /data/embeddings/span-run-20261029/grid/predictions-context.parquet, per site;
    WavJEPA's v in predictions-context-wavjepa-v.parquet beside it, on its 10,328.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
import score_frequency as sf  # noqa: E402
import score_span_grid as sg  # noqa: E402
from capstone import seeds  # noqa: E402

DATE = "20261124"
OLD = rp.SPAN_REPRESENTATIONS                       # wavjepa, wav2vec2, mfcc
CONV = rp.CONV_REPRESENTATIONS                      # hubert, wavlm, and the raw variants
REPORTED_ENCODERS = ("wavjepa", "hubert", "wavlm")  # K-numbers reported, not called
REPS = (*OLD, *CONV)
NEW_BLOCKS = [(rep, span) for rep in REPS for span in rp.CONTEXT_SPANS]
CHECK_BLOCKS = [(rep, "r") for rep in REPS]          # the tripwire
# Each pair's own committed results, for the tripwire.
COMMITTED = {**{rep: f"span-grid-results-{rp.SPAN_DATE}.csv" for rep in OLD},
             **{rep: "conv-encoders-results-20261124.csv" for rep in CONV}}
N_CALLED = 3

SPLIT_SCHEME = sg.SPLIT_SCHEME


def check_r_scores(rows):
    """The re-scored ɾ rows must equal each representation's committed results,
    as written (six decimals)."""
    import pandas as pd

    new = pd.DataFrame([r for r in rows if r["span"] == "r"]).astype({"layer": int})
    n = 0
    bad = []
    for fname in sorted(set(COMMITTED.values())):
        reps = [r for r, f in COMMITTED.items() if f == fname]
        old = pd.read_csv(sg.OUT / fname, dtype=str)
        old = old[old.representation.isin(reps) & (old.span == "r")].astype({"layer": int})
        cols = [c for c in old.columns if c not in ("representation", "span", "layer")]
        merged = old.merge(new, on=["representation", "span", "layer"], suffixes=("_old", ""))
        expected = sum(1 if r == "mfcc" else 13 for r in reps)
        if len(merged) != len(old) or len(old) != expected:
            raise SystemExit(f"re-scored {len(merged)} ɾ rows of {fname}'s {len(old)}")
        for c in cols:
            fmt = merged[c].map(lambda v: f"{v:.6f}" if isinstance(v, float) else str(v))
            if (fmt != merged[f"{c}_old"]).any():
                bad.append(f"{fname}:{c}")
        n += len(merged)
    if bad:
        raise SystemExit(f"re-scoring the ɾ blocks moved {bad}; nothing new is read")
    return n


def subset_ctx(ctx, keep):
    """The scoring context restricted to the grid sites where `keep` (over the
    grid's rows) is true: for WavJEPA's v, which has no frames at 6 sites. The
    bootstrap weights stay per type, so resamples are the same as everywhere."""
    d2 = copy.copy(ctx["d"])
    in_grid = d2.in_grid.copy()
    in_grid[ctx["g"][~keep]] = False
    d2.in_grid = in_grid
    out = dict(ctx, d=d2, g=ctx["g"][keep], y=ctx["y"][keep], tix=ctx["tix"][keep],
               cy=ctx["cy"][keep], lsub=ctx["lsub"][keep], unclipped=ctx["unclipped"][keep])
    return out


def chance_gap(W, tix, y, correct):
    """Best-layer balanced accuracy minus 0.5, with the word-type bootstrap
    interval, the best layer re-selected in every resample. Returns gap, lo, hi."""
    import numpy as np

    from capstone.probes.scoring import (balanced_accuracy_from_correct,
                                         boot_balanced_accuracy, interval)

    ba = max(balanced_accuracy_from_correct(y, c) for c in correct)
    boot = boot_balanced_accuracy(W, tix, y, correct)
    lo, hi = interval((np.nanmax(boot, axis=1) - 0.5)[:, None])
    return float(ba - 0.5), float(lo[0]), float(hi[0])


def frequency_bands(d, g):
    """score_frequency.py's decade bands for the grid's sites, with its §8 checks."""
    import numpy as np

    from count_flap_sites import DEFAULT_ROOT, read_words

    counts = Counter()
    for wf in DEFAULT_ROOT.rglob("*.words"):
        for _, _, word, *_ in read_words(wf):
            counts[word] += 1
    freq = np.array([counts[d.inv.word[i]] for i in g])
    band = np.array(sf.BANDS)[np.digitize(freq, sf.EDGES)]
    y = d.is_d[g]
    keys = np.array([d.keys[i] for i in g])
    for b, want in sf.EXPECT.items():
        m = band == b
        got = (int(m.sum()), int(y[m].sum()), len(set(keys[m])),
               len(set(keys[m & ~y])), len(set(keys[m & y])))
        if got != want:
            raise SystemExit(f"band {b}: {got}, §8 (2026-11-21) registered {want}")
    return band


def stratified(W, tix, y, band, pred):
    """score_frequency's stratified score for one layer's predictions: balanced
    accuracy within each of the three lower bands, weighted by sites. Returns
    point, lo, hi."""
    import numpy as np

    from capstone.probes.scoring import balanced_accuracy, boot_balanced_accuracy, interval

    cor = pred == y
    masks = [band == b for b in sf.SCORED]
    w = np.array([m.sum() for m in masks], float)
    w /= w.sum()
    point = sum(wi * balanced_accuracy(y[m], pred[m]) for wi, m in zip(w, masks))
    bs = sum(wi * boot_balanced_accuracy(W, tix[m], y[m], cor[None, m])[:, 0]
             for wi, m in zip(w, masks))
    lo, hi = interval(bs[:, None])
    return float(point), float(lo[0]), float(hi[0])


def main() -> int:
    import numpy as np
    import pandas as pd

    from capstone.predictions import build_manifest, write_prediction_columns

    sg.check_against_criterion_1()
    d = rp.SpanData()
    rp.check_conv_features(d)
    sites = rp.check_context_features(d)
    ctx_manifest = json.loads((rp.CONTEXT_DIR / "extract-manifest.json").read_text())
    manifest = build_manifest(
        rp.INVENTORY, split_scheme=SPLIT_SCHEME, seeds=seeds.REGISTERED,
        n_sites_scored=int(d.in_grid.sum()), synthetic=False,
        encoders=tuple(f"{r}_{s}" for r, s in NEW_BLOCKS), run_id=f"context-{DATE}")
    ctx = sg.setup(d)
    W, g, tix, y = ctx["W"], ctx["g"], ctx["tix"], ctx["y"]

    rows, cell_rows, summary, correct, pred_cols, keep_of = [], [], {}, {}, [], {}
    for rep, span in CHECK_BLOCKS + [("mfcc", "vr")] + [(r, "vr") for r in REPS if r != "mfcc"] \
            + NEW_BLOCKS:
        if (rep, span) in summary:
            continue
        keep = sites[rep, span][g] if (rep, span) in sites else np.ones(len(g), bool)
        bctx = ctx if keep.all() else subset_ctx(ctx, keep)
        r, c, s, cor, col = sg.score_block(bctx, rep, span)
        rows += r
        cell_rows += c
        summary[rep, span] = s
        # Correctness on the full grid's columns (False where the block has no site),
        # so a pair can be restricted to the sites both blocks have via keep_of.
        full = np.zeros((cor.shape[0], len(g)), dtype=bool)
        full[:, keep] = cor
        correct[rep, span] = full
        keep_of[rep, span] = keep
        if (rep, span) in NEW_BLOCKS:
            pred_cols.append((col, keep))
    n_checked = check_r_scores(rows)
    print(f"re-scored ɾ blocks: {n_checked} rows identical to the committed results",
          flush=True)

    comps = []

    def add(cid, rep, a, b, gap, lo, hi, called, n):
        comps.append({"comparison": cid, "representation": rep, "a": a, "b": b, "n": n,
                      "gap": gap, "ci_lo": lo, "ci_hi": hi,
                      "call": sg.call(gap, lo) if called else "reported, not called"})

    def vs_chance(cid, rep, called):
        gap, lo, hi = chance_gap(W, tix, y, correct[rep, "ctx"])
        add(cid, rep, f"{rep}_ctx", "0.5", gap, lo, hi, called, len(g))

    def paired(cid, rep, a, b, called):
        keep = keep_of[rep, a] & keep_of[rep, b]
        gap, lo, hi = sg.paired_best_gap(W, tix[keep], y[keep], correct[rep, a][:, keep],
                                         correct[rep, b][:, keep])
        add(cid, rep, f"{rep}_{a}", f"{rep}_{b}", gap, lo, hi, called, int(keep.sum()))

    vs_chance("K1", "wav2vec2", True)                      # registered
    paired("K2", "wav2vec2", "r", "ctx", True)
    vs_chance("K3", "mfcc", True)
    assert sum(c["call"] != "reported, not called" for c in comps) == N_CALLED
    for rep in REPORTED_ENCODERS:                          # reported from here on
        vs_chance("K1", rep, False)
    for rep in ("wavjepa", "mfcc", "hubert", "wavlm"):
        paired("K2", rep, "r", "ctx", False)
    for rep in ("wavjepa", "wav2vec2", "mfcc", "hubert", "wavlm"):
        paired("V", rep, "v", "vr", False)

    # The frequency re-score's stratified score on each context block, at its best layer.
    band = frequency_bands(d, g)
    strat = []
    for rep in ("wavjepa", "wav2vec2", "mfcc", "hubert", "wavlm"):
        z = np.load(rp.GRID_DIR / rp.block_name("probe", rep, "ctx", "joint", False))
        L = summary[rep, "ctx"]["best_layer"]
        p, lo, hi = stratified(W, tix, y, band, z["real"][L, g] > 0.5)
        strat.append({"representation": rep, "span": "ctx", "layer": L,
                      "stratified_balanced_acc": p, "ci_lo": lo, "ci_hi": hi})

    # ---------------------------------------------------------- write
    new_rows = [r for r in rows if (r["representation"], r["span"]) in NEW_BLOCKS]
    new_cells = [r for r in cell_rows if (r["representation"], r["span"]) in NEW_BLOCKS]
    sg.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).to_csv(sg.OUT / f"context-results-{DATE}.csv", index=False,
                                  float_format="%.6f")
    pd.DataFrame(new_cells).to_csv(sg.OUT / f"context-cells-{DATE}.csv", index=False,
                                   float_format="%.6f")
    comp_df = pd.DataFrame(comps)
    strat_df = pd.DataFrame(strat).assign(comparison="stratified", call="reported, not called")
    pd.concat([comp_df, strat_df], ignore_index=True).to_csv(
        sg.OUT / f"context-comparisons-{DATE}.csv", index=False, float_format="%.6f")
    # The writer needs a prediction at every site it is told is scored, so WavJEPA's
    # v (no frames at 6 grid sites) goes to a file of its own with its own mask.
    full = [col for col, keep in pred_cols if keep.all()]
    write_prediction_columns(full, d.inv.site_id.to_numpy(), d.in_grid, d.is_d,
                             rp.GRID_DIR / "predictions-context.parquet", synthetic=False)
    for col, keep in pred_cols:
        if not keep.all():
            scored = d.in_grid.copy()
            scored[g[~keep]] = False
            write_prediction_columns([col], d.inv.site_id.to_numpy(), scored, d.is_d,
                                     rp.GRID_DIR / f"predictions-context-{col[0]}-{col[1]}.parquet",
                                     synthetic=False)
    manifest.write(sg.OUT / "run-manifest-context.json")

    new = [b for b in summary if b in NEW_BLOCKS]
    nc = sum(summary[b]["nonconverged"] + summary[b]["nonconverged_control"] for b in new)
    fits = sum(summary[b]["fits"] for b in new)
    lines = [
        "Span grid, pooled design — the context-only span and the preceding stretch "
        "without the flap (§8, 2026-11-23; HuBERT and WavLM, 2026-11-24)",
        f"Run {DATE}. `joint`, out-of-fold over 25 cells, {len(g):,} sites "
        f"({int(y.sum()):,} /d/), the grid's sites, folds and probes. Chance is 0.50.",
        "",
        f"code commit       {manifest.code_commit}",
        f"features          {rp.CONTEXT_DIR} (extracted at {ctx_manifest['code_commit']}); "
        f"{rp.CONV_SPANS_DIR} for HuBERT, WavLM and the raw variants",
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"new fits          {fits:,}, {nc} did not converge (none refit)",
        f"tripwire          every representation's ɾ re-scored: {n_checked} rows equal "
        "the committed results",
        f"WavJEPA's v       {int(keep_of['wavjepa', 'v'].sum()):,} sites; "
        f"{int((~keep_of['wavjepa', 'v']).sum())} have no preceding-stretch frames",
        "",
        "Per span x representation, at the best layer (95% word-type bootstrap interval)",
        "-------------------------------------------------------------------------------",
        f"{'block':<20}{'best':>5}  {'bal. acc.':>9}  {'95% CI':>17}  {'best-layer CI':>13}  "
        f"{'ctrl p':>6}  {'lemma':>6}  {'unclip':>6}  cells min/med/max",
    ]
    order = [(rep, span) for rep in REPS for span in ("r", "vr", "ctx", "v")]
    for rep, span in order:
        s = summary[rep, span]
        best_ci = f"[{s['best_lo']:.0f}, {s['best_hi']:.0f}]"
        tag = "" if (rep, span) in NEW_BLOCKS else "   (committed, for comparison)"
        lines.append(
            f"{rep + '_' + span:<20}{s['best_layer']:>5}  {s['ba']:>9.4f}  "
            f"[{s['lo']:.4f}, {s['hi']:.4f}]  {best_ci:>13}  "
            f"{s['control_p']:>6.4f}  {s['lemma_ba']:>6.4f}  {s['unclipped_ba']:>6.4f}  "
            + "/".join(f"{c:.3f}" for c in s["cells"]) + tag)
    lines += ["", "Registered: K1, K2, K3 (§6: gap >= 0.05 and paired 95% lower bound > 0)",
              "-----------------------------------------------------------------------"]
    for c in comps[:N_CALLED]:
        lines.append(f"{c['comparison']:<3} {c['a']:<16} - {c['b']:<14} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "Reported, not called", "--------------------"]
    for c in comps[N_CALLED:]:
        n = "" if c["n"] == len(g) else f"   (on {c['n']:,} sites)"
        lines.append(f"{c['comparison']:<3} {c['a']:<16} - {c['b']:<14} {c['gap']:+.4f}  "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]{n}")
    lines += ["", "Frequency-stratified context score (score_frequency.py's three lower "
              "bands, site-weighted), at each block's best layer; reported, not called"]
    for s in strat:
        lines.append(f"    {s['representation'] + '_ctx':<16} layer {s['layer']:>2}  "
                     f"{s['stratified_balanced_acc']:.4f}  [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")
    lines += ["", "K1-K2 reading (§8, 2026-11-23): effect/effect concentrated at the flap; "
              "effect/no effect the flap adds nothing measurable; no effect/effect only the "
              "flap; no effect/no effect unsettled. No effect on K2 is not equality. "
              "K3 is read separately.",
              f"No multiple-comparison correction is registered; {N_CALLED} gaps are called."]
    text = "\n".join(lines) + "\n"
    (sg.OUT / f"context-summary-{DATE}.txt").write_text(text)
    print(text)
    print(f"wrote {sg.OUT}/context-*-{DATE}.* and {rp.GRID_DIR / 'predictions-context.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
