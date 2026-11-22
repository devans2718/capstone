#!/usr/bin/env python3
"""The frequency re-score (span grid §8, 2026-11-21): scores with word frequency held roughly fixed.

HOW TO RUN (in .venv-app, after score_span_grid.py; the corpus mounted)
    .venv-app/bin/python scripts/score_frequency.py

    Reads the span grid's cached out-of-fold blocks under
    /data/embeddings/span-run-<date>/grid/ and fits nothing. Registered in
    experiments/span-grid/README.md §8 (2026-11-21, `9464163`) before any
    frequency split was read.

WHAT IT SCORES
    A flap's frequency is its word's token count over every Buckeye .words
    file, any pronunciation. Bands are decades: under 10, 10-99, 100-999, and
    1,000 or more. The top band is almost all /t/, so it is left out of the
    called scores. The frequency-stratified score is balanced accuracy within
    each of the three lower bands, averaged with each band weighted by its
    observed site count (the same weights in every resample). Every block is
    read at its span-grid best layer, fixed in REGISTERED_LAYER and checked
    against the block's own argmax.

    Registered calls, §6's rule (gap >= 0.05 and the paired word-type
    bootstrap 95% interval excludes zero; pilot index, seed 20260928):
      F1  wav2vec2-base, ɾ, layer 9: stratified BA - 0.5
      F2  wav2vec2-base's stratified BA - MFCC's, ɾ span
    Reported, not called: F1 and F2 for WavJEPA and the other spans; MFCC's
    F1; each band's BA with /t/ and /d/ recall and type counts; the
    unstratified BA on the same sites; the stratified BA within word-final and
    within word-medial flaps; /t/ recall in the top band.

WHAT IT WRITES (aggregates only, nothing Buckeye-derived beyond counts)
    results/span-grid/frequency-<date>.csv              block x subset
    results/span-grid/frequency-comparisons-<date>.csv  F1 and F2 per block
    results/span-grid/frequency-summary-<date>.txt      read first
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
from capstone import seeds  # noqa: E402
from count_flap_sites import DEFAULT_ROOT, read_words  # noqa: E402

OUT = rp.REPO / "results" / "span-grid"
DATE = "20261121"
REGISTRATION = "9464163 (corrected 7842a84)"

# §8 (2026-11-21): the same fixed layers as the position re-score.
REGISTERED_LAYER = {
    ("wav2vec2", "r"): 9, ("wav2vec2", "vr"): 9, ("wav2vec2", "rv"): 9, ("wav2vec2", "vrv"): 9,
    ("wavjepa", "r"): 3, ("wavjepa", "vr"): 4, ("wavjepa", "rv"): 4, ("wavjepa", "vrv"): 6,
    ("mfcc", "r"): 0, ("mfcc", "vr"): 0, ("mfcc", "rv"): 0, ("mfcc", "vrv"): 0,
}
EDGES = (10, 100, 1000)
BANDS = ("under 10", "10-99", "100-999", "1000+")
SCORED = BANDS[:3]
# The §8 table, counted before registration from the inventory and word tier alone:
# (sites, /d/, word types, /t/ types, /d/ types).
EXPECT = {"under 10": (1094, 433, 655, 384, 287), "10-99": (2126, 828, 276, 147, 136),
          "100-999": (3007, 1141, 57, 26, 32), "1000+": (4107, 111, 11, 10, 1)}
GAP = 0.05


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

    counts = Counter()
    for wf in DEFAULT_ROOT.rglob("*.words"):
        for _, _, word, *_ in read_words(wf):
            counts[word] += 1
    freq = np.array([counts[inv.word[i]] for i in g])
    band = np.array(BANDS)[np.digitize(freq, EDGES)]
    for b, want in EXPECT.items():
        m = band == b
        got = (int(m.sum()), int(y[m].sum()), len(set(keys[m])),
               len(set(keys[m & ~y])), len(set(keys[m & y])))
        if got != want:
            raise SystemExit(f"band {b}: {got}, §8 registered {want}")
    # §8 correction (2026-11-21): frequency is per word string, and the type key
    # merges clitic forms (that, that's), so some keys span two bands. The
    # bootstrap is unaffected (a resampled type brings each site into its own
    # band); the count is checked against the correction and reported.
    spanning = [k for k in set(keys) if len(set(band[keys == k])) > 1]
    n_spanning_sites = int(np.isin(keys, spanning).sum())
    if (len(spanning), n_spanning_sites) != (15, 1964):
        raise SystemExit(f"{len(spanning)} keys / {n_spanning_sites} sites span bands; "
                         "the §8 correction says 15 / 1,964")

    idx = inv.phone_idx_in_word.to_numpy()[g]
    ncanon = np.array([len(c) for c in inv.canonical_phones])[g]
    position = np.where(idx == ncanon - 1, "word-final",
                        np.where(idx == 0, "word-initial", "word-medial"))

    def subsets():
        """(name, mask) for every subset scored on its own."""
        for b in BANDS:
            yield f"band {b}", band == b
        yield "lower bands pooled", np.isin(band, SCORED)
        for p in ("word-final", "word-medial"):
            for b in SCORED:
                yield f"{p}, band {b}", (position == p) & (band == b)

    # The called scores' bands must keep both classes in every resample. The
    # reported per-position bands may not (word-medial 100-999 has 6 /d/ types);
    # their intervals skip those resamples (interval() is nan-aware) and the
    # count of skipped resamples is reported.
    called_bands = {f"band {b}" for b in SCORED}

    rows, boot = [], {}
    for (rep, span), L in REGISTERED_LAYER.items():
        z = np.load(rp.GRID_DIR / rp.block_name("probe", rep, span, "joint", False))
        if not ((z["fold"] >= 0) == d.in_grid).all():
            raise SystemExit(f"{rep}_{span}: scored sites are not the grid's")
        pred = z["real"][:, g] > 0.5
        ba_all = np.array([balanced_accuracy(y, p) for p in pred])
        if int(np.argmax(ba_all)) != L:
            raise SystemExit(f"{rep}_{span}: best layer {int(np.argmax(ba_all))}, §8 fixed {L}")
        p, cor = pred[L], pred[L] == y
        for name, m in subsets():
            row = {"representation": rep, "span": span, "layer": L, "subset": name,
                   "n": int(m.sum()), "n_d": int(y[m].sum()),
                   "n_types_t": len(set(keys[m & ~y])), "n_types_d": len(set(keys[m & y])),
                   "recall_t": float((~p[m & ~y]).mean()) if (m & ~y).any() else np.nan,
                   "recall_d": float(p[m & y].mean()) if (m & y).any() else np.nan}
            # The top band has one /d/ type; only its /t/ recall is reported.
            if name != "band 1000+":
                b = boot_balanced_accuracy(W, tix[m], y[m], cor[None, m])[:, 0]
                n_nan = int(np.isnan(b).sum())
                if n_nan and name in called_bands:
                    raise SystemExit(f"{rep}_{span} {name}: {n_nan} resamples lost a class")
                boot[rep, span, name] = b
                lo, hi = interval(b[:, None])
                row.update(balanced_acc=balanced_accuracy(y[m], p[m]),
                           ci_lo=float(lo[0]), ci_hi=float(hi[0]), boot_nan=n_nan)
            rows.append(row)
        # Stratified scores: whole grid, then within each position group.
        for prefix, label in (("", "stratified"), ("word-final, ", "word-final, stratified"),
                              ("word-medial, ", "word-medial, stratified")):
            names = [f"{prefix}band {b}" for b in SCORED]
            sub = {r["subset"]: r for r in rows
                   if r["representation"] == rep and r["span"] == span}
            w = np.array([sub[n]["n"] for n in names], float)
            w /= w.sum()
            point = float(sum(wi * sub[n]["balanced_acc"] for wi, n in zip(w, names)))
            bs = sum(wi * boot[rep, span, n] for wi, n in zip(w, names))
            boot[rep, span, label] = bs
            lo, hi = interval(bs[:, None])
            rows.append({"representation": rep, "span": span, "layer": L, "subset": label,
                         "n": int(sum(sub[n]["n"] for n in names)),
                         "n_d": int(sum(sub[n]["n_d"] for n in names)),
                         "balanced_acc": point, "ci_lo": float(lo[0]), "ci_hi": float(hi[0]),
                         "boot_nan": int(np.isnan(bs).sum())})

    res = pd.DataFrame(rows)

    def point(rep, span, subset):
        r = res[(res.representation == rep) & (res.span == span) & (res.subset == subset)]
        return float(r.balanced_acc.iloc[0])

    comps = []
    for rep in ("wav2vec2", "wavjepa", "mfcc"):
        for span in rp.SPANS:
            registered = rep == "wav2vec2" and span == "r"
            s = boot[rep, span, "stratified"]
            f1 = point(rep, span, "stratified") - 0.5
            lo, hi = interval((s - 0.5)[:, None])
            comps.append({"comparison": "F1", "representation": rep, "span": span,
                          "gap": f1, "ci_lo": float(lo[0]), "ci_hi": float(hi[0]),
                          "call": call(f1, lo[0], hi[0]) if registered else "reported, not called"})
            if rep == "mfcc":
                continue
            f2 = point(rep, span, "stratified") - point("mfcc", span, "stratified")
            lo, hi = interval((s - boot["mfcc", span, "stratified"])[:, None])
            comps.append({"comparison": "F2", "representation": rep, "span": span,
                          "gap": f2, "ci_lo": float(lo[0]), "ci_hi": float(hi[0]),
                          "call": call(f2, lo[0], hi[0]) if registered else "reported, not called"})
    comp = pd.DataFrame(comps)

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"frequency-{DATE}.csv", index=False, float_format="%.6f")
    comp.to_csv(OUT / f"frequency-comparisons-{DATE}.csv", index=False, float_format="%.6f")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rp.REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    lines = [
        "Span grid, frequency re-score (§8, 2026-11-21) — scores with word frequency held roughly fixed",
        f"Run {DATE}, registered at {REGISTRATION}; run on top of {commit[:7]}, "
        "this script committed with these results.",
        f"`joint` out-of-fold predictions from the span grid's run {rp.SPAN_DATE}; nothing refit.",
        f"Bootstrap: 10,000 resamples of the pilot's 991-type index, seed {seeds.BOOTSTRAP}.",
        "Frequency: the word's token count over all Buckeye .words files. Bands: decades; "
        "the 1000+ band is left out of the stratified scores.",
        f"Type keys spanning two bands (clitic forms; §8 correction): {len(spanning)}, "
        f"{n_spanning_sites:,} sites.",
        "",
        "Calls (F1: stratified BA minus 0.5; F2: stratified BA minus MFCC's). "
        "Called: wav2vec2_r only",
    ]
    for c in comps:
        lines.append(f"  {c['comparison']}  {c['representation'] + '_' + c['span']:<13} "
                     f"{c['gap']:+.4f}  [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "At the registered layer: balanced accuracy [95% CI]; sites, /d/; "
              "recall /t/, /d/; /t/ and /d/ types"]
    for (rep, span), L in REGISTERED_LAYER.items():
        lines.append(f"  {rep}_{span} (layer {L})")
        for r in res[(res.representation == rep) & (res.span == span)].itertuples():
            ba = (f"{r.balanced_acc:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]"
                  if not pd.isna(r.balanced_acc) else "   n/a (one class)      ")
            extra = ""
            if not pd.isna(getattr(r, "recall_t", np.nan)):
                rd = f"{r.recall_d:.3f}" if not pd.isna(r.recall_d) else "  n/a"
                extra = (f"  t {r.recall_t:.3f} d {rd}  "
                         f"types {int(r.n_types_t)}/{int(r.n_types_d)}")
            nan = (f"  ({int(r.boot_nan)} resamples skipped)"
                   if not pd.isna(r.boot_nan) and r.boot_nan else "")
            lines.append(f"    {r.subset:<30} {ba}  {r.n:>5,} {r.n_d:>5,}{extra}{nan}")
    text = "\n".join(lines) + "\n"
    (OUT / f"frequency-summary-{DATE}.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
