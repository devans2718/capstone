#!/usr/bin/env python3
"""The transfer probe, scored: trained on unflapped stops, tested on the grid's
flaps; X1 and X2 (span grid §8, 2026-11-21; HuBERT and WavLM's, 2026-11-25).

HOW TO RUN (in .venv-app, after `run_probes.py span-grid-transfer`)
    .venv-app/bin/python scripts/score_transfer.py

    Reads the cached blocks in /data/embeddings/span-run-20261029/grid/transfer/
    and fits nothing. Scored with the grid's machinery: balanced accuracy over the
    10,334 flaps, the pilot's word-type bootstrap (10,000 resamples, seed
    20260928, the 991-type index).

LAYERS ARE FIXED, NOT CHOSEN ON TRANSFER SCORES
    Each representation at its span-grid best layer for the matching flap span:
    wav2vec2-base 9; WavJEPA 3, 4, 4, 6; MFCC its one (§8, 2026-11-21); HuBERT
    11, 8, 11, 10 and WavLM 12, 10, 11, 10 (§8, 2026-11-25), in the order ɾ, Vɾ,
    ɾV, VɾV. Each is checked against the committed span-grid results before
    anything is read. The raw-input variants take their own span-grid best
    layers, read from the committed results the same way; the registration
    fixes no layer for them, and their numbers are reported only.

WHAT IT CALLS (§6's rule: a gap of at least 0.05 and a word-type bootstrap 95%
lower bound above zero; one layer, so no re-selection)
    X1  wav2vec2-base, stop -> ɾ at layer 9, 0.40 set: balanced accuracy - 0.5
    X2  wav2vec2-base, V+stop -> Vɾ at layer 9, 0.40 set: the same

WHAT IT REPORTS, NOT CALLED
    X1 and X2's numbers for WavJEPA, MFCC, HuBERT, WavLM and the raw variants; the
    stop+V and V+stop+V spans; every layer's curve; the 0.25 and 0.50 sets on the
    stop and V+stop spans; at the fixed layer of every 0.40 block: transfer on
    the flaps at a covered position only (8,916), the flap-trained probe's score
    at the same layer (from the committed results), the frequency re-score's
    stratified score, /t/ and /d/ recall, and each speaker group's and word-type
    group's score.

WHAT IT WRITES
    results/span-grid/  aggregates only, nothing Buckeye-derived:
        transfer-results-<date>.csv      block x layer
        transfer-groups-<date>.csv       speaker and word-type groups, fixed layer
        transfer-comparisons-<date>.csv  every gap, called or reported
        transfer-summary-<date>.txt      read first
        run-manifest-transfer.json
    /data/embeddings/span-run-20261029/grid/transfer/predictions-transfer.parquet,
    per flap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probes as rp  # noqa: E402
import score_context as sc  # noqa: E402
import score_span_grid as sg  # noqa: E402
from capstone import seeds  # noqa: E402

DATE = "20261125"
COMMITTED_DIR = rp.REPO / "results" / "span-grid"
COMMITTED = {**{rep: f"span-grid-results-{rp.SPAN_DATE}.csv" for rep in rp.SPAN_REPRESENTATIONS},
             **{rep: "conv-encoders-results-20261124.csv" for rep in rp.CONV_REPRESENTATIONS}}
FLAP_SPANS = ("r", "vr", "rv", "vrv")
# The registration's layers, in flap-span order (§8, 2026-11-21 and 2026-11-25).
REGISTERED_LAYERS = {"wav2vec2": (9, 9, 9, 9), "wavjepa": (3, 4, 4, 6), "mfcc": (0, 0, 0, 0),
                     "hubert": (11, 8, 11, 10), "wavlm": (12, 10, 11, 10)}
CALLED = {("wav2vec2", "s", "040"): "X1", ("wav2vec2", "vs", "040"): "X2"}
SPAN_ID = {"s": "X1", "vs": "X2", "sv": "stop+V", "vsv": "V+stop+V"}
SPAN_NAME = {"s": "stop -> ɾ", "vs": "V+stop -> Vɾ", "sv": "stop+V -> ɾV", "vsv": "V+stop+V -> VɾV"}
COVERED = (8_916, 0.231)            # the registration's table, 0.40 set

SPLIT_SCHEME = ("capstone.splits, `joint` (span grid §8, 2026-11-21): the pilot's 5x5 grid, "
                "folds assigned over its 10,389 eligible sites; each cell's probe trains on the "
                "transfer stops sharing neither its speaker fold nor its type fold and is scored "
                "on the cell's grid flaps. Out-of-fold predictions pooled over the 25 cells, "
                "10,334 flaps.")


def fixed_layers():
    """Each representation's span-grid best layer per flap span, from the committed
    results; the registered ones must agree with the registration."""
    import pandas as pd

    layers = {}
    for fname in sorted(set(COMMITTED.values())):
        df = pd.read_csv(COMMITTED_DIR / fname)
        for rep in [r for r, f in COMMITTED.items() if f == fname]:
            for span in FLAP_SPANS:
                s = df[(df.representation == rep) & (df.span == span)].sort_values("layer")
                layers[rep, span] = int(s.layer.to_numpy()[s.balanced_acc.to_numpy().argmax()])
                flap_ba = dict(zip(s.layer, s.balanced_acc))
                layers[rep, span, "flap_ba"] = flap_ba
    for rep, want in REGISTERED_LAYERS.items():
        got = tuple(layers[rep, span] for span in FLAP_SPANS)
        if got != want:
            raise SystemExit(f"{rep}: committed best layers {got}, registered {want}")
    return layers


def covered_flaps(d, g):
    """The grid flaps at a position the 0.40 set covers, as the registration's
    table counts them: count_transfer_candidates' flaps inside the §2 class
    context whose (word, position) is a kept position. Its tokens carry the
    track and phone ordinal, so each is located on its .phones tier to find its
    grid site by (track, start), the key collect() matched them on."""
    import numpy as np

    from count_flap_sites import DEFAULT_ROOT, read_phones
    from count_transfer_candidates import MIN_FLAP_SHARE, kept_positions, load_context

    assert MIN_FLAP_SHARE == 0.40
    _, attested, ctx, share, n_flap = load_context()
    keep = kept_positions(share, n_flap, attested, MIN_FLAP_SHARE)
    paths = {p.stem: p for p in DEFAULT_ROOT.rglob("*.phones")}
    tiers, keys = {}, set()
    for t in ctx:
        if t[7] and (t[1], t[2]) in keep:
            track, k = t[8], t[9]
            if track not in tiers:
                tiers[track] = read_phones(paths[track])
            keys.add((track, round(float(tiers[track][k].start), 4)))
    cov = np.array([(d.inv.track[i], round(float(d.inv.start[i]), 4)) in keys for i in g])
    if cov.sum() != len(keys):
        raise SystemExit("a covered flap token did not match a grid site")
    y = d.is_d[g]
    got = (int(cov.sum()), round(float(y[cov].mean()), 3))
    if got != COVERED:
        raise SystemExit(f"covered flaps {got}, the registration's table says {COVERED}")
    return cov


def main() -> int:
    import numpy as np
    import pandas as pd

    from capstone.predictions import build_manifest, write_prediction_columns
    from capstone.probes.scoring import (balanced_accuracy, boot_balanced_accuracy,
                                         interval, resample_weights)
    from capstone.probes.controls import type_index

    run_dir = rp.TRANSFER_RUN_DIR
    out_dir = sg.OUT
    sg.check_against_criterion_1()
    d = rp.SpanData()
    rp.check_conv_features(d)
    st = rp.TransferStops(d)
    blocks = rp.transfer_blocks()
    manifest = build_manifest(
        rp.INVENTORY, split_scheme=SPLIT_SCHEME, seeds=seeds.REGISTERED,
        n_sites_scored=int(d.in_grid.sum()), synthetic=False,
        encoders=tuple(f"{r}_{s}_{t}" for r, s, t in blocks), run_id=f"transfer-{DATE}")
    layers = fixed_layers()
    ctx = sg.setup(d)
    W, g, tix, y = ctx["W"], ctx["g"], ctx["tix"], ctx["y"]
    cov = covered_flaps(d, g)
    cov_types, cov_tix = type_index([d.keys[i] for i in g[cov]])
    W_cov = resample_weights(len(cov_types), seeds.BOOTSTRAP)
    band = sc.frequency_bands(d, g)
    spk_group, type_group = st.flap_fold[g] // 5, st.flap_fold[g] % 5

    def ci_at(boot_col):
        lo, hi = interval(boot_col[:, None])
        return float(lo[0]), float(hi[0])

    rows, group_rows, comps, summary, pred_cols = [], [], [], {}, []
    fits = nonconv = 0
    for rep, span, tset in blocks:
        fspan = rp.TRANSFER_SPANS[span]
        z = np.load(run_dir / rp.transfer_block_name(rep, span, tset))
        if not ((z["fold"] >= 0) == d.in_grid).all() or (z["fold"][g] != st.flap_fold[g]).any():
            raise SystemExit(f"{rep}_{span}_{tset}: scored flaps or folds are not the grid's")
        real = z["real"][:, g]
        if np.isnan(real).any():
            raise SystemExit(f"{rep}_{span}_{tset}: a grid flap has no prediction")
        pred = real > 0.5
        cor = pred == y
        ba = np.array([balanced_accuracy(y, p) for p in pred])
        boot = boot_balanced_accuracy(W, tix, y, cor)
        lo, hi = interval(boot)
        fits += z["real_converged"].size
        nonconv += int((~z["real_converged"]).sum())
        for L in range(len(ba)):
            rows.append({"representation": rep, "stop_span": span, "flap_span": fspan,
                         "set": tset, "layer": L, "n": len(g), "n_d": int(y.sum()),
                         "balanced_acc": ba[L], "ci_lo": lo[L], "ci_hi": hi[L],
                         "recall_t": float((~pred[L][~y]).mean()),
                         "recall_d": float(pred[L][y].mean()),
                         "n_nonconverged": int((~z["real_converged"][L]).sum())})
        L = layers[rep, fspan]
        g_lo, g_hi = ci_at(boot[:, L] - 0.5)
        cid = CALLED.get((rep, span, tset), SPAN_ID[span])
        called = (rep, span, tset) in CALLED
        comps.append({"comparison": cid, "representation": rep, "stop_span": span,
                      "flap_span": fspan, "set": tset, "layer": L, "n": len(g),
                      "gap": float(ba[L] - 0.5), "ci_lo": g_lo, "ci_hi": g_hi,
                      "call": sg.call(ba[L] - 0.5, g_lo) if called else "reported, not called"})
        s = {"layer": L, "ba": float(ba[L]), "lo": float(lo[L]), "hi": float(hi[L]),
             "curve_best": int(np.argmax(ba)), "curve_max": float(ba.max()),
             "recall_t": float((~pred[L][~y]).mean()), "recall_d": float(pred[L][y].mean()),
             "flap_ba": float(layers[rep, fspan, "flap_ba"][L]),
             "n_train": (int(z["n_train"].min()), int(np.median(z["n_train"])),
                         int(z["n_train"].max())),
             "nonconverged": int((~z["real_converged"]).sum()),
             "fits": int(z["real_converged"].size)}
        if tset == "040":
            pc, yc = pred[L][cov], y[cov]
            cb = boot_balanced_accuracy(W_cov, cov_tix, yc, (pc == yc)[None])
            s["covered"] = (float(balanced_accuracy(yc, pc)), *ci_at(cb[:, 0]))
            s["stratified"] = sc.stratified(W, tix, y, band, pred[L])
            for axis, grp in (("speaker", spk_group), ("type", type_group)):
                for k in range(5):
                    m = grp == k
                    group_rows.append({"representation": rep, "stop_span": span,
                                       "flap_span": fspan, "set": tset, "layer": L,
                                       "axis": axis, "group": k, "n": int(m.sum()),
                                       "n_d": int(y[m].sum()),
                                       "balanced_acc": balanced_accuracy(y[m], pred[L][m])})
        summary[rep, span, tset] = s
        pred_cols.append((rep, f"transfer_{span}_{tset}", "joint", False, z["fold"], z["real"]))
    n_called = sum(c["call"] != "reported, not called" for c in comps)
    assert n_called == len(CALLED) == 2

    # ---------------------------------------------------------- write
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"transfer-results-{DATE}.csv", index=False,
                              float_format="%.6f")
    pd.DataFrame(group_rows).to_csv(out_dir / f"transfer-groups-{DATE}.csv", index=False,
                                    float_format="%.6f")
    extra = []
    for (rep, span, tset), s in summary.items():
        if tset != "040":
            continue
        for name, key in (("covered", "covered"), ("stratified", "stratified")):
            p, lo_, hi_ = s[key]
            extra.append({"comparison": name, "representation": rep, "stop_span": span,
                          "flap_span": rp.TRANSFER_SPANS[span], "set": tset,
                          "layer": s["layer"],
                          "n": int(cov.sum()) if key == "covered" else len(g),
                          "gap": p, "ci_lo": lo_, "ci_hi": hi_,
                          "call": "reported, not called (a score, not a gap)"})
    pd.concat([pd.DataFrame(comps), pd.DataFrame(extra)], ignore_index=True).to_csv(
        out_dir / f"transfer-comparisons-{DATE}.csv", index=False, float_format="%.6f")
    write_prediction_columns(pred_cols, d.inv.site_id.to_numpy(), d.in_grid, d.is_d,
                             run_dir / "predictions-transfer.parquet", synthetic=False)
    manifest.write(out_dir / "run-manifest-transfer.json")

    sm = {p: json.loads((p / "extract-manifest.json").read_text())
          for p in (rp.TRANSFER_STOPS_DIR, rp.CONV_STOPS_DIR)}
    lines = [
        "Span grid — the transfer probe: trained on unflapped stops, scored on the flaps "
        "(§8, 2026-11-21; HuBERT and WavLM, 2026-11-25)",
        f"Run {DATE}. `joint`, out-of-fold over 25 cells, {len(g):,} flaps "
        f"({int(y.sum()):,} /d/). Chance is 0.50.",
        "",
        f"code commit       {manifest.code_commit}",
        f"stops             {rp.TRANSFER_STOPS_TSV} (sha256 {st.tsv_sha256[:12]}); "
        + ", ".join(f"{t}: {int(m.sum()):,} ({int(st.is_d[m].sum()):,} /d/)"
                    for t, m in st.sets.items())
        + f"; {int(st.clamped.sum())} clamped stop left out",
        "features          " + "; ".join(f"{p} (extracted at {m['code_commit']})"
                                         for p, m in sm.items()),
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"fits              {fits:,}, {nonconv} did not converge (none refit)",
        "folds             the flaps' re-derived joint folds equal iter_splits'; every grid "
        "flap scored once",
        f"covered flaps     {int(cov.sum()):,} ({float(y[cov].mean()):.1%} /d/), as registered",
        "",
        "Per block, at the fixed layer (95% word-type bootstrap interval)",
        "-----------------------------------------------------------------",
        f"{'block':<24}{'layer':>5}  {'bal. acc.':>9}  {'95% CI':>17}  {'flap-trained':>12}  "
        f"{'rec t':>5}  {'rec d':>5}  {'curve max (layer)':>17}  train min/med/max",
    ]
    for (rep, span, tset), s in summary.items():
        lines.append(
            f"{rep + '_' + span + '_' + tset:<24}{s['layer']:>5}  {s['ba']:>9.4f}  "
            f"[{s['lo']:.4f}, {s['hi']:.4f}]  {s['flap_ba']:>12.4f}  {s['recall_t']:>5.3f}  "
            f"{s['recall_d']:>5.3f}  {s['curve_max']:>10.4f} ({s['curve_best']:>2})  "
            + "/".join(str(x) for x in s["n_train"]))
    lines += ["", "Registered: X1, X2 (§6: gap >= 0.05 and 95% lower bound > 0)",
              "-------------------------------------------------------------"]
    for c in comps:
        if c["call"] != "reported, not called":
            lines.append(f"{c['comparison']:<3} {c['representation']}, {SPAN_NAME[c['stop_span']]}, "
                         f"layer {c['layer']}, 0.40 set: {c['gap']:+.4f}  "
                         f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['call']}")
    lines += ["", "Reported, not called: transfer balanced accuracy - 0.5 at the fixed layer",
              "--------------------------------------------------------------------------"]
    for c in comps:
        if c["call"] == "reported, not called":
            lines.append(f"{c['comparison']:<9} {c['representation']:<13} {c['set']}  "
                         f"{SPAN_NAME[c['stop_span']]:<17} layer {c['layer']:>2}  "
                         f"{c['gap']:+.4f}  [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")
    lines += ["", "0.40 set, at the fixed layer: covered flaps only (8,916) and the frequency "
              "re-score's stratified score (three lower bands, site-weighted); reported",
              f"{'block':<24}{'covered':>8}  {'95% CI':>17}  {'stratified':>10}  {'95% CI':>17}"]
    for (rep, span, tset), s in summary.items():
        if tset == "040":
            c, st_ = s["covered"], s["stratified"]
            lines.append(f"{rep + '_' + span + '_' + tset:<24}{c[0]:>8.4f}  "
                         f"[{c[1]:.4f}, {c[2]:.4f}]  {st_[0]:>10.4f}  [{st_[1]:.4f}, {st_[2]:.4f}]")
    lines += ["", "Speaker and word-type groups: transfer-groups-*.csv.",
              "",
              "What the readings predict (§8, 2026-11-21): a plain lexical shortcut, no transfer; "
              "a phonetic residual of the contrast, transfer mainly in X2; top-down, transfer on "
              "X1 as well as X2. Transfer separates a shortcut from the other two, not those two "
              "from each other. A failed transfer is weak evidence against a trace; a successful "
              "one is strong evidence for one.",
              "The raw-input variants are at their own span-grid best layers (not registered; "
              "reported only).",
              f"No multiple-comparison correction is registered; {n_called} gaps are called."]
    text = "\n".join(lines) + "\n"
    (out_dir / f"transfer-summary-{DATE}.txt").write_text(text)
    print(text)
    print(f"wrote {out_dir}/transfer-*-{DATE}.* and {run_dir / 'predictions-transfer.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
