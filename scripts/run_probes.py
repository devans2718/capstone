#!/usr/bin/env python3
"""The pilot's probe run: every registered probe, then the §5 criteria.

HOW TO RUN (in .venv-app, after scripts/extract_site_features.py)
    .venv-app/bin/python scripts/run_probes.py gate       # criteria first
    .venv-app/bin/python scripts/run_probes.py rest       # everything else
    .venv-app/bin/python scripts/run_probes.py assemble   # CSVs, parquet, manifest

    The span grid (experiments/span-grid/README.md), pooled design only:
    .venv-app/bin/python scripts/run_probes.py span-repro  # §5's reproduction check
    .venv-app/bin/python scripts/run_probes.py span-grid   # the 10,334; then
    .venv-app/bin/python scripts/score_span_grid.py        # scores and C1-C7

    HuBERT, WavLM and the raw-input variants (span grid §8, 2026-11-24):
    .venv-app/bin/python scripts/run_probes.py span-grid-conv
    .venv-app/bin/python scripts/score_conv_encoders.py    # E1, E2, C6

    The context-only span and the preceding stretch without the flap (span grid
    §8, 2026-11-23; HuBERT and WavLM's, 2026-11-24):
    .venv-app/bin/python scripts/run_probes.py span-grid-context
    .venv-app/bin/python scripts/score_context.py          # K1-K3

    The transfer probe, trained on unflapped stops and scored on the flaps
    (span grid §8, 2026-11-21; HuBERT and WavLM's, 2026-11-25):
    .venv-app/bin/python scripts/run_probes.py span-grid-transfer
    .venv-app/bin/python scripts/score_transfer.py         # X1, X2

    The cropped design, the padding-only control and the flap-masked variant
    (span grid §3; §8, 2026-11-26), after scripts/extract_cropped.py:
    .venv-app/bin/python scripts/run_probes.py span-grid-cropped
    .venv-app/bin/python scripts/score_cropped.py          # C2, C3; C4 C5 C7 cropped
    .venv-app/bin/python scripts/score_masked.py           # M1, M2

    gate fits only what the criteria read: mean pooling, `joint`, WavJEPA and
    MFCC, real and control probes, and the lookup. It then writes
    results/flapping/criteria-20260926.txt. The result being waited on is not
    queued behind analyses that gate nothing. rest fits the other three
    conditions, the §3 subsample, max and midpoint pooling, and the
    speaker-ID probe. Each block of fits is cached under the run directory
    and skipped if present, so a stage can be re-run after an interruption
    without refitting. Nothing takes a seed or a hyperparameter as an
    argument: seeds come from capstone.seeds and the probe from
    capstone.probes.linear.

WHAT IS FIT
    For each representation (WavJEPA layers 0-12, MFCC) x pooling x
    condition x fold, taken from capstone.splits.iter_splits over the
    10,389 eligible sites:
      - the real probe (/t/ vs /d/)
      - the control probe, on labels from the §4 mapping (seed 20260927)
      - the speaker-ID probe, under `random` and `word_disjoint` only (§4),
        on mean pooling in the headline run. It is a diagnostic, and the
        pooling comparison is registered for the real probe.
    and, once per condition, the word-identity lookup.

    A fit that does not converge is counted and reported, never refit (§3).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capstone import seeds  # noqa: E402
from capstone.splits import CONDITIONS, iter_splits, type_key  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "flapping" / "sites-20260924.parquet"
LEMMA_TSV = REPO / "results" / "flapping" / "lemma-overlap-joint-20260926.tsv"
FEATURES = Path("/data/embeddings/features-b378c303ca90")
RUN_DIR = Path("/data/embeddings/probe-run-20260926")
OUT = REPO / "results" / "flapping"
DATE = "20260926"

REPRESENTATIONS = ("wavjepa", "mfcc")
POOLINGS = ("mean", "max", "midpoint")
SPEAKER_CONDITIONS = ("random", "word_disjoint")

# The span grid (experiments/span-grid/README.md): pooled design only (§8,
# 2026-10-29). A span takes the place of the pilot's pooling: span s for
# representation rep is SPANS_DIR/{rep}_{s}.npy, in the pilot's row layout.
SPANS_DIR = Path("/data/embeddings/spans-b378c303ca90")
SPAN_DATE = "20261029"
SPAN_RUN_DIR = Path(f"/data/embeddings/span-run-{SPAN_DATE}")
SPANS = ("r", "vr", "rv", "vrv")
SPAN_REPRESENTATIONS = ("wavjepa", "wav2vec2", "mfcc")
SPAN_REVISIONS = {"wavjepa": "6be4a50", "wav2vec2": "0b5b8e8"}   # span grid §4
# The grid proper: the 10,334 (§1). Its ɾ is a second fit of the span, so it
# lives apart from the reproduction check's block in SPAN_RUN_DIR.
GRID_DIR = SPAN_RUN_DIR / "grid"
GRID_FEATURES = GRID_DIR / "features"
# §5's duration baselines, from spans.tsv: one column each, then the clipped
# duration appended to the span's MFCC for Vɾ and VɾV (§8, 2026-10-29).
DURATIONS = {"clipped": "stretch_clipped_ms", "full": "stretch_ms", "vowel": "vowel_ms"}
MFCC_DURATION_SPANS = ("vr", "vrv")
# HuBERT and WavLM, normalized, and the raw-input variant of all three
# (§8, 2026-11-24): scripts/extract_conv_encoders.py. Their blocks join the
# grid's in GRID_DIR, under names of their own.
CONV_SPANS_DIR = Path("/data/embeddings/conv-spans-b378c303ca90")
CONV_REPRESENTATIONS = ("hubert", "wavlm", "wav2vec2_raw", "hubert_raw", "wavlm_raw")
CONV_REVISIONS = {"hubert": "dba3bb0", "wavlm": "efa81aa", "wav2vec2": "0b5b8e8"}
# The context-only span (ctx) and the preceding stretch without the flap (v)
# (§8, 2026-11-23): scripts/extract_transfer_context.py, for the grid's three
# representations. HuBERT, WavLM and the raw variants have theirs in
# CONV_SPANS_DIR (§8, 2026-11-24). Blocks join the grid's in GRID_DIR.
CONTEXT_DIR = Path("/data/embeddings/context-b378c303ca90")
CONTEXT_SPANS = ("ctx", "v")
# The cropped design, the padding-only control and the flap-masked variant
# (§3; §8, 2026-11-26): scripts/extract_cropped.py, all four encoders and MFCC.
# A "pool" takes the place of the span: crop_<span>, pad_<span>, pad_ctx,
# mask_ctx. Blocks join the grid's in GRID_DIR under those names.
CROPPED_DIR = Path("/data/embeddings/cropped-b378c303ca90")
CROPPED_REPRESENTATIONS = ("wavjepa", "wav2vec2", "hubert", "wavlm", "mfcc")
CROPPED_POOLS = (*(f"crop_{s}" for s in SPANS), *(f"pad_{s}" for s in SPANS),
                 "pad_ctx", "mask_ctx")
CROPPED_REVISIONS = {"wavjepa": "6be4a50", "wav2vec2": "0b5b8e8", "hubert": "dba3bb0",
                     "wavlm": "efa81aa"}
CROPPED_L_MS = 533.875          # 8,542 samples; §3's 533.9
# The transfer probe (§8, 2026-11-21; HuBERT and WavLM, 2026-11-24 and
# 2026-11-25): /t/ vs /d/ trained on unflapped stops, scored on the grid's flaps.
# scripts/export_transfer_stops.py wrote the stops (the 0.25 set, each row
# flagged for 0.25/0.40/0.50); extract_transfer_context.py encoded them for the
# grid's three representations, extract_conv_encoders.py for the rest.
TRANSFER_STOPS_TSV = Path("/data/span-grid/transfer-stops.tsv")
TRANSFER_STOPS_DIR = Path("/data/embeddings/transfer-stops-f81597b4ced8")
CONV_STOPS_DIR = Path("/data/embeddings/conv-stops-f81597b4ced8")
TRANSFER_SPANS = {"s": "r", "vs": "vr", "sv": "rv", "vsv": "vrv"}   # stop span -> flap span
# The registration's table: set -> (stops, /d/, word types). 040 is registered,
# 025 and 050 are the sensitivity sets.
TRANSFER_SETS = {"040": (2_247, 1_011, 252), "025": (2_433, 1_162, 303),
                 "050": (1_749, 746, 220)}
# The sensitivity sets are fit on the spans X1 and X2 are called on.
TRANSFER_SENSITIVITY_SPANS = ("s", "vs")
TRANSFER_RUN_DIR = GRID_DIR / "transfer"

assert seeds.FOLDS == seeds.SUBSAMPLE  # iter_splits takes one seed for both


# ------------------------------------------------------------------- the data


class Data:
    """The inventory, the eligible set and every per-site array the run needs."""

    def __init__(self):
        import numpy as np
        import pyarrow.parquet as pq

        from capstone.probes.controls import control_mapping, d_share, type_index

        inv = pq.read_table(INVENTORY).to_pandas()
        ids = (FEATURES / "site_ids.txt").read_text().split()
        if ids != list(inv.site_id):
            raise SystemExit(f"{FEATURES}/site_ids.txt is not in inventory order")
        fm = json.loads((FEATURES / "extract-manifest.json").read_text())
        if fm.get("limit"):
            raise SystemExit(f"{FEATURES} is a --limit smoke test, not a full extraction")

        self.inv = inv
        self.n = len(inv)
        self.features_manifest = fm
        self.eligible = ~inv.excerpt_clamped.to_numpy()
        self.elig_idx = np.flatnonzero(self.eligible)
        self.is_d = (inv.underlying == "d").to_numpy()
        self.keys = [type_key(w) for w in inv.word]

        e = self.elig_idx
        self.types, tix = type_index([self.keys[i] for i in e])
        self.type_idx = np.full(self.n, -1, dtype=np.int64)
        self.type_idx[e] = tix
        self.share = d_share(self.is_d[e])

        assert len(e) == 10_389 and int(self.is_d[e].sum()) == 2_529, "§3 pool changed"
        assert len(self.types) == 991, f"{len(self.types)} type keys, §3 says 991"

        mapping = control_mapping(len(self.types), self.share, seeds.CONTROL)
        self.control_mapping = mapping
        self.control_is_d = np.zeros(self.n, dtype=bool)
        self.control_is_d[e] = mapping[tix]

        spk = sorted(set(inv.speaker))
        self.speaker_code = np.array([spk.index(s) for s in inv.speaker])

    def splits(self, condition, subsampled):
        seed = seeds.FOLDS if (condition == "random" or subsampled) else None
        out = list(iter_splits(condition, self.inv.speaker, self.inv.word,
                               self.inv.underlying, eligible=self.eligible,
                               seed=seed, subsample=subsampled))
        # Every eligible site in exactly one test set; no ineligible site anywhere.
        import numpy as np

        seen = np.zeros(self.n, dtype=np.int64)
        for s in out:
            seen[s.test] += 1
            assert self.eligible[s.train].all()
        assert (seen[self.eligible] == 1).all() and (seen[~self.eligible] == 0).all()
        return out


def features_path(rep, pooling, features=FEATURES):
    return features / f"{rep}_{pooling}.npy"


def block_name(kind, rep, pooling, condition, subsampled):
    return f"{kind}_{rep}_{pooling}_{condition}{'_sub' if subsampled else ''}.npz"


# --------------------------------------------------------------------- fitting


def _fit(path, layer, train, test, y_train):
    import numpy as np

    from capstone.probes.linear import fit_predict

    X = np.load(path, mmap_mode="r")
    return fit_predict(np.asarray(X[train, layer]), y_train, np.asarray(X[test, layer]))


def run_block(d: Data, kind, rep, pooling, condition, subsampled, n_jobs,
              features=FEATURES, run_dir=RUN_DIR, sites=None):
    """Fit one block and cache it. kind: 'probe' (real + control) or 'speaker'.

    The span grid passes its own features and run directories, with a span in
    place of the pooling; the pilot's stages take the defaults. `sites`, a bool
    mask, drops every other site from each cell's train and test sets after the
    folds are assigned (span grid §5): the grid's 55 exclusions. Unscored sites
    keep fold -1 and NaN predictions.
    """
    import numpy as np
    from joblib import Parallel, delayed

    out = run_dir / block_name(kind, rep, pooling, condition, subsampled)
    if out.exists():
        return out
    path = features_path(rep, pooling, features)
    n_layers = np.load(path, mmap_mode="r").shape[1]
    splits = d.splits(condition, subsampled)
    if sites is not None:
        splits = [replace(s, train=s.train[sites[s.train]], test=s.test[sites[s.test]])
                  for s in splits]

    targets = {"probe": {"real": d.is_d, "control": d.control_is_d},
               "speaker": {"speaker": d.speaker_code}}[kind]
    jobs = [(t, L, k) for t in targets for L in range(n_layers) for k in range(len(splits))]
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit)(path, L, splits[k].train, splits[k].test, targets[t][splits[k].train])
        for t, L, k in jobs)

    fold = np.full(d.n, -1, dtype=np.int32)
    for s in splits:
        fold[s.test] = s.fold
    arrays = {"fold": fold}
    for t in targets:
        dtype = np.float64 if kind == "probe" else np.int64
        fill = np.nan if kind == "probe" else -1
        arrays[t] = np.full((n_layers, d.n), fill, dtype=dtype)
        arrays[f"{t}_converged"] = np.zeros((n_layers, len(splits)), dtype=bool)
    for (t, L, k), (pred, conv) in zip(jobs, results):
        arrays[t][L, splits[k].test] = pred
        arrays[f"{t}_converged"][L, k] = conv

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    nc = sum(int((~arrays[f"{t}_converged"]).sum()) for t in targets)
    print(f"  {out.name}: {len(jobs)} fits, {nc} did not converge", flush=True)
    return out


def run_lookup(d: Data, condition, subsampled):
    import numpy as np

    from capstone.probes.controls import lookup_predict

    out = RUN_DIR / f"lookup_{condition}{'_sub' if subsampled else ''}.npz"
    if out.exists():
        return out
    pred = np.zeros(d.n, dtype=bool)
    fold = np.full(d.n, -1, dtype=np.int32)
    for s in d.splits(condition, subsampled):
        pred[s.test] = lookup_predict(d.type_idx[s.train], d.is_d[s.train], d.type_idx[s.test])
        fold[s.test] = s.fold
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, pred_d=pred, fold=fold)
    return out


# -------------------------------------------------------------------- criteria


def gate(n_jobs) -> int:
    import numpy as np

    from capstone.predictions import code_commit
    from capstone.probes import criteria as C
    from capstone.probes.controls import null_mappings, type_index
    from capstone.probes.scoring import (balanced_accuracy, f1_d,
                                         resample_weights)

    d = Data()
    print("gate: joint, mean pooling, WavJEPA and MFCC", flush=True)
    for rep in REPRESENTATIONS:
        run_block(d, "probe", rep, "mean", "joint", False, n_jobs)
    run_lookup(d, "joint", False)

    e = d.elig_idx
    y = d.is_d[e]
    w = np.load(RUN_DIR / block_name("probe", "wavjepa", "mean", "joint", False))
    m = np.load(RUN_DIR / block_name("probe", "mfcc", "mean", "joint", False))
    pred_w = w["real"][:, e] > 0.5                   # (13, n)
    pred_m = m["real"][0, e] > 0.5
    ctrl_w = w["control"][:, e] > 0.5
    lookup = np.load(RUN_DIR / "lookup_joint.npz")["pred_d"][e]

    W = resample_weights(len(d.types), seeds.BOOTSTRAP)
    tix = d.type_idx[e]
    c1, n1 = C.criterion_1(W, tix, y, pred_w == y, pred_m == y)
    nulls = null_mappings(len(d.types), d.share, seeds.CONTROL_NULL)
    c2, n2 = C.criterion_2(ctrl_w, tix, d.control_mapping, nulls)
    ba_layers = [balanced_accuracy(y, p) for p in pred_w]
    c3, _ = C.criterion_3(ba_layers)
    best = n1["best_layer"]
    c4, _ = C.criterion_4(f1_d(y, pred_w[best]), best)

    # Criterion 1 re-scored on the fixed lemma-unseen subset (§5): reported,
    # not gated. Same predictions, its own types, the same bootstrap seed.
    import pandas as pd

    lem = pd.read_csv(LEMMA_TSV, sep="\t")
    unseen = set(lem.site_id[lem.lemma_in_train.astype(str).str.lower() == "false"])
    sub = np.array([sid in unseen for sid in d.inv.site_id.to_numpy()[e]])
    assert sub.sum() == 7_586 and int(y[sub].sum()) == 1_735, "the §5 subset moved"
    sub_types, sub_tix = type_index([d.keys[i] for i in e[sub]])
    W_sub = resample_weights(len(sub_types), seeds.BOOTSTRAP)
    c1_sub, n1_sub = C.criterion_1(W_sub, sub_tix, y[sub],
                                   (pred_w == y)[:, sub], (pred_m == y)[sub])

    nc = {rep: int((~np.load(RUN_DIR / block_name("probe", rep, "mean", "joint", False))
                    ["real_converged"]).sum()) for rep in REPRESENTATIONS}
    ctrl_nc = int((~w["control_converged"]).sum())

    lines = [
        "Pilot: English flapping probe — §5 criteria, revision 3",
        f"Run {DATE}. Scored on `joint`, mean pooling, headline run, pooled over "
        f"{len(e):,} sites ({int(y.sum()):,} /d/). Chance is 0.50 balanced accuracy.",
        "",
        f"code commit       {code_commit()}",
        f"features          {FEATURES} (extracted at {d.features_manifest['code_commit']})",
        f"inventory sha256  {d.features_manifest['inventory_sha256']}",
        f"checkpoint        {d.features_manifest['checkpoint_revision']}",
        "seeds             " + ", ".join(f"{k} {v}" for k, v in seeds.REGISTERED.items()),
        f"non-converged     WavJEPA real {nc['wavjepa']}/325, control {ctrl_nc}/325; "
        f"MFCC real {nc['mfcc']}/25",
        "",
    ]
    for c in (c1, c2, c3, c4):
        lines += [f"Criterion {c.number} — {c.name}: {'PASS' if c.passed else 'FAIL'}",
                  f"  {c.detail}", ""]
    verdict = all(c.passed for c in (c1, c2, c3, c4))
    lines += [f"PILOT: {'MET — all four criteria pass' if verdict else 'FAILED — see §5 for how to read a failure'}",
              "",
              "Reported, not gated",
              "-------------------",
              f"WavJEPA balanced accuracy by layer (joint, mean):",
              "  " + "  ".join(f"{L}:{b:.4f}" for L, b in enumerate(ba_layers)),
              f"Control balanced accuracy by layer:",
              "  " + "  ".join(f"{L}:{b:.4f}" for L, b in enumerate(n2['control_ba_by_layer'])),
              f"Word-identity lookup under joint: {balanced_accuracy(y, lookup):.4f}",
              f"Criterion 1 on the {int(sub.sum()):,} lemma-unseen sites "
              f"({int(y[sub].sum()):,} /d/, {len(sub_types)} type keys): {c1_sub.detail}",
              f"  (would {'pass' if c1_sub.passed else 'not pass'} criterion 1's thresholds; "
              "this does not change whether criterion 1 passes)",
              ""]
    text = "\n".join(lines)
    path = OUT / f"criteria-{DATE}.txt"
    path.write_text(text)
    (RUN_DIR / "criteria.json").write_text(json.dumps(
        {"c1": n1, "c2": n2, "c1_lemma_unseen": n1_sub, "ba_layers": ba_layers,
         "passed": {c.number: c.passed for c in (c1, c2, c3, c4)}}, indent=2))
    print("\n" + text)
    print(f"wrote {path}")
    return 0


# ------------------------------------------------------------------ span grid


class SpanData(Data):
    """Data, plus the span extraction's checks, made before any span is read.

    At least what the pilot checks of its features (inventory order, a full
    extraction), and that the spans are the registered extraction: the same
    inventory and exclusion list, the pinned revisions, float16 frames, ɾ equal
    to the pilot's features, and NaN exactly where no site is scored.
    """

    def __init__(self):
        import csv

        import numpy as np

        from measure_span_grid import EXPECTED_SHA256, exclusion_digest

        super().__init__()
        ids = (SPANS_DIR / "site_ids.txt").read_text().split()
        if ids != list(self.inv.site_id):
            raise SystemExit(f"{SPANS_DIR}/site_ids.txt is not in inventory order")
        sm = json.loads((SPANS_DIR / "extract-manifest.json").read_text())
        problems = []
        if sm.get("limit"):
            problems.append("a --limit smoke test, not a full extraction")
        if sm["inventory_sha256"] != self.features_manifest["inventory_sha256"]:
            problems.append("a different inventory from the pilot's features")
        if sm["exclusion_sha256"] != EXPECTED_SHA256:
            problems.append("an exclusion list other than the registered one (§1)")
        if sm["checkpoint_revisions"] != SPAN_REVISIONS:
            problems.append(f"revisions {sm['checkpoint_revisions']}, §4 pins {SPAN_REVISIONS}")
        if sm["frame_dtype"] != "float16" or sm["pooling"] != "mean":
            problems.append("not float16 frames mean-pooled (§4, §8)")
        for rep, r in sm["r_vs_pilot"].items():
            if r["rows_identical"] != r["rows"] or r["max_abs_diff"] != 0:
                problems.append(f"ɾ for {rep} differs from the pilot's features")

        with (SPANS_DIR / "spans.tsv").open(newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if [r["site_id"] for r in rows] != list(self.inv.site_id.to_numpy()[self.elig_idx]):
            problems.append("spans.tsv is not the eligible sites in inventory order")
        out = [(r["site_id"], None) for r in rows if r["in_grid"] == "0"]
        if exclusion_digest(out) != EXPECTED_SHA256:
            problems.append("spans.tsv's excluded sites are not the registered 55")
        if problems:
            raise SystemExit(f"{SPANS_DIR}: " + "; ".join(problems))

        # The grid's sites (§1): eligible, less the 55. Folds stay the pilot's,
        # assigned over the 10,389; the 55 leave after assignment (§5).
        self.in_grid = np.zeros(self.n, dtype=bool)
        self.in_grid[self.elig_idx] = [r["in_grid"] == "1" for r in rows]
        assert self.in_grid.sum() == 10_334, "§1's 10,334 changed"
        self.spans_manifest = sm

        def column(name):
            a = np.full(self.n, np.nan)
            a[self.elig_idx] = [float(r[name]) if r[name] else np.nan for r in rows]
            return a

        self.durations = {k: column(c) for k, c in DURATIONS.items()}
        self.clipped = np.zeros(self.n, dtype=bool)
        self.clipped[self.elig_idx] = [r["clipped_pre"] == "1" or r["clipped_post"] == "1"
                                       for r in rows]
        self.clipped_post = np.zeros(self.n, dtype=bool)
        self.clipped_post[self.elig_idx] = [r["clipped_post"] == "1" for r in rows]
        self.spans_rows = rows
        for k, a in self.durations.items():
            if (np.isnan(a) != ~self.in_grid).any():
                raise SystemExit(f"spans.tsv: {DURATIONS[k]} is not set for exactly the grid's sites")

        # NaN exactly where no site is scored: the clamped for ɾ, and the
        # clamped and the 55 for the other spans.
        for span in SPANS:
            scored = self.eligible if span == "r" else self.in_grid
            for rep in SPAN_REPRESENTATIONS:
                X = np.load(features_path(rep, span, SPANS_DIR), mmap_mode="r")
                if X.shape[0] != self.n:
                    raise SystemExit(f"{rep}_{span}.npy has {X.shape[0]} rows, not {self.n}")
                nan = np.isnan(X[:, 0, 0])
                if (nan != ~scored).any():
                    raise SystemExit(f"{rep}_{span}.npy: NaN rows are not the unscored sites")


def span_repro(n_jobs) -> int:
    """§5's reproduction check: the pooled design's ɾ for WavJEPA, on the pilot's
    10,389 sites and folds, against the pilot's `joint` mean-pooled curve.

    The ɾ features are the pilot's bit for bit, so the curve has to match
    exactly: any difference in an out-of-fold prediction fails the check, and
    nothing else in the span grid is read. Aggregates only are printed and
    written (span-run dir, /data); no site ID leaves it.
    """
    import time

    import numpy as np

    from capstone.predictions import code_commit
    from capstone.probes.scoring import balanced_accuracy

    d = SpanData()
    e = d.elig_idx
    y = d.is_d[e]
    print("span-repro: WavJEPA ɾ, joint, the pilot's 10,389 sites", flush=True)
    t0 = time.perf_counter()
    path = run_block(d, "probe", "wavjepa", "r", "joint", False, n_jobs,
                     features=SPANS_DIR, run_dir=SPAN_RUN_DIR)
    seconds = time.perf_counter() - t0

    new = np.load(path)
    old = np.load(RUN_DIR / block_name("probe", "wavjepa", "mean", "joint", False))
    rows, worst = [], {}
    for t in ("real", "control"):
        if not np.array_equal(new["fold"], old["fold"]):
            raise SystemExit("the folds differ from the pilot's")
        a, b = new[t][:, e], old[t][:, e]
        ba_new = [balanced_accuracy(y if t == "real" else d.control_is_d[e], p > 0.5) for p in a]
        ba_old = [balanced_accuracy(y if t == "real" else d.control_is_d[e], p > 0.5) for p in b]
        worst[t] = {"max_abs_p_d": float(np.abs(a - b).max()),
                    "max_abs_ba": float(max(abs(x - z) for x, z in zip(ba_new, ba_old)))}
        rows += [{"target": t, "layer": L, "span_r": x, "pilot_mean": z}
                 for L, (x, z) in enumerate(zip(ba_new, ba_old))]
    passed = all(w["max_abs_p_d"] == 0 and w["max_abs_ba"] == 0 for w in worst.values())

    report = {"code_commit": code_commit(), "spans_dir": str(SPANS_DIR),
              "extracted_at": d.spans_manifest["code_commit"],
              "fit_seconds": seconds, "cached": seconds < 1.0,
              "n_sites": len(e), "n_d": int(y.sum()), "max_abs_diff": worst,
              "passed": passed, "by_layer": rows}
    (SPAN_RUN_DIR / "repro.json").write_text(json.dumps(report, indent=2))
    print("layer  span ɾ   pilot mean   (real probe, balanced accuracy)")
    for r in rows:
        if r["target"] == "real":
            print(f"{r['layer']:>5}  {r['span_r']:.6f} {r['pilot_mean']:.6f}")
    print(f"max |diff|: {worst}")
    print(f"fit time: {seconds:.0f} s{' (cached block; not a timing)' if report['cached'] else ''}")
    print(f"REPRODUCTION: {'PASS' if passed else 'FAIL — nothing else in the span grid is read'}")
    print(f"wrote {SPAN_RUN_DIR / 'repro.json'}")
    return 0 if passed else 1


def build_baselines(d: SpanData):
    """§5's duration baselines as feature arrays, in the spans' row layout.

    duration_{clipped,full,vowel}.npy (n, 1, 1), and mfcc_duration_{vr,vrv}.npy
    (n, 1, 40): the span's 39 MFCC dimensions and the clipped duration. NaN where
    the grid scores no site. Under /data with the rest of the run.
    """
    import numpy as np

    GRID_FEATURES.mkdir(parents=True, exist_ok=True)
    for k, a in d.durations.items():
        np.save(features_path("duration", k, GRID_FEATURES), a.astype(np.float32)[:, None, None])
    clipped = d.durations["clipped"].astype(np.float32)[:, None, None]
    for span in MFCC_DURATION_SPANS:
        m = np.load(features_path("mfcc", span, SPANS_DIR))
        np.save(features_path("mfcc_duration", span, GRID_FEATURES),
                np.concatenate([m, clipped], axis=2))


def span_grid(n_jobs) -> int:
    """The pooled design's probes on the 10,334, `joint` only (span grid §5).

    Every span x representation, then the duration baselines, each a real and a
    control probe in every cell. Blocks are cached under GRID_DIR, so the stage
    resumes after an interruption. Scoring is scripts/score_span_grid.py.
    """
    import time

    d = SpanData()
    build_baselines(d)
    blocks = [(rep, span, SPANS_DIR) for rep in SPAN_REPRESENTATIONS for span in SPANS]
    blocks += [("duration", k, GRID_FEATURES) for k in DURATIONS]
    blocks += [("mfcc_duration", span, GRID_FEATURES) for span in MFCC_DURATION_SPANS]
    print(f"span-grid: {len(blocks)} blocks on {int(d.in_grid.sum()):,} sites -> {GRID_DIR}",
          flush=True)
    t0 = time.perf_counter()
    timing = {}
    for rep, span, features in blocks:
        t = time.perf_counter()
        run_block(d, "probe", rep, span, "joint", False, n_jobs,
                  features=features, run_dir=GRID_DIR, sites=d.in_grid)
        timing[f"{rep}_{span}"] = time.perf_counter() - t
        print(f"    {timing[f'{rep}_{span}']:.0f} s; {(time.perf_counter() - t0) / 60:.1f} min "
              "so far", flush=True)
    (GRID_DIR / "timing.json").write_text(json.dumps(
        {"blocks_seconds": timing, "total_seconds": time.perf_counter() - t0}, indent=2))
    return 0


def check_conv_features(d: SpanData):
    """The conv-encoder features are a full extraction on the grid's own rows.

    Their extraction's wav2vec2 check must have matched the stored span grid on
    every row, and each file must be NaN exactly where SpanData's are.
    """
    import numpy as np

    from measure_span_grid import EXPECTED_SHA256

    if (CONV_SPANS_DIR / "site_ids.txt").read_text().split() != list(d.inv.site_id):
        raise SystemExit(f"{CONV_SPANS_DIR}/site_ids.txt is not in inventory order")
    m = json.loads((CONV_SPANS_DIR / "extract-manifest.json").read_text())
    problems = []
    if m.get("limit"):
        problems.append("a --limit smoke test, not a full extraction")
    if m["inventory_sha256"] != d.spans_manifest["inventory_sha256"]:
        problems.append("a different inventory from the span grid's")
    if m["exclusion_sha256"] != EXPECTED_SHA256:
        problems.append("an exclusion list other than the registered one (§1)")
    if m["checkpoint_revisions"] != CONV_REVISIONS:
        problems.append(f"revisions {m['checkpoint_revisions']}, §8 pins {CONV_REVISIONS}")
    for span, r in m["check_vs_stored"].items():
        if r["rows_identical"] != r["rows"]:
            problems.append(f"the wav2vec2 check failed on {span}")
    if problems:
        raise SystemExit(f"{CONV_SPANS_DIR}: " + "; ".join(problems))
    for span in SPANS:
        scored = d.eligible if span == "r" else d.in_grid
        for rep in CONV_REPRESENTATIONS:
            X = np.load(features_path(rep, span, CONV_SPANS_DIR), mmap_mode="r")
            if X.shape[0] != d.n or (np.isnan(X[:, 0, 0]) != ~scored).any():
                raise SystemExit(f"{rep}_{span}.npy: NaN rows are not the unscored sites")
    return m


def span_grid_conv(n_jobs) -> int:
    """HuBERT, WavLM and the raw-input variants on the grid's four spans (§8,
    2026-11-24): the same sites, folds and probes as span_grid, cached beside
    its blocks in GRID_DIR. Scoring is scripts/score_conv_encoders.py."""
    import time

    d = SpanData()
    check_conv_features(d)
    blocks = [(rep, span) for rep in CONV_REPRESENTATIONS for span in SPANS]
    print(f"span-grid-conv: {len(blocks)} blocks on {int(d.in_grid.sum()):,} sites -> "
          f"{GRID_DIR}", flush=True)
    t0 = time.perf_counter()
    timing = {}
    for rep, span in blocks:
        t = time.perf_counter()
        run_block(d, "probe", rep, span, "joint", False, n_jobs,
                  features=CONV_SPANS_DIR, run_dir=GRID_DIR, sites=d.in_grid)
        timing[f"{rep}_{span}"] = time.perf_counter() - t
        print(f"    {timing[f'{rep}_{span}']:.0f} s; {(time.perf_counter() - t0) / 60:.1f} min "
              "so far", flush=True)
    (GRID_DIR / "timing-conv.json").write_text(json.dumps(
        {"blocks_seconds": timing, "total_seconds": time.perf_counter() - t0}, indent=2))
    return 0


def check_context_features(d: SpanData):
    """The context extraction is a full one on the grid's rows: the registered
    inventory, exclusions and revisions, its re-pooled ɾ equal to the stored span
    grid on every row, and NaN on every row the grid does not score.

    Returns, per (rep, span), the sites to fit: the grid's, less any row whose
    span is empty. Only WavJEPA's preceding stretch has such rows (the
    extraction's v_empty: 6 sites, §8 2026-11-23's manifest); they are left out of
    that block alone and reported by the scorer.
    """
    import numpy as np

    from measure_span_grid import EXPECTED_SHA256

    if (CONTEXT_DIR / "site_ids.txt").read_text().split() != list(d.inv.site_id):
        raise SystemExit(f"{CONTEXT_DIR}/site_ids.txt is not in inventory order")
    m = json.loads((CONTEXT_DIR / "extract-manifest.json").read_text())
    problems = []
    if m.get("limit"):
        problems.append("a --limit smoke test, not a full extraction")
    if m["inventory_sha256"] != d.spans_manifest["inventory_sha256"]:
        problems.append("a different inventory from the span grid's")
    if m["exclusion_sha256"] != EXPECTED_SHA256:
        problems.append("an exclusion list other than the registered one (§1)")
    if m["checkpoint_revisions"] != SPAN_REVISIONS:
        problems.append(f"revisions {m['checkpoint_revisions']}, §4 pins {SPAN_REVISIONS}")
    if m["frame_dtype"] != "float16" or m["pooling"] != "mean":
        problems.append("not float16 frames mean-pooled (§4, §8)")
    for rep, r in m["r_vs_stored"].items():
        if r["rows_identical"] != r["rows"] or r["max_abs_diff"] != 0:
            problems.append(f"the re-pooled ɾ for {rep} differs from the stored span grid")
    if problems:
        raise SystemExit(f"{CONTEXT_DIR}: " + "; ".join(problems))

    sites = {}
    for features, reps in ((CONTEXT_DIR, SPAN_REPRESENTATIONS), (CONV_SPANS_DIR, CONV_REPRESENTATIONS)):
        for rep in reps:
            for span in CONTEXT_SPANS:
                X = np.load(features_path(rep, span, features), mmap_mode="r")
                if X.shape[0] != d.n:
                    raise SystemExit(f"{rep}_{span}.npy has {X.shape[0]} rows, not {d.n}")
                nan = np.isnan(X[:, 0, 0])
                if nan[~d.in_grid].sum() != (~d.in_grid).sum():
                    raise SystemExit(f"{rep}_{span}.npy has values on rows the grid does not score")
                empty = int((nan & d.in_grid).sum())
                expected = m["v_empty"].get(rep, 0) if (span == "v" and features == CONTEXT_DIR) else 0
                if empty != expected:
                    raise SystemExit(f"{rep}_{span}.npy: {empty} empty grid rows, expected {expected}")
                sites[(rep, span)] = d.in_grid & ~nan
    return sites


def span_grid_context(n_jobs) -> int:
    """The context-only span and the preceding stretch without the flap, for
    every representation (§8, 2026-11-23 and 2026-11-24): the grid's sites, folds
    and probes, cached beside its blocks in GRID_DIR. Scoring is
    scripts/score_context.py."""
    import time

    d = SpanData()
    check_conv_features(d)
    sites = check_context_features(d)
    blocks = [(rep, span, CONTEXT_DIR) for rep in SPAN_REPRESENTATIONS for span in CONTEXT_SPANS]
    blocks += [(rep, span, CONV_SPANS_DIR) for rep in CONV_REPRESENTATIONS for span in CONTEXT_SPANS]
    print(f"span-grid-context: {len(blocks)} blocks on {int(d.in_grid.sum()):,} sites -> "
          f"{GRID_DIR}", flush=True)
    t0 = time.perf_counter()
    timing = {}
    for rep, span, features in blocks:
        t = time.perf_counter()
        run_block(d, "probe", rep, span, "joint", False, n_jobs,
                  features=features, run_dir=GRID_DIR, sites=sites[(rep, span)])
        timing[f"{rep}_{span}"] = time.perf_counter() - t
        print(f"    {timing[f'{rep}_{span}']:.0f} s; {(time.perf_counter() - t0) / 60:.1f} min "
              "so far", flush=True)
    (GRID_DIR / "timing-context.json").write_text(json.dumps(
        {"blocks_seconds": timing, "total_seconds": time.perf_counter() - t0}, indent=2))
    return 0


def check_cropped_features(d: SpanData, cropped_dir=CROPPED_DIR, allow_limit=False):
    """The cropped extraction is a full one on the grid's rows (§8, 2026-11-26):
    the registered inventory, exclusions and revisions, L as registered, its full
    excerpt's ɾ equal to the stored pooled design on every row for every
    representation, and a value on exactly the grid's rows of every file.

    allow_limit admits a --limit smoke extraction, for testing the scorers
    only; nothing fit on one is ever written under results/.
    """
    import numpy as np

    from measure_span_grid import EXPECTED_SHA256

    if (cropped_dir / "site_ids.txt").read_text().split() != list(d.inv.site_id):
        raise SystemExit(f"{cropped_dir}/site_ids.txt is not in inventory order")
    m = json.loads((cropped_dir / "extract-manifest.json").read_text())
    problems = []
    if m.get("limit") and not allow_limit:
        problems.append("a --limit smoke test, not a full extraction")
    if not m.get("limit") and m["n_encoded"] != int(d.in_grid.sum()):
        problems.append(f"{m['n_encoded']:,} sites encoded, the grid has {int(d.in_grid.sum()):,}")
    if m["inventory_sha256"] != d.spans_manifest["inventory_sha256"]:
        problems.append("a different inventory from the span grid's")
    if m["exclusion_sha256"] != EXPECTED_SHA256:
        problems.append("an exclusion list other than the registered one (§1)")
    if m["checkpoint_revisions"] != CROPPED_REVISIONS:
        problems.append(f"revisions {m['checkpoint_revisions']}, §8 pins {CROPPED_REVISIONS}")
    if m["frame_dtype"] != "float16" or m["pooling"] != "mean":
        problems.append("not float16 frames mean-pooled (§4, §8)")
    if m["L_ms"] != CROPPED_L_MS:
        problems.append(f"L = {m['L_ms']} ms, not {CROPPED_L_MS}")
    if m["code_commit"].endswith("-dirty") and not allow_limit:
        problems.append("extracted from a dirty tree")
    for rep, r in m["r_vs_stored"].items():
        if r["rows_differing"] or r["max_abs_diff"] != 0:
            problems.append(f"the full excerpt's ɾ for {rep} differs from the stored pooled design")
    if problems:
        raise SystemExit(f"{cropped_dir}: " + "; ".join(problems))

    sites = d.in_grid.copy()
    if m.get("limit"):
        X = np.load(features_path("mfcc", "pad_ctx", cropped_dir), mmap_mode="r")
        sites &= ~np.isnan(X[:, 0, 0])
    for rep in CROPPED_REPRESENTATIONS:
        for pool in CROPPED_POOLS:
            X = np.load(features_path(rep, pool, cropped_dir), mmap_mode="r")
            if X.shape[0] != d.n:
                raise SystemExit(f"{rep}_{pool}.npy has {X.shape[0]} rows, not {d.n}")
            if (np.isnan(X[:, 0, 0]) != ~sites).any():
                raise SystemExit(f"{rep}_{pool}.npy: NaN rows are not the unscored sites")
    return m, sites


def build_cropped_baselines(d: SpanData, cropped_dir=CROPPED_DIR, features=GRID_FEATURES):
    """MFCC + clipped duration for the cropped design's Vɾ and VɾV (C5's
    cropped half): the crop's 39 MFCC dimensions and §5's clipped duration."""
    import numpy as np

    features.mkdir(parents=True, exist_ok=True)
    clipped = d.durations["clipped"].astype(np.float32)[:, None, None]
    for span in MFCC_DURATION_SPANS:
        m = np.load(features_path("mfcc", f"crop_{span}", cropped_dir))
        np.save(features_path("mfcc_duration", f"crop_{span}", features),
                np.concatenate([m, clipped], axis=2))


def cropped_blocks(cropped_dir=CROPPED_DIR, features=GRID_FEATURES):
    """(rep, pool, features dir) for every block the cropped stage fits."""
    blocks = [(rep, pool, cropped_dir) for rep in CROPPED_REPRESENTATIONS
              for pool in CROPPED_POOLS]
    blocks += [("mfcc_duration", f"crop_{span}", features) for span in MFCC_DURATION_SPANS]
    return blocks


def span_grid_cropped(n_jobs) -> int:
    """The cropped design, the padding-only control and the flap-masked context
    for every representation (§3; §8, 2026-11-26), plus MFCC + duration on the
    crops: the grid's sites, folds and probes, cached beside its blocks in
    GRID_DIR. Scoring is scripts/score_cropped.py and scripts/score_masked.py."""
    import time

    d = SpanData()
    _, sites = check_cropped_features(d)
    build_cropped_baselines(d)
    blocks = cropped_blocks()
    print(f"span-grid-cropped: {len(blocks)} blocks on {int(sites.sum()):,} sites -> "
          f"{GRID_DIR}", flush=True)
    t0 = time.perf_counter()
    timing = {}
    for rep, pool, features in blocks:
        t = time.perf_counter()
        run_block(d, "probe", rep, pool, "joint", False, n_jobs,
                  features=features, run_dir=GRID_DIR, sites=sites)
        timing[f"{rep}_{pool}"] = time.perf_counter() - t
        print(f"    {timing[f'{rep}_{pool}']:.0f} s; {(time.perf_counter() - t0) / 60:.1f} min "
              "so far", flush=True)
    (GRID_DIR / "timing-cropped.json").write_text(json.dumps(
        {"blocks_seconds": timing, "total_seconds": time.perf_counter() - t0}, indent=2))
    return 0


# ------------------------------------------------------------- transfer probe


def transfer_dirs(rep):
    """(stop features, flap features) for a representation."""
    if rep in SPAN_REPRESENTATIONS:
        return TRANSFER_STOPS_DIR, SPANS_DIR
    return CONV_STOPS_DIR, CONV_SPANS_DIR


class TransferStops:
    """The transfer probe's training stops (§8, 2026-11-21), checked before any is read.

    The stops TSV must be the file both extractions encoded (its sha256) and in
    their row order; each set must have the registration's counts and the
    extraction's ID digest; both extractions must be full ones on the grid's
    inventory at the pinned revisions, NaN exactly at the clamped stop.

    Each stop gets the joint split's speaker fold and type fold, from the same
    assignment over the pilot's 10,389 eligible sites that gives the flaps
    theirs (capstone.splits). That assignment is re-derived here and checked
    against the flaps' own folds before it is used for a stop.
    """

    def __init__(self, d: SpanData):
        import csv
        import hashlib

        import numpy as np

        from capstone.splits import N_FOLDS, speaker_folds, type_folds, type_key

        raw = TRANSFER_STOPS_TSV.read_bytes()
        tsv_hash = hashlib.sha256(raw).hexdigest()
        body = [ln for ln in raw.decode().splitlines() if not ln.startswith("#")]
        rows = list(csv.DictReader(body, delimiter="\t"))
        ids = [r["site_id"] for r in rows]
        problems = []

        for stops_dir, revisions in ((TRANSFER_STOPS_DIR, SPAN_REVISIONS),
                                     (CONV_STOPS_DIR, CONV_REVISIONS)):
            m = json.loads((stops_dir / "extract-manifest.json").read_text())
            if (stops_dir / "site_ids.txt").read_text().split() != ids:
                problems.append(f"{stops_dir}/site_ids.txt is not the stops TSV's order")
            if m.get("limit"):
                problems.append(f"{stops_dir}: a --limit smoke test, not a full extraction")
            if m.get("part") != "stops" or m["stops_tsv_sha256"] != tsv_hash:
                problems.append(f"{stops_dir}: encoded from another stops TSV")
            if m["inventory_sha256"] != d.spans_manifest["inventory_sha256"]:
                problems.append(f"{stops_dir}: a different inventory from the span grid's")
            if m["checkpoint_revisions"] != revisions:
                problems.append(f"{stops_dir}: revisions {m['checkpoint_revisions']}, "
                                f"pinned {revisions}")
            if m["frame_dtype"] != "float16" or m["pooling"] != "mean":
                problems.append(f"{stops_dir}: not float16 frames mean-pooled")
            for span, r in m.get("check_vs_stored", {}).items():
                if r["rows_identical"] != r["rows"]:
                    problems.append(f"{stops_dir}: the wav2vec2 check failed on {span}")
            for tset in TRANSFER_SETS:
                col = [r["site_id"] for r in rows if r[f"in_{tset}"] == "1"]
                got = hashlib.sha256("".join(f"{s}\n" for s in sorted(col)).encode()).hexdigest()
                if got != m["site_id_sha256"][f"{int(tset) / 100:.2f}"]:
                    problems.append(f"{stops_dir}: the {tset} set's IDs differ from the extraction's")
        for tset, want in TRANSFER_SETS.items():
            sub = [r for r in rows if r[f"in_{tset}"] == "1"]
            got = (len(sub), sum(r["label"] == "d" for r in sub), len({r["word"] for r in sub}))
            if got != want:
                problems.append(f"the {tset} set has {got} (stops, /d/, types), registered {want}")
        if problems:
            raise SystemExit("transfer stops: " + "; ".join(problems))

        self.rows = rows
        self.n = len(rows)
        self.is_d = np.array([r["label"] == "d" for r in rows])
        self.clamped = np.array([r["excerpt_clamped"] == "1" for r in rows])
        self.sets = {t: np.array([r[f"in_{t}"] == "1" for r in rows]) & ~self.clamped
                     for t in TRANSFER_SETS}
        self.tsv_sha256 = tsv_hash

        # The joint split's two axes, assigned over the eligible sites as
        # iter_splits assigns them, checked against the flaps' folds.
        e = d.elig_idx
        spk = speaker_folds(d.inv.speaker.to_numpy()[e])
        typ = type_folds(d.inv.word.to_numpy()[e])
        a = np.array([spk[s] for s in d.inv.speaker.to_numpy()[e]])
        b = np.array([typ[type_key(w)] for w in d.inv.word.to_numpy()[e]])
        self.flap_fold = np.full(d.n, -1, dtype=np.int64)
        self.flap_fold[e] = N_FOLDS * a + b
        for s in d.splits("joint", False):
            if not (self.flap_fold[s.test] == s.fold).all():
                raise SystemExit("the re-derived joint folds differ from iter_splits'")
        missing = sorted({r["speaker"] for r in rows if r["speaker"] not in spk} |
                         {r["word"] for r in rows if type_key(r["word"]) not in typ})
        if missing:
            raise SystemExit(f"stops outside the flaps' speakers or word types: {missing[:10]}")
        self.a = np.array([spk[r["speaker"]] for r in rows])
        self.b = np.array([typ[type_key(r["word"])] for r in rows])

        # NaN exactly at the clamped stop, in every file the stage reads.
        for rep in (*SPAN_REPRESENTATIONS, *CONV_REPRESENTATIONS):
            stops_dir, _ = transfer_dirs(rep)
            for span in TRANSFER_SPANS:
                X = np.load(features_path(rep, span, stops_dir), mmap_mode="r")
                if X.shape[0] != self.n or (np.isnan(X[:, 0, 0]) != self.clamped).any():
                    raise SystemExit(f"{stops_dir}/{rep}_{span}.npy: NaN rows are not the "
                                     "clamped stops")

    def cells(self, d: SpanData, tset):
        """(cell, train stops, test flaps) for the 25 cells of the joint grid: the
        set's stops sharing neither the cell's speaker fold nor its type fold, and
        the cell's grid flaps."""
        import numpy as np

        from capstone.splits import N_FOLDS

        out = []
        for i in range(N_FOLDS):
            for j in range(N_FOLDS):
                k = N_FOLDS * i + j
                train = np.flatnonzero(self.sets[tset] & (self.a != i) & (self.b != j))
                test = np.flatnonzero(d.in_grid & (self.flap_fold == k))
                y = self.is_d[train]
                if not (0 < y.sum() < len(y)):
                    raise SystemExit(f"cell {k}: the {tset} set's training stops are one class")
                out.append((k, train, test))
        assert sum(len(t) for _, _, t in out) == int(d.in_grid.sum())
        return out


def transfer_block_name(rep, span, tset):
    return f"transfer_{rep}_{span}_{tset}_joint.npz"


def _fit_transfer(stop_path, flap_path, layer, train, test, y_train):
    import numpy as np

    from capstone.probes.linear import fit_predict

    X = np.load(stop_path, mmap_mode="r")
    F = np.load(flap_path, mmap_mode="r")
    return fit_predict(np.asarray(X[train, layer]), y_train, np.asarray(F[test, layer]))


def run_transfer_block(d: SpanData, st: TransferStops, rep, span, tset, n_jobs,
                       run_dir=TRANSFER_RUN_DIR):
    """Fit one transfer block and cache it: §5's probe, every layer, trained on
    the stops' span and scored on the flaps' matching span. A real probe only;
    the registration asks for no control task here."""
    import numpy as np
    from joblib import Parallel, delayed

    out = run_dir / transfer_block_name(rep, span, tset)
    if out.exists():
        return out
    stops_dir, flaps_dir = transfer_dirs(rep)
    stop_path = features_path(rep, span, stops_dir)
    flap_path = features_path(rep, TRANSFER_SPANS[span], flaps_dir)
    n_layers = np.load(stop_path, mmap_mode="r").shape[1]
    if np.load(flap_path, mmap_mode="r").shape[1:] != np.load(stop_path, mmap_mode="r").shape[1:]:
        raise SystemExit(f"{rep}: stop and flap features differ in shape")
    cells = st.cells(d, tset)
    jobs = [(L, c) for L in range(n_layers) for c in range(len(cells))]
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_transfer)(stop_path, flap_path, L, cells[c][1], cells[c][2],
                               st.is_d[cells[c][1]])
        for L, c in jobs)

    fold = np.full(d.n, -1, dtype=np.int32)
    for k, _, test in cells:
        fold[test] = k
    real = np.full((n_layers, d.n), np.nan)
    converged = np.zeros((n_layers, len(cells)), dtype=bool)
    for (L, c), (pred, conv) in zip(jobs, results):
        real[L, cells[c][2]] = pred
        converged[L, c] = conv
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, fold=fold, real=real, real_converged=converged,
             n_train=np.array([len(t) for _, t, _ in cells]),
             n_train_d=np.array([int(st.is_d[t].sum()) for _, t, _ in cells]))
    print(f"  {out.name}: {len(jobs)} fits, {int((~converged).sum())} did not converge",
          flush=True)
    return out


def transfer_blocks():
    """Every transfer block, the two called ones first: the 0.40 set on every span
    for every representation; the sensitivity sets on the stop and V+stop spans,
    for every representation but the raw-input variants."""
    blocks = [("wav2vec2", "s", "040"), ("wav2vec2", "vs", "040")]   # X1, X2
    for tset in TRANSFER_SETS:
        for rep in (*SPAN_REPRESENTATIONS, *CONV_REPRESENTATIONS):
            if tset != "040" and rep.endswith("_raw"):
                continue
            for span in (TRANSFER_SPANS if tset == "040" else TRANSFER_SENSITIVITY_SPANS):
                if (rep, span, tset) not in blocks:
                    blocks.append((rep, span, tset))
    return blocks


def span_grid_transfer(n_jobs) -> int:
    """The transfer probe (§8, 2026-11-21 and 2026-11-25): trained on the stops,
    scored on the grid's flaps in the joint split's cells, cached in
    TRANSFER_RUN_DIR. Scoring is scripts/score_transfer.py."""
    import time

    d = SpanData()
    check_conv_features(d)
    st = TransferStops(d)
    blocks = transfer_blocks()
    print(f"span-grid-transfer: {len(blocks)} blocks, {st.n:,} stops -> "
          f"{int(d.in_grid.sum()):,} flaps -> {TRANSFER_RUN_DIR}", flush=True)
    t0 = time.perf_counter()
    timing = {}
    for rep, span, tset in blocks:
        t = time.perf_counter()
        run_transfer_block(d, st, rep, span, tset, n_jobs)
        timing[f"{rep}_{span}_{tset}"] = time.perf_counter() - t
        print(f"    {timing[f'{rep}_{span}_{tset}']:.0f} s; "
              f"{(time.perf_counter() - t0) / 60:.1f} min so far", flush=True)
    (TRANSFER_RUN_DIR / "timing-transfer.json").write_text(json.dumps(
        {"blocks_seconds": timing, "total_seconds": time.perf_counter() - t0}, indent=2))
    return 0


# ------------------------------------------------------------------------ rest


def rest(n_jobs) -> int:
    d = Data()
    blocks = []
    for pooling in POOLINGS:
        for cond in CONDITIONS:
            for sub in ((False, True) if cond != "joint" else (False,)):
                blocks.append((pooling, cond, sub))
    for pooling, cond, sub in blocks:
        for rep in REPRESENTATIONS:
            run_block(d, "probe", rep, pooling, cond, sub, n_jobs)
    for cond in CONDITIONS:
        for sub in ((False, True) if cond != "joint" else (False,)):
            run_lookup(d, cond, sub)
    for cond in SPEAKER_CONDITIONS:
        for rep in REPRESENTATIONS:
            run_block(d, "speaker", rep, "mean", cond, False, n_jobs)
    return 0


# -------------------------------------------------------------------- assemble


def assemble() -> int:
    import numpy as np
    import pandas as pd

    from capstone.predictions import build_manifest, write_prediction_columns
    from capstone.probes.scoring import (balanced_accuracy,
                                         balanced_accuracy_multiclass,
                                         boot_balanced_accuracy, f1_d, interval,
                                         resample_weights)

    d = Data()
    e = d.elig_idx
    # Built first: it records the commit, and this stage writes its CSVs into
    # the tree, which would otherwise mark its own provenance dirty.
    manifest = build_manifest(
        INVENTORY, split_scheme=SPLIT_SCHEME, seeds=seeds.REGISTERED,
        n_sites_scored=len(e), synthetic=False, encoders=("wavjepa", "mfcc", "lookup"),
        run_id=f"pilot-{DATE}")
    y = d.is_d[e]
    tix = d.type_idx[e]
    W = resample_weights(len(d.types), seeds.BOOTSTRAP)

    rows, fold_rows, pred_cols, ctrl_cols = [], [], [], []
    site_ids = d.inv.site_id.to_numpy()

    def lookup_ba(cond, sub):
        z = np.load(RUN_DIR / f"lookup_{cond}{'_sub' if sub else ''}.npz")
        return balanced_accuracy(y, z["pred_d"][e]), z

    def speaker_ba(rep, cond, sub, pooling):
        p = RUN_DIR / block_name("speaker", rep, pooling, cond, sub)
        if not p.exists():
            return None
        z = np.load(p)
        return [balanced_accuracy_multiclass(d.speaker_code[e], z["speaker"][L, e])
                for L in range(z["speaker"].shape[0])]

    for pooling in POOLINGS:
        for cond in CONDITIONS:
            for sub in ((False, True) if cond != "joint" else (False,)):
                lk, lz = lookup_ba(cond, sub)
                for rep in REPRESENTATIONS:
                    p = RUN_DIR / block_name("probe", rep, pooling, cond, sub)
                    if not p.exists():
                        raise SystemExit(f"missing {p.name}; run `gate` and `rest` first")
                    z = np.load(p)
                    real, ctrl = z["real"][:, e], z["control"][:, e]
                    pred, cpred = real > 0.5, ctrl > 0.5
                    cy = d.control_is_d[e]
                    boot = boot_balanced_accuracy(W, tix, y, pred == y)
                    lo, hi = interval(boot)
                    spk = speaker_ba(rep, cond, sub, pooling)
                    for L in range(real.shape[0]):
                        ba = balanced_accuracy(y, pred[L])
                        cba = balanced_accuracy(cy, cpred[L])
                        rows.append({
                            "encoder": rep, "pooling": pooling, "layer": L,
                            "condition": cond, "subsampled": sub,
                            "balanced_acc": ba, "ci_lo": lo[L], "ci_hi": hi[L],
                            "f1_d": f1_d(y, pred[L]),
                            "control_balanced_acc": cba, "selectivity": ba - cba,
                            "lookup_balanced_acc": lk,
                            "speaker_balanced_acc": spk[L] if spk else np.nan,
                            "n": len(e), "n_d": int(y.sum()),
                            "n_nonconverged": int((~z["real_converged"][L]).sum()),
                            "n_nonconverged_control": int((~z["control_converged"][L]).sum()),
                        })
                        for k in np.unique(z["fold"][e]):
                            m = z["fold"][e] == k
                            fold_rows.append({
                                "encoder": rep, "pooling": pooling, "layer": L,
                                "condition": cond, "subsampled": sub, "fold": int(k),
                                "n": int(m.sum()), "n_d": int(y[m].sum()),
                                "balanced_acc": balanced_accuracy(y[m], pred[L][m])
                                if 0 < y[m].sum() < m.sum() else np.nan,
                            })
                    pred_cols.append((rep, pooling, cond, sub, z["fold"], z["real"]))
                    ctrl_cols.append((rep, pooling, cond, sub, z["fold"], z["control"]))
                if pooling == POOLINGS[0]:
                    # A hard 0/1 for eligible sites; NaN for the excluded,
                    # which the lookup never scored.
                    pd_ = np.where(d.eligible, np.where(lz["pred_d"], 1.0, 0.0), np.nan)[None]
                    pred_cols.append(("lookup", None, cond, sub, lz["fold"], pd_))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"probe-results-{DATE}.csv", index=False, float_format="%.6f")
    pd.DataFrame(fold_rows).to_csv(OUT / f"probe-folds-{DATE}.csv", index=False,
                                   float_format="%.6f")

    write_prediction_columns(pred_cols, site_ids, d.eligible, d.is_d,
                             RUN_DIR / "predictions.parquet", synthetic=False)
    _write_control(ctrl_cols, site_ids, d, RUN_DIR / "control-predictions.parquet")

    manifest.write(RUN_DIR / "run-manifest.json")
    manifest.write(OUT / "run-manifest-pilot.json")
    print(f"wrote {OUT / f'probe-results-{DATE}.csv'} ({len(res):,} rows)")
    print(f"wrote {OUT / f'probe-folds-{DATE}.csv'} ({len(fold_rows):,} rows)")
    print(f"wrote {RUN_DIR / 'predictions.parquet'}, control-predictions.parquet, run-manifest.json")
    return 0


SPLIT_SCHEME = ("capstone.splits, pre-registration revision 3 §3. Clamped sites excluded "
                "before fold assignment and scoring; folds over the 10,389 eligible, keyed by "
                "type_key. random: stratified 5-fold by label, seed 20260926; speaker_disjoint: "
                "5 speaker folds; word_disjoint: 5 type folds; joint: 5x5 grid, all 25 cells. "
                "Secondary: single-axis training subsampled to 80% per class, keyed on "
                "(20260926, fold).")


def _write_control(cols, site_ids, d, path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for rep, pooling, cond, sub, fold, p_d in cols:
        R, n = p_d.shape
        e = np.tile(d.eligible, R)
        tables.append(pa.table({
            "site_id": np.tile(site_ids, R),
            "encoder": [rep] * (R * n),
            "layer": np.repeat(np.arange(R, dtype=np.int32), n),
            "pooling": [pooling] * (R * n),
            "condition": [cond] * (R * n),
            "subsampled": np.full(R * n, sub),
            "fold": pa.array(np.tile(fold, R), mask=~e, type=pa.int32()),
            "control_is_d": pa.array(np.tile(d.control_is_d, R), mask=~e),
            "p_d": pa.array(p_d.ravel(), mask=~e),
        }))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("gate", "rest", "assemble", "span-repro",
                                          "span-grid", "span-grid-conv",
                                          "span-grid-context", "span-grid-transfer",
                                          "span-grid-cropped"))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel fits (default: all cores)")
    args = ap.parse_args()
    if args.stage == "gate":
        return gate(args.jobs)
    if args.stage == "rest":
        return rest(args.jobs)
    if args.stage == "span-repro":
        return span_repro(args.jobs)
    if args.stage == "span-grid":
        return span_grid(args.jobs)
    if args.stage == "span-grid-conv":
        return span_grid_conv(args.jobs)
    if args.stage == "span-grid-context":
        return span_grid_context(args.jobs)
    if args.stage == "span-grid-transfer":
        return span_grid_transfer(args.jobs)
    if args.stage == "span-grid-cropped":
        return span_grid_cropped(args.jobs)
    return assemble()


if __name__ == "__main__":
    sys.exit(main())
