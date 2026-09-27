#!/usr/bin/env python3
"""The pilot's probe run: every registered probe, then the §5 criteria.

HOW TO RUN (in .venv-app, after scripts/extract_site_features.py)
    .venv-app/bin/python scripts/run_probes.py gate       # criteria first
    .venv-app/bin/python scripts/run_probes.py rest       # everything else
    .venv-app/bin/python scripts/run_probes.py assemble   # CSVs, parquet, manifest

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


def features_path(rep, pooling):
    return FEATURES / f"{rep}_{pooling}.npy"


def block_name(kind, rep, pooling, condition, subsampled):
    return f"{kind}_{rep}_{pooling}_{condition}{'_sub' if subsampled else ''}.npz"


# --------------------------------------------------------------------- fitting


def _fit(path, layer, train, test, y_train):
    import numpy as np

    from capstone.probes.linear import fit_predict

    X = np.load(path, mmap_mode="r")
    return fit_predict(np.asarray(X[train, layer]), y_train, np.asarray(X[test, layer]))


def run_block(d: Data, kind, rep, pooling, condition, subsampled, n_jobs):
    """Fit one block and cache it. kind: 'probe' (real + control) or 'speaker'."""
    import numpy as np
    from joblib import Parallel, delayed

    out = RUN_DIR / block_name(kind, rep, pooling, condition, subsampled)
    if out.exists():
        return out
    path = features_path(rep, pooling)
    n_layers = np.load(path, mmap_mode="r").shape[1]
    splits = d.splits(condition, subsampled)

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

    RUN_DIR.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("stage", choices=("gate", "rest", "assemble"))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel fits (default: all cores)")
    args = ap.parse_args()
    if args.stage == "gate":
        return gate(args.jobs)
    if args.stage == "rest":
        return rest(args.jobs)
    return assemble()


if __name__ == "__main__":
    sys.exit(main())
