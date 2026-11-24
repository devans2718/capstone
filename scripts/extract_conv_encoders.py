#!/usr/bin/env python3
"""HuBERT and WavLM base on every span the pooled design has, plus a raw-input variant.

HOW TO RUN
    docker compose run --rm dev python scripts/extract_conv_encoders.py \
        [--part grid|stops|all] [--limit N]

    --part grid:  the pilot's sites: §2's four spans, the context-only span and
                  the preceding stretch without the flap (§8, 2026-11-23)
    --part stops: the transfer probe's training stops (§8, 2026-11-21)
    --part all:   both, grid first (the default)
    --limit N:    a smoke test, N sites per part; the manifest records it

WHAT IT DOES (span-grid README §8, 2026-11-24)
    Five representations, each a checkpoint and an input normalization:
        hubert, wavlm          normalized per excerpt, as the registered
                               wav2vec2 run is: the primary protocol
        wav2vec2_raw, hubert_raw, wavlm_raw
                               the raw waveform, as fairseq pretrained
                               wav2vec2-base and HuBERT-base: a variant
    All three checkpoints share wav2vec2-base's conv stack, so one frame rule
    serves all: by centre with §4's nearest-frame fallback, through
    Wav2Vec2Extractor.window_frames. Excerpts, float16 frames and mean pooling
    are the pooled design's (§3, §4), planned by the same functions as
    extract_span_features.py and extract_transfer_context.py.

    GRID: ɾ for the 10,389 eligible sites, Vɾ, ɾV, VɾV, ctx and v for the
    10,334 grid sites, exactly as those two scripts define them. ctx is every
    frame not in ɾ's; v is Vɾ's frames less ɾ's, NaN when empty.
    STOPS: s, vs, sv, vsv for the 0.25 set's 2,433 stops; the clamped one stays
    NaN.

    A sixth pass, wav2vec2 normalized, is the registered run re-done through
    the generalized loader. It is not written; every span it pools is compared
    row by row with the stored file, and a mismatch exits non-zero. That is
    the check that this script's frames and spans are the stored ones.

WHAT IT WRITES (Buckeye-derived, so /data only, never the repository)
    /data/embeddings/conv-spans-<inventory sha256[:12]>/
        {rep}_{r,vr,rv,vrv,ctx,v}.npy   (10392, 13, 768) float32
    /data/embeddings/conv-stops-<stops TSV sha256[:12]>/
        {rep}_{s,vs,sv,vsv}.npy          (2433, 13, 768) float32
    each with frames.tsv (frames pooled per span, the same for every rep),
    site_ids.txt and extract-manifest.json. Rows not encoded are NaN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_site_features import (CONTEXT_MS, EMBEDDINGS, SAMPLE_RATE,  # noqa: E402
                                   read_sites)
from extract_span_features import FRAME_DTYPE, mean_pool, plan_spans  # noqa: E402
from extract_transfer_context import (STOP_SPANS, read_stops,  # noqa: E402
                                      read_track)

# rep -> (checkpoint name in CONV_ENCODERS, normalize the input)
VARIANTS = {
    "hubert": ("hubert", True),
    "wavlm": ("wavlm", True),
    "wav2vec2_raw": ("wav2vec2", False),
    "hubert_raw": ("hubert", False),
    "wavlm_raw": ("wavlm", False),
}
CHECK = "wav2vec2"          # normalized: the registered run, compared not written
GRID_SPANS = ("r", "vr", "rv", "vrv", "ctx", "v")
SHAPE = (13, 768)


class Encoders:
    """The six passes over one excerpt, and the one frame rule they share."""

    def __init__(self):
        from capstone.encoders.extraction import CONV_ENCODERS, Wav2Vec2Extractor

        self.ext = {rep: Wav2Vec2Extractor.load_named(name, normalize=norm)
                    for rep, (name, norm) in {**VARIANTS, CHECK: ("wav2vec2", True)}.items()}
        self.rule = self.ext[CHECK]
        for rep, e in self.ext.items():
            if e.geometry != self.rule.geometry:
                raise SystemExit(f"{rep}'s frame geometry {e.geometry} is not "
                                 f"wav2vec2-base's {self.rule.geometry}")
        self.revisions = {name: CONV_ENCODERS[name][1] for name, _ in VARIANTS.values()}
        self.normalize = {rep: e.feature_extractor.do_normalize for rep, e in self.ext.items()}
        print(f"device {self.rule.device}; revisions {self.revisions}; "
              f"normalize {self.normalize}", flush=True)

    def encode(self, wav_t, ex):
        """{rep: (13, n, 768)} for one excerpt, float16 frames as float32."""
        from capstone.encoders.extraction import excerpt_samples

        a, b = excerpt_samples(ex, SAMPLE_RATE)
        return {rep: e.encode(wav_t[a:b], dtype=FRAME_DTYPE).float().numpy()
                for rep, e in self.ext.items()}

    def idx(self, ex, n_frames, start_ms, end_ms):
        """Frame indices of one span, by centre with §4's fallback."""
        import numpy as np
        from dataclasses import replace

        rng = self.rule.window_frames(
            replace(ex, target_start_ms=start_ms, target_end_ms=end_ms), n_frames)
        return np.arange(rng.start, rng.stop)


class Check:
    """The CHECK pass against a stored file, row by row, NaN rows included."""

    def __init__(self, stored: dict):
        import numpy as np

        self.stored = {span: np.load(path, mmap_mode="r") for span, path in stored.items()}
        self.same = {span: 0 for span in stored}
        self.rows = {span: 0 for span in stored}
        self.max_diff = {span: 0.0 for span in stored}

    def __call__(self, span, i, new):
        import numpy as np

        old = np.asarray(self.stored[span][i])
        self.rows[span] += 1
        if np.array_equal(old, new, equal_nan=True):
            self.same[span] += 1
        else:
            d = np.abs(old - new)
            self.max_diff[span] = max(self.max_diff[span], float(np.nanmax(d)) if
                                      np.isfinite(d).any() else float("inf"))

    def report(self):
        ok = True
        out = {}
        for span in self.stored:
            out[span] = {"rows_identical": self.same[span], "rows": self.rows[span],
                         "max_abs_diff": self.max_diff[span]}
            ok &= self.same[span] == self.rows[span]
            print(f"{CHECK} {span} vs stored: {self.same[span]:,} of {self.rows[span]:,} "
                  f"rows identical; max |diff| {self.max_diff[span]:.3e}")
        return ok, out


def open_arrays(out, n, spans):
    import numpy as np

    arrays = {}
    for rep in VARIANTS:
        for span in spans:
            a = np.lib.format.open_memmap(out / f"{rep}_{span}.npy", mode="w+",
                                          dtype=np.float32, shape=(n, *SHAPE))
            a[:] = np.nan
            arrays[rep, span] = a
    return arrays


def finish(out, arrays, spans, ids, n_frames, encoded, manifest):
    """Flush, check for inf, write frames.tsv, site_ids.txt and the manifest."""
    import numpy as np

    for (rep, span), a in arrays.items():
        a.flush()
        if np.isinf(a).any():
            raise SystemExit(f"{rep}_{span} contains inf")
    with (out / "frames.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["site_id", *(f"n_{span}" for span in spans)])
        for i in sorted(encoded):
            w.writerow([ids[i], *(int(n_frames[span][i]) or "" for span in spans)])
    (out / "site_ids.txt").write_text("".join(s + "\n" for s in ids))
    rows = np.array(sorted(encoded), dtype=int)
    medians = {}
    for span in spans:
        k = n_frames[span][rows]
        k = k[k > 0]
        medians[span] = float(np.median(k)) if len(k) else None
    manifest["median_frames"] = medians
    (out / "extract-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("  median frames  " + "  ".join(f"{s} {m}" for s, m in medians.items()))
    print(f"wrote {out}", flush=True)


def base_manifest(enc, limit):
    from capstone.predictions import code_commit

    return {
        "code_commit": code_commit(),
        "variants": {rep: {"checkpoint": name, "normalize": norm}
                     for rep, (name, norm) in VARIANTS.items()},
        "checkpoint_revisions": enc.revisions,
        "check": f"{CHECK}, normalized, against the stored registered run",
        "context_ms": CONTEXT_MS,
        "sample_rate": SAMPLE_RATE,
        "frame_dtype": FRAME_DTYPE,
        "stored_dtype": "float32",
        "pooling": "mean",
        "frame_rule": "centre, §4 nearest-frame fallback",
        "limit": limit,
    }


def by_track(rows, members):
    tracks: dict[str, list[int]] = {}
    for i in members:
        tracks.setdefault(rows[i]["track"], []).append(i)
    return sorted(tracks.items())


# --------------------------------------------------------------------- grid


def extract_grid(enc, limit):
    import numpy as np

    sites, inv_hash = read_sites()
    n = len(sites)
    plans, digest = plan_spans(sites)
    out = EMBEDDINGS / f"conv-spans-{inv_hash[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    spans_dir = EMBEDDINGS / f"spans-{inv_hash[:12]}"
    context_dir = EMBEDDINGS / f"context-{inv_hash[:12]}"
    check = Check({**{s: spans_dir / f"wav2vec2_{s}.npy" for s in ("r", "vr", "rv", "vrv")},
                   **{s: context_dir / f"wav2vec2_{s}.npy" for s in ("ctx", "v")}})
    print(f"\nGRID: {len(plans):,} sites planned -> {out}", flush=True)
    arrays = open_arrays(out, n, GRID_SPANS)
    n_frames = {span: np.zeros(n, dtype=np.int32) for span in GRID_SPANS}
    n_v_empty = 0

    done, encoded = 0, []
    for track, members in by_track(sites, plans):
        _, wav_t, rec_ms = read_track(sites[members[0]]["speaker"], track)
        for i in members:
            s = sites[i]
            ex, spans, info = plans[i]
            if ex.offset_ms + ex.duration_ms > rec_ms:
                raise SystemExit(f"{s['site_id']}: excerpt runs past the recording, "
                                 "but the inventory says it is not clamped")
            frames = enc.encode(wav_t, ex)
            n_all = frames[CHECK].shape[-2]
            idx = {span: enc.idx(ex, n_all, *bounds) for span, bounds in spans.items()}
            if info["in_grid"]:
                idx["ctx"] = np.setdiff1d(np.arange(n_all), idx["r"])
                idx["v"] = np.setdiff1d(idx["vr"], idx["r"])
                if len(idx["v"]) == 0:
                    n_v_empty += 1
            for span, sel in idx.items():
                if len(sel) == 0:
                    if span == "v":
                        check(span, i, np.full(SHAPE, np.nan, dtype=np.float32))
                        continue
                    raise SystemExit(f"{s['site_id']}: no frame in span {span}")
                n_frames[span][i] = len(sel)
                for rep in VARIANTS:
                    arrays[rep, span][i] = mean_pool(frames[rep], sel)
                check(span, i, mean_pool(frames[CHECK], sel))
            encoded.append(i)
            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        print(f"  {track}: {done:,} sites", flush=True)

    identical, report = check.report()
    manifest = base_manifest(enc, limit) | {
        "part": "grid",
        "inventory_sha256": inv_hash,
        "exclusion_sha256": digest,
        "spans": list(GRID_SPANS),
        "n_sites": n,
        "n_planned": len(plans),
        "n_encoded": done,
        "v_empty": n_v_empty,
        "check_vs_stored": report,
    }
    print(f"encoded {done:,} sites; v empty {n_v_empty}")
    finish(out, arrays, GRID_SPANS, [s["site_id"] for s in sites], n_frames, encoded,
           manifest)
    return 0 if identical else 1


# -------------------------------------------------------------------- stops


def extract_stops(enc, limit):
    import numpy as np

    from capstone.encoders.extraction import plan_excerpt
    from extract_transfer_context import STOPS_TSV

    stops, inv_hash, set_hashes = read_stops()
    n = len(stops)
    tsv_hash = hashlib.sha256(STOPS_TSV.read_bytes()).hexdigest()
    out = EMBEDDINGS / f"conv-stops-{tsv_hash[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    stored = EMBEDDINGS / f"transfer-stops-{tsv_hash[:12]}"
    check = Check({span: stored / f"wav2vec2_{span}.npy" for span in STOP_SPANS})
    print(f"\nSTOPS: {n:,} stops -> {out}", flush=True)
    arrays = open_arrays(out, n, STOP_SPANS)
    n_frames = {span: np.zeros(n, dtype=np.int32) for span in STOP_SPANS}

    done, encoded, skipped_clamped = 0, [], []
    for track, members in by_track(stops, range(n)):
        _, wav_t, rec_ms = read_track(stops[members[0]]["speaker"], track)
        for i in members:
            s = stops[i]
            start_ms, end_ms = float(s["start"]) * 1000, float(s["end"]) * 1000
            ex = plan_excerpt(start_ms, end_ms, rec_ms, context_ms=CONTEXT_MS)
            if ex.clamped != (s["excerpt_clamped"] == "1"):
                raise SystemExit(f"{s['site_id']}: clamped {ex.clamped} here, "
                                 f"{s['excerpt_clamped']} in the TSV")
            if ex.clamped:
                skipped_clamped.append(s["site_id"])
                continue
            lo, hi = ex.offset_ms, ex.offset_ms + ex.duration_ms
            pre = max(float(s["vowel_start"]) * 1000, lo)
            post = min(float(s["following_end"]) * 1000, hi)
            spans = {"s": (start_ms, end_ms), "vs": (pre, end_ms),
                     "sv": (start_ms, post), "vsv": (pre, post)}

            frames = enc.encode(wav_t, ex)
            n_all = frames[CHECK].shape[-2]
            for span, bounds in spans.items():
                sel = enc.idx(ex, n_all, *bounds)
                if len(sel) == 0:
                    raise SystemExit(f"{s['site_id']}: no frame in span {span}")
                n_frames[span][i] = len(sel)
                for rep in VARIANTS:
                    arrays[rep, span][i] = mean_pool(frames[rep], sel)
                check(span, i, mean_pool(frames[CHECK], sel))
            encoded.append(i)
            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        print(f"  {track}: {done:,} stops", flush=True)

    identical, report = check.report()
    manifest = base_manifest(enc, limit) | {
        "part": "stops",
        "stops_tsv": str(STOPS_TSV),
        "stops_tsv_sha256": tsv_hash,
        "inventory_sha256": inv_hash,
        "site_id_sha256": set_hashes,
        "spans": list(STOP_SPANS),
        "n_sites": n,
        "n_encoded": done,
        "skipped_clamped": skipped_clamped,
        "check_vs_stored": report,
    }
    print(f"encoded {done:,} stops; skipped as clamped {len(skipped_clamped)}")
    finish(out, arrays, STOP_SPANS, [s["site_id"] for s in stops], n_frames, encoded,
           manifest)
    return 0 if identical else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("grid", "stops", "all"), default="all")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    enc = Encoders()
    rc = 0
    if args.part in ("grid", "all"):
        rc |= extract_grid(enc, args.limit)
    if args.part in ("stops", "all"):
        rc |= extract_stops(enc, args.limit)
    if rc:
        print("\nThe wav2vec2 check is NOT bit-identical to the stored run. Nothing "
              "downstream is read until this is understood.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
