#!/usr/bin/env python3
"""Pooled features for the transfer probe's stops and the context-only span (span-grid §8).

HOW TO RUN
    docker compose run --rm dev python scripts/extract_transfer_context.py \
        [--part stops|context|all] [--limit N]

    --part stops:   the transfer probe's training stops (§8, 2026-11-21, f22cad8)
    --part context: the context-only span on the grid's flap sites (§8, 2026-11-23)
    --part all:     both, stops first (the default)
    --limit N:      a smoke test, N sites per part; the manifest records it

WHAT IT DOES
    The pooled design's extraction (§3, §4), unchanged: each site's excerpt is
    the target ± 200 ms through plan_excerpt, encoded once per encoder at the
    pinned revisions with float16 frames, and every span mean-pooled from that
    one encoding. Frame membership follows extract_span_features.py line for
    line: WavJEPA by onset, trimmed to frames that read wholly real audio;
    wav2vec 2.0 by centre with §4's nearest-frame fallback; MFCC by centre.

    STOPS reads /data/span-grid/transfer-stops.tsv (export_transfer_stops.py),
    all 2,433 rows of the 0.25 set, and pools §2's four spans with the stop in
    the flap's place:
        s     the stop                      (ɾ's counterpart)
        vs    the preceding stretch + stop  (Vɾ)
        sv    the stop + following segment  (ɾV)
        vsv   all three                     (VɾV)
    Spans are clipped to the excerpt, as §2 clips them. A stop whose excerpt
    is clamped at a recording edge is not encoded and its rows stay NaN, as
    the pilot does with its clamped sites; the count is printed.

    CONTEXT reads the pilot's sites (read_sites, hash-checked) and plans them
    with extract_span_features.plan_spans, so the excerpts and Vɾ bounds are
    the grid's own. For the 10,334 grid sites it pools:
        ctx   every frame of the excerpt not in that encoder's ɾ frames
              (WavJEPA: clean frames only)
        v     Vɾ's frames less ɾ's: the preceding stretch, flap excluded.
              Reported, not called. Empty for a stretch too short to own a
              frame of its own; such rows stay NaN and are counted.
    and re-pools ɾ, which has to equal the stored ɾ span
    (spans-<inventory hash>/{rep}_r.npy) bit for bit.

WHAT IT WRITES (Buckeye-derived, so /data only, never the repository)
    /data/embeddings/transfer-stops-<stops TSV sha256[:12]>/
        {wavjepa,wav2vec2}_{s,vs,sv,vsv}.npy   (2433, 13, 768) float32
        mfcc_{s,vs,sv,vsv}.npy                 (2433, 1, 39)   float32
        frames.tsv, site_ids.txt, extract-manifest.json
    /data/embeddings/context-<inventory sha256[:12]>/
        {wavjepa,wav2vec2}_{ctx,v}.npy         (10392, 13, 768) float32
        mfcc_{ctx,v}.npy                       (10392, 1, 39)   float32
        frames.tsv, site_ids.txt, extract-manifest.json
    Rows follow the input's order; rows not encoded are NaN.

WHAT IT CHECKS, AND STOPS ON
    The stops TSV's site-ID hashes against its own rows; each stop's
    clamped flag against the recording's length; a frame in every
    registered span of every representation (the stops' four, ctx); no inf.
    After the context part, the re-pooled ɾ against the stored ɾ: a mismatch
    is reported with its size and exits non-zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_site_features import (CONTEXT_MS, CORPUS, DELTA_WIDTH,  # noqa: E402
                                   EMBEDDINGS, MFCC, SAMPLE_RATE, mfcc_frames,
                                   mfcc_site_frames, read_sites)
from extract_span_features import FRAME_DTYPE, mean_pool, plan_spans  # noqa: E402

ENCODERS = ("wavjepa", "wav2vec2")
REPS = (*ENCODERS, "mfcc")
STOPS_TSV = Path("/data/span-grid/transfer-stops.tsv")
STOP_SPANS = ("s", "vs", "sv", "vsv")
CONTEXT_SPANS = ("ctx", "v")
N_STOPS = 2_433          # the 0.25 set (export_transfer_stops.REGISTERED)


# ------------------------------------------------------------------ shared


class Encoder:
    """The two encoders and MFCC over one excerpt, with the span grid's frame rules."""

    def __init__(self):
        from capstone.encoders.extraction import (REVISION, W2V2_REVISION,
                                                  Wav2Vec2Extractor, WavJEPAExtractor)
        self.ext = {"wavjepa": WavJEPAExtractor.load(), "wav2vec2": Wav2Vec2Extractor.load()}
        self.revisions = {"wavjepa": REVISION, "wav2vec2": W2V2_REVISION}
        self.n_clean_trimmed = 0
        print(f"device {self.ext['wavjepa'].device}; WavJEPA {REVISION}, "
              f"wav2vec2 {W2V2_REVISION}", flush=True)

    def encode(self, wav, wav_t, ex):
        """Frames for one excerpt: {rep: (13|1, n, 768|39)}, and WavJEPA's clean count."""
        from capstone.encoders.extraction import excerpt_samples

        a, b = excerpt_samples(ex, SAMPLE_RATE)
        frames = {rep: e.encode(wav_t[a:b], dtype=FRAME_DTYPE).float().numpy()
                  for rep, e in self.ext.items()}
        frames["mfcc"] = mfcc_frames(wav[a:b]).T[None]
        return frames, self.ext["wavjepa"].geometry.n_clean_frames(b - a)

    def span_idx(self, frames, n_clean, ex, start_ms, end_ms):
        """{rep: frame indices} for one span, exactly as extract_span_features selects them."""
        import numpy as np

        exs = replace(ex, target_start_ms=start_ms, target_end_ms=end_ms)
        idx = {}
        for rep, e in self.ext.items():
            rng = e.window_frames(exs, frames[rep].shape[-2])
            if rep == "wavjepa" and rng.stop > n_clean:
                rng = range(rng.start, max(rng.start, n_clean))
                self.n_clean_trimmed += 1
            idx[rep] = np.arange(rng.start, rng.stop)
        t0 = int(round(exs.target_start_local_ms / 1000 * SAMPLE_RATE))
        t1 = int(round(exs.target_end_local_ms / 1000 * SAMPLE_RATE))
        idx["mfcc"], _ = mfcc_site_frames(frames["mfcc"].shape[-2], t0, t1, t0)
        return idx


def open_arrays(out, n, spans):
    import numpy as np

    arrays = {}
    for rep, shape in (("wavjepa", (n, 13, 768)), ("wav2vec2", (n, 13, 768)),
                       ("mfcc", (n, 1, 39))):
        for span in spans:
            a = np.lib.format.open_memmap(out / f"{rep}_{span}.npy", mode="w+",
                                          dtype=np.float32, shape=shape)
            a[:] = np.nan
            arrays[rep, span] = a
    return arrays


def read_track(speaker, track):
    import soundfile as sf
    import torch

    wav, sr = sf.read(CORPUS / speaker / f"{track}.wav", dtype="float32")
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{track}.wav is {sr} Hz, expected {SAMPLE_RATE}")
    if wav.ndim != 1:
        raise SystemExit(f"{track}.wav has shape {wav.shape}, expected mono")
    return wav, torch.from_numpy(wav), len(wav) / SAMPLE_RATE * 1000


def finish(out, arrays, spans, ids, n_frames, encoded, manifest):
    """Flush, check for inf, write frames.tsv, site_ids.txt and the manifest."""
    import numpy as np

    for (rep, span), a in arrays.items():
        a.flush()
        if np.isinf(a).any():
            raise SystemExit(f"{rep}_{span} contains inf")
    cols = [f"n_{rep}_{span}" for rep in REPS for span in spans]
    with (out / "frames.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["site_id", *cols])
        for i in sorted(encoded):
            w.writerow([ids[i], *(int(n_frames[rep, span][i]) or ""
                                  for rep in REPS for span in spans)])
    (out / "site_ids.txt").write_text("".join(s + "\n" for s in ids))
    rows = np.array(sorted(encoded))
    medians = {}
    for rep in REPS:
        for span in spans:
            k = n_frames[rep, span][rows] if len(rows) else n_frames[rep, span][:0]
            k = k[k > 0]
            medians[f"{rep}_{span}"] = float(np.median(k)) if len(k) else None
    manifest["median_frames"] = medians
    (out / "extract-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for rep in REPS:
        print(f"  median frames, {rep:<8} " + "  ".join(
            f"{span} {manifest['median_frames'][f'{rep}_{span}']}" for span in spans))
    print(f"wrote {out}", flush=True)


def base_manifest(enc, limit):
    from capstone.predictions import code_commit

    return {
        "code_commit": code_commit(),
        "checkpoint_revisions": enc.revisions,
        "context_ms": CONTEXT_MS,
        "sample_rate": SAMPLE_RATE,
        "frame_dtype": FRAME_DTYPE,
        "stored_dtype": "float32",
        "pooling": "mean",
        "mfcc": {**MFCC, "delta_width": DELTA_WIDTH, "dims": 39,
                 "librosa": __import__("librosa").__version__},
        "limit": limit,
    }


# ------------------------------------------------------------------- stops


def read_stops():
    """The stops TSV's rows, after checking its recorded hashes against them."""
    with STOPS_TSV.open() as fh:
        inv_line, ids_line = fh.readline(), fh.readline()
        rows = list(csv.DictReader(fh, delimiter="\t"))
    recorded = json.loads(ids_line.split("=", 1)[1])
    for thr, want in recorded.items():
        col = f"in_{int(round(float(thr) * 100)):03d}"
        ids = sorted(r["site_id"] for r in rows if r[col] == "1")
        got = hashlib.sha256("".join(f"{s}\n" for s in ids).encode()).hexdigest()
        if got != want:
            raise SystemExit(f"{STOPS_TSV}: the {thr} set hashes to {got[:12]}, "
                             f"its header says {want[:12]}")
    if len(rows) != N_STOPS:
        raise SystemExit(f"{STOPS_TSV} has {len(rows):,} rows, expected {N_STOPS:,}")
    return rows, inv_line.strip().split("=", 1)[1], recorded


def extract_stops(enc, limit):
    import numpy as np

    from capstone.encoders.extraction import plan_excerpt

    stops, inv_hash, set_hashes = read_stops()
    n = len(stops)
    tsv_hash = hashlib.sha256(STOPS_TSV.read_bytes()).hexdigest()
    out = EMBEDDINGS / f"transfer-stops-{tsv_hash[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nSTOPS: {n:,} stops -> {out}", flush=True)
    arrays = open_arrays(out, n, STOP_SPANS)
    n_frames = {(rep, span): np.zeros(n, dtype=np.int32) for rep in REPS for span in STOP_SPANS}

    by_track: dict[str, list[int]] = {}
    for i, s in enumerate(stops):
        by_track.setdefault(s["track"], []).append(i)

    done, encoded, skipped_clamped = 0, [], []
    for track, members in sorted(by_track.items()):
        wav, wav_t, rec_ms = read_track(stops[members[0]]["speaker"], track)
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

            frames, n_clean = enc.encode(wav, wav_t, ex)
            for span, (a, b) in spans.items():
                for rep, sel in enc.span_idx(frames, n_clean, ex, a, b).items():
                    if len(sel) == 0:
                        raise SystemExit(f"{s['site_id']}: no {rep} frame in span {span}")
                    arrays[rep, span][i] = mean_pool(frames[rep], sel)
                    n_frames[rep, span][i] = len(sel)
            encoded.append(i)
            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        print(f"  {track}: {done:,} stops", flush=True)

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
        "wavjepa_spans_trimmed_to_clean_frames": enc.n_clean_trimmed,
    }
    print(f"encoded {done:,} stops; skipped as clamped {len(skipped_clamped)}")
    finish(out, arrays, STOP_SPANS, [s["site_id"] for s in stops], n_frames, encoded, manifest)
    return 0


# ----------------------------------------------------------------- context


def extract_context(enc, limit):
    import numpy as np

    sites, inv_hash = read_sites()
    n = len(sites)
    plans, digest = plan_spans(sites)
    grid = [i for i in plans if plans[i][2]["in_grid"]]
    out = EMBEDDINGS / f"context-{inv_hash[:12]}"
    stored = EMBEDDINGS / f"spans-{inv_hash[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nCONTEXT: {len(grid):,} grid sites -> {out}", flush=True)
    arrays = open_arrays(out, n, CONTEXT_SPANS)
    r_new = {rep: np.full((n, *shape), np.nan, dtype=np.float32)
             for rep, shape in (("wavjepa", (13, 768)), ("wav2vec2", (13, 768)),
                                ("mfcc", (1, 39)))}
    n_frames = {(rep, span): np.zeros(n, dtype=np.int32)
                for rep in REPS for span in CONTEXT_SPANS}
    n_v_empty = {rep: 0 for rep in REPS}

    by_track: dict[str, list[int]] = {}
    for i in grid:
        by_track.setdefault(sites[i]["track"], []).append(i)

    done, encoded = 0, []
    for track, members in sorted(by_track.items()):
        wav, wav_t, rec_ms = read_track(sites[members[0]]["speaker"], track)
        for i in members:
            s = sites[i]
            ex, spans, _ = plans[i]
            if ex.offset_ms + ex.duration_ms > rec_ms:
                raise SystemExit(f"{s['site_id']}: excerpt runs past the recording, "
                                 "but the inventory says it is not clamped")
            frames, n_clean = enc.encode(wav, wav_t, ex)
            r_idx = enc.span_idx(frames, n_clean, ex, *spans["r"])
            vr_idx = enc.span_idx(frames, n_clean, ex, *spans["vr"])
            for rep in REPS:
                n_all = n_clean if rep == "wavjepa" else frames[rep].shape[-2]
                ctx = np.setdiff1d(np.arange(n_all), r_idx[rep])
                v = np.setdiff1d(vr_idx[rep], r_idx[rep])
                if len(ctx) == 0:
                    raise SystemExit(f"{s['site_id']}: no {rep} frame outside the flap")
                r_new[rep][i] = mean_pool(frames[rep], r_idx[rep])
                arrays[rep, "ctx"][i] = mean_pool(frames[rep], ctx)
                n_frames[rep, "ctx"][i] = len(ctx)
                if len(v):
                    arrays[rep, "v"][i] = mean_pool(frames[rep], v)
                    n_frames[rep, "v"][i] = len(v)
                else:
                    n_v_empty[rep] += 1
            encoded.append(i)
            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        print(f"  {track}: {done:,} sites", flush=True)

    # The re-pooled ɾ against the stored span grid's ɾ.
    rows = np.array(sorted(encoded))
    report, identical = {}, True
    for rep in REPS:
        old = np.load(stored / f"{rep}_r.npy", mmap_mode="r")[rows]
        new = r_new[rep][rows]
        same = int(np.all(old == new, axis=tuple(range(1, old.ndim))).sum())
        diff = float(np.abs(old - new).max())
        report[rep] = {"rows_identical": same, "rows": len(rows), "max_abs_diff": diff}
        identical &= same == len(rows)
        print(f"ɾ vs stored {rep}_r.npy: {same:,} of {len(rows):,} rows identical; "
              f"max |diff| {diff:.3e}")

    manifest = base_manifest(enc, limit) | {
        "part": "context",
        "inventory_sha256": inv_hash,
        "exclusion_sha256": digest,
        "spans": list(CONTEXT_SPANS),
        "n_sites": n,
        "n_grid": len(grid),
        "n_encoded": done,
        "v_empty": n_v_empty,
        "wavjepa_spans_trimmed_to_clean_frames": enc.n_clean_trimmed,
        "r_vs_stored": report,
    }
    print(f"encoded {done:,} sites; v empty {n_v_empty}")
    finish(out, arrays, CONTEXT_SPANS, [s["site_id"] for s in sites], n_frames, encoded,
           manifest)
    if not identical:
        print("\nɾ is NOT bit-identical to the stored span grid. Nothing downstream is "
              "read until this is understood.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("stops", "context", "all"), default="all")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    enc = Encoder()
    rc = 0
    if args.part in ("stops", "all"):
        rc |= extract_stops(enc, args.limit)
        enc.n_clean_trimmed = 0
    if args.part in ("context", "all"):
        rc |= extract_context(enc, args.limit)
    return rc


if __name__ == "__main__":
    sys.exit(main())
