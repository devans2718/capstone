#!/usr/bin/env python3
"""Pooled WavJEPA and MFCC features for every flap site (pre-registration §2, §4).

HOW TO RUN (two environments, in this order)
    .venv-app/bin/python scripts/extract_site_features.py export-sites
    docker compose run --rm dev python scripts/extract_site_features.py extract

    The CUDA image has torch, transformers and librosa but no pyarrow, and
    .venv-app has pyarrow but no torch. export-sites turns the committed
    parquet inventory into a plain TSV under /data, recording the parquet's
    sha256. extract reads that TSV, re-hashes the parquet in the repo, and
    refuses to run if the two disagree. It takes ~10-20 min on the RTX 3060.

WHAT IT WRITES, under /data/embeddings/features-<inventory hash[:12]>/
    wavjepa_{mean,max,midpoint}.npy   (10392, 13, 768) float32
    mfcc_{mean,max,midpoint}.npy      (10392, 1, 39)   float32
    site_ids.txt                      one per line, inventory order
    extract-manifest.json             commit, inventory hash, checkpoint, context

    Rows are in inventory order, all 10,392 of them. The 3 clamped sites are
    not encoded and their rows are NaN: they are excluded before scoring (§1),
    and a NaN that reaches a probe fails loudly where a zero would not.

    Stored as float32, but each WavJEPA frame is rounded to float16 first:
    encode_excerpt calls encode() at its default dtype, so a frame keeps
    about three significant digits before it is pooled. MFCC never passes
    through encode and is float32 throughout. *Corrected 2026-10-18: this
    said float32 removed the precision question; that held for the stored
    pools only.* No value overflowed float16 (the inf check below ran
    clean), and the rounding is kept: the span grid's reproduction check
    needs these exact features (span-grid README §8).

ONE EXCERPT, TWO REPRESENTATIONS
    Each site's excerpt is planned once: plan_excerpt on the FLOAT times (as
    retrieval.plan_clip does), then excerpt_samples. WavJEPA encodes it
    through encode_excerpt. MFCC is computed on exactly those samples. The
    frames are not the same and cannot be (§4). The reduction is the same: a
    mean over the frames that describe the site.

    WavJEPA frames belong to a site by ONSET (frames_for_interval). MFCC
    frames belong to it by CENTRE: frame t covers samples [160t, 160t + 400),
    with its centre at 160t + 200, and is the site's frame if that centre lies
    in [target start, target end). Midpoint pooling is the §7 clarification
    of 2026-09-26. For WavJEPA it is the last onset at or before the site's
    midpoint, and for MFCC the centre nearest it, with ties going to the
    earlier frame.

WHAT IT CHECKS, AND STOPS ON RATHER THAN SKIPS
    Every recording is 16 kHz. Each site's clamp flag and WavJEPA frame count
    match the inventory's excerpt_clamped and n_frame_onsets. Every eligible
    site has at least one frame of each kind. The midpoint frame lies inside
    the site. No output value is inf. A failure means the geometry this run
    uses is not the geometry the inventory was built with, and every number
    downstream would be about something else.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "flapping" / "sites-20260924.parquet"
CORPUS = Path("/data/corpora/buckeye")
EMBEDDINGS = Path("/data/embeddings")
SITES_TSV = EMBEDDINGS / "sites-20260924.tsv"

SAMPLE_RATE = 16_000
CONTEXT_MS = 200.0
POOLINGS = ("mean", "max", "midpoint")

# §4, registered. Frame t spans samples [HOP*t, HOP*t + WIN) of the excerpt.
MFCC = dict(n_mfcc=13, n_fft=400, win_length=400, hop_length=160,
            window="hann", n_mels=40, center=False)
HOP, WIN = 160, 400
DELTA_WIDTH = 9

COLUMNS = ("site_id", "speaker", "track", "start", "end",
           "excerpt_clamped", "n_frame_onsets")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------- export-sites


def export_sites() -> int:
    import pyarrow.parquet as pq

    inv = pq.read_table(INVENTORY, columns=list(COLUMNS)).to_pandas()
    SITES_TSV.parent.mkdir(parents=True, exist_ok=True)
    with SITES_TSV.open("w", newline="") as fh:
        fh.write(f"# inventory_sha256={sha256(INVENTORY)}\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in inv.itertuples(index=False):
            # repr() on the float times: the full double round-trips exactly,
            # and plan_excerpt must see the same float the inventory holds.
            w.writerow([r.site_id, r.speaker, r.track, repr(float(r.start)),
                        repr(float(r.end)), int(r.excerpt_clamped), int(r.n_frame_onsets)])
    print(f"wrote {SITES_TSV}: {len(inv):,} sites")
    return 0


def read_sites():
    with SITES_TSV.open() as fh:
        header = fh.readline().strip()
        recorded = header.split("=", 1)[1]
        if recorded != sha256(INVENTORY):
            raise SystemExit(
                f"{SITES_TSV} was exported from inventory {recorded[:12]}, but "
                f"{INVENTORY.name} now hashes differently. Re-run export-sites."
            )
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["start"], r["end"] = float(r["start"]), float(r["end"])
        r["excerpt_clamped"] = r["excerpt_clamped"] == "1"
        r["n_frame_onsets"] = int(r["n_frame_onsets"])
    return rows, recorded


# ---------------------------------------------------------------------- pooling


def pool(frames, idx, mid):
    """(mean, max, midpoint) over frames[..., idx, :]. `mid` indexes frames."""
    sel = frames[..., idx, :]
    return sel.mean(axis=-2), sel.max(axis=-2), frames[..., mid, :]


def mfcc_frames(samples):
    """(39, n_frames) MFCC + deltas over the whole excerpt, as registered."""
    import librosa
    import numpy as np

    m = librosa.feature.mfcc(y=samples, sr=SAMPLE_RATE, **MFCC)
    d1 = librosa.feature.delta(m, width=DELTA_WIDTH, order=1)
    d2 = librosa.feature.delta(m, width=DELTA_WIDTH, order=2)
    return np.concatenate([m, d1, d2], axis=0)


def mfcc_site_frames(n_frames: int, start_s: int, end_s: int, mid_s: float):
    """Frames whose centre lies in [start_s, end_s), and the one nearest mid_s."""
    import numpy as np

    centres = HOP * np.arange(n_frames) + WIN / 2
    idx = np.flatnonzero((centres >= start_s) & (centres < end_s))
    if len(idx) == 0:
        return idx, None
    mid = int(idx[np.argmin(np.abs(centres[idx] - mid_s))])  # argmin: first on a tie
    return idx, mid


# ---------------------------------------------------------------------- extract


def extract(limit: int | None) -> int:
    import numpy as np
    import soundfile as sf
    import torch

    from capstone.encoders.extraction import (N_LAYERS, REVISION, WIDTH,
                                              WavJEPAExtractor, excerpt_samples,
                                              plan_excerpt)
    from capstone.predictions import code_commit

    sites, inv_hash = read_sites()
    n = len(sites)
    out = EMBEDDINGS / f"features-{inv_hash[:12]}"
    out.mkdir(parents=True, exist_ok=True)

    ext = WavJEPAExtractor.load()
    geo = ext.geometry
    print(f"device {ext.device}; checkpoint {REVISION}; {n:,} sites -> {out}")

    arrays = {}
    for rep, shape in (("wavjepa", (n, N_LAYERS, WIDTH)), ("mfcc", (n, 1, 39))):
        for p in POOLINGS:
            a = np.lib.format.open_memmap(out / f"{rep}_{p}.npy", mode="w+",
                                          dtype=np.float32, shape=shape)
            a[:] = np.nan
            arrays[rep, p] = a
    n_mfcc_frames = np.zeros(n, dtype=np.int32)

    by_track: dict[str, list[int]] = {}
    for i, s in enumerate(sites):
        by_track.setdefault(s["track"], []).append(i)

    done = 0
    for track, members in sorted(by_track.items()):
        speaker = sites[members[0]]["speaker"]
        wav, sr = sf.read(CORPUS / speaker / f"{track}.wav", dtype="float32")
        if sr != SAMPLE_RATE:
            raise SystemExit(f"{track}.wav is {sr} Hz, expected {SAMPLE_RATE}")
        if wav.ndim != 1:
            raise SystemExit(f"{track}.wav has shape {wav.shape}, expected mono")
        rec_ms = len(wav) / SAMPLE_RATE * 1000
        wav_t = torch.from_numpy(wav)

        for i in members:
            s = sites[i]
            ex = plan_excerpt(s["start"] * 1000, s["end"] * 1000, rec_ms,
                              context_ms=CONTEXT_MS)
            if ex.clamped != s["excerpt_clamped"]:
                raise SystemExit(f"{s['site_id']}: clamped={ex.clamped} here, "
                                 f"{s['excerpt_clamped']} in the inventory")
            if ex.clamped:
                continue                       # excluded before scoring; row stays NaN

            frames, target = ext.encode_excerpt(wav_t, ex, SAMPLE_RATE)
            frames = frames.float().numpy()    # (13, n, 768)
            n_fr = frames.shape[-2]
            if len(target) != s["n_frame_onsets"] or len(target) == 0:
                raise SystemExit(f"{s['site_id']}: {len(target)} frames here, "
                                 f"{s['n_frame_onsets']} in the inventory")
            if target.stop > n_fr:
                raise SystemExit(f"{s['site_id']}: target frames {target} run past "
                                 f"the {n_fr} frames encoded")

            mid_ms = (ex.target_start_local_ms + ex.target_end_local_ms) / 2
            mid = geo.sample_to_frame(int(round(mid_ms / 1000 * SAMPLE_RATE)))
            if mid not in target:
                raise SystemExit(f"{s['site_id']}: midpoint frame {mid} not in {target}")
            idx = np.arange(target.start, target.stop)
            for p, v in zip(POOLINGS, pool(frames, idx, mid)):
                arrays["wavjepa", p][i] = v

            a, b = excerpt_samples(ex, SAMPLE_RATE)
            m = mfcc_frames(wav[a:b])
            t0 = int(round(ex.target_start_local_ms / 1000 * SAMPLE_RATE))
            t1 = int(round(ex.target_end_local_ms / 1000 * SAMPLE_RATE))
            midx, mmid = mfcc_site_frames(m.shape[1], t0, t1,
                                          mid_ms / 1000 * SAMPLE_RATE)
            if mmid is None:
                raise SystemExit(f"{s['site_id']}: no MFCC frame centre in "
                                 f"[{t0}, {t1}) samples. §4 says this cannot happen.")
            n_mfcc_frames[i] = len(midx)
            for p, v in zip(POOLINGS, pool(m.T[None], midx, mmid)):
                arrays["mfcc", p][i] = v

            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        print(f"  {track}: {done:,} sites", flush=True)

    eligible = np.array([not s["excerpt_clamped"] for s in sites])
    for (rep, p), a in arrays.items():
        a.flush()
        if np.isinf(a).any():
            raise SystemExit(f"{rep}_{p} contains inf")
        if not limit and np.isnan(a[eligible]).any():
            raise SystemExit(f"{rep}_{p}: an eligible site has no features")

    np.save(out / "mfcc_n_frames.npy", n_mfcc_frames)
    (out / "site_ids.txt").write_text("".join(s["site_id"] + "\n" for s in sites))
    manifest = {
        "code_commit": code_commit(),
        "inventory_sha256": inv_hash,
        "checkpoint_revision": REVISION,
        "context_ms": CONTEXT_MS,
        "sample_rate": SAMPLE_RATE,
        "mfcc": {**MFCC, "delta_width": DELTA_WIDTH, "dims": 39, "librosa": __import__("librosa").__version__},
        "poolings": list(POOLINGS),
        "n_sites": n,
        "n_encoded": int(done),
        "n_clamped_not_encoded": int((~eligible).sum()),
        "limit": limit,
        "dtype": "float32",
    }
    (out / "extract-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    elig_frames = n_mfcc_frames[eligible]
    print(f"\nencoded {done:,} sites; {int((~eligible).sum())} clamped left NaN")
    if not limit:
        print(f"MFCC frames per site: min {elig_frames.min()}, "
              f"median {int(np.median(elig_frames))}, max {elig_frames.max()}")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("export-sites", "extract"))
    ap.add_argument("--limit", type=int, default=None,
                    help="encode only this many sites (a smoke test; the "
                         "manifest records it)")
    args = ap.parse_args()
    return export_sites() if args.step == "export-sites" else extract(args.limit)


if __name__ == "__main__":
    sys.exit(main())
