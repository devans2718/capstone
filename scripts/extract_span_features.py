#!/usr/bin/env python3
"""Span-pooled WavJEPA, wav2vec 2.0 and MFCC features: the span grid's pooled design.

HOW TO RUN
    docker compose run --rm dev python scripts/extract_span_features.py [--limit N]

    Reads the pilot's sites TSV (extract_site_features.py export-sites), which is
    hash-checked against the committed inventory, and the .phones tier, which
    measure_span_grid.locate_spans walks exactly as it does for the registered
    exclusion list. Stops if that list's sha256 is not the registered one.
    ~20-30 min on the RTX 3060 for both encoders.

WHAT IT DOES (span-grid README §2-§4, §8)
    Each site's excerpt is planned and encoded exactly as the pilot's was (pilot
    §2): the same plan_excerpt, the same samples, float16 frames (§8, 2026-10-18).
    It is encoded once per encoder; each span's frames are then pooled from that
    one encoding. Spans are clipped to the excerpt and clipped sites kept (§2).

    Frames belong to a span by each encoder's own rule, through window_frames on
    a copy of the excerpt whose target is the span: WavJEPA by onset, wav2vec 2.0
    by centre with §4's nearest-midpoint fallback. MFCC by centre, as the pilot.
    WavJEPA also drops any frame that does not read wholly real audio
    (n_clean_frames): its last returned frame is ~95% padding, and a following
    span clipped at the excerpt end would otherwise pool it (§8, 2026-10-18).

    Pooling is the mean, the only one registered (§4).

WHICH SITES
    ɾ for all 10,389 eligible sites, so that the built-in reproduction check
    (§5) runs on the pilot's set. Vɾ, ɾV and VɾV for the 10,334 span-grid sites;
    the 55 excluded have no preceding vowel, and their rows stay NaN. Rows are
    in inventory order, 10,392 of them; the 3 clamped sites are NaN throughout.

WHAT IT WRITES, under /data/embeddings/spans-<inventory hash[:12]>/
    {wavjepa,wav2vec2}_{r,vr,rv,vrv}.npy   (10392, 13, 768) float32
    mfcc_{r,vr,rv,vrv}.npy                 (10392, 1, 39)   float32
    spans.tsv          per site: clipped span bounds, clip flags, the three
                       durations, frames pooled per encoder and span
    site_ids.txt, extract-manifest.json

    Buckeye-derived, so /data only, never the repository.

WHAT IT CHECKS, AND STOPS ON
    Everything extract_site_features.py checks, plus: every span has a frame in
    every representation; the clipped-site counts are §2's (269 preceding, 442
    following); and, after the run, the ɾ span equals the pilot's mean-pooled
    features bit for bit. That last check is not §5's reproduction check, which
    is the probe curve; it is a cheaper one ahead of it. GPU inference is
    reproducible only to a tolerance (probe_frame_counts.py), so a failure is
    reported with its size and exits non-zero rather than being judged here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_site_features import (CONTEXT_MS, CORPUS, DELTA_WIDTH,  # noqa: E402
                                   EMBEDDINGS, MFCC, SAMPLE_RATE, mfcc_frames,
                                   mfcc_site_frames, read_sites)
from measure_span_grid import (EXPECTED_SHA256, clipped,  # noqa: E402
                               exclusion_digest, locate_spans)

SPANS = ("r", "vr", "rv", "vrv")
ENCODERS = ("wavjepa", "wav2vec2")
FRAME_DTYPE = "float16"          # the pilot's frame precision, kept (§8)
CLIPPED_PRE, CLIPPED_POST = 269, 442   # §2, registered


def mean_pool(frames, idx):
    """The pilot's mean, written as pool() writes it, so ɾ can match bit for bit."""
    return frames[..., idx, :].mean(axis=-2)


# ------------------------------------------------------------------ planning


def plan_spans(sites):
    """Every eligible site's excerpt and clipped spans, before any audio is read.

    Returns {inventory row: (excerpt, {span: (start_ms, end_ms)}, info)}, times in
    the recording's base. Span-grid sites get all four spans; the 55 excluded
    get ɾ alone.
    """
    from capstone.encoders.extraction import plan_excerpt

    eligible = [s for s in sites if not s["excerpt_clamped"]]
    kept, excluded = locate_spans(eligible)
    digest = exclusion_digest(excluded)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"exclusion list hashes to {digest[:12]}, registered "
                         f"{EXPECTED_SHA256[:12]}. The sites are not the span grid's.")
    context = {r["site_id"]: (v, f, clipped(r, v, f)) for r, v, _, f in kept}

    plans = {}
    n_pre = n_post = 0
    for i, s in enumerate(sites):
        if s["excerpt_clamped"]:
            continue
        # Recording length only matters for clamping, which the inventory has
        # already settled for these sites; the extract loop re-checks it.
        ex = plan_excerpt(s["start"] * 1000, s["end"] * 1000, float("inf"),
                          context_ms=CONTEXT_MS)
        lo, hi = ex.offset_ms, ex.offset_ms + ex.duration_ms
        flap = (s["start"] * 1000, s["end"] * 1000)
        spans = {"r": flap}
        info = {"in_grid": s["site_id"] in context}
        if info["in_grid"]:
            v, f, (clip_pre, clip_post) = context[s["site_id"]]
            v_start, f_end = v.start * 1000, f.end * 1000
            # The bounds clip by min/max; the flags are §2's own comparison.
            # Where the two could disagree, the bound moves by float rounding.
            pre, post = max(v_start, lo), min(f_end, hi)
            spans |= {"vr": (pre, flap[1]), "rv": (flap[0], post), "vrv": (pre, post)}
            info |= {
                "clipped_pre": clip_pre,
                "clipped_post": clip_post,
                "stretch_ms": flap[0] - v_start,
                "stretch_clipped_ms": flap[0] - pre,
                "vowel_ms": (v.end - v.start) * 1000,
            }
            n_pre += info["clipped_pre"]
            n_post += info["clipped_post"]
        plans[i] = (ex, spans, info)

    if (n_pre, n_post) != (CLIPPED_PRE, CLIPPED_POST):
        raise SystemExit(f"clipped sites {n_pre} preceding / {n_post} following; "
                         f"§2 registers {CLIPPED_PRE} / {CLIPPED_POST}")
    return plans, digest


# ---------------------------------------------------------------------- extract


def extract(limit: int | None) -> int:
    import numpy as np
    import soundfile as sf
    import torch

    from capstone.encoders.extraction import (REVISION, W2V2_REVISION,
                                              Wav2Vec2Extractor, WavJEPAExtractor,
                                              excerpt_samples)
    from capstone.predictions import code_commit

    sites, inv_hash = read_sites()
    n = len(sites)
    plans, digest = plan_spans(sites)
    out = EMBEDDINGS / f"spans-{inv_hash[:12]}"
    pilot = EMBEDDINGS / f"features-{inv_hash[:12]}"
    out.mkdir(parents=True, exist_ok=True)

    ext = {"wavjepa": WavJEPAExtractor.load(), "wav2vec2": Wav2Vec2Extractor.load()}
    jepa_geo = ext["wavjepa"].geometry
    print(f"device {ext['wavjepa'].device}; WavJEPA {REVISION}, wav2vec2 "
          f"{W2V2_REVISION}; {len(plans):,} sites planned -> {out}")

    arrays = {}
    for rep, shape in (("wavjepa", (n, 13, 768)), ("wav2vec2", (n, 13, 768)),
                       ("mfcc", (n, 1, 39))):
        for span in SPANS:
            a = np.lib.format.open_memmap(out / f"{rep}_{span}.npy", mode="w+",
                                          dtype=np.float32, shape=shape)
            a[:] = np.nan
            arrays[rep, span] = a
    n_frames = {(rep, span): np.zeros(n, dtype=np.int32)
                for rep in (*ENCODERS, "mfcc") for span in SPANS}
    n_clean_trimmed = 0     # WavJEPA spans that lost a padded frame
    n_w2v2_fallback = 0     # ɾ spans with no wav2vec2 frame centre (§4)

    by_track: dict[str, list[int]] = {}
    for i in plans:
        by_track.setdefault(sites[i]["track"], []).append(i)

    done, encoded = 0, []
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
            ex, spans, _ = plans[i]
            if ex.offset_ms + ex.duration_ms > rec_ms:
                raise SystemExit(f"{s['site_id']}: excerpt runs past the recording, "
                                 "but the inventory says it is not clamped")
            a, b = excerpt_samples(ex, SAMPLE_RATE)
            n_samples = b - a

            frames = {rep: e.encode(wav_t[a:b], dtype=FRAME_DTYPE).float().numpy()
                      for rep, e in ext.items()}          # (13, n, 768) each
            m = mfcc_frames(wav[a:b]).T[None]               # (1, n, 39)
            n_clean = jepa_geo.n_clean_frames(n_samples)

            # The pilot's own frame count, checked as extract_site_features does.
            flap_jepa = ext["wavjepa"].window_frames(ex, frames["wavjepa"].shape[-2])
            if len(flap_jepa) != s["n_frame_onsets"]:
                raise SystemExit(f"{s['site_id']}: {len(flap_jepa)} WavJEPA frames, "
                                 f"{s['n_frame_onsets']} in the inventory")

            for span, (start_ms, end_ms) in spans.items():
                exs = replace(ex, target_start_ms=start_ms, target_end_ms=end_ms)
                idx = {}
                for rep, e in ext.items():
                    rng = e.window_frames(exs, frames[rep].shape[-2])
                    if rep == "wavjepa" and rng.stop > n_clean:
                        rng = range(rng.start, max(rng.start, n_clean))
                        n_clean_trimmed += 1
                    idx[rep] = np.arange(rng.start, rng.stop)
                t0 = int(round(exs.target_start_local_ms / 1000 * SAMPLE_RATE))
                t1 = int(round(exs.target_end_local_ms / 1000 * SAMPLE_RATE))
                idx["mfcc"], _ = mfcc_site_frames(m.shape[-2], t0, t1, t0)
                if span == "r" and not len(ext["wav2vec2"].geometry.frames_for_interval(
                        exs.target_start_local_ms, exs.target_end_local_ms)):
                    n_w2v2_fallback += 1

                for rep, sel in idx.items():
                    if len(sel) == 0:
                        raise SystemExit(f"{s['site_id']}: no {rep} frame in span {span}")
                    src = m if rep == "mfcc" else frames[rep]
                    arrays[rep, span][i] = mean_pool(src, sel)
                    n_frames[rep, span][i] = len(sel)

            encoded.append(i)
            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        print(f"  {track}: {done:,} sites", flush=True)

    for key, a in arrays.items():
        a.flush()
        if np.isinf(a).any():
            raise SystemExit(f"{key[0]}_{key[1]} contains inf")

    # spans.tsv: Buckeye-derived, /data only.
    cols = ["site_id", "in_grid", "clipped_pre", "clipped_post", "stretch_ms",
            "stretch_clipped_ms", "vowel_ms"]
    cols += [f"{span}_{edge}_local_ms" for span in SPANS for edge in ("start", "end")]
    cols += [f"n_{rep}_{span}" for rep in (*ENCODERS, "mfcc") for span in SPANS]
    with (out / "spans.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for i in encoded:
            ex, spans, info = plans[i]
            row = [sites[i]["site_id"], int(info["in_grid"])]
            row += [(int(info[k]) if isinstance(info[k], bool) else repr(info[k]))
                    if k in info else "" for k in cols[2:7]]
            for span in SPANS:
                if span in spans:
                    row += [repr(spans[span][0] - ex.offset_ms),
                            repr(spans[span][1] - ex.offset_ms)]
                else:
                    row += ["", ""]
            row += [int(n_frames[rep, span][i]) or ""
                    for rep in (*ENCODERS, "mfcc") for span in SPANS]
            w.writerow(row)
    (out / "site_ids.txt").write_text("".join(s["site_id"] + "\n" for s in sites))

    # The ɾ span against the pilot's mean-pooled features.
    rows = np.array(sorted(encoded))
    report, identical = {}, True
    for rep, pilot_file in (("wavjepa", "wavjepa_mean.npy"), ("mfcc", "mfcc_mean.npy")):
        old = np.load(pilot / pilot_file, mmap_mode="r")[rows]
        new = arrays[rep, "r"][rows]
        same_rows = int(np.all(old == new, axis=tuple(range(1, old.ndim))).sum())
        diff = float(np.abs(old - new).max())
        report[rep] = {"rows_identical": same_rows, "rows": len(rows), "max_abs_diff": diff}
        identical &= same_rows == len(rows)
        print(f"\nɾ vs pilot {pilot_file}: {same_rows:,} of {len(rows):,} rows identical; "
              f"max |diff| {diff:.3e}")

    manifest = {
        "code_commit": code_commit(),
        "inventory_sha256": inv_hash,
        "exclusion_sha256": digest,
        "checkpoint_revisions": {"wavjepa": REVISION, "wav2vec2": W2V2_REVISION},
        "context_ms": CONTEXT_MS,
        "sample_rate": SAMPLE_RATE,
        "frame_dtype": FRAME_DTYPE,
        "stored_dtype": "float32",
        "pooling": "mean",
        "spans": list(SPANS),
        "mfcc": {**MFCC, "delta_width": DELTA_WIDTH, "dims": 39,
                 "librosa": __import__("librosa").__version__},
        "n_sites": n,
        "n_encoded": done,
        "limit": limit,
        "wavjepa_spans_trimmed_to_clean_frames": n_clean_trimmed,
        "wav2vec2_r_spans_on_fallback": n_w2v2_fallback,
        "r_vs_pilot": report,
    }
    (out / "extract-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nencoded {done:,} sites; WavJEPA spans trimmed to clean frames "
          f"{n_clean_trimmed:,}; wav2vec2 ɾ spans on the §4 fallback {n_w2v2_fallback:,}")
    for rep in (*ENCODERS, "mfcc"):
        counts = []
        for span in SPANS:
            k = n_frames[rep, span][rows]
            counts.append(f"{span} {int(np.median(k[k > 0])) if (k > 0).any() else '-'}")
        print(f"  median frames, {rep:<8} {'  '.join(counts)}")
    print(f"wrote {out}")
    if not identical:
        print("\nɾ is NOT bit-identical to the pilot. Nothing downstream is read until "
              "this is understood.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="encode only this many sites (a smoke test; the "
                         "manifest records it)")
    return extract(ap.parse_args().limit)


if __name__ == "__main__":
    sys.exit(main())
