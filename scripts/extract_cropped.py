#!/usr/bin/env python3
"""The cropped design, the padding-only control and the flap-masked variant (span-grid §3, §8).

HOW TO RUN
    docker compose run --rm dev python scripts/extract_cropped.py [--limit N]

    --limit N: a smoke test on the first N grid sites (track order). It writes
    to a separate cropped-*-smoke/ directory, never over a full run's output.

WHAT IT BUILDS, PER GRID SITE (§3; §8 of 2026-10-18 and 2026-11-26)
    The site's excerpt (flap ± 200 ms, unclamped for every grid site) and one
    noise draw of L samples, shaped to the track's pauses (capstone.cropped),
    seeded by the site. From them, six inputs of exactly L samples, each the
    excerpt's kept samples at their own indices over that noise:

        crop_{r,vr,rv,vrv}  only that span's samples kept       (the cropped design)
        pad                 the whole excerpt kept               (the padding-only control)
        mask                the whole excerpt but the flap kept  (the flap-masked variant)

    with 2.5 ms raised-cosine ramps inside the kept speech wherever it meets
    noise. Each encoder's input normalization is computed once, on the full
    excerpt, and applied to every input with the feature extractor skipped
    (Extractor.input_norm). MFCC reads the composed inputs as they are.

    L is recomputed from the grid's excerpts and must be 533.9 ms to the
    tenth of a millisecond (§3); the script stops otherwise.

WHAT IT POOLS (mean over frames; float16 frames, stored float32)
    crop_{span}  each span from its own crop, by §4's frame rules
    pad_{span}   the four spans and ctx from the padded full excerpt
    mask_ctx     ctx from the masked input
    Frames are located in the padded input exactly as in the excerpt. ctx is
    every frame of the excerpt outside that encoder's ɾ frames (§8,
    2026-11-23): WavJEPA's clean frames only, wav2vec 2.0-type and MFCC frames
    that lie within the excerpt, so no pooled frame starts in the end padding
    and no WavJEPA frame reads it.

THE CHECK, ON EVERY SITE (§3: "verified before the run")
    The full excerpt, unpadded, through the same normalization path, pooled
    over ɾ, has to equal the stored pooled design's ɾ (spans-*/, conv-spans-*/)
    bit for bit, for every encoder and MFCC. A mismatch is counted, reported
    with its size, and makes the script exit non-zero. The same forward pass
    gives the GroupNorm reference.

THE GROUPNORM MEASURE (§8, 2026-10-18; reported, not compared)
    Conv layer 0's GroupNorm, one group per channel, standardizes each channel
    over the whole input. Its per-channel mean and standard deviation are
    captured for the full excerpt and each of the six inputs, and per site
    and encoder the script stores the median over channels of
    |log(σ_in / σ_full)| and of |μ_in − μ_full| / σ_full.
    WavJEPA also z-scores each 2.01 s segment, zero padding included, inside
    the model; that segment's mean and std are stored per input too.

NOISE SOURCE PER TRACK
    §3: SIL intervals of at least 200 ms in that track. A track with less
    than MIN_PAUSE_SECONDS of them uses its speaker's tracks pooled instead
    (§8, 2026-11-26, build details); which tracks did is recorded.

WHAT IT WRITES (Buckeye-derived, so /data only, never the repository)
    /data/embeddings/cropped-<inventory sha256[:12]>[-smoke]/
        {rep}_{pool}.npy      (10392, 13, 768) float32; mfcc (10392, 1, 39)
        groupnorm.npy         (10392, 4 encoders, 6 inputs, 2) float32
        segment_norm.npy      (10392, 7 inputs incl. full, 2) float32, WavJEPA
        frames.tsv, site_ids.txt, noise-sources.tsv, extract-manifest.json
    Rows follow the inventory's order; rows not encoded are NaN.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_site_features import (CONTEXT_MS, CORPUS, DELTA_WIDTH,  # noqa: E402
                                   EMBEDDINGS, MFCC, SAMPLE_RATE, mfcc_frames,
                                   mfcc_site_frames, read_sites)
from extract_span_features import FRAME_DTYPE, mean_pool, plan_spans  # noqa: E402

ENCODERS = ("wavjepa", "wav2vec2", "hubert", "wavlm")
REPS = (*ENCODERS, "mfcc")
SPANS = ("r", "vr", "rv", "vrv")
INPUTS = (*(f"crop_{s}" for s in SPANS), "pad", "mask")
POOLS = (*(f"crop_{s}" for s in SPANS), *(f"pad_{s}" for s in SPANS), "pad_ctx", "mask_ctx")
L_MS_REGISTERED = 533.9
MIN_PAUSE_SECONDS = 2.0
FALLBACK = "speaker"        # §8, 2026-11-26: pool the speaker's tracks
STORED = {"wavjepa": "spans", "wav2vec2": "spans", "mfcc": "spans",
          "hubert": "conv-spans", "wavlm": "conv-spans"}


# ------------------------------------------------------------------ encoders


class Encoders:
    """The four encoders at their pinned revisions, all normalizing their input."""

    def __init__(self):
        import torch

        from capstone.encoders.extraction import (CONV_ENCODERS, REVISION,
                                                  Wav2Vec2Extractor, WavJEPAExtractor)
        self.ext = {"wavjepa": WavJEPAExtractor.load()}
        for name in ENCODERS[1:]:
            self.ext[name] = Wav2Vec2Extractor.load_named(name, normalize=True)
        self.revisions = {"wavjepa": REVISION,
                          **{n: CONV_ENCODERS[n][1] for n in ENCODERS[1:]}}
        self._captured = {}
        for rep, e in self.ext.items():
            mod = next(m for m in e.model.modules() if isinstance(m, torch.nn.GroupNorm))
            if mod.num_groups != mod.num_channels:
                raise SystemExit(f"{rep}: first GroupNorm has {mod.num_groups} groups "
                                 f"for {mod.num_channels} channels, expected one per channel")
            self._captured[rep] = []
            mod.register_forward_pre_hook(self._keep(rep))
        print(f"device {self.ext['wavjepa'].device}; revisions {self.revisions}", flush=True)

    def _keep(self, rep):
        def hook(_module, args):
            x = args[0].detach().double()               # (1, C, T)
            self._captured[rep].append(
                (x.mean(dim=-1)[0].cpu().numpy(),
                 x.var(dim=-1, unbiased=False).add(0).sqrt()[0].cpu().numpy()))
        return hook

    def encode(self, rep, wav_t, norm):
        """(frames (13, n, 768) float32, GroupNorm (mean, std) per channel)."""
        self._captured[rep].clear()
        frames = self.ext[rep].encode(wav_t, dtype=FRAME_DTYPE, norm=norm).float().numpy()
        if len(self._captured[rep]) != 1:
            raise SystemExit(f"{rep}: GroupNorm fired {len(self._captured[rep])} times "
                             "for one input, expected once (one segment)")
        return frames, self._captured[rep][0]

    def n_excerpt_frames(self, rep, n_samples):
        """Frames of an excerpt of n_samples that read only its own audio."""
        g = self.ext[rep].geometry
        if rep == "wavjepa":
            return g.n_clean_frames(n_samples)
        return g.n_output_frames(n_samples)

    def span_idx(self, rep, n_frames, n_ex_frames, ex, start_ms, end_ms):
        """Indices of a span's frames, as extract_span_features selects them,
        never past the excerpt's own frames."""
        import numpy as np

        exs = replace(ex, target_start_ms=start_ms, target_end_ms=end_ms)
        rng = self.ext[rep].window_frames(exs, n_frames)
        if rep == "wavjepa" and rng.stop > n_ex_frames:
            rng = range(rng.start, max(rng.start, n_ex_frames))
        return np.arange(rng.start, min(rng.stop, n_ex_frames))


def mfcc_idx(n_ex_frames, ex, start_ms, end_ms):
    exs = replace(ex, target_start_ms=start_ms, target_end_ms=end_ms)
    t0 = int(round(exs.target_start_local_ms / 1000 * SAMPLE_RATE))
    t1 = int(round(exs.target_end_local_ms / 1000 * SAMPLE_RATE))
    idx, _ = mfcc_site_frames(n_ex_frames, t0, t1, t0)
    return idx


def gn_measure(stats, ref):
    """Median over channels of |log(σ/σ_full)| and |μ − μ_full| / σ_full."""
    import numpy as np

    (mu, sd), (mu0, sd0) = stats, ref
    return (float(np.median(np.abs(np.log(sd / sd0)))),
            float(np.median(np.abs(mu - mu0) / sd0)))


# --------------------------------------------------------------------- noise


def noise_sources(sites, grid, by_name):
    """{track: PauseNoise}, falling back to the speaker's tracks where a track is short."""
    import count_flap_sites as cfs
    import soundfile as sf

    from capstone.cropped import fit_pause_noise, pause_intervals

    tracks = sorted({sites[i]["track"] for i in grid})
    speaker_of = {sites[i]["track"]: sites[i]["speaker"] for i in grid}
    pauses = {}
    for trk in sorted(by_name):
        pauses[trk] = pause_intervals(cfs.read_phones(by_name[trk]))
    seconds = {t: sum(e - s for s, e in iv) for t, iv in pauses.items()}

    fits, rows = {}, []
    for trk in tracks:
        if seconds[trk] >= MIN_PAUSE_SECONDS:
            wav, _ = sf.read(CORPUS / speaker_of[trk] / f"{trk}.wav", dtype="float32")
            fits[trk] = fit_pause_noise(wav, pauses[trk])
            rows.append((trk, "track", trk, fits[trk].seconds, fits[trk].n_intervals))
            continue
        spk = speaker_of[trk]
        donors = [t for t in sorted(by_name) if t.startswith(spk) and seconds[t] > 0]
        fits[trk] = _fit_many(spk, donors, pauses)
        rows.append((trk, FALLBACK, ",".join(donors), fits[trk].seconds,
                     fits[trk].n_intervals))
    return fits, rows


def _fit_many(spk, donors, pauses):
    """One spectrum and level over several tracks' pauses, weighted by frames/samples."""
    import numpy as np
    import soundfile as sf

    from capstone.cropped import PauseNoise, fit_pause_noise

    parts = []
    for t in donors:
        wav, _ = sf.read(CORPUS / spk / f"{t}.wav", dtype="float32")
        try:
            parts.append(fit_pause_noise(wav, pauses[t]))
        except ValueError:
            continue
    if not parts:
        raise SystemExit(f"speaker {spk}: no pause long enough in any track")
    secs = np.array([p.seconds for p in parts])
    psd = sum(p.psd * s for p, s in zip(parts, secs)) / secs.sum()
    rms = float(np.sqrt(sum(p.rms ** 2 * s for p, s in zip(parts, secs)) / secs.sum()))
    return PauseNoise(psd=psd, rms=rms, seconds=float(secs.sum()),
                      n_intervals=sum(p.n_intervals for p in parts))


# ---------------------------------------------------------------------- main


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch

    from capstone.cropped import RAMP_SAMPLES, SEED_BASE, compose, segment_norm_stats, \
        site_seed, synth_noise
    from capstone.encoders.extraction import excerpt_samples
    from capstone.predictions import code_commit

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    t_start = time.time()

    sites, inv_hash = read_sites()
    n = len(sites)
    plans, digest = plan_spans(sites)
    grid = [i for i in plans if plans[i][2]["in_grid"]]
    bounds = {i: excerpt_samples(plans[i][0], SAMPLE_RATE) for i in grid}
    L = max(b - a for a, b in bounds.values())
    if round(L / SAMPLE_RATE * 1000, 1) != L_MS_REGISTERED or any(plans[i][0].clamped for i in grid):
        raise SystemExit(f"L = {L} samples ({L / 16:.3f} ms), registered {L_MS_REGISTERED} ms; "
                         "or a grid excerpt is clamped. Nothing is built.")
    print(f"{len(grid):,} grid sites; L = {L} samples ({L / 16:.3f} ms)", flush=True)

    by_name = {p.stem: p for p in CORPUS.rglob("*.phones")}
    fits, noise_rows = noise_sources(sites, grid, by_name)
    fell_back = [r for r in noise_rows if r[1] != "track"]
    print(f"noise: {len(fits)} tracks, {len(fell_back)} from the speaker's tracks: "
          f"{[r[0] for r in fell_back]}", flush=True)

    enc = Encoders()
    segment = enc.ext["wavjepa"].geometry.target_length   # samples per 2.01 s segment

    out = EMBEDDINGS / (f"cropped-{inv_hash[:12]}" + ("-smoke" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for rep in REPS:
        shape = (n, 1, 39) if rep == "mfcc" else (n, 13, 768)
        for pool in POOLS:
            a = np.lib.format.open_memmap(out / f"{rep}_{pool}.npy", mode="w+",
                                          dtype=np.float32, shape=shape)
            a[:] = np.nan
            arrays[rep, pool] = a
    gn = np.full((n, len(ENCODERS), len(INPUTS), 2), np.nan, dtype=np.float32)
    seg = np.full((n, len(INPUTS) + 1, 2), np.nan, dtype=np.float32)
    n_frames = {(rep, pool): np.zeros(n, dtype=np.int32) for rep in REPS for pool in POOLS}
    stored = {rep: np.load(EMBEDDINGS / f"{STORED[rep]}-{inv_hash[:12]}" / f"{rep}_r.npy",
                           mmap_mode="r") for rep in REPS}
    mismatch = {rep: [0, 0.0] for rep in REPS}

    by_track: dict[str, list[int]] = {}
    for i in grid:
        by_track.setdefault(sites[i]["track"], []).append(i)

    done, encoded = 0, []
    for track, members in sorted(by_track.items()):
        spk = sites[members[0]]["speaker"]
        wav, sr = sf.read(CORPUS / spk / f"{track}.wav", dtype="float32")
        if sr != SAMPLE_RATE or wav.ndim != 1:
            raise SystemExit(f"{track}.wav: {sr} Hz, shape {wav.shape}")
        for i in members:
            s = sites[i]
            ex, spans, _ = plans[i]
            a, b = bounds[i]
            x = wav[a:b]
            n_ex = b - a
            local = {k: (int(round((lo - ex.offset_ms) / 1000 * SAMPLE_RATE)),
                         int(round((hi - ex.offset_ms) / 1000 * SAMPLE_RATE)))
                     for k, (lo, hi) in spans.items()}
            noise = synth_noise(fits[track], L, site_seed(s["site_id"]))
            f0, f1 = local["r"]
            keeps = {**{f"crop_{k}": [local[k]] for k in SPANS},
                     "pad": [(0, n_ex)], "mask": [(0, f0), (f1, n_ex)]}
            inputs = {k: compose(x, keep, noise) for k, keep in keeps.items()}

            # MFCC: no normalization, frames of the excerpt only.
            mf_full = mfcc_frames(x).T[None]
            n_mf_ex = mf_full.shape[-2]
            r_idx = mfcc_idx(n_mf_ex, ex, *spans["r"])
            got = mean_pool(mf_full, r_idx)
            _check(mismatch, "mfcc", got, stored["mfcc"][i])
            mf = {k: mfcc_frames(v).T[None] for k, v in inputs.items()}
            _pool_all(arrays, n_frames, "mfcc", i, mf, n_mf_ex, ex, spans,
                      lambda *a_: mfcc_idx(*a_[1:]))

            # WavJEPA's in-model segment normalization: its stats per input.
            g = enc.ext["wavjepa"].input_norm(torch.from_numpy(x)).gain
            seg[i, 0] = segment_norm_stats(x * np.float32(g), segment)
            for j, k in enumerate(INPUTS):
                seg[i, j + 1] = segment_norm_stats(inputs[k] * np.float32(g), segment)

            for e_i, rep in enumerate(ENCODERS):
                norm = enc.ext[rep].input_norm(torch.from_numpy(x))
                full, ref = enc.encode(rep, torch.from_numpy(x), norm)
                n_ex_fr = enc.n_excerpt_frames(rep, n_ex)
                ridx = enc.span_idx(rep, full.shape[-2], n_ex_fr, ex, *spans["r"])
                _check(mismatch, rep, mean_pool(full, ridx), stored[rep][i])
                fr = {}
                for j, k in enumerate(INPUTS):
                    fr[k], st = enc.encode(rep, torch.from_numpy(inputs[k]), norm)
                    gn[i, e_i, j] = gn_measure(st, ref)
                _pool_all(arrays, n_frames, rep, i, fr, n_ex_fr, ex, spans,
                          lambda nf, nex, ex_, lo, hi, rep=rep:
                          enc.span_idx(rep, nf, nex, ex_, lo, hi))
            encoded.append(i)
            done += 1
            if args.limit and done >= args.limit:
                break
        print(f"  {track}: {done:,} sites, {time.time() - t_start:,.0f} s", flush=True)
        if args.limit and done >= args.limit:
            break

    for a in arrays.values():
        a.flush()
        if np.isinf(a).any():
            raise SystemExit("inf in the output")
    np.save(out / "groupnorm.npy", gn)
    np.save(out / "segment_norm.npy", seg)
    ids = [s["site_id"] for s in sites]
    (out / "site_ids.txt").write_text("".join(x_ + "\n" for x_ in ids))
    with (out / "frames.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["site_id", *(f"n_{r}_{p}" for r in REPS for p in POOLS)])
        for i in sorted(encoded):
            w.writerow([ids[i], *(int(n_frames[r, p][i]) for r in REPS for p in POOLS)])
    with (out / "noise-sources.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["track", "source", "tracks_used", "pause_seconds", "n_intervals"])
        w.writerows(noise_rows)

    rows = np.array(sorted(encoded))
    gn_summary = {}
    for e_i, rep in enumerate(ENCODERS):
        for j, k in enumerate(INPUTS):
            v = gn[rows, e_i, j]
            gn_summary[f"{rep}_{k}"] = {
                "log_sd_median": float(np.median(v[:, 0])), "log_sd_p95": float(np.percentile(v[:, 0], 95)),
                "mu_median": float(np.median(v[:, 1])), "mu_p95": float(np.percentile(v[:, 1], 95))}
    manifest = {
        "code_commit": code_commit(),
        "checkpoint_revisions": enc.revisions,
        "inventory_sha256": inv_hash,
        "exclusion_sha256": digest,
        "context_ms": CONTEXT_MS, "sample_rate": SAMPLE_RATE,
        "L_samples": L, "L_ms": L / SAMPLE_RATE * 1000,
        "ramp_samples": RAMP_SAMPLES, "seed_base": SEED_BASE,
        "min_pause_seconds_per_track": MIN_PAUSE_SECONDS, "fallback": FALLBACK,
        "fallback_tracks": [r[0] for r in fell_back],
        "frame_dtype": FRAME_DTYPE, "stored_dtype": "float32", "pooling": "mean",
        "mfcc": {**MFCC, "delta_width": DELTA_WIDTH, "dims": 39,
                 "librosa": __import__("librosa").__version__},
        "inputs": list(INPUTS), "pools": list(POOLS), "encoders": list(ENCODERS),
        "limit": args.limit, "n_grid": len(grid), "n_encoded": done,
        "r_vs_stored": {rep: {"rows_differing": m[0], "max_abs_diff": m[1]}
                        for rep, m in mismatch.items()},
        "groupnorm": gn_summary,
        "median_frames": {f"{r}_{p}": float(np.median(n_frames[r, p][rows]))
                          for r in REPS for p in POOLS},
        "seconds": round(time.time() - t_start),
    }
    (out / "extract-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"encoded {done:,} sites in {manifest['seconds']:,} s -> {out}")
    for rep, m in mismatch.items():
        print(f"  full excerpt ɾ vs stored {rep}_r: {m[0]} of {done:,} rows differ, "
              f"max |diff| {m[1]:.3e}")
    for k, v in gn_summary.items():
        print(f"  GroupNorm {k:<20} |log σ| median {v['log_sd_median']:.3f} p95 "
              f"{v['log_sd_p95']:.3f}; |Δμ|/σ median {v['mu_median']:.3f}")
    bad = any(m[0] for m in mismatch.values())
    if bad:
        print("\nThe full excerpt through the cropped path does NOT reproduce the stored "
              "pooled design. Nothing built here is read until that is understood.")
    return 1 if bad else 0


def _check(mismatch, rep, got, want):
    import numpy as np

    want = np.asarray(want)
    if not np.array_equal(got, want):
        mismatch[rep][0] += 1
        mismatch[rep][1] = max(mismatch[rep][1], float(np.nanmax(np.abs(got - want))))


def _pool_all(arrays, n_frames, rep, i, frames, n_ex_frames, ex, spans, idx_fn):
    """Pool the crops, the padded control's spans and ctx, and the masked ctx."""
    import numpy as np

    def put(pool, fr, sel):
        if len(sel) == 0:
            raise SystemExit(f"row {i}: no {rep} frame for {pool}")
        arrays[rep, pool][i] = mean_pool(fr, sel)
        n_frames[rep, pool][i] = len(sel)

    nf = {k: v.shape[-2] for k, v in frames.items()}
    for sp in SPANS:
        put(f"crop_{sp}", frames[f"crop_{sp}"],
            idx_fn(nf[f"crop_{sp}"], n_ex_frames, ex, *spans[sp]))
        put(f"pad_{sp}", frames["pad"], idx_fn(nf["pad"], n_ex_frames, ex, *spans[sp]))
    r_pad = idx_fn(nf["pad"], n_ex_frames, ex, *spans["r"])
    r_mask = idx_fn(nf["mask"], n_ex_frames, ex, *spans["r"])
    put("pad_ctx", frames["pad"], np.setdiff1d(np.arange(n_ex_frames), r_pad))
    put("mask_ctx", frames["mask"], np.setdiff1d(np.arange(n_ex_frames), r_mask))


if __name__ == "__main__":
    sys.exit(main())
