"""The cropped design's inputs: pause-matched noise, ramps, and composition (span-grid §3).

WHAT §3 ASKS FOR, AND WHERE EACH PART LIVES HERE
    - Noise matched to the recording: `fit_pause_noise` takes a track's pauses
      (`SIL` intervals of at least 200 ms, `pause_intervals`) and keeps their
      long-term average power spectrum and RMS level. `synth_noise` draws
      Gaussian noise with that spectrum at that level. Synthesized, not cut
      from the pauses, which can hold breaths and the interviewer.
    - One noise seed per site, shared across the site's inputs: `site_seed`.
    - 2.5 ms raised-cosine ramps inside the kept speech: `compose`.
    - Fixed length L, the excerpt at its own position: `compose` lays the
      excerpt's kept samples at their excerpt indices in an L-sample buffer of
      noise, so the input starts with the excerpt's first sample and noise
      fills from the excerpt's end to L (§8, 2026-11-26).

    Normalization is not here. It belongs to each encoder's extractor
    (`input_norm` in encoders/extraction.py), computed on the full excerpt and
    applied to the whole composed input.

RAMP RULE
    A ramp sits at every edge where kept speech meets noise, inside the
    speech, so no sample outside the kept interval survives. An edge that is
    the input's own start has no noise before it and gets no ramp. The padded
    full excerpt (the padding-only control) therefore has one ramp, at the
    excerpt's end, and the flap-masked input has three: either side of the
    flap and at the excerpt's end.

No torch here, so the module and its tests run without the GPU stack.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

SAMPLE_RATE = 16_000
RAMP_MS = 2.5
RAMP_SAMPLES = int(round(RAMP_MS / 1000 * SAMPLE_RATE))   # 40
MIN_PAUSE_S = 0.2           # §3: SIL intervals of >= 200 ms
PAUSE_LABEL = "SIL"
N_FFT = 1024                # spectrum resolution, 15.6 Hz bins at 16 kHz
SEED_BASE = 20261126        # fixed with the build; one seed per site from it


@dataclass(frozen=True)
class PauseNoise:
    """A track's pause spectrum and level: what `synth_noise` reproduces.

    psd:  mean power per rfft bin of N_FFT-sample Hann frames, (N_FFT//2 + 1,)
    rms:  RMS of the pause samples themselves, in the recording's units
    seconds, n_intervals: how much pause the estimate rests on
    """

    psd: np.ndarray
    rms: float
    seconds: float
    n_intervals: int


def pause_intervals(phones, min_s: float = MIN_PAUSE_S, label: str = PAUSE_LABEL):
    """(start_s, end_s) of every `label` interval at least `min_s` long.

    `phones` are count_flap_sites.Phone objects (start, end, label, ...).
    A microsecond of slack keeps an interval of exactly min_s, which float
    subtraction of the tier's times can put a hair under it.
    """
    return [(p.start, p.end) for p in phones
            if p.label == label and p.end - p.start >= min_s - 1e-6]


def fit_pause_noise(wav: np.ndarray, intervals: Iterable[tuple[float, float]],
                    sr: int = SAMPLE_RATE, n_fft: int = N_FFT) -> PauseNoise:
    """Long-term average spectrum and RMS of `wav` over `intervals` (seconds).

    The spectrum is the mean of |rfft|^2 over Hann-windowed frames of n_fft
    samples, hop n_fft // 2, taken inside each interval only, so no frame
    straddles a pause's edge. The RMS is over every pause sample. Only the
    spectrum's shape is used (synth_noise rescales to `rms`), so the window's
    gain needs no correction.
    """
    hop = n_fft // 2
    win = np.hanning(n_fft).astype(np.float64)
    power = np.zeros(n_fft // 2 + 1)
    n_frames, sq, n_samples, n_int = 0, 0.0, 0, 0
    for s, e in intervals:
        a, b = int(round(s * sr)), int(round(e * sr))
        seg = np.asarray(wav[a:b], dtype=np.float64)
        if len(seg) == 0:
            continue
        n_int += 1
        sq += float(np.sum(seg ** 2))
        n_samples += len(seg)
        for k in range(0, len(seg) - n_fft + 1, hop):
            power += np.abs(np.fft.rfft(seg[k:k + n_fft] * win)) ** 2
            n_frames += 1
    if n_frames == 0:
        raise ValueError("no pause frame long enough for the spectrum: "
                         f"{n_int} intervals, {n_samples} samples")
    return PauseNoise(psd=power / n_frames, rms=float(np.sqrt(sq / n_samples)),
                      seconds=n_samples / sr, n_intervals=n_int)


def site_seed(site_id: str, base: int = SEED_BASE) -> int:
    """One deterministic seed per site, independent of processing order."""
    h = hashlib.sha256(f"{base}:{site_id}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def synth_noise(noise: PauseNoise, n: int, seed: int) -> np.ndarray:
    """n samples of Gaussian noise with `noise`'s spectrum, at `noise.rms`.

    White Gaussian noise is shaped in the frequency domain by the square root
    of the pause spectrum, interpolated onto this length's rfft bins, and the
    result is scaled to the pauses' RMS exactly. float32, as the audio is.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    n_fft = 2 * (len(noise.psd) - 1)
    f_src = np.fft.rfftfreq(n_fft)
    f_dst = np.fft.rfftfreq(n)
    shaped = np.fft.irfft(spec * np.sqrt(np.interp(f_dst, f_src, noise.psd)), n=n)
    shaped *= noise.rms / np.sqrt(np.mean(shaped ** 2))
    return shaped.astype(np.float32)


def ramp(n: int = RAMP_SAMPLES) -> np.ndarray:
    """A rising raised-cosine of n samples, strictly between 0 and 1."""
    k = np.arange(n) + 0.5
    return (0.5 - 0.5 * np.cos(np.pi * k / n)).astype(np.float64)


def compose(excerpt: np.ndarray, keep: Sequence[tuple[int, int]], noise: np.ndarray,
            ramp_n: int = RAMP_SAMPLES) -> np.ndarray:
    """An input of len(noise) samples: `excerpt`'s kept intervals over noise.

    keep:  sample intervals [a, b) of the excerpt, sorted and disjoint, each
           placed at its own excerpt indices. Everything else is `noise`.
    A raised-cosine crossfade of ramp_n samples runs inside each kept interval
    at every edge that meets noise (not at index 0, which nothing precedes).
    Within the ramp the output is w * speech + (1 - w) * noise. An interval
    shorter than two ramps gets the product of its rising and falling ramps.
    """
    n = len(noise)
    if len(excerpt) > n:
        raise ValueError(f"excerpt of {len(excerpt)} samples is longer than L = {n}")
    out = noise.astype(np.float64).copy()
    up = ramp(ramp_n)
    prev_end = 0
    for a, b in keep:
        if not 0 <= a < b <= len(excerpt) or a < prev_end:
            raise ValueError(f"bad keep interval [{a}, {b}) for {len(excerpt)} samples")
        prev_end = b
        w = np.ones(b - a)
        m = min(ramp_n, b - a)
        if a > 0:
            w[:m] *= up[:m]
        if b < n:
            w[-m:] *= up[:m][::-1]
        seg = excerpt[a:b].astype(np.float64)
        out[a:b] = w * seg + (1 - w) * out[a:b]
    return out.astype(np.float32)


def segment_norm_stats(x: np.ndarray, segment: int) -> tuple[float, float]:
    """Mean and std WavJEPA's per-segment `normalize` computes for a one-segment input.

    The model zero-pads its input to a whole number of segments and z-scores
    each segment, padding included (std with Bessel's correction, as torch's
    default). Only one segment is handled: every cropped input, L samples, is
    shorter than one.
    """
    if len(x) > segment:
        raise ValueError(f"{len(x)} samples is more than one segment ({segment})")
    padded = np.zeros(segment, dtype=np.float64)
    padded[:len(x)] = x
    return float(padded.mean()), float(padded.std(ddof=1))
