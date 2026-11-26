"""The cropped design's noise and composition (capstone.cropped), no GPU needed."""

import numpy as np
import pytest

from capstone.cropped import (RAMP_SAMPLES, PauseNoise, compose, fit_pause_noise,
                              pause_intervals, ramp, segment_norm_stats, site_seed,
                              synth_noise)

SR = 16_000


class _Phone:
    def __init__(self, start, end, label):
        self.start, self.end, self.label = start, end, label


def _lowpass_noise(n, seed=0):
    """Noise with most of its power below ~1 kHz: a moving average of white noise."""
    white = np.random.default_rng(seed).standard_normal(n + 15)
    return np.convolve(white, np.ones(16) / 16, mode="valid")[:n]


def test_ramp_is_monotone_and_inside_zero_one():
    r = ramp()
    assert len(r) == RAMP_SAMPLES == 40
    assert np.all(np.diff(r) > 0) and 0 < r[0] < r[-1] < 1
    assert np.allclose(r + r[::-1], 1)   # crossfade keeps a constant sum


def test_pause_intervals_keeps_long_sil_only():
    ph = [_Phone(0, 0.5, "SIL"), _Phone(0.5, 0.6, "SIL"), _Phone(0.6, 1.0, "VOCNOISE"),
          _Phone(1.0, 1.2, "SIL"), _Phone(1.2, 1.3, "ah")]
    assert pause_intervals(ph) == [(0, 0.5), (1.0, 1.2)]


def test_fit_reads_only_the_pauses():
    """Loud speech outside the intervals must not touch the level or spectrum."""
    quiet = 0.01 * _lowpass_noise(SR)
    wav = np.concatenate([quiet, 5 * np.random.default_rng(1).standard_normal(SR)])
    fit = fit_pause_noise(wav, [(0.0, 1.0)])
    assert fit.rms == pytest.approx(np.sqrt(np.mean(quiet ** 2)))
    assert fit.n_intervals == 1 and fit.seconds == pytest.approx(1.0)


def test_fit_refuses_without_a_whole_frame():
    with pytest.raises(ValueError):
        fit_pause_noise(np.zeros(SR), [(0.0, 0.01)])


def test_synth_matches_level_and_spectrum_shape():
    src = 0.02 * _lowpass_noise(20 * SR)
    fit = fit_pause_noise(src, [(0.0, 20.0)])
    out = synth_noise(fit, 8542, seed=7)
    assert out.dtype == np.float32 and len(out) == 8542
    assert np.sqrt(np.mean(out.astype(np.float64) ** 2)) == pytest.approx(fit.rms, rel=1e-5)
    # Shape: low-passed source, so the synthesis keeps most power below 1 kHz.
    p = np.abs(np.fft.rfft(out)) ** 2
    f = np.fft.rfftfreq(len(out), 1 / SR)
    assert p[f < 1000].sum() / p.sum() > 0.8


def test_synth_is_deterministic_per_seed():
    fit = PauseNoise(psd=np.ones(513), rms=0.1, seconds=1.0, n_intervals=1)
    a, b, c = synth_noise(fit, 1000, 3), synth_noise(fit, 1000, 3), synth_noise(fit, 1000, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_site_seed_is_stable_and_distinct():
    assert site_seed("x") == site_seed("x") != site_seed("y")
    assert 0 <= site_seed("x") < 2 ** 64


def test_compose_places_speech_at_its_own_indices():
    ex = np.arange(1, 201, dtype=np.float32)        # never zero
    noise = np.zeros(300, dtype=np.float32)
    out = compose(ex, [(100, 160)], noise)
    assert np.all(out[:100] == 0) and np.all(out[160:] == 0)
    # 60 samples is under two ramps: the product of both, attenuated but nonzero.
    assert np.all(out[100:160] != 0) and np.all(np.abs(out[100:160]) <= ex[100:160])
    long_ = compose(ex, [(20, 180)], noise)
    assert np.array_equal(long_[20 + RAMP_SAMPLES:180 - RAMP_SAMPLES],
                          ex[20 + RAMP_SAMPLES:180 - RAMP_SAMPLES])


def test_compose_ramps_only_at_noise_edges():
    ex = np.ones(200, dtype=np.float32)
    noise = np.zeros(300, dtype=np.float32)
    full = compose(ex, [(0, 200)], noise)            # the padding-only control
    assert np.all(full[:200 - RAMP_SAMPLES] == 1)     # no ramp at index 0
    assert np.allclose(full[200 - RAMP_SAMPLES:200], ramp()[::-1])
    assert np.all(full[200:] == 0)
    whole = compose(np.ones(300, dtype=np.float32), [(0, 300)], noise)
    assert np.all(whole == 1)                         # excerpt == L: no edge meets noise


def test_compose_mask_keeps_no_flap_sample():
    ex = np.ones(200, dtype=np.float32)
    noise = np.full(260, 0.5, dtype=np.float32)
    out = compose(ex, [(0, 80), (120, 200)], noise)   # flap [80, 120) masked
    assert np.all(out[80:120] == 0.5)
    assert np.allclose(out[80 - RAMP_SAMPLES:80], 1 - 0.5 * ramp())        # fade out
    assert np.allclose(out[120:120 + RAMP_SAMPLES], 0.5 + 0.5 * ramp())    # fade in
    assert np.all(out[:80 - RAMP_SAMPLES] == 1)


def test_compose_rejects_bad_intervals():
    ex, noise = np.ones(10, dtype=np.float32), np.zeros(20, dtype=np.float32)
    for keep in ([(5, 5)], [(0, 11)], [(4, 8), (6, 9)]):
        with pytest.raises(ValueError):
            compose(ex, keep, noise)
    with pytest.raises(ValueError):
        compose(np.ones(30, dtype=np.float32), [(0, 30)], noise)


def test_segment_norm_stats_matches_torch_normalize():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(2).standard_normal(500).astype(np.float32)
    padded = torch.nn.functional.pad(torch.tensor(x)[None, None], (0, 300))
    mu, sd = segment_norm_stats(x, 800)
    assert mu == pytest.approx(float(padded.mean()), abs=1e-6)
    assert sd == pytest.approx(float(padded.std()), rel=1e-5)
