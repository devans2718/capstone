"""Frame arithmetic tests. No torch, no GPU, no checkpoint — pure integer math.

Keeping these dependency-free means they run anywhere in milliseconds, CI
included, where downloading a 785 MB checkpoint and reserving a GPU is not on
offer. Every expected value is measured from the real model by
scripts/probe_frame_counts.py and scripts/probe_frame_positions.py.

The model-dependent counterpart is in test_frame_geometry_model.py, marked
`model` and skipped by default.
"""

import pytest

from capstone.encoders.wavjepa import FrameGeometry, reassemble

# Measured from labhamlet/wavjepa-base @ 6be4a50 on 2026-08-21.
WAVJEPA = FrameGeometry(
    sample_rate=16_000,
    target_length=32_159,
    output_steps=200,
    stride=160,
    receptive_field=240,
)

SR = WAVJEPA.sample_rate


# --------------------------------------------------------------- the staircase

def test_frames_advance_by_exactly_ten_ms_within_a_segment():
    """The conv stride is exactly 160 samples. Inside a segment there is no drift."""
    assert WAVJEPA.frame_period_ms == 10.0
    assert WAVJEPA.frame_start_sample(0) == 0
    assert WAVJEPA.frame_start_sample(1) == 160
    assert WAVJEPA.frame_start_sample(199) == 31_840


def test_segment_boundary_jumps_past_the_slack():
    """Frame 200 starts a new segment at target_length, not at 200*160.

    This is the whole reason a uniform period fails: 319 samples of gap between
    frame 199's onset and frame 200's, where a uniform map expects 160.
    """
    assert WAVJEPA.frame_start_sample(200) == 32_159
    gap = WAVJEPA.frame_start_sample(200) - WAVJEPA.frame_start_sample(199)
    # The jump spans frame 199's whole window plus the tail nothing reads,
    # not one stride: 240 + 79.
    assert gap == 319 == WAVJEPA.receptive_field + WAVJEPA.slack_samples


def test_seventy_nine_samples_per_segment_are_read_by_no_frame():
    """Frame 199 reaches sample 32,080; the segment is 32,159 long."""
    assert WAVJEPA.slack_samples == 79
    assert not WAVJEPA.is_slack_sample(32_079)
    assert WAVJEPA.is_slack_sample(32_080)
    assert WAVJEPA.is_slack_sample(32_158)
    assert not WAVJEPA.is_slack_sample(32_159)  # first sample of the next segment


# ------------------------------------------- why the obvious constants are wrong

def test_uniform_ten_ms_accumulates_error_across_segment_boundaries():
    """~9.94 ms per boundary crossed: ~40 ms by the end of a 10 s clip."""
    index = 995  # last usable frame of a 10 s clip
    uniform = index * WAVJEPA.stride
    true = WAVJEPA.frame_start_sample(index)
    drift_ms = (true - uniform) / SR * 1000
    assert true - uniform == 636
    assert drift_ms == pytest.approx(39.75, abs=0.01)


def test_uniform_mean_period_sawtooths_within_a_segment():
    """10.0497 ms is an average: exact at segment starts, ~9.9 ms off by the end."""
    at_boundary = 200 * WAVJEPA.mean_frame_period_ms / 1000 * SR
    assert at_boundary == pytest.approx(WAVJEPA.frame_start_sample(200), abs=1.0)

    mid = 199 * WAVJEPA.mean_frame_period_ms / 1000 * SR
    error_ms = (mid - WAVJEPA.frame_start_sample(199)) / SR * 1000
    assert error_ms == pytest.approx(9.9, abs=0.1)


# ------------------------------------------------------------- observed counts

@pytest.mark.parametrize(
    "seconds, segments, frames",
    [
        (2, 1, 200),   # single segment: nothing is trimmed
        (4, 2, 399),
        (6, 3, 598),
        (10, 5, 996),  # the original 996-vs-1000 observation
    ],
)
def test_output_frame_counts_match_the_model(seconds, segments, frames):
    n = SR * seconds
    assert WAVJEPA.n_segments(n) == segments
    assert WAVJEPA.n_output_frames(n) == frames


def test_hooked_frames_exceed_output_frames_by_exactly_the_padding():
    """1000 hooked vs 996 out, and the difference is exactly pad_steps."""
    n = SR * 10
    hooked = WAVJEPA.output_steps * WAVJEPA.n_segments(n)
    assert hooked == 1000
    assert hooked - WAVJEPA.n_output_frames(n) == WAVJEPA.pad_steps(n) == 4


# --------------------------------------------------------------------- padding

def test_padding_is_never_zero_even_at_an_exact_multiple():
    """Upstream pads a whole extra segment rather than none. Surprising; real."""
    assert WAVJEPA.pad_samples(WAVJEPA.target_length) == WAVJEPA.target_length


def test_a_padded_frame_survives_the_floor():
    """4.97 frames of padding exist, 4 are removed — hence a dirty last frame."""
    n = SR * 10
    exact = WAVJEPA.pad_samples(n) / SR * WAVJEPA.output_sr
    assert exact > WAVJEPA.pad_steps(n)
    assert WAVJEPA.n_clean_frames(n) < WAVJEPA.n_output_frames(n)


def test_clean_frame_count_for_a_ten_second_clip():
    assert WAVJEPA.n_clean_frames(SR * 10) == 995
    assert WAVJEPA.n_output_frames(SR * 10) == 996


# ----------------------------------------------------------------- conversions

def test_ms_and_frame_round_trip_across_a_boundary():
    for index in (0, 1, 97, 199, 200, 201, 500, 995):
        assert WAVJEPA.ms_to_frame(WAVJEPA.frame_to_ms(index)) == index


def test_sample_to_frame_never_overshoots_the_sample():
    for sample in (0, 159, 160, 16_000, 32_079, 32_159, 100_000):
        frame = WAVJEPA.sample_to_frame(sample)
        assert WAVJEPA.frame_start_sample(frame) <= sample


def test_frame_span_is_the_receptive_field_not_the_stride():
    """A per-frame label labels an overlapping 15 ms window, not a 10 ms cell."""
    start, end = WAVJEPA.frame_span_ms(0)
    assert start == 0.0
    assert end == pytest.approx(15.0, abs=1e-6)

    next_start, _ = WAVJEPA.frame_span_ms(1)
    assert end - next_start == pytest.approx(5.0, abs=1e-6)  # 5 ms overlap


def test_frames_for_interval_covers_a_label_span():
    frames = WAVJEPA.frames_for_interval(1000.0, 1100.0)
    assert len(frames) == 10  # 100 ms at 10 ms/frame, all inside one segment
    assert WAVJEPA.frame_to_ms(frames[0]) == pytest.approx(1000.0, abs=1e-6)


# ------------------------------------------------------- the upstream tripwire

def test_short_audio_does_not_trip_the_segment_count_guard():
    WAVJEPA.n_segments(SR * 200)  # 200 s, ~100 segments — fine


def test_very_long_audio_raises_instead_of_silently_mismatching():
    """Past ~201 segments the checkpoint's two segment counts disagree.

    process_seconds floors 2.0099 s to 2 s, and that 0.5% compounds until the
    padding mask believes there is one more segment than the loop produced.
    Better to refuse than to return a cut_off the captures cannot satisfy.
    """
    with pytest.raises(ValueError, match="segment counts disagree"):
        WAVJEPA.n_segments(WAVJEPA.target_length * 251)


# ------------------------------------------------------------------ reassembly

def test_reassemble_rejects_the_wrong_number_of_captures():
    """A hook list from more than one utterance is the realistic failure."""
    torch = pytest.importorskip("torch")
    captures = [torch.zeros(1, 200, 768) for _ in range(4)]  # should be 5
    with pytest.raises(ValueError, match="expected 5 segment captures"):
        reassemble(captures, WAVJEPA, SR * 10)


def test_reassemble_truncates_to_the_models_own_length():
    torch = pytest.importorskip("torch")
    captures = [torch.zeros(1, 200, 768) for _ in range(5)]
    out = reassemble(captures, WAVJEPA, SR * 10)
    assert out.shape == (1, 996, 768)
