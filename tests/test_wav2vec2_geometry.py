"""wav2vec 2.0 frame arithmetic. No torch, no GPU, no checkpoint.

Every expected value is measured from facebook/wav2vec2-base @ 0b5b8e8 by
scripts/probe_w2v2_frame_positions.py (positions) and test_wav2vec2_model.py
(counts). The model-dependent counterpart is test_wav2vec2_model.py, marked
`model` and skipped by default.
"""

from types import SimpleNamespace

import pytest

from capstone.encoders.extraction import Wav2Vec2Extractor, plan_excerpt
from capstone.encoders.wav2vec2 import ConvGeometry

# Measured from facebook/wav2vec2-base @ 0b5b8e8 on 2026-10-17.
W2V2 = ConvGeometry(sample_rate=16_000, stride=320, receptive_field=400)

CONFIG = SimpleNamespace(conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                         conv_stride=[5, 2, 2, 2, 2, 2, 2])


# ---------------------------------------------------------------- construction

def test_geometry_derives_from_the_conv_spec():
    """320 and 400 come out of the recursion, not out of a constant."""
    assert ConvGeometry.from_config(CONFIG, 16_000) == W2V2


def test_mismatched_conv_spec_is_refused():
    bad = SimpleNamespace(conv_kernel=[10, 3], conv_stride=[5])
    with pytest.raises(ValueError):
        ConvGeometry.from_config(bad, 16_000)


def test_frames_advance_by_exactly_twenty_ms():
    assert W2V2.frame_period_ms == 20.0
    assert W2V2.frame_span_ms(1) == pytest.approx((20.0, 45.0))


# ------------------------------------------------------------- observed counts

@pytest.mark.parametrize(
    "n_samples, frames",
    [(400, 1), (719, 1), (720, 2), (1039, 2), (6400, 19), (6832, 21),
     (8543, 26), (16_000, 49), (48_000, 149)],
)
def test_output_frame_counts_match_the_model(n_samples, frames):
    assert W2V2.n_output_frames(n_samples) == frames


def test_input_shorter_than_one_window_is_refused():
    with pytest.raises(ValueError, match="receptive field"):
        W2V2.n_output_frames(399)


# ------------------------------------------------------- selection by centre

def test_centres_sit_half_a_window_after_onsets():
    assert W2V2.frame_centre_sample(0) == 200
    assert W2V2.frame_centre_sample(20) == 6600


def test_interval_takes_the_frames_whose_centres_it_contains():
    # [10 ms, 40 ms) = samples [160, 640): centres 200 and 520.
    assert W2V2.frames_for_interval(10.0, 40.0) == range(0, 2)


def test_a_centre_on_the_start_is_in_and_on_the_end_is_out():
    # Frame 1's centre is sample 520 = 32.5 ms.
    assert 1 in W2V2.frames_for_interval(32.5, 50.0)
    assert 1 not in W2V2.frames_for_interval(20.0, 32.5)


def test_interval_between_two_centres_is_empty():
    assert len(W2V2.frames_for_interval(13.0, 32.0)) == 0


def test_nearest_frame_breaks_a_tie_towards_the_earlier():
    """Sample 360 is equidistant from centres 200 and 520, as MFCC's rule has it."""
    assert W2V2.nearest_frame(360) == 0
    assert W2V2.nearest_frame(360.5) == 1
    assert W2V2.nearest_frame(0) == 0


# ------------------------------------------------- the extractor's frame rule
# window_frames is pure arithmetic, so the extractor is built without a model.

EXTRACTOR = Wav2Vec2Extractor(None, None, W2V2, "cpu")


def test_centred_flap_of_one_hop_gets_frame_ten():
    """plan_excerpt puts the flap onset at 200 ms, frame 10's onset; frame 10's
    centre is 12.5 ms later, so a 27.4 ms flap (the median) contains it alone."""
    ex = plan_excerpt(5_000.0, 5_027.4, recording_duration_ms=60_000, context_ms=200)
    assert EXTRACTOR.window_frames(ex, 21) == range(10, 11)


def test_flap_containing_no_centre_takes_the_nearest_to_its_midpoint():
    """A 10 ms flap ends before frame 10's centre: the §4 fallback applies."""
    ex = plan_excerpt(5_000.0, 5_010.0, recording_duration_ms=60_000, context_ms=200)
    assert len(W2V2.frames_for_interval(ex.target_start_local_ms,
                                        ex.target_end_local_ms)) == 0
    # Midpoint 205 ms = sample 3280: centres 3080 (frame 9) and 3400 (frame 10).
    assert EXTRACTOR.window_frames(ex, 21) == range(10, 11)


def test_window_frames_is_clamped_to_the_frames_encoded():
    ex = plan_excerpt(5_000.0, 5_027.4, recording_duration_ms=60_000, context_ms=200)
    assert EXTRACTOR.window_frames(ex, 10) == range(10, 10)
