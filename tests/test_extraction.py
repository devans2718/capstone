"""Extraction tests.

Split deliberately: the excerpt planning is pure arithmetic and runs in
milliseconds anywhere, while anything touching hooks needs the 785 MB
checkpoint and is marked `model`. Keeping them in one file makes the boundary
visible — if a test needs the checkpoint, it should be obvious why.
"""

import pytest

from capstone.encoders.extraction import Excerpt, excerpt_samples, plan_excerpt
from capstone.encoders.wavjepa import FrameGeometry

from .test_frame_geometry import WAVJEPA


# ------------------------------------------------------------- excerpt planning


def test_excerpt_centres_on_target():
    ex = plan_excerpt(1000.0, 1030.0, recording_duration_ms=60_000, context_ms=200)
    assert ex.offset_ms == 800.0
    assert ex.duration_ms == 430.0
    assert ex.target_start_local_ms == 200.0
    assert ex.target_end_local_ms == 230.0


def test_excerpt_clamps_at_recording_start():
    """A target near the start gets a short excerpt, not silence padding.

    Padding would be signal to the encoder — it normalises per segment and has
    no way to know the leading zeros are ours rather than the recording's.
    """
    ex = plan_excerpt(50.0, 80.0, recording_duration_ms=60_000, context_ms=200)
    assert ex.offset_ms == 0.0
    assert ex.target_start_local_ms == 50.0
    assert ex.clamped


def test_excerpt_clamps_at_recording_end():
    ex = plan_excerpt(59_900.0, 59_950.0, recording_duration_ms=60_000, context_ms=200)
    assert ex.offset_ms == 59_700.0
    assert ex.duration_ms == 300.0
    assert ex.clamped


def test_unclamped_excerpt_is_not_flagged():
    ex = plan_excerpt(1000.0, 1030.0, recording_duration_ms=60_000, context_ms=200)
    assert not ex.clamped


def test_centred_excerpts_put_every_target_at_the_same_frame():
    """The property that makes the positional encoding cancel across sites.

    Identical audio shifted 10 frames within its segment differs by cosine 0.394
    at layer 0. Constant context is what stops that varying with the label, so
    it is asserted rather than left as a comment.
    """
    g = WAVJEPA
    onsets = set()
    for start in (1000.0, 5000.0, 12_345.0, 40_000.0):
        ex = plan_excerpt(start, start + 30.0, recording_duration_ms=60_000,
                          context_ms=200)
        assert not ex.clamped
        onsets.add(g.ms_to_frame(ex.target_start_local_ms))
    assert len(onsets) == 1, f"targets landed on different frames: {onsets}"


def test_target_stays_findable_through_the_staircase():
    """Excerpt-local times must survive conversion by the real geometry.

    The frame map is piecewise, so a target 200 ms into an excerpt is not at
    frame 20 in general. This asserts the round trip lands inside the target,
    not that it lands on a particular index.
    """
    g = WAVJEPA
    ex = plan_excerpt(5_000.0, 5_030.0, recording_duration_ms=60_000, context_ms=200)
    frames = g.frames_for_interval(ex.target_start_local_ms, ex.target_end_local_ms)
    assert len(frames) >= 1
    for i in frames:
        onset = g.frame_to_ms(i)
        assert ex.target_start_local_ms <= onset < ex.target_end_local_ms


def test_very_short_centred_target_still_gets_its_onset_frame():
    """A target shorter than the frame hop, centred by plan_excerpt, keeps a frame.

    Centring puts the target's start exactly on a frame onset, and that onset
    is inside [start, end). This test used to assert the opposite (zero frames),
    which pinned the off-by-one resolved as D1 rather than the docstring's
    contract.
    """
    g = WAVJEPA
    ex = plan_excerpt(1_000.0, 1_006.0, recording_duration_ms=60_000, context_ms=200)
    frames = g.frames_for_interval(ex.target_start_local_ms, ex.target_end_local_ms)
    assert len(frames) == 1
    assert g.frame_to_ms(frames[0]) == pytest.approx(ex.target_start_local_ms, abs=1e-6)


def test_short_target_off_an_onset_can_yield_no_frames():
    """Empty ranges are still possible, so callers must still handle them.

    A 6 ms target starting 2 ms after an onset contains no onset at all.
    Nothing plan_excerpt produces lands here, but a raw timestamp can.
    """
    frames = WAVJEPA.frames_for_interval(1_002.0, 1_008.0)
    assert len(frames) == 0


def test_generic_phase_excludes_the_frame_that_started_before_the_target():
    """The other half of D1: a frame whose onset precedes start is not inside."""
    frames = WAVJEPA.frames_for_interval(1_003.0, 1_033.0)
    assert [WAVJEPA.frame_to_ms(i) for i in frames] == pytest.approx([1010.0, 1020.0, 1030.0])


def test_excerpt_samples_matches_what_the_encoder_consumes():
    """One window, one arithmetic. Anything else cutting audio calls this.

    encode_excerpt used to carry these two lines inline, which made them
    impossible to reuse from a torch-free caller — so a clip exporter would have
    had to restate them, and a restatement that drifts shows a human audio the
    model never saw.
    """
    ex = plan_excerpt(5_000.0, 5_030.0, recording_duration_ms=60_000, context_ms=200)
    start, end = excerpt_samples(ex)
    assert start == int(round(ex.offset_ms / 1000 * 16_000))
    assert end - start == int(round(ex.duration_ms / 1000 * 16_000))


def test_end_is_not_derived_from_target_end_plus_context():
    """The restatement that looks right and lands one sample away.

    `end` is start PLUS a separately rounded duration. Deriving it from
    target_end_ms + context_ms rounds differently, and the two disagree on 40%
    of windows — measured over 200,000 offsets, not estimated. At 16 kHz one
    sample is 62.5 us, which sounds like nothing and is not: identical audio
    shifted within its segment differs by cosine 0.394 at layer 0.
    """
    disagreements = 0
    for i in range(2_000):
        t0 = 1_000.0 + i * 0.0137
        ex = plan_excerpt(t0, t0 + 27.4, recording_duration_ms=600_000, context_ms=200)
        naive_end = int(round((ex.target_end_ms + 200.0) / 1000 * 16_000))
        if excerpt_samples(ex)[1] != naive_end:
            disagreements += 1
    assert disagreements > 0, (
        "the naive derivation agreed everywhere, which means this test no longer "
        "guards anything — check whether the rounding changed"
    )


# ------------------------------------------------------------- checkpoint-bound

model_test = pytest.mark.model


@pytest.fixture(scope="module")
def extractor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from capstone.encoders.extraction import WavJEPAExtractor

    return WavJEPAExtractor.load(device="cpu")


@model_test
def test_encode_returns_thirteen_layers(extractor):
    torch = pytest.importorskip("torch")

    wav = torch.randn(16_000 * 3)
    out = extractor.encode(wav)

    assert out.shape[0] == 13, "expected 12 encoder layers plus the input"
    assert out.shape[2] == 768
    assert out.shape[1] == extractor.geometry.n_output_frames(16_000 * 3)


@model_test
def test_layer_twelve_matches_model_output_up_to_final_norm(extractor):
    """Index 12 is pre-final-LayerNorm, and that is deliberate.

    Asserting the relationship rather than equality documents the convention: if
    upstream ever moves the norm inside the stack, this fails loudly instead of
    silently changing what every probe reads.
    """
    torch = pytest.importorskip("torch")

    wav = torch.randn(16_000 * 3)
    out = extractor.encode(wav, dtype="float32")

    batch = extractor.feature_extractor(wav.unsqueeze(0), return_tensors="pt")
    with torch.inference_mode():
        final = extractor.model(batch["input_values"].to(extractor.device))
    final = final[0] if isinstance(final, (tuple, list)) else final

    assert out[12].shape == final.squeeze(0).shape
    assert not torch.allclose(out[12], final.squeeze(0).float(), atol=1e-3), (
        "index 12 equals the model output — the final LayerNorm may have moved"
    )


@model_test
def test_hooks_do_not_leak_between_calls(extractor):
    """Two encodes in a row must not accumulate captures.

    A leaked hook is silent: the second call's reassemble sees twice the
    segments and raises a count error that points nowhere near the cause.
    """
    torch = pytest.importorskip("torch")

    wav = torch.randn(16_000 * 3)
    first = extractor.encode(wav)
    second = extractor.encode(wav)
    assert first.shape == second.shape


@model_test
def test_batching_is_refused(extractor):
    """The feature extractor derives one RMS gain per waveform."""
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="one utterance"):
        extractor.encode(torch.randn(2, 16_000))
