"""wav2vec 2.0 checks that need the real checkpoint. Marked `model`; skipped by default.

Run explicitly, in the container, with the checkpoint cached:

    docker compose run --rm dev python -m pytest -m model tests/test_wav2vec2_model.py

The counterpart of test_frame_geometry_model.py and the checkpoint-bound half
of test_extraction.py. Random noise only: nothing here touches the corpus.
"""

import pytest

from .test_wav2vec2_geometry import W2V2

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def extractor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from capstone.encoders.extraction import Wav2Vec2Extractor

    return Wav2Vec2Extractor.load(device="cpu")


def test_geometry_read_from_checkpoint_matches_recorded_values(extractor):
    """The tripwire for the pinned revision's conv spec moving."""
    assert extractor.geometry == W2V2, (
        "wav2vec2-base's frame geometry is not the one verified on 2026-10-17. "
        "Re-run scripts/probe_w2v2_frame_positions.py before anything else."
    )


@pytest.mark.parametrize("n_samples", [400, 719, 720, 6832, 8543, 48_000])
def test_encode_returns_thirteen_states_at_the_predicted_length(extractor, n_samples):
    torch = pytest.importorskip("torch")

    torch.manual_seed(n_samples)
    out = extractor.encode(torch.randn(n_samples) * 0.1)
    assert out.shape == (13, W2V2.n_output_frames(n_samples), 768)


def test_state_twelve_is_the_model_output(extractor):
    """No final norm in the base model, so index 12 is last_hidden_state.

    WavJEPA's index 12 is pre-final-norm; here there is no norm to be before.
    If upstream adds one, this fails rather than silently changing what the
    deepest layer means.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    wav = torch.randn(8000) * 0.1
    out = extractor.encode(wav, dtype="float32")
    batch = extractor.feature_extractor(wav.numpy(), sampling_rate=16_000,
                                        return_tensors="pt")
    with torch.inference_mode():
        final = extractor.model(batch["input_values"]).last_hidden_state
    assert torch.equal(out[12], final[0].float())


def test_state_zero_is_the_transformer_input_with_position(extractor):
    """Index 0 is LayerNorm(projection + positional conv), not the CNN output."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    wav = torch.randn(8000) * 0.1
    out = extractor.encode(wav, dtype="float32")
    m = extractor.model
    batch = extractor.feature_extractor(wav.numpy(), sampling_rate=16_000,
                                        return_tensors="pt")
    with torch.inference_mode():
        conv = m.feature_extractor(batch["input_values"]).transpose(1, 2)
        proj, _ = m.feature_projection(conv)
        manual = m.encoder.layer_norm(proj + m.encoder.pos_conv_embed(proj))
    assert torch.equal(out[0], manual[0].float())
    assert not torch.allclose(out[0], proj[0].float(), atol=1e-3)


def test_encoding_is_deterministic(extractor):
    """eval() keeps SpecAugment's time masking off; two runs are identical."""
    torch = pytest.importorskip("torch")

    wav = torch.randn(6832) * 0.1
    assert torch.equal(extractor.encode(wav, dtype="float32"),
                       extractor.encode(wav, dtype="float32"))


def test_batching_is_refused(extractor):
    """do_normalize standardizes each waveform by its own statistics."""
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="one utterance"):
        extractor.encode(torch.randn(2, 16_000))
