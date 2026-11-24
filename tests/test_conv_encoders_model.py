"""HuBERT and WavLM base through Wav2Vec2Extractor. Marked `model`; skipped by default.

Run explicitly, in the container, with the checkpoints cached:

    docker compose run --rm dev python -m pytest -m model tests/test_conv_encoders_model.py

The same tripwires test_wav2vec2_model.py sets for wav2vec2-base: the frame
geometry read from the checkpoint, 13 states at the predicted length, and
state 12 being the model's output. Plus the normalize override, which the span
grid relies on (WavLM's own feature extractor does not normalize). Random noise
only: nothing here touches the corpus.
"""

import pytest

from .test_wav2vec2_geometry import W2V2

pytestmark = pytest.mark.model

NAMES = ("hubert", "wavlm")


@pytest.fixture(scope="module", params=NAMES)
def extractor(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from capstone.encoders.extraction import Wav2Vec2Extractor

    return Wav2Vec2Extractor.load_named(request.param, normalize=True, device="cpu")


def test_geometry_is_wav2vec2_bases(extractor):
    """One frame rule serves all three only if the conv stacks match."""
    assert extractor.geometry == W2V2


@pytest.mark.parametrize("n_samples", [400, 719, 720, 8543])
def test_encode_returns_thirteen_states_at_the_predicted_length(extractor, n_samples):
    torch = pytest.importorskip("torch")

    torch.manual_seed(n_samples)
    out = extractor.encode(torch.randn(n_samples) * 0.1)
    assert out.shape == (13, W2V2.n_output_frames(n_samples), 768)


def test_state_twelve_is_the_model_output(extractor):
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    wav = torch.randn(8000) * 0.1
    out = extractor.encode(wav, dtype="float32")
    batch = extractor.feature_extractor(wav.numpy(), sampling_rate=16_000,
                                        return_tensors="pt")
    with torch.inference_mode():
        final = extractor.model(batch["input_values"]).last_hidden_state
    assert torch.equal(out[12], final[0].float())


def test_normalize_override_reaches_the_model(extractor):
    """normalize=True must standardize the input whatever the checkpoint says.

    A quiet input is the case where it matters: conv layer 0's GroupNorm does
    not absorb the gain once the eps term is comparable to the variance.
    """
    torch = pytest.importorskip("torch")

    assert extractor.feature_extractor.do_normalize is True
    torch.manual_seed(1)
    quiet = torch.randn(8543) * 0.01
    standardized = (quiet - quiet.mean()) / quiet.std(unbiased=False)
    a = extractor.encode(quiet, dtype="float32")
    b = extractor.encode(standardized, dtype="float32")
    assert torch.allclose(a, b, atol=1e-3)


def test_load_named_wav2vec2_matches_the_registered_loader():
    """The generalized loader must give wav2vec2-base exactly as before."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from capstone.encoders.extraction import Wav2Vec2Extractor

    old = Wav2Vec2Extractor.load(device="cpu")
    new = Wav2Vec2Extractor.load_named("wav2vec2", normalize=True, device="cpu")
    torch.manual_seed(2)
    wav = torch.randn(8543) * 0.05
    assert torch.equal(old.encode(wav, dtype="float32"), new.encode(wav, dtype="float32"))
