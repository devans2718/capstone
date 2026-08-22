"""Checks that need the real checkpoint. Marked `model`; skipped by default.

Run explicitly, in the container, with a GPU and the checkpoint cached:

    docker compose run --rm dev python -m pytest -m model

These are the tests that cannot run in CI — the checkpoint is 785 MB and lives
behind a network fetch. They earn their keep by catching the one failure the
pure-arithmetic tests structurally cannot: upstream changing the numbers.

WavJEPA is loaded with trust_remote_code=True, so its modeling code is fetched
from the Hub and is not covered by transformers' deprecation policy. A silent
revision that changes process_audio_seconds would move the frame period under
every cached embedding and every probe label already computed. That is worth a
test even though it will pass every time for months.
"""

import pytest

from capstone.encoders.wavjepa import FrameGeometry

from .test_frame_geometry import WAVJEPA

pytestmark = pytest.mark.model

MODEL_ID = "labhamlet/wavjepa-base"


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    del torch
    m = transformers.AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    m.eval()
    return m


def test_geometry_read_from_checkpoint_matches_recorded_values(model):
    """The tripwire for an upstream revision moving the frame arithmetic."""
    live = FrameGeometry.from_model(model)
    assert live == WAVJEPA, (
        "WavJEPA's frame geometry changed upstream. Every cached embedding and "
        "every probe label derived from the old geometry is now suspect. Re-run "
        "scripts/probe_frame_counts.py before doing anything else."
    )


def test_model_output_length_matches_predicted_cut_off(model):
    """End to end: the formula predicts what the model actually returns."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    extractor = transformers.AutoFeatureExtractor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    geometry = FrameGeometry.from_model(model)

    for seconds in (2, 6, 10):
        n = geometry.sample_rate * seconds
        torch.manual_seed(seconds)
        batch = extractor(torch.randn(1, n), return_tensors="pt")
        with torch.inference_mode():
            result = model(batch["input_values"])
        final = result[0] if isinstance(result, (tuple, list)) else result
        assert final.shape[-2] == geometry.n_output_frames(n), (
            f"{seconds}s: model returned {final.shape[-2]} frames, "
            f"formula predicted {geometry.n_output_frames(n)}"
        )
