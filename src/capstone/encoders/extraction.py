"""Layer-wise extraction from WavJEPA via forward hooks.

WHY HOOKS
    WavJEPAModel.forward(self, tensor) takes no kwargs, so output_hidden_states
    raises rather than degrading, and the encoder beneath is a stock
    nn.TransformerEncoder called as a black box. RQ1's layer gap and all of RQ3
    need one representation per layer, so hooks on the individual layers are the
    mechanism, not a workaround. That they are stock TransformerEncoderLayers is
    what makes the same pattern generalise to the HF baselines later.

THIRTEEN LAYERS, NOT TWELVE
    Index 0 is the INPUT to encoder.layers[0] — conv features after projection
    and positional encoding, captured with a forward PRE-hook. Indices 1..12 are
    the outputs of the 12 encoder layers.

    Layer 0 is not padding out the count. A single frame reads 15 ms, and stop
    closure voicing spans 50-100 ms, so RQ1's cue cannot be present in the conv
    features and must be assembled by the transformer. Layer 0 is therefore a
    real control: signal there means the probe is reading something acoustic and
    local, not something the encoder built.

    With one caveat that has to be honoured by the caller. Layer 0 also carries
    the positional encoding, and it carries it strongly — see plan_excerpt. The
    early layers are only a clean control if every site is centred with the same
    context; otherwise position leaks into exactly the layers the control uses.

    Note index 12 is NOT the model's own output. A final LayerNorm sits after the
    stack, so model(x) == LayerNorm(layer 12). Kept unnormalised deliberately —
    every encoder gets treated the same way, and the baselines' hidden states are
    likewise pre-final-norm.

WHY EXCERPTS, NOT WHOLE RECORDINGS
    Two independent ceilings meet here:

      - FrameGeometry.n_segments refuses past ~201 segments (~6.7 min), where the
        checkpoint's own two segment counts diverge. Buckeye tracks are longer.
      - The positional embedding table is exactly 200 entries and no attention
        crosses a 2.01 s segment, so context beyond 2.01 s is not merely
        expensive, it is unreachable by the model.

    The second makes the first harmless. Encoding a short excerpt around a site
    discards nothing the model could have used. It also does something better
    than whole-track encoding: segment boundaries are laid on a grid from the
    excerpt's first sample, so centring the excerpt on the target puts the target
    mid-segment, where it has the most within-segment context available. Encoding
    a whole track leaves boundary placement to chance, and a site landing next to
    a seam silently loses context on one side.

    The cost is that the representation depends on the excerpt window. That is a
    real experimental parameter, which is why it is named and recorded rather
    than hidden.

WHAT THIS MODULE DOES NOT DO
    It does not pool. Sites are 2-3 frames wide at this geometry (median 27.4 ms
    against a 10 ms hop), so mean vs. max vs. midpoint are materially different
    vectors, not variants of one summary — and flapping is conditioned by the
    surrounding vowels, so the cue may live in the transitions rather than in the
    flap interval. Pooling is a decision the pilot should inform, so extraction
    keeps frames and `window_frames` records where the target sits inside them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

from .wavjepa import FrameGeometry, reassemble

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

MODEL_ID = "labhamlet/wavjepa-base"
# Pinned: WavJEPA loads with trust_remote_code=True, so its modelling code is
# fetched from the Hub and is outside transformers' deprecation policy. An
# unpinned revision can move the frame arithmetic under already-cached
# embeddings. tests/test_frame_geometry_model.py is the tripwire.
REVISION = "6be4a50"

N_ENCODER_LAYERS = 12
N_LAYERS = N_ENCODER_LAYERS + 1  # + the pre-transformer input at index 0
WIDTH = 768


@dataclass(frozen=True)
class Excerpt:
    """An audio window cut around a target, and where the target sits in it.

    Times are relative to the source recording; `offset_ms` is what converts
    them into excerpt-relative coordinates for FrameGeometry.
    """

    offset_ms: float        # excerpt start, in the source recording
    duration_ms: float
    target_start_ms: float  # target onset, in the source recording
    target_end_ms: float
    clamped: bool = False   # hit a recording edge, so the target is off-centre

    @property
    def target_start_local_ms(self) -> float:
        return self.target_start_ms - self.offset_ms

    @property
    def target_end_local_ms(self) -> float:
        return self.target_end_ms - self.offset_ms


def plan_excerpt(
    target_start_ms: float,
    target_end_ms: float,
    recording_duration_ms: float,
    context_ms: float = 200.0,
) -> Excerpt:
    """Centre an excerpt on a target, with `context_ms` on each side.

    KEEP context_ms CONSTANT ACROSS SITES. It is not merely a storage knob.

    The positional encoding is fixed, non-learned, and re-added from position 0
    every segment, so a frame's representation depends on where it sits inside
    its segment. Measured on identical audio shifted 10 frames: cosine distance
    0.394 at layer 0 and 0.459 at layer 1, decaying to 0.021 by layer 12,
    against a run-to-run noise floor of exactly 0.

    Centring every target with the same context makes that contribution a
    constant across sites, so it cannot correlate with the probe's label. Vary
    the width between sites and it becomes a confound concentrated in precisely
    the early layers the design uses as its control.

    Clamped at the recording edges, because padding with silence would be signal
    the encoder normalises against. A clamped excerpt is off-centre and thus
    carries a different positional contribution — `clamped` is set so callers can
    drop those sites rather than discover the imbalance in the results.

    context_ms defaults to 200 because the model cannot attend beyond 2.01 s, so
    wider windows buy context the encoder provably cannot use while costing
    storage linearly. It stays a parameter so the pilot can test that.
    """
    start = max(0.0, target_start_ms - context_ms)
    end = min(recording_duration_ms, target_end_ms + context_ms)
    was_clamped = (
        start > target_start_ms - context_ms or end < target_end_ms + context_ms
    )
    return Excerpt(
        offset_ms=start,
        duration_ms=end - start,
        target_start_ms=target_start_ms,
        target_end_ms=target_end_ms,
        clamped=was_clamped,
    )


def excerpt_samples(excerpt: Excerpt, sample_rate: int = 16_000) -> tuple[int, int]:
    """The sample window an Excerpt cuts out of its recording: (start, end).

    ONE COPY, DELIBERATELY
        encode_excerpt cuts the audio the encoder consumes. Anything else that
        cuts the same window — a clip exporter, a listening tool — has to land on
        the same two integers or it is showing a human something the model never
        saw. Measured positional sensitivity is cosine distance 0.394 at layer 0
        for a 10-frame shift, so "close enough" is not a category that exists
        here: a clip off by one frame makes a conclusion drawn by ear unfounded.

        The rounding is the part that does not survive restatement. `end` is
        start PLUS a separately rounded duration, not a rounding of
        target_end_ms + context_ms. Those two differ by one sample whenever the
        fractional parts fall either side of a half, which is often.

        So this is a function rather than a comment telling the next caller to
        be careful. It lives at module scope, and it touches no torch, so the
        application can import it without the CUDA environment.
    """
    start = int(round(excerpt.offset_ms / 1000 * sample_rate))
    end = start + int(round(excerpt.duration_ms / 1000 * sample_rate))
    return start, end


class WavJEPAExtractor:
    """Frozen WavJEPA with per-layer hooks. One utterance per call.

    Never exposes a training path: the model is eval()'d with requires_grad
    cleared, and every forward runs under inference_mode. This is a probing
    study — the encoder is frozen by construction, not by convention.
    """

    def __init__(self, model, feature_extractor, geometry: FrameGeometry, device: str):
        self.model = model
        self.feature_extractor = feature_extractor
        self.geometry = geometry
        self.device = device
        self._layers = _find_encoder_layers(model)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        model_id: str = MODEL_ID,
        revision: str | None = REVISION,
        device: str | None = None,
    ) -> "WavJEPAExtractor":
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device)

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        geometry = FrameGeometry.from_model(model)
        return cls(model, feature_extractor, geometry, device)

    # ----------------------------------------------------------------- encoding

    def encode(self, waveform, dtype: str = "float16"):
        """Encode ONE utterance and return (13, n_frames, 768).

        Args:
            waveform: 1-D float samples at 16 kHz, or a (1, n) tensor/array.
                Must be one utterance — see the module docstring on why this is
                not batched and must not be re-chunked by the caller.
            dtype: storage dtype for the returned tensor. float16 halves the
                cache for a precision loss far below the between-layer
                differences a probe reads.

        Returns:
            (13, n_frames, 768), where n_frames == geometry.n_output_frames.
            Index 0 is the pre-transformer input; 1..12 are encoder outputs.
        """
        import torch

        wav = _as_1d(waveform)
        n_samples = int(wav.shape[-1])
        # Raises before any GPU work if the excerpt is past the point where the
        # checkpoint's own segment counts disagree.
        self.geometry.n_segments(n_samples)

        batch = self.feature_extractor(
            wav.unsqueeze(0) if wav.dim() == 1 else wav, return_tensors="pt"
        )
        inputs = batch["input_values"].to(self.device)

        with _hooks(self._layers) as captured:
            with torch.inference_mode():
                self.model(inputs)

            layers = []
            for idx in range(N_LAYERS):
                segments = captured[idx]
                if not segments:
                    raise RuntimeError(
                        f"layer {idx} captured nothing — the hook did not fire. "
                        "Upstream may have changed the encoder's call path."
                    )
                # reassemble validates the segment count against n_samples, which
                # catches captures accidentally spanning two utterances.
                layers.append(reassemble(segments, self.geometry, n_samples))

        stacked = torch.cat(layers, dim=0)  # (13, n_frames, 768)
        return stacked.to(getattr(torch, dtype))

    def encode_excerpt(self, recording, excerpt: Excerpt, sample_rate: int = 16_000):
        """Cut `excerpt` out of `recording`, encode it, and locate the target.

        Returns (frames, target_range) where frames is (13, n, 768) and
        target_range indexes the frames whose onsets fall inside the target.
        The range can be empty for a very short target — 4 of Buckeye's 10,392
        flap sites are under 13.4 ms and contain no frame onset at all — so
        callers must handle that rather than assume at least one frame.
        """
        wav = _as_1d(recording)
        start, end = excerpt_samples(excerpt, sample_rate)
        frames = self.encode(wav[start:end])

        target = self.geometry.frames_for_interval(
            excerpt.target_start_local_ms, excerpt.target_end_local_ms
        )
        return frames, target

    def window_frames(self, excerpt: Excerpt, n_frames: int) -> range:
        """Frames of an encoded excerpt whose onsets fall inside the target.

        Separated from encode_excerpt so labels can be computed without a GPU.
        Clamped to n_frames: the model's final retained frame is ~95% padding
        (pad_steps floors), and a target abutting the excerpt end could otherwise
        index into it.
        """
        rng = self.geometry.frames_for_interval(
            excerpt.target_start_local_ms, excerpt.target_end_local_ms
        )
        return range(min(rng.start, n_frames), min(rng.stop, n_frames))


# --------------------------------------------------------------------- internals


def _as_1d(waveform):
    """Accept a tensor, an array, or a (1, n) batch; return a 1-D float tensor."""
    import torch

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.float().squeeze()
    if waveform.dim() != 1:
        raise ValueError(
            f"expected one utterance as 1-D samples, got shape {tuple(waveform.shape)}. "
            "Batching is deliberately unsupported: the feature extractor derives one "
            "RMS gain per waveform, so batching changes the numbers."
        )
    return waveform


def _find_encoder_layers(model):
    """Locate the ModuleList of encoder layers to hook.

    model.model.encoder.layers is undocumented remote code that upstream can
    rename in any push, so a miss searches the module tree and reports where the
    layers went, rather than failing with a bare AttributeError.
    """
    import torch

    try:
        return model.model.encoder.layers
    except AttributeError:
        pass

    for name, mod in model.named_modules():
        if (
            isinstance(mod, torch.nn.ModuleList)
            and len(mod) == N_ENCODER_LAYERS
            and any(isinstance(m, torch.nn.TransformerEncoderLayer) for m in mod)
        ):
            return mod

    raise AttributeError(
        "cannot locate WavJEPA's encoder layers to hook — upstream moved them. "
        "scripts/smoke_wavjepa.py prints the module tree."
    )


@contextmanager
def _hooks(layers) -> Iterator[dict[int, list]]:
    """Register the 13 capture hooks; always remove them.

    A context manager because a leaked hook is silent and cumulative: the next
    forward appends to the same lists, and reassemble then sees too many
    segments. That surfaces as a confusing count error far from its cause.

    get_audio_representation loops over 2.01 s segments in Python, so every hook
    fires once PER SEGMENT. The lists are per-layer segment sequences, which is
    exactly what reassemble consumes.
    """
    captured: dict[int, list] = {i: [] for i in range(N_LAYERS)}
    handles = []

    def keep(idx):
        def hook(_module, _inputs, output):
            # .detach().float().cpu() immediately: keeping segment captures on
            # the GPU across a long recording is how this OOMs.
            captured[idx].append(output.detach().float().cpu())

        return hook

    def keep_input(_module, args):
        captured[0].append(args[0].detach().float().cpu())

    # Index 0: the input to the first encoder layer, i.e. conv features after
    # projection and positional encoding.
    handles.append(layers[0].register_forward_pre_hook(keep_input))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(keep(i + 1)))

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
