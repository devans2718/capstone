"""Frame geometry for WavJEPA: the conversions between time, samples and frames.

WHY THIS MODULE EXISTS
    Probe labels come from forced-alignment timestamps in milliseconds and have
    to become frame indices. Every obvious way to do that conversion is wrong,
    and all of them fail silently:

        10.00 ms/frame     wrong: ignores the 79-sample gap at each segment
                           boundary. Accumulates ~9.94 ms per boundary crossed,
                           ~40 ms by the end of a 10 s clip.
        10.0497 ms/frame   wrong: that is the AVERAGE over a whole segment.
                           Sawtooths from 0 to ~9.9 ms within each segment and
                           resets. Bounded, but not small.
        get_timestamps()   wrong: divides clip duration by surviving frame
                           count, mixing two spans that differ by the retained
                           padding fraction.

    The true map is piecewise, and this module is the only place it lives.

THE ACTUAL GEOMETRY  (measured, see scripts/probe_frame_positions.py)
    The conv stack's stride product is exactly 160 samples with a 240-sample
    receptive field, so within one segment frame j reads input samples
    [j*160, j*160 + 240) — a clean 10.000 ms hop.

    But a segment is 32,159 samples and 200 frames only reach sample 32,080.
    The last 79 samples of every segment are read by NO frame. Verified by
    spiking single samples and finding where the conv response peaks: spikes at
    32,100 and 32,158 produce no local peak at all.

    So globally:

        frame_start_sample(i) = (i // 200) * 32159  +  (i % 200) * 160
                                 ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
                                 segment origin        exact 10 ms hop

    A staircase: flat 10 ms steps, with a 4.94 ms jump every 2.01 s.

DELIBERATELY IMPORT-LIGHT
    No torch at module scope. FrameGeometry is plain integer arithmetic, so all
    of it is testable without a GPU, a checkpoint download, or a 2 GB torch
    install. Only `from_model` and `reassemble` touch torch, on call.

THE PADDING ARITHMETIC IT MIRRORS
    From the checkpoint's remote code (labhamlet/wavjepa-base @ 6be4a50):

        model.py:    pad_frames = target_length - (cur_frames % target_length)
                     ... loop over segments, hstack, then x = x[:, :cut_off, :]
        utils.py:    total_frames = int((total / sr) / process_seconds)
                     output_sr    = int(output_steps / process_seconds)
                     pad_steps    = int((pad_frames / sr) * output_sr)
                     cut_off      = output_steps * total_frames - pad_steps

    The int() floors are reproduced exactly rather than cleaned up. They are
    what the model does, and two of this project's gotchas live inside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch


@dataclass(frozen=True)
class FrameGeometry:
    """Every time/sample/frame conversion WavJEPA implies, derived not assumed.

    Construct with `from_model` in real code. The explicit constructor exists so
    the arithmetic can be tested against known values without loading anything.
    """

    sample_rate: int
    target_length: int      # samples the encoder consumes in one forward pass
    output_steps: int       # frames it emits for those samples
    stride: int             # conv stride product: samples between frame onsets
    receptive_field: int    # samples one frame actually reads

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_model(cls, model) -> "FrameGeometry":
        """Read the geometry off a loaded checkpoint.

        Accepts either the HF wrapper (WavJEPAModel) or the inner WavJEPA. The
        attribute path is undocumented remote code and upstream can rename it,
        so a miss raises here with a clear message rather than surfacing later
        as a wrong number.
        """
        inner = getattr(model, "model", model)
        try:
            extractor = inner.extract_audio
            return cls(
                sample_rate=inner.sample_rate,
                target_length=inner.target_length,
                output_steps=inner.output_steps,
                stride=prod(s for _, _, s in extractor.conv_layers_spec),
                # receptive_fields[0] is the input-side field of one output
                # frame, computed by the extractor from its own conv spec.
                receptive_field=extractor.receptive_fields[0],
            )
        except AttributeError as exc:
            raise AttributeError(
                "could not read frame geometry from this model — upstream moved "
                f"an attribute. Missing: {exc}"
            ) from exc

    # ------------------------------------------------------------------ basic units

    @property
    def process_seconds(self) -> int:
        """Segment length in whole seconds, as the model computes it.

        Integer division, mirroring utils.calculate_padding_mask. For a 32,159
        sample window this is 2, not 2.0099 — the model rounds its own window
        down here, which is where output_sr comes from.
        """
        return self.target_length // self.sample_rate

    @property
    def output_sr(self) -> int:
        """Nominal frame rate the model pads against: 100 Hz."""
        return int(self.output_steps / self.process_seconds)

    @property
    def frame_period_ms(self) -> float:
        """Time between consecutive frames WITHIN a segment: exactly 10.000 ms.

        Not valid across a segment boundary — use frame_start_sample for that.
        """
        return self.stride / self.sample_rate * 1000

    @property
    def mean_frame_period_ms(self) -> float:
        """Average period across a whole segment: 10.0497 ms.

        Correct only as an average. Reporting it as "the frame rate" is what
        produces up to ~10 ms of sawtooth error inside each segment. Kept here
        because it is the number the model's own get_timestamps approximates,
        so comparisons against upstream behaviour need it.
        """
        return self.target_length / self.output_steps / self.sample_rate * 1000

    @property
    def slack_samples(self) -> int:
        """Samples at the end of each segment that no frame reads: 79.

        200 frames reach sample 32,080; the segment is 32,159. Audio landing
        here is invisible to the encoder — a 4.94 ms hole every 2.01 s.
        """
        return self.target_length - ((self.output_steps - 1) * self.stride + self.receptive_field)

    # ------------------------------------------------------------------- padding

    def pad_samples(self, n_samples: int) -> int:
        """Zeros the model appends to reach a whole number of segments.

        Never zero: when n_samples is already an exact multiple, the model pads
        a full extra segment rather than none. Upstream's behaviour, not a bug
        here.
        """
        return self.target_length - (n_samples % self.target_length)

    def n_segments(self, n_samples: int) -> int:
        """Segments the encoder loop will run.

        The checkpoint counts segments two different ways — `padded //
        target_length` for the loop, `int((padded / sr) / process_seconds)` for
        the padding mask — and they agree only while the audio is short enough.
        They diverge past ~201 segments because process_seconds rounds the
        window down from 2.0099 s to 2 s, and that 0.5% compounds.

        Raise rather than return a number that would silently mismatch the
        model's own truncation. Nothing here encodes 6.7-minute audio in one
        call, so this is a tripwire, not a limitation.
        """
        padded = n_samples + self.pad_samples(n_samples)
        loop_count = padded // self.target_length
        mask_count = int((padded / self.sample_rate) / self.process_seconds)
        if loop_count != mask_count:
            raise ValueError(
                f"{n_samples} samples ({n_samples / self.sample_rate:.1f} s) is past "
                f"the point where the checkpoint's own segment counts disagree "
                f"(loop={loop_count}, mask={mask_count}). Encode in shorter pieces."
            )
        return loop_count

    def pad_steps(self, n_samples: int) -> int:
        """Frames the model discards, floored — which is why one survives dirty."""
        return int((self.pad_samples(n_samples) / self.sample_rate) * self.output_sr)

    # -------------------------------------------------------------- frame counts

    def n_output_frames(self, n_samples: int) -> int:
        """`cut_off`: frames the model returns for this many input samples."""
        return self.output_steps * self.n_segments(n_samples) - self.pad_steps(n_samples)

    def n_clean_frames(self, n_samples: int) -> int:
        """Frames whose receptive field lies wholly inside real audio.

        Always <= n_output_frames. pad_steps is floored, so a fraction of a
        padding frame survives the truncation: the last frame the model returns
        is typically ~95% silence, and it is not masked from attention either
        since the mask uses the same floored count.

        Use this, not n_output_frames, to bound where probe targets may sit.
        """
        total = self.n_output_frames(n_samples)
        clean = 0
        for i in range(total):
            if self.frame_start_sample(i) + self.receptive_field <= n_samples:
                clean = i + 1
            else:
                break
        return clean

    # -------------------------------------------------------------- conversions

    def frame_start_sample(self, index: int) -> int:
        """First input sample that frame `index` reads. The staircase.

        Segment origins advance by target_length (32,159) while frames inside a
        segment advance by stride (160). The two differ by the slack, so a
        uniform period is wrong in both directions depending which you pick.
        """
        segment, within = divmod(index, self.output_steps)
        return segment * self.target_length + within * self.stride

    def frame_to_ms(self, index: int) -> float:
        """Onset time of frame `index`, in ms. Exact, not averaged."""
        return self.frame_start_sample(index) / self.sample_rate * 1000

    def frame_span_ms(self, index: int) -> tuple[float, float]:
        """The (start, end) of what frame `index` reads, in ms.

        The span is receptive_field wide (15 ms), not stride wide (10 ms), so
        consecutive frames within a segment overlap by 5 ms. A per-frame label
        is a label on an overlapping window, not on a partition.
        """
        start = self.frame_to_ms(index)
        return start, start + self.receptive_field / self.sample_rate * 1000

    def sample_to_frame(self, sample: int) -> int:
        """Input sample -> the frame whose onset most recently preceded it.

        A sample is read by several overlapping frames; this returns the last
        one to start at or before it, which is the convention that makes
        frame_start_sample(sample_to_frame(s)) <= s hold.

        Samples in the per-segment slack are read by no frame at all. Rather
        than invent an index, this clamps to the segment's final frame — check
        `is_slack_sample` first when it matters.
        """
        segment, within = divmod(sample, self.target_length)
        frame_within = min(within // self.stride, self.output_steps - 1)
        return segment * self.output_steps + frame_within

    def ms_to_frame(self, t_ms: float) -> int:
        """Alignment timestamp -> frame index, through the true staircase."""
        return self.sample_to_frame(int(round(t_ms / 1000 * self.sample_rate)))

    def is_slack_sample(self, sample: int) -> bool:
        """True if no frame reads this sample — it lands in a segment's tail gap.

        Worth checking for any alignment boundary that matters: a phone whose
        acoustic evidence falls entirely in the slack is not represented at all.
        """
        within = sample % self.target_length
        return within >= (self.output_steps - 1) * self.stride + self.receptive_field

    def frames_for_interval(self, start_ms: float, end_ms: float) -> range:
        """Frames whose onsets fall in [start_ms, end_ms) — a probe's label span."""
        return range(self.ms_to_frame(start_ms), self.ms_to_frame(end_ms))


def reassemble(
    captures: Sequence["torch.Tensor"],
    geometry: FrameGeometry,
    n_samples: int,
) -> "torch.Tensor":
    """Turn per-segment hook captures into one utterance-length representation.

    The hooks fire once per segment because get_audio_representation loops, so
    layer-wise extraction gets a list where it wanted an array. The model's own
    reassembly is a concatenation followed by a single tail truncation, and the
    index map is identity: hooked frame i is output frame i for all i < cut_off.
    There is no per-seam correction to apply.

    Args:
        captures: one tensor per segment, in segment order, each
            (batch, output_steps, width).
        geometry: read from the same model that produced the captures.
        n_samples: length of the ORIGINAL waveform, before the model padded it.
            This is what determines how much tail to drop.

    Returns:
        (batch, cut_off, width), matching what the model itself returns.
    """
    import torch

    expected = geometry.n_segments(n_samples)
    if len(captures) != expected:
        raise ValueError(
            f"expected {expected} segment captures for {n_samples} samples, "
            f"got {len(captures)}. A hook fired the wrong number of times, which "
            f"usually means the captures come from more than one utterance."
        )

    stacked = torch.cat(list(captures), dim=-2)
    return stacked[:, : geometry.n_output_frames(n_samples), :]
