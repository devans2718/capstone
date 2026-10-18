"""Frame geometry for wav2vec 2.0: the conversions between time, samples and frames.

WHY THIS MODULE EXISTS
    The same job as wavjepa.FrameGeometry, for a model whose geometry is much
    simpler. It is still derived from the checkpoint's config rather than typed
    in, because the span grid's registration requires it (span-grid README §4),
    and because a frame rate typed in is the one number nothing would catch.

THE ACTUAL GEOMETRY  (measured, see scripts/probe_w2v2_frame_positions.py)
    The conv stack has kernels (10, 3, 3, 3, 3, 2, 2) and strides
    (5, 2, 2, 2, 2, 2, 2): stride product 320 samples (20 ms), receptive field
    400 samples (25 ms). No conv pads its input, so frame t reads exactly

        [320 t, 320 t + 400)

    of the waveform it was given. Verified by spiking single samples: frame 20's
    response rises above the GroupNorm floor over [6400, 6800) and nowhere else.

    Unlike WavJEPA there are no segments, so the map is uniform: no staircase,
    no slack inside the input. What is unread is the tail, the last 0-319
    samples after the final whole window.

FRAMES BELONG TO AN INTERVAL BY CENTRE
    A frame is 25 ms wide against a median flap of 27.4 ms, so selection by
    onset (WavJEPA's rule) would credit a flap with a frame that reads mostly
    the preceding vowel. By centre, as MFCC (span-grid README §4): frame t
    belongs to [t0, t1) if 320 t + 200 lies in it.

DELIBERATELY IMPORT-LIGHT
    No torch, no transformers. `from_config` reads two lists off any object that
    has them, so the arithmetic is testable without the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class ConvGeometry:
    """Every time/sample/frame conversion a strided conv front end implies.

    Construct with `from_config` in real code. The explicit constructor exists so
    the arithmetic can be tested against known values without loading anything.
    """

    sample_rate: int
    stride: int             # conv stride product: samples between frame onsets
    receptive_field: int    # samples one frame reads

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_config(cls, config, sample_rate: int) -> "ConvGeometry":
        """Derive stride product and receptive field from the conv spec.

        `config` is a Wav2Vec2Config or anything with `conv_kernel` and
        `conv_stride`. The receptive field grows by (k - 1) input-side strides
        per layer: the standard recursion, with no padding term because the
        model's convs have none (the spike test is what establishes that).
        """
        stride, field = 1, 1
        for kernel, step in zip(config.conv_kernel, config.conv_stride, strict=True):
            field += (kernel - 1) * stride
            stride *= step
        return cls(sample_rate=sample_rate, stride=stride, receptive_field=field)

    # ------------------------------------------------------------------ basic units

    @property
    def frame_period_ms(self) -> float:
        """Time between consecutive frame onsets: exactly 20.000 ms."""
        return self.stride / self.sample_rate * 1000

    def n_output_frames(self, n_samples: int) -> int:
        """Frames the model returns for this many input samples.

        Every frame is a whole window; a partial one at the tail is dropped.
        Raises below one window, where the conv stack itself fails.
        """
        if n_samples < self.receptive_field:
            raise ValueError(
                f"{n_samples} samples is shorter than one frame's receptive field "
                f"({self.receptive_field}); the model returns no frames"
            )
        return (n_samples - self.receptive_field) // self.stride + 1

    # -------------------------------------------------------------- conversions

    def frame_start_sample(self, index: int) -> int:
        """First input sample frame `index` reads."""
        return index * self.stride

    def frame_centre_sample(self, index: int) -> float:
        """Centre of what frame `index` reads, in samples: 320 t + 200."""
        return index * self.stride + self.receptive_field / 2

    def frame_span_ms(self, index: int) -> tuple[float, float]:
        """The (start, end) of what frame `index` reads, in ms.

        25 ms wide on a 20 ms hop, so consecutive frames overlap by 5 ms.
        """
        start = self.frame_start_sample(index) / self.sample_rate * 1000
        return start, start + self.receptive_field / self.sample_rate * 1000

    def frames_for_interval(self, start_ms: float, end_ms: float) -> range:
        """Frames whose centres fall in [start_ms, end_ms).

        Both ends are rounded to samples first, as extract_site_features.py
        rounds a site's bounds for MFCC, so the two centre rules agree on what
        an interval is. Can be empty: an interval under one hop may contain no
        centre. Not clamped to the output length; the caller knows it.

        Closed form, in integers: centre(t) >= t0  <=>  t >= (2 t0 - rf) / 2s.
        """
        t0, t1 = (int(round(t / 1000 * self.sample_rate)) for t in (start_ms, end_ms))
        rf, s = self.receptive_field, self.stride
        first = max(0, _ceil_div(2 * t0 - rf, 2 * s))
        stop = max(0, _ceil_div(2 * t1 - rf, 2 * s))
        return range(first, max(first, stop))

    def nearest_frame(self, sample: float) -> int:
        """The frame whose centre is nearest `sample`; a tie goes to the earlier.

        The tie rule is MFCC's midpoint rule (extract_site_features.py), so the
        two centre-based representations break ties the same way.
        """
        x = (sample - self.receptive_field / 2) / self.stride
        lo = floor(x)
        return max(0, lo if x - lo <= 0.5 else lo + 1)
