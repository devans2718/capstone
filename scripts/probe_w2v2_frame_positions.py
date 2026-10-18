#!/usr/bin/env python3
"""Locate which input samples each wav2vec 2.0 conv frame actually reads.

WHY
    The counterpart of probe_frame_positions.py, which did this for WavJEPA.
    The config gives kernels and strides, and from them a stride product of 320
    and a receptive field of 400 (capstone.encoders.wav2vec2.ConvGeometry). It
    does not say how the code applies them: padding in a conv, or an offset,
    would keep the frame count and the hop and shift every window in time.
    Every frame label would then be off by a constant, and nothing would raise.
    The span grid's registration requires the geometry verified, not read off
    the config (span-grid README §4).

HOW
    Spike one input sample and find which frame's conv output moves most.

    Conv layer 0 has a GroupNorm that normalizes each channel over the whole
    time axis, so a spike anywhere nudges every frame: "did this frame change"
    tells frames apart by nothing. The test reads the PEAK, and reports
    peak/median as the contrast. A spike inside some frame's window peaks in the
    predicted frames; a spike in the unread tail gives a flat response.

    Then a sweep: one fixed frame, the spike moved one sample at a time across
    its predicted window and 80 samples either side. The samples whose spike
    raises that frame above twice the GroupNorm floor are its receptive field,
    measured rather than computed.

    Calls the conv stack directly, which removes do_normalize from the path and
    leaves only the GroupNorm floor.

Exits 0 regardless: a diagnostic, not a gate. The result it establishes is
encoded in ConvGeometry and tested in tests/test_wav2vec2_geometry.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from capstone.encoders.extraction import W2V2_REVISION, Wav2Vec2Extractor  # noqa: E402

N = 16_000            # one second of input: 49 frames
SPIKE = 50.0          # against noise of std 0.1: dominates the GroupNorm floor locally
SWEEP_FRAME = 20
SWEEP_MARGIN = 80     # samples either side of the predicted window


def main() -> int:
    ext = Wav2Vec2Extractor.load(device="cpu")
    conv = ext.model.feature_extractor
    g = ext.geometry
    stride, rf = g.stride, g.receptive_field
    T = g.n_output_frames(N)
    last_read = g.frame_start_sample(T - 1) + rf

    print(f"=== geometry from the config @ {W2V2_REVISION} ===")
    print(f"  conv stride     {stride} samples between frame onsets")
    print(f"  receptive field {rf} samples per frame")
    print(f"  {N} samples -> {T} frames; frame {T - 1} reads up to sample {last_read}")
    print(f"  => predicted: {N - last_read} tail samples read by no frame\n")

    torch.manual_seed(1)
    base = torch.randn(1, N) * 0.1
    with torch.inference_mode():
        ref = conv(base).float()                        # (1, 512, T)

    def response(p: int) -> torch.Tensor:
        """Per frame, the largest change over the 512 channels."""
        x = base.clone()
        x[0, p] += SPIKE
        with torch.inference_mode():
            out = conv(x).float()
        return (out - ref).abs().amax(dim=1)[0]         # (T,)

    print("=== where a single-sample spike peaks ===")
    header = (f"{'spike at':>10} {'predicted':>10} {'peak frame':>11} "
              f"{'peak/median':>12} {'verdict':>9}")
    print(header)
    print("-" * len(header))
    ok = True
    for p in (0, 199, 320, 399, 400, 5000, 8000, last_read - 1, last_read, N - 1):
        r = response(p)
        peak = int(r.argmax())
        median = float(r.median())
        ratio = float(r.max()) / median if median > 0 else float("inf")
        lo = max(0, math.ceil((p - rf + 1) / stride))
        hi = min(T - 1, p // stride)
        if p >= last_read:
            # No frame reads it: only the flat GroupNorm floor may move. A sharp
            # peak here would falsify the account.
            predicted, good = "none", ratio < 2.0 or float(r.max()) < 1e-3
        else:
            predicted, good = f"{lo}..{hi}", lo <= peak <= hi
        ok &= good
        print(f"{p:>10} {predicted:>10} {peak:>11} {ratio:>12.1f} "
              f"{'ok' if good else 'MISMATCH':>9}")

    t = SWEEP_FRAME
    start, end = g.frame_start_sample(t), g.frame_start_sample(t) + rf
    lo_s, hi_s = start - SWEEP_MARGIN, end + SWEEP_MARGIN
    sens = torch.stack([response(p)[t] for p in range(lo_s, hi_s)])
    # The floor: how much the same spikes move a frame 10 away, which reads none
    # of them. Sampled every 37 samples; it is flat, so this is plenty.
    floor = float(torch.stack([response(p)[t + 10]
                               for p in range(lo_s, hi_s, 37)]).max())
    above = [lo_s + i for i, v in enumerate(sens) if v > 2 * floor]
    w = (sens - floor).clamp(min=0)
    centroid = float((w * torch.arange(lo_s, hi_s)).sum() / w.sum())
    measured = (above[0], above[-1] + 1)
    ok &= measured == (start, end)

    print(f"\n=== frame {t}'s receptive field, swept ===")
    print(f"  predicted reads   [{start}, {end})")
    print(f"  measured          [{measured[0]}, {measured[1]})  "
          f"(above 2x the GroupNorm floor, {floor:.2e})")
    print(f"  predicted centre  {g.frame_centre_sample(t):.1f}; response centroid "
          f"{centroid:.1f} ({centroid - g.frame_centre_sample(t):+.1f} samples)")

    print("\n=== verdict ===")
    if ok:
        print("  Frame t reads [t*stride, t*stride + receptive_field) of its input,")
        print("  with no padding or offset. The map is uniform; a frame's centre is")
        print("  t*stride + receptive_field/2.")
    else:
        print("  A spike landed outside its predicted frames, or the swept field is")
        print("  not the computed one. ConvGeometry does not describe this checkpoint;")
        print("  do not trust it until this is understood.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
