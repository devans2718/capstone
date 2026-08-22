#!/usr/bin/env python3
"""Locate which input samples each conv frame actually reads.

WHY
    probe_frame_counts.py established how MANY frames come out. This one
    establishes WHERE they sit, which is what alignment timestamps need. The
    two obvious answers are both wrong:

        160.795 samples/frame   the average across a segment (32,159 / 200)
        160 samples/frame       the conv stride, correct only within a segment

    The conv stack has stride product 160 and receptive field 240, so frame j
    of a segment reads [j*160, j*160+240) — frame 199 ends at sample 32,080.
    But a segment is 32,159 samples. If the arithmetic is right, the last 79
    samples of every segment are read by no frame at all, and the global map is
    a staircase:

        frame_start_sample(i) = (i // 200) * 32159 + (i % 200) * 160

HOW
    Spike one input sample and find where the conv response peaks.

    Two normalizations make a naive "did anything change" test useless: the
    model's normalize() standardizes over the whole segment, and GroupNorm at
    conv layer 0 normalizes each channel over the whole time axis. Both mean a
    perturbation anywhere nudges every frame. Neither moves positions, only
    scales — so this reads the PEAK, and reports peak/median as the contrast.
    A spike inside some frame's window peaks sharply in the predicted frames; a
    spike in the slack produces a flat diffuse response with ratio near 1.

    Calls the extractor directly rather than the whole model, which removes
    normalize() from the path and leaves only the GroupNorm floor.

Exits 0 regardless: a diagnostic, not a gate. The result it establishes is
encoded in capstone.encoders.wavjepa.FrameGeometry and tested there.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModel

MODEL_ID = "labhamlet/wavjepa-base"
SPIKE = 50.0          # large enough to dominate the GroupNorm floor locally


def main() -> int:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    inner = model.model
    conv = inner.extract_audio

    target_length = inner.target_length
    output_steps = inner.output_steps
    stride = 1
    for _, _, s in conv.conv_layers_spec:
        stride *= s
    rf = conv.receptive_fields[0]
    last_read = (output_steps - 1) * stride + rf
    slack = target_length - last_read

    print("=== geometry from the checkpoint ===")
    print(f"  target_length   {target_length}")
    print(f"  output_steps    {output_steps}")
    print(f"  conv stride     {stride} samples between frame onsets")
    print(f"  receptive field {rf} samples per frame")
    print(f"  frame {output_steps - 1} reads up to sample {last_read}")
    print(f"  => predicted slack: {slack} samples per segment, read by no frame\n")

    torch.manual_seed(0)
    base = torch.randn(1, 1, target_length) * 0.1
    with torch.inference_mode():
        ref = conv(base).float()

    probes = [0, stride * 10, target_length // 2, last_read - 1,
              last_read, last_read + slack // 2, target_length - 1]

    print("=== where a single-sample spike peaks ===")
    print("  Slack splits into two regimes. A sample past the final layer's")
    print("  reach but still inside earlier layers' coverage shifts the")
    print("  GroupNorm statistics, giving a flat response with no peak. A")
    print("  sample past every layer's reach changes nothing at all.\n")
    header = (f"{'spike at':>10} {'predicted':>14} {'peak frame':>11} "
              f"{'peak/median':>12} {'max change':>11} {'verdict':>10}")
    print(header)
    print("-" * len(header))

    ok = True
    for p in probes:
        x = base.clone()
        x[:, :, p] += SPIKE
        with torch.inference_mode():
            out = conv(x).float()

        per_frame = (out - ref).abs().amax(dim=-1)[0]
        peak = int(per_frame.argmax())
        median = float(per_frame.median())
        ratio = float(per_frame.max()) / median if median > 0 else float("inf")

        lo = max(0, -(-(p - rf + 1) // stride))          # ceil division
        hi = min(output_steps - 1, p // stride)
        in_slack = p >= last_read
        predicted = "none (slack)" if in_slack else f"{lo}..{hi}"

        largest = float(per_frame.max())
        if in_slack:
            # No frame reads it, so either nothing changes at all, or the flat
            # GroupNorm floor shifts with no local peak. A sharp peak here would
            # falsify the whole account.
            good = largest < 1e-3 or ratio < 2.0
        else:
            good = lo <= peak <= hi
        ok &= good

        print(f"{p:>10} {predicted:>14} {peak:>11} {ratio:>12.1f} "
              f"{largest:>11.2e} {'ok' if good else 'MISMATCH':>10}")

    print("\n=== verdict ===")
    if ok:
        print("  Frame j reads [j*stride, j*stride + receptive_field) within its")
        print("  segment, and the segment tail is unread. The global map is")
        print("  piecewise:")
        print("      frame_start_sample(i) = (i // output_steps) * target_length")
        print("                            + (i %  output_steps) * stride")
        print("  A uniform frames-per-second figure is wrong in both directions.")
    else:
        print("  A probe landed outside its predicted frames. The conv arithmetic")
        print("  assumed here does not describe this checkpoint — do not trust")
        print("  FrameGeometry until this is understood.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
