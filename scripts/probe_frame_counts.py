#!/usr/bin/env python3
"""Explain the hooked-frames vs. output-frames gap, from the model's own numbers.

THE OBSERVATION
    A 10 s clip fires each encoder-layer hook 5 times at (1, 200, 768) per call
    -> 1000 hooked frames. The model's own output is (1, 996, 768). Four frames
    are unaccounted for. Probe labels come from forced-alignment timestamps, so
    an unexplained off-by-N does not shift RQ1's labels, it corrupts them.

WHY COUNTING ALONE CANNOT SETTLE IT
    An earlier version of this script compared clip lengths and concluded "one
    frame lost per seam between segments," because losses went 0, 1, 2, 4 for
    1, 2, 3, 5 segments. That fits. So does "N frames trimmed off the tail," and
    the two imply different index maps. Totals cannot separate them. Reading
    model.py did:

        x = torch.hstack(embeddings)   # every hooked frame, concatenated
        x = x[:, :cut_off, :]          # one truncation, off the end

    There are no per-seam deletions. The frames go off the tail, and they are
    the frames corresponding to zero-padding the model added itself.

WHAT cut_off ACTUALLY IS  (utils.calculate_padding_mask)
        n_segments   = padded_samples // target_length
        total_steps  = output_steps * n_segments        # 200 per 2 s segment
        output_sr    = output_steps // process_seconds  # 100 Hz
        pad_steps    = int((pad_frames / sample_rate) * output_sr)
        cut_off      = total_steps - pad_steps

    So the model pads the waveform up to a whole number of 2 s segments, encodes
    every segment including the padded tail, then discards exactly the frames
    that the padding produced.

WHAT THIS SCRIPT DOES
    Prints the real intermediate quantities per clip length rather than
    inferring them: what the feature extractor emits, target_length, pad_frames,
    segment count, pad_steps, predicted cut_off, and the observed output length.
    A mismatch between predicted and observed means the account above is wrong
    somewhere and should be read, not patched over.

    It also checks segment independence directly. normalize() is applied per
    segment inside the loop, so segment 0's representation should not depend on
    what follows it. If a 2 s prefix encoded alone equals the first 200 frames
    of the same audio encoded as part of a 10 s clip, then reassembly really is
    plain concatenation and hooked frame i is output frame i for all i < cut_off.

Exits 0 regardless: this is a diagnostic, not a gate.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL_ID = "labhamlet/wavjepa-base"
SAMPLE_RATE = 16_000
CLIP_SECONDS = [2, 4, 6, 10]


def find_encoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    try:
        return model.model.encoder.layers
    except AttributeError:
        raise SystemExit(
            "FAIL: model.model.encoder.layers is gone — upstream moved the layers. "
            "Run scripts/smoke_wavjepa.py, which searches the module tree."
        )


def encode(model, extractor, layers, device, audio: torch.Tensor) -> dict:
    """Encode one waveform, capturing per-call hook shapes alongside the output."""
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().float().cpu())

    handle = layers[0].register_forward_hook(hook)
    try:
        batch = extractor(audio, return_tensors="pt")
        inputs = batch["input_values"].to(device)
        with torch.inference_mode():
            result = model(inputs)
    finally:
        handle.remove()

    # forward() returns (x, timestamps); x is the representation.
    final = result[0] if isinstance(result, (tuple, list)) else result

    return {
        "extractor_samples": inputs.shape[-1],
        "calls": len(captured),
        "per_call": [tuple(c.shape) for c in captured],
        "hooked": sum(c.shape[-2] for c in captured),
        "output": final.shape[-2],
        "repr": final.detach().float().cpu(),
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== load === ({device})")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    layers = find_encoder_layers(model)

    inner = model.model
    target_length = inner.target_length      # samples per internal segment
    output_steps = inner.output_steps        # frames emitted per segment
    sr = inner.sample_rate
    process_seconds = target_length // sr
    output_sr = int(output_steps / process_seconds)

    print(f"  {MODEL_ID}, {len(layers)} encoder layers")
    print(f"  sample_rate    {sr}")
    print(f"  target_length  {target_length} samples = {process_seconds}s per segment")
    print(f"  output_steps   {output_steps} frames per segment -> {output_sr} Hz\n")

    print("=== per clip: the model's own arithmetic vs. what came out ===")
    header = (f"{'clip':>5} {'ext.samples':>12} {'pad':>7} {'segs':>5} "
              f"{'hooked':>7} {'padsteps':>9} {'pred':>6} {'actual':>7} {'ok':>4}")
    print(header)
    print("-" * len(header))

    runs = {}
    all_ok = True
    for seconds in CLIP_SECONDS:
        torch.manual_seed(seconds)
        audio = torch.randn(1, SAMPLE_RATE * seconds)
        r = encode(model, extractor, layers, device, audio)
        runs[seconds] = r

        cur = r["extractor_samples"]
        # model.py: pad up to a whole number of segments. Note this pads a FULL
        # segment when cur is already an exact multiple, which is worth seeing
        # explicitly rather than assuming it away.
        pad_frames = target_length - (cur % target_length)
        n_seg = (cur + pad_frames) // target_length
        pad_steps = int((pad_frames / sr) * output_sr)
        pred = output_steps * n_seg - pad_steps

        ok = pred == r["output"]
        all_ok &= ok
        print(f"{seconds:>4}s {cur:>12} {pad_frames:>7} {n_seg:>5} "
              f"{r['hooked']:>7} {pad_steps:>9} {pred:>6} {r['output']:>7} "
              f"{'yes' if ok else 'NO':>4}")

    print("\n=== hook call shapes ===")
    for seconds, r in runs.items():
        uniform = len(set(r["per_call"])) <= 1
        shown = r["per_call"][0] if uniform else r["per_call"]
        print(f"  {seconds:>2}s: {r['calls']} call(s), "
              f"{'all ' + str(shown) if uniform else 'NON-UNIFORM ' + str(shown)}")

    # ---- segment independence -------------------------------------------------
    # Encode the opening segment of the 10 s waveform on its own and compare.
    # normalize() runs on the segment slice, so segment 0 should not depend on
    # what follows it — if so, concatenation is plain and the map is identity.
    #
    # The prefix must be exactly target_length samples, NOT 2 s. A 2 s prefix is
    # 32000 samples and gets 159 zeros appended, while segment 0 of the longer
    # clip is 32159 samples of real audio. Comparing those is comparing two
    # different inputs, and it reports a difference that means nothing.
    print("\n=== segment independence ===")
    torch.manual_seed(10)
    long_audio = torch.randn(1, SAMPLE_RATE * 10)
    prefix = long_audio[:, :target_length].clone()

    long_r = encode(model, extractor, layers, device, long_audio)
    pre_r = encode(model, extractor, layers, device, prefix)

    k = min(pre_r["output"], long_r["output"], output_steps)
    a = pre_r["repr"][:, :k, :]
    b = long_r["repr"][:, :k, :]
    max_abs = (a - b).abs().max().item()

    # Control: the same waveform encoded twice. Non-deterministic kernel
    # selection and TF32 matmuls make float32 GPU inference reproducible only to
    # a tolerance, and that tolerance — not an arbitrary threshold — is what the
    # comparison above has to beat to mean anything.
    long_again = encode(model, extractor, layers, device, long_audio)
    noise = (long_r["repr"][:, :k, :] - long_again["repr"][:, :k, :]).abs().max().item()
    scale = b.abs().max().item()

    # The question that actually matters is not "is there any difference" but
    # "is it an indexing difference." A one-frame misalignment would make frame i
    # resemble frame i+1, so the size of a deliberate one-frame shift is the
    # yardstick: an index error lands near it, an amplitude residual lands orders
    # of magnitude below.
    shifted = (b[:, :-1, :] - b[:, 1:, :]).abs().max().item()

    print(f"  comparing first {k} frames: {target_length}-sample prefix alone")
    print(f"    vs. the same samples inside the 10 s clip")
    print(f"  max abs difference   : {max_abs:.3e}")
    print(f"  run-to-run noise     : {noise:.3e}  (same input encoded twice)")
    print(f"  representation scale : {scale:.3e}")
    print(f"  a ONE-FRAME shift is : {shifted:.3e}  (adjacent-frame distance)")

    if max_abs > 0.1 * shifted:
        print("  => difference is comparable to a one-frame shift. This IS an")
        print("     indexing problem; do not build the extractor until it is read.")
    elif max_abs <= max(noise * 10, 1e-6 * scale):
        print("  => difference is at the numerical-noise floor; segments are")
        print("     encoded independently and concatenation is plain.")
    else:
        print("  => amplitude-level residual, orders below a one-frame shift, so")
        print("     the index map is intact: hooked frame i == output frame i.")
        print("     Source is the feature extractor: _normalize_audio applies one")
        print("     RMS-derived gain computed over the WHOLE waveform, and the")
        print("     epsilon in per-segment (x-mean)/(std+1e-5) keeps that gain")
        print("     from cancelling exactly. Practical rule: encode an utterance")
        print("     in ONE call. Re-chunking it perturbs representations slightly.")

    # ---- frame period ---------------------------------------------------------
    # THREE rates are in play and none of them agree. This script can only see
    # two, because both are derived from output *lengths*:
    #
    #   target_length / output_steps   the AVERAGE across a segment, 10.0497 ms
    #   get_timestamps()               clip duration / surviving frame count
    #
    # The third is the one that actually governs alignment: the conv stride,
    # exactly 160 samples = 10.000 ms between frame onsets within a segment,
    # with segment origins advancing by target_length instead. Counting frames
    # cannot see it — an earlier version of this block called the segment
    # average "encoder-internal" and was wrong for that reason.
    #
    # scripts/probe_frame_positions.py locates frames rather than counting them
    # and settles it. capstone.encoders.wavjepa.FrameGeometry is the only place
    # any of this should be computed.
    print("\n=== frame period (lengths only — see probe_frame_positions.py) ===")
    mean_ms = target_length / sr * 1000 / output_steps
    print(f"  segment average : {target_length}/{output_steps} samples per frame "
          f"= {mean_ms:.4f} ms")
    for seconds, r in runs.items():
        ts_ms = seconds * 1000 / r["output"]
        spread = (mean_ms - ts_ms) * r["output"]
        print(f"  {seconds:>2}s get_timestamps(): {ts_ms:.4f} ms/frame "
              f"-> {spread:+.1f} ms apart by the final frame")

    print("\n=== verdict ===")
    if all_ok:
        print("  cut_off = output_steps*n_segments - pad_steps reproduces every")
        print("  observed output length. The missing frames are padding-derived")
        print("  and sit at the TAIL, not at segment seams.")
        print("  Extractor rule: hstack the per-segment hook captures, then")
        print("  truncate to the model's output length. No per-seam correction.")
    else:
        print("  The formula did NOT reproduce every output length (see 'ok' column).")
        print("  Do not build the extractor on it — read model.py and utils.py.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
