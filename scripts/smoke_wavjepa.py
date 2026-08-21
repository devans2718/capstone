#!/usr/bin/env python3
"""Verify WavJEPA runs end to end AND that per-layer hidden states are reachable.

This is the gate for the advisor's step 1. "Does the model load" is the easy half
and not the half that matters: RQ1's layer-wise probe gap and all of RQ3 need one
representation *per encoder layer*, and WavJEPA does not give you that for free.

    class WavJEPAModel(PreTrainedModel):
        def forward(self, tensor):
            return self.model.get_audio_representation(tensor)

That signature takes no kwargs, so the usual `output_hidden_states=True` raises
rather than degrading, and the encoder underneath is a stock nn.TransformerEncoder
invoked as a black box. Forward hooks on the individual layers are therefore the
extraction mechanism for this project, not a workaround. This script proves they
work and pins down the tensor shapes the rest of the pipeline will assume.

Exits nonzero on any failure so it can gate a build.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL_ID = "labhamlet/wavjepa-base"
SAMPLE_RATE = 16_000          # model card: audio must be resampled to 16 kHz
CLIP_SECONDS = 10             # deliberately > the model's 2 s internal segment
EXPECTED_LAYERS = 12          # config.json: encoder_cfg.num_layers
EXPECTED_WIDTH = 768          # config.json: encoder_cfg.d_model


def find_encoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the ModuleList of transformer encoder layers to hook.

    The expected path is model.model.encoder.layers, but that is an undocumented
    detail of remote code that upstream can rename in any push. Rather than fail
    with an AttributeError, fall back to a search and report what was found, so a
    break here tells us where the layers moved instead of just that they moved.
    """
    try:
        layers = model.model.encoder.layers
        print(f"  encoder layers found at the expected path: model.model.encoder.layers")
        return layers
    except AttributeError:
        print("  !! model.model.encoder.layers missing — searching module tree")

    candidates = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.ModuleList)
        and len(mod) == EXPECTED_LAYERS
        and any(isinstance(m, torch.nn.TransformerEncoderLayer) for m in mod)
    ]
    if not candidates:
        print("  !! no ModuleList of TransformerEncoderLayer found. Module tree:")
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList):
                print(f"       {name}: ModuleList(len={len(mod)}) of {type(mod[0]).__name__}")
        raise SystemExit("FAIL: cannot locate encoder layers to hook")

    name, layers = candidates[0]
    print(f"  encoder layers found at a NEW path: {name} (update this script)")
    return layers


def main() -> int:
    print("=== environment ===")
    print(f"  torch        {torch.__version__}")
    print(f"  cuda built   {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"  cuda avail   {cuda_ok}")
    if cuda_ok:
        print(f"  device       {torch.cuda.get_device_name(0)}")
    else:
        print("  !! no GPU visible — check the compose device reservation")
    device = "cuda" if cuda_ok else "cpu"

    print("\n=== load ===")
    # trust_remote_code: the modeling code lives on the Hub, not in transformers.
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {MODEL_ID}: {n_params/1e9:.2f}B params on {device}")

    print("\n=== hook encoder layers ===")
    layers = find_encoder_layers(model)
    print(f"  layer count: {len(layers)} (expected {EXPECTED_LAYERS})")

    # Each hook appends once per forward call. get_audio_representation() chunks
    # the waveform into 2 s segments and loops, so a 10 s clip fires every hook
    # ~5 times. Capturing a list (not a tensor) is what makes that visible here
    # and is the reassembly problem the real extractor has to solve.
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(layers))}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            captured[idx].append(output.detach().float().cpu())
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    print("\n=== forward ===")
    audio = torch.randn(1, SAMPLE_RATE * CLIP_SECONDS)  # noise is fine; shapes are the assertion
    print(f"  input: {CLIP_SECONDS}s @ {SAMPLE_RATE}Hz -> {tuple(audio.shape)}")
    batch = extractor(audio, return_tensors="pt")
    inputs = batch["input_values"].to(device)
    with torch.inference_mode():
        result = model(inputs)

    for h in handles:
        h.remove()

    final = result[0] if isinstance(result, (tuple, list)) else result
    print(f"  model output: {tuple(final.shape) if hasattr(final, 'shape') else type(final)}")

    print("\n=== captured per-layer states ===")
    ok = True
    for idx in sorted(captured):
        chunks = captured[idx]
        if not chunks:
            print(f"  layer {idx:2d}: NOTHING CAPTURED")
            ok = False
            continue
        shape = tuple(chunks[0].shape)
        if idx in (0, len(layers) - 1):
            print(f"  layer {idx:2d}: {len(chunks)} call(s), each {shape}")
        width = chunks[0].shape[-1]
        if width != EXPECTED_WIDTH:
            print(f"  layer {idx:2d}: width {width} != expected {EXPECTED_WIDTH}")
            ok = False
    print(f"  ... layers 1-{len(layers)-2} identical in shape, omitted")

    calls = len(captured[0])
    frames = sum(c.shape[-2] for c in captured[0])
    print(f"\n  segments per {CLIP_SECONDS}s clip: {calls}")
    print(f"  total frames across segments: {frames}")
    # Deliberately NOT reported as a frame rate. Dividing hooked frames by clip
    # duration mixes a count that includes the model's own zero-padding with a
    # duration that excludes it, and there is no single frame rate to report
    # anyway: frames advance 160 samples inside a segment while segment origins
    # advance 32,159. Use capstone.encoders.wavjepa.FrameGeometry for any
    # frame/time conversion; this script only proves the hooks fire.
    print(f"  ({frames} hooked frames includes padded tail — not a frame rate)")

    if len(layers) != EXPECTED_LAYERS:
        print(f"\nFAIL: expected {EXPECTED_LAYERS} layers, hooked {len(layers)}")
        ok = False

    print("\n" + ("PASS: per-layer extraction works." if ok else "FAIL: see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
