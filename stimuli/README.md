# stimuli

Self-recorded audio for the RQ1 context-disambiguation control: код/кот
tokens placed in disambiguating sentence contexts (plus any TTS-synthesized
supplements). These recordings exist nowhere else — committing them here is
deliberate; the private GitHub remote is their offsite backup.

Suggested naming: `<word>_<context-id>_<take>.wav` with a `manifest.csv`
mapping filenames to sentence context, recording date, and notes.

Everything below is build-time instruction. Read it before recording, not
after — every constraint here is cheap to satisfy while a microphone is set up
and expensive or impossible to repair afterward.

---

## The 2.01 s context window is a hard ceiling

WavJEPA's positional embedding table is exactly 200 entries, fixed and not
learned, so the transformer can only ever receive 200 tokens. Audio longer
than one window is cut into consecutive 32,159-sample (2.01 s) segments and
each is encoded by a *separate* forward pass. No attention crosses a segment
boundary; there is no cross-segment mechanism at all.

**The target token and the context that disambiguates it must fall inside the
same 2.01 s window, or the model provably cannot use the context.** A probe
that fails under those conditions has measured the chunking, not the encoder.

This is not a tuning parameter. It is the shape of the weights.

### Boundaries sit on a grid measured from sample zero

Segment boundaries are laid out from the first sample of whatever you hand the
model, so they are a property of the *file*, not of the speech:

```
file A:  [--------seg 0--------][--------seg 1--------]
              кот  ^context                                  both in seg 0 — usable

file B:  [--------seg 0--------][--------seg 1--------]
                          кот   ^context                     split — context invisible
```

File B differs from file A only by some leading silence. Consequences for
recording and for preprocessing:

- **Keep each stimulus under 2.01 s end to end** where the design allows it.
  One window, one forward pass, no boundary to reason about. This is the
  simplest way to make the whole problem disappear.
- If a stimulus must be longer, **place the target and its context near the
  middle** so a boundary is unlikely to fall between them, and verify per file
  rather than assuming.
- **Trim leading silence consistently across conditions.** Uneven leading
  silence moves the grid by condition, which manufactures a difference between
  conditions that has nothing to do with the linguistics.
- Note that this makes leading silence a *confound*, not just untidiness.

## Amplitude is renormalized every 2.01 s

Two normalizations run before the encoder sees anything: one clip-global gain
to −14 dBFS, then per-segment `(x - mean) / (std + 1e-5)` on each window.

The encoder therefore **never sees absolute amplitude**, and relative amplitude
is only preserved *within* a segment. Practical consequences:

- Do not design a contrast that depends on one condition being louder than
  another. It will be normalized away.
- Do not worry about matching absolute recording levels between takes. It will
  also be normalized away. Consistency of *room, mic, distance and mouth
  position* is what matters; gain is not.
- Voicing, duration, spectral shape and F0 all survive both normalizations, so
  RQ1's actual cues are unaffected.

## Splice mechanics

Coarticulation will not match when a token recorded in one carrier is dropped
into another, and a click or spectral jump at the seam is a difference the
encoder can detect that has nothing to do with context. A probe can score well
on splice artifacts alone.

- **Record both carriers with the target in place, then cross-splice**, so each
  condition contains one native token and one transplant. The artifact is then
  balanced across conditions instead of aligned with them.
- **Cut at zero crossings.** A discontinuity in the waveform is a broadband
  transient — exactly the kind of thing early conv layers respond to.
- **Listen to every stimulus.** If the seam is audible, it is in the signal.
- Prefer splicing at a **silent closure** rather than mid-vowel, where formant
  trajectories have to line up.

The analysis-side counterpart is a free control: early layers have a 15 ms
receptive field and no context, so if a probe succeeds on layer 0 as well as
layer 8, it is reading acoustics rather than contextual representation. Run it.

## Recording format

Record at **16 kHz mono or higher and resample down**; the model requires
16 kHz. Keep the original higher-rate masters if recording above it — they are
tier-1 irreplaceable, and resampling is reproducible while un-resampling is
not.

Do not normalize, compress, or noise-reduce on the way in. Every one of those
is either undone by the model's own normalization or is an uncontrolled
transformation applied unevenly across conditions.
