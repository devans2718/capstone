# Pilot: English flapping probe — pre-registration

**Written 2026-08-22, before any probe was trained.** This is the advisor's step 4,
and it exists so that step 5's result can be read against criteria that were fixed
in advance. The commit date is the evidence; that is the point of the file.

Scope: WavJEPA (frozen) on the Buckeye flapping set, one model, one dataset,
layer-wise. This is the gate before extending to the baseline encoders.

**The thresholds below are judgment calls, not derived bounds.** They were chosen
without a result in hand, which is the only property that matters about them. If
one turns out to be badly calibrated, that is a finding to record — not a licence
to move it after seeing the numbers. See *Deviations* at the bottom.

---

## 1. Task and labels

**Task.** Binary classification of the underlying segment beneath a realized flap:
was this `[dx]` an underlying /t/ or /d/?

**Label source.** Buckeye's hand-corrected `.phones` tier, aligned to `.words`.
A site is a `[dx]` inside a word whose canonical and actual phone strings are the
same length, so position maps to an underlying segment unambiguously.
Implemented in `scripts/count_flap_sites.py`; inventory at
`results/corpus-verification/buckeye-flap-sites-20260821.{txt,tsv}`.

**Labels are underlying, not surface.** Surface is `[dx]` in every case by
construction — that is what makes the task a test of abstraction rather than of
acoustic classification.

**Exclusions, all fixed in advance:**

- Flaps carrying the `*` annotator mark — hedged labels.
- Words where canonical and actual strings differ in length (**21% of
  flap-bearing words**). These are the heavily reduced tokens (`about → b ah dx`);
  recovering them needs edit-distance alignment between tiers, which is out of
  scope here. They are reported, never silently dropped.
- Sites whose excerpt cannot be centred at the chosen context width
  (`Excerpt.clamped`) — **3 of 10,392 at 200 ms**.
- The 4 sites containing no frame onset at all.

**Resulting inventory: 10,392 sites, 7,863 /t/ and 2,529 /d/** — a 75.7 / 24.3
split across 40 speakers, every speaker carrying both classes (min 96, median 249).

## 2. Representations

`capstone.encoders.extraction.encode()` → `(13, n_frames, 768)` per utterance.
Index 0 is the input to `layers[0]`; 1–12 are encoder layer outputs; index 12 is
pre-final-LayerNorm. All 13 are probed.

**Fixed by the extraction findings, not open to tuning here:**

- **`context_ms = 200`, constant across every site.** Constancy is the binding
  constraint, not the value: the positional encoding is re-added from position 0
  each segment, so a frame's representation depends on where it sits within its
  segment (cosine distance 0.394 at layer 0 for a 10-frame shift, decaying to
  0.021 by layer 12). Holding context constant makes that contribution a constant
  that cannot correlate with the label. Vary it and the confound lands precisely
  in the early layers used as the control. 200 ms is the judgment call — wide
  enough for the conditioning vowels, and it clamps only 3 sites against 35 at
  990 ms.
- **One `encode()` call per utterance; excerpts are never re-chunked.** The
  feature extractor applies a single RMS-derived gain over the whole waveform.
- **Frame→time conversion goes through `FrameGeometry`.** The map is a staircase,
  not a uniform rate; no frame rate is retyped anywhere.

**Pooling: mean over the site's frame onsets.** Registered as the default so the
pilot has one rule, *not* as a settled finding. Sites are 2–3 frames wide (median
27.4 ms, 41% have ≤2 onsets), so mean / max / midpoint are materially different
vectors rather than variants of one summary. Max and midpoint are named here as
the pre-declared secondary comparison; anything beyond those three is exploratory
and gets labelled as such.

## 3. Probe

**Linear (logistic regression) on frozen features.** Deliberately low capacity:
selectivity degrades as the probe gets expressive enough to memorize, and a
linear probe keeps "the encoder represents this" the simplest reading of a
positive result. A stronger probe is a follow-up, not part of the gate.

**Class imbalance: `class_weight='balanced'`, all 10,392 sites retained.**
The alternative — subsampling to the balanced ceiling of 5,058 — discards 5,334
sites to buy the convenience of accuracy and balanced accuracy coinciding. Not
worth it when the metric can simply be chosen correctly instead.

**Splits: 5-fold `GroupKFold` grouped by speaker.** Speaker-disjoint by
construction. Chosen over a single held-out split because 40 speakers is few
enough that one split's composition could carry the result, and the fold-to-fold
spread is itself needed to judge whether differences between layers are real.
Report mean ± sd across folds at every layer.

## 4. Controls

**Control task (Hewitt & Liang 2019, *Designing and Interpreting Probes with
Control Tasks*).** Each word *type* is assigned a random /t/-or-/d/ label,
matching the real marginal; every token of that type inherits it. The mapping is
seeded and recorded. A probe can only fit this by memorizing, so control accuracy
measures probe capacity rather than encoder content.

**Selectivity = real balanced accuracy − control balanced accuracy**, computed
per layer, same probe and hyperparameters throughout.

**Speaker-leakage check.** A probe trained on the same representations to predict
speaker identity, reported alongside. Not a pass/fail gate — a diagnostic to
consult if the main result looks too strong.

## 5. Criteria — what counts as the pilot failing

Fixed 2026-08-22, before training. All refer to the **best layer** unless stated.

| # | Criterion | Threshold |
|---|---|---|
| 1 | Balanced accuracy | **≥ 0.65** |
| 2 | Selectivity (real − control) | **≥ 0.10** |
| 3 | Layer-wise curve is not flat | best − worst layer **≥ 0.05** balanced accuracy |
| 4 | Minority-class F1 (/d/) | **≥ 0.50** |

Chance is 0.50 balanced accuracy. Criterion 1 is set well above it because the
deliverable is a *comparison across layers*, and a signal too weak to separate
layers cannot support RQ3 regardless of significance. Criterion 4 exists so an
aggregate cannot hide a probe that has quietly stopped predicting /d/.

**Minimum sample count.** The registered analysis needs ≥ 500 sites of the
minority class per fold; at 2,529 /d/ over 5 folds (~506 each) this is met with
almost no margin, which is itself worth knowing before the run.

**If criteria are met:** proceed to step 5's write-up and then to the baseline
encoders (RQ3).

**If they are not, the pilot has failed, and failure is diagnostic rather than
terminal.** Read in this order:

- *Near chance at every layer* — suspect the label/alignment path or the 2–3 frame
  site width before concluding anything about the encoder.
- *High real accuracy with high control accuracy* — probe too expressive, or
  speaker leakage. Check the speaker probe.
- *Good but flat across layers* — a real result, and one that reframes RQ3 rather
  than ending it.

None of these outcomes licences re-running with adjusted thresholds and reporting
the second run as the first.

## 6. Deviations

Any departure from the above gets appended here with a date and a reason, and the
original text stays as written. An amended pre-registration with a visible
amendment history is still worth something; a silently edited one is worth
nothing.

*(none yet)*
