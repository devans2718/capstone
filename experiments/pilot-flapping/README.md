# Pilot: English flapping probe — pre-registration

**Revision 3 — in progress, not final.** Nothing has been run. Still pending
before this revision is final and pushed:

- whichever Challenge 2 feedback items are adopted: the MFCC feature
  configuration (§4), and optionally ABX, a random-split condition, and
  publishing the token list and split files;
- the §3 split construction moved into `src/` as shared code, used by the
  probe run and the demo slice alike. The slice currently draws random fold
  numbers for its synthetic rows.

**Changes in revision 3**, each from measuring the corpus or the code, none
from a result:

1. **Word-type tie-break registered (§3).** 985 of the 1,022 types share a
   site count with another type. Revision 2 left their order to
   `Counter.most_common()`, which is the inventory file's row order. So its
   claim that the split was "reproducible from the inventory alone" held only
   while that order held. Ties now break alphabetically; the two rules place
   777 types (2,848 sites) in different folds. Measured in
   `scripts/analyze_flap_lexical_structure.py` §5.
2. **All 25 cells of the speaker × type grid are scored (§3, §4, §6).**
   Revision 2 tested only the 5 diagonal cells, which left ~80% of sites
   unscored. The same construction scores every site once, and no probe trains
   on less than before.
3. **Frame counting fixed (§1, §2).** `frames_for_interval` dropped the frame
   starting exactly at each flap's onset (decision D1 in
   `docs/challenges/challenge-2-slice-decisions.md`). Fixed, every site has at
   least one frame, and **the §1 exclusion of sites with no frame onset now
   selects nothing**. Revision 2 said 4 sites; the pre-fix extractor would have
   found 39. Sites with ≤2 frames fall from 41% to 18.4%.
4. **§5 criteria finalized.** Criterion 1 compares against the MFCC baseline
   (newly registered in §4) instead of an absolute 0.60. Criterion 2 is a leak
   bound instead of a selectivity floor, because under this split Hewitt &
   Liang's own ceiling argument puts the control at chance. Criterion 4 keeps
   0.45 with its trivial floors stated. Hewitt & Liang, the input revision 2
   named, turned out to state no thresholds at all; see §5.
5. **Transcription neutralisation recorded (§1a).** 30.5% of sites share a
   transcribed pronunciation with the other class. This is a population note,
   not a change to the design.
6. **Correction.** Revision 2 gave the pooled diagonal as "≈1,990 sites". Its
   own script summed to 2,104. The /d/ figure, 516, was right.

**Revision 2, 2026-08-22. Nothing has been run at any point in this file's
history.** Revision 1 (`b0b2bce`, pushed) registered speaker-disjoint splits
only. It was revised, not deviated from, after measuring that **word identity
alone scores 0.9951 on the real task** — a probe could clear every revision-1
threshold without listening to anything. Both revisions predate the first
trained probe; the git history is the evidence, and revision 1 is left intact
in it rather than rewritten.

This is the advisor's step 4. It exists so that step 5's result can be read
against criteria fixed in advance.

Scope: WavJEPA (frozen) on the Buckeye flapping set, one model, one dataset,
layer-wise, measured against an MFCC baseline. This is the gate before
extending to the baseline encoders.

**Thresholds below are judgment calls, not derived bounds.** If one turns out
badly calibrated, that is a finding to record — not a licence to move it after
seeing results. See *Deviations*.

---

## 1. Task and labels

**Task.** Binary classification of the underlying segment beneath a realized
flap: was this `[dx]` an underlying /t/ or /d/?

**Label source.** Buckeye's hand-corrected `.phones` tier, aligned to `.words`.
A site is a `[dx]` inside a word whose canonical and actual phone strings are
the same length, so position maps to an underlying segment unambiguously.
Implemented in `scripts/count_flap_sites.py`; inventory at
`results/corpus-verification/buckeye-flap-sites-20260821.{txt,tsv}`.

**Labels are underlying, not surface.** Surface is `[dx]` in every case by
construction — that is what makes the task a test of abstraction rather than of
acoustic classification.

**Exclusions, all fixed in advance:**

- Flaps carrying the `*` annotator mark — hedged labels.
- Words where canonical and actual strings differ in length (**21% of
  flap-bearing words**). These are the heavily reduced tokens (`about → b ah
  dx`); recovering them needs edit-distance alignment between tiers, out of
  scope here. Reported, never silently dropped.
- Sites that cannot be centred at the chosen context width (`Excerpt.clamped`)
  — **3 of 10,392 at 200 ms**.
- Sites containing no frame onset at all. **Selects none**: centring puts
  every target's start on a frame onset, and that onset counts (min 1 frame
  per site, `results/flapping/sites-20260924.txt`). Kept as a rule so a
  future change to context width or geometry cannot let an empty site through
  silently.

**Inventory: 10,392 sites, 7,863 /t/ and 2,529 /d/**, 75.7 / 24.3, across 40
speakers, every speaker carrying both classes (min 96, median 249).

## 1a. The lexical structure of this task — measured 2026-08-22

Registered here because it determines the split design, and because a reader
must be able to see that it was known before the run.

- **1,022 distinct word types.** Heavy head: top 10 cover **41.2%** of sites,
  top 50 **69.6%**, top 200 **85.8%**. 484 types (47.4%) occur exactly once,
  accounting for 4.7% of sites.
- The top 10 — `to, that, but, lot, it, get, what, out, at, got` — are **all
  /t/ function words**. This is the source of the class imbalance: /t/ is
  concentrated in a few very frequent items, /d/ is spread across the tail
  (553 /t/-only types vs 444 /d/-only types, but 3× the sites).
- **Only 25 types carry both classes** (220 sites, 2.12%), and they appear to
  be words with *two* flap sites (`dated`, `hated`, `dedicated`) rather than
  genuinely ambiguous types. The inventory has no position-within-word column,
  so this is an inference and is recorded as one.
- **A word-identity lookup table scores 0.9951** on the real task, using no
  audio at all. Keyed on (word, position) it would be effectively 1.0.
- **30.5% of sites are neutralised in the transcription** (measured
  2026-09-19, decision D5). 59 cells of (transcribed pronunciation, position)
  are attested as both /t/ and /d/ across different words: `got ~ god`,
  `what ~ would`, `write ~ ride`. That covers 3,173 sites (2,601 /t/, 572 /d/),
  flagged as `surface_ambiguous` in the inventory. **This is a claim about
  Buckeye's labels, not the audio.** The transcription hides the vowel-length
  difference before /t/ vs /d/, which is plausibly the cue that separates
  them. Recorded as a known population; not an evaluation subset in this
  revision.

**Two consequences, both binding:**

1. Any split in which word types recur between train and test measures
   vocabulary, not phonology.
2. **The Hewitt & Liang control task cannot function under such a split.**
   Control labels are type-determined by construction; real labels are
   type-determined at 99.5%. The two tasks become structurally identical, so
   selectivity approaches 0 for any probe able to identify words, whatever the
   encoder contains.

## 2. Representations

`capstone.encoders.extraction.encode()` → `(13, n_frames, 768)` per utterance.
Index 0 is the input to `layers[0]`; 1–12 are encoder layer outputs; index 12
is pre-final-LayerNorm. All 13 are probed.

**Fixed by the extraction findings, not open to tuning here:**

- **`context_ms = 200`, constant across every site.** Constancy is the binding
  constraint, not the value: the positional encoding is re-added from position
  0 each segment, so a frame's representation depends on where it sits within
  its segment (cosine distance 0.394 at layer 0 for a 10-frame shift, decaying
  to 0.021 by layer 12). Holding context constant makes that a constant that
  cannot correlate with the label. Vary it and the confound lands precisely in
  the early layers used as the control. 200 ms is the judgment call — wide
  enough for the conditioning vowels, and it clamps 3 sites against 35 at
  990 ms.
- **One `encode()` call per utterance; excerpts are never re-chunked.** The
  feature extractor applies a single RMS-derived gain over the whole waveform.
- **Frame→time conversion goes through `FrameGeometry`.** The map is a
  staircase, not a uniform rate; no frame rate is retyped anywhere.

**Pooling: mean over the site's frame onsets.** Registered as the default so
the pilot has one rule, *not* as a settled finding. Sites are 2–4 frames wide
(median 27.4 ms and 3 frames; 18.4% have ≤2), so mean / max / midpoint are
materially different vectors. Max and midpoint are the pre-declared secondary comparison;
anything beyond those three is exploratory and gets labelled so.

## 3. Probe and splits

**Linear (logistic regression) on frozen features.** Deliberately low capacity:
a linear probe keeps "the encoder represents this" the simplest reading of a
positive result. A stronger probe is a follow-up, not part of the gate.

**Class imbalance: `class_weight='balanced'`, all sites retained.**
Subsampling to the balanced ceiling of 5,058 would discard 5,334 sites to buy
the convenience of accuracy and balanced accuracy coinciding. Choosing the
metric correctly is cheaper.

### Splits: word-type-disjoint **and** speaker-disjoint

Both axes, jointly. Type-disjointness is the one that makes the task
phonological rather than lexical; speaker-disjointness is the standard guard
the advisor asked for, kept.

Construction, 5 speaker folds × 5 type folds:

1. Sort speakers; assign speaker *i* to fold *i mod 5*.
2. Rank word types by site count, highest first, **ties broken by the word's
   spelling in code-point order** (Python's default string sort, so a
   capitalised type sorts before lowercase ones). Assign the *i*-th ranked type
   to type fold *i mod 5*. Walking the ranked list spreads each fold's types
   across the whole frequency range, and needs **no random seed**: the
   assignment is a function of the inventory's contents alone. A
   frequency-blind random draw was measured and rejected. It swung diagonal
   test-fold /d/ counts between 71 and 253, against 87–113 for the stratified
   walk.
3. **Every cell of the 5 × 5 grid is a test set.** Cell (*a*, *b*) = sites in
   speaker fold *a* and type fold *b*. For each cell, **test** = the cell;
   **train** = sites in neither speaker fold *a* nor type fold *b*; one probe
   per cell, 25 in all. Speaker folds and type folds are both partitions, so
   every site is in exactly one cell and receives exactly one out-of-fold
   prediction. Measured: all 25 cells populated, no speaker or word type
   shared between any cell's test and train set.

**The discard is from training, not from scoring: ~32% of sites per probe.** Each probe trains on 6,255–7,072 sites (1,504–1,760 /d/). Sites that
share exactly one axis with the test cell are left out of that probe's
training, but every site is scored by its own cell. Test cells run 281–535
sites, 67–151 of them /d/, drawn from 27–48 distinct /d/ word types.

**Scoring is pooled, not averaged.** Out-of-fold predictions from all 25 cells
are collected and scored **once** over the pooled set: every site that survives
§1's exclusions, **10,389 sites, of which 2,529 are /d/**. That is the
inventory's 10,392 less the 3 clamped sites, all /t/. Per-cell scores are
reported as a spread of 25 numbers, not as the headline.

*(No minimum-sample threshold is registered. Revision 1 carried one — "≥500
minority sites per fold" — that had been chosen with the inventory already in
view and could only pass. The pooled counts above are stated as facts instead.)*

## 4. Controls

**Registered baseline: the word-identity lookup, 0.9951.** Not a probe — a
lookup table over the training types. Under the type-disjoint split it collapses
toward chance on held-out types, and that collapse is the check that the split
is doing its job. Report it.

The lookup, the control task and the speaker-leakage check all run on the same
25 cells as the real probe (§3) and are pooled the same way. A control scored
over a different set of sites than the probe it calibrates would not be a
control.

**Control task (Hewitt & Liang 2019, *Designing and Interpreting Probes with
Control Tasks*), demoted to a sanity check.** Each word type draws a random
/t/-or-/d/ label matching the real marginal; every token of that type inherits
it. The mapping is seeded and recorded.

Its role has changed from revision 1 and the change is deliberate: under a
type-disjoint split, memorization is *already* neutralized, because held-out
types were never seen with any label. A memorizing probe therefore scores near
chance on the control run by construction. **The control is now a leak
detector, not the primary calibration** — it should come out near chance, and
if it does not, word or speaker information is reaching the probe through a
route not accounted for here.

Hewitt & Liang give the reason directly: the control ceiling is "the fraction
of tokens in the evaluation set whose types occur in the training set (plus
chance accuracy on all other tokens)" (§2.3). Under this split that fraction
is zero, so the ceiling is chance. The control's gate is therefore a leak
bound (§5, criterion 2), not a selectivity floor.

**Selectivity = real balanced accuracy − control balanced accuracy**, per layer,
same probe and hyperparameters, **reported but not gated on**. With the control
at chance by construction it is close to (real − 0.50) and adds nothing to
criterion 1. It is in balanced-accuracy points rather than the paper's plain
accuracy because of the 75.7/24.3 imbalance. Hewitt & Liang state no absolute
selectivity threshold anywhere; they use it only to compare probes and layers.

**Registered baseline: MFCC.** The same linear probe on MFCC features. The
sites, excerpts, frames, pooling, class weighting and 25 cells are all the
same as for WavJEPA, and it is scored the same pooled way. It plays the role
of Hewitt & Liang's random-representation baseline (their Proj0, §4; Zhang &
Bowman 2018): features with no learned structure. That shows how much of the
probe's result is available from the acoustics alone. Criterion 1 is
measured against it. *(Feature configuration, meaning coefficients, deltas and
MFCC vs log-mel, is still to be fixed in this revision.)*

**Speaker-leakage check.** A probe trained on the same representations to
predict speaker identity, reported alongside. Diagnostic, not a gate.

## 5. Criteria — what counts as the pilot failing

All are scored on the pooled out-of-fold set (10,389 sites, 2,529 /d/). "Best
layer" is WavJEPA's layer with the highest balanced accuracy.

| # | Criterion | Threshold |
|---|---|---|
| 1 | Best layer beats the MFCC baseline | gap **≥ 0.05** balanced accuracy, **and** bootstrap 95% lower bound of the gap **> 0** |
| 2 | Control task shows no leak | control balanced accuracy within **0.50 ± 0.03** at **every** layer |
| 3 | Layer-wise curve is not flat | best − worst **≥ 0.05** balanced accuracy |
| 4 | Minority-class F1 (/d/), best layer | **≥ 0.45** |

Chance is 0.50 balanced accuracy. None of these is derived from Hewitt &
Liang, who state no thresholds; each is a judgment call with its reasoning
below. Finalized in revision 3, before any probe has been trained.
**Running before this revision is pushed voids the pre-registration.**

**Criterion 1 is comparative, not absolute.** An absolute bar (rev 2: ≥ 0.60)
was set for a task where word types recurred, and nothing calibrates it for
unseen words. Beating features with no learned structure is the claim the
project actually needs. The gap uses the same 0.05 as criterion 3, the
smallest difference this design treats as meaningful. The bootstrap resamples
**word types** (10,000 resamples, seed recorded) because word types are the
unit this design generalizes over, and it **re-selects the best layer inside
every resample**, so that picking the best of 13 layers cannot inflate the
gap. Sampling noise on a single balanced accuracy at this *n* is about ±0.011
(95%, independent sites); the word-type bootstrap will be wider, and that is
the conservative direction.

**Criterion 2 is a leak bound.** Under this split the control should sit at
chance (§4). ±0.03 is about three times the ±0.011 sampling band, wide enough
not to fire on noise. It applies at every layer, not only the best, because a
leak at any layer means information is reaching the probe by a route this
design does not account for.

**Criterion 4 is a guard, not a second performance bar.** It exists so an
aggregate cannot hide a probe that has quietly stopped predicting /d/. Trivial
floors at 24.3% /d/: always answering /d/ gives F1 **0.391**, a coin flip
**0.327**, and guessing /d/ at its base rate **0.243**. For reference, a probe
at balanced accuracy 0.60 with errors split evenly between the classes reaches
F1 0.42, so 0.45 is not lenient.

**If criteria are met:** proceed to the step-5 write-up, then the baseline
encoders (RQ3).

**If not, the pilot has failed, and failure is diagnostic rather than
terminal.** Read in this order:

- *Near chance at every layer* — suspect the label/alignment path, the 2–4 frame
  site width, or that the cue genuinely does not generalize across vocabulary,
  before concluding anything about the encoder.
- *High real accuracy with high control accuracy* — a leak. The split should
  have made this impossible; find the route before reading anything else.
- *Good but flat across layers* — a real result, and one that reframes RQ3
  rather than ending it.

None of these licences re-running with adjusted thresholds and reporting the
second run as the first.

## 6. Open questions for the advisor

Neither should be settled unilaterally:

- **Is leaving ~32% of sites out of each probe's training acceptable**, or is
  there a split design that buys type-disjointness more cheaply? Every site is
  still scored (§3). The cost is to training, not to the test set.
- **Do 10,389 pooled test sites (2,529 /d/) support a 13-layer comparison?**
  The layer-wise curve is the actual RQ3 deliverable, and this is the number
  that limits how finely layers can be distinguished.

## 7. Deviations

Any departure from the above gets appended here with a date and a reason, and
the original text stays as written. An amended pre-registration with a visible
amendment history is worth something; a silently edited one is worth nothing.

Revision 1 → 2 is **not** a deviation: nothing had been run, and the change was
driven by measuring the corpus, not by seeing a result. It is described at the
top of this file and preserved in git.

*(none yet)*
