# Pilot: English flapping probe — pre-registration

**Revision 3 — in progress, not final.** Nothing has been run. Updated
2026-09-24 in §3, §4 and §6. The word-type ranking now has an explicit
tie-break, and scoring moves from the 5 diagonal cells of the speaker × type
grid to all 25. Both changes came from measuring the inventory
(`scripts/analyze_flap_lexical_structure.py` §5), not from any result. Still
pending before this revision is final and pushed: the provisional thresholds in
§5, and whatever is adopted from the Challenge 2 research feedback.

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
layer-wise. This is the gate before extending to the baseline encoders.

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
- The 4 sites containing no frame onset at all.

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
the pilot has one rule, *not* as a settled finding. Sites are 2–3 frames wide
(median 27.4 ms, 41% have ≤2 onsets), so mean / max / midpoint are materially
different vectors. Max and midpoint are the pre-declared secondary comparison;
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
   across the whole frequency range, and needs **no random seed**.
   *(Rev 3: the tie-break is registered because 985 of 1,022 types share a
   count with another type. Rev 2 left ties to `Counter.most_common()`, which
   orders them by row order in the inventory file. Its claim that the split was
   "reproducible from the inventory alone" held only while that order held.
   The two rules differ for 777 types, 2,848 sites.)*
   A frequency-blind random draw was measured at rev 2 and rejected: it swung
   test-fold /d/ counts between 71 and 253 across folds, against 57–134 for the
   stratified walk under row-order ties. Under the registered tie-break the
   stratified range is 87–113.
3. **Every cell of the 5 × 5 grid is a test set.** Cell (*a*, *b*) = sites in
   speaker fold *a* and type fold *b*. For each cell, **test** = the cell;
   **train** = sites in neither speaker fold *a* nor type fold *b*; one probe
   per cell, 25 in all. Speaker folds and type folds are both partitions, so
   every site is in exactly one cell and receives exactly one out-of-fold
   prediction. Cell (*k*, *k*)'s train set is exactly rev 2's fold *k*, so no
   probe trains on less than under the diagonal design.
   *(Rev 2 tested only the 5 diagonal cells, which left ~80% of sites unscored.
   Measured 2026-09-24: all 25 cells populated, no speaker or word type shared
   between any cell's test and train set.)*

**The discard is now from training, not from scoring: ~32% of sites per
probe.** Each probe trains on 6,255–7,072 sites (1,504–1,760 /d/). Sites that
share exactly one axis with the test cell are left out of that probe's
training, but every site is scored by its own cell. Test cells run 281–535
sites, 67–151 of them /d/, drawn from 27–48 distinct /d/ word types.

**Scoring is pooled, not averaged.** Out-of-fold predictions from all 25 cells
are collected and scored **once** over the pooled set: **every site that
survives §1's exclusions, 10,392 before exclusions, of which 2,529 are /d/**.
After exclusions that is **10,385 sites (2,528 /d/)** if the zero-onset
exclusion counts in the recording's timebase (4 sites), or **10,349 (2,521
/d/)** if it counts in the excerpt-local timebase (39 sites). Which one applies
is decision D1 in `docs/challenges/challenge-2-slice-decisions.md`, and it must
be settled before the run. Per-cell scores are reported as a spread of 25
numbers, not as the headline.

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

**Selectivity = real balanced accuracy − control balanced accuracy**, per layer,
same probe and hyperparameters. Note this is in balanced-accuracy points, a
deviation from the paper's plain accuracy, taken because of the 75.7/24.3
imbalance — their reported magnitudes will not transfer one-for-one.

**Speaker-leakage check.** A probe trained on the same representations to
predict speaker identity, reported alongside. Diagnostic, not a gate.

## 5. Criteria — what counts as the pilot failing

All refer to the **best layer**, scored on the pooled out-of-fold set.

| # | Criterion | Threshold |
|---|---|---|
| 1 | Balanced accuracy | **≥ 0.60** *(provisional)* |
| 2 | Selectivity (real − control) | **≥ 0.10** *(provisional)* |
| 3 | Layer-wise curve is not flat | best − worst **≥ 0.05** balanced accuracy |
| 4 | Minority-class F1 (/d/) | **≥ 0.45** *(provisional)* |

Chance is 0.50 balanced accuracy.

⚠ **The three marked *provisional* must be finalized before the first probe is
trained, and this file updated, in one further revision.** They were set for a
task where word types recurred; generalizing a phonological cue to *unseen*
words is materially harder, and revision 1's values are no longer calibrated to
what is being asked. The finalizing input is Hewitt & Liang's reported
selectivity magnitudes. **Running before that revision voids the
pre-registration** — that is the whole point of the file.

Criterion 1 is set above chance by a margin because the deliverable is a
*comparison across layers*; a signal too weak to separate layers cannot support
RQ3 whatever its significance. Criterion 4 exists so an aggregate cannot hide a
probe that has quietly stopped predicting /d/.

**If criteria are met:** proceed to the step-5 write-up, then the baseline
encoders (RQ3).

**If not, the pilot has failed, and failure is diagnostic rather than
terminal.** Read in this order:

- *Near chance at every layer* — suspect the label/alignment path, the 2–3 frame
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

- **Is the ~32% per-fold discard acceptable**, or is there a split design that
  buys type-disjointness more cheaply?
  *(Rev 3: partly answered by the full grid. No site goes unscored any more,
  but each probe still trains without ~32% of the data. Whether that training
  cost is acceptable remains the advisor's call.)*
- **Do the pooled test sites support a 13-layer comparison?** The layer-wise
  curve is the actual RQ3 deliverable, and this is the number that limits how
  finely layers can be distinguished. *(Rev 2 put the pool at ≈1,990, and the
  measured diagonal was 2,104. Under rev 3 it is ~10,350–10,385, with
  ~2,520 /d/.)*

## 7. Deviations

Any departure from the above gets appended here with a date and a reason, and
the original text stays as written. An amended pre-registration with a visible
amendment history is worth something; a silently edited one is worth nothing.

Revision 1 → 2 is **not** a deviation: nothing had been run, and the change was
driven by measuring the corpus, not by seeing a result. It is described at the
top of this file and preserved in git.

*(none yet)*
