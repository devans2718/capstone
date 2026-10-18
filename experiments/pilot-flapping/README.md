# Pilot: English flapping probe — pre-registration

**Revision 3, 2026-09-26. Nothing has been run.** Final as pushed. From here,
departures are appended to §7, never edited in above.

**Changes in revision 3**, each from measuring the corpus or the code, or from
the Challenge 2 feedback, none from a result:

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
7. **Four split conditions, not one (§3).** `random`, `speaker_disjoint`,
   `word_disjoint` and `joint`, the last being revision 2's design. The three
   new ones reuse the grid's own axes, so the drop across them is attributable
   to the constraint rather than to differently drawn folds. Only `joint`
   gates (§5). The others are the result the feedback asked for: how much of
   the score survives as the lexical and speaker routes are closed.
8. **Folds assigned over the 10,389 eligible sites (§1, §3).** The 3 clamped
   sites used to count in the type ranking while being excluded from scoring.
   Counting them moves 186 singleton types to a different fold, so they now
   leave before assignment. With change 18, this moves every grid figure in
   §3; the old ones are in
   `results/corpus-verification/buckeye-flap-lexical-structure-20260924.txt`.
9. **Secondary run at matched training size (§3).** The single-axis
   conditions retrained on 80% of their own training set, ≈64% of sites, to
   match `joint`.
10. **Controls under every condition; speaker-leakage check placed (§4).** The
    lookup and the control task run under all four. The speaker-ID probe runs
    under `random` and `word_disjoint` only, the two conditions where it is
    defined.
11. **MFCC configuration fixed (§4)**, with "same frames" corrected to what
    the code can do: MFCC frames are not the encoder's frames.
12. **Per-layer confidence intervals registered (§5).**
13. **Second encoder, exploratory (scope).** `facebook/wav2vec2-base`.
14. **Split construction moved to shared code.** `capstone.splits`, used by the
    probe run, the demo slice and the analysis script alike.
    `tests/test_splits.py` holds §3's figures.
15. **Every seed registered by value (§3, *Registered seeds*).** Revision 2
    said the control mapping and the bootstrap were "seeded and recorded"
    without saying with what, which leaves the seed free to be chosen after a
    result. The bootstrap seed bears directly on criterion 1.
16. **Criterion 2 restated as a randomization test (§4, §5).** The ±0.03
    bound assumed independent sites. Control labels are drawn per word type,
    and simulated with no leak at all, the bound fires 59–80% of the time on a
    single mapping (`scripts/simulate_control_spread.py`,
    `results/corpus-verification/buckeye-control-spread-20260926.txt`). The
    new test compares the real control score against the same statistic under
    10,000 redrawn mappings.
17. **Probe hyperparameters registered (§3).** Revision 2 said "the same probe
    and hyperparameters" without giving any, which left the regularization
    free to be chosen after a result. It bears on criterion 1, where 768
    dimensions are compared against 39.
18. **"Word type" defined, with contractions merged onto their host (§1a,
    §3).** Revision 2 never said what a word type was. It was the raw word
    form, so `do` could train while `don't` was tested. The type key is now
    the word form, lowercased, with 34 contracted forms merged onto the host
    that carries the flap (187 sites). 1,021 word forms become 991 keys. The
    key is not a lemma, and lemma overlap across the split is measured and
    registered as a limitation (§1a).
19. **Criterion 1 re-scored on unseen lemmas, reported and not gated (§5).**
    The subset of 7,586 sites is fixed now, as a committed file, so that the
    re-score is not a choice made after the run.

**Considered in revision 3 and not adopted:** ABX as a secondary measure
(not costed for flap sites; a follow-up), and publishing the token list,
split files, configs and results CSV (a question of what leaves the private
repository, not of design; decided separately).

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

**Second encoder, exploratory: `facebook/wav2vec2-base`**, chosen because
Pasad et al. 2022 (arXiv:2107.04734v3) analyze it layer by layer, so there is
a published curve to check against. It runs the same design and gates nothing, and every result
from it is labelled exploratory. It needs its own frame geometry, derived from
its conv stack and verified the way WavJEPA's was. WavJEPA's `FrameGeometry` is
not reused and no frame rate is typed in. Dropping it for time is not a
deviation.

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
  — **3 of 10,392 at 200 ms**. They are removed before folds are assigned
  (§3), not only before scoring. A site that is never scored does not decide
  where other sites go.
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
- **The figures above count raw word forms. The split's unit is the type
  key** (§3, measured 2026-09-26): the form, lowercased, with contractions
  merged onto their host. That gives 991 keys over the 10,389 eligible sites.
- **Known limitation: the split is disjoint by form, not by lemma**
  (measured 2026-09-26, `scripts/measure_lemma_overlap.py`,
  `results/corpus-verification/buckeye-lemma-overlap-20260926.txt`). Under
  `joint`, **25.1% of test sites (2,612) have their lemma in their probe's
  training set with their own label** (24.5% of /t/, 27.2% of /d/). That is
  the share a probe memorizing lemmas could exploit. Most of it comes from a
  few verb families: `get/got/getting/gotten` (619 sites), `out/outer` (258),
  `do/did/does` (226), `put/putting` (118), `start/started/starting` (90).
  Counting the lemma regardless of label gives 27.0%. Lemmas are from WordNet
  3.0, guided by Buckeye's POS tags, so this is an estimate. "Unseen words"
  in this design means unseen word forms. A lemma-disjoint split was not
  attempted: it would reshuffle every fold, and it is a follow-up. Criterion
  1 is also re-scored on the sites whose lemma is unseen (§5).

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

**Hyperparameters, fixed and never tuned.** scikit-learn (1.9.0 in the lock)
`LogisticRegression` with an L2 penalty, **C = 1.0**, solver `lbfgs`,
`max_iter = 1000`. Features are **standardized per probe**: a `StandardScaler`
is fit on that probe's training sites only, then applied to its test sites.
One setting for everything: every representation (WavJEPA, MFCC, wav2vec2),
layer, condition and run, the control task and the speaker-ID probe included.
A fit that does not converge is reported as such, and is not refit with other
settings.

**Class imbalance: `class_weight='balanced'`, all sites retained.**
Subsampling to the balanced ceiling of 5,058 would discard 5,334 sites to buy
the convenience of accuracy and balanced accuracy coinciding. Choosing the
metric correctly is cheaper.

### Splits: four conditions from one grid, one of them gated

Implemented once, in `capstone.splits`. The probe run, the demo slice and the
analysis script all take their folds from it, and `tests/test_splits.py`
reproduces every figure in this section.

**The gated condition is `joint`: word-type-disjoint and speaker-disjoint.**
Type-disjointness is the one that makes the task phonological rather than
lexical; speaker-disjointness is the standard guard the advisor asked for,
kept.

Construction, 5 speaker folds × 5 type folds, **over the 10,389 eligible
sites**. The §1 exclusions are applied first, so excluded sites are neither
ranked nor counted, and belong to no fold:

1. Sort speakers; assign speaker *i* to fold *i mod 5*.
2. Rank word types by site count, highest first, **ties broken by spelling in
   code-point order** (Python's default string sort). Assign the *i*-th ranked
   type to type fold *i mod 5*. Walking the ranked list spreads each fold's
   types across the whole frequency range, and needs **no random seed**: the
   assignment is a function of the inventory's contents alone. A
   frequency-blind random draw was measured and rejected. It swung diagonal
   test-fold /d/ counts between 71 and 253, against 87–113 for the stratified
   walk (both measured over all 10,392 sites and raw word forms, before
   changes 8 and 18).

   **A word type is a type key** (`capstone.splits.type_key`): the
   inventory's `word` column (the orthographic word form, with no
   pronunciation or POS in it), lowercased, with contracted forms merged onto
   the host word that carries the flap. The merge is an explicit table
   covering every contracted form in the inventory, not "cut at the
   apostrophe":
   - `'s`, `'ll`, `'ve` → the host (`city's` → `city`): the clitic contains no
     /t/ or /d/.
   - `it'd`, `that'd` → the host: labelled /t/, the host's segment.
   - `don't`, `didn't`, `doesn't`, `wouldn't`, `couldn't` → `do`, `did`,
     `does`, `would`, `could`: the flap is the host's /d/.
   - `i'd`, `they'd`, `you'd`, `he'd`, `there'd`, `haven't` keep their own
     key: the flap is in the clitic.

   34 forms (187 sites) are merged, and 1,021 word forms become 991 keys. The
   same key is the unit of the lookup table and the control task (§4).
3. **Every cell of the 5 × 5 grid is a test set.** Cell (*a*, *b*) = sites in
   speaker fold *a* and type fold *b*. For each cell, **test** = the cell;
   **train** = sites in neither speaker fold *a* nor type fold *b*; one probe
   per cell, 25 in all. Speaker folds and type folds are both partitions, so
   every site is in exactly one cell and receives exactly one out-of-fold
   prediction. Measured: all 25 cells populated, no speaker or word type
   shared between any cell's test and train set.

**The discard is from training, not from scoring: ~32% of sites per probe.**
Each probe trains on 6,257–7,100 sites (1,389–1,820 /d/). Sites that share
exactly one axis with the test cell are left out of that probe's training, but
every site is scored by its own cell. Test cells run 310–535 sites, 57–189 of
them /d/, drawn from 25–39 distinct /d/ type keys. The diagonal alone would
score 2,187 sites (543 /d/). Measured in
`results/corpus-verification/buckeye-flap-lexical-structure-20260926.txt`.

**/d/ is unevenly spread across type folds**, and this is known before the
run. Type fold 1 holds about twice the /d/ of type fold 3 (718 against 362
test sites). The walk balances site counts, not labels, and some merged keys
are large and entirely /d/: `do` (113 sites), `did` (104). Class weighting
handles the training imbalance, and scoring is pooled, so no cell's
proportion is the headline. The per-cell spread is read with it in mind.

**Scoring is pooled, not averaged.** Out-of-fold predictions from all 25 cells
are collected and scored **once** over the pooled set: every site that survives
§1's exclusions, **10,389 sites, of which 2,529 are /d/**. That is the
inventory's 10,392 less the 3 clamped sites, all /t/. Per-cell scores are
reported as a spread of 25 numbers, not as the headline.

### The three reported conditions

Each reuses one of the grid's two partitions, or neither:

| Condition | Test | Train | Probes | `fold` |
| --- | --- | --- | --- | --- |
| `random` | site fold *k* | the other four site folds | 5 | 0–4 |
| `speaker_disjoint` | speaker fold *a* (a grid row) | all other speaker folds | 5 | 0–4 |
| `word_disjoint` | type fold *b* (a grid column) | all other type folds | 5 | 0–4 |
| `joint` | cell (*a*, *b*) | neither *a* nor *b* | 25 | 0–24, as 5*a* + *b* |

Because the axes are shared, **`joint`'s train set for cell (*a*, *b*) is
exactly the intersection of `speaker_disjoint`'s train set for *a* and
`word_disjoint`'s for *b*.** A difference between conditions is then
attributable to the constraint, not to differently drawn folds.

**`random` is shuffled 5-fold, stratified by underlying label**
(scikit-learn `StratifiedKFold(shuffle=True)`), **seed 20260926**, over the
same 10,389 sites. It is the only condition with a seed. Tokens of one word
landing in both train and test is the point of this condition, not a defect.

Every condition scores every eligible site exactly once and is pooled the same
way as `joint`. They are **reported, not gated** (§5): the drop across them is
the result, and under `random` and `speaker_disjoint` the controls cannot
behave as they do under `joint` (§4).

### Secondary run: training matched to `joint`

The single-axis conditions train on ~80% of sites and `joint` on ~64%, so part
of any drop from `word_disjoint` to `joint` is only less training data. **In a
secondary run, each of `random`, `speaker_disjoint` and `word_disjoint` trains
on 80% of its own training set (≈64% of sites):** `round(0.8 × n)` sites per
class, drawn without replacement from a generator keyed on (20260926, fold).
Test sets are unchanged. The subsample is marked by its own column in the
predictions and results, not encoded in the condition name. Reported beside
the headline runs, never in place of them, and gated on nothing.

### Registered seeds

Every random draw in the run, in one place. The values are labels in date
form, chosen before any result and meaning nothing else. Each kind of draw
has its own seed, except that `random`'s folds and the subsample share
20260926. They still never share a random stream: the folds come from
scikit-learn's generator, and the subsample from a numpy generator keyed on
(20260926, fold). A run records every seed in its manifest.

| Seed | Drives | Where |
| --- | --- | --- |
| **20260926** | `random`'s fold assignment (`StratifiedKFold(shuffle=True, random_state=…)`) | §3 |
| **20260926** | the secondary run's subsample, drawn from a generator keyed on (seed, fold) | §3 |
| **20260927** | the control task's word-type → label mapping | §4 |
| **20260928** | the word-type bootstrap: criterion 1's gap, and every per-layer interval | §5 |
| **20260929** | criterion 2's 10,000 null control mappings, drawn by the same procedure as the real one | §4, §5 |

The speaker and type folds need no seed (§3 construction). The demo slice's
synthetic predictions use 20260919, which is not part of this design.

*(No minimum-sample threshold is registered. Revision 1 carried one — "≥500
minority sites per fold" — that had been chosen with the inventory already in
view and could only pass. The pooled counts above are stated as facts instead.)*

## 4. Controls

**Registered baseline: the word-identity lookup, 0.9951.** Not a probe — a
lookup table over the training types. Under the type-disjoint split it collapses
toward chance on held-out types, and that collapse is the check that the split
is doing its job. Report it.

**The lookup and the control task run under all four conditions (§3)**, each
on the same folds as the real probe in that condition, and pooled the same
way. A control scored over a different set of sites than the probe it
calibrates would not be a control. Expected shape: the lookup near 0.99 under
`random` and `speaker_disjoint`, where word types recur between train and
test, and collapsing under `word_disjoint` and `joint`, where they do not. That
collapse is the plainest single exhibit of what the type-disjoint split
removes.

**Control task (Hewitt & Liang 2019, *Designing and Interpreting Probes with
Control Tasks*), demoted to a sanity check.** Each word type draws a random
/t/-or-/d/ label, and every token of that type inherits it.

**The mapping procedure**, written once because criterion 2's randomization
test (§5) redraws it, and a redraw that differs from the real draw would test
against the wrong distribution:

1. Take the type keys (§3) of the 10,389 eligible sites, 991 of them, in
   code-point order.
2. With a numpy `default_rng(seed)`, draw one uniform per type in that order.
   A type is /d/ if its draw is below **2,529 / 10,389**, the eligible /d/
   share, and /t/ otherwise.

Each type is drawn independently, so no type's label depends on any other
type's, and in particular not on the training types'. The randomization test
relies on that. The real mapping is one draw with **seed 20260927**. The
10,000 null mappings are 10,000 consecutive draws from a single generator
with **seed 20260929** (§3, *Registered seeds*).

Its role has changed from revision 1 and the change is deliberate: under a
type-disjoint split, memorization is *already* neutralized, because held-out
types were never seen with any label. A memorizing probe therefore scores
chance on the control run in expectation. Any single mapping can land well off
0.50 (sd ≈ 0.056, change 16). **The control is now a leak detector, not the
primary calibration**: it should come out where a random mapping puts it, and
if it does not, word or speaker information is reaching the probe through a
route not accounted for here.

Hewitt & Liang give the reason directly: the control ceiling is "the fraction
of tokens in the evaluation set whose types occur in the training set (plus
chance accuracy on all other tokens)" (§2.3). Under this split that fraction
is zero, so the ceiling is chance. The control's gate is therefore a leak
bound (§5, criterion 2), not a selectivity floor.

That argument holds for the type-disjoint conditions only. **Under `random`
and `speaker_disjoint` the control is expected to sit well above chance**:
word types recur, the control's labels are a function of the type, and a
probe that identifies words memorizes them. There the control is reported for
the comparison, and is not a leak test. This is why criterion 2 is scored on
`joint` alone (§5).

**Selectivity = real balanced accuracy − control balanced accuracy**, per layer,
same probe and hyperparameters, **reported but not gated on**. The control
sits at chance in expectation, not in any one draw. A single mapping's score
can land well off 0.50 with no leak at all (sd ≈ 0.056; change 16), so
selectivity is (real − control) as measured, not (real − 0.50), and it is read
with that spread in mind. It is in balanced-accuracy points rather than the paper's plain
accuracy because of the 75.7/24.3 imbalance. Hewitt & Liang state no absolute
selectivity threshold anywhere; they use it only to compare probes and layers.

**Registered baseline: MFCC.** The same linear probe on MFCC features. The
sites, excerpts, pooling rule, class weighting and folds are all the same as
for WavJEPA, in every condition, and it is scored the same pooled way. It
plays the role of Hewitt & Liang's random-representation baseline (their
Proj0, §4; Zhang & Bowman 2018): features with no learned structure. That shows
how much of the probe's result is available from the acoustics alone.
Criterion 1 is measured against it.

The **frames are not the same, and cannot be**: MFCC frames sit on a uniform
10 ms grid, and the encoder's sit on its own staircase (§2). Nor is the rule
for which frames belong to a site the same (see *Pooling* below). What is the
same is the reduction: a mean over the frames that describe the site.
Configuration:

- **13 cepstral coefficients including c0, plus first and second deltas: 39
  dimensions.** Computed with librosa (0.11.0 in the lock) on the site's 16 kHz
  §2 excerpt, the same samples the encoder receives.
- **25 ms Hann window (400 samples), 10 ms hop (160), 40 mel filters,
  uncentred frames** (`center=False`), so frame *t* spans 10*t* to 10*t* + 25 ms
  of the excerpt and no padding is invented.
- **Deltas over the whole excerpt, before pooling**, with librosa's default
  9-frame regression window, so a site's deltas see the context the excerpt
  provides.
- **Pooling: mean over the frames whose centre lies in the site's interval.**
  WavJEPA's frames are selected by onset (§2). MFCC frames are selected by
  centre, because an MFCC frame is a 25 ms Hann window weighted toward its
  middle. Its onset sits 12.5 ms before the audio it mostly describes, and
  with sites a median 27 ms long, an onset rule would skew each site's frames
  toward the audio after it. Every eligible excerpt starts exactly 200 ms
  before its flap, so the first frame centre falls 2.5 ms into every site. The shortest eligible site is
  6.25 ms, so every site has at least one frame, by construction. The run
  asserts this, and stops rather than excluding a site if it ever fails.
- Max and midpoint pooling run as the same pre-declared secondary as for
  WavJEPA (§2).

**Speaker-leakage check.** A probe trained on the same representations to
predict speaker identity, reported alongside. Diagnostic, not a gate. **It runs
under `random` and `word_disjoint` only.** It needs every test speaker to occur
in training, and those are the two conditions where every speaker does. Under
`speaker_disjoint` and `joint` a speaker-ID probe is asked to name speakers it
never saw, so the check is undefined there, not merely weak.

## 5. Criteria — what counts as the pilot failing

**All four are scored on the `joint` condition only**, on its pooled
out-of-fold set (10,389 sites, 2,529 /d/), in the headline run. "Best layer" is
WavJEPA's layer with the highest balanced accuracy under `joint`. `random`,
`speaker_disjoint` and `word_disjoint`, the secondary run, and the exploratory
encoder are reported, not gated.

**Why only `joint`.** Under `random` and `speaker_disjoint`, word types recur
between train and test. The control task can memorize per-type labels there,
so it cannot sit at chance, and criterion 2 would fire on a design that is
working correctly (§4). Real labels are 99.5% type-determined (§1a), so
criterion 1 there would mostly measure word identification. The drop from
`random` to `joint` is the result, not a gate.

| # | Criterion | Threshold |
|---|---|---|
| 1 | Best layer beats the MFCC baseline | gap **≥ 0.05** balanced accuracy, **and** bootstrap 95% lower bound of the gap **> 0** |
| 2 | Control task shows no leak | the control's highest balanced accuracy over the 13 layers does **not exceed the 95th percentile** of the same maximum under 10,000 redrawn mappings |
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
**word types** (10,000 resamples, **seed 20260928**) because word types are the
unit this design generalizes over, and it **re-selects the best layer inside
every resample**, so that picking the best of 13 layers cannot inflate the
gap. Sampling noise on a single balanced accuracy at this *n* is about ±0.011
(95%, independent sites); the word-type bootstrap will be wider, and that is
the conservative direction.

**Criterion 2 is a randomization test for a leak.** With no leak, the control
probe's predictions on held-out types are independent of those types' control
labels. So the distribution of its score with no leak is known exactly: keep
the predictions fixed and redraw the labels.

- **Statistic:** the control's pooled balanced accuracy under `joint`, at each
  of WavJEPA's 13 layers. Take the maximum over layers.
- **Null:** for each of the 10,000 null mappings (§4, seed 20260929), score
  the same fixed predictions at all 13 layers against that mapping, and keep
  the maximum. The probes are not retrained.
- **Fires** if the real maximum exceeds the 95th percentile of the 10,000
  null maxima.

The maximum over layers is used rather than a per-layer Bonferroni
correction. It holds the chance of a false alarm across all 13 layers at 5%
exactly, where Bonferroni would treat 13 strongly correlated layers as
independent and be needlessly conservative. It is one-sided: a leak raises
the control's score, and a score below the null is not a leak. It still covers
every layer, because a leak at any layer means information is reaching the
probe by a route this design does not account for.

**Why the ±0.03 of earlier drafts was dropped.** It was justified as three
times a ±0.011 band that assumes independent sites. Control labels are shared
by every token of a type, and the head is heavy, so a leak-free probe's score
on a single mapping has sd ≈ 0.056. ±0.03 would fire 59–80% of the time on a
design that is working (change 16).

**Criterion 4 is a guard, not a second performance bar.** It exists so an
aggregate cannot hide a probe that has quietly stopped predicting /d/. Trivial
floors at 24.3% /d/: always answering /d/ gives F1 **0.391**, a coin flip
**0.327**, and guessing /d/ at its base rate **0.243**. For reference, a probe
at balanced accuracy 0.60 with errors split evenly between the classes reaches
F1 0.42, so 0.45 is not lenient.

**Per-layer confidence intervals, reported and not gated.** Every balanced
accuracy on a layer-wise curve, in every condition and for every
representation, carries a 95% interval from the same word-type bootstrap as
criterion 1 (10,000 resamples of word types, seed 20260928), computed layer by
layer. The curves are the RQ3 deliverable, and a curve without intervals
invites reading differences between adjacent layers that the data cannot
support.

**Criterion 1 re-scored on unseen lemmas, reported and not gated.** A quarter
of `joint`'s test sites have their lemma in their probe's training set (§1a).
So criterion 1's statistic is also computed on the **7,586 sites (1,735 /d/)
whose lemma occurs nowhere in their probe's training set**. The subset is
fixed before the run, as the rows of
`results/flapping/lemma-overlap-joint-20260926.tsv` with
`lemma_in_train = false`. It uses the same predictions, with no new probes:
the best-layer gap to MFCC, the same word-type bootstrap (seed 20260928),
and the best layer re-selected in every resample. If the gap holds on this
subset, lemma memorization is not what produces it. If it shrinks, that
share of the result is lemma-level, and is reported as such. Neither outcome
changes whether criterion 1 passes.

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
  still scored (§3). The cost is to training, not to the test set. The §3
  secondary run measures part of this: it shows how much the single-axis
  conditions lose when trained at `joint`'s size.
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

### 2026-09-26 — Clarifications fixed before the first run

Not deviations: revision 3 left these unspecified, and they are fixed here
before any feature has been extracted or any probe trained. Nothing above is
changed.

- **Midpoint pooling (§2, §4).** For WavJEPA, the site's frame whose onset is
  the last at or before the site's temporal midpoint. The target starts on an
  onset, so this frame is always one of the site's own frames. For MFCC, the
  site's frame whose centre is nearest the midpoint, with a tie going to the
  earlier frame.
- **The lookup on a type it never saw in training (§4).** It predicts the
  training set's majority label, which is /t/ in every fold. A held-out type
  therefore scores at chance in balanced accuracy, and that collapse is what
  §4 expects under `word_disjoint` and `joint`.
- **Speaker-ID metric (§4).** Balanced accuracy over the 40 speakers (chance
  0.025), with the §3 probe and settings unchanged, multinomial over the 40
  classes.
- **Criterion 1's bootstrap is paired (§5).** Each of the 10,000 resamples of
  word types scores WavJEPA's 13 layers and the MFCC baseline on the same
  resampled types. The gap's interval is then the interval of a difference
  measured on one sample, not of two independent intervals.

### 2026-09-26 — A prediction recorded before extraction

Stated by the author before any feature exists, and recorded so it cannot be
revised after: WavJEPA's information "sits higher than wav2vec2-base's", per
the argument from pretraining objectives in Pasad, Shi & Livescu 2023
(*Comparative layer-wise analysis of self-supervised speech models*). It is a
prediction, not a criterion. It gates nothing, and wav2vec2-base remains
exploratory.

### 2026-09-26 — Written after the run: the lookup's rule for tied types

**This entry was written after the run, not before it.** The run's code made
a choice that nothing above registers. It is recorded here so the file says
what was done, and does not appear to have been settled in advance.

The first clarification above says what the lookup predicts for a type it
never saw in training. It does not say what it predicts for a type seen
equally often as /t/ and /d/ in training. The code (`lookup_predict` in
`capstone.probes.controls`) gives such a type the same fallback: the training
set's majority label, /t/.

Measured effect: no test site under `word_disjoint` or `joint` belongs to a
tied type, because held-out types never occur in training. So the lookup's
0.5000 under both is untouched. Under `random` the rule reaches 6 of 10,389
test sites (8 in the subsampled run), and under `speaker_disjoint` 18 (20
subsampled). The lookup gates nothing (§5), and no criterion reads it.

### 2026-09-26 — Registered before listening: an audit of the sites by ear

**Written after the run, before any of the audit below has been listened to.**
It adds a measurement; it changes nothing above, and no criterion reads it.

**Why.** §5's first reading of a failure near chance is the alignment path:
the sites may not be what the inventory says. The `random` layer curve argues
against a broken pipeline, but only indirectly. This checks the sites
directly, by ear. It measures the inventory, not the probe.

**Sample.** 40 /t/ and 40 /d/ sites, drawn uniformly without replacement from
the 10,389 eligible sites (clamped sites excluded, as in §1), with numpy's
`default_rng(20260930)` (`capstone.seeds.AUDIT`), in a shuffled review order.
The draw sees the inventory only, never a prediction. Code:
`capstone.audit.draw_sample`; the sample is committed as
`results/flapping/audit-sample-20260926.tsv` in the same commit as this entry,
and a test checks the file equals the seeded draw.

**Questions**, each answered yes, no, or can't tell:

- `flap` — is there a flap: a quick tap, not a full stop, a deletion or a
  glottal stop? This checks the transcribers' `[dx]`.
- `word` — is the word the one the transcript shows? The /t/ or /d/ label
  follows from the word, and cannot itself be heard under a flap.
- `clip` — is the flap roughly centred in the 200 ms-context clip, and not
  cut off?

**Blind.** The audit view shows the clip, the word and its neighbours, and
nothing from the probe: no guess, no score, no condition.

**Listener.** One listener, the author, not a trained phonetician, on
headphones. The rates bound what one untrained ear agrees with, and are
reported as that.

**Counting.** The log is append-only; the last verdict for a site and question
is the verdict. Only lines recorded in the audit view, for run
`pilot-20260926`, on sampled sites are counted. Verdicts recorded while
browsing are kept but never counted.

**Reported.** Per question and per class: yes / (yes + no) with a Wilson 95%
interval, with the can't-tell count reported beside it and never folded in,
and how many of the 40 were reviewed. The sample is balanced and the
inventory is not (75.7 / 24.3), so no pooled rate is reported. Output:
`results/flapping/audit-<date>.txt` from `scripts/summarize_audit.py`.

### 2026-10-17 — Not a deviation: the span grid is registered separately

**Nothing above is changed.** §5's first reading of a failure near chance
names "the 2–4 frame site width" among its suspects. The follow-up that takes
it up, asking where the cue sits and whether the neighbouring speech carries
it, is a new study with its own sites, spans, inputs, baselines and predictions.
It is registered in `experiments/span-grid/README.md`, not here. It reuses
this file's inventory, probe, `joint` folds and seeds, and its WavJEPA ɾ span
on these 10,389 sites must reproduce this pilot's `joint` mean-pooled curve
before any of its results are read. This pilot's result stands as recorded.
