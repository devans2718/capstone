# Challenge 2 Enhancement — brief for the repo-side agent

**Written 2026-09-25 against the repo at `~/code/capstone`, checked the same evening.** Deadline: **Mon 2026-09-28, 11:59 PM**. A written report follows on Wed 09-30 and reuses everything built here.

Read `experiments/pilot-flapping/README.md` first. This work finishes that pre-registration. This brief records decisions made on 2026-09-25, names what is still open, and sets the order of work. Everything you need is in this file and the repo.

---

## What is being graded

The assignment is to **improve the existing Co-Design app, not restart it**, and to add a small research and evaluation component. It is submitted as **6–8 slides + a deployed app or a reproducible local demo + the updated GitHub repo**. The parts that land in the repo:

- **Addressing 3–5 suggestions from the feedback on the previous submission**, each shown as *Feedback → What Changed → Evidence*. Not every suggestion has to be implemented.
- **One research question**, tied to the app.
- **A baseline and one small experiment with actual results.** Results must be real; nothing is to be fabricated.
- **5–10 test cases**, including at least one failure case. The slice already has nine.
- **Result + Evidence + Provenance + Uncertainty** shown in the app.
- **A local demo is explicitly allowed** when deployment is blocked by data restrictions. Buckeye is restricted, so state that limitation and give run instructions.
- **README must cover:** what the app does · the Human–AI workflow · data used · AI capability · baseline and evaluation · how to run · limitations · next steps · references to datasets, GitHub projects and papers.
- **No confidential or restricted data in the repo.**

**What the feedback asks for, which this brief implements most of:**
- Finish flapping end to end on Buckeye before adding Russian or RAG
- At least two models plus an MFCC/log-mel baseline: same probe, same token list
- Random, speaker-disjoint and word-disjoint results, layer by layer, plus the combined condition if the sample size permits
- ABX or another training-free phonetic check (undecided, see §3)
- Publish the token-selection script, split files, configs and results CSV (undecided, see §3)

---

## 0. State — checked 2026-09-25, not assumed

| Fact | Consequence |
| --- | --- |
| Branch `challenge-2-slice` @ `be23fff`, clean, **11 commits ahead of `origin`** | Nothing from 2026-09-24 is pushed yet, revision 3 included |
| Pre-registration marked *Revision 3 — in progress*; §5 finalized in `980f84c` | The criteria are done; §3 and §4 are what this brief extends |
| Only the **joint** condition has a construction anywhere, in `scripts/analyze_flap_lexical_structure.py` §5 (lines ~111–121) | `ordinary` and `word_disjoint` are **names only**. `predictions.CONDITIONS` labels synthetic rows, and `synthesize()` draws their folds at random (`predictions.py:297`) |
| The slice has **no speaker-disjoint condition** | The feedback requires one |
| Docker is back as of 2026-09-24, and the image's tests pass with the GPU attached | Extraction can run in the container |

---

## 1. Invariants

1. ⚠ **Nothing runs before revision 3 is on `origin`.** No probe, no MFCC features, no embeddings scored against labels. Verify with `git log origin/challenge-2-slice` that the revision-3 commit is there before the first run. Running first voids the pre-registration, which is the only thing that file is for.
2. ⚠ **Registered text is appended to, never edited in place**, once it has been pushed. Revision 3 isn't pushed, so its body can still change. §7 *Deviations* is append-only from the moment of the push.
3. ⚠ **The inventory stays 10,392 sites / 7,863 /t/ / 2,529 /d/**, pooled scoring stays 10,389 sites / 2,529 /d/, and `count_t_realizations.py` still passes.
4. ⚠ **The existing joint-grid figures must reproduce exactly** from the new shared code: test cells 281–535 sites, /d/ 67–151; train 6,255–7,072; diagonal 2,137 sites / 513 /d/; full grid 10,392 / 2,529; zero speakers or types shared between any cell's test and train.
5. ⚠ **No Buckeye audio in the repo.** Embeddings and predictions go under `/data`, not `results/`.
6. **Don't decide anything in §3 on your own.** Draft options and their costs if that helps, but register nothing from that list without the user's sign-off.

---

## 2. Decided 2026-09-25 — register these in revision 3

### 2.1 Four conditions, all from the grid's own axes

The 5 × 5 grid already defines two partitions in §3: **speaker folds** (sorted speakers, `i mod 5`) and **type folds** (frequency-ranked walk, alphabetical tie-break). The three new conditions reuse them instead of building new ones:

| Condition | Test | Train | Probes | `fold` values |
| --- | --- | --- | --- | --- |
| `random` | site fold *k* | the other four site folds | 5 | 0–4 |
| `speaker_disjoint` | speaker fold *a* (a grid row) | all other speaker folds | 5 | 0–4 |
| `word_disjoint` | type fold *b* (a grid column) | all other type folds | 5 | 0–4 |
| `joint` | cell (*a*, *b*) | neither *a* nor *b* | 25 | 0–24 (`5a + b`) |

**Why the shared axes matter, so the prose can say it:** joint's train set for cell (*a*, *b*) is exactly the intersection of `speaker_disjoint`'s train set for *a* and `word_disjoint`'s for *b*. Differences between conditions are then attributable to the constraint, not to differently drawn folds.

**`random` is seeded stratified 5-fold by underlying label** (`StratifiedKFold(shuffle=True)`), seed recorded in the pre-registration and the manifest. It is the only condition with a seed. Tokens of one word landing in both train and test is the point of this condition, not a defect.

Every condition scores every eligible site exactly once and is pooled the same way as joint.

### 2.2 Secondary run: training subsampled to 64%

The single-axis conditions train on ~80% of sites, joint on ~64%, so part of any drop from `word_disjoint` to `joint` is just less training data. **Register a secondary run in which each single-axis condition's training set is subsampled to 80% of itself (≈ 64% of sites), stratified by label and seeded.** The headline runs stay as they are. `random` is included for completeness.

The subsample must be distinguishable in `predictions.parquet` by a column, not by parsing the condition name.

### 2.3 The §5 criteria gate `joint` only

State it explicitly in §5. `random`, `speaker_disjoint` and `word_disjoint` are **reported conditions, not gates**. The drop across them is the result, and it's the "drop from random to disjoint splits" plot the feedback asks for. Reason to register alongside it:
- Under `random` and `speaker_disjoint`, word types recur between train and test. The control task can memorize per-type labels, so it **cannot** sit at chance, and criterion 2 would fire on a correctly working design.
- Real labels are 99.5% type-determined, so criterion 1 there would mostly measure word identification.

This also follows the project's framing as a controlled representation-analysis study: *the comparison across control conditions is the result.*

**Report the word-identity lookup and the control task under all four conditions.** Expected shape: lookup ~0.99 under `random` and `speaker_disjoint`, collapsing under `word_disjoint` and `joint`. That collapse is the cleanest single exhibit of what the controls do.

### 2.4 Names

`random / speaker_disjoint / word_disjoint / joint` everywhere: `predictions.CONDITIONS`, the app, the tests, `make_synthetic_predictions.py`, the manifest, the pre-registration. **Currently 48 occurrences across 7 files** of `ordinary`, `word_disjoint`, `combined` and `N_FOLDS`. `combined` and `ordinary` go; this matches the feedback's vocabulary and removes a translation step from the report. Slice rows are synthetic, so nothing real is lost. Past commit messages and dated decision records keep their old wording.

### 2.5 Shared split code in `src/`

Move the construction into a module (e.g. `capstone.splits`) that the probe run, the slice, and the analysis script all use:
- speaker fold, type fold and cell per site
- `(train_idx, test_idx)` iteration for each condition, with the 64% subsample as an option
- **A test that reproduces invariant 4's figures** from the shared code, plus: every site in exactly one test fold per condition, and zero leakage on the constrained axis per condition

`synthesize()` then takes its folds from this module (the `TODO(pre-reg rev 3)` at `predictions.py:291`). Point `analyze_flap_lexical_structure.py` at it too, or at minimum have the test assert the two agree.

---

## 3. Still open — do not register without the user's sign-off

| Item | Default on the table | Note |
| --- | --- | --- |
| **MFCC feature configuration** (§4) | — | MFCC vs log-mel, coefficient count, deltas. The criterion-1 baseline, so it must be fixed before the push |
| **ABX** as a secondary measure | — | Depends on a study of `zerospeech/libri-light-abx2`, in progress. A cost estimate for flap sites (whether it runs on our pooled site vectors or needs frame sequences) would help decide |
| **Second encoder: confirmatory or exploratory** | — | `facebook/wav2vec2-base`, chosen because Pasad et al. 2022 analyzes it layer by layer, so there is a published curve to check against |
| **Per-layer CIs** for the curves | same word-type bootstrap as criterion 1, per layer, per condition | Criterion 1's gap CI is registered; the per-layer CIs aren't |
| **Publishing** the token list, split files, configs, results CSV | — | Touches whether the repo stays private (it is private because of the Buckeye download script) |

### ⚠ Two things found while writing this brief — raise with the user, don't resolve

- **Are fold assignments computed over all 10,392 sites or the 10,389 after the clamp exclusion?** The type ranking is by site count, and 985 of 1,022 types share a count with another type, so dropping 3 sites can move tie-breaks and folds. §3 says "rank word types by site count" without saying which inventory. **Measure whether it changes any fold**; if it does, the choice is registered text.
- **The §4 speaker-leakage check can't run under speaker-disjoint splits.** A speaker-ID probe can't predict speakers it never saw, so under `joint` it's undefined. With `random` now registered, that condition is the natural home for it. It needs saying in §4.

---

## 4. Order of work

**Saturday 09-26, morning — revision 3 (§2, plus whatever gets settled from §3):**
1. `capstone.splits` + its test (§2.5); invariant 4 reproduces
2. Renames (§2.4); the slice's tests still pass
3. Pre-registration: §3 gets the four conditions and the 64% secondary; §4 gets the lookup/control-under-all-four line and the speaker-leakage placement; §5 gets joint-only gating; the change list at the top gets entries for all of it
4. Update the top-of-file status: the pending list shrinks to nothing, and "in progress" goes
5. **Push. Confirm on `origin`.** Report the commit hash

**Saturday afternoon through Sunday 09-27 — the run.** WavJEPA + MFCC first, because it's the pilot and the gate. wav2vec2-base only after that.
- Embeddings for all sites, 13 layers, mean-pooled per §2 (max and midpoint as the registered secondary). ~400 MB per model pooled
- MFCC/log-mel for the same sites, same excerpts, same pooling
- Probe harness → **one results CSV**: encoder · layer · condition · subsample flag · balanced accuracy · /d/ F1 · control balanced accuracy · lookup · n. Plus `predictions.parquet` with real rows (`synthetic = false`)
- Layer-wise plots per condition, with CIs if §3 settles them
- Criteria evaluated on `joint` and written up plainly, **pass or fail**. A fail is diagnostic per §5, not a reason to adjust anything
- ⚠ **wav2vec2-base needs its own frame geometry**, derived from its conv stack and verified the way WavJEPA's was. **Do not reuse `FrameGeometry` or retype a frame rate.** If that can't be done cleanly by Sunday, the fallback is WavJEPA + MFCC only, which still meets the assignment
- ABX on the same embeddings only if it is adopted and time allows

**Sunday evening — app and README:** swap the slice's synthetic predictions for the real ones and let the app show all four conditions. The `SYNTHETIC` banner has to *go* when the rows are real and stay when they aren't, driven by the data rather than hardcoded. Then the README items from *What is being graded*. The slides are out of scope for this brief.

---

## 5. Definition of done — revision 3

- [ ] `capstone.splits` exists; its test reproduces invariant 4 and checks per-condition coverage and leakage
- [ ] Renames done; slice tests pass; `count_t_realizations.py` passes
- [ ] Pre-registration registers four conditions, the 64% secondary, joint-only gating, lookup and control under all four, and the speaker-leakage placement
- [ ] Every §3 item is either registered with the user's sign-off or explicitly listed as not adopted
- [ ] The two open questions in §3 answered or registered
- [ ] Revision-3 commit **on `origin`**, hash reported

---

## Amendment 1 — 2026-09-25, evening

Appended after work on §4 began. The text above is left as written, and this section records what changed since. Where the two disagree, this section wins.

### A1.1 Progress on §4

| Step | State | Commit |
| --- | --- | --- |
| 1. `capstone.splits` + test | Done | `cb1369d` |
| 2. Renames | Done. Slice tests pass, `count_t_realizations.py` passes | `edd98f3` |
| Fold assignment over eligible sites (A1.2) | Done | `8a97660` |
| Manifest re-recorded from a clean tree | Done | `0563101` |
| 3–5. Pre-registration text, status, push | **Not started** | — |

Nothing is pushed.

### A1.2 Decided: folds are assigned over the 10,389 eligible sites

This answers the first open question in §3. **Measured:** counting the 3 clamped sites in the type ranking does move folds. `nineties` is a singleton whose only site is clamped. Counting it shifts every singleton ranked after it, so 191 types (one site each, 87 /d/) change fold. Speaker folds don't move.

**Decision:** the clamped sites are removed before any fold is assigned, not only before scoring. A site that is never scored should not decide which fold other sites land in. The split does now depend on `context_ms`, which decides which sites clamp, but §2 already fixes that value and puts it off-limits for tuning.

**Invariant 4 is replaced.** The figures in §1 describe the same grid built over all 10,392 sites and are superseded:

| | Invariant 4 as written (10,392) | **Now (10,389)** |
| --- | --- | --- |
| Test sites per cell | 281–535 | **279–534** |
| /d/ per cell | 67–151 | **66–149** |
| Distinct /d/ types per cell | 27–48 | **28–46** |
| Train per cell | 6,255–7,072 | **6,245–7,068** |
| Train /d/ per cell | 1,504–1,760 | **1,505–1,759** |
| Diagonal | 2,137 / 513 | **2,136 / 510** |
| Full grid | 10,392 / 2,529 | **10,389 / 2,529** |
| Speakers or types shared between test and train | 0 | **0** |

Invariant 3 is unchanged. Recorded in `results/corpus-verification/buckeye-flap-lexical-structure-20260925.txt`. The 20260924 report stays as the record of the grid over all sites. **Pre-reg §3 quotes the old figures** (the "discard is from training" paragraph and change #2 in the top list), so step 3 has to update them and add this rule to §3's construction.

### A1.3 Still open

- **Speaker-leakage placement** (the second open question in §3). One addition: a speaker-ID probe is also well defined under `word_disjoint`, where every speaker appears in training. So the candidates are `random`, `word_disjoint` or both. In either case, §4's sentence "the lookup, the control task and the speaker-leakage check all run on the same 25 cells" has to change, not just gain a line.
- **`random`'s seed value.** §2.1 says it is recorded in the pre-registration, but no value has been chosen. The code refuses to run `random` or the subsample without an explicit seed. One `seed` argument drives both: the fold shuffle, and the subsample, which draws from a generator keyed on `(seed, fold)`.
- Everything else in §3 is unchanged.

### A1.4 Naming, confirmed

*What is being graded* quotes the feedback as asking for "the **combined** condition". §2.4's claim that `joint` "matches the feedback's vocabulary" is therefore not quite right. The decision stands, and `joint` is used everywhere. The report needs one sentence mapping the feedback's "combined" to `joint`. Separately, §2.4's "48 occurrences across 7 files" includes the dated slice brief, which keeps its wording by §2.4's own rule. The live rename was 29 lines across 5 files.

### A1.5 How the code behaves, for the run

- `iter_splits(condition, speakers, words, labels, *, eligible, seed=None, subsample=False)`. `eligible` is **required**, so a caller cannot silently fall back to assigning over every site. Indices refer to the full inventory, so per-site embedding arrays for all 10,392 sites can be indexed directly.
- Subsample: `round(0.8 × n)` per class from each training set, stratified by label. It is refused on `joint`.
- A clamped site's `fold` is **null** in `predictions.parquet`, because it belongs to no fold.
- `scripts/analyze_flap_lexical_structure.py` now reads the parquet, which is the only file with the clamp flag, so it runs in `.venv-app`, not bare `python3`.
- `scikit-learn==1.9.0` was added to `requirements-app.txt`, pinned to the lock.
- `tests/test_splits.py` needs pyarrow, so the **CUDA image skips it**. Invariant 4 is checked in `.venv-app` only.

### A1.6 Found on the way, not yet done

These are needed by later steps in §4:

- **Manifest seeds:** `Manifest` has one `random_seed`. The real run has at least four: `random`'s folds and subsample, the control-task mapping, the criterion-1 bootstrap, and the synthetic draw.
- **Manifest `synthetic` flag:** `build_manifest` hardcodes `synthetic=True`, and `Manifest` defaults to it.
- **Prediction schema:** it has no subsample column yet (§2.2).
- **App refusal:** `app/streamlit_app.py:79` refuses to render any manifest that isn't `synthetic`. Sunday's data-driven banner replaces that refusal. The banner is not a second flag to add on top of it.
- **Pre-reg §6:** the 64% secondary run partly answers its first advisor question, about the ~32% training discard. Worth a cross-reference in step 3.

---

## Amendment 2 — 2026-09-26

**Revision 3 is on `origin` at `43b2896`**, pushed 2026-09-26 before any run. §4 steps 3–5 are done, and invariant 1 is satisfied: the run may start. From here the pre-registration is append-only (§7 *Deviations*).

**Invariant 4 moved a second time.** A review of the draft on 2026-09-26 found that "word type" was never defined and that contractions split one word across two keys (`do` in training, `don't` in test). The split now keys on `capstone.splits.type_key()`: the word form, lowercased, with 34 contracted forms merged onto their host (187 sites). That supersedes A1.2's figures:

| | A1.2 (10,389, word forms) | **Registered (10,389, type keys)** |
| --- | --- | --- |
| Test sites per cell | 279–534 | **310–535** |
| /d/ per cell | 66–149 | **57–189** |
| Distinct /d/ types per cell | 28–46 | **25–39** |
| Train per cell | 6,245–7,068 | **6,257–7,100** |
| Train /d/ per cell | 1,505–1,759 | **1,389–1,820** |
| Diagonal | 2,136 / 510 | **2,187 / 543** |
| Full grid | 10,389 / 2,529 | **10,389 / 2,529** |
| Leakage | 0 | **0** |

**Settled since Amendment 1** (all registered in revision 3):
- Speaker-leakage check under `random` and `word_disjoint`.
- Seeds 20260926 (folds and subsample), 20260927 (control mapping), 20260928 (bootstrap), 20260929 (criterion 2's null mappings).
- Probe hyperparameters: L2, C = 1.0, per-probe standardization, no tuning.
- Criterion 2 replaced by a max-over-layers randomization test, because ±0.03 fired 59–80% of the time with no leak.
- Lemma overlap measured (25.1% of `joint` test sites, same label), plus a reported re-score of criterion 1 on the 7,586 sites whose lemma is unseen.

**For the run**, beyond A1.6: the probe harness should read the four seeds from one place in code, not take each as an argument. The control task and lookup are keyed by `type_key`, not the raw `word` column.

---

## Amendment 3 — 2026-09-26, evening

**The pilot has run, and it failed.** WavJEPA and MFCC ran through every registered probe, and the §5 criteria are scored on `joint`. Everything is on `origin`. From here the pre-registration's §7 records what was done. It changes nothing above it.

### A3.1 What ran, in order

| Step | Commit |
| --- | --- |
| §7 clarifications, pushed before any feature existed: midpoint pooling, the lookup on unseen types, the speaker-ID metric, and the paired bootstrap. Also the author's wav2vec2 prediction, under its own heading | `8d1630b` |
| Seeds in `capstone.seeds`, a manifest with seeds by name and a required `synthetic` flag, and `subsampled` and `pooling` columns (closes A1.6's first three items) | `647ec90`, `003235f` |
| Feature extraction: WavJEPA and MFCC, mean/max/midpoint, stored as float32. *Corrected 2026-10-18: WavJEPA's frames were rounded to float16 before pooling (`129dfe1`); MFCC is float32 throughout* | `c53d4c7`, run at `afb5811` |
| Probe harness, `capstone.probes`, with tests | `4881f42` |
| **Criteria report** | `6915b83` |
| Results CSV, folds CSV, manifest | `2d61fba` |
| Figures | `459565f` |
| §7 note, written after the run: the lookup's tie rule (reaches 0 sites under the type-disjoint conditions) | `2847927` |

### A3.2 The result

On `joint`, mean pooling, pooled over 10,389 sites (2,529 /d/), from `results/flapping/criteria-20260926.txt`:

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Best layer beats MFCC by ≥ 0.05, with the CI's lower bound > 0 | **FAIL**: layer 3 scores 0.5438 against MFCC's 0.5274, gap +0.016, CI [−0.015, +0.056] |
| 2 | Control shows no leak | **PASS**: 0.5377, under the null 95th percentile of 0.5502 (p = 0.13) |
| 3 | Curve not flat, spread ≥ 0.05 | **FAIL**: 0.025 |
| 4 | /d/ F1 ≥ 0.45 | **FAIL**: 0.356 |

Reported, not gated (mean pooling, best WavJEPA layer; from `probe-results-20260926.csv`):

| Condition | WavJEPA | MFCC | Lookup | Control |
| --- | --- | --- | --- | --- |
| `random` | 0.652 (L6) | 0.582 | 0.941 | up to 0.648 |
| `speaker_disjoint` | 0.627 (L7) | 0.574 | 0.930 | up to 0.631 |
| `word_disjoint` | 0.559 (L3) | 0.534 | 0.500 | up to 0.534 |
| `joint` | 0.544 (L3) | 0.527 | 0.500 | up to 0.538 |

- **The pipeline is not obviously broken.** §5's first reading of "near chance at every layer" is the alignment path. Under `random`, though, there is a real layer curve (0.567 at L0, 0.652 at L6). The lookup collapses to exactly 0.500 under both type-disjoint conditions. Extraction checked every site's clamp flag and frame count against the inventory.
- **Closing word identity costs most of the score** (−0.093, random to word_disjoint). Closing speakers costs −0.025.
- The subsample secondary costs only 0.006–0.012, so training size is not the story.
- Pooling barely matters under `joint`: 0.5438 / 0.5409 / 0.5438 for mean / max / midpoint.
- Speaker-ID reaches 0.92 for WavJEPA, 0.69 for MFCC (chance 0.025).
- Lemma-unseen re-score (7,586 sites): gap +0.030, CI [−0.004, +0.073].

These results give the *Feedback → What Changed → Evidence* slides their evidence: random / speaker-disjoint / word-disjoint / joint, layer by layer, with an MFCC baseline. A failed pilot is a real result, and §5 says how to read it.

### A3.3 Where things are

- **Repo, committed:** `results/flapping/`: `criteria-20260926.txt`, `probe-results-20260926.csv` (294 rows), `probe-folds-20260926.csv`, `run-manifest-pilot.json`, and `figures/` (layer curves, condition drop, controls, pooling).
- **`/data` only (invariant 5):**
  - `features-b378c303ca90/`: pooled features and their extraction manifest.
  - `probe-run-20260926/`: cached fits, `predictions.parquet` (3,127,992 rows, `synthetic = false`), `control-predictions.parquet`, `run-manifest.json`.
- **How to rerun:** in the docstrings of `scripts/extract_site_features.py`, `scripts/run_probes.py` and `scripts/plot_layer_curves.py`. Extraction needs both environments, because the CUDA image has no pyarrow. `compose.yaml` now sets git's `safe.directory` for `/workspace`, so manifests written in the container record a commit (`afb5811`).

### A3.4 Not done from §4

- **wav2vec2-base (exploratory).** Not started. It still needs its own frame geometry, derived and verified (§4's warning stands). Its checkpoint is not cached under `/data/checkpoints/hf`.
- **App swap.** Not started, and it touches three things:
  - The app still reads the synthetic run at `/data/embeddings/predictions.parquet`, which the real run deliberately did not overwrite.
  - `render_guard` (`app/streamlit_app.py`) still refuses a non-synthetic manifest. Sunday's data-driven banner replaces it (A1.6).
  - The real parquet has one row per site per encoder × layer × pooling × condition × subsampled. So `retrieval.SliceStore` queries must also filter on layer, pooling and `subsampled`. Today they filter only on condition and encoder.
- **README** items from *What is being graded*. The result above, the limitations (§1a lemma overlap, the pilot's failure) and next steps now have content to draw on.
- **Following from the failure, for a later round:** a lemma-disjoint split, the 2–4 frame site width, context beyond the flap interval, and a stronger probe. §5 rules out adjusting anything and rerunning under the same registration. Each is a new registration.

### A3.5 Noticed, not acted on

`src/capstone/data/` is gitignored, probably by a blanket `data/` rule, so its `__init__.py` is not in the repo. Nothing imports it today.

## Amendment 4 — 2026-09-26, night

**The app now shows the pilot run, and the per-site verdicts are replaced by a blind listening audit.** This closes A3.4's *App swap*. Nothing was rerun or retuned (§5).

### A4.1 What changed, in order

| Step | Commit |
| --- | --- |
| `SliceStore` pins one slice: layer (required, no default), pooling and `subsampled`. Counts are taken within the slice, and a site returned twice raises an error | `c4ca2b8` |
| App reads `/data/embeddings/probe-run-20260926/`. `render_guard` is replaced by a banner chosen by the manifest's own `synthetic` field. The sidebar sets the whole slice, and its default layer is labelled as the run's highest, picked after the fact. Uncertainty §2 shows per-layer intervals from the CSV | `5221ac1` |
| Verdict lines record which prediction was on screen | `f3397be` |
| Result tab, first in the reading order: the criteria table parsed from the report, the committed PNGs, A3.2's one line of reading, and a split selector linked to the site view | `4320b3f` |
| Co-Design fix: the `label` verdict was misread as judging the model | `a440624`, superseded by `5bdc95e` (A4.2) |
| Co-Design fix: coverage captions printed raw `**` | `a63f8e7` |
| Pre-registration §7: the audit registered before listening, with its sample committed | `f92e503` |
| Audit (blind) tab. Browse verdicts are kept and never counted. `retrieval` and `grounding` are removed | `5bdc95e` |
| `scripts/summarize_audit.py` writes the audit report; the Result tab shows it | `961cd16` |
| Compare tab counted sites compared as rows shown ("10,389" over 639) | `53c4ed6` |

### A4.2 Why the verdicts became an audit

The four per-site verdicts did no research work:
- They went to a log nothing read, on sites picked from filtered tables with the model's guess on screen.
- `label` asked whether the /t/ or /d/ under a flap was right, which cannot be heard. That is the project's premise.
- `retrieval` and `grounding` checked the app, not the data.

The failed pilot gives human review a real question. §5's first reading of a result near chance is the alignment path: are the sites what the inventory says? The audit answers it by ear:
- 40 /t/ + 40 /d/ sites, seed 20260930, blind to the probe;
- the questions `flap`, `word` and `clip`, each answered yes, no or can't tell;
- per-class rates with Wilson intervals.

All of it was registered in pre-registration §7 before any listening. **Status: 4 of 80 sites reviewed.** The report (`results/flapping/audit-<date>.txt`) and the README's limitations line wait on the rest.

**For slide 2:** the `label` row's *What Changed* is two steps. The first, `a440624`, reworded the prompt. The second, `5bdc95e`, is the real change: the question could not be answered by ear, so `word` replaced it, inside a blind audit. Cite both.

### A4.3 The real parquet against the synthetic one

- **The columns and types are identical.** The difference is one row per site for every encoder × layer × pooling × condition × subsampled combination: 3,127,992 rows.
- **The lookup rows have NULL layer and pooling, and MFCC has only layer 0.** The app shows WavJEPA only.
- **`joint` has no subsampled rows.** The app falls back to the headline run there and says so.
- **Every site has a row**: 10,389 scored and 3 excluded, so the "not reached" state is empty for this run.

### A4.4 Slide 3

Four crops, not whole tabs:
1. The Result tab: the verdict box, the criteria table and the layer curves.
2. Compare, `random` against `joint`, /d/, layer 3: 482 sites went from right to wrong.
3. Browse, one site: `sNNNNx:NNNNN`, "[utterance]", with P(/t/) 0.854 at `joint` layer 3 against 0.25 under `random`.
4. The Audit (blind) tab.

### A4.5 Noticed, not acted on

- **Result tab selector and caption disagreed once.** In one browser screenshot, the selector read `random` while the caption below it said the site view showed `joint` at layer 3. 40 random headless interactions did not reproduce it. It is most likely a screenshot taken while the page was rerunning, but that is unconfirmed.
- **Listen picks from the first 200 rows only.** Browse's Listen picker lists the first 200 matching rows (`LOOKUP_LIMIT`). A site beyond them can be reached by narrowing the filters, or by identifier in the lookup tab, which has no audio.
- **A3.5 was resolved before this amendment** (`34da84a`).
