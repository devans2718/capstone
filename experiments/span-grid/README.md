# Span grid: where the flap's /t/–/d/ cue sits — pre-registration

**Registered 2026-10-17. Nothing has been run.** Final as pushed, with the
author's predictions in §6. From here, departures are appended to §8, never
edited in above.

The pilot (`experiments/pilot-flapping/`) asked whether WavJEPA's frames *at
the flap* separate underlying /t/ from /d/ on unseen words. It failed, 1 of 4
criteria (`6915b83`), and the blind audit cleared the inventory. This study
asks the next question: **where is the cue, and does hearing the neighbouring
speech carry it?** The expected cue is lengthening of the sonorant stretch
before an underlying /d/, which the transcription hides (pilot §1a). Under that
reading the pilot's near-chance result is what a linguist predicts for frames
that contain the flap alone.

Nothing here changes how the pilot is read. Its registration, deviations and
result stand as written; this file is a separate study that reuses its
inventory, probe and split.

**Terms.** A **span** is a stretch of the `.phones` tier around a flap (§2);
every site has four. The **pooled** and **cropped designs** are the two ways a
span is probed (§3). *Cell* keeps the pilot's meaning: one of the split's 25
test sets.

---

## 1. Sites

**The pilot's 10,389 eligible sites less 55 whose environment is not the one
under study: 10,334.** The 55, fixed now from the `.phones` tier:

- 18 directly after another flap (`dx`)
- 23 whose preceding segment carries a `+1` suffix (`ae+1`, `ih+1`, …), whose
  meaning in Buckeye's notation is not established here
- 12 after an obstruent (`v`, `s`, `th`, `k`, `f`), two of them through a
  coda `r`
- 2 after `w`, one through a coda `r`

The exact list is fixed now as a set, not a rule re-applied later. Site IDs
are Buckeye-derived and stay out of this repository, so what is committed with
this file is the script that regenerates the list from a registered copy,
`scripts/measure_span_grid.py`, and the sha256 of the sorted list:
`964e7ab99882340560b6da10fffb7a54f6efb92e519b8247f95f0602b480db21`. The
script exits non-zero if the list it produces differs, and the build checks
the same hash before it runs. The figures in §1–§3 are that script's output.

## 2. Spans

Every span is in the recording's time base, from the `.phones` tier, parsed by
`count_flap_sites.read_phones` (the parser carrying the ~7% fix).

- **ɾ** — the flap.
- **Vɾ** — the preceding sonorant stretch through the flap. **The stretch runs
  back from the flap through any sonorant coda** (`r`, `l`, `n`, `m`, `ng`,
  `nx`) **to the start of the vowel before it**: *hardy* is `aa r` + flap, as
  *heeded* is `iy` + flap, because the cue under study is lengthening of the
  stretch, not of one segment. V includes nasalized vowels, `er`, and syllabic
  consonants (`el`, `em`, `en`, `eng`): a flap before *little*'s `el` is
  intervocalic. Codas found: `r` 557, `l` 211, `n` 71, `m` 18, `ng` 6, `nx` 3,
  `r n` 1.
- **ɾV** — the flap through the following segment, **whatever that segment
  is**: a vowel, a sonorant, `hh` (*but he*), a pause.
- **VɾV** — the preceding stretch through the following segment.

**Spans are clipped to the excerpt (flap ± 200 ms), and clipped sites are
kept.** 677 sites (6.55%) run past it: 269 on the preceding side (overshoot
median 28.9 ms), 442 on the following (44.4 ms). Dropping them would select on
the cue itself, since a stretch runs past the edge by being long. Clipping
censors the stretch's duration at 200 ms rather than erasing it (the
median stretch is 80.1 ms), and the clipped audio was never in the encoder's
input in the pilot either. Clipped sites are flagged in the build output, and
every span is also reported on the unclipped sites alone.

## 3. Two designs

The pilot's excerpt is the flap ± 200 ms, and both encoders' transformers see
all of it: every frame can carry both neighbours. So "which frames are pooled"
and "which audio was heard" are different manipulations, and each design takes
one.

### The pooled design — where the cue sits in the representation

The pilot's excerpt, encoded exactly as the pilot encoded it (pilot §2), with
frames pooled over each span. A difference between spans says where in the
representation the information is, not whether hearing more speech helps.

**Needs a re-extraction.** The pilot stored only per-site pools over the
flap's own frames (`(10392, 13, 768)` per pooling), so no stored frame reaches
the neighbours. The new extraction stores each span's pool; it is
deterministic at the pinned revision.

### The cropped design — whether hearing the neighbouring speech helps

Each span's audio alone is encoded. **Built so that spans differ only in which
speech they contain:**

- **Fixed length L**: the longest eligible excerpt (533.9 ms, measured
  2026-10-17), computed from the inventory by the build, not typed in. Input
  length then cannot differ between spans.
- **Same position**: each crop sits where it sat in its excerpt, so the flap is
  centred in every input. The pilot measured why this matters: a 10-frame
  shift moves layer 0 by cosine distance 0.394 (pilot §2).
- **Padding is noise matched to the recording, not silence.** Per track:
  Gaussian noise shaped to the long-term average spectrum of that track's
  pauses (`SIL` intervals of ≥ 200 ms), at their RMS level. Silence is out of
  distribution for every encoder here. Synthesized rather than cut from the
  pauses, which can hold breaths and the interviewer.
- **Ramps**: a 2.5 ms raised-cosine crossfade between noise and speech at each
  crop edge, inside the crop, the same for every span. It costs the shortest
  span (ɾ, median 27.4 ms) proportionally the most, and that is stated rather
  than avoided: a ramp outside the crop would let the neighbouring audio back
  in.
- **One noise seed per site**, shared across that site's spans. The noise then
  differs between sites and is identical between a site's spans.
- **Normalization happens once, on the full excerpt.** Both encoders scale
  each input waveform: WavJEPA's feature extractor by one RMS-derived gain
  (pilot §2), wav2vec 2.0's by `do_normalize` (zero mean, unit variance). The
  encoder's own normalization is computed on the full excerpt and applied
  before cropping, and crops are fed with the extractor's normalization
  disabled. Otherwise mostly-noise inputs would rescale the speech, by a
  different amount for each span. **Verified before the run:** the full
  excerpt through this path matches the standard path within float tolerance.
- **For MFCC the two designs nearly coincide.** Its frames are local, but its
  deltas reach ±40 ms, so near a crop edge the cropped design's deltas see
  noise where the pooled design's see speech.

**Padding-only control.** The full excerpt, padded to L as above, against the
excerpt as is. It changes nothing but the padding. If the two agree, the
padding is inert; if not, that is known before the short spans are read.

## 4. Representations

- **WavJEPA**, frozen, at the pilot's pinned revision (`6be4a50`), 13 layers.
- **`facebook/wav2vec2-base`** at revision `0b5b8e8`, 13 hidden states (CNN
  output + 12 layers, 768 wide). Self-supervised on 960 h of LibriSpeech with
  no transcript supervision, so nothing it carries came from text. Already the
  pilot's registered exploratory encoder; the pilot's 2026-09-26 prediction
  ("sits higher than wav2vec2-base's") applies here unchanged. **Frame
  geometry derived from its conv stack and verified, as the pilot requires**
  (pilot, *Second encoder*); no frame rate is typed in. From its config the
  stride product is 320 samples (20 ms) and the receptive field 400 (25 ms),
  to be confirmed by the same test WavJEPA's geometry passed.
- **MFCC**, the pilot's configuration exactly (pilot §4).

**Which frames belong to a span:** WavJEPA by onset, MFCC by centre, both as
in the pilot. wav2vec 2.0 by centre, as MFCC, since its frames are as coarse as
the median flap. On the config's geometry, 168 flaps (1.6%) contain no frame
centre, and take the frame whose centre is nearest the flap's midpoint; 8,065
contain exactly one. These counts are recomputed once the geometry is verified,
and any change is recorded in §8. So for
the ɾ span, wav2vec 2.0 is mostly one frame.

**Pooling: mean over the span's frames.** One rule. Max and midpoint, the
pilot's secondaries, are not run: over a multi-segment span they do not mean
what they meant over a flap.

**Mean pooling discards duration.** The mean of a span's frames does not
depend on how many frames there were. An MFCC gain in Vɾ can therefore come
only from spectral differences, and an encoder carries the stretch's length
only if its frames encode it themselves. The duration baselines (§5) are what
make length visible.

## 5. Probe, split, baselines, checks

**Probe and hyperparameters: the pilot's, unchanged** (pilot §3): L2 logistic
regression, C = 1.0, `lbfgs`, `max_iter = 1000`, per-probe `StandardScaler`,
`class_weight='balanced'`.

**Split: `joint` only**, out-of-fold predictions pooled over its 25 cells
(pilot §3). The other three conditions are not run: the pilot's drop across
them is already measured, and this study's question is between spans.

**Folds: the pilot's, assigned over its 10,389 sites; the 55 exclusions (§1)
leave after assignment.** This departs on purpose from the pilot's rule that "a
site that is never scored does not decide where other sites go" (pilot §3).
Applied here, that rule would re-rank word types and move sites between folds.
The pilot's committed lemma subset would no longer match them, and the
reproduction check would need a second set of folds. Comparability with the
pilot is worth more here than the rule. The 55 still shape the type ranking,
and nothing else.

**Baselines, each its own probe under the same split:**

- **MFCC** for every span (above).
- **Duration of the preceding stretch, clipped to the excerpt** — one number
  per site, the stretch as it lies inside the audio the encoder receives, so
  the baseline knows nothing the encoder could not hear. Compared against the
  Vɾ and VɾV spans: does an encoder carry anything beyond one hand-measured
  number?
- **MFCC + clipped duration** (40 dimensions) for the Vɾ and VɾV spans: the
  acoustic floor that has the length. Mean pooling gives MFCC no duration
  (§4), so beating MFCC alone could mean only that duration beats spectra.
  C5 reads both.
- **Secondary, reported:** the stretch's full duration (unclipped) and the
  vowel's alone. The first shows what the clipping costs; the second whether
  lengthening sits on the vowel or spreads across the coda.

**Checks for every span and representation:**

- **The control task's randomization test**, the pilot's criterion 2
  procedure and seeds (20260927, 20260929), maximum over layers.
- **The lemma-unseen re-score**, on the pilot's fixed subset of 7,586 sites
  (pilot §5). The file is Buckeye-derived and held privately since the public
  release; its role is unchanged. It matters more here: wider spans carry more
  of the word.
- **Per-layer 95% intervals** from the pilot's word-type bootstrap (10,000
  resamples, seed 20260928).

**Built-in reproduction check, before any span is read.** The pooled design's
ɾ span for WavJEPA, on the pilot's 10,389 sites and folds, has to reproduce the
pilot's `joint` mean-pooled curve. If it does not, the new extraction is wrong
and nothing else is read.

## 6. Registered comparisons and predictions

**The span grid gates nothing.** It is the study's stopping point (decided
2026-10-16): the result is reported whatever it is. What is fixed here is
which comparisons are read, how an effect is called, and what was predicted.

**An effect is called with the pilot's machinery:** a best-layer gap of
**≥ 0.05** balanced accuracy **and** a paired word-type bootstrap 95% lower
bound **> 0**, with the best layer re-selected in every resample (pilot §5,
criterion 1).

| # | Question | Computed as | Per |
|---|---|---|---|
| C1 | **Does pooling over more of the neighbouring speech help, when the encoder heard the whole excerpt?** | Vɾ − ɾ, ɾV − ɾ, VɾV − ɾ | pooled design, each representation |
| C2 | **Does it help when the encoder hears only the span?** | the same three gaps | cropped design, each representation |
| C3 | **Do a span's frames carry speech the span itself doesn't contain?** | each span, pooled − cropped | each encoder |
| C4 | **Does an encoder know more than the stretch's duration?** | Vɾ and VɾV best layer − clipped-duration baseline | both designs, each encoder |
| C5 | **Does an encoder beat the MFCC features?** | each span, best layer − MFCC, and best layer − MFCC + duration | both designs, each encoder |
| C6 | **At what depth does the information peak?** | best layer per span | each encoder; with intervals, no threshold |
| C7 | **Which encoder is better?** | each span, WavJEPA best layer − wav2vec 2.0 best layer | both designs |

**The author's predictions** — *to be written by the author before this file
is pushed: a direction, and a rough size where the author has one, per
comparison; "no prediction" where there is none.*

*Recorded 2026-10-17, before any span has been extracted.*

- **C1.** Vɾ beats ɾ. ɾV shows no discernible change from ɾ. VɾV improves on ɾ
  by the same amount as Vɾ, the gain carried by the preceding stretch.
- **C2.** The same pattern as C1.
- **C3.** The frames do carry speech outside their span, but the gap is
  present and below the effect threshold.
- **C4.** About level: what an encoder carries in Vɾ is, in effect, the
  stretch's duration.
- **C5.** Against MFCC: Vɾ and VɾV beat it together, or neither does. No
  prediction about how the encoders differ in this, and none against MFCC +
  duration.
- **C6.** WavJEPA's layer-wise curves resemble the pilot's. The pilot's
  2026-09-26 prediction stands: WavJEPA's information sits higher than
  wav2vec2-base's.
- **C7.** No prediction.
- **The duration baseline's own score:** no prediction.

## 7. Reported, not compared

The padding-only control; each span on unclipped sites only; the control task
and lemma-unseen re-score per span; per-span spread over the split's 25 cells;
the three duration baselines side by side.

## 8. Deviations

Any departure from the above gets appended here with a date and a reason, and
the original text stays as written.

**2026-10-18 — wav2vec2-base's geometry, verified (§4).** Checked live on
2026-10-17 at `0b5b8e8` (resolving to `0b5b8e868dd8…`) in the pinned
container, by the tests WavJEPA's geometry passed
(`scripts/probe_w2v2_frame_positions.py`, `tests/test_wav2vec2_model.py`).
Stride product 320 samples (20 ms) and receptive field 400 (25 ms), as the
config says. The model's frame count equals `(n − 400) // 320 + 1` at every
length tried. A single-sample spike peaks in the predicted frames, and frame t
reads samples [320t, 320t + 400) of its input and nothing outside them, beyond
conv layer 0's GroupNorm floor. The map is uniform: frame t's centre is sample
320t + 200. Two corrections to §4:

- **Hidden state 0 is not the CNN output.** It is LayerNorm(projection +
  convolutional positional embedding), so like WavJEPA's index 0 it carries
  position. The positional convolution (kernel 128) also mixes neighbouring
  frames. Index 12 is the last layer's output and equals the model's
  `last_hidden_state`.
- **Recount.** With the flap's bounds rounded to samples, as the extraction
  rounds them for MFCC, 160 of the 10,334 sites (1.5%) contain no frame centre
  and 8,031 (77.7%) exactly one; on the pilot's 10,389, 169 and 8,064. The
  provisional 168 and 8,065 were computed outside the repository and could not
  be reproduced. "Mostly one frame" stands.

**2026-10-18 — The cropped design's normalization covers one of two (§3).** §3
applies the encoder's input normalization once, on the full excerpt, so that
mostly-noise crops do not rescale the speech. Both encoders also normalize
inside the model: conv layer 0 is a GroupNorm with one group per channel, which
standardizes each channel over its whole input (for WavJEPA, over each 2.01 s
segment, longer than L). It cannot be disabled without changing the model. In a
padded crop the noise therefore still sets the scale of the speech's conv
features, by a different amount for each span. Both spike tests show it: a
spike anywhere moves every frame (`scripts/probe_frame_positions.py`,
`scripts/probe_w2v2_frame_positions.py`). How it is handled is decided before
the cropped inputs are built, and recorded here. The padding-only control
measures the effect for the full excerpt only.

**2026-10-18 — GroupNorm in the cropped design: kept, and measured (§3;
resolves the entry above).** Conv layer 0's GroupNorm is kept as part of each
encoder. §3 controls the normalization outside the model, the feature
extractor's, because preprocessing is the experimenter's to fix. How the
encoder's own layers respond to an input holding only the span is part of what
the cropped design asks. Freezing GroupNorm's statistics to the full excerpt's
is rejected: they summarize the neighbouring speech, as 512 channel means and
512 standard deviations, and would carry it into every crop. §3's input
normalization already carries a leak of that kind, one gain and offset computed
on the full excerpt. It stays, because it is one pair of numbers, and it is
stated here so the line between the two is visible.

**Measured and reported, not compared:** the per-channel scale and offset
GroupNorm applies to the crop, against those it applies to the full excerpt.
Per site, the median over channels of |log(σ_crop / σ_full)| and of
|μ_crop − μ_full| / σ_full, reported per span and encoder as a median and 95th
percentile. Applied to the same speech samples, these are the whole of the
difference GroupNorm makes, so the measure is defined for every site, including
ɾ spans too short to hold a frame clear of the crop edges.

**2026-10-18 — WavJEPA's spans take only frames that read real audio (§2,
§4).** A span clipped at the excerpt's end would, by onset, include WavJEPA's
last returned frame, which reads mostly the model's zero padding
(`FrameGeometry.n_clean_frames`). Every WavJEPA span drops any frame whose
receptive field runs past the excerpt. The ɾ span never reaches the end, so the
pilot's features are unchanged; wav2vec 2.0 and MFCC return only whole frames
and are unaffected. The number of spans affected is recorded in the extraction
manifest.

**2026-10-18 — Frame precision: the pilot's float16 is kept (§3, §5).** The
pilot's WavJEPA frames were rounded to float16 by `WavJEPAExtractor.encode`'s
default dtype before pooling, then stored as float32; the extraction script's
docstring said float32 throughout and is corrected. Every encoder here keeps
float16 frames, so the pooled design's ɾ span gets the pilot's exact features
and the reproduction check (§5) compares like with like, with no tolerance to
set.

**2026-10-29 — The cropped design is deferred (§3, §6, §7).** Decided
2026-10-24 and appended before any span has been probed: the pooled design's
spans are extracted (2026-10-18), and no probe has been fit to them. Only the
pooled design runs now. The cropped design's machinery (§3's normalization
path and its check against the standard path, noise matched to each track's
pauses, the padding-only control, and the GroupNorm measure in the entries
above) is a lot to build and to understand, and the remaining time goes to
the pooled design and to reviewing the study. It is deferred, not dropped:
§3's specification and §6's predictions stand as registered, and if it runs
later it is read against them as written, with that run's date stated.

What this leaves untested, by §6's comparisons:

- **C2 and C3: not tested.** Both need the cropped design.
- **C4, C5 and C7: the pooled design only.** Each is registered for both
  designs; only its pooled half is read.
- **C1 and C6: unaffected.** Both are pooled-design comparisons.

From §7, the padding-only control and the GroupNorm measure are not produced;
the rest of §7 is reported for the pooled design.

**2026-10-29 — The 55 exclusions meet the pilot's fixed machinery, and two
readings of the text (§1, §2, §5–§7).** §5 keeps the pilot's folds, assigned
over its 10,389 sites, with the 55 leaving after assignment, and does not say
how the 55 meet the rest of the pilot's fixed machinery. Fixed here before any
grid result is read: the 55 leave every probe's training and test sets, and
everything else is the pilot's, restricted to the 10,334 and never rebuilt or
redrawn.

- **Control task.** The pilot's control mapping over its 991 type keys (seed
  20260927) and its 10,000 null mappings (seed 20260929) are kept. Each site
  keeps its type's control label. The 8 types whose sites are all among the
  55 drop out of scoring.
- **Word-type bootstrap.** The pilot's index over the 991 types and its
  resample weights (seed 20260928) are kept; the 8 types carry no sites.
- **Lemma-unseen re-score.** The pilot's fixed subset of 7,586 sites,
  intersected with the 10,334: 7,546 sites (1,727 /d/) in 797 of its 805 type
  keys. Its bootstrap keeps the pilot's 805-type index.
- **Counts.** The 10,334 hold 2,513 /d/ (the pilot's 10,389, 2,529). The
  smallest of the 25 test cells holds 310 sites, 57 of them /d/.
- **§7's unclipped sites.** "Each span on the unclipped sites alone" (§2, §7)
  is read as the 9,657 sites with neither clip flag, scored from the existing
  out-of-fold predictions without refitting, as the lemma-unseen re-score is.
- **C5's MFCC + duration gap.** §6's table says "each span"; §5 builds the
  MFCC + clipped-duration baseline for Vɾ and VɾV only, and the duration is
  that of the stretch before the flap, which ɾ and ɾV do not contain. That
  half of C5 is read for Vɾ and VɾV.

The ɾ span is fit twice: on the 10,389 for §5's reproduction check, and on the
10,334 for the grid, which is the one compared.

**2026-10-29 — C7's rule is one-sided; the call stands (§6). Written after
the pooled design's results were read.** §6 calls an effect when a gap is
≥ 0.05 and its paired bootstrap 95% lower bound is > 0, and C7 is computed as
WavJEPA − wav2vec 2.0. As written, the rule can call only a WavJEPA advantage,
though C7 asks which encoder is better and no direction was predicted. The
registered call stands: no effect on all four spans. Reported, not called:
the gap is negative on every span, wav2vec2-base ahead by 0.062 to 0.075 with
every 95% interval below zero (`results/span-grid/`, `004ea19`). The rule is
not re-read in reverse here because the reading would be chosen after the
result. The encoders' order is supported by registered comparisons: C5 calls
wav2vec2-base ahead of MFCC on every span and WavJEPA on none.

**2026-11-21 — A re-score by the flap's position in its word, registered
before any position split is read (§6, §7).** A new analysis of the pooled
design's existing out-of-fold predictions (`grid/predictions.parquet`, from
`004ea19`): no probe is fit and nothing is re-extracted. It asks where
wav2vec2-base's lead on unseen words comes from. Pasad et al. (*What Do
Self-Supervised Speech Models Know About Words?*, Fig. 4, §5.1.2) find
that frames near the centre of a word segment are the most informative of
word identity, in wav2vec 2.0 base and every other model they test. If
the score is a lexical shortcut, it should come mostly from flaps inside a
word. If it is phonetic, position should matter little. *Lexical* here
includes parts of words: word-medial sites include flaps at a morpheme's
edge (*started*, *getting*, *thirty*), and the inventory has no morpheme
segmentation, so the re-score cannot tell a word-level shortcut from a
morpheme-level one.

**Groups**, from the inventory's `phone_idx_in_word` against the length of
the citation form:

- **word-final**: the flap is the citation form's last phone (*that, but,
  lot, it*)
- **word-medial**: neither first nor last (*little, better, city*)
- **word-initial**: the first phone, after a vowel-final word (*to, do*)

Counted 2026-11-21 from the inventory and the grid's site mask alone, with
no prediction loaded (`run_probes.SpanData`):

| Group | Sites | /d/ | Word types | Top 10 types' share of sites |
|---|---|---|---|---|
| word-final | 6,428 | 1,462 (22.7%) | 387 | 58.2% |
| word-medial | 2,987 | 867 (29.0%) | 610 | 29.3% |
| word-initial | 919 | 184 (20.0%) | 31 | 97.1% (*to* alone 722) |

37 types have sites in both word-final and word-medial: two-flap words such as
*dated*.

**Layers are fixed, not re-selected.** Each block is read at its best layer in
the span grid's summary (`20261029`): wav2vec2-base layer 9 on every span,
WavJEPA layers 3, 4, 4 and 6 (ɾ, Vɾ, ɾV, VɾV), and MFCC's single
representation. Position plays no part in choosing a layer.

**Registered comparisons**, each called by §6's rule made two-sided: a gap of
at least 0.05 in absolute value **and** a paired word-type bootstrap 95%
interval that excludes zero. The C7 entry above shows why the rule has to be
two-sided here. The bootstrap is the pilot's 991-type index and weights
(seed 20260928), with both groups scored on the same resampled types.

- **P1.** wav2vec2-base, ɾ span, layer 9: word-medial minus word-final
  balanced accuracy.
- **P2.** P1's gap minus MFCC's ɾ gap (word-medial minus word-final).
  Position also changes the phonetics: a word-final flap sits across a word
  boundary, before a different mix of vowels. MFCC picks up that part with
  no knowledge of words, so P2 asks whether the encoder's position difference
  goes beyond the acoustic features'.

**Confounds, and how each is handled:**

- **Class balance moves with position** (22.7% /d/ word-final, 29.0%
  word-medial). Balanced accuracy is computed within each group, and /t/
  and /d/ recall are reported per group.
- **Frequency moves with position.** Word-final sites concentrate in a few
  function words (the top ten types hold 58% of them). Reported, not called:
  each group's balanced accuracy with every word type weighted equally, each
  site weighted by one over its type's site count in the group.
- **Type counts** per group and class are reported with every score.

**Reported, not called:** P1 and P2 for WavJEPA and for the other three spans;
the word-initial group's score, whose 31 types, mostly *to*, are too few for
an interval to mean much; per-layer curves for each group.

**What each reading predicts.** A lexical shortcut predicts P1 and P2 both
positive. A phonetic cue predicts P1 near zero, or a P1 that MFCC shares
(P2 near zero).

**The author's prediction:** I predict a lexical shortcut.

The scoring script is `scripts/score_position.py`. It writes aggregates only,
to `results/span-grid/position-<date>.*`.

**2026-11-21 — A re-score by the frequency of the flap's word, registered
before any frequency split is read (§6, §7).** Like the position re-score
above, a new analysis of the pooled design's existing out-of-fold predictions
(`grid/predictions.parquet`, from `004ea19`): no probe is fit and nothing is
re-extracted. The position re-score named frequency as a confound and did
not measure it. Measured afterwards (`scripts/count_position_frequency.py`,
`63fa6d6`), the words behind word-final flaps are far more frequent than the
words behind word-medial ones, and within word-final flaps frequency follows
the label: /t/ words are common function words, /d/ words mostly are not.
Frequent words are reduced more in conversation. A probe could therefore
separate /t/ from /d/ by how reduced the flap is, a property of the sound
that follows the word rather than the contrast, and it would need no
knowledge of words to do it: MFCC also does better word-final. This
re-score asks how much of each representation's score survives when
frequency is held roughly constant.

**Frequency** is the word's token count over every Buckeye `.words` file,
any pronunciation, so it is measured on the same corpus. **Bands** are
decades of that count, fixed at round numbers: under 10, 10–99, 100–999,
and 1,000 or more. Counted 2026-11-21 from the inventory, the corpus's word
tier and the grid's site mask, with no prediction loaded:

| Band | Sites | /d/ | Word types | /t/ types | /d/ types |
|---|---|---|---|---|---|
| under 10 | 1,094 | 433 (39.6%) | 655 | 384 | 287 |
| 10–99 | 2,126 | 828 (38.9%) | 276 | 147 | 136 |
| 100–999 | 3,007 | 1,141 (37.9%) | 57 | 26 | 32 |
| 1,000 or more | 4,107 | 111 (2.7%) | 11 | 10 | 1 |

The top band is almost all /t/: its word-final sites (3,272) are all /t/,
and its 111 /d/ sites are one word-initial type. Balanced accuracy needs
both classes, so **the top band is left out of the called scores**. Below
it, each band's /d/ share is 38–40%, close to even.

**The frequency-stratified score:** balanced accuracy computed within each
of the three lower bands, then averaged with each band weighted by its site
count. Within a band, word frequency varies by at most a factor of ten and
no longer predicts the label much, so a probe that reads only reduction
loses most of its advantage there.

**Layers are fixed**, as in the position re-score: each block at its best
layer in the span grid's summary (`20261029`). Frequency plays no part in
choosing a layer.

**Registered comparisons**, by §6's rule (a gap of at least 0.05 and a
paired word-type bootstrap 95% interval excluding zero; the pilot's 991-type
index and weights, seed 20260928). Every word type falls in exactly one band,
so the type bootstrap is unchanged. Band weights stay at the observed site
counts in every resample.

- **F1.** wav2vec2-base, ɾ span, layer 9: frequency-stratified balanced
  accuracy minus chance (0.5). Does the encoder's signal survive the
  frequency control?
- **F2.** wav2vec2-base's frequency-stratified balanced accuracy minus
  MFCC's, on the ɾ span. This is C5 under the frequency control: does the
  encoder still beat the representation that knows no words?

**Reported, not called:**

- F1 and F2 for WavJEPA and the other three spans; MFCC's stratified score
  against chance.
- Each band's own balanced accuracy, with /t/ and /d/ recall and type counts.
- The unstratified balanced accuracy on the same sites (all three lower
  bands pooled), so the stratification's effect can be seen.
- The stratified score within word-final and within word-medial flaps
  separately.
- /t/ recall in the top band.

**Already seen, so stated here:** the frequency counts above and in
`count_position_frequency.py`'s output (frequencies only, no scores). The
position re-score's reported type-weighted scores, which count every word
type once and so damp the most frequent types, were read: wav2vec2-base ɾ
drops from 0.607 to 0.572 word-medially and from 0.641 to 0.610
word-finally. No score by frequency band has been computed.

**Limits.** Bands are a decade wide, so some frequency remains within each;
the stratification reduces the confound and does not remove it. The top
band, where the confound is strongest, cannot be scored at all. The bands
also move with position (word-final sites sit mostly in the upper bands), so
the per-position scores below the top band rest on fewer sites.

**What each reading predicts.** If the encoder's signal is frequency-driven
reduction, F1 falls toward chance and F2 toward zero: under the control,
wav2vec2-base does no better than MFCC. If it is a trace of the contrast, F1
stays near the unstratified score and F2 stays positive.

**The author's prediction:** controlling for frequency will bring their
performances closer to chance. F1: I think it will land either closer to
chance or no longer reliably beat chance; I do think its score will drop
measurably. F2: I'm not sure it'll clearly beat MFCC because of that (MFCC
isn't that much better than chance to begin with).

The scoring script is `scripts/score_frequency.py`. It writes aggregates only,
to `results/span-grid/frequency-<date>.*`.

*Correction, 2026-11-21, before any score was read.* The claim above that
every word type falls in exactly one band is wrong. Frequency is counted per
word string, and the split's type key merges clitic and possessive forms
(*that* with *that's*, *it* with *it's*), so 15 type keys have sites in two
bands: 1,964 sites, of which 63 sit outside their key's main band. Nothing in
the method changes. The bootstrap does not need the claim: a resampled type
brings each of its sites into that site's own band. Frequency stays per word
string, as counted for the table. The scoring script stopped on its own check
of the claim before loading any prediction.

**2026-11-21 — A transfer probe: trained on unflapped stops, tested on
flaps, registered before any stop is extracted (§6, §7).** Every probe so
far has trained on flaps. This one trains /t/ against /d/ on stops that
were pronounced as stops, [t] and [d], where the flaps occur. It is then
scored on the flaps of word types and speakers it never trained on. If the
direction that separates a clear [t] from a clear [d] also separates the
flaps, the flap carries a residual of *that* contrast. That claim is
stronger than "something in the flap's frames predicts the label". A probe
that only learned which words go with which label has nothing to transfer.

**Training sites** (`scripts/count_transfer_candidates.py`, `6e5d077`;
counted 2026-11-21, with no representation loaded).

- **Each token** is a canonical /t/ or /d/ surfacing as the same stop,
  under `count_flap_sites`' token rules.
- **Its neighbours on the phone tier** must fall in §2's classes. Before
  it: a vowel, a sonorant coda or a syllabic consonant. After it: a vowel,
  a syllabic consonant or `hh`.
- **Its position must be attested:** the same word type at the same
  position in its citation form must be among the grid's 10,334 flap
  sites, and must flap in at least 40% of its tokens in this context.
- **Word-initial positions are out.**

Buckeye marks no stress, so the corpus supplies it. A position Buckeye
doesn't flap (*attack*) is out, which a phone-context rule alone would let
in (Bauer 2005, p. 31).

| Set | Stops | /d/ | Word types | Speakers with a /d/ | Grid flaps at a covered position |
|---|---|---|---|---|---|
| ≥ 0.40 (registered) | 2,247 | 1,011 (45.0%) | 252 | 40 | 8,916 (23.1% /d/) |
| ≥ 0.25 (sensitivity) | 2,433 | 1,162 (47.8%) | 303 | 40 | 9,030 |
| ≥ 0.50 (sensitivity) | 1,749 | 746 (42.7%) | 220 | 40 | 8,198 |

The threshold was chosen from the counts as the point where the set
stops being cheap to shrink. No threshold brings the set's /d/ share near
the flaps' (about 23%). The probe's balanced class weights absorb the
difference in training, and balanced accuracy on the flaps doesn't depend
on their class balance.

**Extraction:** the same as the pooled design's (§3, §4). Each stop is
encoded inside its own excerpt, the stop ± 200 ms. The same encoders
(wav2vec2-base, WavJEPA, MFCC), revisions and frame mapping are used. §2's
spans are defined with the stop in the flap's place:

- **stop** — the stop alone, closure and release as transcribed (the
  flap span's counterpart);
- **V+stop**, **stop+V** and **V+stop+V** — the same stretches as Vɾ, ɾV
  and VɾV.

**Split:** the joint split's 5 × 5 grid, unchanged. Every training
stop's word type is a flap type by construction, so it already has a type
group. Every stop's speaker has a speaker group. For each of the 25 test
cells, the probe trains on the stops sharing neither the cell's speakers
nor its word types, and is scored on the cell's flaps. Every grid flap is
scored once. Each fit trains on about 64% of the stops, roughly 1,440.

**Probe:** §5's probe, unchanged (L2 logistic regression, C = 1, balanced
class weights, standardized on each fit's training sites).

**Layers are fixed**, as in the two re-scores above: each representation at
its span-grid best layer for the matching flap span (wav2vec2-base 9;
WavJEPA 3, 4, 4, 6; MFCC its one). No layer is chosen on transfer scores.

**Registered comparisons**, by §6's rule (a gap of at least 0.05 and a
paired word-type bootstrap 95% interval excluding zero; the pilot's
991-type index and weights, seed 20260928), over all 10,334 grid flaps:

- **X1.** wav2vec2-base, stop → ɾ at layer 9: transfer balanced accuracy
  minus chance (0.5).
- **X2.** wav2vec2-base, V+stop → Vɾ at layer 9: the same. The asymmetry
  below makes this the span where a shared cue can exist.

**Reported, not called:**

- X1 and X2 for WavJEPA and MFCC, and for the stop+V and V+stop+V spans.
- Every layer's transfer curve.
- The sensitivity sets at 0.25 and 0.50, with the same calls' numbers.
- Transfer scored only on flaps at a covered position (8,916).
- The flap-trained probe's score beside each transfer score: the span
  grid's, from its cached predictions.
- The frequency re-score's stratified score, applied to the transfer
  predictions.
- /t/ and /d/ recall on the flaps.
- Each speaker group's and each word-type group's score.

**The asymmetry, stated in advance.** A clear stop's strongest cues are
closure voicing and the release burst. A flap has neither, so a probe
trained on stops may lean on cues the flap lacks, and fail to transfer
even if the flap keeps a trace. **A failed transfer is weak evidence
against a trace.** A successful one is strong evidence for one. The cue
the two do share is the length of the preceding vowel or sonorant
stretch, which is why X2 is called alongside X1.

**What the readings predict:**

- **Plain lexical shortcut** (the probe learns which words carry which
  label): no transfer. Stops and flaps in held-out words share no labels
  through memory.
- **A phonetic residual of the contrast** (incomplete neutralization):
  transfer, mainly in X2, where the stretch's length is shared.
- **Top-down** (the model recognizes the word, and its lexical knowledge
  pulls the flap toward where its stop lands, as listeners' lexical bias
  does): transfer, on X1 as well as X2.

So transfer separates a shortcut from the other two readings. It doesn't
separate those two from each other.

**Frequency.** The training stops' words and the flaps' words differ in
frequency the way the position groups did. The decided set drops the
word-initial *to*s, and the covered flap positions include the common
function words. The frequency re-score found frequency doesn't explain the
flap-trained score. The reported stratified score checks the same for
transfer.

**HuBERT and WavLM**, if they are extracted with these sites, get a dated
entry of their own before any of their scores is read.

**The author's prediction:** I think it'll do better than chance this
time. The vowel lengthening before voiced consonants (/d/) might be the
clue that transfers, rather than the stop's other markers.

The scoring script will be `scripts/score_transfer.py`, writing
aggregates only to `results/span-grid/transfer-<date>.*`.

---

**2026-11-23 — A context-only span, registered before any of its features is
extracted (§2, §6, §7).** A new span for the pooled design: the excerpt's
frames *outside* the flap. It was not in the original grid; the idea came
after the grid's results were read, from the leak-or-shortcut question
(2026-11-08): the labels are fully lexical, and every encoder hears the
neighbouring words. If the frames around the flap predict /t/ against /d/ as
well as the flap's own frames do, the flap's frames are not where the label
is read.

**What it can and cannot show.** In the pooled design the encoder hears
the whole excerpt (§3), so wav2vec 2.0's and WavJEPA's frames outside the
flap have attended to the flap. For them, the span says whether the label
can be read *away from the flap's frames*, not whether the context's own
audio carries it. MFCC frames do not attend: outside the flap they contain
only the flap's edges, through their 25 ms windows and ±40 ms deltas. MFCC
is therefore the representation in which this span answers whether the
context's audio carries the label (K3).

**Sites:** the grid's 10,334.

**Spans**, in each encoder's frames:

- **context** — every frame of the excerpt (the flap ± 200 ms) that is not
  in that encoder's ɾ span, by that encoder's own rule (§4): WavJEPA by
  onset, wav2vec 2.0 and MFCC by centre. The complement is exact, with no
  margin: a margin would remove frames whose windows overlap the flap
  slightly, while in wav2vec 2.0 and WavJEPA every frame has heard the flap
  anyway. Where wav2vec 2.0's ɾ span is the nearest-frame fallback (§4), that
  frame is the one left out. WavJEPA keeps only frames that read wholly real
  audio (§8, 2026-10-18).
- **preceding stretch, flap excluded** — Vɾ (§2) without the flap's frames:
  the vowel and any coda before it, clipped to the excerpt as Vɾ is.
  Reported, not called. It asks whether the stretch's own frames carry what
  Vɾ carries. Mean pooling discards duration (§4), so this span cannot see
  the stretch's length.

**Extraction:** the pooled design's (§3, §4), unchanged: the same excerpts,
revisions, float16 frames and mean pooling. Only the pooled spans were
stored, so the sites are re-encoded. The ɾ span is re-pooled in the same pass
and has to equal the stored ɾ span bit for bit, as the grid's extraction
checked against the pilot's.

**Probe, split, checks:** §5's, unchanged (joint split, the pilot's folds,
L2 logistic regression with balanced class weights). The best layer is
re-selected in every bootstrap resample, as for the grid's own spans (§6).

**Registered comparisons**, by §6's rule (a best-layer gap of at least 0.05
and a paired word-type bootstrap 95% lower bound above zero; the pilot's
991-type index and weights, seed 20260928):

- **K1.** wav2vec2-base, context: balanced accuracy minus chance (0.5). Is
  the label readable away from the flap's frames?
- **K2.** wav2vec2-base, ɾ minus context, paired. Do the flap's own frames
  carry more than the frames around them? §6's rule is one-sided (§8,
  2026-10-29), so only this direction is called. The gap's sign and interval
  are reported either way: context ahead of ɾ is a possible outcome (in the
  grid, every wider span's point value sits slightly above ɾ's, VɾV by
  +0.009, none an effect; C1, `20261029`).
- **K3.** MFCC, context: balanced accuracy minus chance. Does the context's
  own audio carry the label?

**How they read together.** No effect on K2 is not equality: it says only
that a gap of 0.05 was not shown.

| K1 | K2 | Reading |
|---|---|---|
| effect | effect | Readable around the flap, more so at it: concentrated at the flap |
| effect | no effect | The flap's frames add nothing measurable over the frames around them |
| no effect | effect | Only the flap's frames carry it |
| no effect | no effect | Unsettled |

K3 is read separately: an effect says the neighbouring audio itself carries
the label, which the K1–K2 table cannot say.

**Reported, not called:**

- K1 and K3's numbers for WavJEPA; K2 for WavJEPA and MFCC.
- The preceding stretch, flap excluded, for all three representations, and
  its gap to Vɾ.
- Every layer's curve.
- §5's checks per span: the control task and the lemma-unseen re-score.
- Each span on unclipped sites only (§7).
- The frequency re-score's stratified score, applied to the context
  predictions.

**A flap-masked variant, planned, not registered here.** The clean test of
whether the context alone carries the label is to encode the excerpt with
the flap replaced by noise, so that no frame hears it. It needs §3's padding
machinery (pause-matched noise, ramps, normalization on the full excerpt),
which the cropped design builds. It runs with the cropped design and gets a
dated entry of its own before it is extracted.

**HuBERT and WavLM**, if they are extracted with these sites, get a dated
entry of their own before any of their scores is read.

**The author's prediction:** I do overall think that the context carries
the label. I would expect the flap frames to carry some of the context,
since their head and tail ends aren't a clean dissection of the flap, and
from the context mixing after attention. So realistically I expect similar
performance (K2 no effect). With the cropped design it might be a cleaner difference.

The scoring script will be `scripts/score_context.py`, writing aggregates
only to `results/span-grid/context-<date>.*`.

---

**2026-11-24 — HuBERT and WavLM, registered before any of their scores is
read (§4, §6, §7).** Two more encoders for the pooled design:
`facebook/hubert-base-ls960` at `dba3bb0` and `microsoft/wavlm-base` at
`efa81aa`. Like wav2vec2-base, both are self-supervised on LibriSpeech's
960 h without transcripts. Their objectives differ: wav2vec 2.0 recovers
its own quantized local features, HuBERT predicts cluster IDs taken from an
intermediate layer of an earlier HuBERT, and WavLM predicts HuBERT's cluster
IDs from input augmented with noise and overlapping speech. WavJEPA differs
in both data (AudioSet) and objective. If HuBERT and WavLM land near
wav2vec2-base, WavJEPA is the outlier among these encoders. If they spread
out, C7's gap belongs to checkpoints, not to JEPA against the rest.

Their features were extracted before this entry was written, as the entries
of 2026-11-21 and 2026-11-23 allow, by `scripts/extract_conv_encoders.py`
(`e40c225`). No score has been computed.

**Frames:** wav2vec 2.0's rule (§4), by centre with the nearest-frame
fallback. Both checkpoints have wav2vec2-base's conv stack and 13 × 768
states (configs compared; `tests/test_conv_encoders_model.py`), so the
geometry is the same. The extraction re-pooled wav2vec2-base through the same
loader, and every span matched the stored features bit for bit.

**Input normalization: all three normalized per excerpt, as the registered
wav2vec2 run is.** The checkpoints' own feature extractors disagree:
wav2vec2's and HuBERT's normalize, WavLM's does not. fairseq pretrained
wav2vec2-base and HuBERT-base on raw waveforms (`normalize: false` in both
pretraining configs). The choice is not small on quiet input: conv layer 0's
GroupNorm absorbs gain only while the variance is well above its eps term. On
noise at gain 0.01, normalizing moved the layer-12 states of all three
encoders by a relative difference of 0.2 to 1.0. Normalizing all three keeps
their input identical, so a gap between them belongs to the checkpoint.
**A raw-input variant of all three, wav2vec2-base included, was extracted
beside it. It is reported, not called.**

**Spans:** §2's four; the context-only span and the preceding stretch without
the flap (§8, 2026-11-23); the transfer probe's stops (§8, 2026-11-21).

**Probe, split, checks:** §5's, unchanged.

**Registered comparisons**, by §6's rule (a best-layer gap of at least 0.05
and a paired word-type bootstrap 95% lower bound above zero, the best layer
re-selected in every resample):

- **E1.** Each span, HuBERT − WavJEPA and WavLM − WavJEPA. Is WavJEPA behind
  every LibriSpeech encoder, not only wav2vec2-base? This direction is
  called because C7's reported gap (§8, 2026-10-29) put wav2vec2-base ahead,
  and that result was read before this entry.
- **E2.** Each span, HuBERT − MFCC and WavLM − MFCC: C5's first half for the
  new encoders.
- **C6 for HuBERT and WavLM.** The best layer per span, with its interval, as
  §6 has it for the first two encoders: no threshold. For comparison, the
  grid's curves: wav2vec2-base is flat to layer 5, peaks at 9 on every span
  and falls back by 11–12. WavJEPA steps up by layers 3–4 and stays on a
  plateau, its best layer between 3 and 6 with wide intervals (`20261029`).

**Reported, not called:**

- The three LibriSpeech encoders against each other, per span, with paired
  intervals in both directions. Their spread is the size of a checkpoint
  effect.
- C1's gaps, C4 and C5's second half for HuBERT and WavLM.
- K1 to K3's numbers for HuBERT and WavLM, and the transfer probe's.
- The raw-input variant of all three, and its gap to the normalized
  features.
- §5's checks per span.

**The author's predictions:**

- **E1.** I think WavJEPA's performance will be worse than the librispeech
  ones.
- **E2.** I think they will beat mfcc like wav2vec2-base did.
- **C6.** I think there will be a similar pattern between hubert and wavlm,
  but that shared pattern will be different from both wavjepa and
  wav2vec2-base (based on similarities observed in Pasad et al. 2023, *What
  Do Self-Supervised Speech Models Know About Words?*). The pattern meant is
  the layer pattern.
- **Score level among the three LibriSpeech encoders:** no prediction.

The paper's finding behind the C6 prediction (§5.2.1): models trained to
recover local features (wav2vec 2.0) correlate best with word-level
properties at central layers, 5–7 for Base, and models trained on discrete
units from an intermediate layer (HuBERT, WavLM) at much higher layers. Its
probe is word-level; this one is a phone-level contrast.

The scores join `scripts/score_span_grid.py`'s output, written to
`results/span-grid/` as aggregates only.

---

