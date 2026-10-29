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

---

