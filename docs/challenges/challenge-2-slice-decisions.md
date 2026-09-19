# Open decisions — Challenge 2 slice, after step 1

| | |
|---|---|
| **Status** | Questionnaire. Questions committed before answers, same as the brief. |
| **Date** | 2026-09-19 |
| **Raised by** | `dfbd487` (site inventory extension) and what measuring it turned up |
| **Blocks** | D1–D3 block step 2 in one direction only; see each entry |

Fill in the **Decision** line under each item. Answers get committed on top of
this file, so the questions keep their own timestamp.

Anything already settled is in *Not asking about* at the bottom, so this file
is a list of live choices rather than a status report.

---

## D1 — Is `frames_for_interval` off by one, or is its docstring wrong?

**The disagreement.** `FrameGeometry.frames_for_interval` says it returns
"Frames whose onsets fall in `[start_ms, end_ms)`". It computes
`range(frame_at(start), frame_at(end))`, where `frame_at(x)` is the last frame
whose onset is ≤ x. At a generic phase that makes two errors which cancel: it
includes a frame whose onset precedes `start`, and excludes the one preceding
`end`. At phase zero — `start` exactly on a frame onset — the first error
disappears and the second does not.

**Why every site hits it.** `plan_excerpt` centres on `context_ms = 200`, which
is exactly 3,200 samples, which is exactly 20 frame steps at a 160-sample
stride. So in excerpt-local coordinates every target starts exactly on an
onset. The error never cancels and never varies.

```
excerpt-local, as frames_for_interval computes it : 24,016 onsets,  39 sites with none
excerpt-local, as its docstring defines it        : 33,216 onsets,   0 sites with none
```

9,200 of 10,392 sites are short by exactly one frame. The remaining 1,192 have
durations that are exact multiples of 10 ms, so `end` lands on an onset too and
the errors cancel again. The missing frame is the one whose receptive field
begins at the flap onset — the best-aligned frame in the span.

**This is not a Challenge 2 question.** It sits in
`src/capstone/encoders/wavjepa.py`, upstream of the slice, and it reaches the
pilot rather than the demo. Nothing has been changed.

- [ ] **(a)** The docstring is the contract. Fix the implementation.
      `tests/test_extraction.py:85` (`test_very_short_target_can_yield_no_frames`)
      pins the current behaviour and would change with it.
- [ ] **(b)** The implementation is the contract. Fix the docstring, and record
      why a frame whose onset coincides with the target's start should not
      count as inside it.
- [ ] **(c)** Defer. Keep both columns, decide before the first probe is
      trained. Costs nothing now; costs a re-extraction if decided after
      embeddings exist.

**Decision:**

**Notes:**

---

## D2 — If D1 is (a), does the pre-registration take a revision or a deviation?

§1 excludes "The 4 sites containing no frame onset at all." Under (a) that set
is **empty** — no site has zero onsets, because every site's own starting frame
counts. §2's "41% have ≤2 onsets" also moves.

§7 says any departure is appended with a date and a reason and the original
text stays. But §7 also records that revision 1 → 2 was *not* a deviation,
because nothing had been run and the change came from measuring the corpus
rather than from seeing a result. That description fits this exactly.

A revision is already owed regardless: §5 marks three thresholds *provisional*
and says they must be finalized before the first probe, in one further
revision, or the pre-registration is void.

- [ ] **(a)** Fold into the owed revision 3, described as measurement-driven,
      not a deviation. One revision, one push, before anything is trained.
- [ ] **(b)** Revision 3 for the thresholds; a separate §7 deviation entry for
      the exclusion change, on the stricter reading.
- [ ] **(c)** Something else —

**Decision:**

**Notes:**

---

## D3 — Does `sites-20260919.parquet` get regenerated?

`n_frame_onsets` currently holds the undercounted value, and the file is
committed and pushed. Nothing consumes it yet.

- [ ] **(a)** Regenerate in place once D1 lands. Same filename, corrected
      column; the git history carries the before/after.
- [ ] **(b)** New dated file, leave `20260919` as the artifact that matches
      what the report says. Matches how `results/corpus-verification/` treats
      dated outputs.
- [ ] **(c)** Leave it; correct at step 2 when predictions are generated
      against it.

**Decision:**

**Notes:**

---

## D4 — What happens to the published 28,392?

`results/corpus-verification/buckeye-flap-sites-20260821.txt` reports "total
frames over all sites 28,392". That figure is correct for the recording's
timebase and describes a coordinate system nothing downstream uses. The
pre-registration cites the file.

Invariant 3 freezes the `.tsv`, not the `.txt`.

- [ ] **(a)** Leave it untouched as the historical artifact; the 2026-09-19
      report supersedes it and says so.
- [ ] **(b)** Append a dated correction note to the `.txt` pointing at the
      newer figures.
- [ ] **(c)** Regenerate it.

**Decision:**

**Notes:**

---

## D5 — Flap-induced neutralisation: 30.5% of the inventory

Not in any design document, found while answering a question about D‑1's
sibling measurement. Distinct from pre-registration §1a, which asked whether a
single *spelling* carries both classes at one position (it never does — all 25
both-class types are positionally determined).

This is the across-spelling case: 59 (surface phone string, position) cells are
attested as both /t/ and /d/, covering **3,173 sites, 30.53%** of the
inventory. `got ~ god`, `what ~ would`, `right ~ write ~ ride`,
`said ~ set ~ sit`. Your literal two-flap swapped case
(/XdXXXtX/ vs /XtXXXdX/) does **not** occur: of 168 two-flap sites, zero
surface strings have even one ambiguous flap position.

Caveat that cuts the other way: Buckeye's labels are coarse, so `[r ay dx]` is
one string for *write* and *ride* while hiding the pre-fortis vowel-length
difference — which is plausibly the cue that makes the pair separable. These
sites are neutralised in the transcription, not necessarily in the audio, which
is what makes them interesting rather than merely hard.

- [ ] **(a)** Record in the pre-registration as a known population; no code
      change now.
- [ ] **(b)** Add a derived column to the inventory (`surface_ambiguous`) so it
      can be filtered on without recomputing.
- [ ] **(c)** Carve out as a named evaluation subset — the sites where
      segmental identity cannot answer and a durational cue must.
- [ ] **(d)** Note here and defer entirely.

**Decision:**

**Notes:**

---

## D6 — Step 2 scope

Brief §5 specifies `predictions.parquet` under `/data/embeddings/` and the run
manifest. Nothing found in step 1 contradicts it.

- [ ] Proceed exactly as §5 specifies.
- [ ] Changes first —

**Decision:**

**Notes:**

---

## D7 — Merging this branch

Six prior branches were merged with a plain `Merge branch 'x'` and kept; there
are zero GitHub PR merges in this history. Recommendation is to stay
consistent: no PR, `--no-ff` merge with a substantive body, branch retained.
The off-by-one finding wants a durable home, and the merge body is where this
repo already keeps that kind of thing.

- [ ] Merge to `main` with a written body, keep the branch. **(recommended)**
- [ ] Open a PR on GitHub instead.
- [ ] Hold the branch unmerged until D1 is resolved.

**Decision:**

**Notes:**

---

## Not asking about

Settled, recorded so the list above stays short.

- **`site_id` format.** `track:%05d` on the phone's tier ordinal. Max observed
  index is 6,493 across 254 tracks, so five digits leaves 93,506 of headroom
  and lexical sort matches numeric sort. Unique across all 10,392 rows.
- **Both onset columns stay.** `n_frame_onsets` (excerpt-local) and
  `n_frame_onsets_abs` cost two int32 columns and are the only way a reader can
  see D1 exists at all.
- **The committed TSV.** Byte-identical, verified; invariant 3 holds whatever
  D1 decides, because the seven-column writer does not touch any of this.
- **The pre-commit hook.** Already active via `core.hooksPath=ops/hooks`, with
  `core.fileMode=false`. Verified it refuses a shebang file staged 100644.
- **`requirements-app.txt` extras.** `pytest` and `-e .` beyond brief §11's
  five, because without them the test suite does not collect.
