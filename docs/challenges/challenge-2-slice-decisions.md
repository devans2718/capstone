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

**Reviewer** blocks are a second opinion, added 2026-09-19 after reading the
branch. They sit above the checkboxes deliberately: they are input to the
decision, not the decision. The **Decision** and **Notes** lines stay empty
until you fill them.

> ⏱ **Sequencing, before any of the substance.** D1–D5 are research-track
> decisions and **none of them blocks the Challenge 2 demo** — the slice runs
> on synthetic predictions, and `n_frame_onsets` is a column the app
> *displays*, not one it computes against. What they block is pre-registration
> revision 3. **Answer D6 and D7 now; D1–D5 can wait until the app is
> standing.** They will be no harder Tuesday, and the measurement that
> produced them is already written down.

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

**Reviewer — (a).** This is less of a judgment call than the framing above
suggests, because **the test suite already contains the docstring contract.**
`test_excerpt_local_times_round_trip`, immediately above the test you flagged
(`tests/test_extraction.py:78-81`), asserts
`target_start_local_ms <= onset < target_end_local_ms` for *every* frame in
the returned range. That is the docstring, written as an assertion. It passes
today only because `plan_excerpt` hands it a phase-zero excerpt; at a generic
phase the implementation returns a leading frame whose onset precedes `start`
and the assertion fails.

So D1 is not docstring-versus-implementation. It is **test-versus-test**, and
the two cannot both survive: one encodes the contract and passes by the same
coincidence that hides the bug, the other pins the arithmetic. Given that the
excluded frame is the one whose receptive field *begins* at the flap onset —
the best-aligned frame in the span — option (b) requires writing down a reason
to drop the best frame. No such reason exists, which is the tell.

**Decision:** (a), 2026-09-24. Fix the implementation.

**Notes:** `frames_for_interval` now uses the first onset at or after each end.
The pinned test was replaced by one asserting the centred 6 ms target keeps
its onset frame, plus two covering the cases that can still be empty or were
wrong at a generic phase. Rebuilt inventory: excerpt-local total **33,079**
onsets, 0 sites with none. That was checked by brute-force enumeration against
every site, with 0 mismatches. It is not the 33,216 estimated above; the
brute-force figure is the definition. The absolute timebase moved too, from
28,392 to 28,382 total, still 4 sites with none, because slack-sample handling
changed with it.

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

**Reviewer — (a).** The §7 precedent for revision 1 → 2 fits this exactly:
nothing has been run, and the change came from measuring the corpus rather
than from seeing a result. One revision, one push.

One amendment to how it gets written. Make the §7 entry state explicitly that
the exclusion set became **empty**, rather than just deleting the sentence. A
silently removed exclusion reads like an oversight to anyone auditing later;
"this exclusion no longer selects anything, and here is the measurement that
emptied it" reads like the finding it actually is.

**Decision:** (a), 2026-09-24. Folded into revision 3, which is still in
progress and not yet pushed.

**Notes:** Per the reviewer's amendment, the pre-registration says the
no-onset exclusion now selects nothing and cites the measurement that emptied
it, rather than dropping the sentence.

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

**Reviewer — (a), and fold D5's column into the same regeneration.** The
`results/corpus-verification/` dating convention exists for published reports
that other documents cite. This parquet is an intermediate that nothing
consumes yet, so the argument for (b) does not reach it.

One caveat on timing: `sites-20260919.txt` moves with the parquet, and if D1
lands after today the filename and its contents disagree about the date. If
this slips past tonight, use a fresh date for the corrected pair rather than
regenerating `20260919` on the 20th.

**Decision:** (a) with the reviewer's fresh-date caveat, 2026-09-24. Claude
applied it under the go-ahead to fix D1, and it was confirmed the same day.
The corrected inventory is `sites-20260924.{parquet,stats.json,txt}`, and the
`20260919` trio is removed from the tree and kept in git history. D5's column
was added in a second rebuild the same day, once D5 was decided.

**Notes:** Only `n_frame_onsets` and `n_frame_onsets_abs` changed. Every
other column is identical, and the 2026-08-21 TSV regenerates byte-identical.

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

**Reviewer — (b), narrowly, and it is low stakes either way.** (a) is
defensible. But a one-line dated pointer at the top costs nothing and helps
whoever opens the `.txt` directly rather than arriving at it through the
newer report — which is the same discipline as a dated supersession header.
It changes no reported figure, so the pre-registration's citation stays
honest.

**Decision:** (b), 2026-09-24.

**Notes:** Added a six-line dated note above the report's header, pointing to
`results/flapping/sites-20260924.txt`. Nothing below it was edited. D1's fix
also moved the absolute-timebase total itself, from 28,392 to 28,382, because
slack samples now resolve forward. The published figure is therefore wrong
under the current code as well as describing the wrong timebase. The note
says it is superseded rather than quoting the new absolute number, since
nothing downstream uses that timebase.

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

**Reviewer — (b) plus (a); not (c) yet.** The derived column is additive in
exactly the way `dfbd487` established, and folding it into D3's regeneration
makes it free. It also keeps (c) available later without recomputing, which
is the whole argument for doing it now rather than deciding the research
question now.

Hold (c) back. Carving a named evaluation subset is a research decision, and
revision 3 is already carrying the thresholds and the exclusion change; a
third item on the same push is how one of them gets underthought.

⚠ **The caveat is the important half of this entry and should travel with the
number wherever it goes.** 30.53% neutralised is a claim about Buckeye's
label granularity, not about the audio. If the figure reaches a slide without
the caveat it becomes "a third of the data is ambiguous," which is a
different statement and a false one.

**Decision:** (b) + (a), 2026-09-24. Not (c) yet.

**Notes:** `surface_ambiguous` (bool) is appended to
`sites-20260924.parquet` by `mark_surface_ambiguous` in
`scripts/count_flap_sites.py`: 3,173 sites (2,601 /t/, 572 /d/) in 59 cells,
matching the figures above. All existing columns are unchanged, the TSV is
still byte-identical, and `n_surface_ambiguous` is in the stats JSON. Recorded
in pre-registration §1a with the caveat attached.

---

## D6 — Step 2 scope

Brief §5 specifies `predictions.parquet` under `/data/embeddings/` and the run
manifest. Nothing found in step 1 contradicts it.

- [ ] Proceed exactly as §5 specifies.
- [ ] Changes first —

**Reviewer — proceed as §5**. One of the two to answer today.

Nothing step 1 turned up moves the step 2 spec, and the reason is worth stating rather than asserting: **predictions are keyed on `site_id`, and nothing in D1–D5 touches it**. `site_id` is built from the phone's ordinal in its track's `.phones` tier, so it is stable across any regeneration of the inventory. D1 and D3 change the value of `n_frame_onsets` and rewrite the parquet; D5(b) appends a column. Under all of them the join key is unchanged, which means predictions generated today still join correctly against whatever the inventory becomes.

That is what makes deferring D1–D5 safe — not that they are research-track and can wait, but that step 2's output survives every resolution of them.

**Decision:** Agreed.

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

**Reviewer — agreed: `--no-ff` with a body, branch kept, no PR.** The other
half of what to answer today.

I considered whether holding until D1 resolves keeps a known-wrong column off
`main`, and it does not apply. `dfbd487`'s design decision was to keep *both*
columns precisely so the discrepancy is visible; `main` carrying both is not
carrying an error, it is carrying the record. Holding the branch would
actually suppress the finding until the fix lands, which is backwards.

**Decision:** Agreed.

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
