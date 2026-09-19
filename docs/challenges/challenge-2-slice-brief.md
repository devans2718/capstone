# Challenge 2 — implementation brief for the first vertical slice

| | |
|---|---|
| **Status** | Specification. Written before implementation; see *Provenance* below. |
| **Date** | 2026-09-19 |
| **Branch** | `challenge-2-slice`, branched from `talk3-realization-inventory`, current with `main` |
| **Implements** | `docs/challenges/challenge-2-ai-design.md` (AI Design, v2, 2026-09-16) |
| **Deliverable deadline** | 2026-09-21 |

## Provenance

This document was written and committed **before any implementation code**, and is
the specification the slice was built against rather than a description written
afterward. That ordering is deliberate and follows the same principle as
`experiments/pilot-flapping/README.md`, which was pushed to `origin` before any
probe was run: a specification whose priority is established by commit history is
checkable, where one reconstructed after the fact is not.

It also records the two points at which the AI Design, taken literally, would have
produced the wrong build (§3). Those are stated here rather than silently corrected
in the code.

---

## 1. Scope

Build one vertical slice of the project's first P1 task:

```
User -> Streamlit -> DuckDB tools / audio export -> result + evidence -> human review
```

The slice establishes the join spine that later retrieval work sits on. It is
explicitly **not** the full system, and five designed components are deferred (§10).

### 1.1 No probe is trained

The pre-registration at `experiments/pilot-flapping/README.md` finalizes three
provisional thresholds in a revision that is not yet written, and the file states
that running before that revision is pushed voids the pre-registration.

The slice therefore runs on **synthetic predictions**, marked as such throughout.
This costs nothing that matters: the retrieval contract is exactly as testable
against synthetic predictions as against real ones, because retrieval does not
depend on whether a probability is meaningful, only on whether it round-trips with
its provenance intact.

### 1.2 Data scale

The site inventory remains **10,392**. The number of *scored* sites in this run is
**40**, per §3.1.

---

## 2. Invariants

Violating any of these is worse than delivering less functionality.

1. **`scripts/count_t_realizations.py` imports `DEFAULT_ROOT`, `phones_in`,
   `read_phones` and `read_words` from `scripts/count_flap_sites.py`, and asserts
   `EXPECTED_T_FLAPS = 7863`.** Those four signatures and the existing fields of
   `Phone` are frozen. New fields are added with defaults. Run
   `count_t_realizations.py` after any change to the inventory and confirm it still
   reproduces 7,863.
2. **The site count remains exactly 10,392** — 7,863 underlying /t/ and 2,529 /d/.
   A changed count means a selection rule moved, which invalidates the corpus
   figures cited in the approved proposal and in the Challenge 2 Human Design.
3. **`results/corpus-verification/buckeye-flap-sites-20260821.tsv` keeps its exact
   seven columns in their exact order.** Parquet is written alongside it, never in
   place of it.
4. **`export_clip` calls `plan_excerpt`; it does not re-derive the window.**
   Measured positional sensitivity is cosine distance 0.394 at layer 0 for a
   10-frame shift, so a clip misaligned by 10 ms makes any conclusion drawn by ear
   unfounded.
5. **`synthetic` is a four-place invariant** — the run manifest, every prediction
   row, the rendered uncertainty string, and a persistent banner in the user
   interface. The renderer must not be able to omit it.
6. **No corpus audio enters the repository.** Exported clips are written under
   `/data`. Buckeye is distributed under a registration agreement with OSU;
   `.gitignore` already excludes `*.wav` outside `stimuli/`, and that exclusion is
   load-bearing rather than incidental.
7. **No float participates in a join.** `site_id` is the join key and
   `start_sample` is the audio anchor; both are integers.

---

## 3. Two corrections to the AI Design

Both are cases where implementing the design as written would produce a build that
fails a stated requirement.

### 3.1 Scope: 40 sites, not 10,392

The AI Design, §9 step 3, specifies synthetic predictions for "all 10,392 sites in
each" condition.

**This is superseded.** The Challenge 2 lecture deck, slide 19, sizes the first task
at "~20–50 flap sites, one evidence object per site linking audio, transcript,
timestamp, speaker, and model result," and the Human Design was returned with the
comment that the first P1 task could be narrower and more specific.

**Resolution: score 40 sites.** The inventory is unchanged at 10,392; `n_scored` is
40.

This strengthens rather than weakens the coverage mechanism. Coverage then reports
`n_total: 10392, n_scored: 40`, which is a genuine partial denominator — the
condition the mechanism exists to make visible. A run that scores every site cannot
demonstrate it.

A second consequence is that human review becomes honest. A `label` verdict requires
listening to the clip; 40 sites is an evening's work and 10,392 is not something a
person does. The review step is only meaningful at this scale.

### 3.2 The failure case has no reachable target

The AI Design, §8, designates evaluation question 5 — a site "not scored, discarded
by the split" — as the failure case, and the assignment requires at least one
difficult or failure case.

**As specified, it cannot be tested.** §9 step 3 generates only the `ordinary` and
`word_disjoint` conditions, and both score all 10,392 sites out-of-fold. The state
`eligible = false` therefore never occurs anywhere in the slice.

**Resolution: generate a third condition, `combined_diagonal`**, over the same 40
sites, with approximately 80% of sites `eligible = false`. This matches the measured
discard of the diagonal fold scheme, under which only 2,104 of 10,392 sites are
scored.

This also makes the three-state eligibility field real rather than nominal. All
three states become reachable in the slice:

| State | Meaning | Reached by |
|---|---|---|
| `eligible = false` | discarded by the split; no prediction exists | off-diagonal sites under `combined_diagonal` |
| `eligible, scored` | a prediction exists | the 40 scored sites |
| `eligible, not scored` | run incomplete | the remaining 10,352 sites under `ordinary` |

Rendering these three distinctly is the point of the field. Collapsing any two of
them into "no data" is the defect test case 4 exists to catch.

---

## 4. Step 1 — extend the site inventory

Independent of Challenge 2, this step converts an inference in the pre-registration
into a measurement: whether the 25 word types carrying both underlying classes are
words with two flap sites (*dated*, *hated*, *dedicated*) or genuinely ambiguous
types. It should be committed on its own merits.

`Phone` gains `index: int = -1`, the phone's ordinal in the track's `.phones` tier —
its position in the list returned by `read_phones`.

`Site` gains the following, all with defaults:

| Field | Derivation | Purpose |
|---|---|---|
| `site_id` | `f"{track}:{phone.index:05d}"` | stable join key. `start` is a float accumulated through a text round-trip and must not be keyed on |
| `phone_idx_in_word` | the `idx` already computed in the `zip(canon, actual)` loop and currently discarded | the measurement above |
| `n_flaps_in_word` | count of `dx` in `actual` | disambiguates two sites within one word |
| `start_sample`, `end_sample` | `round(start * 16000)` | integer audio anchor |
| `duration_ms` | derived | |
| `word_start_sample`, `word_end_sample` | from the `.words` interval | |
| `canonical_phones`, `actual_phones` | retained as lists | joining them into strings destroys the label, which is defined positionally between the two tiers |
| `prev_phone`, `next_phone` | neighbours in the track-level phone tier, crossing word boundaries | supports conditioning analyses over the following segment |
| `next_word`, `next_word_first_phone` | lookahead in `.words` | connects to the word-final conditioning result in `8c8e4f9` |
| `annotator_mark` | `phone.mark` | empty for all 10,392 retained sites; 754 further sites exist under `--keep-marked` |
| `n_frame_onsets` | `FrameGeometry.frames_for_interval` | see §4.1 |
| `excerpt_clamped` | `plan_excerpt(...).clamped` at `context_ms = 200` | sites at a recording edge cannot be centred |

Recording duration is read from the `.wav` header; the standard-library `wave`
module is sufficient and adds no dependency.

`read_words` retains its current signature as a generator (invariant 1). Lookahead
is obtained by materializing it per track inside `collect_sites`.

### 4.1 Frame onsets must be computed in excerpt-local coordinates

The existing report computes frame onsets from the site's position in the
recording's own timebase. What the extraction pipeline indexes is the *excerpt-local*
span, because `encode_excerpt` lays the segment grid from the excerpt's first
sample. These can disagree: the frame geometry is a staircase with a 79-sample
interval at the end of each 2.01 s segment that no frame reads.

Compute the excerpt-local value, and report both. The known figures in the absolute
timebase are 4 zero-onset sites and 28,392 onsets in total. If the excerpt-local
count differs, that is a result to record, not a discrepancy to reconcile away.

### 4.2 Output and acceptance

Add a `--parquet PATH` option writing `results/flapping/sites-20260919.parquet`. The
existing `--dump` TSV path is unchanged.

Do not proceed to step 2 until all four hold:

- total is 10,392, with 7,863 /t/ and 2,529 /d/
- `count_t_realizations.py` reproduces 7,863
- the committed TSV is byte-identical
- both frame-onset counts are reported

---

## 5. Step 2 — prediction schema and run manifest

`predictions.parquet` is written under `/data/embeddings/`, not into the repository.

| Column | Type | Notes |
|---|---|---|
| `site_id` | str | joins `sites.parquet` |
| `encoder` | str | `wavjepa`; the lookup baseline will use `lookup` |
| `layer` | int | nullable, for encoders without layers |
| `condition` | str | `ordinary`, `word_disjoint`, `combined_diagonal` |
| `fold` | int | |
| `eligible` | bool | false = discarded by the split |
| `scored` | bool | false = eligible but not scored in this run |
| `p_t` | float | **nullable; null whenever `scored` is false** |
| `pred` | str | `t` or `d`; nullable |
| `correct` | bool | nullable |
| `synthetic` | bool | true for this run |

`p_t` must be null rather than `0.0` when a site is unscored. A zero is a number,
and a number will be rendered, aggregated and believed.

The run manifest is written as JSON beside the parquet file and records `run_id`,
`inventory_hash` (of `sites.parquet`), `code_commit`, `checkpoint_revision`
(`6be4a50`), `context_ms` (200), `split_scheme`, `random_seed`, `n_sites_scored`,
and `synthetic: true`. Enforced refusal on hash mismatch is deferred (§10); this
step records the fields.

---

## 6. Step 3 — synthetic prediction cell

WavJEPA, layer 8, 40 sites, three conditions per §3.2.

- Draw so that a plausible number of sites change correctness between `ordinary`
  and `word_disjoint`, so the differential question in §7 has a non-empty answer.
  Sites correct under `ordinary` and incorrect under `word_disjoint` should be the
  larger group, that being the direction a lexical shortcut predicts.
- Seed the generator and record the seed in the manifest. A synthetic result that
  cannot be regenerated is worse than none.
- Select the 40 sites deliberately to include at least one zero-onset site, one
  clamped site, and both underlying classes. Edge cases belong inside the
  demonstration set, not merely reachable elsewhere in the inventory.

---

## 7. Step 4 — retrieval tools

Typed, deterministic, no language model inside. DuckDB over the two parquet files.

| Tool | Returns |
|---|---|
| `query_sites(filters)` | scored-observation rows and a coverage block |
| `compare_conditions(cond_a, cond_b, ...)` | sites whose correctness changed, with both probabilities |
| `export_clip(site_id, context_ms=200)` | path to a wav file |

A coverage block is returned on **every** response, including comparisons where
denominators are equal and the block is uninformative. It costs one dictionary and
it is the only mechanism that makes a wrong denominator visible rather than silent:

```json
{"n_total": 10392, "n_eligible": 10392, "n_scored": 40, "compared_on": 40}
```

`compare_conditions` intersects on commonly-scored sites by default and reports that
it has done so.

`export_clip` must reproduce the sample arithmetic in `encode_excerpt` exactly:

```python
start = int(round(excerpt.offset_ms / 1000 * sample_rate))
end   = start + int(round(excerpt.duration_ms / 1000 * sample_rate))
```

Deriving `end` from `target_end_ms + context_ms` rounds differently and lands one
sample away. Reuse these lines rather than restating them.

### 7.1 Division of labour

Tools compute. Retrieval returns rows. A language model does exactly two things:
translate a question into a typed tool call, and narrate returned rows in prose
where every number is traceable to a returned field.

It does not compute a statistic, infer a label, decide which sites are interesting
without an `n`, or assert a cause. If the tool layer returns no rows, the answer is
"no rows," not a hypothesis. This constraint is what makes the output citable in the
same sense as the reports in `results/corpus-verification/`.

---

## 8. Step 5 — Streamlit application

A presentation layer over working parts. Functionality over visual complexity.

| Element | Requirement it satisfies |
|---|---|
| Filters: encoder, layer, condition, underlying class | human control of the query |
| Evidence table with the coverage block rendered, not hidden | provenance |
| Audio player per site | the evidence must remain audio; a transcript is not a substitute |
| Two separately labelled uncertainty sections | see §8.2 |
| Persistent `SYNTHETIC` banner | invariant 5 |
| Review control writing `reviewed.jsonl` | human review |

### 8.1 Review is a four-way verdict, not accept/reject

On a synthetic run the probe output carries no information, so "accept this answer"
is meaningless, and an interface inviting it trains the reviewer to approve noise.
Every verdict names which of four things it judges:

| Target | Question the reviewer answers | Judged against |
|---|---|---|
| `label` | Is this a flap, and is the underlying /t/ vs /d/ assignment correct? | the audio, by ear, against `word` and the canonical/actual strings |
| `clip` | Does the exported audio contain the site, centred and uncut? | the clip against `start_sample` and `duration_ms` |
| `retrieval` | Is this row responsive to the question asked? | the row against the filter and the coverage block |
| `grounding` | Did the prose stay within the returned fields? | the narration against the evidence |

Each line carries `inventory_hash`: a verdict against a site is a verdict against a
version of that site, and regenerating the inventory under `--keep-marked` adds 754
sites. The hash is what determines whether a prior verdict still applies.

`label` verdicts are a reusable result rather than a demonstration artifact. They
constitute a hand-checked subset of the inventory and supply the pre-registration's
label-uncertainty channel with measurements in place of estimates.

### 8.2 Two uncertainty channels, never merged

| Channel | Status in this run | Content |
|---|---|---|
| Label and population | **real, measured, citable today** | 2,986 of 14,238 flap-bearing words (21.0%) excluded for canonical/actual length mismatch; 754 annotator-marked flaps; 198 flaps not from /t/ or /d/; 4 sites with zero frame onsets; 3 clamped at 200 ms |
| Probe | **synthetic; carries no information** | `p_t` and margin exist in the schema and are labelled synthetic wherever rendered |

These are never combined into a single confidence figure. They have different
provenance and different remedies — the first is addressed by better alignment or by
recovering the 21%, the second by more data or a better probe — and a blended number
would make both unfixable, because it would no longer be possible to tell which had
moved.

The first channel carries a consequence that should be stated plainly rather than
buried: **the retained set is the cleanly-aligned set.** The 21% excluded are the
most reduced and most coarticulated tokens, which is precisely where flapping is
most interesting. No probe result on the remaining 10,392 sites can speak to them,
however narrow its confidence interval. That is a bound on what any result can mean,
and it is knowable now, before any result exists.

---

## 9. Step 6 — test cases

Ground truth is computed independently in pandas within each test, so each case is a
comparison rather than a judgment. Synthetic predictions are sufficient for all of
them.

| # | Case | Input | Expected |
|--:|---|---|---|
| 1 | Basic filter | misclassified sites, WavJEPA layer 8, `word_disjoint` | exact integer, equal to the pandas count |
| 2 | Condition differential | /d/ sites correct under `ordinary`, incorrect under `word_disjoint` | set equality with pandas, and a coverage block showing equal denominators |
| 3 | Provenance round-trip | `export_clip('sNNNNx:NNNNN')` | sample offsets identical to `plan_excerpt` with the `encode_excerpt` rounding |
| 4 | **Failure case** — off-diagonal null | a site with `eligible = false` under `combined_diagonal` | "not scored — discarded by the split". Never `0`, never a probability, never an inferred value |
| 5 | Empty result | a filter matching no rows | `n = 0` with a coverage block; not an exception, and not an empty table without a denominator |
| 6 | Unknown identifier | `export_clip('s9999z:00000')` | raises, with a message naming the identifier |
| 7 | Zero-onset site | a site with `n_frame_onsets == 0` | returned, with the zero surfaced in the label-uncertainty channel; neither dropped silently nor raising |

Cases 3 and 4 are the two a superficially plausible implementation fails: 3 because
re-deriving a window is easier than calling `plan_excerpt`, and 4 because an absent
row is easily rendered as a zero. If time is short, these two must still pass.

---

## 10. Deferred

Designed but deliberately outside this slice. None requires a schema change; all are
additive.

`group_errors` and contingency tables · `metric` with cluster bootstrap confidence
intervals · feature-space nearest neighbours · manifest hash **refusal** · the
word-identity lookup baseline rows · cross-encoder comparison · any trained probe.

One of these carries a caveat that must travel with it. Grouping errors by word
under a word-disjoint condition partially reconstructs by hand the 99.5% lexical
lookup that the split exists to prevent — leakage through the analyst rather than
through the model. The caveat must fire in the interface wherever grouping by word
is offered. The simplest correct treatment in this slice is not to offer it.

---

## 11. Development environment

The slice requires no GPU and no CUDA image. `plan_excerpt`, `Excerpt` and
`FrameGeometry` import no torch at module scope, and the slice trains nothing, so
the application runs against pandas, pyarrow, duckdb, soundfile and streamlit alone.

These are declared in `requirements-app.txt`, kept separate from `requirements.txt`,
which remains the full pinned environment for extraction and probing. The split is
deliberate: it allows the application to be run and reviewed without building the
CUDA base image.

---

## 12. Definition of done

- [ ] `count_t_realizations.py` reproduces 7,863; the inventory is 10,392 / 7,863 / 2,529; the committed TSV is unchanged
- [ ] The condition-differential question answers end to end, returning answer, evidence, coverage, provenance and both uncertainty channels
- [ ] A clip plays in the application whose samples are identical to what the encoder would consume
- [ ] All seven test cases pass, cases 3 and 4 included
- [ ] `README.md` states what the application does, how to run it, what data it uses, what AI capability it demonstrates, and what remains to be developed
- [ ] Work is committed on `challenge-2-slice` and pushed
