"""Hand-written facts for the knowledge graph's design layer.

WHAT THIS FILE IS
    Everything in the graph that no results file already records: the study's
    questions, the pre-registration lineage, criteria, parameters, papers,
    linguistic concepts and the findings. kg/build_design.py adds what *is*
    recorded (curves from the results CSV, outcomes from the criteria report,
    the run from its manifest, artifact commits from git) and writes the Cypher.

    Schema and reasoning: ai-design-plan.md in the vault (Challenge 3).

THE BUCKEYE RULE
    Nothing here may carry Buckeye-derived data: no words, speaker codes or
    timestamps. Aggregate figures (site counts, balanced accuracies) are fine.
    The corpus layer (sites, tokens, type keys, lemmas, speakers) is built
    locally from a registered copy, never from this file.

SHAPE
    NODES: one dict per node. `key` is unique across the whole graph and is how
    EDGES refer to it; `labels` is a list (the control task is both a Task and
    a Control). Every other entry becomes a node property. A value of None is
    left off.
    EDGES: (source key, TYPE, target key, properties).
"""

from __future__ import annotations

REPO = "capstone"  # the public successor; history rewritten from capstone-prep


def node(key: str, *labels: str, **props) -> dict:
    return {"key": key, "labels": list(labels), **props}


NODES: list[dict] = []
EDGES: list[tuple[str, str, str, dict]] = []


def edge(src: str, rel: str, dst: str, **props) -> None:
    EDGES.append((src, rel, dst, props))


# ---------------------------------------------------------------------------
# Research: study, questions, experiments, datasets
# ---------------------------------------------------------------------------

NODES += [
    node("study", "Study", name="PhonoProbe",
         description="Layer-wise probing of frozen audio-JEPA representations "
                     "(WavJEPA) against self-supervised baselines"),
    node("rq1", "ResearchQuestion", id="RQ1", status="active",
         text="Do frozen audio-JEPA representations encode phonemes, or only "
              "phones? Tested at neutralization sites: Russian word-final "
              "devoicing, with English flapping as the control."),
    node("rq2", "ResearchQuestion", id="RQ2", status="deferred",
         text="English morphosyntax. Deferred 2026-08-01 until the phonological "
              "experiments and the probing pipeline work."),
    node("rq3", "ResearchQuestion", id="RQ3", status="active",
         text="Are those patterns specific to the JEPA objective, or general to "
              "self-supervised speech encoders?"),
    node("exp-flapping", "Experiment", name="English flapping", status="active",
         question="Is the underlying /t/ or /d/ beneath a flap recoverable "
                  "from the encoder's representation of the flap?"),
    node("exp-devoicing", "Experiment", name="Russian word-final devoicing",
         status="planned"),
    node("ds-buckeye", "Dataset", name="Buckeye Corpus of Conversational Speech",
         status="used",
         citation="Pitt, M.A., Dilley, L., Johnson, K., Kiesling, S., Raymond, "
                  "W., Hume, E. & Fosler-Lussier, E. (2007). Buckeye Corpus of "
                  "Conversational Speech (2nd release). Columbus, OH: Department "
                  "of Psychology, Ohio State University.",
         access="free registration at buckeyecorpus.osu.edu; not redistributed"),
    node("ds-common-voice-ru", "Dataset", name="Mozilla Common Voice (Russian)",
         status="planned"),
]

edge("study", "ASKS", "rq1")
edge("study", "ASKS", "rq2")
edge("study", "ASKS", "rq3")
edge("study", "HAS_EXPERIMENT", "exp-flapping")
edge("study", "HAS_EXPERIMENT", "exp-devoicing")
edge("exp-flapping", "ADDRESSES", "rq1", role="English control")
edge("exp-flapping", "ADDRESSES", "rq3", role="layer-wise curves")
edge("exp-devoicing", "ADDRESSES", "rq1", role="primary")
edge("exp-devoicing", "ADDRESSES", "rq3", role="layer-wise curves")
edge("exp-flapping", "USES_DATASET", "ds-buckeye")
edge("exp-devoicing", "USES_DATASET", "ds-common-voice-ru")

# ---------------------------------------------------------------------------
# Pre-registration lineage. Commits from `git log` on
# experiments/pilot-flapping/README.md.
# ---------------------------------------------------------------------------

PREREG = "experiments/pilot-flapping/README.md"

NODES += [
    node("rev1", "Registration", revision=1, date="2026-08-22", commit="12a0349",
         path=PREREG, pushed=True,
         summary="Speaker-disjoint splits only; absolute thresholds."),
    node("rev2", "Registration", revision=2, date="2026-08-22", commit="c7323d0",
         path=PREREG,
         summary="Word-type-disjoint splits, after measuring that word identity "
                 "alone scores 0.9951 on the real task. Three thresholds marked "
                 "provisional. A revision, not a deviation: nothing had run."),
    node("rev3", "Registration", revision=3, date="2026-09-26", commit="cb09c9d",
         path=PREREG, pushed=True,
         summary="Final before any run. Four split conditions, all 25 grid "
                 "cells scored, MFCC registered as the baseline to beat, "
                 "criteria finalized, every seed registered by value."),
]
edge("rev2", "SUPERSEDES", "rev1", reason="word identity scores 0.9951; "
     "a probe could clear revision 1 without listening to anything")
edge("rev3", "SUPERSEDES", "rev2", reason="finalize the provisional "
     "thresholds; 19 listed changes, none from a result")

# §7: dated entries appended after revision 3, never edited in above.
NODES += [
    node("amend-clarifications", "Amendment", date="2026-09-26", commit="5356109",
         title="Clarifications fixed before the first run",
         written_after_run=False,
         summary="Midpoint pooling defined; the lookup's rule for unseen "
                 "types; the speaker-ID metric; criterion 1's bootstrap is "
                 "paired."),
    node("amend-prediction", "Amendment", date="2026-09-26", commit="5356109",
         title="A prediction recorded before extraction",
         written_after_run=False,
         summary="WavJEPA's information sits higher than wav2vec2-base's. A "
                 "prediction, not a criterion; gates nothing."),
    node("amend-lookup-ties", "Amendment", date="2026-09-26", commit="f678cbf",
         title="Written after the run: the lookup's rule for tied types",
         written_after_run=True,
         summary="A type seen equally often as /t/ and /d/ falls back to the "
                 "training majority, /t/. Reaches no site under the "
                 "type-disjoint conditions; gates nothing."),
    node("amend-audit", "Amendment", date="2026-09-26", commit="9fd2e16",
         title="Registered before listening: an audit of the sites by ear",
         written_after_run=True,
         summary="40 /t/ + 40 /d/ sites, seed 20260930, blind to the probe; "
                 "questions flap / word / clip; Wilson intervals."),
    node("amend-span-grid", "Amendment", date="2026-10-17", commit="1715e95",
         title="Not a deviation: the span grid is registered separately",
         written_after_run=True,
         summary="Nothing changed. The follow-up to §5's 'site width' "
                 "suspect is a new study with its own registration; its "
                 "WavJEPA ɾ span must reproduce the pilot's joint curve."),
]
for a in ("amend-clarifications", "amend-prediction", "amend-lookup-ties",
          "amend-audit", "amend-span-grid"):
    edge(a, "AMENDS", "rev3")

# The span grid: a follow-up registered in its own file, not a revision of the
# pilot's. Its departures will go in that file's §8.
SPAN_GRID = "experiments/span-grid/README.md"

NODES += [
    node("sg-rev1", "Registration", revision=1, date="2026-10-17",
         commit="1715e95", path=SPAN_GRID, pushed=True,
         summary="Where the cue sits: four spans (ɾ, Vɾ, ɾV, VɾV) over 10,334 "
                 "sites, a pooled and a cropped design, WavJEPA and "
                 "wav2vec2-base against MFCC and duration baselines. Seven "
                 "registered comparisons with the author's predictions; the "
                 "WavJEPA ɾ span must first reproduce the pilot's joint "
                 "curve."),
]
edge("sg-rev1", "FOLLOWS_UP", "rev3", reason="§5 names the 2–4 frame site "
     "width among the suspects for a failure near chance")
edge("amend-span-grid", "POINTS_TO", "sg-rev1")

# Its §8: dated entries appended after the span grid's registration.
NODES += [
    node("amend-sg-w2v2-geometry", "Amendment", date="2026-10-18", commit="0b530b7",
         title="wav2vec2-base's geometry, verified", written_after_run=False,
         summary="Stride 320 samples and receptive field 400, as the config "
                 "says. Hidden state 0 is LayerNorm(projection + positional "
                 "conv), not the CNN output. 160 of the 10,334 flaps hold no "
                 "frame centre."),
    node("amend-sg-crop-norm", "Amendment", date="2026-10-18", commit="0b530b7",
         title="The cropped design's normalization covers one of two",
         written_after_run=False,
         summary="Conv layer 0's GroupNorm also normalizes over the whole "
                 "input and can't be disabled. Resolved by the next entry."),
    node("amend-sg-groupnorm", "Amendment", date="2026-10-18", commit="136d7a3",
         title="GroupNorm in the cropped design: kept, and measured",
         written_after_run=False,
         summary="Kept as part of each encoder; its shift per crop is to be "
                 "reported, not compared."),
    node("amend-sg-clean-frames", "Amendment", date="2026-10-18", commit="8a362f6",
         title="WavJEPA's spans take only frames that read real audio",
         written_after_run=False,
         summary="A span drops any WavJEPA frame whose receptive field runs "
                 "past the excerpt. 616 spans affected, all ɾV or VɾV."),
    node("amend-sg-float16", "Amendment", date="2026-10-18", commit="0b530b7",
         title="Frame precision: the pilot's float16 is kept",
         written_after_run=False,
         summary="So the ɾ span gets the pilot's exact features and the "
                 "reproduction check has no tolerance to set."),
    node("amend-sg-deferred", "Amendment", date="2026-10-29", commit="84a9b55",
         title="The cropped design is deferred", written_after_run=False,
         summary="Only the pooled design runs. C2 and C3 are not tested; C4, "
                 "C5 and C7 are read for the pooled design only."),
    node("amend-sg-exclusions", "Amendment", date="2026-10-29", commit="f5a4b33",
         title="The 55 exclusions meet the pilot's fixed machinery",
         written_after_run=False,
         summary="The pilot's control mapping, null mappings, bootstrap index "
                 "and lemma subset are kept, restricted to the 10,334. Also "
                 "fixes two readings: §7's unclipped sites (9,657) and C5's "
                 "MFCC + duration half (Vɾ and VɾV)."),
    node("amend-sg-c7", "Amendment", date="2026-10-29", commit="9afc0e4",
         title="Written after the results: C7's rule is one-sided; the call "
               "stands", written_after_run=True,
         summary="No effect as registered; wav2vec2-base ahead by 0.062–0.075 "
                 "on every span, reported, not called."),
]
for a in ("amend-sg-w2v2-geometry", "amend-sg-crop-norm", "amend-sg-groupnorm",
          "amend-sg-clean-frames", "amend-sg-float16", "amend-sg-deferred",
          "amend-sg-exclusions", "amend-sg-c7"):
    edge(a, "AMENDS", "sg-rev1")

# ---------------------------------------------------------------------------
# Criteria: one node per criterion per revision.
# ---------------------------------------------------------------------------

CRITERIA = {
    "rev1": [
        (1, "Balanced accuracy at the best layer", ">= 0.65"),
        (2, "Selectivity (real - control) at the best layer", ">= 0.10"),
        (3, "Layer-wise curve is not flat", "best - worst >= 0.05"),
        (4, "Minority-class F1 (/d/) at the best layer", ">= 0.50"),
    ],
    "rev2": [
        (1, "Balanced accuracy at the best layer", ">= 0.60 (provisional)"),
        (2, "Selectivity (real - control) at the best layer", ">= 0.10 (provisional)"),
        (3, "Layer-wise curve is not flat", "best - worst >= 0.05"),
        (4, "Minority-class F1 (/d/) at the best layer", ">= 0.45 (provisional)"),
    ],
    "rev3": [
        (1, "Best WavJEPA layer beats the MFCC baseline",
         "gap >= 0.05 balanced accuracy, and paired word-type bootstrap 95% "
         "lower bound > 0"),
        (2, "Control task shows no leak",
         "control's max balanced accuracy over layers does not exceed the "
         "95th percentile of 10,000 redrawn mappings"),
        (3, "Layer-wise curve is not flat", "best - worst >= 0.05"),
        (4, "Minority-class F1 (/d/) at the best layer", ">= 0.45"),
    ],
}
for rev, rows in CRITERIA.items():
    for n, statement, threshold in rows:
        NODES.append(node(f"crit-{rev}-{n}", "Criterion", number=n,
                          revision=int(rev[-1]), statement=statement,
                          threshold=threshold))
        edge(rev, "SETS_CRITERION", f"crit-{rev}-{n}")
for n in (1, 2, 3, 4):
    edge(f"crit-rev2-{n}", "SUPERSEDES", f"crit-rev1-{n}")
    edge(f"crit-rev3-{n}", "SUPERSEDES", f"crit-rev2-{n}")

# What changed, where it changed. The ±0.03 bound for criterion 2 was drafted
# inside revision 3's work (b9c861b) and replaced before the push, so it is a
# property here rather than a revision of its own.
CRITERION_NOTES = {
    "crit-rev3-1": "Comparative instead of absolute: nothing calibrates an "
                   "absolute bar for unseen words.",
    "crit-rev3-2": "A leak bound, not a selectivity floor: under a "
                   "type-disjoint split the control task's ceiling is chance. "
                   "Drafted as 0.50 ± 0.03 (b9c861b); replaced before the push "
                   "because ±0.03 fires 59–80% of the time with no leak "
                   "(61bf29e).",
    "crit-rev3-4": "A guard, not a second performance bar. Trivial floors: "
                   "0.391 always-/d/, 0.327 coin flip, 0.243 base rate.",
}
for n in NODES:
    if n["key"] in CRITERION_NOTES:
        n["note"] = CRITERION_NOTES[n["key"]]

# All rev 3 criteria are scored on `joint` only.
for n in (1, 2, 3, 4):
    edge(f"crit-rev3-{n}", "MEASURED_ON", "cond-joint")
edge("crit-rev3-1", "COMPARES_TO", "rep-mfcc")

# ---------------------------------------------------------------------------
# Tasks, controls, representations, probes
# ---------------------------------------------------------------------------

NODES += [
    node("task-voicing", "Task", name="Underlying voicing beneath a flap",
         description="Binary: was this [ɾ] an underlying /t/ or /d/? Labels "
                     "from Buckeye's hand-corrected phone tier, aligned to "
                     "the words."),
    node("task-control", "Task", "Control", name="Control task (Hewitt & Liang)",
         description="Each word type draws a random /t/-or-/d/ label at the "
                     "real marginal; every token inherits it. Under a "
                     "type-disjoint split it is a leak detector.",
         seed=20260927),
    node("task-speaker", "Task", name="Speaker identification",
         description="40-way speaker ID from the same representations. A "
                     "speaker-leakage diagnostic; defined only under random "
                     "and word_disjoint."),
    node("ctrl-lookup", "Control", name="Word-identity lookup",
         description="A lookup table over training word types; no audio. "
                     "Scores 0.9951 on the real task when types recur, and "
                     "chance when they don't."),
    node("rep-wavjepa", "Representation", name="WavJEPA", status="used",
         learned=True, n_layers=13, dim=768, checkpoint="6be4a50",
         note="Index 0 is the input to the first layer; 1–12 are layer "
              "outputs. Pretraining data not yet checked against the paper."),
    node("rep-mfcc", "Representation", name="MFCC", status="used",
         learned=False, n_layers=1, dim=39,
         note="13 coefficients incl. c0 + deltas + delta-deltas, librosa, "
              "25 ms window / 10 ms hop. Features with no learned structure."),
    node("rep-wav2vec2", "Representation", name="facebook/wav2vec2-base",
         status="used", learned=True, n_layers=13, dim=768,
         checkpoint="0b5b8e8",
         note="Run in the span grid's pooled design (2026-10-29), on the "
              "10,334 sites under joint: best layer 9 on every span, "
              "balanced accuracy 0.616–0.625. Ahead of MFCC on every span "
              "(C5). Index 0 is not the CNN output but LayerNorm(projection "
              "+ positional conv), so it carries position (span grid §8)."),
    node("rep-duration", "Representation", name="Duration", status="used",
         learned=False, n_layers=1, dim=1,
         note="§5's baseline: the length of the sonorant stretch before the "
              "flap, clipped to the excerpt (the compared one), unclipped, "
              "or the vowel's alone; one number per site."),
    node("rep-hubert", "Representation", name="facebook/hubert-base-ls960",
         status="planned", learned=True),
    node("rep-wavlm", "Representation", name="microsoft/wavlm-base",
         status="planned", learned=True),
    node("probe-linear", "Probe", name="Linear probe", status="used",
         implementation="scikit-learn LogisticRegression", penalty="l2", C=1.0,
         solver="lbfgs", max_iter=1000, class_weight="balanced",
         standardization="StandardScaler fit per probe on its training sites"),
    node("probe-stronger", "Probe", name="Stronger probe", status="planned",
         note="A §5 follow-up, as a new registration."),
]

# ---------------------------------------------------------------------------
# Parameters: conditions, pooling, environment. Construction details go on
# the SPECIFIES edges below, so a node means the same thing in every revision.
# ---------------------------------------------------------------------------

NODES += [
    node("cond-random", "Condition", name="random", held_out="nothing",
         description="Shuffled 5-fold, stratified by label. Word types and "
                     "speakers recur between train and test."),
    node("cond-speaker", "Condition", name="speaker_disjoint",
         held_out="speakers"),
    node("cond-word", "Condition", name="word_disjoint",
         held_out="word types (type keys)"),
    node("cond-joint", "Condition", name="joint",
         held_out="speakers and word types",
         description="The gated condition."),
    node("pool-mean", "Pooling", name="mean",
         description="Mean over the site's frames."),
    node("pool-max", "Pooling", name="max"),
    node("pool-midpoint", "Pooling", name="midpoint",
         description="The site's frame at or before its temporal midpoint."),
    node("env-flap", "Environment", name="[ɾ]", status="used",
         description="The flap's own frames only."),
    node("env-v-flap", "Environment", name="[Vɾ]", status="used",
         description="The preceding sonorant stretch through the flap: back "
                     "through any sonorant coda (r, l, n, m, ng, nx) to the "
                     "start of the vowel before it. V includes syllabic "
                     "consonants."),
    node("env-flap-v", "Environment", name="[ɾV]", status="used",
         description="The flap through the following segment, whatever it "
                     "is."),
    node("env-v-flap-v", "Environment", name="[VɾV]", status="used",
         description="The preceding stretch through the following segment."),
]

# Revision 1
edge("rev1", "SPECIFIES", "rep-wavjepa", role="primary")
edge("rev1", "SPECIFIES", "probe-linear", role="primary")
edge("rev1", "SPECIFIES", "task-voicing", role="primary")
edge("rev1", "SPECIFIES", "task-control", role="calibration",
     construction="selectivity against a per-type control task")
edge("rev1", "SPECIFIES", "cond-speaker", role="gated",
     construction="5-fold GroupKFold by speaker")
edge("rev1", "SPECIFIES", "pool-mean", role="primary")
edge("rev1", "SPECIFIES", "env-flap", role="primary")
# Revision 2
edge("rev2", "SPECIFIES", "rep-wavjepa", role="primary")
edge("rev2", "SPECIFIES", "probe-linear", role="primary")
edge("rev2", "SPECIFIES", "task-voicing", role="primary")
edge("rev2", "SPECIFIES", "task-control", role="calibration")
edge("rev2", "SPECIFIES", "ctrl-lookup", role="baseline",
     construction="registered at 0.9951")
edge("rev2", "SPECIFIES", "cond-joint", role="gated",
     construction="5 x 5 speaker x type grid; the 5 diagonal cells tested")
edge("rev2", "SPECIFIES", "pool-mean", role="primary")
edge("rev2", "SPECIFIES", "env-flap", role="primary")
# Revision 3
edge("rev3", "SPECIFIES", "rep-wavjepa", role="primary")
edge("rev3", "SPECIFIES", "rep-mfcc", role="baseline")
edge("rev3", "SPECIFIES", "rep-wav2vec2", role="exploratory")
edge("rev3", "SPECIFIES", "probe-linear", role="primary")
edge("rev3", "SPECIFIES", "task-voicing", role="primary")
edge("rev3", "SPECIFIES", "task-control", role="leak detector",
     construction="seed 20260927; criterion 2's null from seed 20260929")
edge("rev3", "SPECIFIES", "task-speaker", role="diagnostic")
edge("rev3", "SPECIFIES", "ctrl-lookup", role="baseline")
edge("rev3", "SPECIFIES", "cond-random", role="reported",
     construction="StratifiedKFold(shuffle=True), seed 20260926, 5 folds")
edge("rev3", "SPECIFIES", "cond-speaker", role="reported",
     construction="5 speaker folds (grid rows)")
edge("rev3", "SPECIFIES", "cond-word", role="reported",
     construction="5 type folds (grid columns); types ranked by site count, "
                  "ties by spelling")
edge("rev3", "SPECIFIES", "cond-joint", role="gated",
     construction="5 x 5 speaker x type grid; all 25 cells scored, pooled")
edge("rev3", "SPECIFIES", "pool-mean", role="primary")
edge("rev3", "SPECIFIES", "pool-max", role="secondary")
edge("rev3", "SPECIFIES", "pool-midpoint", role="secondary")
edge("rev3", "SPECIFIES", "env-flap", role="primary",
     construction="context_ms = 200, constant across sites")
# Span grid
edge("sg-rev1", "SPECIFIES", "rep-wavjepa", role="compared")
edge("sg-rev1", "SPECIFIES", "rep-wav2vec2", role="compared")
edge("sg-rev1", "SPECIFIES", "rep-mfcc", role="baseline",
     construction="alone, and with the span's clipped duration appended")
edge("sg-rev1", "SPECIFIES", "rep-duration", role="baseline")
edge("sg-rev1", "SPECIFIES", "probe-linear", role="primary")
edge("sg-rev1", "SPECIFIES", "task-voicing", role="primary")
edge("sg-rev1", "SPECIFIES", "task-control", role="leak detector")
edge("sg-rev1", "SPECIFIES", "cond-joint", role="only condition",
     construction="the pilot's folds over 10,389 sites; 55 dropped after "
                  "assignment")
edge("sg-rev1", "SPECIFIES", "pool-mean", role="only pooling")
for env in ("env-flap", "env-v-flap", "env-flap-v", "env-v-flap-v"):
    edge("sg-rev1", "SPECIFIES", env, role="span",
         construction="clipped to flap ± 200 ms; clipped sites kept")

# ---------------------------------------------------------------------------
# Code. A reference (repo, path), not the code. Commits live on artifacts.
# ---------------------------------------------------------------------------

SCRIPTS = [
    "count_flap_sites", "extract_site_features", "run_probes",
    "plot_layer_curves", "draw_audit_sample", "summarize_audit",
    "measure_lemma_overlap", "simulate_control_spread", "measure_span_grid",
    "score_span_grid",
]
NODES += [node(f"code-{s}", "Code", kind="script", repo=REPO,
               path=f"scripts/{s}.py") for s in SCRIPTS]
MODULES = {
    "splits": "src/capstone/splits.py",
    "controls": "src/capstone/probes/controls.py",
    "extraction": "src/capstone/encoders/extraction.py",
    "audit": "src/capstone/audit.py",
}
NODES += [node(f"code-mod-{m}", "Code", kind="module", repo=REPO, path=p,
               name=f"capstone.{p[len('src/capstone/'):-3].replace('/', '.')}")
          for m, p in MODULES.items()]

for c in ("cond-random", "cond-speaker", "cond-word", "cond-joint"):
    edge(c, "IMPLEMENTED_BY", "code-mod-splits")
edge("ctrl-lookup", "IMPLEMENTED_BY", "code-mod-controls")
edge("task-control", "IMPLEMENTED_BY", "code-mod-controls")
edge("rep-wavjepa", "IMPLEMENTED_BY", "code-mod-extraction")

# ---------------------------------------------------------------------------
# Artifacts. kg/build_design.py fills `commit` from git (the commit that
# added the file). local_only = carries Buckeye-derived rows; never published.
# ---------------------------------------------------------------------------

ARTIFACTS = [
    # key, path, kind, generated by, local_only
    ("art-criteria", "results/flapping/criteria-20260926.txt", "report", "run_probes", False),
    ("art-results", "results/flapping/probe-results-20260926.csv", "csv", "run_probes", False),
    ("art-folds", "results/flapping/probe-folds-20260926.csv", "csv", "run_probes", False),
    ("art-manifest", "results/flapping/run-manifest-pilot.json", "manifest", "run_probes", False),
    ("art-fig-curves", "results/flapping/figures/layer-curves-mean.png", "figure", "plot_layer_curves", False),
    ("art-fig-drop", "results/flapping/figures/condition-drop.png", "figure", "plot_layer_curves", False),
    ("art-fig-controls", "results/flapping/figures/controls.png", "figure", "plot_layer_curves", False),
    ("art-fig-pooling", "results/flapping/figures/pooling.png", "figure", "plot_layer_curves", False),
    ("art-audit-sample", "results/flapping/audit-sample-20260926.tsv", "tsv", "draw_audit_sample", True),
    ("art-audit-report", "results/flapping/audit-20260928.txt", "report", "summarize_audit", False),
    ("art-lemma-sites", "results/flapping/lemma-overlap-joint-20260926.tsv", "tsv", "measure_lemma_overlap", True),
    ("art-lemma-report", "results/corpus-verification/buckeye-lemma-overlap-20260926.txt", "report", "measure_lemma_overlap", True),
    ("art-control-spread", "results/corpus-verification/buckeye-control-spread-20260926.txt", "report", "simulate_control_spread", False),
    ("art-inventory", "results/corpus-verification/buckeye-flap-sites-20260821.tsv", "tsv", "count_flap_sites", True),
    ("art-sg-results", "results/span-grid/span-grid-results-20261029.csv", "csv", "score_span_grid", False),
    ("art-sg-cells", "results/span-grid/span-grid-cells-20261029.csv", "csv", "score_span_grid", False),
    ("art-sg-comparisons", "results/span-grid/span-grid-comparisons-20261029.csv", "csv", "score_span_grid", False),
    ("art-sg-summary", "results/span-grid/span-grid-summary-20261029.txt", "report", "score_span_grid", False),
    ("art-sg-manifest", "results/span-grid/run-manifest-span-grid.json", "manifest", "score_span_grid", False),
]
# Which run_probes.py stage wrote each of its outputs. The stages recorded
# different commits (see build_design.run_nodes).
STAGE = {"art-criteria": "gate", "art-results": "assemble",
         "art-folds": "assemble", "art-manifest": "assemble"}
for key, path, kind, script, local in ARTIFACTS:
    NODES.append(node(key, "Artifact", path=path, kind=kind, repo=REPO,
                      local_only=local))
    if key in STAGE:
        edge(key, "GENERATED_BY", f"code-{script}", stage=STAGE[key])
    else:
        edge(key, "GENERATED_BY", f"code-{script}")

# ---------------------------------------------------------------------------
# Papers. JUSTIFIES carries the section that does the justifying.
# ---------------------------------------------------------------------------

NODES += [
    node("paper-hewitt-liang", "Paper", short="Hewitt & Liang 2019",
         title="Designing and Interpreting Probes with Control Tasks",
         authors="John Hewitt, Percy Liang", year=2019, arxiv="1909.03368"),
    node("paper-zhang-bowman", "Paper", short="Zhang & Bowman 2018",
         title="Language Modeling Teaches You More Syntax than Translation "
               "Does: Lessons Learned Through Auxiliary Task Analysis",
         authors="Kelly W. Zhang, Samuel R. Bowman", year=2018,
         arxiv="1809.10040"),
    node("paper-pasad-2021", "Paper", short="Pasad, Chou & Livescu 2021",
         title="Layer-wise Analysis of a Self-supervised Speech "
               "Representation Model",
         authors="Ankita Pasad, Ju-Chieh Chou, Karen Livescu", year=2021,
         arxiv="2107.04734"),
    node("paper-pasad-2023", "Paper", short="Pasad, Shi & Livescu 2023",
         title="Comparative Layer-wise Analysis of Self-supervised Speech "
               "Models",
         authors="Ankita Pasad, Bowen Shi, Karen Livescu", year=2023,
         arxiv="2211.03929"),
    node("paper-wavjepa", "Paper", short="Yuksel et al. 2025",
         title="WavJEPA: Semantic Learning Unlocks Robust Audio Foundation "
               "Models for Raw Waveforms",
         authors="Goksenin Yuksel, Pierre Guetschel, Michael Tangermann, "
                 "Marcel van Gerven, Kiki van der Heijden",
         year=2025, arxiv="2509.23238"),
]

edge("paper-hewitt-liang", "JUSTIFIES", "task-control", section="§2",
     claim="control tasks assign random outputs per word type")
edge("paper-hewitt-liang", "JUSTIFIES", "crit-rev3-2", section="§2.3",
     claim="the control's ceiling is the share of test tokens whose types "
           "occur in training; zero under a type-disjoint split")
edge("paper-hewitt-liang", "JUSTIFIES", "probe-linear", section="§3.6",
     claim="linear and bilinear probes are the most selective")
edge("paper-hewitt-liang", "JUSTIFIES", "rep-mfcc", section="§4",
     claim="a random-representation baseline (Proj0)")
edge("paper-hewitt-liang", "JUSTIFIES", "rq3", section="§4",
     claim="layer accuracy confounds representing a property with exposing "
           "word identity")
edge("paper-zhang-bowman", "JUSTIFIES", "rep-mfcc",
     claim="the source Hewitt & Liang cite for Proj0")
edge("paper-pasad-2021", "JUSTIFIES", "rep-wav2vec2",
     claim="a published layer-wise curve for wav2vec2-base to check against")
edge("paper-pasad-2023", "JUSTIFIES", "amend-prediction",
     claim="the pretraining-objective argument behind the prediction")
edge("paper-pasad-2023", "JUSTIFIES", "rq3",
     claim="the same layer-wise analysis across wav2vec 2.0, HuBERT and WavLM")
edge("paper-pasad-2023", "JUSTIFIES", "rep-hubert")
edge("paper-pasad-2023", "JUSTIFIES", "rep-wavlm")
edge("paper-wavjepa", "JUSTIFIES", "rep-wavjepa", claim="the primary encoder")

# ---------------------------------------------------------------------------
# Concepts. Plain-language definitions for a reader without linguistics.
# Reviewed by Daniel 2026-10-10.
# ---------------------------------------------------------------------------

NODES += [
    node("con-phone", "Concept", term="phone",
         definition="A speech sound as actually produced, described by how it "
                    "is articulated or how it sounds, whatever word it "
                    "belongs to."),
    node("con-phoneme", "Concept", term="phoneme",
         definition="An abstract sound category a language uses to tell words "
                    "apart. English /t/ and /d/ are different phonemes "
                    "because 'tin' and 'din' are different words."),
    node("con-allophone", "Concept", term="allophone",
         definition="One of the predictable pronunciations of a phoneme in a "
                    "particular context. The flap is an allophone of both /t/ "
                    "and /d/ in American English."),
    node("con-flap", "Concept", term="flap",
         definition="A very quick tap of the tongue tip against the ridge "
                    "behind the upper teeth, written [ɾ]. In American English "
                    "/t/ and /d/ are usually pronounced this way between "
                    "vowels when the second is unstressed: 'water', "
                    "'ladder'."),
    node("con-neutralization", "Concept", term="neutralization",
         definition="A context where the difference between two phonemes is "
                    "lost in pronunciation. Flapping neutralizes /t/ vs /d/ "
                    "('writer' and 'rider' can sound alike); Russian "
                    "word-final devoicing neutralizes voicing (код 'code' "
                    "and кот 'cat' both end in [t]). Small cues, such as the "
                    "length of the preceding vowel, may survive."),
    node("con-underlying", "Concept", term="underlying form",
         definition="The phoneme a word has in its stored, dictionary form: "
                    "/t/ in 'writer', /d/ in 'rider'. The pilot's labels are "
                    "underlying forms."),
    node("con-surface", "Concept", term="surface form",
         definition="What is actually pronounced. At every pilot site it is "
                    "a flap, by construction, so the label can't be read off "
                    "the sound category."),
    node("con-intervocalic", "Concept", term="intervocalic",
         definition="Between two vowels: the usual environment for flapping. "
                    "Flapping also follows r, l and n ('hardy', 'party'), so "
                    "the span grid's [Vɾ] and [VɾV] spans reach back through "
                    "those to the vowel."),
    node("con-probing", "Concept", term="probing",
         definition="Training a simple classifier (a probe) on a model's "
                    "frozen internal representations, to test whether some "
                    "property can be read out of them."),
    node("con-ssl-speech", "Concept", term="self-supervised speech model",
         definition="A model trained on unlabelled audio only, by predicting "
                    "parts of its own input. No transcripts are used, so "
                    "anything phonological it encodes was learned from sound "
                    "alone."),
]

edge("con-allophone", "BROADER", "con-phone")
edge("con-flap", "BROADER", "con-phone")
edge("con-allophone", "INVOLVES", "con-phoneme")
edge("con-neutralization", "INVOLVES", "con-phoneme")
edge("con-underlying", "INVOLVES", "con-phoneme")
edge("con-surface", "INVOLVES", "con-phone")
edge("rq1", "INVOLVES", "con-phoneme")
edge("rq1", "INVOLVES", "con-phone")
edge("rq1", "INVOLVES", "con-neutralization")
edge("rq3", "INVOLVES", "con-ssl-speech")
edge("exp-flapping", "INVOLVES", "con-flap")
edge("exp-flapping", "INVOLVES", "con-neutralization")
edge("task-voicing", "INVOLVES", "con-underlying")
edge("task-voicing", "INVOLVES", "con-surface")
edge("probe-linear", "INVOLVES", "con-probing")
edge("env-v-flap-v", "INVOLVES", "con-intervocalic")
edge("rep-wavjepa", "INVOLVES", "con-ssl-speech")

# ---------------------------------------------------------------------------
# Findings: the answer to "Why did the pilot fail?" Statuses confirmed by
# Daniel 2026-10-10; wording reviewed the same day.
# Outcome keys (outcome-1 … outcome-4) and curve keys come from
# kg/build_design.py.
# ---------------------------------------------------------------------------

NODES += [
    node("find-sites", "Finding", candidate_cause="Sites aren't what the "
         "inventory says", status="ruled out for the sites, not the labels",
         statement="A blind listening audit of 80 sites heard a flap in 72 of "
                   "74 decidable cases (6 /d/ can't-tells), and the right "
                   "word and a well-placed clip in all 80. The labels come "
                   "from the dictionary form and can't be heard under a "
                   "flap, so the audit clears the sites, not the labels."),
    # Corrected 2026-10-14: said a cue outside the site's frames "is invisible
    # to the probe", true only for MFCC. WavJEPA encodes the whole excerpt
    # (scripts/extract_site_features.py:206-230), so attention mixes context in.
    # Updated 2026-10-29 from the span grid's pooled design; wording and
    # status approved by Daniel the same day.
    node("find-width", "Finding", candidate_cause="Site too narrow (2–4 frames)",
         status="ruled out for MFCC; open for the encoders until the cropped "
                "design runs",
         statement="Sites are 2–4 encoder frames wide (median 3, 27.4 ms). "
                   "MFCC's 25 ms frames see only local audio (deltas reach "
                   "±40 ms), so for MFCC a cue outside the site's frames is "
                   "invisible. WavJEPA and wav2vec2-base encode the flap "
                   "± 200 ms as one excerpt, so at depth their flap frames "
                   "can carry that context; for WavJEPA the pilot shows only "
                   "that the cue wasn't linearly recoverable from those "
                   "frames. The span grid's pooled design (2026-10-29) "
                   "widened the pooled frames to the preceding stretch, the "
                   "following segment, or both, and no representation gained "
                   "(largest +0.015, none called). For MFCC, whose frames are "
                   "local, that rules width out, and the stretch's duration "
                   "alone is near chance (about 0.53). For the encoders the "
                   "pooled design cannot rule it out: every span's frames had "
                   "heard the whole excerpt, so pooling more of them changes "
                   "which frames are read, not what was heard. That test is "
                   "the cropped design, deferred (span grid §8, 2026-10-29). "
                   "Narrow frames don't explain WavJEPA's score on their own: "
                   "from the same ɾ span, wav2vec2-base reaches 0.616 under "
                   "joint, against WavJEPA's 0.543."),
    node("find-words", "Finding", candidate_cause="The cue doesn't generalize "
         "to unseen words", status="consistent with the result",
         statement="The drop from random to joint is mostly the word axis: "
                   "holding out speakers costs little, holding out word "
                   "types costs most of the gap above chance."),
    node("find-training", "Finding", candidate_cause="Too little training data",
         status="ruled out",
         statement="Retraining the single-axis conditions at joint's training "
                   "size (about 64% of sites) costs 0.006–0.013 at the best "
                   "layer, against a 0.108 drop from random to joint."),
    node("find-power", "Finding", candidate_cause="Too little statistical power",
         status="limitation",
         statement="The gap to MFCC's paired word-type bootstrap interval is "
                   "[-0.015, +0.056]: no evidence of an advantage over MFCC, "
                   "which is not evidence of none."),
    node("find-lemma", "Finding", candidate_cause="Lemma overlap",
         status="limitation, registered before the run",
         statement="25.1% of joint test sites (2,612) have their lemma in "
                   "training with the same label; the split is disjoint by "
                   "word form, not lemma. On the 7,586 lemma-unseen sites the "
                   "gap to MFCC is +0.030, CI [-0.004, +0.073], still short "
                   "of criterion 1."),
]

# What each finding explains. Criterion 2 passed; the rest failed.
for f in ("find-sites", "find-width", "find-words", "find-training"):
    for n in (1, 3, 4):
        edge(f, "EXPLAINS", f"outcome-{n}")
edge("find-power", "EXPLAINS", "outcome-1")
edge("find-lemma", "EXPLAINS", "outcome-1")

# What each finding rests on.
edge("find-sites", "EVIDENCED_BY", "art-audit-report")
edge("find-sites", "EVIDENCED_BY", "art-audit-sample")  # -SAMPLES-> its 80 Sites (corpus layer)
edge("find-sites", "EVIDENCED_BY", "amend-audit")
edge("find-width", "EVIDENCED_BY", "rev3", section="§2")
edge("find-width", "EVIDENCED_BY", "env-v-flap-v")
edge("sg-rev1", "TESTS", "find-width")
for cond in ("random", "speaker_disjoint", "word_disjoint", "joint"):
    edge("find-words", "EVIDENCED_BY", f"curve:wavjepa:mean:{cond}:full")
edge("find-words", "EVIDENCED_BY", "art-fig-drop")
for cond in ("random", "speaker_disjoint", "word_disjoint"):
    edge("find-training", "EVIDENCED_BY", f"curve:wavjepa:mean:{cond}:subsampled")
edge("find-power", "EVIDENCED_BY", "art-criteria")
edge("find-lemma", "EVIDENCED_BY", "art-criteria")
edge("find-lemma", "EVIDENCED_BY", "art-lemma-sites")
edge("find-lemma", "EVIDENCED_BY", "rev3", section="§1a")

# ---------------------------------------------------------------------------
# The span grid's findings: what its pooled design found (2026-10-29), one per
# registered question. Wording approved by Daniel 2026-10-29. Every number is
# from results/span-grid/ (004ea19); kg/build_design.py reprints them. The
# Run and the sg Curves come from kg/build_design.py.
# ---------------------------------------------------------------------------

NODES += [
    node("find-sg-span", "Finding", question="Does pooling over more of the "
         "neighbouring speech help, when the encoder heard the whole excerpt? "
         "(C1)", status="no effect",
         statement="Widening the pooled span helps no representation. Of the "
                   "nine gaps (Vɾ, ɾV and VɾV minus ɾ, for WavJEPA, "
                   "wav2vec2-base and MFCC), none reaches 0.05 and every "
                   "paired 95% interval includes zero; the largest is +0.015 "
                   "(MFCC, VɾV). The registered prediction, that Vɾ beats ɾ, "
                   "is not supported."),
    node("find-sg-encoders", "Finding", question="Does an encoder know more "
         "than the stretch's duration, and does it beat MFCC? (C4, C5)",
         status="effect for wav2vec2-base; none for WavJEPA",
         statement="wav2vec2-base scores 0.616–0.625 at its best layer on "
                   "every span. It is called ahead of MFCC on all four spans "
                   "(gaps +0.081 to +0.092), of MFCC with the clipped "
                   "duration on Vɾ and VɾV (+0.070, +0.079), and of the "
                   "clipped duration alone on Vɾ and VɾV (+0.089, +0.098). "
                   "WavJEPA scores 0.543–0.555 and is called ahead of none of "
                   "them; its largest gap to MFCC is +0.025 (Vɾ), half the "
                   "threshold."),
    node("find-sg-duration", "Finding", question="How far does the stretch's "
         "duration alone separate /t/ from /d/? (§5's baselines)",
         status="reported, not compared",
         statement="The preceding stretch's duration alone is near chance: "
                   "0.527 clipped to the excerpt, 0.529 unclipped, 0.524 for "
                   "the vowel alone, each 95% interval including 0.50. "
                   "Clipping costs 0.0015. Under a linear probe, lengthening "
                   "before an underlying /d/, the cue the study set out to "
                   "locate, barely separates /t/ from /d/ on these sites."),
    node("find-sg-depth", "Finding", question="At what depth does the "
         "information peak? (C6)", status="reported, no threshold",
         statement="WavJEPA peaks early: layer 3 for ɾ, 4 for Vɾ and ɾV, 6 "
                   "for VɾV, as the pilot's ɾ curve did (layer 3). "
                   "wav2vec2-base peaks at layer 9 on every span, and each "
                   "span's 95% range for the best layer lies within layers "
                   "6–10. "
                   "The pilot's 2026-09-26 prediction, that WavJEPA's "
                   "information sits higher than wav2vec2-base's, is "
                   "reversed."),
    node("find-sg-c7", "Finding", question="Which encoder is better? (C7)",
         status="no effect, as registered; the reverse reported, not called",
         statement="C7 is computed as WavJEPA − wav2vec2-base, and §6's rule "
                   "is one-sided, so it can call only a WavJEPA advantage: "
                   "the registered call is no effect on all four spans. "
                   "Reported, not called: wav2vec2-base is ahead by "
                   "0.062–0.075 on every span, every 95% interval below zero "
                   "(§8, 2026-10-29). The encoders' order rests on C5, which "
                   "calls wav2vec2-base ahead of MFCC on every span and "
                   "WavJEPA on none."),
    node("find-sg-checks", "Finding", question="Can the grid's numbers be "
         "trusted? (§5's checks)", status="passed",
         statement="The WavJEPA ɾ span reproduced the pilot's joint curve "
                   "exactly on its 10,389 sites (largest difference 0.0). On "
                   "the grid's 10,334, every control-task p-value is above "
                   "0.05 (smallest 0.095); the lemma-unseen and unclipped "
                   "re-scores track the full scores within about 0.02; all "
                   "5,650 fits converged."),
]

SG_REPS = ("wavjepa", "wav2vec2", "mfcc")
SG_SPANS = ("r", "vr", "rv", "vrv")
for f in ("find-sg-span", "find-sg-encoders", "find-sg-duration",
          "find-sg-depth", "find-sg-c7", "find-sg-checks"):
    edge(f, "FROM_RUN", "run-span-grid")
    edge(f, "EVIDENCED_BY", "art-sg-summary")
for f in ("find-sg-span", "find-sg-encoders", "find-sg-c7"):
    edge(f, "EVIDENCED_BY", "art-sg-comparisons")
for rep in SG_REPS:
    for span in SG_SPANS:
        edge("find-sg-span", "EVIDENCED_BY", f"curve:sg:{rep}:{span}")
for enc in ("wavjepa", "wav2vec2"):
    for span in SG_SPANS:
        for f in ("find-sg-encoders", "find-sg-depth", "find-sg-c7"):
            edge(f, "EVIDENCED_BY", f"curve:sg:{enc}:{span}")
for span in SG_SPANS:
    edge("find-sg-encoders", "EVIDENCED_BY", f"curve:sg:mfcc:{span}")
for span in ("vr", "vrv"):
    edge("find-sg-encoders", "EVIDENCED_BY", f"curve:sg:mfcc_duration:{span}")
for variant in ("clipped", "full", "vowel"):
    edge("find-sg-duration", "EVIDENCED_BY", f"curve:sg:duration:{variant}")
edge("find-sg-encoders", "EVIDENCED_BY", "curve:sg:duration:clipped")
edge("find-sg-depth", "TESTS", "amend-prediction")
edge("find-sg-c7", "EVIDENCED_BY", "amend-sg-c7")
edge("find-sg-checks", "EVIDENCED_BY", "art-sg-results")

# What the span grid says about the pilot's width suspect.
edge("find-width", "EVIDENCED_BY", "find-sg-span")
edge("find-width", "EVIDENCED_BY", "find-sg-duration")
edge("find-width", "EVIDENCED_BY", "amend-sg-deferred")
edge("amend-sg-deferred", "POINTS_TO", "find-width")

# ---------------------------------------------------------------------------
# Display names. Neo4j Browser captions a node with its `name`, so every node
# gets a short one; longer text stays in its own property.
# ---------------------------------------------------------------------------

SHORT_NAMES = {
    "amend-clarifications": "Clarifications",
    "amend-prediction": "Prediction",
    "amend-lookup-ties": "Lookup ties (after run)",
    "amend-audit": "Audit (after run)",
    "amend-span-grid": "Span grid pointer",
    "sg-rev1": "Span grid (rev 1)",
    "amend-sg-w2v2-geometry": "wav2vec2 geometry",
    "amend-sg-crop-norm": "Crop normalization",
    "amend-sg-groupnorm": "GroupNorm kept",
    "amend-sg-clean-frames": "Clean frames only",
    "amend-sg-float16": "float16 kept",
    "amend-sg-deferred": "Cropped design deferred",
    "amend-sg-exclusions": "The 55 and the pilot's machinery",
    "amend-sg-c7": "C7 one-sided (after results)",
    "task-voicing": "Voicing task",
    "task-control": "Control task",
    "task-speaker": "Speaker ID",
    "ctrl-lookup": "Word lookup",
    "find-sites": "Sites mislabelled?",
    "find-width": "Sites too narrow?",
    "find-words": "Doesn't generalize to unseen words",
    "find-training": "Too little training data?",
    "find-power": "Too little power",
    "find-lemma": "Lemma overlap",
    "find-sg-span": "Wider spans don't help",
    "find-sg-encoders": "wav2vec2 beats the baselines; WavJEPA doesn't",
    "find-sg-duration": "Duration alone near chance",
    "find-sg-depth": "Depth: WavJEPA early, wav2vec2 layer 9",
    "find-sg-c7": "C7: no effect as registered",
    "find-sg-checks": "Checks passed",
}
for n in NODES:
    labels = n["labels"]
    if n["key"] in SHORT_NAMES:
        n["name"] = SHORT_NAMES[n["key"]]
    elif "Registration" in labels:
        n["name"] = f"Revision {n['revision']}"
    elif "Criterion" in labels:
        n["name"] = f"C{n['number']} (rev {n['revision']})"
    elif "Paper" in labels:
        n["name"] = n["short"]
    elif "Concept" in labels:
        n["name"] = n["term"]
    elif "ResearchQuestion" in labels:
        n["name"] = n["id"]
    elif "Code" in labels and "name" not in n:
        n["name"] = n["path"].rsplit("/", 1)[-1]
