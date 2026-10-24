"""Challenge 2 — evidence-first interface over the flapping pilot.

WHAT THIS IS FOR
    One vertical slice: a question goes in, typed tools answer it, and what
    comes back carries its own provenance — which sites, out of how many, cut
    from which samples, under which run. Since 2026-09-26 it reads the real
    pilot run, which failed its pre-registered criteria. The point was never
    that the numbers are good. It is that a number on this screen can be walked
    back to the audio it claims to describe, and a failed result is exactly
    where that matters most.

WHY THE BANNER IS CHOSEN BY THE MANIFEST
    Until the pilot ran, every probability here was a seeded random draw, and
    render_guard() refused any manifest that did not say synthetic. That was
    right while no real run existed. Now one does, and the guard is replaced by
    render_banner(): it reads the manifest's own `synthetic` field, which has
    no default and is written by the run itself, and draws the synthetic
    warning or the real-run provenance before anything else on the page. There
    is no second flag in this file to forget. Point the app back at the
    synthetic run and the synthetic banner and wording return with it.

WHY THERE ARE TWO UNCERTAINTY SECTIONS AND NOT ONE NUMBER
    Label-and-population uncertainty comes from the inventory: what was
    excluded and why. Probe uncertainty comes from the run: the per-layer
    bootstrap intervals. Averaging them would produce one confidence figure
    that is not wrong so much as unfixable: when it moved, you could not tell
    which half had moved, and the two have different remedies — better
    alignment versus a different probe or split.

    So they are rendered in separate sections, labelled by provenance, and the
    code has no function that combines them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capstone.audit import AUDIT_TARGETS, summarize  # noqa: E402
from capstone.predictions import CONDITIONS, Manifest  # noqa: E402
from capstone.retrieval import NOT_REACHED, NOT_SCORED, SliceStore, UnknownSite  # noqa: E402

INVENTORY = ROOT / "results" / "flapping" / "sites-20260924.parquet"
STATS = ROOT / "results" / "flapping" / "sites-20260924.stats.json"
# The pilot run. The synthetic run at /data/embeddings/predictions.parquet is
# kept, untouched: pointing these two back at it restores the synthetic banner
# and wording, because both are chosen from the manifest (render_banner).
RUN_DIR = Path("/data/embeddings/probe-run-20260926")
PREDICTIONS = RUN_DIR / "predictions.parquet"
MANIFEST = RUN_DIR / "run-manifest.json"
PROBE_RESULTS = ROOT / "results" / "flapping" / "probe-results-20260926.csv"
CRITERIA = ROOT / "results" / "flapping" / "criteria-20260926.txt"
# Shown as committed, never redrawn here: one plotting implementation
# (scripts/plot_layer_curves.py), so this page and the slides cannot disagree.
FIGURES = ROOT / "results" / "flapping" / "figures"
REVIEWS = Path("/data/reviews/reviewed.jsonl")
CLIPS = Path("/data/clips")

# How many sites the lookup dropdown lists per state. 10,352 sites are never
# reached by a 40-site run, and any one of them demonstrates that state just as
# well as all of them — the true count is always shown beside the cap, so the
# list cannot be read as the total.
LOOKUP_LIMIT = 200

# The listening audit's sample (pre-registration §7, 2026-09-26). Drawn once
# by scripts/draw_audit_sample.py; the questions are capstone.audit's.
AUDIT_SAMPLE = ROOT / "results" / "flapping" / "audit-sample-20260926.tsv"
VERDICT_LABELS = {"yes": "yes", "no": "no", "cant_tell": "can't tell"}

# The Ask the graph tab runs kg/graphrag.py's pipeline as it is. The five
# questions are the ones GraphRAG was evaluated on; each has a saved run in
# kg/graphrag-runs/.
GRAPHRAG = ROOT / "kg" / "graphrag.py"
GRAPHRAG_QUESTIONS = [
    "What's a flap?",
    "How did criterion 2 get its final form?",
    "Why did the pilot fail, and what's planned to test it?",
    "How does WavJEPA's best layer compare with MFCC under each split condition?",
    "What does registration revision 3 specify that the pilot never used?",
]


def criteria_commit() -> str:
    """The commit that recorded the criteria report, for the banner.

    Asked of git rather than typed here, so the banner cannot go on naming a
    commit after the report is re-recorded. Falls back to the report's own
    `code commit` line, which names the code that computed it.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--",
             str(CRITERIA.relative_to(ROOT))],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    if out:
        return out
    m = re.search(r"^code commit\s+(\w{7})", CRITERIA.read_text(), re.M)
    return m.group(1) if m else "unknown"


def render_banner(manifest: Manifest, criteria_text: str | None) -> None:
    """The first thing on the page, and chosen by the manifest alone.

    A synthetic manifest gets the warning it always got. A real one gets its
    provenance and the report's `PILOT:` line, quoted rather than restated. A manifest with no boolean `synthetic` is refused: that is
    the one case with no correct banner.
    """
    if not isinstance(manifest.synthetic, bool):
        raise RuntimeError(
            f"run manifest {MANIFEST} has synthetic={manifest.synthetic!r}; it "
            "must say true or false, and the page will not guess which."
        )
    if manifest.synthetic:
        st.error(
            "**SYNTHETIC RUN — the model scores on this page are made up.**  \n"
            "No probe has been trained yet. Every `p_t` below is a random number from a "
            "fixed seed. They are here so the plumbing can be tested end to end — which "
            "works just as well with noise, because the plumbing does not care whether a "
            "number means anything. **Nothing on this page is a finding about /t/ or /d/.**"
        )
        return
    verdict = next((line for line in (criteria_text or "").splitlines()
                    if line.startswith("PILOT:")), "criteria report not found")
    st.info(
        f"**REAL RUN — `{manifest.run_id}`.** Predictions from code "
        f"`{manifest.code_commit[:7]}` (run manifest); criteria report committed "
        f"in `{criteria_commit()}`.  \n**{verdict}**"
    )


def parse_criteria(text: str) -> list[dict]:
    """The criteria report's four criteria as rows: statistic, threshold, result.

    Split out of the report's own lines rather than retyped, so the table
    cannot drift from the file. A detail line's `(needs …)` parentheticals are
    the threshold, and the rest is the statistic. Criterion 2 states its
    threshold as a quantity instead (the null's 95th percentile), so when
    there is no `needs` the first clause is the statistic and the rest is what
    it was held against.
    """
    rows, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^Criterion (\d+) — (.+): (PASS|FAIL)$", line)
        if not m:
            continue
        detail = lines[i + 1].strip() if i + 1 < len(lines) else ""
        needs = re.findall(r"\(((?:[\w ]+ )?needs [^)]*)\)", detail)
        if needs:
            statistic = re.sub(r"\s*\((?:[\w ]+ )?needs [^)]*\)", "", detail)
            threshold = "; ".join(needs)
        else:
            first, _, rest = detail.partition("; ")
            statistic, threshold = first, rest
        rows.append({"#": int(m.group(1)), "criterion": m.group(2),
                     "statistic": statistic, "threshold": threshold,
                     "result": m.group(3)})
    return rows


def best_layers(results) -> dict:
    """Per condition: the WavJEPA layer that scored highest on this run.

    Mean pooling, headline run, read from the committed results CSV. This is
    a default for the view, not a finding: it is where this run happened to
    peak, chosen after the fact, and on joint the layers are close together.
    The sidebar says so next to it.
    """
    head = results[(results.encoder == "wavjepa") & (results.pooling == "mean")
                   & (~results.subsampled)]
    out = {}
    for cond, g in head.groupby("condition"):
        top = g.loc[g.balanced_acc.idxmax()]
        out[cond] = {"layer": int(top.layer), "ba": float(top.balanced_acc),
                     "spread": float(g.balanced_acc.max() - g.balanced_acc.min())}
    return out


def input_stamp() -> tuple:
    """Modification times of everything the cached loaders are built from.

    Passed as the first argument to each one, so a cache key changes when its
    inputs do. Streamlit keys a no-argument cached function on nothing at all,
    which means it never expires for the life of the session.

    For load_store that is a development annoyance: the object outlives an edit
    to retrieval.py and the next call hits a method the stale instance does not
    have.

    For load_manifest it is a provenance bug, and a bad one. Regenerating
    predictions writes a new run_id, a new code_commit and a new inventory_hash,
    and a cached manifest would go on asserting the old three in the caption
    above rows that came from the new run. A page whose stated provenance
    quietly stops describing what is on it is the exact failure the rest of this
    interface exists to prevent, so it does not get to happen here by accident
    of caching.
    """
    import capstone.predictions
    import capstone.retrieval

    paths = [INVENTORY, STATS, PREDICTIONS, MANIFEST, PROBE_RESULTS, CRITERIA,
             Path(capstone.retrieval.__file__), Path(capstone.predictions.__file__)]
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


@st.cache_resource
def load_store(stamp: tuple) -> SliceStore:
    return SliceStore(INVENTORY, PREDICTIONS)


@st.cache_data
def load_manifest(stamp: tuple) -> Manifest:
    return Manifest.read(MANIFEST)


@st.cache_data
def load_stats(stamp: tuple) -> dict:
    return json.loads(STATS.read_text())


@st.cache_data
def load_probe_results(stamp: tuple):
    import pandas as pd

    return pd.read_csv(PROBE_RESULTS)


@st.cache_data
def load_criteria(stamp: tuple) -> str | None:
    return CRITERIA.read_text() if CRITERIA.exists() else None


@st.cache_data
def load_states(stamp: tuple, condition: str, layer: int, pooling: str,
                subsampled: bool) -> dict:
    """Sites grouped by eligibility state. Cached — it scans all 10,392 sites."""
    return load_store(stamp).sites_by_state(condition, layer=layer, pooling=pooling,
                                            subsampled=subsampled)


def transcript(row) -> str:
    """The flap site's word with one word either side: 'seven ladies who'.

    A reviewer asked to judge a label by ear needs to know what they are about
    to hear. The word alone does not tell them — 'ladies' could be anywhere in
    anything — and without the frame it is guesswork dressed up as a verdict.

    An ellipsis marks a pause rather than a missing word, and it is shown rather
    than closed over: 1,360 sites have no preceding word and 826 no following
    one. Silently joining across the gap would put two words next to each other
    that the speaker never said next to each other.
    """
    left = row.prev_word if getattr(row, "prev_word", "") else "…"
    right = row.next_word if getattr(row, "next_word", "") else "…"
    return f"{left} {row.word} {right}"


def label_for(row) -> str:
    """How a site is named anywhere a human has to pick one."""
    return f'"{transcript(row)}"  ({row.site_id})'


def coverage_sentence(cov) -> str:
    """The denominators, in a sentence rather than as four bare numbers."""
    d = cov.as_dict()
    return (f"Showing **{d['compared_on']:,}** sites. This run scored "
            f"**{d['n_scored']:,}** sites in total, out of **{d['n_total']:,}** "
            f"in the corpus — so these are {d['compared_on']:,} of "
            f"{d['n_scored']:,} scored, **not** {d['compared_on']:,} of "
            f"{d['n_total']:,}.")


def compare_sentence(comp) -> str:
    """The Compare tab's denominators: rows shown, against sites compared.

    coverage_sentence says "showing N" with N = compared_on, which in a query
    is the number of rows. In a comparison, compared_on is the number of sites
    scored under both splits, and the table holds only those that changed. So
    the general sentence read "Showing 10,389 sites" over a table of 639.
    """
    d = comp.coverage.as_dict()
    return (f"Showing **{comp.n:,}** sites whose result differs, out of "
            f"**{d['compared_on']:,}** scored under both splits. This run "
            f"scored **{d['n_scored']:,}** sites, out of **{d['n_total']:,}** "
            "in the corpus.")


def coverage_line(cov) -> str:
    d = cov.as_dict()
    return (f"**{d['compared_on']:,}** shown · **{d['n_scored']:,}** scored · "
            f"**{d['n_eligible']:,}** eligible · **{d['n_total']:,}** in inventory")


def record_verdict(site_id: str, target: str, verdict: str, note_key: str,
                   manifest: Manifest, view: dict, audit: bool) -> None:
    """Append one verdict to reviewed.jsonl.

    inventory_hash travels on every line. A verdict about a site is a verdict
    about a VERSION of that site: regenerating the inventory under --keep-marked
    admits 754 further sites and changes the hash, and a verdict recorded
    against the old one should stop applying rather than silently attach itself
    to different audio.

    run_id and synthetic keep verdicts from different runs apart: the file
    holds verdicts made on the synthetic run (20260920T000056Z) as well as
    this one. The view (encoder, condition, layer, pooling, subsampled) says
    which prediction was on screen while browsing; the blind audit shows none,
    and records none.

    `audit` is what decides whether a line is ever counted (capstone.audit):
    true only from the blind Audit tab. A verdict from the browser is kept
    but never counted, because the site was picked by a filter with the
    model's guess on screen.

    Append-only: earlier lines, which predate these fields, are never
    rewritten. Called as a button callback, so the page that follows already
    shows it.
    """
    note = st.session_state.get(note_key, "")
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "site_id": site_id,
        "target": target,
        "verdict": verdict,
        "note": note,
        "inventory_hash": manifest.inventory_hash,
        "run_id": manifest.run_id,
        "code_commit": manifest.code_commit,
        "synthetic": manifest.synthetic,
        "audit": audit,
        **view,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with REVIEWS.open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    st.toast(f"{target}: {VERDICT_LABELS[verdict]}"
             + ("" if audit else " (kept, not counted)"))


def read_reviews() -> list[dict]:
    """Every line of reviewed.jsonl, every run. Filtering is capstone.audit's."""
    if not REVIEWS.exists():
        return []
    return [json.loads(x) for x in REVIEWS.read_text().splitlines() if x.strip()]


def verdict_form(site_id: str, *, audit: bool, view: dict, prefix: str,
                 current: dict | None = None) -> None:
    """The three questions, each answered yes / no / can't tell, with a note.

    `current` maps question to the last verdict recorded, shown beside it so
    the reviewer can see what the log holds and change it (last line wins).
    """
    for target, question in AUDIT_TARGETS.items():
        cols = st.columns([4, 1, 1, 1, 3])
        now = (current or {}).get(target)
        cols[0].markdown(f"**{target}**: {question}"
                         + (f"  \n*recorded: {VERDICT_LABELS[now]}*" if now else ""))
        note_key = f"{prefix}_n_{target}_{site_id}"
        cols[4].text_input("note", key=note_key, label_visibility="collapsed",
                           placeholder="note (optional)")
        for col, verdict in zip(cols[1:4], ("yes", "no", "cant_tell")):
            col.button(VERDICT_LABELS[verdict], key=f"{prefix}_{verdict}_{target}_{site_id}",
                       type="primary" if now == verdict else "secondary",
                       on_click=record_verdict,
                       args=(site_id, target, verdict, note_key, manifest, view, audit))


# --------------------------------------------------------------------- layout

st.set_page_config(page_title="Flapping slice", layout="wide")

missing = [p for p in (INVENTORY, PREDICTIONS, MANIFEST) if not p.exists()]
if missing:
    st.error("Missing inputs:\n\n" + "\n".join(f"- `{p}`" for p in missing))
    st.code("python scripts/count_flap_sites.py --parquet "
            "results/flapping/sites-20260924.parquet\n"
            "# the pilot's predictions: see scripts/run_probes.py's docstring")
    st.stop()

STAMP = input_stamp()
manifest = load_manifest(STAMP)
criteria_text = load_criteria(STAMP)
render_banner(manifest, criteria_text)
SYNTHETIC = manifest.synthetic
store = load_store(STAMP)
stats = load_stats(STAMP)
results = load_probe_results(STAMP)
BEST = best_layers(results)

st.title("Was that flap a /t/ or a /d/?")
st.write(
    "In American English the *t* in **water** and the *d* in **ladder** both come "
    "out as the same quick tap — a flap. This page is for looking at individual "
    "flaps in recorded speech, hearing them, and checking what the model said "
    "about each one. Every site is 10–130 ms of audio inside a real conversation."
)
st.caption(
    f"run `{manifest.run_id}` · code `{manifest.code_commit[:12]}` · "
    f"inventory `{manifest.inventory_hash[:12]}` · checkpoint "
    f"`{manifest.checkpoint_revision}` · context {manifest.context_ms:.0f} ms · "
    "seeds " + ", ".join(f"{k} {v}" for k, v in sorted(manifest.seeds.items()))
)

# The condition and layer live in session_state under fixed keys, so every
# view reads the same slice. Changing the condition moves the layer to that
# condition's default; the layer can then be moved by hand.
st.session_state.setdefault("condition", "joint")
st.session_state.setdefault("layer", BEST["joint"]["layer"])
st.session_state.setdefault("pooling", "mean")


@st.cache_resource
def load_graphrag():
    """kg/graphrag.py as a module, loaded from its path.

    kg/ is a folder of scripts, not a package, and stays one: loading the one
    file by path keeps facts.py and the loaders off sys.path, and leaves the
    command line (`python kg/graphrag.py ...`) exactly as it was. A missing
    neo4j or anthropic raises here, inside the tab, not at the top of the app.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("graphrag", GRAPHRAG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_default_layer() -> None:
    st.session_state.layer = BEST[st.session_state.condition]["layer"]
    st.session_state.result_condition = st.session_state.condition


def from_result_view() -> None:
    """The Result tab's selector drives the site view: same condition, its layer."""
    st.session_state.condition = st.session_state.result_condition
    st.session_state.layer = BEST[st.session_state.condition]["layer"]


st.session_state.setdefault("result_condition", st.session_state.condition)


with st.sidebar:
    st.header("Query")
    condition = st.selectbox("How train/test was split", CONDITIONS,
                             key="condition", on_change=to_default_layer)
    st.caption(
        "`random` — words and speakers can appear in both train and test.  \n"
        "`speaker_disjoint` — a speaker is in one or the other, never both.  \n"
        "`word_disjoint` — a word type is in one or the other, never both.  \n"
        "`joint` — word-split *and* speaker-split: the pre-registered design."
    )
    encoder = st.selectbox("Encoder", ["wavjepa"])
    layer = st.selectbox("Encoder layer", list(range(13)), key="layer")
    best = BEST[condition]
    st.caption(
        f"Starts at layer {best['layer']}: the layer that happened to score "
        f"highest under `{condition}` on this run (mean pooling, "
        f"{best['ba']:.3f}). Picked after the fact, not in advance, and not a "
        f"finding — the 13 layers span only {best['spread']:.3f} here."
    )
    pooling = st.selectbox("Pooling over the flap", ["mean", "max", "midpoint"],
                           key="pooling")
    if condition == "joint":
        subsampled = False
        st.caption("`joint` has no subsampled run; showing the headline run.")
    else:
        subsampled = st.toggle("Subsampled training (secondary run)", value=False)
    underlying = st.selectbox("Underlying sound", ["either", "t", "d"],
                              format_func=lambda v: {"either": "either",
                                                     "t": "/t/ (water, city)",
                                                     "d": "/d/ (ladder, ready)"}[v])
    correctness = st.selectbox("Model got it", ["either", "correct", "misclassified"],
                               format_func=lambda v: {"either": "either",
                                                      "correct": "right",
                                                      "misclassified": "wrong"}[v])
    st.divider()
    st.caption(f"Split scheme recorded in the manifest:\n\n`{manifest.split_scheme}`")

u = None if underlying == "either" else underlying
c = None if correctness == "either" else (correctness == "correct")

(tab_result, tab_evidence, tab_compare, tab_uncertainty, tab_state, tab_audit,
 tab_graph) = st.tabs(
    ["Result", "Browse and listen", "Compare two splits",
     "What this can and can't tell you", "Look up one site", "Audit (blind)",
     "Ask the graph"]
)

# --------------------------------------------------------------------- result

with tab_result:
    if criteria_text is None:
        st.warning(f"No criteria report at `{CRITERIA}`.")
    else:
        lines = criteria_text.splitlines()
        verdict = next((x for x in lines if x.startswith("PILOT:")), "")
        crit = parse_criteria(criteria_text)
        n_pass = sum(r["result"] == "PASS" for r in crit)
        st.subheader("The pre-registered result")
        st.error(f"**{verdict}**  \n{n_pass} of {len(crit)} criteria pass.")
        # The report's second line states what was scored, on what, over
        # how many sites. Quoted, not paraphrased.
        st.caption(lines[1] if len(lines) > 1 else "")
        st.dataframe(crit, hide_index=True, width="stretch",
                     column_config={"#": st.column_config.NumberColumn(width="small"),
                                    "result": st.column_config.TextColumn(width="small")})

        st.image(str(FIGURES / "layer-curves-mean.png"),
                 caption="Balanced accuracy by layer, mean pooling, each "
                         "condition (figures/layer-curves-mean.png)")
        fig, reading = st.columns([3, 2])
        fig.image(str(FIGURES / "condition-drop.png"),
                  caption="Best layer per condition (figures/condition-drop.png)")
        with reading:
            st.markdown(
                "**Reading, from Amendment 3 (A3.2):** closing word identity "
                "costs most of the score (−0.093, `random` → `word_disjoint`); "
                "closing speakers costs −0.025."
            )
            st.caption("Interpretation beyond that line is in the report, "
                       "not on this page.")
        with st.expander("Controls and pooling"):
            st.image(str(FIGURES / "controls.png"),
                     caption="Probe vs control task (figures/controls.png)")
            st.image(str(FIGURES / "pooling.png"),
                     caption="Mean, max and midpoint pooling (figures/pooling.png)")
        with st.expander("The full criteria report, verbatim"):
            st.code(criteria_text, language=None)

    # The audit checks the sites behind this result, by ear (§7). Its report
    # is written by scripts/summarize_audit.py from the Audit tab's verdicts.
    reports = sorted(p for p in FIGURES.parent.glob("audit-*.txt"))
    st.markdown("#### Are the sites what the inventory says?")
    if reports:
        with st.expander(f"Listening audit, blind to the probe ({reports[-1].name})"):
            st.code(reports[-1].read_text(), language=None)
    else:
        st.caption("A blind listening audit of 80 sampled sites is registered "
                   "(pre-registration §7) and not yet reported. It checks the "
                   "data, not the model: see the Audit tab.")

    st.divider()
    st.markdown("#### From the result to the sites behind it")
    st.selectbox("Look at the sites under this split", CONDITIONS,
                 key="result_condition", on_change=from_result_view)
    st.caption(
        f"The other tabs now show `{condition}` at layer {layer}, "
        f"{pooling} pooling — the same slice as the sidebar. Choosing a split "
        "here moves both to it, at the layer where that split scored highest "
        "on this run."
    )

# ------------------------------------------------------------------- evidence

with tab_evidence:
    result = store.query_sites(condition=condition, encoder=encoder, layer=layer,
                               pooling=pooling, subsampled=subsampled,
                               underlying=u, correct=c)
    st.info(coverage_sentence(result.coverage))
    st.caption(
        f"{coverage_line(result.coverage)} — shown on every answer, even when "
        "it tells you nothing new. It is the only thing stopping a partial table "
        "from reading like the whole corpus."
    )
    if result.n == 0:
        st.warning("**Nothing matches these filters.** That is an answer, not an "
                   "error — the counts above still hold. Try widening one filter.")
    else:
        shown = result.rows.copy()
        shown.insert(1, "transcript", [transcript(r) for r in shown.itertuples()])
        st.dataframe(shown, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Listen to one")
        st.caption(
            "Pick a site. You will hear the flap with 200 ms either side — about "
            "a word's worth of run-up and follow-through, which is what makes it "
            "possible to hear at all."
        )
        # Capped like the lookup tab's list, with the true count beside it.
        # The table above still holds every row.
        options = {label_for(r): r.site_id
                   for r in result.rows.head(LOOKUP_LIMIT).itertuples()}
        if result.n > LOOKUP_LIMIT:
            st.caption(f"Listing the first {LOOKUP_LIMIT} of {result.n:,} rows "
                       "above. Narrow the filters in the sidebar to reach others.")
        picked_label = st.selectbox("Site", list(options))
        pick = options[picked_label]
        if pick:
            plan = store.plan_clip(pick)
            row = result.rows[result.rows.site_id == pick].iloc[0]
            a, b = st.columns([2, 3])
            with a:
                st.markdown(f"### “{transcript(row)}”")
                st.caption(f"`{row.site_id}` · speaker `{row.speaker}`")
                st.write(
                    f"The flap is in **{row.word}**. It is really a "
                    f"**/{row.underlying}/** underneath; the model guessed "
                    f"**/{row.pred}/** — {'right' if row.correct else 'wrong'}."
                )
                if SYNTHETIC:
                    st.metric("Model's score for /t/ (made up)", f"{row.p_t:.3f}")
                else:
                    st.metric(f"Probe's probability of /t/ (layer {layer}, "
                              f"{pooling}; site held out in fold {row.fold})",
                              f"{row.p_t:.3f}")
                st.caption(
                    f"Sounds either side: `{row.prev_phone}` … `{row.next_phone}`. "
                    f"The flap itself is {row.duration_ms:.0f} ms long."
                )
            with b:
                st.write(f"**Clip:** {plan['n_samples']/16:.0f} ms "
                         f"(samples `{plan['start_sample']:,}`–`{plan['end_sample']:,}`)")
                st.caption(
                    "These are the exact samples the encoder would read, not a "
                    "close approximation — the window comes from `plan_excerpt` "
                    "and `excerpt_samples`, the same two calls `encode_excerpt` "
                    "makes. A clip off by one frame would make anything you "
                    "concluded by ear unfounded."
                )
                try:
                    clip = store.export_clip(pick, out_dir=CLIPS)
                    st.audio(str(clip))
                except FileNotFoundError as exc:
                    st.warning(f"No corpus audio mounted: {exc}")

            st.markdown("#### Your verdict: kept, not counted")
            # Three questions about what can be heard. The /t/ or /d/ under a
            # flap cannot be (that is the project's premise); it follows from
            # the word, so `word` is the question.
            st.caption(
                "About the site, not the model: whether the model was right is "
                "judged across all 10,389 sites by the criteria report. "
                "Verdicts here are logged but never counted, because this site "
                "was chosen by a filter with the model's guess on screen. The "
                "counted review is the **Audit** tab: a random sample, blind "
                "to the model."
            )
            view = {"encoder": encoder, "condition": condition, "layer": layer,
                    "pooling": pooling, "subsampled": subsampled}
            verdict_form(pick, audit=False, view=view, prefix="browse")

# ---------------------------------------------------------------- differential

with tab_compare:
    st.markdown("#### Is the model listening, or just recognising the word?")
    st.write(
        "A model that has memorised *water → /t/* will look good until you test "
        "it on words it never saw in training. That is what `word_disjoint` "
        "does. If accuracy drops when you switch to it, memorising is part of "
        "what was going on — and the sites that flipped are the evidence."
    )
    a, b = st.columns(2)
    cond_a = a.selectbox("Compare this split…", CONDITIONS,
                         index=CONDITIONS.index("random"))
    cond_b = b.selectbox("…against this one", CONDITIONS,
                         index=CONDITIONS.index("word_disjoint"))
    cmp_u = st.selectbox("Underlying sound ", ["either", "t", "d"], key="cmp_u",
                         format_func=lambda v: {"either": "either",
                                                "t": "/t/ (water, city)",
                                                "d": "/d/ (ladder, ready)"}[v])

    # joint has no subsampled run, so a comparison involving it uses the
    # headline run on both sides rather than coming back empty.
    cmp_sub = subsampled and "joint" not in (cond_a, cond_b)
    st.caption(f"Both sides at layer {layer}, {pooling} pooling, "
               f"{'subsampled' if cmp_sub else 'headline'} run — the sidebar's "
               "slice.")
    comp = store.compare_conditions(cond_a, cond_b, encoder=encoder, layer=layer,
                                    pooling=pooling, subsampled=cmp_sub,
                                    underlying=None if cmp_u == "either" else cmp_u)
    st.info(compare_sentence(comp))
    st.caption(comp.note)
    if comp.n == 0:
        st.warning("**No site changed.** Every site the two splits share was "
                   "judged the same way by both. That is an answer too.")
    else:
        flipped_out = int((comp.rows.correct_a & ~comp.rows.correct_b).sum())
        flipped_in = int((~comp.rows.correct_a & comp.rows.correct_b).sum())
        st.markdown(
            f"**{flipped_out} sites** went from right under `{cond_a}` to wrong "
            f"under `{cond_b}`. **{flipped_in}** went the other way."
        )
        st.caption(
            "These are counts of the rows below — not percentages, not a "
            "significance test, and not a claim about why. "
            + ("Saying why would need a trained probe, which does not exist yet."
               if SYNTHETIC else
               "The aggregate comparison, with intervals, is in the criteria "
               "report and the results CSV.")
        )
        shown = comp.rows.copy()
        shown.insert(1, "transcript", [transcript(r) for r in shown.itertuples()])
        st.dataframe(shown, width="stretch", hide_index=True)

# ----------------------------------------------------------------- uncertainty

with tab_uncertainty:
    st.write(
        "Two different things here are uncertain, and they are kept apart on "
        "purpose. "
        + ("One is real and can be quoted today. The other is noise."
           if SYNTHETIC else
           "One comes from what was left out of the data; the other from the "
           "probe run itself.")
    )

    st.subheader("1 · What got left out of the data — real, and quotable now")
    st.caption("Counted by `scripts/count_flap_sites.py` and read from "
               "`sites-20260924.stats.json`, not typed into this page.")
    m = st.columns(4)
    excl = stats["excluded_length_mismatch"]
    m[0].metric("words thrown out",
                f"{excl:,}", f"{excl / stats['flap_bearing_words']:.1%} of all flap words",
                delta_color="off")
    m[1].metric("flaps the annotator wasn't sure about",
                f"{stats['excluded_annotator_mark']:,}")
    m[2].metric("flaps from neither /t/ nor /d/", f"{stats['excluded_not_t_or_d']:,}")
    m[3].metric("too near the start or end to centre", f"{stats['n_excerpt_clamped']:,}")

    st.warning(
        f"**What's left is the tidy half.** Those {excl:,} thrown-out words "
        f"({excl / stats['flap_bearing_words']:.1%}) were dropped because the "
        "written-down pronunciation and the actual one had different numbers of "
        "sounds — which happens when a speaker slurs. So the excluded words are "
        "the *most* casually pronounced ones, and casual pronunciation is "
        f"exactly where flapping gets interesting. Whatever the remaining "
        f"{stats['n_sites']:,} sites end up showing, it cannot speak for those. "
        "That limit is knowable right now, before any result exists."
    )

    st.markdown("**Flaps too short for the model to see**")
    st.metric("flaps the model takes no sample inside",
              f"{stats['n_zero_onset_excerpt_local']:,}")
    st.caption(
        "The model samples audio every 10 ms, and some flaps are shorter than "
        "that. None is missed, because every flap is cut out so that it starts "
        "exactly on a sample. Measured against the raw recording instead, "
        f"{stats['n_zero_onset_absolute']:,} flaps would contain no sample, but "
        "that is not how the model reads them. Until 2026-09-24 an off-by-one "
        "bug dropped the sample at each flap's start and left 39 flaps empty "
        "(D1 in `docs/challenges/challenge-2-slice-decisions.md`)."
    )

    st.divider()
    if SYNTHETIC:
        st.subheader("2 · How confident the model is — meaningless here")
        st.error(
            "Every `p_t` on this page is a random number. It is labelled wherever it "
            "appears. It is not weak evidence about /t/ and /d/ — it is no evidence, "
            "and averaging a lot of it together does not turn it into any."
        )
        fix = "the second needs a real probe"
    else:
        st.subheader("2 · How sure the probe's score is — per layer, from this run")
        sub_label = "subsampled" if subsampled else "headline"
        st.caption(
            f"`{condition}`, {pooling} pooling, {sub_label} run. Read from "
            "`results/flapping/probe-results-20260926.csv`, not recomputed here. "
            "Balanced accuracy over all scored sites, with the 95% interval "
            "from the word-type bootstrap in `scripts/run_probes.py`. Chance "
            "is 0.50. The control column is the same probe on the control "
            "task."
        )
        sel = results[(results.condition == condition) & (results.pooling == pooling)
                      & (results.subsampled == subsampled)]
        wj = sel[sel.encoder == "wavjepa"]
        table = wj[["layer", "balanced_acc", "ci_lo", "ci_hi",
                    "control_balanced_acc", "f1_d", "n"]].rename(columns={
            "balanced_acc": "balanced accuracy", "ci_lo": "95% low",
            "ci_hi": "95% high", "control_balanced_acc": "control",
            "f1_d": "/d/ F1", "n": "sites"})
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["balanced accuracy", "95% low",
                                              "95% high", "control", "/d/ F1"]})
        mf = sel[sel.encoder == "mfcc"]
        if not mf.empty:
            m0 = mf.iloc[0]
            st.caption(f"MFCC baseline, same slice: {m0.balanced_acc:.3f} "
                       f"[{m0.ci_lo:.3f}, {m0.ci_hi:.3f}].")
        fix = "the second is a matter of the probe and the split"
    st.caption(
        "These two are never added together into one confidence score. They come "
        "from different places and have different fixes — the first needs better "
        f"alignment or recovering those thrown-out words, {fix} — and a combined "
        "number would hide which of the two had moved, leaving you unable to fix "
        "either."
    )

# ----------------------------------------------------------------- site lookup

with tab_state:
    st.write(
        "There are three reasons a site might have no score, and they mean "
        "different things. This page keeps them apart. Showing all three as a "
        "blank cell — or worse, as a zero — is the specific mistake this view "
        "exists to make impossible."
    )
    cond = st.selectbox("Under which split", CONDITIONS,
                        index=CONDITIONS.index("joint"), key="state_cond")

    # joint has no subsampled run; the lookup falls back to the headline run.
    state_sub = subsampled and cond != "joint"
    st.caption(f"At layer {layer}, {pooling} pooling, "
               f"{'subsampled' if state_sub else 'headline'} run — the sidebar's slice.")
    groups = load_states(STAMP, cond, layer, pooling, state_sub)
    kinds = {
        "scored": f"Scored in this run ({len(groups['scored']):,})",
        "excluded": f"Excluded before scoring ({len(groups['excluded']):,})",
        "not_reached": f"Not looked at in this run ({len(groups['not_reached']):,})",
    }
    kind = st.radio("Show me a site that was…", list(kinds), horizontal=True,
                    format_func=kinds.get, key="state_kind")

    pool = groups[kind]
    if pool.empty:
        st.info(f"No sites are in that state under `{cond}`.")
        site_id = None
    else:
        # Capped, and the true count is in the radio label above so the cap
        # cannot be mistaken for the total. Any one site demonstrates the state;
        # shipping 10,352 options would make the page heavy to prove nothing
        # extra. The identifier box below still reaches every site.
        capped = pool.head(LOOKUP_LIMIT)
        options = {label_for(r): r.site_id for r in capped.itertuples()}
        if len(pool) > LOOKUP_LIMIT:
            st.caption(f"Listing the first {LOOKUP_LIMIT} of {len(pool):,}. "
                       "Any site can still be reached by identifier below.")
        site_id = options[st.selectbox("Site", list(options), key="state_site")]

    with st.expander("Or look one up by identifier"):
        typed = st.text_input("Site identifier", value="",
                              placeholder="sNNNNx:NNNNN",
                              help="Format is track:phone_number")
        if typed.strip():
            site_id = typed.strip()

    if site_id:
        try:
            state = store.site_state(site_id.strip(), cond, layer=layer,
                                     pooling=pooling, subsampled=state_sub)
        except UnknownSite as exc:
            st.error(str(exc))
        else:
            left = state["prev_word"] or "…"
            right = state["next_word"] or "…"
            st.markdown(f"### “{left} {state['word']} {right}”")
            st.caption(f"`{state['site_id']}` · really a /{state['underlying']}/")

            if state["status"] == NOT_SCORED:
                st.warning(
                    "**Excluded before scoring.** The site exists, but the flap "
                    "sits too near the start or end of its recording for the "
                    "fixed 200 ms window to centre on it, so it was set aside "
                    "before any split, and no prediction was ever made. Not a "
                    "zero, not a low score — nothing."
                )
            elif state["status"] == NOT_REACHED:
                st.info(
                    "**Not looked at in this run.** The site was eligible; this "
                    f"run scored {manifest.n_sites_scored:,} sites and this was "
                    "not one of them. Different from being excluded."
                )
            else:
                made_up = "made-up " if SYNTHETIC else ""
                st.success(
                    f"**Scored.** The model guessed /{state['pred']}/ — "
                    f"{'right' if state['correct'] else 'wrong'} — with a "
                    f"{made_up}score of {state['p_t']:.3f} for /t/."
                )
            with st.expander("Raw record"):
                st.json(state)

# ---------------------------------------------------------------------- audit

with tab_audit:
    st.write(
        "Does the inventory hold what it says it does? A random sample of "
        "sites — 40 /t/, 40 /d/, drawn before anyone listened (pre-registration "
        "§7) — checked by ear. This tests the data, not the model, so nothing "
        "from the probe is shown here."
    )
    if not AUDIT_SAMPLE.exists():
        st.warning(f"No audit sample at `{AUDIT_SAMPLE}`. Draw it with "
                   "`scripts/draw_audit_sample.py`.")
    else:
        import pandas as pd

        sample = pd.read_csv(AUDIT_SAMPLE, sep="\t")
        reviews = read_reviews()
        from capstone.audit import counted_verdicts

        got = counted_verdicts(reviews, sample, manifest.run_id)
        done = {s for s in sample.site_id
                if all((s, t) in got for t in AUDIT_TARGETS)}
        st.progress(len(done) / len(sample),
                    text=f"{len(done)} of {len(sample)} sites fully reviewed")

        ids = sample.site_id.tolist()
        if "audit_site" not in st.session_state:
            st.session_state.audit_site = next((s for s in ids if s not in done), ids[0])

        def step(delta: int) -> None:
            i = ids.index(st.session_state.audit_site)
            st.session_state.audit_site = ids[(i + delta) % len(ids)]

        def next_unreviewed() -> None:
            i = ids.index(st.session_state.audit_site)
            rest = ids[i + 1:] + ids[:i + 1]
            st.session_state.audit_site = next((s for s in rest if s not in done),
                                               st.session_state.audit_site)

        order = dict(zip(sample.site_id, sample.order))
        nav = st.columns([5, 1, 1, 2])
        nav[0].selectbox("Site", ids, key="audit_site",
                         format_func=lambda s: f"{order[s]}. {s}"
                                               + ("  ✓" if s in done else ""))
        nav[1].button("◀", on_click=step, args=(-1,), key="audit_prev")
        nav[2].button("▶", on_click=step, args=(1,), key="audit_next")
        nav[3].button("Next unreviewed", on_click=next_unreviewed, key="audit_skip")

        site = st.session_state.audit_site
        info = store.site_state(site, "joint", layer=layer)
        left, right = info["prev_word"] or "…", info["next_word"] or "…"
        a, b = st.columns([2, 3])
        with a:
            st.markdown(f"### “{left} {info['word']} {right}”")
            st.caption(f"Listen for the flap in **{info['word']}**. `{site}`")
        with b:
            try:
                st.audio(str(store.export_clip(site, out_dir=CLIPS)))
            except FileNotFoundError as exc:
                st.warning(f"No corpus audio mounted: {exc}")
        verdict_form(site, audit=True, view={}, prefix="audit",
                     current={t: v for (s, t), v in got.items() if s == site})

        with st.expander("Running tally (counted verdicts only)"):
            tally = summarize(reviews, sample, manifest.run_id)
            st.dataframe(tally, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("rate", "ci_lo", "ci_hi")})
            st.caption("yes / (yes + no), Wilson 95% interval; can't tell is "
                       "reported beside the rate, not folded in. Per class "
                       "only: the sample is balanced, the inventory is not.")

# -------------------------------------------------------------- ask the graph


def use_example() -> None:
    st.session_state.graph_question = st.session_state.graph_example


def ask_graph(graphrag, question: str) -> dict:
    """route -> retrieve -> answer, as kg/graphrag.py runs them. Never raises.

    Neo4j is checked before the first model call, so a stopped server costs
    nothing. Whatever fails, the error is returned for this tab to show.
    """
    import neo4j.exceptions

    graphrag.load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"error": "No `ANTHROPIC_API_KEY`: set it in the environment or "
                         "in `.env` at the repo root (see kg/README.md, Setup)."}
    try:
        with graphrag.GraphDatabase.driver(graphrag.URI, auth=graphrag.AUTH) as d:
            d.verify_connectivity()
    except (neo4j.exceptions.DriverError, neo4j.exceptions.Neo4jError) as exc:
        return {"error": f"No knowledge graph at `{graphrag.URI}` ({exc}). Start "
                         "it with `docker compose -f kg/compose.yaml up -d` "
                         "(see kg/README.md, Setup)."}
    try:
        choice = graphrag.route(question)
        rows = graphrag.retrieve(choice)
        text = graphrag.answer(question, rows)
    except graphrag.anthropic.APIError as exc:
        return {"error": f"The model call failed: {exc}"}
    except (neo4j.exceptions.DriverError, neo4j.exceptions.Neo4jError) as exc:
        return {"error": f"The retrieval failed: {exc}"}
    except SystemExit as exc:  # complete() exits on a refusal
        return {"error": str(exc)}
    return {"question": question, "choice": choice, "rows": rows, "answer": text}


with tab_graph:
    st.write(
        "Questions about the study's design, results and reasoning, answered "
        "from its knowledge graph (`kg/`). The model picks one of five "
        "reviewed retrievals and fills its parameter; it never writes Cypher. "
        "It then answers from the returned rows only, citing their node keys. "
        "The rows are shown first, so every claim can be checked against them."
    )
    try:
        graphrag = load_graphrag()
    except ImportError as exc:
        st.warning(f"GraphRAG needs `neo4j` and `anthropic` "
                   f"(requirements-app.txt): {exc}")
    else:
        st.selectbox("Evaluated questions", GRAPHRAG_QUESTIONS, index=None,
                     placeholder="Pick one, or type your own below",
                     key="graph_example", on_change=use_example)
        question = st.text_input("Question", key="graph_question")
        # Kept in session state: any widget on the page reruns the script, and
        # that must neither drop the answer nor pay for it again.
        if st.button("Ask", disabled=not question.strip(), key="graph_ask"):
            with st.spinner(f"Routing, retrieving and answering ({graphrag.MODEL})"):
                st.session_state.graph_run = ask_graph(graphrag, question.strip())

        run = st.session_state.get("graph_run")
        if run and "error" in run:
            st.error(run["error"])
        elif run:
            choice = run["choice"]
            st.markdown(f"**Question:** {run['question']}")
            st.markdown(f"**Retrieval:** `{choice['retrieval']}` "
                        f"(term = `{choice['term']!r}`, "
                        f"criterion = `{choice['criterion']}`)")
            statements = graphrag.RETRIEVALS[choice["retrieval"]]["statements"]
            for name, cypher in statements.items():
                rows = run["rows"][name]
                st.markdown(f"##### `{name}`: {len(rows)} rows")
                with st.expander("Cypher"):
                    st.code(textwrap.dedent(cypher).strip(), language="cypher")
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch")
            st.markdown(f"#### Answer ({graphrag.MODEL})")
            st.markdown(run["answer"])
            st.caption("Keys in [brackets] are node keys from the rows above.")
