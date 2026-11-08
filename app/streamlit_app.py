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
    warning before anything else on the page. A real run gets no banner (the
    Result tab states the verdict); its provenance is in Run details, at the
    foot of How it was built, out of the way of readers who aren't checking it. There is no
    second flag in this file to forget. Point the app back at the
    synthetic run and the synthetic banner and wording return with it.

WHY THERE ARE TWO UNCERTAINTY SECTIONS AND NOT ONE NUMBER
    Label-and-population uncertainty comes from the inventory: what was
    excluded and why. Probe uncertainty comes from the run: the per-layer
    bootstrap intervals. Averaging them would produce one confidence figure
    that is not wrong so much as unfixable: when it moved, you could not tell
    which half had moved, and the two have different remedies — better
    alignment versus a different probe or split.

    So they are rendered in separate places, labelled by provenance: what was
    left out under "Can the data be trusted?", the intervals under the Result
    tab's per-layer table. The code has no function that combines them.

THE TABS FOLLOW THE STUDY'S ARGUMENT
    The result; why (the word split); the flaps themselves; whether the data
    holds; then questions to the graph. Looking up one site's state comes
    last: it is for checking the run, not for following the argument.
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

# The poster's and video's figures, rendered for each theme by
# app_figures.py in the vault's video folder (they are drawn there, from the
# poster's and the frames' sources). The four intro clips are Kokoro-82M
# syntheses of "Say ___ again."
ASSETS = ROOT / "app" / "assets"
INTRO_CLIPS = [  # (file stem, word, dictionary sound, accent, IPA)
    # As heard, not as sent: Kokoro's British voice r-colours the ending
    # too (F3 measured 2026-11-08), though its phonemes asked for a plain ə.
    ("us-writer-flap", "writer", "t", "American", "[ˈɹaɪɾɚ]"),
    ("us-rider-flap", "rider", "d", "American", "[ˈɹaɪɾɚ]"),
    ("gb-writer-text", "writer", "t", "British", "[ˈɹaɪtɚ]"),
    ("gb-rider-text", "rider", "d", "British", "[ˈɹaɪdɚ]"),
]

# The Hear the flaps table, as a listener reads it. The store's other columns
# (samples, frame counts, the slice) stay under "All columns".
LISTENER_COLUMNS = ["transcript", "word", "underlying", "pred", "correct", "p_t",
                    "speaker", "duration_ms"]
LISTENER_LABELS = {
    "underlying": "dictionary sound",
    "pred": "model guessed",
    "correct": st.column_config.CheckboxColumn("right?"),
    "p_t": st.column_config.NumberColumn("probability of /t/", format="%.3f"),
    "duration_ms": st.column_config.NumberColumn("flap (ms)", format="%.0f"),
}

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
    # A real run needs no warning at the top: the Result tab states the
    # verdict, and the run's provenance is in Run details (How it was built).


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
    Each tab's layer picker says so next to it (slice_controls).
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

st.set_page_config(page_title="PhonoProbe", layout="wide")
# Wide, for the tables, but capped: on a large monitor every box and line of
# text otherwise stretches edge to edge.
# Prose and alert boxes are capped at a readable line length on top of that;
# tables and figures keep the full width. Columns stack on a phone by
# themselves, so the side-by-side layouts below need no mobile case.
st.markdown(
    "<style>.block-container { max-width: 1600px; }"
    " .stMarkdown p, .stMarkdown li { max-width: 85ch; }"
    " [data-testid='stAlert'] { max-width: 85ch; }</style>",
    unsafe_allow_html=True)

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

head, intro = st.columns([2, 3], vertical_alignment="center")
head.title("PhonoProbe")
head.subheader("Was that flap a /t/ or a /d/?")
intro.write(
    "In American English the *t* in **water** and the *d* in **ladder** both come "
    "out as the same quick tap of the tongue — a *flap* — so by ear you usually "
    "can't tell which it was. PhonoProbe asks whether a speech model, WavJEPA, "
    "still keeps track: a small classifier trained on the model's internal "
    "layers guesses /t/ or /d/ for flaps in recorded conversation. This page "
    "shows the result, plays the flaps behind it, and answers questions about "
    "the study from its knowledge graph."
)
# The page's readers know data science, not phonetics: the terms they will
# meet below, in one place, instead of a gloss on every line.
with intro.expander("Terms used on this page"):
    st.markdown(
        "- **/t/, /d/** — the sound the word has in its dictionary form: "
        "*water* has a /t/, even when it's pronounced with a flap.\n"
        "- **Flap** — the quick tap that both become between vowels, as in "
        "*water* and *ladder*.\n"
        "- **/ / and [ ]** — slashes hold the sound in the word's dictionary "
        "form; square brackets hold what was actually said, in the "
        "International Phonetic Alphabet.\n"
        "- **ɹ, ɾ, ɚ** — **ɹ** is the ordinary English r, **ɾ** the flap, "
        "and **ɚ** the r-colored vowel at the end of *writer*. In the "
        "follow-up chart, **V** stands for a vowel.\n"
        "- **Underlying** — linguists' word for the dictionary sound: an "
        "underlying /t/ is the /t/ in *water*.\n"
        "- **Lemma** — a word's dictionary headword: *write*, *writes* and "
        "*writing* share one.\n"
        "- **Site** — one flap in the recordings: one word, said once, by "
        "one speaker.\n"
        "- **WavJEPA** — a self-supervised speech model: it learned from audio "
        "alone, with no transcripts. It has 13 layers (0–12). "
        "**wav2vec2-base**, a widely used model of the same kind, is the "
        "comparison in the follow-ups.\n"
        "- **Frames, pooling** — a model reads audio as a sequence of short "
        "slices, frames, and a flap spans a few. Pooling turns those few into "
        "one vector: their mean, their max, or the middle one.\n"
        "- **Probe** — a simple classifier trained on one layer's output, to "
        "test whether the information is in there.\n"
        "- **Control task** — the same probe trained on made-up labels: each "
        "word assigned /t/ or /d/ at random. With no word in both training "
        "and test, it should score at chance; if it doesn't, word identity "
        "is leaking from one to the other (criterion 2).\n"
        "- **MFCC** — a standard, hand-designed summary of a sound's spectrum, "
        "long used in speech recognition. The baseline: if the model's layers "
        "can't beat it, they add nothing a plain acoustic feature doesn't.\n"
        "- **Splits** — how flaps were divided into training and test: "
        "`random`; `speaker_disjoint` (no speaker in both); `word_disjoint` "
        "(no word in both); `joint` (no speaker and no word in both), the "
        "design registered in advance.\n"
        "- **Balanced accuracy** — the average of the accuracy on /t/ and on "
        "/d/, so guessing the commoner one scores 0.50: chance.\n"
        "- **Pre-registered** — the pass/fail criteria were written down "
        "before the run. **Amendments** are later, dated changes to that "
        "registration, each saying whether it came before or after the "
        "results."
    )

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


# No sidebar: each tab carries the controls it uses, where they apply, and
# starts at the registered design (joint, mean pooling). The run has one
# encoder in its predictions, and the page shows the headline run; the
# subsampled secondary run stays in the results CSV and the knowledge graph
# (find-training).
ENCODER = "wavjepa"
SUBSAMPLED = False
POOLINGS = ["mean", "max", "midpoint"]
LAYERS = list(range(13))


def slice_controls(prefix: str, *, split: bool = True,
                   default: str = "joint") -> tuple[str, int, str]:
    """A tab's own split, layer and pooling pickers, keyed by `prefix`.

    The layer starts at the split's best layer and moves with the split; it
    can then be moved by hand. Without `split`, the
    tab chooses its splits itself and the layer starts at `default`'s best.
    """
    ck, lk, pk = f"{prefix}_condition", f"{prefix}_layer", f"{prefix}_pooling"
    st.session_state.setdefault(ck, default)
    st.session_state.setdefault(lk, BEST[st.session_state[ck]]["layer"])
    st.session_state.setdefault(pk, "mean")

    def follow_split() -> None:
        st.session_state[lk] = BEST[st.session_state[ck]]["layer"]

    if split:
        a, b, c = st.columns(3)
        cond = a.selectbox("Split", CONDITIONS, key=ck, on_change=follow_split)
    else:
        b, c = st.columns(2)
        cond = st.session_state[ck]
    layer = b.selectbox("WavJEPA layer", LAYERS, key=lk)
    pooling = c.selectbox("Pooling over the flap", POOLINGS, key=pk)
    best = BEST[cond]
    st.caption(
        f"The layer starts at {best['layer']}, the one that happened to score "
        f"highest under `{cond}` on this run (mean pooling, {best['ba']:.3f}): "
        f"picked after the fact, not a finding. The 13 layers span only "
        f"{best['spread']:.3f} there."
    )
    return cond, layer, pooling


def render_run_details() -> None:
    """Provenance for anyone checking the run, out of everyone else's way."""
    if SYNTHETIC:
        return
    with st.expander("Run details"):
        st.caption(
            f"run `{manifest.run_id}` · predictions from code "
            f"`{manifest.code_commit[:12]}` · criteria report committed in "
            f"`{criteria_commit()}` · inventory "
            f"`{manifest.inventory_hash[:12]}` · checkpoint "
            f"`{manifest.checkpoint_revision}` · context "
            f"{manifest.context_ms:.0f} ms · seeds "
            + ", ".join(f"{k} {v}" for k, v in sorted(manifest.seeds.items()))
        )
        st.caption(f"Split scheme:\n\n`{manifest.split_scheme}`")


def themed(name: str) -> str:
    """The figure's variant for the theme the viewer sees; dark if unknown."""
    theme = getattr(st.context.theme, "type", None) or "dark"
    return str(ASSETS / "figures" / theme / name)


def render_layer_table() -> None:
    """The probe's own uncertainty: per-layer intervals, for a chosen split.

    Kept apart from what the inventory left out (the data tab); see the
    module docstring for why the two are never combined.
    """
    if SYNTHETIC:
        st.error(
            "Every `p_t` on this page is a random number. It is labeled wherever it "
            "appears. It is not weak evidence about /t/ and /d/ — it is no evidence, "
            "and averaging a lot of it together does not turn it into any."
        )
        return
    # Every layer is in the table, so only the split and the pooling are picked.
    a, b = st.columns(2)
    condition = a.selectbox("Split", CONDITIONS, index=CONDITIONS.index("joint"),
                            key="table_condition")
    pooling = b.selectbox("Pooling over the flap", POOLINGS, key="table_pooling")
    st.caption(
        "Balanced accuracy over all scored sites, with the 95% interval from "
        "a bootstrap over word types. Chance is 0.50. The control column is "
        "the same probe on the control task."
    )
    sel = results[(results.condition == condition) & (results.pooling == pooling)
                  & (results.subsampled == SUBSAMPLED)]
    wj = sel[sel.encoder == "wavjepa"]
    table = wj[["layer", "balanced_acc", "ci_lo", "ci_hi",
                "control_balanced_acc", "f1_d", "n"]].rename(columns={
        "balanced_acc": "balanced accuracy", "ci_lo": "95% low",
        "ci_hi": "95% high", "control_balanced_acc": "control",
        "f1_d": "/d/ F1", "n": "sites"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={col: st.column_config.NumberColumn(format="%.3f")
                                for col in ["balanced accuracy", "95% low",
                                            "95% high", "control", "/d/ F1"]})
    mf = sel[sel.encoder == "mfcc"]
    if not mf.empty:
        m0 = mf.iloc[0]
        st.caption(f"MFCC baseline, same slice: {m0.balanced_acc:.3f} "
                   f"[{m0.ci_lo:.3f}, {m0.ci_hi:.3f}].")


# The listening audit's committed report, one line per question and class:
#   flap  /t/  reviewed 40/40  yes 38  no  2  can't tell  0  rate 0.950 [0.835, 0.986]
AUDIT_LINE = re.compile(
    r"^\s*(\w+)\s+/(\w)/\s+reviewed (\d+)/(\d+)\s+yes\s+(\d+)\s+no\s+(\d+)\s+"
    r"can't tell\s+(\d+)\s+rate ([\d.]+) \[([\d.]+), ([\d.]+)\]")


def parse_audit(text: str) -> list[dict]:
    """The audit report's per-question rows, read from the report, not retyped."""
    rows = []
    for line in text.splitlines():
        m = AUDIT_LINE.match(line)
        if m:
            q, cls, _, _, yes, no, cant, rate, lo, hi = m.groups()
            rows.append({"question": AUDIT_TARGETS.get(q, q), "key": q,
                         "class": f"/{cls}/", "yes": int(yes), "no": int(no),
                         "can't tell": int(cant), "rate": float(rate),
                         "95% interval": f"[{lo}, {hi}]"})
    return rows

(tab_question, tab_result, tab_compare, tab_evidence, tab_data, tab_graph,
 tab_built, tab_state) = st.tabs(
    ["The question", "The result", "Listening, or recognizing the word?",
     "Hear the flaps", "Can the data be trusted?", "Ask the graph",
     "How it was built", "Look up one site"]
)

# ------------------------------------------------------------------- question

with tab_question:
    grid, words = st.columns([3, 2], gap="large")
    # Two by two, as in the video's frame: accents down, words across.
    with grid:
        for accent in ("American", "British"):
            st.markdown(f"**{accent}**")
            cells = st.columns(2)
            for cell, (stem, word, sound, _, ipa) in zip(
                    cells, [c for c in INTRO_CLIPS if c[3] == accent]):
                with cell:
                    st.markdown(f"*{word}* · /{sound}/ · heard as {ipa}")
                    st.image(themed(f"wave-{stem}.png"), width="stretch")
                    st.audio(str(ASSETS / "audio" / f"{stem}.wav"))
    words.write(
        "Two words, two accents. In American English *writer* and *rider* can "
        "sound alike: the /t/ and the /d/ both become the same quick tap. "
        "British English keeps them apart. Listen, and compare the shaded "
        "stretch of each waveform."
    )
    words.caption(
        "Synthesized with Kokoro-82M, “Say ___ again.” The American pair was "
        "made from the same sounds with the tap forced, so it is identical by "
        "construction: the extreme case. Real speakers differ at most a "
        "little, and not reliably."
    )
    words.info(
        "Listeners still know which word they heard: the difference is gone "
        "from the sound but kept in the language. **The question: does a "
        "self-supervised speech model keep it too — the dictionary /t/ or /d/ "
        "— or only the sound it actually heard?** The next tab has the "
        "pre-registered answer, measured on 10,389 real flaps from recorded "
        "conversation."
    )

# ---------------------------------------------------------------- how built

with tab_built:
    # Half width: the diagram is tall, and wider would push its lower bands
    # off the first screen.
    fig, words = st.columns(2, gap="large")
    fig.image(themed("architecture.png"), width="stretch")
    with words:
        st.markdown(
            "**Data.** Flaps from the Buckeye corpus: recorded conversations "
            "of 40 speakers of American English, each flap labeled /t/ or "
            "/d/ from its word's dictionary form.\n\n"
            "**Probing.** Around each flap, a short excerpt goes through "
            "frozen speech models — WavJEPA and wav2vec2-base — and through "
            "MFCCs, the baseline. A linear probe at every layer guesses /t/ "
            "or /d/, scored on speakers and words it never trained on. The "
            "pass/fail criteria were registered before any probe ran.\n\n"
            "**Knowledge graph.** The study itself — registrations, runs, "
            "results, findings, code — in Neo4j, linked to the flaps they "
            "describe.\n\n"
            "**App.** This page. Ask the graph lets Claude pick one of seven "
            "reviewed queries and answer only from the rows it returns."
        )
        st.caption("The diagram's 10,334 is the follow-up span study's count: "
                   "the pilot's 10,389 less 55 registered exclusions.")
        render_run_details()

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
        # The report's pointer to its own §5 means nothing on this page.
        verdict = verdict.split(" — see ")[0]
        box, scope = st.columns([1, 3], vertical_alignment="center")
        box.error(f"**{verdict}**  \n{n_pass} of {len(crit)} criteria pass.")
        # The report's second line states what was scored, on what, over
        # how many sites. Quoted, not paraphrased.
        scope.caption(lines[1] if len(lines) > 1 else "")
        # PASS/FAIL beside the criterion, so a narrow screen that scrolls
        # the long statistic column never hides it.
        st.dataframe(crit, hide_index=True, width="stretch",
                     column_order=("#", "criterion", "result", "statistic",
                                   "threshold"),
                     column_config={"#": st.column_config.NumberColumn(width="small"),
                                    "result": st.column_config.TextColumn(width="small")})

        fig, more = st.columns([3, 2])
        fig.image(str(FIGURES / "layer-curves-mean.png"),
                  caption="Balanced accuracy by layer, mean pooling, each "
                          "condition")
        with more:
            st.caption("Why the pilot failed: the next tab, and the Ask the "
                       "graph tab for the causes and what's planned.")
            with st.expander("Every layer, with 95% intervals"):
                render_layer_table()
            with st.expander("Controls and pooling"):
                st.image(str(FIGURES / "controls.png"),
                         caption="Probe vs control task")
                st.image(str(FIGURES / "pooling.png"),
                         caption="Mean, max and midpoint pooling")
            with st.expander("The full criteria report, verbatim"):
                st.code(criteria_text, language=None)

# ------------------------------------------------------------------- evidence

with tab_evidence:
    st.write(
        "Every flap the probe scored, filtered here: the split, the layer, "
        "/t/ or /d/, and whether the probe got it right. Pick one below to "
        "hear it."
    )
    condition, layer, pooling = slice_controls("hear")
    a, b = st.columns(2)
    underlying = a.selectbox("Dictionary sound", ["either", "t", "d"],
                             format_func=lambda v: {"either": "either",
                                                    "t": "/t/ (water, city)",
                                                    "d": "/d/ (ladder, ready)"}[v])
    correctness = b.selectbox("Probe got it", ["either", "correct", "misclassified"],
                              format_func=lambda v: {"either": "either",
                                                     "correct": "right",
                                                     "misclassified": "wrong"}[v])
    result = store.query_sites(
        condition=condition, encoder=ENCODER, layer=layer, pooling=pooling,
        subsampled=SUBSAMPLED,
        underlying=None if underlying == "either" else underlying,
        correct=None if correctness == "either" else correctness == "correct")
    st.info(coverage_sentence(result.coverage))
    st.caption(coverage_line(result.coverage))
    if result.n == 0:
        st.warning("**Nothing matches these filters.** That is an answer, not an "
                   "error — the counts above still hold. Try widening one filter.")
    else:
        shown = result.rows.copy()
        shown.insert(1, "transcript", [transcript(r) for r in shown.itertuples()])
        # What a listener needs; every column the store returns is one click away.
        st.dataframe(shown, width="stretch", hide_index=True,
                     column_order=[col for col in LISTENER_COLUMNS
                                   if col in shown.columns],
                     column_config=LISTENER_LABELS)
        with st.expander("All columns"):
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
                       "above. Narrow the filters above to reach others.")
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
                    f"The flap is in **{row.word}**, which has a "
                    f"**/{row.underlying}/** in its dictionary form. The probe "
                    f"guessed **/{row.pred}/** — "
                    f"{'right' if row.correct else 'wrong'}."
                )
                if SYNTHETIC:
                    st.metric("Model's score for /t/ (made up)", f"{row.p_t:.3f}")
                else:
                    st.metric(f"Probe's probability of /t/ (layer {layer}, "
                              f"{pooling}; site held out in fold {row.fold})",
                              f"{row.p_t:.3f}")
                # The neighbouring phone codes (prev_phone, next_phone) are
                # under "All columns"; here they would need a phonetics key.
                st.caption(f"The flap itself is {row.duration_ms:.0f} ms long.")
            with b:
                # The clip is planned by plan_excerpt and excerpt_samples, the
                # same two calls encode_excerpt makes: a clip off by one frame
                # would make anything concluded by ear unfounded.
                st.write(f"**Clip:** {plan['n_samples']/16:.0f} ms")
                st.caption("Exactly the audio the model heard: the flap, with "
                           "200 ms either side.")
                try:
                    clip = store.export_clip(pick, out_dir=CLIPS)
                    st.audio(str(clip))
                except FileNotFoundError as exc:
                    st.warning(f"No corpus audio mounted: {exc}")

            # Three questions about what can be heard. The /t/ or /d/ under a
            # flap cannot be (that is the project's premise); it follows from
            # the word, so `word` is the question.
            with st.expander("Judge this site yourself (logged, not counted)"):
                st.caption(
                    "About the site, not the model. Verdicts here are logged "
                    "but never counted, because this site was chosen with the "
                    "model's guess on screen. The counted review is the audit "
                    "under **Can the data be trusted?**: a random sample, "
                    "blind to the model."
                )
                view = {"encoder": ENCODER, "condition": condition,
                        "layer": layer, "pooling": pooling,
                        "subsampled": SUBSAMPLED}
                verdict_form(pick, audit=False, view=view, prefix="browse")

# ---------------------------------------------------------------- differential

with tab_compare:
    st.write(
        "A model that has memorised *water → /t/* will look good until you test "
        "it on words it never saw in training. That is what `word_disjoint` "
        "does. If accuracy drops when you switch to it, memorising is part of "
        "what was going on — and the sites that flipped are the evidence."
    )
    fig, reading = st.columns([3, 2])
    fig.image(str(FIGURES / "condition-drop.png"),
              caption="Best layer per condition")
    with reading:
        st.markdown(
            "**Reading:** holding out words costs most of the score "
            "(−0.093, `random` → `word_disjoint`); holding out speakers "
            "costs −0.025. That pattern is consistent with a probe partly "
            "recognizing the words rather than hearing the flaps."
        )
    st.markdown("#### The sites that flipped")
    a, b, d = st.columns(3)
    cond_a = a.selectbox("Compare this split…", CONDITIONS,
                         index=CONDITIONS.index("random"))
    cond_b = b.selectbox("…against this one", CONDITIONS,
                         index=CONDITIONS.index("word_disjoint"))
    cmp_u = d.selectbox("Dictionary sound", ["either", "t", "d"], key="cmp_u",
                        format_func=lambda v: {"either": "either",
                                               "t": "/t/ (water, city)",
                                               "d": "/d/ (ladder, ready)"}[v])
    # Both sides at one layer and pooling, chosen here.
    _, cmp_layer, cmp_pooling = slice_controls("cmp", split=False)
    comp = store.compare_conditions(cond_a, cond_b, encoder=ENCODER, layer=cmp_layer,
                                    pooling=cmp_pooling, subsampled=SUBSAMPLED,
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
               "The comparison across all sites, with intervals, is the "
               "figure above and the result tab's per-layer table.")
        )
        shown = comp.rows.copy()
        shown.insert(1, "transcript", [transcript(r) for r in shown.itertuples()])
        st.dataframe(shown, width="stretch", hide_index=True)

    # Wording from the knowledge graph's findings (kg/facts.py: the span
    # grid's and the representation communities'), approved 2026-10-31.
    st.divider()
    st.markdown("#### Two follow-ups")
    span, comm = st.columns(2, gap="large")
    with span:
        st.markdown("**Was the window too narrow?**")
        st.image(themed("span-grid-results.png"), width="stretch")
        st.caption(
            "A separately registered follow-up widened the stretch of audio "
            "the probe pools over, from the flap alone (ɾ) to the flap with "
            "the vowel before, after, or both (V = vowel). Widening helps no "
            "representation: no gain reaches 0.05, and every paired 95% "
            "interval includes zero. wav2vec2-base, a second speech model, "
            "scores 0.616–0.625 at its best layer on every span, ahead of "
            "MFCC on all four; WavJEPA scores 0.543–0.555, ahead of none. "
            "10,334 flaps; whiskers are 95% intervals."
        )
    with comm:
        st.markdown("**What do the models group flaps by?**")
        st.image(themed("community-ami.png"), width="stretch")
        st.caption(
            "Exploratory, not registered, and no probe: each "
            "representation's flaps are clustered (Louvain communities over "
            "a nearest-neighbor graph), and the clusters compared with ways "
            "of grouping the flaps, by adjusted mutual information (0 = "
            "unrelated). WavJEPA's clusters follow the speaker most (0.234, "
            "the largest for any representation and grouping); MFCC's follow "
            "the surrounding sounds most. None follows /t/ vs /d/ (at most "
            "0.007)."
        )

# ----------------------------------------------------------------- site lookup

with tab_state:
    st.write(
        "For checking the run: any one site's state. A site with no score can "
        "mean three different things, and they are kept apart here, never "
        "shown as a blank or a zero."
    )
    cond, st_layer, st_pooling = slice_controls("state")
    groups = load_states(STAMP, cond, st_layer, st_pooling, SUBSAMPLED)
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
            state = store.site_state(site_id.strip(), cond, layer=st_layer,
                                     pooling=st_pooling, subsampled=SUBSAMPLED)
        except UnknownSite as exc:
            st.error(str(exc))
        else:
            left = state["prev_word"] or "…"
            right = state["next_word"] or "…"
            st.markdown(f"### “{left} {state['word']} {right}”")
            st.caption(f"`{state['site_id']}` · dictionary sound /{state['underlying']}/")

            if state["status"] == NOT_SCORED:
                st.warning(
                    "**Excluded before scoring.** The site exists, but the flap "
                    "sits too near the start or end of its recording for the "
                    "fixed 200 ms window to center on it, so it was set aside "
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

# ------------------------------------------------------------------------ data

with tab_data:
    st.write(
        "Two questions about the data, separate from the model: what the "
        "inventory left out, and whether the sites it kept are what it says."
    )

    st.subheader("What got left out")
    m = st.columns(4)
    excl = stats["excluded_length_mismatch"]
    m[0].metric("words that couldn't be aligned",
                f"{excl:,}", f"{excl / stats['flap_bearing_words']:.1%} of all flap words",
                delta_color="off")
    m[1].metric("flaps the annotator wasn't sure about",
                f"{stats['excluded_annotator_mark']:,}")
    m[2].metric("flaps from neither /t/ nor /d/", f"{stats['excluded_not_t_or_d']:,}")
    m[3].metric("too near the start or end to center", f"{stats['n_excerpt_clamped']:,}")

    # A scope statement, not an apology: what the result covers, why the rest
    # was set aside, and what including it would take.
    st.info(
        "**What the result covers.** Each site's /t/ or /d/ comes from lining "
        "up the word's dictionary pronunciation with the one the transcriber "
        "heard, sound by sound. That only works when the two are the same "
        f"length. In {excl:,} flap words "
        f"({excl / stats['flap_bearing_words']:.1%}) they weren't — a sound "
        "dropped, added or merged, as happens in fast speech — so those were "
        "set aside rather than guessed at. The result speaks for flaps in "
        "words that kept their dictionary shape; flaps in more reduced words "
        "would need a different alignment to include."
    )
    st.caption(
        "Every flap that was kept is long enough for the model to see: each "
        "is cut so that it starts on one of the model's 10 ms frames "
        f"({stats['n_zero_onset_excerpt_local']:,} flaps with no frame inside)."
    )

    st.divider()
    st.subheader("Are the sites what the inventory says?")
    # The audit checks the sites behind the result, by ear (pre-registration
    # §7). Its report is written by scripts/summarize_audit.py from the
    # verdicts entered below; the summary here is parsed from that report.
    reports = sorted(p for p in FIGURES.parent.glob("audit-*.txt"))
    audit_rows = parse_audit(reports[-1].read_text()) if reports else []
    if audit_rows:
        flaps = [r for r in audit_rows if r["key"] == "flap"]
        heard = sum(r["yes"] for r in flaps)
        decided = sum(r["yes"] + r["no"] for r in flaps)
        undecided = sum(r["can't tell"] for r in flaps)
        st.success(
            f"**A flap was heard in {heard} of the {decided} sites that could "
            f"be decided** ({undecided} couldn't be). Every word matched its "
            "transcript, and every clip was centered on its flap."
            if all(r["no"] == 0 and r["can't tell"] == 0
                   for r in audit_rows if r["key"] != "flap") else
            f"**A flap was heard in {heard} of the {decided} sites that could "
            f"be decided** ({undecided} couldn't be)."
        )
        st.write(
            "80 sites, 40 /t/ and 40 /d/, drawn at random before anyone "
            "listened, and judged by one listener (the author) with nothing "
            "from the model on screen. This clears the sites, not the labels: "
            "whether a flap was /t/ or /d/ underneath can't be heard, which is "
            "the study's premise."
        )
        st.dataframe([{k: v for k, v in r.items() if k != "key"}
                      for r in audit_rows], hide_index=True, width="stretch",
                     column_config={"rate": st.column_config.NumberColumn(format="%.3f")})
        with st.expander("The audit report, verbatim"):
            st.code(reports[-1].read_text(), language=None)
    else:
        st.caption("A blind listening audit of 80 sampled sites is registered "
                   "(pre-registration §7) and not yet reported. It checks the "
                   "data, not the model.")

    st.markdown("#### The audit, site by site")
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
        # For the words around the site only; nothing from the probe is shown.
        info = store.site_state(site, "joint", layer=BEST["joint"]["layer"])
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


def graph_log(message: str) -> None:
    """One line per Ask step on the server's stderr, so a run can be traced."""
    print(f"{datetime.now():%H:%M:%S} [ask the graph] {message}",
          file=sys.stderr, flush=True)


@st.cache_data(show_spinner=False)
def run_graphrag(question: str) -> dict:
    """route -> retrieve -> answer, cached per question across sessions.

    A lost session (a reconnect, a reload) drops session state; this cache
    survives it, so asking again shows the answer already paid for. Errors
    raise, and Streamlit caches only returns, so a failure is never reused.
    `answered_at` is when the model actually ran, which is how the tab tells
    a live answer from a cached one. Cleared when the server restarts.
    """
    graphrag = load_graphrag()
    start = datetime.now()
    graph_log(f"live run: {question!r}")
    choice = graphrag.route(question)
    graph_log(f"routed to {choice['retrieval']} (term={choice['term']!r}, "
              f"criterion={choice['criterion']})")
    rows = graphrag.retrieve(choice)
    graph_log("rows: " + ", ".join(f"{k}={len(v)}" for k, v in rows.items()))
    text = graphrag.answer(question, rows)
    graph_log(f"answered in {(datetime.now() - start).total_seconds():.1f} s")
    return {"question": question, "choice": choice, "rows": rows,
            "answer": text, "answered_at": start}


def ask_graph(graphrag, question: str) -> dict:
    """run_graphrag behind the tab's checks. Never raises.

    Neo4j is checked before the first model call, so a stopped server costs
    nothing. Whatever fails, the error is logged and returned for this tab to
    show. `cached` is true when the answer predates this click.
    """
    import neo4j.exceptions

    graph_log(f"Ask: {question!r}")
    graphrag.load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        graph_log("error: no ANTHROPIC_API_KEY")
        return {"error": "No `ANTHROPIC_API_KEY`: set it in the environment or "
                         "in `.env` at the repo root (see kg/README.md, Setup)."}
    try:
        with graphrag.GraphDatabase.driver(graphrag.URI, auth=graphrag.AUTH) as d:
            d.verify_connectivity()
    except (neo4j.exceptions.DriverError, neo4j.exceptions.Neo4jError) as exc:
        graph_log(f"error: no knowledge graph at {graphrag.URI}: {exc}")
        return {"error": f"No knowledge graph at `{graphrag.URI}` ({exc}). Start "
                         "it with `docker compose -f kg/compose.yaml up -d` "
                         "(see kg/README.md, Setup)."}
    clicked = datetime.now()
    try:
        run = run_graphrag(question)
    except graphrag.anthropic.APIError as exc:
        error = f"The model call failed: {exc}"
    except (neo4j.exceptions.DriverError, neo4j.exceptions.Neo4jError) as exc:
        error = f"The retrieval failed: {exc}"
    except SystemExit as exc:  # complete() exits on a refusal
        error = str(exc)
    else:
        cached = run["answered_at"] < clicked
        if cached:
            graph_log(f"cached answer from {run['answered_at']:%H:%M:%S}, "
                      "no model calls")
        return {**run, "cached": cached}
    graph_log(f"error: {error}")
    return {"error": error}


with tab_graph:
    st.write(
        "Questions about the study's design, results and reasoning, answered "
        "from its knowledge graph (`kg/`). The model picks one of seven "
        "reviewed retrievals and fills its parameter; it never writes Cypher. "
        "It then answers from the returned rows only, citing their node keys. "
        "The rows are shown beside the answer, so every claim can be checked "
        "against them."
    )
    try:
        graphrag = load_graphrag()
    except ImportError as exc:
        st.warning(f"GraphRAG needs `neo4j` and `anthropic` "
                   f"(requirements-app.txt): {exc}")
    else:
        # Two independent inputs, no callback between them. With a callback
        # filling the text box from the pick, picking sent a real Chrome back
        # to the first tab (2026-10-24; not reproducible headless). Without
        # it, confirmed in real Chrome, the tab stays put.
        pick_col, type_col = st.columns(2)
        picked = pick_col.selectbox("Evaluated questions", GRAPHRAG_QUESTIONS,
                                    index=None, placeholder="Pick one",
                                    key="graph_example")
        typed = type_col.text_input("Or type your own question",
                                    key="graph_question")
        question = typed.strip() or picked or ""
        ask_col, will_col = st.columns([1, 11], vertical_alignment="center")
        if question:
            will_col.markdown(f"**Will ask:** {question}")
        # Kept in session state: any widget on the page reruns the script, and
        # that must neither drop the answer nor pay for it again.
        if ask_col.button("Ask", disabled=not question, key="graph_ask"):
            with st.spinner(f"Routing, retrieving and answering ({graphrag.MODEL})"):
                st.session_state.graph_run = ask_graph(graphrag, question)

        run = st.session_state.get("graph_run")
        if run and "error" in run:
            st.error(run["error"])
        elif run:
            choice = run["choice"]
            # Repeated only when the inputs have moved on since the click.
            if run["question"] != question:
                st.markdown(f"**Question:** {run['question']}")
            # Rows on the left, answer on the right: the rows still read
            # first, and the claims sit beside the rows they cite. On a phone
            # the columns stack, rows first.
            rows_col, answer_col = st.columns(2, gap="large")
            with rows_col:
                # The parameters the model filled, leaving out the unused ones.
                params = ", ".join(f"{k} = `{choice[k]!r}`"
                                   for k in ("term", "criterion") if choice.get(k))
                st.markdown(f"**Retrieval the model picked:** "
                            f"`{choice['retrieval']}`"
                            + (f" ({params})" if params else ""))
                statements = graphrag.RETRIEVALS[choice["retrieval"]]["statements"]
                for name, cypher in statements.items():
                    rows = run["rows"][name]
                    st.markdown(f"##### `{name}`: {len(rows)} rows")
                    with st.expander("Cypher"):
                        st.code(textwrap.dedent(cypher).strip(), language="cypher")
                    if rows:
                        st.dataframe(rows, hide_index=True, width="stretch")
            with answer_col:
                at = f"{run['answered_at']:%H:%M:%S}"
                st.markdown(f"#### Answer ({graphrag.MODEL})")
                # Shown above the answer, so a screenshot says whether it was live.
                if run["cached"]:
                    st.warning(f"**Cached:** answered at {at} and reused, with no "
                               "new model calls. The cache clears when the server "
                               "restarts.")
                else:
                    st.success(f"**Live:** answered at {at} by two model calls "
                               "(route, then answer).")
                st.markdown(run["answer"])
                st.caption("Keys in [brackets] are node keys from the rows "
                           "beside the answer.")
