"""Challenge 2 slice — evidence-first interface over the flapping inventory.

WHAT THIS IS FOR
    One vertical slice: a question goes in, typed tools answer it, and what
    comes back carries its own provenance — which sites, out of how many, cut
    from which samples, under which run. The point is not that the numbers are
    good. They are synthetic and carry no information. The point is that a
    number on this screen can be walked back to the audio it claims to describe.

WHY THE BANNER CANNOT BE TURNED OFF
    Every probability here comes from a seeded random draw. A person looking at
    a table of probabilities will read them as results — that is what tables of
    probabilities are for — so 'synthetic' is asserted in four places: the run
    manifest, every prediction row, the rendered uncertainty string, and a
    banner drawn before any query can run.

    It is structural rather than disciplined. render_guard() below raises if the
    manifest it is handed does not say synthetic, so a future edit that forgets
    the banner fails loudly instead of quietly showing noise as findings.

WHY THERE ARE TWO UNCERTAINTY SECTIONS AND NOT ONE NUMBER
    Label-and-population uncertainty is real, measured, and citable today. Probe
    uncertainty is synthetic and means nothing. Averaging them would produce one
    confidence figure that is not wrong so much as unfixable: when it moved, you
    could not tell which half had moved, and the two have different remedies —
    better alignment versus more data.

    So they are rendered in separate sections, labelled by provenance, and the
    code has no function that combines them.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capstone.predictions import CONDITIONS, Manifest  # noqa: E402
from capstone.retrieval import NOT_REACHED, NOT_SCORED, SliceStore, UnknownSite  # noqa: E402

INVENTORY = ROOT / "results" / "flapping" / "sites-20260924.parquet"
STATS = ROOT / "results" / "flapping" / "sites-20260924.stats.json"
PREDICTIONS = Path("/data/embeddings/predictions.parquet")
MANIFEST = Path("/data/embeddings/run-manifest.json")
REVIEWS = Path("/data/reviews/reviewed.jsonl")
CLIPS = Path("/data/clips")

# How many sites the lookup dropdown lists per state. 10,352 sites are never
# reached by a 40-site run, and any one of them demonstrates that state just as
# well as all of them — the true count is always shown beside the cap, so the
# list cannot be read as the total.
LOOKUP_LIMIT = 200

# The four things a reviewer can judge. Deliberately not accept/reject: on a
# synthetic run the probe output carries no information, so "accept this answer"
# is meaningless and an interface offering it trains the reviewer to approve
# noise. Every verdict names which of four claims it is about.
REVIEW_TARGETS = {
    "label": "Did you hear a flap, and does the /t/ or /d/ under it look right?",
    "clip": "Does the audio contain the flap, roughly centred and not cut off?",
    "retrieval": "Is this site actually an answer to what you filtered for?",
    "grounding": "Does the wording on screen stay within what the data says?",
}


def render_guard(manifest: Manifest) -> None:
    """Refuse to render anything if the run does not declare itself synthetic.

    The brief's invariant is that the renderer must not be ABLE to omit the
    synthetic marker. A flag every call site is supposed to remember is not that
    — this is.
    """
    if not manifest.synthetic:
        raise RuntimeError(
            "run manifest does not declare synthetic=true. This interface "
            "renders synthetic runs and labels them as such; it has no correct "
            "way to present a real one, and showing it unlabelled is the "
            "failure this check exists to prevent."
        )


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

    paths = [INVENTORY, STATS, PREDICTIONS, MANIFEST,
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
def load_states(stamp: tuple, condition: str) -> dict:
    """Sites grouped by eligibility state. Cached — it scans all 10,392 rows."""
    return load_store(stamp).sites_by_state(condition)


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


def coverage_line(cov) -> str:
    d = cov.as_dict()
    return (f"**{d['compared_on']:,}** shown · **{d['n_scored']:,}** scored · "
            f"**{d['n_eligible']:,}** eligible · **{d['n_total']:,}** in inventory")


def record_verdict(site_id: str, target: str, verdict: str, note: str,
                   manifest: Manifest) -> None:
    """Append one verdict to reviewed.jsonl.

    inventory_hash travels on every line. A verdict about a site is a verdict
    about a VERSION of that site: regenerating the inventory under --keep-marked
    admits 754 further sites and changes the hash, and a verdict recorded
    against the old one should stop applying rather than silently attach itself
    to different audio.
    """
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "site_id": site_id,
        "target": target,
        "verdict": verdict,
        "note": note,
        "inventory_hash": manifest.inventory_hash,
        "run_id": manifest.run_id,
        "synthetic": manifest.synthetic,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with REVIEWS.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


# --------------------------------------------------------------------- layout

st.set_page_config(page_title="Flapping slice", layout="wide")

missing = [p for p in (INVENTORY, PREDICTIONS, MANIFEST) if not p.exists()]
if missing:
    st.error("Missing inputs:\n\n" + "\n".join(f"- `{p}`" for p in missing))
    st.code("python scripts/count_flap_sites.py --parquet "
            "results/flapping/sites-20260924.parquet\n"
            "python scripts/make_synthetic_predictions.py")
    st.stop()

STAMP = input_stamp()
manifest = load_manifest(STAMP)
render_guard(manifest)
store = load_store(STAMP)
stats = load_stats(STAMP)

st.error(
    "**SYNTHETIC RUN — the model scores on this page are made up.**  \n"
    "No probe has been trained yet. Every `p_t` below is a random number from a "
    "fixed seed. They are here so the plumbing can be tested end to end — which "
    "works just as well with noise, because the plumbing does not care whether a "
    "number means anything. **Nothing on this page is a finding about /t/ or /d/.**"
)

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
    f"seed {manifest.random_seed}"
)

with st.sidebar:
    st.header("Query")
    condition = st.selectbox("How train/test was split", CONDITIONS, index=1)
    st.caption(
        "`ordinary` — words can appear in both train and test.  \n"
        "`word_disjoint` — a word type is in one or the other, never both.  \n"
        "`combined` — word-split *and* speaker-split: the pre-registered design."
    )
    encoder = st.selectbox("Encoder", ["wavjepa"])
    layer = st.selectbox("Encoder layer", list(range(13)), index=8)
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

tab_evidence, tab_compare, tab_uncertainty, tab_state = st.tabs(
    ["Browse and listen", "Compare two splits",
     "What this can and can't tell you", "Look up one site"]
)

# ------------------------------------------------------------------- evidence

with tab_evidence:
    result = store.query_sites(condition=condition, encoder=encoder, layer=layer,
                               underlying=u, correct=c)
    st.info(coverage_sentence(result.coverage))
    st.caption(
        f"`{coverage_line(result.coverage)}` — shown on every answer, even when "
        "it tells you nothing new. It is the only thing stopping a 40-row table "
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
        options = {label_for(r): r.site_id for r in result.rows.itertuples()}
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
                st.metric("Model's score for /t/ (made up)", f"{row.p_t:.3f}")
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

            st.markdown("#### Your verdict")
            st.caption(
                "Four separate judgements, because they can disagree — the clip "
                "can be cut correctly while the label is wrong. There is no "
                "single approve button on purpose: the model's score here is "
                "random, so approving \"the answer\" would mean approving noise."
            )
            for target, question in REVIEW_TARGETS.items():
                cols = st.columns([3, 1, 1, 2])
                cols[0].markdown(f"**{target}** — {question}")
                note = cols[3].text_input("note", key=f"n_{target}_{pick}",
                                          label_visibility="collapsed")
                if cols[1].button("✓", key=f"y_{target}_{pick}"):
                    record_verdict(pick, target, "pass", note, manifest)
                    st.toast(f"{target}: pass")
                if cols[2].button("✗", key=f"x_{target}_{pick}"):
                    record_verdict(pick, target, "fail", note, manifest)
                    st.toast(f"{target}: fail")

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
    cond_a = a.selectbox("Compare this split…", CONDITIONS, index=0)
    cond_b = b.selectbox("…against this one", CONDITIONS, index=1)
    cmp_u = st.selectbox("Underlying sound ", ["either", "t", "d"], key="cmp_u",
                         format_func=lambda v: {"either": "either",
                                                "t": "/t/ (water, city)",
                                                "d": "/d/ (ladder, ready)"}[v])

    comp = store.compare_conditions(cond_a, cond_b, encoder=encoder, layer=layer,
                                    underlying=None if cmp_u == "either" else cmp_u)
    st.info(coverage_sentence(comp.coverage))
    st.caption(f"{comp.note} · `{coverage_line(comp.coverage)}`")
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
            "significance test, and not a claim about why. Saying why would "
            "need a trained probe, which does not exist yet."
        )
        shown = comp.rows.copy()
        shown.insert(1, "transcript", [transcript(r) for r in shown.itertuples()])
        st.dataframe(shown, width="stretch", hide_index=True)

# ----------------------------------------------------------------- uncertainty

with tab_uncertainty:
    st.write(
        "Two different things here are uncertain, and they are kept apart on "
        "purpose. One is real and can be quoted today. The other is noise."
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
    st.subheader("2 · How confident the model is — meaningless here")
    st.error(
        "Every `p_t` on this page is a random number. It is labelled wherever it "
        "appears. It is not weak evidence about /t/ and /d/ — it is no evidence, "
        "and averaging a lot of it together does not turn it into any."
    )
    st.caption(
        "These two are never added together into one confidence score. They come "
        "from different places and have different fixes — the first needs better "
        "alignment or recovering those thrown-out words, the second needs a real "
        "probe — and a combined number would hide which of the two had moved, "
        "leaving you unable to fix either."
    )

# ----------------------------------------------------------------- site lookup

with tab_state:
    st.write(
        "There are three reasons a site might have no score, and they mean "
        "different things. This page keeps them apart. Showing all three as a "
        "blank cell — or worse, as a zero — is the specific mistake this view "
        "exists to make impossible."
    )
    cond = st.selectbox("Under which split", CONDITIONS, index=2, key="state_cond")

    groups = load_states(STAMP, cond)
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
            state = store.site_state(site_id.strip(), cond)
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
                    "run only looked at 40 sites and this was not one of them. "
                    "Different from being excluded."
                )
            else:
                st.success(
                    f"**Scored.** The model guessed /{state['pred']}/ — "
                    f"{'right' if state['correct'] else 'wrong'} — with a made-up "
                    f"score of {state['p_t']:.3f} for /t/."
                )
            with st.expander("Raw record"):
                st.json(state)
