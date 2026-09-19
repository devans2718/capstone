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

INVENTORY = ROOT / "results" / "flapping" / "sites-20260919.parquet"
STATS = ROOT / "results" / "flapping" / "sites-20260919.stats.json"
PREDICTIONS = Path("/data/embeddings/predictions.parquet")
MANIFEST = Path("/data/embeddings/run-manifest.json")
REVIEWS = Path("/data/reviews/reviewed.jsonl")
CLIPS = Path("/data/clips")

# The four things a reviewer can judge. Deliberately not accept/reject: on a
# synthetic run the probe output carries no information, so "accept this answer"
# is meaningless and an interface offering it trains the reviewer to approve
# noise. Every verdict names which of four claims it is about.
REVIEW_TARGETS = {
    "label": "Is this a flap, and is the underlying /t/ vs /d/ assignment right?",
    "clip": "Does the clip contain the site, centred and uncut?",
    "retrieval": "Is this row responsive to the question asked?",
    "grounding": "Did the prose stay inside the returned fields?",
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


@st.cache_resource
def load_store() -> SliceStore:
    return SliceStore(INVENTORY, PREDICTIONS)


@st.cache_data
def load_manifest() -> Manifest:
    return Manifest.read(MANIFEST)


@st.cache_data
def load_stats() -> dict:
    return json.loads(STATS.read_text())


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
            "results/flapping/sites-20260919.parquet\n"
            "python scripts/make_synthetic_predictions.py")
    st.stop()

manifest = load_manifest()
render_guard(manifest)
store = load_store()
stats = load_stats()

st.error(
    "**SYNTHETIC RUN — these probabilities carry no information.** No probe has "
    "been trained. Every `p_t` below is a seeded random draw, kept because "
    "retrieval is exactly as testable against noise as against a real result. "
    "Nothing here is a finding about /t/ or /d/."
)

st.title("Underlying /t/ vs /d/ beneath a flap — evidence slice")
st.caption(
    f"run `{manifest.run_id}` · code `{manifest.code_commit[:12]}` · "
    f"inventory `{manifest.inventory_hash[:12]}` · checkpoint "
    f"`{manifest.checkpoint_revision}` · context {manifest.context_ms:.0f} ms · "
    f"seed {manifest.random_seed}"
)

with st.sidebar:
    st.header("Query")
    condition = st.selectbox("Condition", CONDITIONS, index=1)
    encoder = st.selectbox("Encoder", ["wavjepa"])
    layer = st.selectbox("Layer", list(range(13)), index=8)
    underlying = st.selectbox("Underlying class", ["either", "t", "d"])
    correctness = st.selectbox("Correctness", ["either", "correct", "misclassified"])
    st.divider()
    st.caption(f"Split scheme recorded in the manifest:\n\n`{manifest.split_scheme}`")

u = None if underlying == "either" else underlying
c = None if correctness == "either" else (correctness == "correct")

tab_evidence, tab_compare, tab_uncertainty, tab_state = st.tabs(
    ["Evidence", "Condition differential", "Uncertainty", "Site lookup"]
)

# ------------------------------------------------------------------- evidence

with tab_evidence:
    result = store.query_sites(condition=condition, encoder=encoder, layer=layer,
                               underlying=u, correct=c)
    st.markdown(f"### Coverage — {coverage_line(result.coverage)}")
    st.caption(
        "Shown on every answer, including when it is uninformative. It is the "
        "only thing that stops a 40-row table reading like a complete result."
    )
    if result.n == 0:
        st.info("**n = 0.** No sites match this filter. That is an answer, not "
                "an error — the denominators above still apply.")
    else:
        st.dataframe(result.rows, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Listen to one site")
        pick = st.selectbox("Site", result.rows.site_id.tolist())
        if pick:
            plan = store.plan_clip(pick)
            row = result.rows[result.rows.site_id == pick].iloc[0]
            a, b = st.columns([2, 3])
            with a:
                st.metric("p_t (synthetic)", f"{row.p_t:.3f}")
                st.write(f"**{row.word}** · underlying /{row.underlying}/ · "
                         f"predicted /{row.pred}/")
                st.write(f"context: `{row.prev_phone}` [dx] `{row.next_phone}`"
                         + (f" → *{row.next_word}*" if row.next_word else ""))
            with b:
                st.write(f"samples `{plan['start_sample']:,}`–`{plan['end_sample']:,}` "
                         f"({plan['n_samples']:,} = {plan['n_samples']/16:.0f} ms)")
                st.caption(
                    "Planned by `plan_excerpt` then `excerpt_samples` — the same "
                    "call `encode_excerpt` makes. This is the audio the encoder "
                    "would consume, not an approximation of it."
                )
                try:
                    clip = store.export_clip(pick, out_dir=CLIPS)
                    st.audio(str(clip))
                except FileNotFoundError as exc:
                    st.warning(f"No corpus audio mounted: {exc}")

            st.markdown("**Review — four separate verdicts, not accept/reject**")
            for target, question in REVIEW_TARGETS.items():
                cols = st.columns([3, 1, 1, 2])
                cols[0].caption(f"`{target}` — {question}")
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
    st.caption(
        "Does correctness change when word types stop recurring between train "
        "and test? A probe leaning on word identity should lose more here than "
        "anywhere else — that is what the word-disjoint split is for."
    )
    a, b = st.columns(2)
    cond_a = a.selectbox("Condition A", CONDITIONS, index=0)
    cond_b = b.selectbox("Condition B", CONDITIONS, index=1)
    cmp_u = st.selectbox("Underlying class ", ["either", "t", "d"], key="cmp_u")

    comp = store.compare_conditions(cond_a, cond_b, encoder=encoder, layer=layer,
                                    underlying=None if cmp_u == "either" else cmp_u)
    st.markdown(f"### Coverage — {coverage_line(comp.coverage)}")
    st.caption(comp.note)
    if comp.n == 0:
        st.info("**n = 0.** No site changed correctness between these two "
                "conditions under this filter.")
    else:
        flipped_out = int((comp.rows.correct_a & ~comp.rows.correct_b).sum())
        flipped_in = int((~comp.rows.correct_a & comp.rows.correct_b).sum())
        st.write(f"**{flipped_out}** correct under `{cond_a}` and incorrect under "
                 f"`{cond_b}`; **{flipped_in}** the other way.")
        st.caption(
            "Both counts are of returned rows. Nothing here is a rate, a "
            "significance claim, or a statement about why — those would need a "
            "denominator this run does not have and a probe that exists."
        )
        st.dataframe(comp.rows, width="stretch", hide_index=True)

# ----------------------------------------------------------------- uncertainty

with tab_uncertainty:
    st.subheader("1 · Label and population — real, measured, citable today")
    st.caption("Measured by `scripts/count_flap_sites.py` and read from "
               "`sites-20260919.stats.json`. Not transcribed into this file.")
    m = st.columns(4)
    excl = stats["excluded_length_mismatch"]
    m[0].metric("words dropped, length mismatch",
                f"{excl:,}", f"{excl / stats['flap_bearing_words']:.1%} of flap-bearing")
    m[1].metric("annotator-marked flaps", f"{stats['excluded_annotator_mark']:,}")
    m[2].metric("flaps not from /t/ or /d/", f"{stats['excluded_not_t_or_d']:,}")
    m[3].metric("clamped at 200 ms", f"{stats['n_excerpt_clamped']:,}")

    st.warning(
        f"**The retained set is the cleanly-aligned set.** The "
        f"{excl:,} excluded words ({excl / stats['flap_bearing_words']:.1%}) are the "
        "most reduced and most coarticulated tokens — which is exactly where "
        f"flapping is most interesting. No result on the remaining "
        f"{stats['n_sites']:,} sites can speak to them, however narrow its "
        "confidence interval. That is a bound on what any result here can mean, "
        "and it is knowable now, before any result exists."
    )

    z = st.columns(2)
    z[0].metric("sites with zero frame onsets — excerpt-local",
                f"{stats['n_zero_onset_excerpt_local']:,}")
    z[1].metric("— recording's own timebase", f"{stats['n_zero_onset_absolute']:,}")
    st.caption(
        "Both are shown because which is correct is an open question — see D1 in "
        "`docs/challenges/challenge-2-slice-decisions.md`. Showing one would be "
        "picking a side of an unresolved question by way of a layout choice."
    )

    st.divider()
    st.subheader("2 · Probe — synthetic, carries no information")
    st.error(
        "`p_t` and its margin exist in the schema and are rendered wherever they "
        "appear, labelled synthetic. They are seeded random draws. They are not "
        "evidence about /t/ and /d/ and cannot become evidence by being averaged."
    )
    st.caption(
        "These two channels are never combined into a single confidence figure. "
        "They have different provenance and different remedies — the first is "
        "addressed by better alignment or by recovering the excluded tokens, the "
        "second by more data or a better probe — and a blended number would make "
        "both unfixable, because you could no longer tell which had moved."
    )

# ----------------------------------------------------------------- site lookup

with tab_state:
    st.caption(
        "The three eligibility states, rendered distinctly. Collapsing any two "
        "of them into 'no data' is the defect this view exists to catch."
    )
    site_id = st.text_input("Site identifier", value="sNNNNx:NNNNN")
    cond = st.selectbox("Condition ", CONDITIONS, index=2, key="state_cond")
    if site_id:
        try:
            state = store.site_state(site_id.strip(), cond)
        except UnknownSite as exc:
            st.error(str(exc))
        else:
            if state["status"] == NOT_SCORED:
                st.warning(f"**{NOT_SCORED}** — no prediction exists for this "
                           "site under this condition. Not zero, not a "
                           "probability, not an inferred value.")
            elif state["status"] == NOT_REACHED:
                st.info(f"**{NOT_REACHED}** — this site is eligible; the run "
                        "scored 40 sites and did not include it.")
            else:
                st.success(f"**scored** — p_t {state['p_t']:.3f} (synthetic), "
                           f"predicted /{state['pred']}/, "
                           f"{'correct' if state['correct'] else 'incorrect'}")
            st.json(state)
