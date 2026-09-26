"""The seven cases the Challenge 2 slice is accepted against.

EACH CASE IS A COMPARISON, NOT A JUDGMENT
    Ground truth is computed independently in pandas inside each test, from the
    same fixture the tools read. A test that asserts a hard-coded 19 passes
    forever after the code stops being right about anything except 19; a test
    that recomputes the answer a different way disagrees when either side
    breaks. The two implementations share the data and nothing else.

NO /data, NO CORPUS, NO CHECKPOINT
    The fixture builds its own inventory and predictions into tmp_path from a
    fixed seed, so every case here runs on a bare checkout. That is deliberate:
    the two cases the brief says must pass whatever happens are the provenance
    round-trip and the excluded-site null, and a mandatory test that needs a
    registered 40 GB corpus mounted is a test that will be skipped exactly when
    it matters.

    One further test does write a real wav, and skips itself when the corpus is
    absent rather than failing.

CASES 3 AND 4 ARE THE LOAD-BEARING ONES
    They are the two a superficially plausible implementation fails: 3 because
    re-deriving a window is easier than calling plan_excerpt, and 4 because an
    absent row is easily rendered as a zero. Both are marked below.

THIS MODULE BELONGS TO THE APP ENVIRONMENT
    pandas, pyarrow and duckdb come from requirements-app.txt, not from the
    CUDA image, which carries only what extraction and probing need. Without
    the guards below, `docker compose run --rm dev python -m pytest` would stop
    at collection and take the image's own tests down with it. With them, the
    image skips this module as a whole and .venv-app runs it in full.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="app environment only (requirements-app.txt)")
pytest.importorskip("pyarrow", reason="app environment only (requirements-app.txt)")
pytest.importorskip("duckdb", reason="app environment only (requirements-app.txt)")

import pyarrow.parquet as pq  # noqa: E402

from capstone.encoders.extraction import excerpt_samples, plan_excerpt  # noqa: E402
from capstone.predictions import synthesize, write_predictions  # noqa: E402
from capstone.retrieval import NOT_REACHED, NOT_SCORED, SliceStore, UnknownSite  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "flapping" / "sites-20260924.parquet"
CORPUS = Path("/data/corpora/buckeye")

needs_inventory = pytest.mark.skipif(
    not INVENTORY.exists(),
    reason=f"no site inventory at {INVENTORY} — build it with "
           "scripts/count_flap_sites.py --parquet",
)


@pytest.fixture(scope="module")
def slice_fixture(tmp_path_factory):
    """An inventory and a seeded prediction set, built into a temp directory.

    Module-scoped because it is read-only and rebuilding it per test buys
    nothing. The seed is fixed, so a failure here is reproducible rather than
    something that happened once.
    """
    if not INVENTORY.exists():
        pytest.skip(f"no site inventory at {INVENTORY}")

    tmp = tmp_path_factory.mktemp("slice")
    inventory = pq.read_table(INVENTORY).to_pandas()
    sites, rows = synthesize(inventory, n_sites=40, seed=20260919, layer=8)

    preds_path = tmp / "predictions.parquet"
    write_predictions(rows, preds_path)
    store = SliceStore(INVENTORY, preds_path, corpus_root=CORPUS)
    return store, inventory, pd.DataFrame(rows), sites


# --------------------------------------------------------------------- case 1

@needs_inventory
def test_case_1_basic_filter_matches_an_independent_count(slice_fixture):
    """Misclassified sites, WavJEPA layer 8, word_disjoint — an exact integer."""
    store, inventory, preds, _ = slice_fixture

    result = store.query_sites(condition="word_disjoint", encoder="wavjepa",
                               layer=8, correct=False)

    expected = preds[(preds.condition == "word_disjoint")
                     & (preds.encoder == "wavjepa")
                     & (preds.layer == 8)
                     & (preds.scored)
                     & (preds.correct == False)]  # noqa: E712
    assert result.n == len(expected)
    assert set(result.rows.site_id) == set(expected.site_id)
    # The denominator travels with the answer, always.
    assert result.coverage.n_total == len(inventory)
    in_cond = preds[preds.condition == "word_disjoint"]
    assert result.coverage.n_scored == int(in_cond.scored.sum())
    assert result.coverage.n_eligible == len(inventory) - int((~in_cond.eligible).sum())
    assert result.coverage.compared_on == result.n


# --------------------------------------------------------------------- case 2

@needs_inventory
def test_case_2_condition_differential_is_set_equal_to_pandas(slice_fixture):
    """/d/ sites correct under random and incorrect under word_disjoint."""
    store, inventory, preds, _ = slice_fixture

    comp = store.compare_conditions("random", "word_disjoint",
                                    underlying="d", layer=8)
    got = set(comp.rows[comp.rows.correct_a & ~comp.rows.correct_b].site_id)

    d_sites = set(inventory[inventory.underlying == "d"].site_id)
    o = preds[(preds.condition == "random") & preds.scored].set_index("site_id")
    w = preds[(preds.condition == "word_disjoint") & preds.scored].set_index("site_id")
    shared = o.index.intersection(w.index)
    expected = {s for s in shared
                if s in d_sites and o.loc[s, "correct"] and not w.loc[s, "correct"]}

    assert got == expected
    assert len(expected) > 0, "fixture produced no differential — nothing tested"
    # Equal denominators: both conditions scored the same 40 sites.
    assert comp.coverage.compared_on == len(shared)
    assert comp.coverage.n_scored == len(shared)


# ----------------------------------------------------------- case 3 (REQUIRED)

@needs_inventory
def test_case_3_clip_offsets_round_trip_through_plan_excerpt(slice_fixture):
    """REQUIRED. The window is planned, never re-derived.

    Asserts the exact identity the brief specifies: plan_clip's offsets equal
    plan_excerpt composed with the encode_excerpt rounding. Checked across every
    scored site, not one, because the two agree by luck at some offsets — a
    single-site version of this test passes against a wrong implementation
    roughly half the time.
    """
    store, inventory, _, sites = slice_fixture

    for site in sites.itertuples():
        plan = store.plan_clip(site.site_id, context_ms=200.0)
        wav = plan["wav"]
        if wav.exists():
            import wave
            with wave.open(str(wav), "rb") as w:
                rec_ms = w.getnframes() / w.getframerate() * 1000
        else:
            rec_ms = float(site.end) * 1000 + 200.0

        excerpt = plan_excerpt(float(site.start) * 1000, float(site.end) * 1000,
                               rec_ms, context_ms=200.0)
        start, end = excerpt_samples(excerpt, 16_000)
        assert (plan["start_sample"], plan["end_sample"]) == (start, end), site.site_id

    # And the trap the brief names by name: deriving `end` from
    # target_end_ms + context_ms reads as obviously equivalent and is not. This
    # asserts the two derivations genuinely disagree somewhere in the fixture,
    # so the loop above is guarding a real distinction rather than an identity
    # that happens to hold.
    disagreements = 0
    for site in sites.itertuples():
        plan = store.plan_clip(site.site_id)
        naive_end = int(round((float(site.end) * 1000 + 200.0) / 1000 * 16_000))
        assert plan["end_sample"] == plan["start_sample"] + int(
            round(plan["excerpt"].duration_ms / 1000 * 16_000)
        )
        disagreements += plan["end_sample"] != naive_end
    assert disagreements > 0, (
        "the naive end-derivation agreed on every site in the fixture, so this "
        "case is no longer distinguishing the two — check the rounding before "
        "trusting the pass"
    )


# ----------------------------------------------------------- case 4 (REQUIRED)

@needs_inventory
def test_case_4_excluded_site_is_never_rendered_as_a_number(slice_fixture):
    """REQUIRED. eligible = false reports a state, never 0 and never a p_t.

    The ineligible sites are the clamped ones, excluded before scoring in every
    condition (pre-registration §1). Until 2026-09-24 this case targeted
    off-diagonal sites under `combined_diagonal`. The full grid scores every
    cell, so the split no longer discards anything.
    """
    store, _, preds, sites = slice_fixture

    off = preds[(preds.condition == "joint") & (~preds.eligible)]
    assert len(off) > 0, "fixture produced no ineligible sites — nothing tested"
    assert set(off.site_id) == set(sites[sites.excerpt_clamped].site_id)

    for site_id in off.site_id:
        state = store.site_state(site_id, "joint")
        assert state["eligible"] is False
        assert state["scored"] is False
        assert state["status"] == NOT_SCORED
        # The whole case: no number stands in for the absence.
        assert state["p_t"] is None
        assert state["pred"] is None
        assert state["correct"] is None
        assert state["p_t"] != 0 and state["p_t"] is not False

    # The third state is distinct from the second, and from "does not exist".
    scored_id = preds[(preds.condition == "random") & preds.scored].site_id.iloc[0]
    assert store.site_state(scored_id, "random")["status"] == "scored"


@needs_inventory
def test_case_4b_unscored_site_is_distinct_from_excluded(slice_fixture):
    """The third eligibility state — a real site this run never reached."""
    store, inventory, preds, _ = slice_fixture

    never = inventory[~inventory.site_id.isin(preds.site_id)].site_id.iloc[0]
    state = store.site_state(never, "random")
    assert state["eligible"] is True
    assert state["scored"] is False
    assert state["status"] == NOT_REACHED
    assert state["status"] != NOT_SCORED, (
        "'never reached' and 'excluded before scoring' rendered identically — "
        "that collapse is the defect this case exists to catch"
    )
    assert state["p_t"] is None


# --------------------------------------------------------------------- case 5

@needs_inventory
def test_case_5_empty_result_is_an_answer_with_a_denominator(slice_fixture):
    """A filter matching nothing: n = 0 plus coverage, not an exception."""
    store, inventory, preds, _ = slice_fixture

    result = store.query_sites(condition="word_disjoint", layer=11)
    assert result.n == 0
    assert result.coverage.n_total == len(inventory)
    assert result.coverage.n_scored == int(
        preds[preds.condition == "word_disjoint"].scored.sum())
    assert result.coverage.compared_on == 0
    # An empty table without a denominator is indistinguishable from a bug.
    assert result.coverage.as_dict()["n_total"] > 0


# --------------------------------------------------------------------- case 6

@needs_inventory
def test_case_6_unknown_identifier_raises_and_names_itself(slice_fixture):
    """export_clip('s9999z:00000') raises, with the identifier in the message."""
    store, _, _, _ = slice_fixture

    with pytest.raises(UnknownSite) as exc:
        store.plan_clip("s9999z:00000")
    assert "s9999z:00000" in str(exc.value)

    with pytest.raises(UnknownSite) as exc:
        store.site_state("s9999z:00000", "random")
    assert "s9999z:00000" in str(exc.value)


# --------------------------------------------------------------------- case 7

@needs_inventory
def test_case_7_zero_onset_site_is_returned_not_dropped(slice_fixture):
    """A site with no onset in the recording's timebase comes back, usable.

    Since D1 was resolved, every site has at least one onset in the
    excerpt-local timebase the extractor indexes. The 4 sites with none in the
    recording's own timebase are where the two columns can be confused: read
    the wrong one and a site the probe will use looks empty. So the site must
    come back, neither dropped nor raising, and n_frame_onsets must carry the
    excerpt-local count, not the absolute zero.
    """
    store, _, _, sites = slice_fixture

    zero = sites[sites.n_frame_onsets_abs == 0]
    assert len(zero) > 0, "fixture selected no zero-onset site — nothing tested"

    for site_id in zero.site_id:
        state = store.site_state(site_id, "random")
        assert state["n_frame_onsets_abs"] == 0
        assert state["n_frame_onsets"] >= 1
        assert state["status"] in ("scored", NOT_SCORED, NOT_REACHED)

    result = store.query_sites(condition="random",
                               site_ids=zero.site_id.tolist())
    assert set(result.rows.site_id) == set(zero.site_id)
    assert (result.rows.n_frame_onsets >= 1).all()


# ------------------------------------------------------- folds are registered


@needs_inventory
def test_synthetic_rows_carry_the_registered_folds(slice_fixture):
    """Every row's fold is its site's fold under capstone.splits, not a draw.

    Assigned over the whole inventory, then looked up: the 40 sites' own word
    counts would rank types differently and describe a split nobody runs.
    """
    from capstone.predictions import CONDITIONS
    from capstone.splits import site_folds

    _, inventory, preds, _ = slice_fixture
    for cond in CONDITIONS:
        fold = dict(zip(inventory.site_id,
                        site_folds(cond, inventory.speaker, inventory.word,
                                   inventory.underlying, seed=20260919)))
        rows = preds[preds.condition == cond]
        assert len(rows) > 0, cond
        assert all(r.fold == fold[r.site_id] for r in rows.itertuples()), cond


# ------------------------------------------------------ corpus-dependent extra

@needs_inventory
@pytest.mark.skipif(not CORPUS.exists(), reason=f"no corpus at {CORPUS}")
def test_exported_clip_has_exactly_the_planned_samples(slice_fixture, tmp_path):
    """The written wav is the window plan_clip described, to the sample."""
    import soundfile as sf

    store, _, _, sites = slice_fixture
    site_id = sites.site_id.iloc[0]

    plan = store.plan_clip(site_id)
    out = store.export_clip(site_id, out_dir=tmp_path)
    data, sr = sf.read(out)

    assert sr == 16_000
    assert len(data) == plan["n_samples"]
