"""capstone.splits against the figures the pre-registration quotes.

THE GRID FIGURES ARE HARD-CODED, AND THAT IS THE POINT HERE
    The slice tests recompute their ground truth a second way. These do not,
    because the numbers below are registered text: §3 of the pre-registration
    and results/corpus-verification/buckeye-flap-lexical-structure-20260926.txt
    state them, and a change to the split that moves any of them is a change to
    the design, not a refactor. Failing loudly is the correct response.

THE GRID IS OVER THE 10,389 ELIGIBLE SITES, KEYED BY type_key
    The 3 clamped sites are removed before folds are assigned (decided
    2026-09-25), and contracted forms are merged onto their host word before
    ranking (decided 2026-09-26). The 20260924 figures (all sites, word forms)
    and the 20260925 ones (eligible sites, word forms) are superseded, not
    wrong.

READ FROM THE PARQUET, BECAUSE THE CLAMP FLAG LIVES THERE
    The TSV has no excerpt_clamped column, so these tests need pyarrow and run
    in .venv-app. The CUDA image skips them. One test checks the TSV and the
    parquet carry the same sites, so figures measured on either still hold.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn", reason="the random condition uses StratifiedKFold")
pq = pytest.importorskip("pyarrow.parquet", reason="the clamp flag is in the parquet")

from capstone.splits import (  # noqa: E402
    CONDITIONS,
    CONTRACTIONS,
    OWN_KEY,
    SUBSAMPLE_FRACTION,
    iter_splits,
    site_folds,
    type_key,
)

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "results" / "corpus-verification" / "buckeye-flap-sites-20260821.tsv"
PARQUET = REPO / "results" / "flapping" / "sites-20260924.parquet"
SEED = 12345  # a test seed; the registered one lives in the pre-registration

# Test sites and test /d/ sites per cell, rows = speaker fold, cols = type fold.
# Copied from the 20260926 analysis output.
TEST_SITES = [[478, 356, 346, 324, 310],
              [447, 492, 459, 353, 373],
              [401, 438, 398, 363, 319],
              [493, 506, 466, 421, 385],
              [535, 478, 462, 388, 398]]
TEST_D = [[66, 126, 69, 71, 86],
          [82, 152, 111, 66, 133],
          [70, 119, 98, 71, 92],
          [94, 189, 96, 97, 135],
          [93, 132, 94, 57, 130]]


@pytest.fixture(scope="module")
def inv():
    if not PARQUET.exists():
        pytest.skip(f"no inventory at {PARQUET}")
    t = pq.read_table(PARQUET, columns=["speaker", "word", "underlying",
                                        "excerpt_clamped"]).to_pandas()
    return (t.speaker.to_numpy(), t.word.to_numpy(), t.underlying.to_numpy(),
            ~t.excerpt_clamped.to_numpy())


def splits(inv, condition, **kw):
    spk, wrd, lab, eligible = inv
    kw.setdefault("eligible", eligible)
    if condition == "random":
        kw.setdefault("seed", SEED)
    return list(iter_splits(condition, spk, wrd, lab, **kw))


# ------------------------------------------------------- invariant 4: the grid


def test_joint_reproduces_the_registered_grid(inv):
    spk, wrd, lab, _ = inv
    cells = splits(inv, "joint")
    assert [c.fold for c in cells] == list(range(25))

    for c in cells:
        a, b = divmod(c.fold, 5)
        assert len(c.test) == TEST_SITES[a][b], (a, b)
        assert (lab[c.test] == "d").sum() == TEST_D[a][b], (a, b)

    n_train = [len(c.train) for c in cells]
    d_train = [int((lab[c.train] == "d").sum()) for c in cells]
    keys = np.array([type_key(w) for w in wrd])
    d_types = [len(set(keys[c.test][lab[c.test] == "d"])) for c in cells]
    assert (min(n_train), max(n_train)) == (6257, 7100)
    assert (min(d_train), max(d_train)) == (1389, 1820)
    assert (min(d_types), max(d_types)) == (25, 39)

    diag = [cells[6 * k] for k in range(5)]
    assert sum(len(c.test) for c in diag) == 2187
    assert sum(int((lab[c.test] == "d").sum()) for c in diag) == 543
    assert sum(len(c.test) for c in cells) == 10389
    assert sum(int((lab[c.test] == "d").sum()) for c in cells) == 2529


# ------------------------------------------------------ coverage and leakage


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_eligible_site_is_tested_exactly_once(inv, condition):
    eligible = inv[3]
    tested = np.concatenate([s.test for s in splits(inv, condition)])
    assert np.array_equal(np.sort(tested), np.flatnonzero(eligible))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_train_and_test_never_overlap_or_touch_excluded_sites(inv, condition):
    excluded = set(np.flatnonzero(~inv[3]))
    assert len(excluded) == 3
    for s in splits(inv, condition):
        assert not set(s.train) & set(s.test)
        assert not excluded & (set(s.train) | set(s.test))


@pytest.mark.parametrize("condition, speaker_held, word_held", [
    ("speaker_disjoint", True, False),
    ("word_disjoint", False, True),
    ("joint", True, True),
])
def test_no_leakage_on_the_constrained_axes(inv, condition, speaker_held, word_held):
    spk, wrd, _, _ = inv
    keys = np.array([type_key(w) for w in wrd])
    for s in splits(inv, condition):
        if speaker_held:
            assert not set(spk[s.test]) & set(spk[s.train]), s.fold
        if word_held:
            assert not set(keys[s.test]) & set(keys[s.train]), s.fold


def test_random_lets_word_types_recur(inv):
    """The condition's purpose, checked so it cannot become constrained by accident."""
    wrd = inv[1]
    for s in splits(inv, "random"):
        assert set(wrd[s.test]) & set(wrd[s.train])


def test_joint_train_is_the_intersection_of_the_single_axis_trains(inv):
    spk_train = {s.fold: set(s.train) for s in splits(inv, "speaker_disjoint")}
    wrd_train = {s.fold: set(s.train) for s in splits(inv, "word_disjoint")}
    for c in splits(inv, "joint"):
        a, b = divmod(c.fold, 5)
        assert set(c.train) == spk_train[a] & wrd_train[b]


# ---------------------------------------------------- exclusion before folds


@pytest.mark.parametrize("condition", CONDITIONS)
def test_folds_are_assigned_as_if_excluded_sites_never_existed(inv, condition):
    """Masking in place equals building the split over the filtered arrays."""
    spk, wrd, lab, eligible = inv
    idx = np.flatnonzero(eligible)
    kw = {"seed": SEED} if condition == "random" else {}
    masked = splits(inv, condition)
    filtered = list(iter_splits(condition, spk[idx], wrd[idx], lab[idx],
                                eligible=None, **kw))
    for m, f in zip(masked, filtered):
        assert np.array_equal(m.test, idx[f.test])
        assert np.array_equal(m.train, idx[f.train])


def test_counting_the_clamped_sites_would_move_folds(inv):
    """Why the order matters: the record behind the 2026-09-25 decision."""
    spk, wrd, _, eligible = inv
    over_all = site_folds("word_disjoint", spk, wrd)[eligible]
    over_eligible = site_folds("word_disjoint", spk[eligible], wrd[eligible])
    moved = {type_key(w) for w in wrd[eligible][over_all != over_eligible]}
    assert len(moved) == 186


def test_eligible_must_be_stated(inv):
    spk, wrd, lab, _ = inv
    with pytest.raises(TypeError, match="eligible"):
        list(iter_splits("joint", spk, wrd, lab))


# ------------------------------------------------------------------ random


def test_random_is_stratified_seeded_and_needs_its_seed(inv):
    spk, wrd, lab, eligible = inv
    n_d = int((lab[eligible] == "d").sum())
    for s in splits(inv, "random"):
        assert abs(int((lab[s.test] == "d").sum()) - n_d / 5) <= 1

    same = site_folds("random", spk, wrd, lab, seed=SEED)
    assert np.array_equal(same, site_folds("random", spk, wrd, lab, seed=SEED))
    assert not np.array_equal(same, site_folds("random", spk, wrd, lab, seed=SEED + 1))

    with pytest.raises(ValueError, match="seed"):
        site_folds("random", spk, wrd, lab)


# ------------------------------------------------------------- the subsample


@pytest.mark.parametrize("condition", ["random", "speaker_disjoint", "word_disjoint"])
def test_subsample_is_a_stratified_fraction_of_the_training_set(inv, condition):
    lab = inv[2]
    full = splits(inv, condition, seed=SEED)
    sub = splits(inv, condition, seed=SEED, subsample=True)
    again = splits(inv, condition, seed=SEED, subsample=True)

    for f, s, g in zip(full, sub, again):
        assert s.subsampled and not f.subsampled
        assert np.array_equal(s.test, f.test)
        assert set(s.train) <= set(f.train)
        assert np.array_equal(s.train, g.train)
        for cls in ("t", "d"):
            n_full = int((lab[f.train] == cls).sum())
            assert int((lab[s.train] == cls).sum()) == round(SUBSAMPLE_FRACTION * n_full)


def test_subsample_refuses_joint_and_a_missing_seed(inv):
    with pytest.raises(ValueError, match="joint"):
        splits(inv, "joint", subsample=True)
    with pytest.raises(ValueError, match="seed"):
        splits(inv, "word_disjoint", subsample=True)


# --------------------------------------------------- the two inventory files


def test_the_tsv_and_the_parquet_carry_the_same_sites(inv):
    """Figures measured on the TSV (§1, §1a) hold for the parquet the run reads."""
    if not TSV.exists():
        pytest.skip(f"no inventory at {TSV}")
    with open(TSV, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    spk, wrd, lab, _ = inv
    assert [r["speaker"] for r in rows] == list(spk)
    assert [r["word"] for r in rows] == list(wrd)
    assert [r["underlying"] for r in rows] == list(lab)


# ------------------------------------------------------------- the type key


def test_every_contracted_form_is_classified(inv):
    """A new contraction in the inventory must be placed, not silently keyed alone."""
    forms = {w.lower() for w in inv[1] if "'" in w}
    assert forms == set(CONTRACTIONS) | OWN_KEY
    assert not set(CONTRACTIONS) & OWN_KEY


def test_merged_forms_carry_their_hosts_labels(inv):
    """Evidence the flap is in the host: a merged form's labels are the host's.

    A subset rather than equality, because a host can carry two flaps
    (`today` has /t/ and /d/; `today's` was only transcribed with /d/).
    """
    _, wrd, lab, _ = inv
    labels = {}
    for w, y in zip(wrd, lab):
        labels.setdefault(w.lower(), set()).add(y)
    for form, host in CONTRACTIONS.items():
        if host in labels:
            assert labels[form] <= labels[host], (form, host)


def test_type_key_lowercases_then_merges():
    assert type_key("Daddy") == type_key("daddy's") == "daddy"
    assert type_key("don't") == "do"
    assert type_key("i'd") == "i'd"
    assert type_key("water") == "water"
