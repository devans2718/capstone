"""capstone.splits against the figures the pre-registration quotes.

THE GRID FIGURES ARE HARD-CODED, AND THAT IS THE POINT HERE
    The slice tests recompute their ground truth a second way. These do not,
    because the numbers below are registered text: §3 of the pre-registration
    and results/corpus-verification/buckeye-flap-lexical-structure-20260924.txt
    state them, and a change to the split that moves any of them is a change to
    the design, not a refactor. Failing loudly is the correct response.

READ FROM THE TSV, NOT THE PARQUET
    The registered figures were computed from the TSV inventory, and reading it
    needs only csv, so the grid tests run in the CUDA image as well as .venv-app.
    One test checks the parquet the slice reads yields the same folds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn", reason="the random condition uses StratifiedKFold")

from capstone.splits import (  # noqa: E402
    CONDITIONS,
    SUBSAMPLE_FRACTION,
    iter_splits,
    site_folds,
)

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "results" / "corpus-verification" / "buckeye-flap-sites-20260821.tsv"
PARQUET = REPO / "results" / "flapping" / "sites-20260924.parquet"
SEED = 12345  # a test seed; the registered one lives in the pre-registration

# Test sites and test /d/ sites per cell, rows = speaker fold, cols = type fold.
# Copied from the 20260924 analysis output.
TEST_SITES = [[479, 381, 339, 335, 281],
              [456, 487, 458, 376, 347],
              [395, 424, 405, 346, 350],
              [505, 492, 475, 379, 421],
              [535, 481, 451, 407, 387]]
TEST_D = [[87, 99, 69, 73, 90],
          [95, 115, 135, 67, 132],
          [88, 99, 102, 85, 76],
          [101, 121, 151, 96, 142],
          [95, 101, 94, 103, 113]]


@pytest.fixture(scope="module")
def inv():
    if not TSV.exists():
        pytest.skip(f"no inventory at {TSV}")
    with open(TSV, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return (np.array([r["speaker"] for r in rows]),
            np.array([r["word"] for r in rows]),
            np.array([r["underlying"] for r in rows]))


def splits(inv, condition, **kw):
    spk, wrd, lab = inv
    if condition == "random":
        kw.setdefault("seed", SEED)
    return list(iter_splits(condition, spk, wrd, lab, **kw))


# ------------------------------------------------------- invariant 4: the grid


def test_joint_reproduces_the_registered_grid(inv):
    spk, wrd, lab = inv
    cells = splits(inv, "joint")
    assert [c.fold for c in cells] == list(range(25))

    for c in cells:
        a, b = divmod(c.fold, 5)
        assert len(c.test) == TEST_SITES[a][b], (a, b)
        assert (lab[c.test] == "d").sum() == TEST_D[a][b], (a, b)

    n_train = [len(c.train) for c in cells]
    d_train = [int((lab[c.train] == "d").sum()) for c in cells]
    d_types = [len(set(wrd[c.test][lab[c.test] == "d"])) for c in cells]
    assert (min(n_train), max(n_train)) == (6255, 7072)
    assert (min(d_train), max(d_train)) == (1504, 1760)
    assert (min(d_types), max(d_types)) == (27, 48)

    diag = [cells[6 * k] for k in range(5)]
    assert sum(len(c.test) for c in diag) == 2137
    assert sum(int((lab[c.test] == "d").sum()) for c in diag) == 513
    assert sum(len(c.test) for c in cells) == 10392
    assert sum(int((lab[c.test] == "d").sum()) for c in cells) == 2529


# ------------------------------------------------------ coverage and leakage


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_site_is_tested_exactly_once(inv, condition):
    tested = np.concatenate([s.test for s in splits(inv, condition)])
    assert np.array_equal(np.sort(tested), np.arange(len(inv[0])))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_train_and_test_never_overlap(inv, condition):
    for s in splits(inv, condition):
        assert not set(s.train) & set(s.test)


@pytest.mark.parametrize("condition, speaker_held, word_held", [
    ("speaker_disjoint", True, False),
    ("word_disjoint", False, True),
    ("joint", True, True),
])
def test_no_leakage_on_the_constrained_axes(inv, condition, speaker_held, word_held):
    spk, wrd, _ = inv
    for s in splits(inv, condition):
        if speaker_held:
            assert not set(spk[s.test]) & set(spk[s.train]), s.fold
        if word_held:
            assert not set(wrd[s.test]) & set(wrd[s.train]), s.fold


def test_random_lets_word_types_recur(inv):
    """The condition's purpose, checked so it cannot become constrained by accident."""
    _, wrd, _ = inv
    for s in splits(inv, "random"):
        assert set(wrd[s.test]) & set(wrd[s.train])


def test_joint_train_is_the_intersection_of_the_single_axis_trains(inv):
    spk_train = {s.fold: set(s.train) for s in splits(inv, "speaker_disjoint")}
    wrd_train = {s.fold: set(s.train) for s in splits(inv, "word_disjoint")}
    for c in splits(inv, "joint"):
        a, b = divmod(c.fold, 5)
        assert set(c.train) == spk_train[a] & wrd_train[b]


# ------------------------------------------------------------------ random


def test_random_is_stratified_seeded_and_needs_its_seed(inv):
    spk, wrd, lab = inv
    n_d = int((lab == "d").sum())
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
    _, _, lab = inv
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


# ------------------------------------------------------------- eligibility


@pytest.mark.parametrize("condition", CONDITIONS)
def test_excluded_sites_leave_without_moving_any_fold(inv, condition):
    n = len(inv[0])
    eligible = np.arange(n) % 97 != 0
    full = splits(inv, condition)
    kept = splits(inv, condition, eligible=eligible)

    for f, k in zip(full, kept):
        assert np.array_equal(k.test, f.test[eligible[f.test]])
        assert np.array_equal(k.train, f.train[eligible[f.train]])
    tested = np.concatenate([s.test for s in kept])
    assert np.array_equal(np.sort(tested), np.flatnonzero(eligible))


# ------------------------------------------------------ the slice's inventory


@pytest.mark.parametrize("condition", ["speaker_disjoint", "word_disjoint", "joint"])
def test_the_slice_parquet_gives_the_same_folds(inv, condition):
    pq = pytest.importorskip("pyarrow.parquet")
    if not PARQUET.exists():
        pytest.skip(f"no inventory at {PARQUET}")
    sites = pq.read_table(PARQUET, columns=["speaker", "word"]).to_pandas()
    assert np.array_equal(
        site_folds(condition, sites.speaker, sites.word),
        site_folds(condition, inv[0], inv[1]),
    )
