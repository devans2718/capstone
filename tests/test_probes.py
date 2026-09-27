"""The probe run's statistics, checked against slower restatements of themselves.

Each test compares a vectorised implementation with a naive one, or with a
figure the pre-registration states. None needs /data or a checkpoint, apart
from the lookup test, which reads the committed inventory and skips without
it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn is in both environments; skip if absent")

from capstone import seeds  # noqa: E402
from capstone.probes import criteria as C  # noqa: E402
from capstone.probes.controls import (control_mapping, lookup_predict,  # noqa: E402
                                      null_mappings, type_index)
from capstone.probes.scoring import (balanced_accuracy,  # noqa: E402
                                     boot_balanced_accuracy, resample_weights)

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "flapping" / "sites-20260924.parquet"


# ------------------------------------------------------------- control mapping


def test_control_mapping_follows_the_registered_procedure():
    keys = ["that", "a", "Zed", "b", "a", "ça"]
    types, idx = type_index(keys)
    assert types == sorted(set(keys))                    # code-point order
    assert types == ["Zed", "a", "b", "that", "ça"]
    share = 2529 / 10389
    got = control_mapping(len(types), share, seeds.CONTROL)
    u = np.random.default_rng(seeds.CONTROL)
    want = np.array([u.random() < share for _ in types])  # one uniform per type, in order
    assert (got == want).all()
    assert (control_mapping(len(types), share, seeds.CONTROL) == got).all()


def test_null_mappings_are_consecutive_draws_from_one_generator():
    nulls = null_mappings(7, 0.3, seeds.CONTROL_NULL, n=5)
    rng = np.random.default_rng(seeds.CONTROL_NULL)
    for row in nulls:
        assert (row == (rng.random(7) < 0.3)).all()


# ---------------------------------------------------------------------- lookup


def test_lookup_uses_the_training_majority_for_unseen_and_tied_types():
    train_t = np.array([0, 0, 0, 1, 1, 2, 2])
    train_d = np.array([True, True, False, False, False, True, False])  # 0: d, 1: t, 2: tie
    pred = lookup_predict(train_t, train_d, np.array([0, 1, 2, 3]))
    assert pred.tolist() == [True, False, False, False]  # tie and unseen -> majority /t/


@pytest.mark.skipif(not INVENTORY.exists(), reason="no committed inventory")
def test_lookup_in_sample_reproduces_the_lexical_ceiling():
    """§1a: word identity alone scores 0.9951 on the task (raw word forms, all
    10,392 sites, plain accuracy). Keyed by type_key over the 10,389 eligible
    sites it should land within a hair of that."""
    pq = pytest.importorskip("pyarrow.parquet")
    from capstone.splits import type_key

    inv = pq.read_table(INVENTORY).to_pandas()
    inv = inv[~inv.excerpt_clamped]
    _, idx = type_index([type_key(w) for w in inv.word])
    y = (inv.underlying == "d").to_numpy()
    acc = (lookup_predict(idx, y, idx) == y).mean()
    assert abs(acc - 0.9951) < 0.002


# ------------------------------------------------------------------- bootstrap


def _toy(seed=0, n=400, T=40, R=3):
    rng = np.random.default_rng(seed)
    tix = rng.integers(0, T, n)
    y = rng.random(n) < 0.3
    correct = rng.random((R, n)) < np.array([0.6, 0.7, 0.8])[:, None]
    return tix, y, correct


def test_vectorised_bootstrap_equals_a_naive_resample_of_sites():
    tix, y, correct = _toy()
    W = resample_weights(40, seed=1, n=25)
    fast = boot_balanced_accuracy(W, tix, y, correct)
    for b in range(len(W)):
        # Materialise the resample: each site repeated as often as its type was drawn.
        reps = W[b, tix].astype(int)
        yy = np.repeat(y, reps)
        for r in range(correct.shape[0]):
            cc = np.repeat(correct[r], reps)
            naive = 0.5 * (cc[yy].mean() + cc[~yy].mean())
            assert fast[b, r] == pytest.approx(naive)


def test_resample_weights_draw_T_types_with_replacement():
    W = resample_weights(40, seed=2, n=100)
    assert W.shape == (100, 40) and (W.sum(axis=1) == 40).all()
    again = resample_weights(40, seed=2, n=100)
    assert (W == again).all()


def test_criterion_1_is_paired_every_resample_scores_both_on_the_same_types():
    """If the bootstrap were unpaired, a representation compared with an exact
    copy of itself would show a spread of gaps. Paired, every gap is 0."""
    tix, y, correct = _toy()
    W = resample_weights(40, seed=3, n=200)
    _, nums = C.criterion_1(W, tix, y, correct[[1]], correct[1])
    assert nums["gap"] == 0 and nums["gap_ci_lo"] == 0 and nums["gap_ci_hi"] == 0


def test_criterion_1_reselects_the_best_layer_inside_each_resample():
    tix, y, correct = _toy()
    W = resample_weights(40, seed=4, n=300)
    boot = boot_balanced_accuracy(W, tix, y, np.vstack([correct, correct[0][None]]))
    want = np.quantile(boot[:, :-1].max(axis=1) - boot[:, -1], 0.025)
    _, nums = C.criterion_1(W, tix, y, correct, correct[0])
    assert nums["gap_ci_lo"] == pytest.approx(want)


# ------------------------------------------------------------------ criterion 2


def test_criterion_2_fires_about_5_percent_of_the_time_with_no_leak():
    """Predictions independent of the mapping: the real draw is exchangeable
    with the nulls, so the test should fire at its nominal rate."""
    rng = np.random.default_rng(0)
    n, T, L = 1500, 150, 3
    tix = rng.integers(0, T, n)
    nulls = null_mappings(T, 0.25, 11, n=400)
    fired = 0
    trials = 200
    for t in range(trials):
        real = rng.random(T) < 0.25
        preds = rng.random((L, n)) < 0.3          # no route from labels to predictions
        c, _ = C.criterion_2(preds, tix, real, nulls)
        fired += not c.passed
    assert 0.01 <= fired / trials <= 0.10


def test_criterion_2_fires_on_a_leak():
    rng = np.random.default_rng(1)
    n, T = 1500, 150
    tix = rng.integers(0, T, n)
    real = rng.random(T) < 0.25
    preds = np.vstack([real[tix], rng.random(n) < 0.3])   # layer 0 reads the labels
    c, nums = C.criterion_2(preds, tix, real, null_mappings(T, 0.25, 12, n=500))
    assert not c.passed and nums["control_max_layer"] == 0


def test_criterion_2_null_scoring_matches_a_direct_computation():
    rng = np.random.default_rng(2)
    n, T = 300, 30
    tix = rng.integers(0, T, n)
    preds = rng.random((2, n)) < 0.4
    nulls = null_mappings(T, 0.25, 13, n=50)
    _, nums = C.criterion_2(preds, tix, nulls[0], nulls)
    direct = [max(balanced_accuracy(m[tix], p) for p in preds) for m in nulls]
    assert nums["null_q95"] == pytest.approx(np.quantile(direct, 0.95))


# ------------------------------------------------------------ the linear probe


def test_probe_returns_p_d_and_convergence():
    from capstone.probes.linear import fit_predict

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X[:, 0] + 0.3 * rng.normal(size=300) > 0.8
    p, converged = fit_predict(X[:200], y[:200], X[200:])
    assert converged and p.shape == (100,) and ((p >= 0) & (p <= 1)).all()
    assert balanced_accuracy(y[200:], p > 0.5) > 0.8
