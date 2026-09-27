"""The control task and the word-identity lookup (pre-registration §4).

THE MAPPING PROCEDURE IS WRITTEN ONCE, HERE
    Criterion 2 compares the real control score against 10,000 redrawn
    mappings. A redraw made by a different procedure from the real draw would
    test against the wrong distribution. So control_mapping and null_mappings
    share one definition of the draw:

      1. The type keys of the eligible sites, in code-point order (Python's
         default sort).
      2. One uniform per type from numpy's default_rng, in that order. A type
         is /d/ if its draw is below the eligible /d/ share (2,529 / 10,389),
         and /t/ otherwise.

    The real mapping is one draw with seeds.CONTROL. The nulls are 10,000
    consecutive draws from a single generator seeded with seeds.CONTROL_NULL.
    rng.random((B, T)) fills row-major, so row b is exactly the b-th
    consecutive draw of T uniforms.

THE LOOKUP (§4, with the §7 clarification of 2026-09-26)
    A table from type key to its majority label among the probe's training
    sites. A type never seen in training gets the training set's majority
    label. A type tied between the classes in training gets it too. The §7
    entry covers unseen types. It says nothing of ties, and giving them the
    same fallback is the only rule that adds nothing new.
"""

from __future__ import annotations

N_NULL = 10_000


def type_index(keys):
    """(sorted unique keys, per-site index into them). Code-point order."""
    import numpy as np

    types = sorted(set(keys))
    pos = {k: i for i, k in enumerate(types)}
    return types, np.array([pos[k] for k in keys], dtype=np.int64)


def d_share(is_d) -> float:
    import numpy as np

    is_d = np.asarray(is_d, dtype=bool)
    return int(is_d.sum()) / len(is_d)


def control_mapping(n_types: int, share: float, seed: int):
    """One draw: bool per type (True = /d/), types in code-point order."""
    import numpy as np

    return np.random.default_rng(seed).random(n_types) < share


def null_mappings(n_types: int, share: float, seed: int, n: int = N_NULL):
    """n consecutive draws from one generator: (n, n_types) bool."""
    import numpy as np

    return np.random.default_rng(seed).random((n, n_types)) < share


def lookup_predict(train_types, train_is_d, test_types):
    """Word-identity lookup: bool per test site (True = /d/)."""
    import numpy as np

    train_types = np.asarray(train_types)
    train_is_d = np.asarray(train_is_d, dtype=bool)
    test_types = np.asarray(test_types)
    n_types = int(max(train_types.max(initial=-1), test_types.max(initial=-1))) + 1

    majority_d = train_is_d.mean() > 0.5
    n_d = np.bincount(train_types, weights=train_is_d, minlength=n_types)
    n_all = np.bincount(train_types, minlength=n_types)
    n_t = n_all - n_d
    table = np.full(n_types, majority_d)          # unseen, or tied
    table[n_d > n_t] = True
    table[n_t > n_d] = False
    return table[test_types]
