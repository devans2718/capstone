"""Pooled scores (pre-registration §3, §5) and the paired word-type bootstrap.

POOLED, NOT AVERAGED
    Every score here is computed once over a pooled set of out-of-fold
    predictions, never as a mean of per-fold scores (§3).

THE BOOTSTRAP RESAMPLES WORD TYPES, AND IS PAIRED
    Word types are the unit the design generalizes over (§5), so a resample
    draws T type keys with replacement from the T present, and each site
    enters as many times as its type was drawn. Written as counts, a resample
    is a weight vector w over types, and any pooled balanced accuracy under
    it is

        BA(w) = 1/2 * ( w . correct_d / w . n_d  +  w . correct_t / w . n_t )

    where correct_d[k] counts type k's /d/ sites scored correctly. So every
    score in a resample is a matrix product with the same W, (B, T). That is
    what makes the bootstrap PAIRED (§7, 2026-09-26): WavJEPA's 13 layers and
    the MFCC baseline are scored on the same resampled types in every
    resample, because they are rows of one `correct` matrix multiplied by one
    W. A gap computed per resample is then a difference measured on one
    sample, not two independent intervals subtracted.

    W is drawn from seeds.BOOTSTRAP for whatever site set is being
    resampled. The same site set gets the same W wherever it is scored, so
    every per-layer interval on the 10,389 sites shares one set of
    resamples.
"""

from __future__ import annotations

N_BOOT = 10_000


def balanced_accuracy(is_d, pred_d) -> float:
    import numpy as np

    is_d, pred_d = np.asarray(is_d, bool), np.asarray(pred_d, bool)
    return 0.5 * ((pred_d[is_d]).mean() + (~pred_d[~is_d]).mean())


def balanced_accuracy_from_correct(is_d, correct) -> float:
    """The same score from a per-site correctness vector."""
    import numpy as np

    is_d, correct = np.asarray(is_d, bool), np.asarray(correct, bool)
    return 0.5 * (correct[is_d].mean() + correct[~is_d].mean())


def f1_d(is_d, pred_d) -> float:
    import numpy as np

    is_d, pred_d = np.asarray(is_d, bool), np.asarray(pred_d, bool)
    tp = int((is_d & pred_d).sum())
    denom = int(is_d.sum()) + int(pred_d.sum())
    return 2 * tp / denom if denom else 0.0


def balanced_accuracy_multiclass(y, pred) -> float:
    """Mean per-class recall: the speaker-ID metric (§7, 2026-09-26)."""
    import numpy as np

    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def resample_weights(n_types: int, seed: int, n: int = N_BOOT):
    """(n, n_types) counts: how often each type is drawn in each resample."""
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_types, size=(n, n_types))
    flat = (np.arange(n)[:, None] * n_types + draws).ravel()
    return np.bincount(flat, minlength=n * n_types).reshape(n, n_types).astype(np.float32)


def boot_balanced_accuracy(W, type_idx, is_d, correct):
    """(B, R) pooled balanced accuracy per resample, for R rows of `correct`.

    type_idx: (n,) type of each site, indexing W's columns.
    is_d:     (n,) truth.
    correct:  (R, n) bool, one row per representation/layer being scored.

    A resample that happens to draw no /d/ (or no /t/) site has no balanced
    accuracy. It comes out NaN rather than a number, and callers use nan-aware
    percentiles. At 991 types with /d/ spread over hundreds of them, this does
    not occur in practice.
    """
    import numpy as np

    is_d = np.asarray(is_d, bool)
    correct = np.atleast_2d(np.asarray(correct, bool))
    T = W.shape[1]

    def per_type(mask):  # (T,) counts, or (R, T) for a 2-D mask
        if mask.ndim == 1:
            return np.bincount(type_idx, weights=mask, minlength=T).astype(np.float64)
        return np.stack([np.bincount(type_idx, weights=m, minlength=T) for m in mask])

    n_d, n_t = per_type(is_d), per_type(~is_d)
    c_d, c_t = per_type(correct & is_d), per_type(correct & ~is_d)
    W64 = W.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec_d = (W64 @ c_d.T) / (W64 @ n_d)[:, None]
        rec_t = (W64 @ c_t.T) / (W64 @ n_t)[:, None]
    return 0.5 * (rec_d + rec_t)


def interval(samples, level: float = 0.95):
    """Percentile interval along axis 0."""
    import numpy as np

    lo = (1 - level) / 2
    return np.nanquantile(samples, lo, axis=0), np.nanquantile(samples, 1 - lo, axis=0)
