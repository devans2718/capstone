"""The four §5 criteria, scored on `joint` only, pooled over the 10,389 sites.

Thresholds are the registered ones and are constants, not arguments. Moving
one after seeing a result is exactly what the pre-registration forbids, and a
keyword argument would make that a one-line change nobody sees.

    1  best WavJEPA layer beats MFCC: gap >= 0.05 AND the paired word-type
       bootstrap's 95% lower bound of the gap > 0, with the best layer
       re-selected inside every resample
    2  the control's max-over-layers balanced accuracy does not exceed the
       95th percentile of the same maximum under 10,000 null mappings, with
       predictions held fixed and nothing retrained
    3  best - worst WavJEPA layer >= 0.05
    4  /d/ F1 at the best layer >= 0.45
"""

from __future__ import annotations

from dataclasses import dataclass

GAP = 0.05          # criterion 1, and criterion 3's spread
F1_D_FLOOR = 0.45   # criterion 4
NULL_QUANTILE = 0.95


@dataclass(frozen=True)
class Criterion:
    number: int
    name: str
    passed: bool
    detail: str


def criterion_1(W, type_idx, is_d, correct_layers, correct_mfcc):
    """Best layer vs MFCC, paired bootstrap. Returns (Criterion, numbers)."""
    import numpy as np

    from .scoring import boot_balanced_accuracy, interval

    ba = np.array([_ba(is_d, c) for c in correct_layers])
    ba_mfcc = _ba(is_d, correct_mfcc)
    best = int(np.argmax(ba))
    gap = float(ba[best] - ba_mfcc)

    rows = np.vstack([correct_layers, correct_mfcc[None]])
    boot = boot_balanced_accuracy(W, type_idx, is_d, rows)       # paired: one W
    gaps = np.nanmax(boot[:, :-1], axis=1) - boot[:, -1]         # best re-selected per resample
    lo, hi = interval(gaps)
    passed = gap >= GAP and lo > 0
    nums = {"best_layer": best, "ba_best": float(ba[best]), "ba_mfcc": float(ba_mfcc),
            "gap": gap, "gap_ci_lo": float(lo), "gap_ci_hi": float(hi)}
    detail = (f"best layer {best}: {ba[best]:.4f}; MFCC {ba_mfcc:.4f}; gap {gap:+.4f} "
              f"(needs >= {GAP}); paired word-type bootstrap 95% CI "
              f"[{lo:+.4f}, {hi:+.4f}] (lower bound needs > 0)")
    return Criterion(1, "best layer beats the MFCC baseline", bool(passed), detail), nums


def _ba(is_d, correct):
    from .scoring import balanced_accuracy_from_correct

    return balanced_accuracy_from_correct(is_d, correct)


def criterion_2(control_pred_d, type_idx, real_mapping, nulls):
    """Randomization test on the control's max-over-layers balanced accuracy.

    control_pred_d: (13, n) bool, the control probes' pooled predictions.
    real_mapping:   (T,) bool, the mapping the control probes were trained on.
    nulls:          (B, T) bool, the null mappings.
    """
    import numpy as np

    P = np.asarray(control_pred_d, bool)
    real_labels = real_mapping[type_idx]
    real = np.array([_ba(real_labels, p == real_labels) for p in P])
    real_max = float(real.max())

    Pf = P.astype(np.float32).T                 # (n, L)
    null_max = np.empty(len(nulls))
    for s in range(0, len(nulls), 500):
        L = nulls[s:s + 500][:, type_idx].astype(np.float32)   # (b, n) labels
        n_d = L.sum(1, keepdims=True)
        n_t = L.shape[1] - n_d
        hit_d = L @ Pf                          # predicted /d/ and is /d/
        hit_t = (1 - L) @ (1 - Pf)              # predicted /t/ and is /t/
        null_max[s:s + 500] = (0.5 * (hit_d / n_d + hit_t / n_t)).max(axis=1)

    q = float(np.quantile(null_max, NULL_QUANTILE))
    p = float((1 + (null_max >= real_max).sum()) / (1 + len(null_max)))
    passed = real_max <= q
    nums = {"control_ba_by_layer": real.tolist(), "control_max": real_max,
            "control_max_layer": int(real.argmax()), "null_q95": q, "p_value": p}
    detail = (f"control max over layers {real_max:.4f} (layer {int(real.argmax())}); "
              f"95th percentile of {len(nulls):,} null maxima {q:.4f}; "
              f"permutation p = {p:.4f}")
    return Criterion(2, "control task shows no leak", bool(passed), detail), nums


def criterion_3(ba_layers):
    import numpy as np

    ba = np.asarray(ba_layers)
    spread = float(ba.max() - ba.min())
    detail = (f"best {ba.max():.4f} (layer {int(ba.argmax())}) - worst {ba.min():.4f} "
              f"(layer {int(ba.argmin())}) = {spread:.4f} (needs >= {GAP})")
    return Criterion(3, "layer-wise curve is not flat", spread >= GAP, detail), {"spread": spread}


def criterion_4(f1_best: float, best_layer: int):
    detail = (f"/d/ F1 at layer {best_layer} {f1_best:.4f} (needs >= {F1_D_FLOOR}; "
              "trivial floors 0.391 always-/d/, 0.327 coin, 0.243 base rate)")
    return Criterion(4, "minority-class F1 at best layer", f1_best >= F1_D_FLOOR, detail), {"f1_d": f1_best}
