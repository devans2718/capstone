#!/usr/bin/env python3
"""How far does the control task's balanced accuracy stray from 0.50 with no leak?

Read-only, and no probe, features or audio are involved: a measurement of the
corpus and of the control task's design, taken before any run.

WHY THIS EXISTS
    Pre-registration criterion 2 bounded the control at 0.50 ± 0.03, justified
    as about three times the ±0.011 sampling band. That band assumes sites are
    independent. Control labels are not: each word type draws one label and
    every token inherits it, and the type distribution is heavy-headed (the top
    10 types are 41% of sites). A probe that has learned nothing about the
    control labels still tends to answer alike for all tokens of a type, so
    hundreds of sites come out right or wrong together, and the effective
    sample is closer to a count of types than of sites.

WHAT IT DOES
    Over the 10,389 eligible sites, draw the control mapping many times, with
    each type labelled /d/ at the real site-level /d/ share. Score each mapping
    against a FIXED prediction pattern that is independent of it, which is
    exactly what a leak-free probe produces on held-out types under `joint`.
    Three patterns:
      - the real underlying label (a probe that learned the real task, nearly
        constant within a type because labels are 99.5% type-determined);
      - a random label, constant within each type;
      - a random label per site (the independence the ±0.011 band assumed).
    Also reports how averaging K independent mappings narrows the spread.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_INVENTORY = (Path(__file__).resolve().parent.parent
                     / "results/flapping/sites-20260924.parquet")
INVENTORY = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INVENTORY
SEED = 0          # simulation only; not a registered seed
N_MAPPINGS = 10_000
BOUND = 0.03

inv = pq.read_table(INVENTORY, columns=["word", "underlying",
                                        "excerpt_clamped"]).to_pandas()
e = inv[~inv.excerpt_clamped].reset_index(drop=True)
y = (e.underlying == "d").to_numpy()
_, tix = np.unique(e.word.to_numpy(), return_inverse=True)
n_types = tix.max() + 1
p_d = y.mean()


def balanced_accuracy(truth, pred):
    pos, neg = truth.sum(), (~truth).sum()
    if pos == 0 or neg == 0:
        return np.nan
    return 0.5 * ((pred & truth).sum() / pos + (~pred & ~truth).sum() / neg)


rng = np.random.default_rng(SEED)
patterns = {
    "real label (~constant per type)": y,
    "random label, constant per type": (rng.random(n_types) < p_d)[tix],
    "random label, per site": rng.random(len(e)) < p_d,
}

print(f"eligible sites {len(e)}, word types {n_types}, /d/ share {p_d:.4f}")
print(f"control mappings drawn per pattern: {N_MAPPINGS:,} (simulation seed {SEED})\n")
print(f"{'prediction pattern':34s} {'mean':>6} {'sd':>6} {'95% range':>13} "
      f"{'P(|BA-.5|>' + str(BOUND) + ')':>16}")

draws = {}
for name, pred in patterns.items():
    ba = np.array([balanced_accuracy((rng.random(n_types) < p_d)[tix], pred)
                   for _ in range(N_MAPPINGS)])
    ba = ba[~np.isnan(ba)]
    draws[name] = ba
    lo, hi = np.percentile(ba, [2.5, 97.5])
    print(f"{name:34s} {ba.mean():6.3f} {ba.std():6.3f} {lo:6.3f}-{hi:6.3f} "
          f"{(abs(ba - 0.5) > BOUND).mean():16.2f}")

share = np.array([(rng.random(n_types) < p_d)[tix].mean() for _ in range(2000)])
print(f"\ncontrol /d/ site share across mappings: 95% range "
      f"{np.percentile(share, 2.5):.3f}-{np.percentile(share, 97.5):.3f} "
      f"(real {p_d:.3f})")

print("\naveraging K independent mappings (real-label pattern), sd of the mean:")
sd = draws["real label (~constant per type)"].std()
for k in (1, 5, 10, 20, 50):
    print(f"  K={k:>3}: sd {sd / np.sqrt(k):.4f}, "
          f"95% half-width {1.96 * sd / np.sqrt(k):.3f}")
