"""The listening audit: a blind, seeded sample of sites, and what it measured.

WHAT IT IS FOR
    The pilot failed, and §5's first reading of a failure is the alignment
    path: maybe the sites are not what the inventory says they are. The
    random-split layer curve argues against that, but only indirectly. This is
    the direct check: listen to a random sample and say, for each site,
    whether there is a flap, in the word the transcript shows, in a clip that
    is centred and not cut off.

    It measures the inventory, not the model. The /t/ or /d/ under a flap
    cannot be heard (that is the premise of the whole project); it follows
    from the word. So the questions are about what CAN be heard.

WHY THE SAMPLE IS DRAWN HERE AND NOT PICKED ON SCREEN
    Sites picked from a filtered table, especially from the misclassified
    ones, are not a sample of anything, and a defect rate measured on them
    describes the filter. The sample is drawn from the eligible inventory
    alone, stratified 40 /t/ + 40 /d/, from a seed in capstone.seeds, before
    anyone listens. It never sees a prediction.

HOW VERDICTS ARE COUNTED
    reviewed.jsonl is append-only, so a reviewer who changes their mind leaves
    two lines. The LAST line for a (site, question) is the verdict. Only lines
    for this run_id, on a sampled site, recorded in the blind audit view, are
    counted. Verdicts recorded while browsing are kept but never counted:
    the browser shows the model's guess, and the sites were chosen by a
    filter. Rates are yes / (yes + no), per class, with a Wilson 95%
    interval. "Can't tell" is reported beside the rate, never folded into it.
"""

from __future__ import annotations

import math

AUDIT_TARGETS = {
    "flap": "Is there a flap here: a quick tap, not a full stop, a deletion or "
            "a glottal stop?",
    "word": "Is the word the one the transcript shows?",
    "clip": "Is the flap roughly centred in the clip, and not cut off?",
}
VERDICTS = ("yes", "no", "cant_tell")
N_PER_CLASS = 40


def draw_sample(inventory, n_per_class: int = N_PER_CLASS, seed: int | None = None):
    """n_per_class eligible sites of each class, in a shuffled review order.

    Eligible means not excerpt_clamped, the same exclusion §1 applies before
    scoring. The review order is shuffled across classes, so /t/ and /d/ sites
    are interleaved rather than arriving in two blocks.

    Returns a DataFrame with columns order, site_id, underlying.
    """
    import numpy as np
    import pandas as pd

    from . import seeds

    rng = np.random.default_rng(seeds.AUDIT if seed is None else seed)
    eligible = inventory[~inventory.excerpt_clamped.astype(bool)]
    picked = []
    for cls in ("t", "d"):
        ids = np.sort(eligible[eligible.underlying == cls].site_id.to_numpy())
        picked += [(s, cls) for s in rng.choice(ids, size=n_per_class, replace=False)]
    order = rng.permutation(len(picked))
    rows = [picked[i] for i in order]
    return pd.DataFrame({"order": range(1, len(rows) + 1),
                         "site_id": [r[0] for r in rows],
                         "underlying": [r[1] for r in rows]})


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n. (nan, nan) when n is 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def counted_verdicts(lines, sample, run_id: str) -> dict:
    """{(site_id, target): verdict}, last line wins, audit lines only."""
    in_sample = set(sample.site_id)
    out = {}
    for line in lines:
        if (line.get("run_id") == run_id and line.get("audit") is True
                and line.get("site_id") in in_sample
                and line.get("target") in AUDIT_TARGETS
                and line.get("verdict") in VERDICTS):
            out[(line["site_id"], line["target"])] = line["verdict"]
    return out


def summarize(lines, sample, run_id: str):
    """One row per target × class: counts, the rate, its interval, and coverage.

    `reviewed` is how many of the class's sampled sites have a verdict on that
    question, so an unfinished audit reports itself as unfinished rather than
    as a small sample.
    """
    import pandas as pd

    got = counted_verdicts(lines, sample, run_id)
    cls_of = dict(zip(sample.site_id, sample.underlying))
    size = sample.underlying.value_counts().to_dict()
    rows = []
    for target in AUDIT_TARGETS:
        for cls in ("t", "d"):
            vs = [v for (s, t), v in got.items() if t == target and cls_of[s] == cls]
            yes, no, cant = (vs.count(v) for v in VERDICTS)
            lo, hi = wilson(yes, yes + no)
            rows.append({"question": target, "class": f"/{cls}/",
                         "reviewed": len(vs), "of": int(size.get(cls, 0)),
                         "yes": yes, "no": no, "cant_tell": cant,
                         "rate": yes / (yes + no) if yes + no else float("nan"),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
