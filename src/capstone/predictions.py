"""Prediction rows and the run manifest that makes them citable.

WHY A SCHEMA MODULE RATHER THAN A DICT AND A PARQUET CALL
    The slice's whole claim is that a number on a screen can be traced back to
    the audio it came from. That claim is carried by two things: a join key that
    cannot silently fan out, and a manifest saying which code, which inventory,
    and which seed produced the row. Neither survives being assembled ad hoc at
    the call site, so both live here.

THE THREE ELIGIBILITY STATES, AND WHERE EACH ONE LIVES
    A site can be in one of three conditions, and the design brief is explicit
    that rendering any two of them the same way is the defect the failure test
    exists to catch:

      eligible = false      excluded before scoring (pre-registration §1, e.g.
                            a clamped excerpt); no prediction was possible.
                            A ROW EXISTS with eligible=false, scored=false and
                            p_t null.
      eligible, scored      a prediction exists. A row exists, fully populated.
      eligible, not scored  the run did not reach this site. NO ROW EXISTS.

    That last one is the trap. It is represented by ABSENCE, which means the
    only correct way to read this table is a LEFT JOIN from the site inventory.
    An INNER JOIN returns something that looks entirely reasonable — every row
    it shows is true — while making "we never scored this" indistinguishable
    from "this site does not exist". Both then render as no data, and a
    denominator quietly becomes 40 instead of 10,392.

WHY p_t IS NULL AND NEVER ZERO
    A zero is a number. A number gets rendered, gets averaged into a mean, and
    gets believed. There is no value of p_t that honestly represents "no
    prediction was made here", so the column is nullable and carries null.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# The excerpt width every site was cut at, and the checkpoint the geometry came
# from. Both are read from the extraction module rather than retyped: a manifest
# that records a context width different from the one actually used is worse
# than one that records nothing, because it looks like provenance.
from .encoders.extraction import REVISION as CHECKPOINT_REVISION
from .splits import CONDITIONS, site_folds

CONTEXT_MS = 200.0


def prediction_schema():
    """The columns of predictions.parquet, declared rather than inferred.

    Declared for the same reason the site inventory's schema is: an inferred
    schema will widen an int64 column to a double on one null and not mention
    it, and `site_id` is the join key for everything the application shows.

    Nullability is load-bearing here, not incidental. p_t, pred and correct are
    nullable precisely so that an unscored site can be written as null instead
    of as a plausible-looking default.
    """
    import pyarrow as pa

    return pa.schema([
        ("site_id", pa.string()),        # joins sites.parquet; never a float
        ("encoder", pa.string()),        # 'wavjepa'; the lookup baseline is 'lookup'
        ("layer", pa.int32()),           # nullable: not every encoder has layers
        ("condition", pa.string()),
        ("fold", pa.int32()),            # NULL when eligible is false: in no fold
        ("eligible", pa.bool_()),        # false = excluded before scoring
        ("scored", pa.bool_()),          # false = eligible but not scored this run
        ("p_t", pa.float64()),           # NULL whenever scored is false
        ("pred", pa.string()),           # 't' | 'd'; nullable
        ("correct", pa.bool_()),         # nullable
        ("synthetic", pa.bool_()),
    ])


def write_predictions(rows, path: Path) -> Path:
    """Write prediction rows to parquet under the declared schema.

    Refuses rather than writes when a row claims a probability it should not
    have. The check is here and not at the call site because this is the last
    place the invariant can be enforced before the value becomes a file that
    something else will trust.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    for r in rows:
        if not r["scored"] and r["p_t"] is not None:
            raise ValueError(
                f"{r['site_id']} under {r['condition']}: scored is false but p_t "
                f"is {r['p_t']!r}. An unscored site has no probability; writing "
                "one makes a number that will be rendered and believed."
            )
        if r["scored"] and r["p_t"] is None:
            raise ValueError(
                f"{r['site_id']} under {r['condition']}: scored is true but p_t "
                "is null."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=prediction_schema()), path)
    return path


# --------------------------------------------------------------------- manifest


def file_hash(path: Path) -> str:
    """sha256 of a file's bytes, as the manifest's inventory_hash.

    Hashing the bytes rather than the row count or the modification time is what
    lets a later reader tell "the same inventory" from "an inventory with the
    same number of rows". Regenerating under --keep-marked, for instance, adds
    754 sites and changes this; a human review verdict recorded against the old
    hash then correctly stops applying.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_commit() -> str:
    """The commit this run was produced by, with a dirty tree marked as such.

    A dirty tree is recorded, not refused. Refusing would push people to commit
    noise just to run something; recording means the manifest never claims a
    clean provenance it does not have.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass(frozen=True)
class Manifest:
    """Everything needed to say where a prediction row came from.

    Hash REFUSAL is deliberately not implemented — the design brief defers it.
    This records the fields so that the refusal, when it arrives, has something
    to check against rather than needing a schema change first.
    """

    run_id: str
    inventory_hash: str
    code_commit: str
    checkpoint_revision: str
    context_ms: float
    split_scheme: str
    random_seed: int
    n_sites_scored: int
    synthetic: bool = True

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        return cls(**json.loads(path.read_text()))


def build_manifest(inventory: Path, *, split_scheme: str, random_seed: int,
                   n_sites_scored: int, run_id: str | None = None) -> Manifest:
    """Assemble a manifest from the inputs that actually produced a run."""
    return Manifest(
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        inventory_hash=file_hash(inventory),
        code_commit=code_commit(),
        checkpoint_revision=CHECKPOINT_REVISION,
        context_ms=CONTEXT_MS,
        split_scheme=split_scheme,
        random_seed=random_seed,
        n_sites_scored=n_sites_scored,
        synthetic=True,
    )


# ------------------------------------------------------- the synthetic cell (§6)

# Accuracies the draw targets. Not calibrated to anything — no probe has been
# trained — but chosen so the differential question in the brief's §7 has a
# non-empty answer, and so the gap runs in the direction a lexical shortcut
# predicts: a probe leaning on word identity loses more when word types stop
# recurring between train and test than it loses to anything else. Holding out
# speakers costs a little; holding out word types costs most of it.
P_CORRECT = {"random": 0.85, "speaker_disjoint": 0.82,
             "word_disjoint": 0.62, "joint": 0.60}


def select_sites(inventory, n: int, seed: int):
    """Choose the n sites the run scores, edge cases included by construction.

    The brief requires at least one zero-onset site, one clamped site, and both
    underlying classes. Those are not left to a random draw over 10,392 rows,
    where the clamped sites alone are 3-in-10,392 and would essentially never
    appear. An edge case that is merely reachable elsewhere in the inventory is
    not demonstrated by a demonstration set.

    The zero-onset picks are sites with no frame onset in the recording's own
    timebase. Since D1 was resolved (2026-09-24), no site has zero onsets in
    the excerpt-local timebase the extractor indexes: a centred target always
    contains the onset at its own start. These 4 are the sites where reading
    the wrong column would make a usable site look empty, and they stay in
    the set to test for exactly that.

    ALL clamped sites are taken, not a sample. They are the ineligible ones, and
    coverage() counts eligibility from prediction rows. A clamped site left out
    of the run would have no row and would read as "not reached", a state it
    can never leave. Taking all of them makes n_eligible exact: 10,389, the
    pre-registration's pool. A real run writes a row for every excluded site for
    the same reason.

    Returns the selected rows in inventory order, so the output does not depend
    on the order the edge cases happened to be added.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    picked: list[str] = []

    def take(pool, k):
        if len(pool) == 0:
            return
        chosen = rng.choice(pool.site_id.to_numpy(),
                            size=min(k, len(pool)), replace=False)
        picked.extend(x for x in chosen if x not in picked)

    take(inventory[inventory.n_frame_onsets_abs == 0], 2)
    take(inventory[inventory.excerpt_clamped], len(inventory))
    take(inventory[inventory.n_flaps_in_word >= 2], 2)

    # Fill the rest keeping both classes present in roughly corpus proportion.
    remaining = inventory[~inventory.site_id.isin(picked)]
    for cls, share in (("d", 0.35), ("t", 0.65)):
        take(remaining[remaining.underlying == cls],
             max(1, round((n - len(picked)) * share)))
    take(inventory[~inventory.site_id.isin(picked)], n - len(picked))

    return inventory[inventory.site_id.isin(picked[:n])].reset_index(drop=True)


def synthesize(inventory, n_sites: int = 40, seed: int = 20260919,
               encoder: str = "wavjepa", layer: int = 8):
    """Generate the four conditions over one selected set of sites.

    SYNTHETIC, AND THE SCHEMA SAYS SO ON EVERY ROW
        No probe has been trained. These probabilities carry no information
        about /t/ and /d/ whatsoever, and the only thing being demonstrated is
        that a number round-trips from a tool call to a screen with its
        provenance intact. Retrieval is exactly as testable against noise as
        against a real result, because retrieval does not depend on whether a
        probability means anything.

    THE DRAW IS SEEDED AND THE SEED IS RECORDED
        A synthetic result that cannot be regenerated is worse than none: it
        cannot be checked, and it cannot be told apart from a result that was
        adjusted until it looked right.

    WHERE eligible=false COMES FROM
        The sites pre-registration §1 excludes before any probe runs. Today
        that is the 3 clamped excerpts, whose window runs off the recording.
        That holds in every condition, because the exclusion happens before
        the split. `joint` is the pre-registered speaker- and type-disjoint
        split. It scores all 25 cells of its 5 x 5 grid, so the split itself
        discards nothing. It was `combined` until revision 3, and
        `combined_diagonal` before 2026-09-24, when it marked ~80% of sites
        ineligible to match the diagonal-only design. select_sites takes every
        clamped site, so the state stays reachable from real exclusions rather
        than an invented rate.

    THE FOLDS ARE THE REGISTERED ONES
        Each row's fold is the site's real fold under capstone.splits, assigned
        over every ELIGIBLE site in the inventory and then looked up for the
        selected sites. Assigning over the 40 alone would rank 40 sites' word
        types and describe a split nobody runs. An excluded site belongs to no
        fold, so its fold is null, for the same reason its p_t is.
        `random` is seeded with `seed`.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    sites = select_sites(inventory, n_sites, seed)
    pool = inventory[~inventory.excerpt_clamped]
    folds = {
        cond: dict(zip(pool.site_id, site_folds(cond, pool.speaker, pool.word,
                                                pool.underlying, seed=seed)))
        for cond in CONDITIONS
    }

    # One correctness draw per site per condition, then a probability CONSISTENT
    # with it. Drawing p_t independently and deriving correctness from it would
    # let the two disagree, and a row whose probability contradicts its own
    # correct flag is a bug that looks like a subtle modelling choice.
    rows = []
    for cond in CONDITIONS:
        hits = rng.random(len(sites)) < P_CORRECT[cond]
        for i, site in enumerate(sites.itertuples()):
            eligible = not site.excerpt_clamped
            if not eligible:
                rows.append({
                    "site_id": site.site_id, "encoder": encoder, "layer": layer,
                    "condition": cond, "fold": None,
                    "eligible": False, "scored": False,
                    "p_t": None, "pred": None, "correct": None, "synthetic": True,
                })
                continue
            truth = site.underlying
            correct = bool(hits[i])
            pred = truth if correct else ("d" if truth == "t" else "t")
            # p_t on the side its own prediction implies, margin away from 0.5.
            margin = float(rng.uniform(0.08, 0.45))
            p_t = 0.5 + margin if pred == "t" else 0.5 - margin
            rows.append({
                "site_id": site.site_id, "encoder": encoder, "layer": layer,
                "condition": cond, "fold": int(folds[cond][site.site_id]),
                "eligible": True, "scored": True,
                "p_t": round(p_t, 6), "pred": pred, "correct": correct,
                "synthetic": True,
            })
    return sites, rows
