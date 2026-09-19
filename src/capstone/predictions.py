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

      eligible = false      discarded by the split; no prediction was possible.
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

CONTEXT_MS = 200.0

CONDITIONS = ("ordinary", "word_disjoint", "combined_diagonal")


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
        ("fold", pa.int32()),
        ("eligible", pa.bool_()),        # false = discarded by the split
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
