"""The listening audit: the sample is what it claims, and only audit lines count.

App environment only, like test_slice_retrieval: pandas and pyarrow come from
requirements-app.txt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="app environment only (requirements-app.txt)")
pytest.importorskip("pyarrow", reason="app environment only (requirements-app.txt)")

import pyarrow.parquet as pq  # noqa: E402

from capstone import seeds  # noqa: E402
from capstone.audit import (  # noqa: E402
    AUDIT_TARGETS, draw_sample, summarize, wilson,
)

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "flapping" / "sites-20260924.parquet"
COMMITTED = REPO / "results" / "flapping" / "audit-sample-20260926.tsv"

needs_inventory = pytest.mark.skipif(not INVENTORY.exists(),
                                     reason=f"no site inventory at {INVENTORY}")


@pytest.fixture(scope="module")
def inventory():
    return pq.read_table(INVENTORY).to_pandas()


@needs_inventory
def test_sample_is_stratified_eligible_and_interleaved(inventory):
    sample = draw_sample(inventory)
    assert sample.underlying.value_counts().to_dict() == {"t": 40, "d": 40}
    assert not sample.site_id.duplicated().any()
    clamped = set(inventory[inventory.excerpt_clamped].site_id)
    assert not clamped & set(sample.site_id)
    truth = dict(zip(inventory.site_id, inventory.underlying))
    assert all(truth[s] == u for s, u in zip(sample.site_id, sample.underlying))
    # Interleaved: the first half of the review order is not one class.
    assert sample.underlying.iloc[:40].nunique() == 2
    assert list(sample.order) == list(range(1, 81))


@needs_inventory
def test_committed_sample_is_the_registered_draw(inventory):
    """The file on disk is what seeds.AUDIT draws, so it cannot be hand-edited."""
    if not COMMITTED.exists():
        pytest.skip("no committed sample yet")
    on_disk = pd.read_csv(COMMITTED, sep="\t")
    redrawn = draw_sample(inventory, seed=seeds.AUDIT)
    assert on_disk.site_id.tolist() == redrawn.site_id.tolist()
    assert on_disk.underlying.tolist() == redrawn.underlying.tolist()


def test_wilson_matches_known_values():
    lo, hi = wilson(36, 40)
    assert lo == pytest.approx(0.7695, abs=1e-3)
    assert hi == pytest.approx(0.9604, abs=1e-3)
    assert all(v != v for v in wilson(0, 0))  # nan


def _line(site, target, verdict, *, run="pilot-20260926", audit=True):
    return {"site_id": site, "target": target, "verdict": verdict,
            "run_id": run, "audit": audit}


def test_summary_counts_only_this_runs_audit_lines_and_last_wins():
    sample = pd.DataFrame({"order": [1, 2, 3], "site_id": ["a", "b", "c"],
                           "underlying": ["t", "t", "d"]})
    lines = [
        _line("a", "flap", "no"),
        _line("a", "flap", "yes"),                      # changed mind: counts
        _line("b", "flap", "cant_tell"),
        _line("c", "flap", "no"),
        _line("b", "flap", "no", audit=False),          # browsing: ignored
        _line("c", "flap", "yes", run="20260920T000056Z"),  # other run: ignored
        _line("z", "flap", "no"),                       # not sampled: ignored
        _line("a", "label", "fail"),                    # old target: ignored
    ]
    out = summarize(lines, sample, "pilot-20260926").set_index(["question", "class"])
    t = out.loc[("flap", "/t/")]
    assert (t.reviewed, t.of, t.yes, t.no, t.cant_tell) == (2, 2, 1, 0, 1)
    assert t.rate == 1.0
    d = out.loc[("flap", "/d/")]
    assert (d.reviewed, d.yes, d.no) == (1, 0, 1)
    assert set(out.index.get_level_values(0)) == set(AUDIT_TARGETS)
    assert out.loc[("word", "/t/")].reviewed == 0
