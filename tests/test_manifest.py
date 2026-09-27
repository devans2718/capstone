"""The manifest and the prediction schema, as the real run will use them.

Until revision 3's run the manifest carried one seed and hardcoded
synthetic=True, and the schema had no way to tell the §3 subsample from the
headline run except by parsing a condition name. These tests pin the changes
that A1.6 asked for, so the real run cannot write a manifest that claims to be
synthetic, or rows that cannot be told apart.
"""

from __future__ import annotations

import pytest

from capstone import seeds
from capstone.predictions import Manifest, build_manifest

pa = pytest.importorskip("pyarrow", reason="app environment only (requirements-app.txt)")


def test_registered_seeds_are_the_preregistered_values():
    assert seeds.REGISTERED == {
        "folds": 20260926, "subsample": 20260926, "control": 20260927,
        "bootstrap": 20260928, "control_null": 20260929,
    }


def test_manifest_round_trips_every_seed_and_the_synthetic_flag(tmp_path):
    inv = tmp_path / "inv.bin"
    inv.write_bytes(b"inventory")
    for synthetic in (True, False):
        m = build_manifest(inv, split_scheme="x", seeds=seeds.REGISTERED,
                           n_sites_scored=10_389, synthetic=synthetic,
                           encoders=("wavjepa", "mfcc"))
        back = Manifest.read(m.write(tmp_path / f"m-{synthetic}.json"))
        assert back == m
        assert back.synthetic is synthetic
        assert back.seeds == seeds.REGISTERED


def test_build_manifest_has_no_default_for_synthetic(tmp_path):
    inv = tmp_path / "inv.bin"
    inv.write_bytes(b"inventory")
    with pytest.raises(TypeError):
        build_manifest(inv, split_scheme="x", seeds={}, n_sites_scored=0)


def test_schema_marks_the_subsample_and_pooling_by_column():
    from capstone.predictions import prediction_schema

    schema = prediction_schema()
    assert schema.field("subsampled").type == pa.bool_()
    assert schema.field("pooling").type == pa.string()
