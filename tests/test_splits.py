"""Reaction-level splits and leakage (Critical req. 5)."""

from __future__ import annotations

import pytest

from katabasis.data.splits import (
    Split,
    assert_no_leakage,
    load_split,
    make_split,
    save_split,
)


def test_no_leakage_by_construction():
    ids = [f"rxn_{i}" for i in range(100)]
    split = make_split(ids, "t1x_core", seed=0)
    assert_no_leakage(split)
    assert len(split.all_ids) == 100
    assert len(set(split.all_ids)) == 100


def test_leakage_detected():
    bad = Split("s", 0, (0.8, 0.1, 0.1), train=["a", "b"], val=["b"], test=["c"])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(bad)


def test_split_deterministic():
    ids = [f"rxn_{i}" for i in range(50)]
    a = make_split(ids, "x", seed=7)
    b = make_split(ids, "x", seed=7)
    assert a.train == b.train and a.val == b.val and a.test == b.test


def test_split_roundtrip_and_hash(tmp_path):
    ids = [f"rxn_{i}" for i in range(30)]
    split = make_split(ids, "x", seed=1)
    h = save_split(split, tmp_path / "split.json")
    loaded = load_split(tmp_path / "split.json")
    assert loaded.hash() == h
    assert loaded.train == split.train


def test_split_fractions_validated():
    with pytest.raises(ValueError):
        make_split(["a"], "x", fractions=(0.5, 0.3, 0.3))
