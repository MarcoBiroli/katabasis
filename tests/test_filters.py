"""Phase-1 pathological-reaction filters."""

from __future__ import annotations

import numpy as np

from katabasis.data.filters import FilterConfig, apply_filters, is_single_barrier
from katabasis.data.synthetic import make_synthetic_reaction


def test_single_barrier_accepts_clean_profile():
    e = np.array([0.0, 5.0, 10.0, 20.0, 10.0, 5.0, 0.0])
    assert is_single_barrier(e, noise_frac=0.05)


def test_single_barrier_rejects_double_barrier():
    # Two maxima with a real intermediate minimum.
    e = np.array([0.0, 10.0, 2.0, 12.0, 3.0])
    assert not is_single_barrier(e, noise_frac=0.05)


def test_single_barrier_rejects_max_at_endpoint():
    e = np.array([0.0, 2.0, 5.0, 8.0, 10.0])  # monotone increasing
    assert not is_single_barrier(e, noise_frac=0.05)


def test_filter_removes_multistep_reaction():
    good = make_synthetic_reaction("good", seed=0, multistep=False)
    bad = make_synthetic_reaction("bad", seed=1, multistep=True)
    cfg = FilterConfig(min_barrier=1.0, min_rp_rmsd=0.1)
    kept, report = apply_filters([good, bad], cfg)
    kept_ids = {r.reaction_id for r in kept}
    assert "good" in kept_ids
    assert "bad" not in kept_ids
    assert report.removed_multistep == 1


def test_filter_removes_low_barrier():
    r = make_synthetic_reaction("lowbar", seed=0, barrier=2.0)
    cfg = FilterConfig(min_barrier=5.0, min_rp_rmsd=0.1)
    kept, report = apply_filters([r], cfg)
    assert not kept and report.removed_low_barrier == 1
