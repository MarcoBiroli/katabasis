"""End-to-end data orchestration: load -> filter -> split.

Given a data config (subset, db/cache paths, filter thresholds, split fractions)
this returns ``{"train": [...], "val": [...], "test": [...]}`` lists of
:class:`Reaction`, persisting the split and writing a filter report. Works for
both the synthetic fixtures and real Halo8 subsets.
"""

from __future__ import annotations

import json
from pathlib import Path

from katabasis.data.filters import FilterConfig, FilterReport, apply_filters
from katabasis.data.halo8 import Halo8Reader, Halo8Schema
from katabasis.data.splits import assert_no_leakage, load_split, make_split, save_split
from katabasis.data.synthetic import make_synthetic_dataset
from katabasis.data.types import Reaction


def _load_raw(cfg: dict) -> list[Reaction]:
    subset = cfg["subset"]
    if subset == "synthetic":
        return make_synthetic_dataset(cfg.get("n_reactions", 24))

    reader = Halo8Reader(cfg["db_path"], cfg["cache_dir"], schema=Halo8Schema())
    reactions = list(reader.iter_reactions())
    if subset in ("t1x_core", "halogen_only"):
        reactions = [r for r in reactions if r.subset == subset]
    return reactions


def _filter_config(cfg: dict) -> FilterConfig:
    f = cfg.get("filter", {})
    return FilterConfig(
        min_barrier=f.get("min_barrier", 5.0),
        min_rp_rmsd=f.get("min_rp_rmsd", 0.5),
        monotonic_noise_frac=f.get("monotonic_noise_frac", 0.05),
        require_single_barrier=f.get("require_single_barrier", True),
        max_removed_frac=f.get("max_removed_frac", 0.30),
    )


def load_reactions(
    cfg: dict, *, report_path: str | Path | None = None
) -> tuple[dict[str, list[Reaction]], FilterReport]:
    """Load, filter, and split reactions for the configured subset."""
    raw = _load_raw(cfg)
    fcfg = _filter_config(cfg)
    kept, report = apply_filters(raw, fcfg)

    if report.total and report.removed_frac > fcfg.max_removed_frac:
        # Surface (don't silently pass) when filtering is more aggressive than
        # the Phase-1 acceptance gate allows.
        print(
            f"[load_reactions] WARNING: filtering removed {report.removed_frac:.1%} "
            f"(> {fcfg.max_removed_frac:.0%}); inspect thresholds for subset={cfg['subset']}."
        )
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report.as_dict(), indent=2))

    ids = [r.reaction_id for r in kept]
    split_path = cfg.get("split_path")
    if split_path and Path(split_path).exists():
        split = load_split(split_path)
    else:
        split = make_split(
            ids, cfg["subset"], seed=cfg.get("split_seed", 0), fractions=tuple(cfg["fractions"])
        )
        if split_path:
            save_split(split, split_path)
    assert_no_leakage(split)

    by_id = {r.reaction_id: r for r in kept}
    splits = {
        "train": [by_id[i] for i in split.train if i in by_id],
        "val": [by_id[i] for i in split.val if i in by_id],
        "test": [by_id[i] for i in split.test if i in by_id],
    }
    return splits, report
