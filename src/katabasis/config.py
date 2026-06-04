"""Config loading built on OmegaConf (YAML files + dotlist CLI overrides).

We use OmegaConf directly rather than full hydra-core (whose pinned
antlr4-python3-runtime has no prebuilt wheel on some platforms). It supports
everything the project needs: composable YAML, variable interpolation, and
``key=value`` command-line overrides. ``no magic numbers in code`` -- every
tunable lives in ``configs/``.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_config(
    name: str = "config",
    *,
    overrides: list[str] | None = None,
    config_root: Path | None = None,
) -> DictConfig:
    """Load ``configs/<name>.yaml`` and apply ``key=value`` overrides.

    The top-level config uses a ``defaults:`` list of ``group: option`` entries
    (hydra-style) which are merged from ``configs/<group>/<option>.yaml``.
    """
    root = config_root or CONFIG_ROOT
    base = OmegaConf.load(root / f"{name}.yaml")

    # A config "group" is a subdirectory of the config root (e.g. data/, model/,
    # train/). Group selections start in the defaults list and may be overridden
    # on the CLI hydra-style as `group=option`.
    selection: dict[str, str] = {}
    for entry in base.pop("defaults", []) or []:
        for group, option in dict(entry).items():
            selection[str(group)] = str(option)

    # Split CLI overrides: `group=option` (group dir exists, no dotted LHS) is a
    # group selection; everything else (`a.b=value`) is a leaf override.
    leaf_overrides: list[str] = []
    for ov in overrides or []:
        key = ov.split("=", 1)[0]
        if "." not in key and (root / key).is_dir():
            selection[key] = ov.split("=", 1)[1]
        else:
            leaf_overrides.append(ov)

    merged = OmegaConf.create({})
    for group, option in selection.items():
        sub = OmegaConf.load(root / group / f"{option}.yaml")
        merged = OmegaConf.merge(merged, OmegaConf.create({group: sub}))
    merged = OmegaConf.merge(merged, base)

    if leaf_overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(leaf_overrides))
    return merged  # type: ignore[return-value]


def to_container(cfg: DictConfig) -> dict:
    """Resolve interpolations and return a plain dict (for logging/serialization)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
