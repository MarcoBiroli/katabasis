"""Run logging: wandb if available and enabled, else JSONL to ``runs/``.

Every run logs full config and per-epoch validation metrics (cross-cutting:
logging). wandb is an optional dependency so CPU smoke runs never require it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class RunLogger:
    """Minimal logger with an optional wandb backend and a JSONL fallback."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config: dict | None = None,
        use_wandb: bool = False,
        project: str = "katabasis",
        run_name: str | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "metrics.jsonl").open("a")
        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(
                    project=project, name=run_name, dir=str(self.run_dir), config=config
                )
            except Exception as exc:  # wandb missing or offline -> fall back silently
                print(f"[RunLogger] wandb unavailable ({exc}); logging to JSONL only.")

    def log(self, metrics: dict, step: int | None = None) -> None:
        record = {"step": step, "time": time.time(), **metrics}
        self._jsonl.write(json.dumps(record, default=float) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
