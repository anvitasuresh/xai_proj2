"""Thin Weights & Biases wrapper.

W&B is optional: if it isn't installed, or WANDB_MODE=disabled, or no API key is set, every call
becomes a no-op and metrics are still written locally by each stage. That keeps tests and CPU-only
runs working without an account.
"""

from __future__ import annotations

import os
from typing import Any


class Run:
    def __init__(self, wandb_run: Any | None):
        self._run = wandb_run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run:
            self._run.log(metrics, step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        if self._run:
            for k, v in metrics.items():
                self._run.summary[k] = v

    def log_table(self, name: str, rows: list[dict[str, Any]], max_rows: int = 500) -> None:
        """Log example rows (e.g. judge scores with text) so they can be browsed in the W&B UI."""
        if not self._run or not rows:
            return
        import wandb

        cols = list(rows[0].keys())
        table = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows[:max_rows]])
        self._run.log({name: table})

    def log_artifact(self, path: str, name: str, type_: str) -> None:
        """Version a dataset/result file in W&B (content-addressed, so reruns dedupe)."""
        if not self._run:
            return
        import wandb

        art = wandb.Artifact(name=name, type=type_)
        art.add_file(path)
        self._run.log_artifact(art)

    def finish(self) -> None:
        if self._run:
            self._run.finish()


def init_run(stage: str, config: dict[str, Any], tags: list[str] | None = None) -> Run:
    """Start a W&B run for one pipeline stage. Project name comes from config['wandb'] or env."""
    wb_cfg = config.get("wandb", {}) or {}
    if os.environ.get("WANDB_MODE") == "disabled" or not wb_cfg.get("enabled", True):
        return Run(None)
    try:
        import wandb
    except ImportError:
        return Run(None)
    if not (os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc"))):
        print("[tracking] No W&B credentials found; logging locally only.")
        return Run(None)
    run = wandb.init(
        project=wb_cfg.get("project", os.environ.get("WANDB_PROJECT", "who-does-the-tutor-learn-from")),
        entity=wb_cfg.get("entity") or os.environ.get("WANDB_ENTITY"),
        job_type=stage,
        group=wb_cfg.get("group"),
        tags=(tags or []) + [stage],
        config=config,
    )
    return Run(run)
