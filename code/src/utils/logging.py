"""Lightweight metrics logger: JSONL on disk, optional wandb."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


class MetricsLogger:
    def __init__(
        self,
        output_root: str | Path,
        run_name: str,
        use_wandb: bool = False,
        wandb_project: str = "latent_reasoning",
        config: dict | None = None,
        is_main_process: bool = True,
    ):
        self.is_main = is_main_process
        self.dir = Path(output_root) / "logs" / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.stdout_path = self.dir / "stdout.log"
        self.start_t = time.time()
        self._wandb = None
        if use_wandb and self.is_main and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                self._wandb = wandb.init(project=wandb_project, name=run_name, config=config or {})
            except Exception as e:
                self.print(f"[log] wandb init failed: {e}; falling back to JSONL only")
        if self.is_main and config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2))

    def log(self, step: int, metrics: dict):
        if not self.is_main:
            return
        rec = {"step": step, "wall": time.time() - self.start_t, **metrics}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def print(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.is_main:
            with self.stdout_path.open("a") as f:
                f.write(line + "\n")
            print(line, flush=True)

    def close(self):
        if self._wandb is not None:
            self._wandb.finish()
