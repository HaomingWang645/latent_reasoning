"""Checkpoint manager that writes to a non-/home disk and prunes old saves.

Layout under output_root/ckpts/<run_name>/:
    step_000500/  step_001000/  ...  latest -> step_XXXXXX
Each step dir contains:
    model_state.pt            (main backbone + heads, sharded if FSDP)
    optimizer_state.pt        (optional, large; controlled by config)
    rng_state.pt
    meta.json                 (step, epoch, wall_time, val metrics)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist


class CheckpointManager:
    def __init__(
        self,
        output_root: str | Path,
        run_name: str,
        keep_last_n: int = 2,
        save_optimizer: bool = True,
        is_main_process: bool = True,
    ):
        self.root = Path(output_root) / "ckpts" / run_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.is_main = is_main_process

    def _step_dir(self, step: int) -> Path:
        return self.root / f"step_{step:08d}"

    def list_existing(self) -> list[Path]:
        dirs = sorted([p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("step_")])
        return dirs

    def latest(self) -> Path | None:
        existing = self.list_existing()
        return existing[-1] if existing else None

    def save(
        self,
        step: int,
        model_state: dict,
        optimizer_state: dict | None = None,
        rng_state: dict | None = None,
        meta: dict | None = None,
    ) -> Path:
        """Save full state. Each rank only saves its shard if model_state is sharded;
        for the pilot we keep it simple and save only on the main process."""
        target = self._step_dir(step)
        if self.is_main:
            target.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, tmp / "model_state.pt")
            if optimizer_state is not None and self.save_optimizer:
                torch.save(optimizer_state, tmp / "optimizer_state.pt")
            if rng_state is not None:
                torch.save(rng_state, tmp / "rng_state.pt")
            (tmp / "meta.json").write_text(
                json.dumps({"step": step, "wall_time": time.time(), **(meta or {})}, indent=2)
            )
            # atomic rename
            if target.exists():
                shutil.rmtree(target)
            tmp.rename(target)

            # update 'latest' symlink
            latest_link = self.root / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(target.name)

            self._prune()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return target

    def _prune(self):
        existing = self.list_existing()
        if len(existing) <= self.keep_last_n:
            return
        to_remove = existing[: -self.keep_last_n]
        for p in to_remove:
            try:
                shutil.rmtree(p)
            except OSError as e:
                print(f"[ckpt] WARN: failed to prune {p}: {e}")

    def load(self, step: int | None = None) -> dict | None:
        if step is None:
            target = self.latest()
            if target is None:
                return None
        else:
            target = self._step_dir(step)
            if not target.exists():
                return None
        out = {
            "model_state": torch.load(target / "model_state.pt", map_location="cpu", weights_only=False),
            "meta": json.loads((target / "meta.json").read_text()),
        }
        opt_path = target / "optimizer_state.pt"
        if opt_path.exists():
            out["optimizer_state"] = torch.load(opt_path, map_location="cpu", weights_only=False)
        rng_path = target / "rng_state.pt"
        if rng_path.exists():
            out["rng_state"] = torch.load(rng_path, map_location="cpu", weights_only=False)
        return out

    def disk_usage_mb(self) -> float:
        total = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / 1024 / 1024
