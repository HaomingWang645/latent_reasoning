"""Shared markdown training log appended to by all stages + the eval daemon.

One file: <output_root>/TRAINING_LOG.md
- Section per stage with start/end timestamps + final metrics
- Periodic step rows (every N steps)
- Eval-result blocks injected by the eval daemon, attributed to the source stage+step
- Process-safe: each appender opens, writes, closes (one shot per call), so multi-
  process appends from the trainer + eval daemon won't interleave at sub-line scale
  (we accept some interleaving across larger blocks under fcntl-less append).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any


_LOG_NAME = "TRAINING_LOG.md"


def _ts() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _append(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(text)


def log_path(output_root: str | Path) -> Path:
    return Path(output_root) / _LOG_NAME


def log_run_header(output_root: str | Path, config: dict):
    p = log_path(output_root)
    if not p.exists():
        cfg_short = {k: v for k, v in config.items() if k in {"run_name", "model", "data", "train", "ckpt"}}
        text = (
            f"# Training Log — Latent Spatial Reasoning Pipeline\n\n"
            f"**Started:** {_ts()}\n\n"
            f"**Config snapshot:**\n```json\n{json.dumps(cfg_short, indent=2)}\n```\n\n"
            f"---\n"
        )
        _append(p, text)


def log_stage_start(output_root: str | Path, stage: str, n_steps: int, n_gpus: int, extra: str = ""):
    text = (
        f"\n## {stage.upper()} — start\n\n"
        f"- **Time:** {_ts()}\n"
        f"- **Target steps:** {n_steps}\n"
        f"- **GPUs:** {n_gpus}\n"
    )
    if extra:
        text += f"- **Notes:** {extra}\n"
    text += "\n| step | total | text_ce | latent_l2 | latent_cos | extra | wall_s |\n|---|---|---|---|---|---|---|\n"
    _append(log_path(output_root), text)


def log_step(output_root: str | Path, step: int, metrics: dict, wall_s: float, extra: str = ""):
    line = (
        f"| {step} | {metrics.get('loss_total', 0):.4f} | "
        f"{metrics.get('loss_text_ce', 0):.4f} | "
        f"{metrics.get('latent_l2', 0):.4f} | "
        f"{metrics.get('latent_cos', 0):.4f} | "
        f"{extra} | {wall_s:.0f} |\n"
    )
    _append(log_path(output_root), line)


def log_stage_end(output_root: str | Path, stage: str, final_metrics: dict, ckpt_path: str = ""):
    text = (
        f"\n**{stage.upper()} — done at {_ts()}**\n\n"
        f"- final metrics: " + " ".join(f"{k}={v:.4f}" for k, v in final_metrics.items() if isinstance(v, (int, float))) + "\n"
    )
    if ckpt_path:
        text += f"- final checkpoint: `{ckpt_path}`\n"
    text += "\n---\n"
    _append(log_path(output_root), text)


def log_eval(output_root: str | Path, ckpt_path: str, source_stage: str, step: int, results: dict[str, Any]):
    text = (
        f"\n### Eval @ {_ts()}  (ckpt: {ckpt_path}, stage: {source_stage}, step: {step})\n\n"
    )
    for bench_name, bench_results in results.items():
        if isinstance(bench_results, dict):
            text += f"**{bench_name}:**\n"
            for metric, value in bench_results.items():
                if isinstance(value, float):
                    text += f"- {metric}: {value:.4f}\n"
                else:
                    text += f"- {metric}: {value}\n"
        else:
            text += f"- {bench_name}: {bench_results}\n"
    text += "\n"
    _append(log_path(output_root), text)


def log_event(output_root: str | Path, msg: str):
    _append(log_path(output_root), f"\n> [{_ts()}] {msg}\n")
