"""Render a 4-panel training-curve PNG from the run's metrics.jsonl.

Saves <run_log_dir>/curves.png. Safe to call mid-training — re-reads the JSONL
each time and overwrites the PNG. Use for monitoring without wandb.

Usage:
    python scripts/plot_curves.py [run_name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

DEFAULT_CFG = Path(__file__).parents[1] / "configs" / "stage2_pilot.yaml"


def main():
    cfg = yaml.safe_load(open(DEFAULT_CFG))
    run_name = sys.argv[1] if len(sys.argv) > 1 else cfg["run_name"]
    log_dir = Path(cfg["output_root"]) / "logs" / run_name
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"no metrics at {metrics_path}")
        return

    rows = [json.loads(l) for l in metrics_path.open() if l.strip()]
    if not rows:
        print("metrics.jsonl is empty")
        return

    steps = [r["step"] for r in rows]
    text_ce = [r.get("loss_text_ce", float("nan")) for r in rows]
    lat_l2 = [r.get("latent_l2", float("nan")) for r in rows]
    lat_cos = [r.get("latent_cos", float("nan")) for r in rows]
    total = [r.get("loss_total", float("nan")) for r in rows]
    lr_b = [r.get("lr_backbone", float("nan")) for r in rows]
    lr_h = [r.get("lr_head", float("nan")) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(f"{run_name}  —  {len(rows)} log rows, latest step {steps[-1]}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(steps, total, label="total", lw=1)
    ax.plot(steps, text_ce, label="text_ce", lw=1)
    ax.plot(steps, lat_l2, label="latent_l2", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("losses (log-y)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, lat_cos, color="tab:green", lw=1.2)
    ax.axhline(0.8, ls="--", color="gray", alpha=0.5, label="M3 milestone (0.8)")
    ax.set_xlabel("step")
    ax.set_ylabel("cosine sim")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("latent_cos vs target  (the load-bearing metric)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, lat_l2, color="tab:orange", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("L2")
    ax.set_title("latent_l2 (linear)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, lr_b, label="lr_backbone")
    ax.plot(steps, lr_h, label="lr_head")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.legend(fontsize=8)
    ax.set_title("learning-rate schedule")
    ax.grid(True, alpha=0.3)

    out = log_dir / "curves.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}  ({len(rows)} rows, step 1 -> {steps[-1]})")
    print(f"  final lat_cos = {lat_cos[-1]:.4f}, lat_l2 = {lat_l2[-1]:.4f}, text_ce = {text_ce[-1]:.4f}")


if __name__ == "__main__":
    main()
