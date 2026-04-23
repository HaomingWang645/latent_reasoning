"""Pretty-print the training metrics JSONL for a run.

Usage:
    python scripts/inspect_run.py [run_name]
    python scripts/inspect_run.py stage2_pilot_mindcube_v0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

DEFAULT_CFG = Path(__file__).parents[1] / "configs" / "stage2_pilot.yaml"


def main():
    cfg = yaml.safe_load(open(DEFAULT_CFG))
    run_name = sys.argv[1] if len(sys.argv) > 1 else cfg["run_name"]
    out_root = Path(cfg["output_root"])
    log_path = out_root / "logs" / run_name / "metrics.jsonl"
    ckpt_root = out_root / "ckpts" / run_name
    if not log_path.exists():
        print(f"(no metrics yet at {log_path})")
        return
    rows = [json.loads(l) for l in log_path.open() if l.strip()]
    if not rows:
        print("(metrics.jsonl is empty)")
        return

    last = rows[-1]
    print(f"== {run_name}  |  metrics: {log_path.name}  ({len(rows)} rows) ==")
    print(f"latest step: {last['step']:>6d}  wall: {last['wall']:.0f}s")
    keys = ["loss_total", "loss_text_ce", "loss_latent", "latent_l2", "latent_cos",
            "n_latents", "lr_backbone", "lr_head", "steps_per_sec"]
    for k in keys:
        if k in last:
            v = last[k]
            print(f"  {k:>16s} = {v:.4f}" if isinstance(v, float) else f"  {k:>16s} = {v}")

    print(f"\n== curve (every 10th row) ==")
    print(f"{'step':>6s}  {'loss':>8s}  {'text_ce':>8s}  {'lat_l2':>8s}  {'lat_cos':>8s}")
    for row in rows[::max(1, len(rows) // 20)]:
        print(f"{row['step']:>6d}  {row.get('loss_total',0):>8.4f}  {row.get('loss_text_ce',0):>8.4f}  "
              f"{row.get('latent_l2',0):>8.4f}  {row.get('latent_cos',0):>8.4f}")

    print("\n== checkpoints ==")
    if ckpt_root.exists():
        ckpts = sorted([p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")])
        total_mb = sum(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in ckpts) / 1024 / 1024
        for c in ckpts:
            sz = sum(f.stat().st_size for f in c.rglob("*") if f.is_file()) / 1024 / 1024
            print(f"  {c.name}  {sz:>7.1f} MB")
        latest = ckpt_root / "latest"
        if latest.is_symlink():
            print(f"  latest -> {latest.resolve().name}")
        print(f"  total disk: {total_mb:.1f} MB")
    else:
        print("  (no checkpoints yet)")


if __name__ == "__main__":
    main()
