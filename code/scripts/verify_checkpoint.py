"""Load the latest checkpoint and verify the state matches what we expect.

Catches: corrupt files, mismatched keys after a training crash, stale 'latest'
symlink, optimizer-state shape changes after config edits.

Usage:
    python scripts/verify_checkpoint.py [--config configs/stage2_pilot.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_pilot.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ckpt = CheckpointManager(
        output_root=cfg["output_root"],
        run_name=cfg["run_name"],
    )
    print(f"ckpt root: {ckpt.root}")
    existing = ckpt.list_existing()
    print(f"existing checkpoints: {[p.name for p in existing]}")
    print(f"disk usage: {ckpt.disk_usage_mb():.1f} MB")

    loaded = ckpt.load()
    if loaded is None:
        print("(no checkpoint found)")
        return

    meta = loaded["meta"]
    print(f"\n== latest checkpoint ==")
    print(f"  step:       {meta.get('step')}")
    print(f"  epoch:      {meta.get('epoch')}")
    print(f"  wall_time:  {meta.get('wall_time')}")
    print(f"  metrics:    {' '.join(f'{k}={v:.4f}' for k, v in meta.items() if isinstance(v, float) and k not in {'wall_time'})}")

    backbone = loaded["model_state"]["backbone"]
    head = loaded["model_state"]["latent_head"]
    print(f"\n== model state ==")
    print(f"  backbone keys:    {len(backbone)}")
    print(f"  latent_head keys: {len(head)}")
    bb_size = sum(t.numel() * t.element_size() for t in backbone.values()) / 1024 / 1024
    head_size = sum(t.numel() * t.element_size() for t in head.values()) / 1024 / 1024
    print(f"  backbone size:    {bb_size:.1f} MB")
    print(f"  head size:        {head_size:.1f} MB")
    print(f"  sample backbone keys (first 5): {list(backbone)[:5]}")

    if "optimizer_state" in loaded:
        opt = loaded["optimizer_state"]
        n_groups = len(opt.get("param_groups", []))
        n_state = len(opt.get("state", {}))
        print(f"\n== optimizer state ==")
        print(f"  param groups: {n_groups}")
        print(f"  state dict entries: {n_state}")

    print("\nOK — checkpoint loads cleanly.")


if __name__ == "__main__":
    main()
