"""Pre-compute VGGT-1B per-view features for all MindCube samples.

For each (question, K-views) sample, runs VGGT on the K views together and
saves the per-view aggregated features (mean-pooled patches) as a compact
.npy file keyed by sample_id.

Output layout:
    <cache_root>/<sample_id>.npy   shape (K, 2048), dtype float16

Multi-GPU friendly: launch with torchrun and the script will partition the
dataset across ranks. Resumable: skips samples whose .npy already exists.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,5 \
      torchrun --standalone --nproc_per_node=3 \
        -m scripts.precompute_vggt \
        --jsonl /home/haoming/mindcube_data/raw/MindCube_train.jsonl \
        --image_root /home/haoming/mindcube_data \
        --vggt_path /mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs/hf_cache/VGGT-1B \
        --out_dir /mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs/vggt_cache/MindCube_train \
        --min_views 4 --max_views 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vggt_encoder import VGGTTargetBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--vggt_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_views", type=int, default=4)
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument("--micro_batch_samples", type=int, default=2,
                        help="N samples processed per VGGT forward (each = K views)")
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    if rank == 0:
        print(f"[precompute] world={world} out_dir={out_dir} jsonl={args.jsonl}")

    # Load + filter
    records = []
    with open(args.jsonl) as f:
        for line in f:
            rec = json.loads(line)
            n = len(rec.get("images", []))
            if args.min_views <= n <= args.max_views:
                records.append(rec)
    if rank == 0:
        print(f"[precompute] {len(records)} samples after view filter")

    # Partition by rank
    my_records = [r for i, r in enumerate(records) if i % world == rank]
    if rank == 0:
        print(f"[precompute] each rank handles ~{len(my_records)} samples")

    # Build VGGT
    vggt = VGGTTargetBuilder(args.vggt_path, dtype=torch.bfloat16).to(device)

    # Process
    t0 = time.time()
    n_done = 0; n_skip = 0; n_fail = 0
    micro = args.micro_batch_samples
    for i in range(0, len(my_records), micro):
        batch = my_records[i:i+micro]
        # Skip already-done
        todo = []
        for rec in batch:
            sid = rec.get("id", f"idx_{i}")
            if (out_dir / f"{sid}.npy").exists():
                n_skip += 1
                continue
            todo.append(rec)
        if not todo:
            continue

        # Load PIL images per sample
        groups = []
        sids = []
        ok = []
        for rec in todo:
            sid = rec.get("id")
            imgs = []
            valid = True
            for rel in rec["images"]:
                p = image_root / rel
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    valid = False
                    break
            if valid and len(imgs) == args.max_views:
                groups.append(imgs)
                sids.append(sid)
                ok.append(rec)
            else:
                n_fail += 1

        if not groups:
            continue

        try:
            with torch.no_grad():
                # VGGT preprocesses + forwards
                # Re-implement to get per-(b,s) features without going through
                # the flattening logic
                x = vggt._preprocess(groups, device)  # (B, S, 3, 518, 518)
                layer_outputs, _ = vggt.model.aggregator(x)
                last = layer_outputs[-1]            # (B, S, P, D)
                per_view = last.mean(dim=2)         # (B, S, D)
                per_view = per_view.detach().cpu().to(torch.float16).numpy()

            for b, sid in enumerate(sids):
                np.save(out_dir / f"{sid}.npy", per_view[b])  # (K, D)
                n_done += 1
        except Exception as e:
            print(f"[r{rank}] VGGT forward failed: {e}")
            n_fail += len(groups)
            continue

        if (n_done + n_skip) % 100 < micro and rank == 0:
            sps = n_done / max(time.time() - t0, 1e-3)
            eta = (len(my_records) - n_done - n_skip) / max(sps, 1e-3) / 60
            print(f"[r0] done={n_done} skip={n_skip} fail={n_fail} "
                  f"sps={sps:.2f} eta={eta:.0f}min")

    print(f"[r{rank}] FINAL done={n_done} skip={n_skip} fail={n_fail} "
          f"wall={time.time()-t0:.0f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
