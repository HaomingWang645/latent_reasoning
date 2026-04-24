"""Run end-of-stage eval on a specific checkpoint.

MindCube tinybench (held out, same domain) + MMSI-Bench (OOD) + optionally VSI-Bench.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
        python -m scripts.post_stage_eval \
          --run_name full_pipeline_v3_stage1 \
          --stage stage1_align \
          --max_samples 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.spatial_eval import run_eval
from src.eval.mmsi_bench import run_mmsi_eval
from src.models.encoders import FrozenSigLIPEncoder, TargetBuilder, ViewIdxPoseEncoder
from src.models.vlm_wrapper import build_model, LatentHead
from src.utils import training_log as mdlog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_pipeline.yaml")
    parser.add_argument("--run_name", required=True, help="ckpt dir name, e.g., full_pipeline_v3_stage2")
    parser.add_argument("--stage", required=True, help="stage label for the log, e.g., stage2_ground")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--skip_mmsi", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda:0")
    output_root = cfg["output_root"]
    ckpt_root = Path(output_root) / "ckpts"

    # Build the same model architecture as the trainer
    dtype = torch.bfloat16
    handles = build_model(
        backbone_id=cfg["model"]["backbone_id"],
        target_dim=0, dtype=dtype,
        freeze_vision_tower=True,
        lora_cfg=cfg["model"].get("lora"),
    )
    target_dim = cfg["model"].get("vggt_target_dim", 832)
    handles.latent_head = LatentHead(
        hidden_dim=handles.hidden_dim, target_dim=target_dim,
    ).to(device, dtype=torch.float32)
    handles.target_dim = target_dim
    handles.model.to(device)

    # Load ckpt
    src = ckpt_root / args.run_name / "latest"
    if not src.exists():
        raise FileNotFoundError(f"No ckpt at {src}")
    state = torch.load(src.resolve() / "model_state.pt", map_location="cpu", weights_only=False)
    missing, unexpected = handles.model.load_state_dict(state["backbone"], strict=False)
    # latent_head may have a target_dim mismatch (e.g., if SigLIP/Pose was used previously)
    try:
        handles.latent_head.load_state_dict(state["latent_head"])
    except RuntimeError as e:
        print(f"[warn] latent_head load skipped due to shape mismatch: {e}")

    ckpt_name = src.resolve().name
    print(f"[{args.stage}] loaded {args.run_name}/{ckpt_name} | "
          f"backbone keys: {len(state['backbone'])} | missing: {len(missing)} | unexpected: {len(unexpected)}")

    # MindCube tinybench
    t0 = time.time()
    mc = run_eval(
        handles,
        eval_jsonl=cfg["data"]["eval_jsonl"],
        image_root=cfg["data"]["mindcube_root"],
        max_samples=args.max_samples,
        max_views=cfg["model"]["max_views_per_sample"],
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[MindCube tinybench] n={mc.n_samples} acc={mc.accuracy:.4f} "
          f"fmt={mc.format_rate:.4f} wall={mc.wall_seconds:.0f}s")

    results = {
        "MindCube_tinybench": {
            "n_samples": mc.n_samples, "n_correct": mc.n_correct,
            "accuracy": mc.accuracy, "format_rate": mc.format_rate,
            "wall_s": mc.wall_seconds,
        },
    }

    # MMSI-Bench
    if not args.skip_mmsi:
        try:
            mmsi = run_mmsi_eval(
                handles,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                max_views=cfg["model"]["max_views_per_sample"],
            )
            print(f"[MMSI-Bench] n={mmsi.n_samples} acc={mmsi.accuracy:.4f} "
                  f"fmt={mmsi.format_rate:.4f} wall={mmsi.wall_seconds:.0f}s")
            for qt, a in mmsi.per_type_accuracy.items():
                print(f"    {qt}: {a:.3f}")
            results["MMSI-Bench"] = {
                "n_samples": mmsi.n_samples, "n_correct": mmsi.n_correct,
                "accuracy": mmsi.accuracy, "format_rate": mmsi.format_rate,
                "wall_s": mmsi.wall_seconds,
                "per_type": mmsi.per_type_accuracy,
            }
        except Exception as e:
            print(f"[MMSI-Bench] FAILED: {e}")

    # Save + markdown log
    out_json = Path(output_root) / f"post_stage_eval_{args.run_name}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out_json}")
    mdlog.log_eval(output_root, ckpt_name, args.stage, step=-1, results=results)
    print(f"[log] appended to TRAINING_LOG.md")


if __name__ == "__main__":
    main()
