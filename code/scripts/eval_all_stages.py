"""Run MindCube-tinybench eval across: base model + Stage 1/2/3/4 latest checkpoints.

Appends a comparison block to TRAINING_LOG.md and writes eval_results.json.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
        python -m scripts.eval_all_stages --max_samples 500
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
from src.models.encoders import FrozenSigLIPEncoder, TargetBuilder, ViewIdxPoseEncoder
from src.models.vlm_wrapper import build_model, LatentHead
from src.utils import training_log as mdlog


STAGE_MAP = {
    "base":      None,  # no ckpt load
    "stage1":    "full_pipeline_v1_stage1",
    "stage2":    "full_pipeline_v1_stage2",
    "stage3":    "full_pipeline_v1_stage3",
    "stage4":    "full_pipeline_v1_stage4",
    "stage4_v2": "full_pipeline_v1_stage4_v2",  # proper GRPO 1000 steps
    "stage4_v3": "full_pipeline_v1_stage4_v3",  # proper GRPO 3000 steps total (continued)
}


def load_ckpt(handles, ckpt_root: Path, run_name: str):
    src = ckpt_root / run_name / "latest"
    if not src.exists():
        return None
    state = torch.load(src.resolve() / "model_state.pt", map_location="cpu", weights_only=False)
    missing, unexpected = handles.model.load_state_dict(state["backbone"], strict=False)
    handles.latent_head.load_state_dict(state["latent_head"])
    return src.resolve().name, len(state["backbone"]), len(missing), len(unexpected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_pipeline.yaml")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--stages", nargs="+", default=["base", "stage1", "stage2", "stage3", "stage4"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda:0")
    output_root = cfg["output_root"]
    ckpt_root = Path(output_root) / "ckpts"
    dtype = torch.bfloat16

    print(f"[build] loading base Qwen2.5-VL-7B + LoRA ...")
    t0 = time.time()
    handles = build_model(
        backbone_id=cfg["model"]["backbone_id"],
        target_dim=0, dtype=dtype,
        freeze_vision_tower=True,
        lora_cfg=cfg["model"].get("lora"),
    )
    siglip = FrozenSigLIPEncoder(cfg["model"]["siglip_path"], dtype=dtype)
    pose_enc = ViewIdxPoseEncoder(max_views=cfg["model"]["max_views_per_sample"] * 2,
                                  dim=cfg["model"]["view_idx_pose_dim"])
    target_builder = TargetBuilder(siglip, pose_enc).to(device)
    target_dim = target_builder.target_dim
    handles.latent_head = LatentHead(hidden_dim=handles.hidden_dim, target_dim=target_dim).to(device, dtype=torch.float32)
    handles.target_dim = target_dim
    handles.model.to(device)
    print(f"[build] done in {time.time()-t0:.0f}s")

    # Snapshot the initial (base) state so we can restore before each stage load
    init_state = {k: v.detach().cpu() for k, v in handles.model.state_dict().items()}
    init_head = {k: v.detach().cpu() for k, v in handles.latent_head.state_dict().items()}

    results = {}
    for stage in args.stages:
        run_name = STAGE_MAP.get(stage)
        if stage != "base":
            # reset to base first
            handles.model.load_state_dict(init_state, strict=True)
            handles.latent_head.load_state_dict(init_head, strict=True)
            info = load_ckpt(handles, ckpt_root, run_name)
            if info is None:
                print(f"[{stage}] no ckpt at {ckpt_root / run_name}, skipping")
                continue
            ckpt_name, n_keys, n_miss, n_unexp = info
            print(f"[{stage}] loaded {ckpt_name} | ckpt keys={n_keys} miss={n_miss} unexp={n_unexp}")
        else:
            print(f"[base] no ckpt, using fresh init")

        t0 = time.time()
        res = run_eval(
            handles,
            eval_jsonl=cfg["data"]["eval_jsonl"],
            image_root=cfg["data"]["mindcube_root"],
            max_samples=args.max_samples,
            max_views=cfg["model"]["max_views_per_sample"],
            max_new_tokens=args.max_new_tokens,
        )
        results[stage] = {
            "n_samples": res.n_samples,
            "n_correct": res.n_correct,
            "accuracy": res.accuracy,
            "format_rate": res.format_rate,
            "wall_seconds": res.wall_seconds,
        }
        print(f"[{stage}] n={res.n_samples} correct={res.n_correct} "
              f"acc={res.accuracy:.4f} fmt={res.format_rate:.4f} "
              f"wall={res.wall_seconds:.0f}s")

    # Save + log
    out_json = Path(output_root) / "eval_all_stages.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out_json}")

    # Markdown summary
    md = ["\n## Full Eval Sweep — MindCube Tinybench\n\n",
          f"**Run:** {time.strftime('%Y-%m-%d %H:%M:%S')}  |  **Samples per stage:** {args.max_samples}\n\n",
          "| stage | ckpt | samples | correct | accuracy | format_rate | wall(s) |\n",
          "|---|---|---|---|---|---|---|\n"]
    for stage, r in results.items():
        run_name = STAGE_MAP.get(stage, "-") or "(fresh LoRA init)"
        md.append(f"| {stage} | {run_name} | {r['n_samples']} | {r['n_correct']} | "
                  f"{r['accuracy']:.4f} | {r['format_rate']:.4f} | {r['wall_seconds']:.0f} |\n")
    md.append("\n")
    mdlog_path = Path(output_root) / "TRAINING_LOG.md"
    with mdlog_path.open("a") as f:
        f.writelines(md)
    print(f"[log] appended to {mdlog_path}")


if __name__ == "__main__":
    main()
