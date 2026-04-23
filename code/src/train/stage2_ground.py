"""Stage 2 — Latent Thought Grounding trainer.

DDP across N GPUs (devices set via CUDA_VISIBLE_DEVICES before torchrun).
- bf16 autocast on the backbone forward
- FP32 latent head for L2 numerical headroom
- Gradient checkpointing on the backbone
- Saves full state every N steps, prunes old, resumes if --resume given
- All checkpoints + logs live under <output_root>/{ckpts,logs}/<run_name>/

Launch (4 GPUs, devices 2,3,4,5):
    cd /home/haoming/latent_reasoning/code
    CUDA_VISIBLE_DEVICES=2,3,4,5 \
        torchrun --standalone --nproc_per_node=4 \
        -m src.train.stage2_ground --config configs/stage2_pilot.yaml [--resume]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Repo path so `python -m src.train.stage2_ground` works from `code/`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collate import make_collate_fn
from src.data.mindcube import MindCubeDataset
from src.models.encoders import FrozenSigLIPEncoder, TargetBuilder, ViewIdxPoseEncoder
from src.models.vlm_wrapper import build_model, latent_loss
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import MetricsLogger


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def dist_print(*a, **kw):
    if is_main():
        print(*a, **kw, flush=True)


# ---------------------------------------------------------------------------
# Optim / schedule
# ---------------------------------------------------------------------------

def build_optimizer(handles, cfg) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, p in handles.model.named_parameters():
        if not p.requires_grad:
            continue
        backbone_params.append(p)
    head_params += list(handles.latent_head.parameters())
    groups = [
        {"params": backbone_params, "lr": cfg["train"]["lr_backbone"], "weight_decay": cfg["train"]["weight_decay"]},
        {"params": head_params, "lr": cfg["train"]["lr_heads"], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def lr_at(step: int, cfg) -> tuple[float, float]:
    warmup = cfg["train"]["warmup_steps"]
    total = cfg["train"]["num_train_steps"]
    if step < warmup:
        scale = (step + 1) / max(1, warmup)
    elif cfg["train"].get("cosine_schedule", True):
        progress = (step - warmup) / max(1, total - warmup)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    else:
        scale = 1.0
    return scale * cfg["train"]["lr_backbone"], scale * cfg["train"]["lr_heads"]


# ---------------------------------------------------------------------------
# Forward + loss for one micro-batch
# ---------------------------------------------------------------------------

def forward_one_batch(handles, target_builder, batch, device, lam_text, lam_latent, lam_format, use_infonce, lts_id, lte_id):
    input_ids = batch.input_ids.to(device, non_blocking=True)
    attention_mask = batch.attention_mask.to(device, non_blocking=True)
    labels = batch.labels.to(device, non_blocking=True)
    pixel_values = batch.pixel_values.to(device, non_blocking=True) if batch.pixel_values is not None else None
    image_grid_thw = batch.image_grid_thw.to(device, non_blocking=True) if batch.image_grid_thw is not None else None

    # Compute targets via frozen encoders (no grad; happens on same device)
    targets = target_builder(batch.target_pil_images, batch.target_view_indices)  # (Nlat, target_dim)

    out = handles.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    text_loss = out.loss if out.loss is not None else input_ids.new_zeros((), dtype=torch.float32)
    last_hidden = out.hidden_states[-1]

    lat_loss, lat_metrics = latent_loss(
        last_hidden=last_hidden,
        input_ids=input_ids,
        targets=targets,
        target_batch_idx=batch.target_batch_idx,
        target_step_idx=batch.target_view_indices,
        lts_id=lts_id,
        latent_head=handles.latent_head,
        use_infonce=use_infonce,
    )

    # Format-CE: a small extra weight on getting the <lts>/<lte> structural tokens right.
    # Easy approximation: compute the fraction of <lts> and <lte> ground-truth tokens
    # and use it as a weight on the existing CE; we already include them via labels.
    # For monitoring only here.
    n_struct = int(((input_ids == lts_id) | (input_ids == lte_id)).sum().item())

    total = lam_text * text_loss + lam_latent * lat_loss
    metrics = {
        "loss_total": float(total.detach()),
        "loss_text_ce": float(text_loss.detach()),
        "loss_latent": float(lat_loss.detach()) if isinstance(lat_loss, torch.Tensor) else 0.0,
        **lat_metrics,
        "n_struct_tokens": n_struct,
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run 5 steps and exit, no saving")
    parser.add_argument("--max_steps_override", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps_override is not None:
        cfg["train"]["num_train_steps"] = args.max_steps_override

    rank, world, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    dist_print(f"[init] world={world} rank={rank} local_rank={local_rank} device={device}")

    # ------------------------------------------------------------------
    # Logger + ckpt
    # ------------------------------------------------------------------
    logger = MetricsLogger(
        output_root=cfg["output_root"],
        run_name=cfg["run_name"],
        use_wandb=cfg["log"].get("use_wandb", False),
        wandb_project=cfg["log"].get("wandb_project", "latent_reasoning"),
        config=cfg,
        is_main_process=is_main(),
    )
    ckpt = CheckpointManager(
        output_root=cfg["output_root"],
        run_name=cfg["run_name"],
        keep_last_n=cfg["ckpt"].get("keep_last_n", 2),
        save_optimizer=cfg["ckpt"].get("save_optimizer_state", True),
        is_main_process=is_main(),
    )
    logger.print(f"world={world} rank={rank} ckpt_root={ckpt.root}")

    # ------------------------------------------------------------------
    # Model + frozen encoders
    # ------------------------------------------------------------------
    dtype = torch.bfloat16 if cfg["model"]["backbone_dtype"] == "bfloat16" else torch.float32
    handles = build_model(
        backbone_id=cfg["model"]["backbone_id"],
        target_dim=0,  # set after building encoders
        dtype=dtype,
        freeze_vision_tower=cfg["model"].get("freeze_vision_tower", True),
    )

    siglip = FrozenSigLIPEncoder(cfg["model"]["siglip_path"], dtype=dtype)
    pose_enc = ViewIdxPoseEncoder(max_views=cfg["model"]["max_views_per_sample"] * 2,
                                  dim=cfg["model"]["view_idx_pose_dim"])
    target_builder = TargetBuilder(siglip, pose_enc).to(device)
    target_builder.eval()
    target_dim = target_builder.target_dim

    # Re-build the latent head now that we know target_dim
    from src.models.vlm_wrapper import LatentHead
    handles.latent_head = LatentHead(hidden_dim=handles.hidden_dim, target_dim=target_dim).to(device, dtype=torch.float32)
    handles.target_dim = target_dim

    # Move backbone to device, enable gradient checkpointing
    handles.model.to(device)
    if cfg["train"].get("gradient_checkpointing", True):
        handles.model.gradient_checkpointing_enable()
        if hasattr(handles.model, "enable_input_require_grads"):
            handles.model.enable_input_require_grads()

    # Wrap with DDP if multi-GPU
    if world > 1:
        ddp_model = DDP(handles.model, device_ids=[local_rank], find_unused_parameters=True,
                        broadcast_buffers=False, gradient_as_bucket_view=True)
        ddp_head = DDP(handles.latent_head, device_ids=[local_rank], find_unused_parameters=False)
        handles.model = ddp_model.module  # keep .module for state_dict
        handles.latent_head = ddp_head.module
        train_model = ddp_model
        train_head = ddp_head
    else:
        train_model = handles.model
        train_head = handles.latent_head

    n_params_trainable = sum(p.numel() for p in handles.model.parameters() if p.requires_grad) \
                       + sum(p.numel() for p in handles.latent_head.parameters() if p.requires_grad)
    logger.print(f"trainable_params={n_params_trainable/1e6:.1f}M | target_dim={target_dim}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = MindCubeDataset(
        jsonl_path=cfg["data"]["train_jsonl"],
        image_root=cfg["data"]["mindcube_root"],
        max_views=cfg["model"]["max_views_per_sample"],
    )
    if world > 1:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    else:
        sampler = None
    collate = make_collate_fn(handles.processor, handles.lts_id, handles.lat_id, handles.lte_id)
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["micro_batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["data"].get("num_workers", 2),
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    logger.print(f"train_dataset_size={len(train_ds)} | per_device_microbatch={cfg['train']['micro_batch_size']}")

    # ------------------------------------------------------------------
    # Optim + resume
    # ------------------------------------------------------------------
    optimizer = build_optimizer(handles, cfg)
    start_step = 0
    if args.resume and not args.smoke:
        loaded = ckpt.load()
        if loaded is not None:
            handles.model.load_state_dict(loaded["model_state"]["backbone"], strict=False)
            handles.latent_head.load_state_dict(loaded["model_state"]["latent_head"])
            if "optimizer_state" in loaded:
                optimizer.load_state_dict(loaded["optimizer_state"])
            start_step = loaded["meta"]["step"] + 1
            logger.print(f"resumed from step {loaded['meta']['step']}")
        else:
            logger.print("no checkpoint found; starting fresh")

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    train_model.train()
    train_head.train()

    grad_accum = cfg["train"]["grad_accum_steps"]
    total_steps = cfg["train"]["num_train_steps"]
    if args.smoke:
        total_steps = 5
        grad_accum = 1

    step = start_step
    iter_t0 = time.time()
    data_iter = iter(loader)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)

    last_save_t = time.time()

    while step < total_steps:
        # Accumulate grads across grad_accum micro-batches
        optimizer.zero_grad(set_to_none=True)
        accum_metrics = {"loss_total": 0.0, "loss_text_ce": 0.0, "loss_latent": 0.0,
                         "latent_l2": 0.0, "latent_cos": 0.0, "n_latents": 0}
        for accum_idx in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                batch = next(data_iter)

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss, metrics = forward_one_batch(
                    handles, target_builder, batch, device,
                    lam_text=cfg["train"]["loss_lambda_text"],
                    lam_latent=cfg["train"]["loss_lambda_latent"],
                    lam_format=cfg["train"]["loss_lambda_format"],
                    use_infonce=(step >= cfg["train"]["infonce_start_step"]),
                    lts_id=handles.lts_id,
                    lte_id=handles.lte_id,
                )
                loss = loss / grad_accum

            loss.backward()
            for k in accum_metrics:
                accum_metrics[k] = accum_metrics[k] + metrics.get(k, 0.0) / grad_accum

        # LR schedule
        lr_b, lr_h = lr_at(step, cfg)
        optimizer.param_groups[0]["lr"] = lr_b
        optimizer.param_groups[1]["lr"] = lr_h

        torch.nn.utils.clip_grad_norm_(
            list(handles.model.parameters()) + list(handles.latent_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        step += 1

        # ----------------------- log -----------------------
        if step % cfg["log"]["log_every_n_steps"] == 0 or step <= 3:
            t = time.time() - iter_t0
            iter_t0 = time.time()
            sps = (cfg["log"]["log_every_n_steps"] if step > cfg["log"]["log_every_n_steps"] else 1) / max(t, 1e-3)
            logger.print(
                f"step={step:>6d} loss={accum_metrics['loss_total']:.4f} "
                f"text_ce={accum_metrics['loss_text_ce']:.4f} "
                f"lat_l2={accum_metrics['latent_l2']:.4f} "
                f"lat_cos={accum_metrics['latent_cos']:.4f} "
                f"lr_b={lr_b:.2e} lr_h={lr_h:.2e} "
                f"steps/s={sps:.2f}"
            )
            logger.log(step, {**accum_metrics, "lr_backbone": lr_b, "lr_head": lr_h, "steps_per_sec": sps})

        # --------------------- save ckpt --------------------
        if not args.smoke and (step % cfg["ckpt"]["save_every_n_steps"] == 0 or step == total_steps):
            t0 = time.time()
            backbone_state = {k: v.detach().cpu() for k, v in handles.model.state_dict().items()}
            head_state = {k: v.detach().cpu() for k, v in handles.latent_head.state_dict().items()}
            opt_state = optimizer.state_dict() if cfg["ckpt"].get("save_optimizer_state", True) else None
            rng_state = {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            }
            ckpt.save(
                step=step,
                model_state={"backbone": backbone_state, "latent_head": head_state},
                optimizer_state=opt_state,
                rng_state=rng_state,
                meta={"epoch": epoch, **{k: float(v) for k, v in accum_metrics.items() if isinstance(v, (int, float))}},
            )
            if is_main():
                logger.print(f"[ckpt] saved step {step} in {time.time()-t0:.1f}s | disk_usage={ckpt.disk_usage_mb():.1f} MB")

    if is_main():
        logger.print("[done] training complete")
    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
