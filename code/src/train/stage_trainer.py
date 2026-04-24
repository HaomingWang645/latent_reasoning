"""Unified trainer for stages 1, 2, 3.

  - stage1_align: freeze backbone+LoRA, train only latent head + new token embeds + LM head
  - stage2_ground: full LoRA + latent head, text + latent loss
  - stage3_e2e: full LoRA + latent head, text-only loss

Stage 4 (GRPO) lives in stage4_grpo.py.

Common features:
  - DDP across CUDA_VISIBLE_DEVICES
  - LoRA via PEFT, gradient checkpointing
  - Save/resume to /mnt/data3 with keep_last_n pruning
  - --stage <name> reads stage-specific subsection of config
  - init_from <run_name>: load model state from another run's latest ckpt
  - Periodic eval on MindCube tinybench every `eval_interval_hours`,
    rank-0 only, in-process, results appended to TRAINING_LOG.md

Launch:
  CUDA_VISIBLE_DEVICES=2,3,5 torchrun --standalone --nproc_per_node=3 \
    -m src.train.stage_trainer \
    --config configs/full_pipeline.yaml --stage stage2_ground [--resume]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collate import make_collate_fn
from src.data.mindcube import MindCubeDataset
from src.eval.spatial_eval import run_eval
from src.models.encoders import FrozenSigLIPEncoder, TargetBuilder, ViewIdxPoseEncoder
from src.models.vlm_wrapper import build_model, latent_loss, LatentHead
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import MetricsLogger
from src.utils import training_log as mdlog


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


def lr_schedule(step: int, lr: float, warmup: int, total: int, cosine: bool = True) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if cosine:
        prog = (step - warmup) / max(1, total - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return lr


def build_optimizer(handles, scfg) -> torch.optim.Optimizer:
    backbone_params = [p for n, p in handles.model.named_parameters() if p.requires_grad]
    head_params = list(handles.latent_head.parameters())
    groups = [
        {"params": backbone_params, "lr": scfg.get("lr_backbone", 0.0), "weight_decay": scfg.get("weight_decay", 0.01)},
        {"params": head_params, "lr": scfg.get("lr_heads", scfg.get("lr_head", 1e-4)), "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def apply_stage_freezing(handles, stage: str):
    """Stage 1 freezes the LoRA adapters; stages 2+ leave them trainable."""
    if stage == "stage1_align":
        for n, p in handles.model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(False)


def apply_init_from(handles, output_root: Path, init_run_name: str, logger):
    """Load ONLY model state from another run's latest checkpoint (not optimizer)."""
    src = output_root / "ckpts" / init_run_name / "latest"
    if not src.exists():
        logger.print(f"[init_from] no ckpt at {src}, skipping")
        return
    src_resolved = src.resolve()
    state_pt = src_resolved / "model_state.pt"
    if not state_pt.exists():
        logger.print(f"[init_from] no model_state.pt at {src_resolved}")
        return
    state = torch.load(state_pt, map_location="cpu", weights_only=False)
    missing_b, unexpected_b = handles.model.load_state_dict(state["backbone"], strict=False)
    handles.latent_head.load_state_dict(state["latent_head"])
    logger.print(f"[init_from] loaded {init_run_name} latest ({src_resolved.name}); "
                 f"missing={len(missing_b)} unexpected={len(unexpected_b)}")


def maybe_run_eval(handles, eval_cfg, image_root, last_eval_t, interval_h, output_root,
                   source_stage, step, ckpt_dir, logger):
    """Returns new last_eval_t. Only rank 0 runs eval; other ranks barrier."""
    elapsed_h = (time.time() - last_eval_t) / 3600.0
    if elapsed_h < interval_h:
        return last_eval_t
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if is_main():
        try:
            res = run_eval(
                handles,
                eval_jsonl=eval_cfg["eval_jsonl"],
                image_root=image_root,
                max_samples=eval_cfg.get("max_eval_samples", 200),
                max_views=eval_cfg.get("max_views", 4),
                max_new_tokens=eval_cfg.get("max_new_tokens", 64),
            )
            results = {"MindCube_tinybench": {
                "n_samples": res.n_samples,
                "accuracy": res.accuracy,
                "format_rate": res.format_rate,
                "wall_s": res.wall_seconds,
            }}
            logger.print(f"[eval] step={step} acc={res.accuracy:.3f} fmt={res.format_rate:.3f} "
                         f"({res.n_correct}/{res.n_samples}) wall={res.wall_seconds:.0f}s")
            mdlog.log_eval(output_root, ckpt_dir, source_stage, step, results)
        except Exception as e:
            logger.print(f"[eval] FAILED: {type(e).__name__}: {e}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return time.time()


def forward_one_batch(handles, target_builder, batch, device, scfg, dtype, lts_id, lte_id):
    input_ids = batch.input_ids.to(device, non_blocking=True)
    attention_mask = batch.attention_mask.to(device, non_blocking=True)
    labels = batch.labels.to(device, non_blocking=True)
    pixel_values = batch.pixel_values.to(device, non_blocking=True) if batch.pixel_values is not None else None
    image_grid_thw = batch.image_grid_thw.to(device, non_blocking=True) if batch.image_grid_thw is not None else None

    # Prefer cached VGGT features from the loader if present; else fall back to
    # live SigLIP+PoseEnc computation.
    if getattr(batch, "target_features", None) is not None:
        targets = batch.target_features.to(device, dtype=torch.float32, non_blocking=True)
    else:
        targets = target_builder(batch.target_pil_images, batch.target_view_indices)

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
        use_infonce=False,
    )

    lam_text = scfg.get("loss_lambda_text", 1.0)
    lam_lat = scfg.get("loss_lambda_latent", 1.0)
    total = lam_text * text_loss + lam_lat * lat_loss
    metrics = {
        "loss_total": float(total.detach()),
        "loss_text_ce": float(text_loss.detach()),
        "loss_latent": float(lat_loss.detach()) if isinstance(lat_loss, torch.Tensor) else 0.0,
        **lat_metrics,
    }
    return total, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["stage1_align", "stage2_ground", "stage3_e2e"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--eval_interval_hours", type=float, default=5.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scfg = cfg[args.stage]
    if args.max_steps_override is not None:
        scfg["num_train_steps"] = args.max_steps_override

    rank, world, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    output_root = cfg["output_root"]
    run_name = scfg["run_name"]

    logger = MetricsLogger(
        output_root=output_root,
        run_name=run_name,
        config={**cfg, "_active_stage": args.stage},
        is_main_process=is_main(),
    )
    ckpt = CheckpointManager(
        output_root=output_root,
        run_name=run_name,
        keep_last_n=scfg["ckpt"].get("keep_last_n", 2),
        save_optimizer=scfg["ckpt"].get("save_optimizer_state", True),
        is_main_process=is_main(),
    )
    logger.print(f"world={world} rank={rank} stage={args.stage} run_name={run_name}")
    if is_main():
        mdlog.log_run_header(output_root, cfg)
        mdlog.log_stage_start(output_root, args.stage, scfg["num_train_steps"], world,
                              extra=f"backbone={cfg['model']['backbone_id']}, init_from={scfg.get('init_from','none')}")

    dtype = torch.bfloat16 if cfg["model"]["backbone_dtype"] == "bfloat16" else torch.float32
    handles = build_model(
        backbone_id=cfg["model"]["backbone_id"],
        target_dim=0,
        dtype=dtype,
        freeze_vision_tower=cfg["model"].get("freeze_vision_tower", True),
        lora_cfg=cfg["model"].get("lora"),
    )

    # Target-encoder selection: "vggt" (cached) or "siglip_pose" (live)
    target_mode = cfg["model"].get("target_encoder", "siglip_pose")
    if target_mode == "vggt":
        target_builder = None
        target_dim = cfg["model"].get("vggt_target_dim", 2048)
        logger.print(f"[target] vggt cached mode, target_dim={target_dim}")
    else:
        siglip = FrozenSigLIPEncoder(cfg["model"]["siglip_path"], dtype=dtype)
        pose_enc = ViewIdxPoseEncoder(max_views=cfg["model"]["max_views_per_sample"] * 2,
                                      dim=cfg["model"]["view_idx_pose_dim"])
        target_builder = TargetBuilder(siglip, pose_enc).to(device)
        target_builder.eval()
        target_dim = target_builder.target_dim
        logger.print(f"[target] siglip+pose live mode, target_dim={target_dim}")

    handles.latent_head = LatentHead(hidden_dim=handles.hidden_dim, target_dim=target_dim).to(device, dtype=torch.float32)
    handles.target_dim = target_dim

    handles.model.to(device)
    if scfg.get("gradient_checkpointing", True):
        handles.model.gradient_checkpointing_enable()
        if hasattr(handles.model, "enable_input_require_grads"):
            handles.model.enable_input_require_grads()

    apply_stage_freezing(handles, args.stage)

    init_from = scfg.get("init_from")
    if init_from and not args.resume:
        apply_init_from(handles, Path(output_root), init_from, logger)

    if world > 1:
        ddp_model = DDP(handles.model, device_ids=[local_rank], find_unused_parameters=True,
                        broadcast_buffers=False, gradient_as_bucket_view=True)
        ddp_head = DDP(handles.latent_head, device_ids=[local_rank], find_unused_parameters=False)
        train_model = ddp_model
        train_head = ddp_head
        handles.model = ddp_model.module
        handles.latent_head = ddp_head.module
    else:
        train_model = handles.model
        train_head = handles.latent_head

    n_params_trainable = sum(p.numel() for p in handles.model.parameters() if p.requires_grad) \
                       + sum(p.numel() for p in handles.latent_head.parameters() if p.requires_grad)
    logger.print(f"trainable_params={n_params_trainable/1e6:.1f}M | target_dim={target_dim}")

    vggt_cache_train = cfg["data"].get("vggt_cache_train") if target_mode == "vggt" else None
    train_ds = MindCubeDataset(
        jsonl_path=cfg["data"]["train_jsonl"],
        image_root=cfg["data"]["mindcube_root"],
        max_views=cfg["model"]["max_views_per_sample"],
        min_views=cfg["model"].get("min_views_per_sample", cfg["model"]["max_views_per_sample"] if target_mode == "vggt" else 2),
        vggt_cache_dir=vggt_cache_train,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=0) if world > 1 else None
    collate = make_collate_fn(handles.processor, handles.lts_id, handles.lat_id, handles.lte_id)
    loader = DataLoader(
        train_ds, batch_size=scfg["micro_batch_size"], sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg["data"].get("num_workers", 2),
        collate_fn=collate, pin_memory=True, drop_last=True,
    )
    logger.print(f"train_dataset_size={len(train_ds)}")

    optimizer = build_optimizer(handles, scfg)
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

    train_model.train(); train_head.train()
    grad_accum = scfg["grad_accum_steps"]
    total_steps = scfg["num_train_steps"]
    if args.smoke:
        total_steps = 5; grad_accum = 1

    step = start_step
    iter_t0 = time.time()
    train_t0 = time.time()
    last_eval_t = time.time()
    data_iter = iter(loader)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)

    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accum = {"loss_total": 0.0, "loss_text_ce": 0.0, "loss_latent": 0.0,
                 "latent_l2": 0.0, "latent_cos": 0.0, "n_latents": 0}
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                batch = next(data_iter)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss, m = forward_one_batch(handles, target_builder, batch, device, scfg, dtype,
                                            handles.lts_id, handles.lte_id)
                loss = loss / grad_accum
            loss.backward()
            for k in accum:
                accum[k] = accum[k] + m.get(k, 0.0) / grad_accum

        for pg, key in zip(optimizer.param_groups, ["lr_backbone", "lr_heads"]):
            base = scfg.get(key, scfg.get("lr_head" if key == "lr_heads" else key, 0.0))
            pg["lr"] = lr_schedule(step, base, scfg.get("warmup_steps", 50), total_steps,
                                   scfg.get("cosine_schedule", True))

        torch.nn.utils.clip_grad_norm_(
            list(handles.model.parameters()) + list(handles.latent_head.parameters()), max_norm=1.0
        )
        optimizer.step()
        step += 1

        if step % scfg.get("log_every_n_steps", 10) == 0 or step <= 3:
            t = time.time() - iter_t0; iter_t0 = time.time()
            sps = (scfg.get("log_every_n_steps", 10) if step > scfg.get("log_every_n_steps", 10) else 1) / max(t, 1e-3)
            wall_s = time.time() - train_t0
            logger.print(
                f"[{args.stage}] step={step:>6d} loss={accum['loss_total']:.4f} "
                f"text_ce={accum['loss_text_ce']:.4f} lat_l2={accum['latent_l2']:.4f} "
                f"lat_cos={accum['latent_cos']:.4f} sps={sps:.2f}"
            )
            logger.log(step, {**accum, "lr_backbone": optimizer.param_groups[0]["lr"],
                              "lr_head": optimizer.param_groups[1]["lr"], "steps_per_sec": sps})
            if is_main() and step % (scfg.get("log_every_n_steps", 10) * 5) == 0:
                mdlog.log_step(output_root, step, accum, wall_s)

        if not args.smoke and (step % scfg["ckpt"]["save_every_n_steps"] == 0 or step == total_steps):
            t0 = time.time()
            trainable_names = {n for n, p in handles.model.named_parameters() if p.requires_grad}
            backbone_state = {k: v.detach().cpu() for k, v in handles.model.state_dict().items() if k in trainable_names}
            head_state = {k: v.detach().cpu() for k, v in handles.latent_head.state_dict().items()}
            opt_state = optimizer.state_dict() if scfg["ckpt"].get("save_optimizer_state", True) else None
            rng_state = {"torch": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
            ckpt.save(step=step,
                      model_state={"backbone": backbone_state, "latent_head": head_state},
                      optimizer_state=opt_state, rng_state=rng_state,
                      meta={"epoch": epoch, "stage": args.stage,
                            **{k: float(v) for k, v in accum.items() if isinstance(v, (int, float))}})
            if is_main():
                logger.print(f"[ckpt] saved step {step} in {time.time()-t0:.1f}s | disk={ckpt.disk_usage_mb():.1f} MB")
                try:
                    import subprocess
                    subprocess.run(["python", str(Path(__file__).parents[2] / "scripts" / "plot_curves.py"), run_name],
                                   check=False, capture_output=True, timeout=30)
                except Exception:
                    pass

        # 5h periodic eval
        eval_cfg = cfg.get("eval_daemon", {})
        eval_cfg.setdefault("eval_jsonl", cfg["data"]["eval_jsonl"])
        if eval_cfg.get("max_eval_samples", 0) > 0:
            ckpt_dir = ckpt.latest().name if ckpt.latest() else "(none)"
            last_eval_t = maybe_run_eval(
                handles, eval_cfg, cfg["data"]["mindcube_root"], last_eval_t,
                args.eval_interval_hours, output_root, args.stage, step, ckpt_dir, logger
            )

    if is_main():
        logger.print(f"[done] {args.stage} complete")
        ckpt_path = ckpt.latest().name if ckpt.latest() else "(none)"
        mdlog.log_stage_end(output_root, args.stage, accum, ckpt_path)

        # End-of-stage OOD eval: MindCube tinybench (held out) + MMSI-Bench (OOD)
        try:
            logger.print("[post-stage eval] running MindCube tinybench + MMSI-Bench ...")
            ec = cfg.get("eval_daemon", {})
            ec.setdefault("eval_jsonl", cfg["data"]["eval_jsonl"])
            # MindCube tinybench
            mc = run_eval(
                handles,
                eval_jsonl=ec["eval_jsonl"],
                image_root=cfg["data"]["mindcube_root"],
                max_samples=ec.get("max_eval_samples", 500),
                max_views=cfg["model"]["max_views_per_sample"],
            )
            # MMSI-Bench
            from src.eval.mmsi_bench import run_mmsi_eval
            mmsi = run_mmsi_eval(
                handles,
                max_samples=500,
                max_views=cfg["model"]["max_views_per_sample"],
            )
            results = {
                "MindCube_tinybench": {
                    "n_samples": mc.n_samples, "accuracy": mc.accuracy,
                    "format_rate": mc.format_rate, "wall_s": mc.wall_seconds,
                },
                "MMSI-Bench": {
                    "n_samples": mmsi.n_samples, "accuracy": mmsi.accuracy,
                    "format_rate": mmsi.format_rate, "wall_s": mmsi.wall_seconds,
                    "per_type": mmsi.per_type_accuracy,
                },
            }
            logger.print(f"[post-stage eval] MindCube acc={mc.accuracy:.3f} fmt={mc.format_rate:.3f} | "
                         f"MMSI-Bench acc={mmsi.accuracy:.3f} fmt={mmsi.format_rate:.3f}")
            mdlog.log_eval(output_root, ckpt_path, args.stage, step, results)
        except Exception as e:
            logger.print(f"[post-stage eval] FAILED: {type(e).__name__}: {e}")
    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
