"""Stage 4 — Multi-GPU DDP GRPO (hardened from stage4_grpo_v2.py).

Hardenings vs v2:
  - **DDP across N GPUs**: each rank takes its own prompt (DistributedSampler).
    All ranks do K=4 rollouts in parallel, run ref + policy forwards locally,
    compute their own loss; gradients are all-reduced on backward. Reference
    swap happens on every rank simultaneously and consistently.
  - **Logit mask**: prevents the model from sampling `<|image_pad|>` and
    `<|video_pad|>` during rollouts. Eliminates the "image features and image
    tokens do not match" skip cascade we saw in v3.
  - **Multiplicative reward** (default): `format_present * (w_correct * correct
    + w_format)`. Format becomes a precondition for any reward — kills the
    "skip-format reward-hacking" pattern we saw in v3.
  - **KL EMA monitor + early-abort**: if running KL average stays above
    `kl_panic_threshold` for `kl_panic_window` steps, the script aborts (the
    last good ckpt is the keep_last_n=2 backup).

Launch (3 GPUs):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,5 \
        torchrun --standalone --nproc_per_node=3 \
        -m src.train.stage4_grpo_dist \
        --config configs/full_pipeline.yaml \
        --init_run_name full_pipeline_v2_stage3 \
        --ref_run_name  full_pipeline_v2_stage3 \
        --run_suffix _grpo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.mindcube import MindCubeDataset
from src.eval.spatial_eval import run_eval, _ANSWER_RE
from src.models.encoders import FrozenSigLIPEncoder, TargetBuilder, ViewIdxPoseEncoder
from src.models.vlm_wrapper import build_model, LatentHead
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


def parse_letter(text: str) -> str | None:
    m = _ANSWER_RE.search(text or "")
    return m.group(1) if m else None


def compute_rewards(
    decoded_list: list[str],
    gt: str,
    w_correct: float = 1.0,
    w_format: float = 0.05,
    format_gates_correct: bool = True,
) -> torch.Tensor:
    """Multiplicative-by-default reward.

    format_gates_correct=True (default):
        r = w_format * has_format + w_correct * is_correct * has_format
        - correct + format: w_correct + w_format
        - format only:      w_format
        - correct, no format: 0
        - neither:          0
    format_gates_correct=False (legacy v2 behavior):
        r = w_format * has_format + w_correct * is_correct
    """
    rewards = []
    for d in decoded_list:
        ans = parse_letter(d)
        has_format = "<lts>" in d and "<lte>" in d
        is_correct = (ans is not None and ans == gt.strip().upper())
        if format_gates_correct:
            r = w_format * float(has_format) + w_correct * float(is_correct and has_format)
        else:
            r = w_format * float(has_format) + w_correct * float(is_correct)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


class RefSwap:
    """In-place param swap context manager — same as v2 but called on every rank."""
    def __init__(self, model, ref_state):
        self.model, self.ref_state = model, ref_state
        self._saved = None

    def __enter__(self):
        self._saved = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ref_state:
                    self._saved[n] = p.data.clone()
                    p.data.copy_(self.ref_state[n].to(p.device, dtype=p.dtype))
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self._saved:
                    p.data.copy_(self._saved[n])
        self._saved = None


def sequence_logprobs(model, sequences, attention_mask, pixel_values, image_grid_thw):
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits
    lp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = sequences[:, 1:]
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--eval_interval_hours", type=float, default=5.0)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--init_run_name", default="full_pipeline_v1_stage3")
    parser.add_argument("--ref_run_name", default=None)
    parser.add_argument("--run_suffix", default="_grpo_dist")
    parser.add_argument("--w_correct", type=float, default=1.0)
    parser.add_argument("--w_format", type=float, default=0.05)
    parser.add_argument("--format_gates_correct", type=int, default=1)  # 1 = on
    parser.add_argument("--kl_panic_threshold", type=float, default=0.5)
    parser.add_argument("--kl_panic_window", type=int, default=20)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scfg = cfg["stage4_grpo"]
    if args.max_steps_override is not None:
        scfg["num_train_steps"] = args.max_steps_override

    rank, world, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    output_root = cfg["output_root"]
    run_name = scfg["run_name"] + args.run_suffix
    scfg["run_name"] = run_name
    dtype = torch.bfloat16

    logger = MetricsLogger(output_root=output_root, run_name=run_name,
                           config={**cfg, "_active_stage": "stage4_grpo_dist"},
                           is_main_process=is_main())
    ckpt = CheckpointManager(output_root=output_root, run_name=run_name,
                             keep_last_n=scfg["ckpt"].get("keep_last_n", 2),
                             save_optimizer=False, is_main_process=is_main())
    logger.print(f"world={world} rank={rank} stage4_grpo_dist run_name={run_name}")
    if is_main():
        mdlog.log_run_header(output_root, cfg)
        mdlog.log_stage_start(output_root, "stage4_grpo_dist", scfg["num_train_steps"], world,
                              extra=f"DDP K={scfg['num_generations']}, beta={scfg['beta']}, "
                                    f"clip_eps={args.clip_eps}, ref={args.ref_run_name or args.init_run_name}, "
                                    f"reward={'mult' if args.format_gates_correct else 'add'}")

    handles = build_model(
        backbone_id=cfg["model"]["backbone_id"],
        target_dim=0, dtype=dtype,
        freeze_vision_tower=cfg["model"].get("freeze_vision_tower", True),
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

    if scfg.get("gradient_checkpointing", True):
        handles.model.gradient_checkpointing_enable()
        if hasattr(handles.model, "enable_input_require_grads"):
            handles.model.enable_input_require_grads()

    # Load ref + init policy (same logic as v2)
    ref_run = args.ref_run_name or args.init_run_name
    ref_src = Path(output_root) / "ckpts" / ref_run / "latest"
    if ref_src.exists():
        ref_loaded = torch.load(ref_src.resolve() / "model_state.pt", map_location="cpu", weights_only=False)
        handles.model.load_state_dict(ref_loaded["backbone"], strict=False)
        handles.latent_head.load_state_dict(ref_loaded["latent_head"])
        trainable_names = {n for n, p in handles.model.named_parameters() if p.requires_grad}
        ref_state = {n: p.detach().clone() for n, p in handles.model.named_parameters() if n in trainable_names}
        logger.print(f"[ref] loaded ref from {ref_run} ({ref_src.resolve().name}); "
                     f"snapshotted {sum(v.numel() for v in ref_state.values())/1e6:.1f}M params")
    else:
        logger.print(f"[ref] WARNING: ckpt missing at {ref_src}; using fresh init as ref")
        trainable_names = {n for n, p in handles.model.named_parameters() if p.requires_grad}
        ref_state = {n: p.detach().clone() for n, p in handles.model.named_parameters() if n in trainable_names}

    if args.init_run_name != ref_run:
        init_src = Path(output_root) / "ckpts" / args.init_run_name / "latest"
        if init_src.exists():
            init_loaded = torch.load(init_src.resolve() / "model_state.pt", map_location="cpu", weights_only=False)
            handles.model.load_state_dict(init_loaded["backbone"], strict=False)
            handles.latent_head.load_state_dict(init_loaded["latent_head"])
            logger.print(f"[init] loaded policy from {args.init_run_name}")

    # DDP wrap
    if world > 1:
        ddp_model = DDP(handles.model, device_ids=[local_rank],
                        find_unused_parameters=True, broadcast_buffers=False,
                        gradient_as_bucket_view=True)
        ddp_head = DDP(handles.latent_head, device_ids=[local_rank], find_unused_parameters=False)
        train_model = ddp_model
        train_head = ddp_head
        handles.model = ddp_model.module
        handles.latent_head = ddp_head.module
    else:
        train_model = handles.model
        train_head = handles.latent_head

    optimizer = torch.optim.AdamW(
        [p for p in handles.model.parameters() if p.requires_grad] + list(handles.latent_head.parameters()),
        lr=scfg["lr"], betas=(0.9, 0.95), weight_decay=0.0,
    )

    train_ds = MindCubeDataset(
        jsonl_path=cfg["data"]["train_jsonl"],
        image_root=cfg["data"]["mindcube_root"],
        max_views=cfg["model"]["max_views_per_sample"],
    )
    handles.model.train(); handles.latent_head.train()
    handles.model.config.use_cache = True

    K = scfg["num_generations"]
    beta = scfg["beta"]
    max_new = scfg.get("max_completion_length", 128)
    pad_id = handles.processor.tokenizer.pad_token_id or handles.processor.tokenizer.eos_token_id

    # Logit mask: block image_pad, video_pad sampling during rollouts
    bad_token_strings = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    bad_words_ids = []
    for tk in bad_token_strings:
        try:
            tid = handles.processor.tokenizer.convert_tokens_to_ids(tk)
            if tid is not None and tid != handles.processor.tokenizer.unk_token_id:
                bad_words_ids.append([tid])
        except Exception:
            continue
    logger.print(f"[gen] bad_words_ids: {bad_words_ids}")

    total_steps = 3 if args.smoke else scfg["num_train_steps"]
    step = 0
    train_t0 = time.time()
    last_eval_t = time.time()
    n = len(train_ds)
    kl_window = []  # for panic monitor

    while step < total_steps:
        # Each rank picks its own prompt (rank-stratified for diversity)
        s_idx = (step * world + rank) % n
        s = train_ds[s_idx]
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img.resize((224, 224), Image.LANCZOS)} for img in s.images],
                {"type": "text", "text": s.question + "\n\nThink step by step using <lts><lat><lte> latent thoughts, one per view, then state the answer letter (A/B/C/D)."},
            ],
        }]
        text = handles.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc = handles.processor(
            text=[text],
            images=[img.resize((224, 224), Image.LANCZOS) for img in s.images],
            return_tensors="pt", padding=True, truncation=False,
        )
        proc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in proc.items()}
        prompt_len = proc["input_ids"].shape[1]

        with torch.no_grad():
            try:
                gen = handles.model.generate(
                    **proc, max_new_tokens=max_new,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    num_return_sequences=K,
                    pad_token_id=pad_id,
                    bad_words_ids=bad_words_ids if bad_words_ids else None,
                    return_dict_in_generate=True,
                )
            except Exception as e:
                logger.print(f"[r{rank}] gen failed: {e}; skip")
                step += 1
                continue
        sequences = gen.sequences
        gen_only = sequences[:, prompt_len:]
        decoded = [handles.processor.tokenizer.decode(g, skip_special_tokens=False) for g in gen_only]

        rewards = compute_rewards(
            decoded, s.gt_answer,
            w_correct=args.w_correct, w_format=args.w_format,
            format_gates_correct=bool(args.format_gates_correct),
        ).to(device)
        if rewards.std().item() < 1e-6:
            advantages = rewards - rewards.mean()
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        attn = (sequences != pad_id).long().to(device)
        pv = proc.get("pixel_values")
        thw = proc.get("image_grid_thw")
        pv_K = (pv.repeat(K, 1) if pv is not None and pv.dim() == 2 else
                (pv.repeat(K, 1, 1, 1) if pv is not None else None))
        thw_K = thw.repeat(K, 1) if thw is not None else None

        # Ref forward — use unwrapped model to avoid DDP sync (no_grad)
        try:
            with RefSwap(handles.model, ref_state):
                with torch.no_grad():
                    ref_lp = sequence_logprobs(handles.model, sequences, attn, pv_K, thw_K)
        except Exception as e:
            logger.print(f"[r{rank}] ref fwd failed: {e}; skip")
            step += 1
            continue
        ref_lp = ref_lp.detach()

        # Policy forward — DDP-wrapped for grad sync
        try:
            policy_lp = sequence_logprobs(train_model, sequences, attn, pv_K, thw_K)
        except Exception as e:
            logger.print(f"[r{rank}] policy fwd failed: {e}; skip")
            step += 1
            continue

        log_ratio = ref_lp - policy_lp
        kl_per_tok = log_ratio.exp() - log_ratio - 1.0

        log_p_old = policy_lp.detach()
        ratio = (policy_lp - log_p_old).exp()
        gen_mask = torch.zeros_like(policy_lp)
        gen_mask[:, prompt_len - 1:] = 1.0
        gen_mask = gen_mask * (sequences[:, 1:] != pad_id).float()

        adv = advantages.unsqueeze(-1)
        s1 = ratio * adv
        s2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
        policy_loss_per_tok = -torch.minimum(s1, s2)
        denom = gen_mask.sum(-1).clamp(min=1)
        pg_loss = (policy_loss_per_tok * gen_mask).sum(-1) / denom
        kl_loss = (kl_per_tok * gen_mask).sum(-1) / denom

        loss = pg_loss.mean() + beta * kl_loss.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(handles.model.parameters()) + list(handles.latent_head.parameters()), max_norm=1.0
        )
        optimizer.step()
        step += 1

        # KL panic monitor
        kl_value = float(kl_loss.mean().item())
        kl_window.append(kl_value)
        if len(kl_window) > args.kl_panic_window:
            kl_window.pop(0)
        if (len(kl_window) >= args.kl_panic_window and
                sum(kl_window) / len(kl_window) > args.kl_panic_threshold):
            logger.print(f"[PANIC] KL EMA over {args.kl_panic_window} steps = "
                         f"{sum(kl_window)/len(kl_window):.3f} > {args.kl_panic_threshold} — aborting")
            break

        if step % 5 == 0 or step <= 3:
            logger.print(
                f"[r{rank} s4d] step={step:>5d} R_mean={rewards.mean().item():.3f} R_max={rewards.max().item():.2f} "
                f"adv_std={advantages.std().item():.3f} pg={pg_loss.mean().item():.4f} "
                f"kl={kl_value:.4f} loss={loss.item():.4f}"
            )
            if is_main():
                logger.log(step, {
                    "reward_mean": float(rewards.mean()), "reward_max": float(rewards.max()),
                    "loss_total": float(loss), "pg_loss": float(pg_loss.mean()),
                    "kl_loss": kl_value,
                })
                if step % 25 == 0:
                    mdlog.log_step(output_root, step, {
                        "loss_total": float(loss), "loss_text_ce": float(pg_loss.mean()),
                        "latent_l2": kl_value, "latent_cos": float(rewards.mean()),
                    }, time.time() - train_t0,
                        extra=f"R={rewards.mean().item():.3f} kl={kl_value:.4f}")

        if not args.smoke and is_main() and (step % scfg["ckpt"]["save_every_n_steps"] == 0 or step == total_steps):
            t0 = time.time()
            backbone_state = {k: v.detach().cpu() for k, v in handles.model.state_dict().items()
                              if k in trainable_names}
            head_state = {k: v.detach().cpu() for k, v in handles.latent_head.state_dict().items()}
            ckpt.save(step=step,
                      model_state={"backbone": backbone_state, "latent_head": head_state},
                      optimizer_state=None, rng_state=None,
                      meta={"stage": "stage4_grpo_dist",
                            "reward_mean": float(rewards.mean()), "kl": kl_value})
            logger.print(f"[ckpt] saved step {step} in {time.time()-t0:.1f}s | disk={ckpt.disk_usage_mb():.1f} MB")

        if (time.time() - last_eval_t) / 3600.0 >= args.eval_interval_hours and is_main():
            try:
                res = run_eval(handles, eval_jsonl=cfg["eval_daemon"]["eval_jsonl"],
                               image_root=cfg["data"]["mindcube_root"],
                               max_samples=cfg["eval_daemon"].get("max_eval_samples", 200),
                               max_views=cfg["model"]["max_views_per_sample"])
                results = {"MindCube_tinybench": {
                    "n_samples": res.n_samples, "accuracy": res.accuracy,
                    "format_rate": res.format_rate, "wall_s": res.wall_seconds,
                }}
                logger.print(f"[eval] step={step} acc={res.accuracy:.3f} fmt={res.format_rate:.3f}")
                mdlog.log_eval(output_root, ckpt.latest().name if ckpt.latest() else "(none)",
                               "stage4_grpo_dist", step, results)
                handles.model.train()
                last_eval_t = time.time()
            except Exception as e:
                logger.print(f"[eval] FAILED: {e}")

    if is_main():
        logger.print("[done] stage4_grpo_dist complete")
        mdlog.log_stage_end(output_root, "stage4_grpo_dist",
                            {"reward_mean": float(rewards.mean()) if 'rewards' in locals() else 0.0,
                             "kl_loss": float(kl_loss.mean()) if 'kl_loss' in locals() else 0.0,
                             "loss": float(loss) if 'loss' in locals() else 0.0},
                            ckpt.latest().name if ckpt.latest() else "")
    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
