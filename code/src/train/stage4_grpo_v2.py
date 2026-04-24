"""Stage 4 — Proper GRPO (replaces simplified REINFORCE in stage4_grpo.py).

Differences from v1:
  - **Reference model**: snapshot of trainable params at init (Stage 3 ckpt).
    Per-token KL(policy || ref) computed via a real ref forward pass.
  - **Group-relative advantage**: A_k = (R_k - mean(R)) / (std(R) + eps), normalized
    *within the K rollouts of each prompt*. Same formula as in DeepSeek's GRPO paper.
  - **PPO-clipped surrogate**: min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).
    With single-step on-policy training the ratio is identically 1, but the clip
    keeps the loss well-behaved and ready for multi-epoch.
  - **Trust region via KL**: total loss = pg_loss + beta * kl. beta = 0.04 per plan.
    This is the protection against the policy collapse that broke v1.

Reference-model trick: instead of two separate model copies (would need 2x memory
for the 7B backbone), we snapshot only the trainable params (~1.1B × bf16 = 2.2 GB)
to GPU memory and use an in-place param swap helper to switch between policy and
ref states for the ref-side forward pass. Base weights are frozen and identical
on both sides, so they never need swapping.

Single-GPU. Launch:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
        python -m src.train.stage4_grpo_v2 --config configs/full_pipeline.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.mindcube import MindCubeDataset
from src.eval.spatial_eval import run_eval, _ANSWER_RE
from src.models.encoders import FrozenSigLIPEncoder, TargetBuilder, ViewIdxPoseEncoder
from src.models.vlm_wrapper import build_model, LatentHead
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import MetricsLogger
from src.utils import training_log as mdlog


def parse_letter(text: str) -> str | None:
    m = _ANSWER_RE.search(text or "")
    return m.group(1) if m else None


def compute_rewards(decoded_list, gt: str, w_correct: float, w_format: float) -> torch.Tensor:
    rewards = []
    for d in decoded_list:
        r = 0.0
        ans = parse_letter(d)
        if ans is not None and ans == gt.strip().upper():
            r += w_correct
        if "<lts>" in d and "<lte>" in d:
            r += w_format
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


class RefSwap:
    """Context manager: temporarily swaps the model's trainable params with the
    reference snapshot for a no-grad reference forward, then restores."""

    def __init__(self, model, ref_state: dict[str, torch.Tensor]):
        self.model = model
        self.ref_state = ref_state
        self._saved = None

    def __enter__(self):
        # Save current trainable params, load ref in their place.
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
    """Forward `sequences` through model and return per-token logprobs at the
    sampled tokens. Returns (logprobs[K, T-1], logits[K, T, V]).
    `sequences` is the prompt+completion; we score positions 1..T-1.
    """
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits  # (K, T, V)
    lp = F.log_softmax(logits[:, :-1, :], dim=-1)            # (K, T-1, V)
    tgt = sequences[:, 1:]                                    # (K, T-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # (K, T-1)
    return tok_lp, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--eval_interval_hours", type=float, default=5.0)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--init_run_name", default="full_pipeline_v1_stage3",
                        help="run_name to load initial policy weights from")
    parser.add_argument("--ref_run_name", default=None,
                        help="run_name to load REFERENCE (KL anchor) weights from. "
                             "Defaults to init_run_name (canonical GRPO).")
    parser.add_argument("--run_suffix", default="_v2",
                        help="suffix appended to scfg.run_name for ckpt/log dirs")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scfg = cfg["stage4_grpo"]
    if args.max_steps_override is not None:
        scfg["num_train_steps"] = args.max_steps_override

    device = torch.device("cuda:0")
    output_root = cfg["output_root"]
    run_name = scfg["run_name"] + args.run_suffix
    scfg["run_name"] = run_name
    dtype = torch.bfloat16

    logger = MetricsLogger(output_root=output_root, run_name=run_name,
                           config={**cfg, "_active_stage": "stage4_grpo_v2"}, is_main_process=True)
    ckpt = CheckpointManager(output_root=output_root, run_name=run_name,
                             keep_last_n=scfg["ckpt"].get("keep_last_n", 2),
                             save_optimizer=False, is_main_process=True)
    logger.print(f"stage4_grpo_v2 run_name={run_name}")
    mdlog.log_run_header(output_root, cfg)
    mdlog.log_stage_start(output_root, "stage4_grpo_v2", scfg["num_train_steps"], 1,
                          extra=f"PROPER GRPO: K={scfg['num_generations']}, beta={scfg['beta']}, "
                                f"clip_eps={args.clip_eps}, ref=stage3 fixed")

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

    # 1. Load REFERENCE first (so we can snapshot before policy weights change)
    ref_run = args.ref_run_name or args.init_run_name
    ref_src = Path(output_root) / "ckpts" / ref_run / "latest"
    if ref_src.exists():
        ref_loaded = torch.load(ref_src.resolve() / "model_state.pt", map_location="cpu", weights_only=False)
        handles.model.load_state_dict(ref_loaded["backbone"], strict=False)
        handles.latent_head.load_state_dict(ref_loaded["latent_head"])
        trainable_names = {n for n, p in handles.model.named_parameters() if p.requires_grad}
        ref_state = {n: p.detach().clone() for n, p in handles.model.named_parameters() if n in trainable_names}
        logger.print(f"[ref] loaded ref weights from {ref_run} ({ref_src.resolve().name}); "
                     f"snapshotted {sum(v.numel() for v in ref_state.values())/1e6:.1f}M params")
    else:
        logger.print(f"[ref] WARNING: ckpt missing at {ref_src}; using fresh init as ref")
        trainable_names = {n for n, p in handles.model.named_parameters() if p.requires_grad}
        ref_state = {n: p.detach().clone() for n, p in handles.model.named_parameters() if n in trainable_names}

    # 2. Load INITIAL POLICY (overrides the ref weights in the model — but ref_state is already saved on GPU)
    if args.init_run_name != ref_run:
        init_src = Path(output_root) / "ckpts" / args.init_run_name / "latest"
        if init_src.exists():
            init_loaded = torch.load(init_src.resolve() / "model_state.pt", map_location="cpu", weights_only=False)
            handles.model.load_state_dict(init_loaded["backbone"], strict=False)
            handles.latent_head.load_state_dict(init_loaded["latent_head"])
            logger.print(f"[init] loaded policy weights from {args.init_run_name} ({init_src.resolve().name})")
        else:
            logger.print(f"[init] WARNING: ckpt missing at {init_src}; policy = ref")
    else:
        logger.print(f"[init] policy weights = ref weights ({ref_run})")

    optimizer = torch.optim.AdamW(
        [p for p in handles.model.parameters() if p.requires_grad] + list(handles.latent_head.parameters()),
        lr=scfg["lr"], betas=(0.9, 0.95), weight_decay=0.0,
    )

    train_ds = MindCubeDataset(
        jsonl_path=cfg["data"]["train_jsonl"],
        image_root=cfg["data"]["mindcube_root"],
        max_views=cfg["model"]["max_views_per_sample"],
    )
    handles.model.train()
    handles.model.config.use_cache = True

    K = scfg["num_generations"]
    beta = scfg["beta"]
    w_correct = scfg.get("reward_correct", 1.0)
    w_format = scfg.get("reward_format", 0.1)
    max_new = scfg.get("max_completion_length", 128)
    pad_id = handles.processor.tokenizer.pad_token_id or handles.processor.tokenizer.eos_token_id

    total_steps = 3 if args.smoke else scfg["num_train_steps"]
    step = 0
    train_t0 = time.time()
    last_eval_t = time.time()
    n = len(train_ds)

    while step < total_steps:
        s = train_ds[step % n]
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

        # 1. K rollouts (no grad)
        with torch.no_grad():
            try:
                gen = handles.model.generate(
                    **proc, max_new_tokens=max_new,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    num_return_sequences=K,
                    pad_token_id=pad_id,
                    return_dict_in_generate=True,
                )
            except Exception as e:
                logger.print(f"[stage4_v2] gen failed: {e}; skip")
                step += 1
                continue
        sequences = gen.sequences
        gen_only = sequences[:, prompt_len:]
        decoded = [handles.processor.tokenizer.decode(g, skip_special_tokens=False) for g in gen_only]

        rewards = compute_rewards(decoded, s.gt_answer, w_correct, w_format).to(device)
        # Group-relative advantage normalized within the K rollouts of THIS prompt
        if rewards.std().item() < 1e-6:
            advantages = rewards - rewards.mean()  # all zeros if no variance
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # Replicate image conditioning to K rollouts
        attn = (sequences != pad_id).long().to(device)
        pv = proc.get("pixel_values")
        thw = proc.get("image_grid_thw")
        pv_K = (pv.repeat(K, 1) if pv is not None and pv.dim() == 2 else
                (pv.repeat(K, 1, 1, 1) if pv is not None else None))
        thw_K = thw.repeat(K, 1) if thw is not None else None

        # 2. Reference log_probs (no grad, ref params)
        try:
            with RefSwap(handles.model, ref_state):
                with torch.no_grad():
                    ref_lp, _ = sequence_logprobs(handles.model, sequences, attn, pv_K, thw_K)
        except Exception as e:
            logger.print(f"[stage4_v2] ref fwd failed: {e}; skip")
            step += 1
            continue
        ref_lp = ref_lp.detach()

        # 3. Policy log_probs (with grad, current policy params)
        try:
            policy_lp, _ = sequence_logprobs(handles.model, sequences, attn, pv_K, thw_K)
        except Exception as e:
            logger.print(f"[stage4_v2] policy fwd failed: {e}; skip")
            step += 1
            continue

        # 4. Per-token KL(policy || ref). Approx via single sample at sampled tokens:
        #    KL ≈ exp(ref - policy) - (ref - policy) - 1   (Schulman's "k3" estimator,
        #    unbiased, non-negative, low variance). Standard in trl GRPO.
        log_ratio = ref_lp - policy_lp  # (K, T-1)
        kl_per_tok = log_ratio.exp() - log_ratio - 1.0

        # 5. PPO-clipped surrogate. Single-step on-policy → log_p_old = policy_lp.detach().
        log_p_old = policy_lp.detach()
        ratio = (policy_lp - log_p_old).exp()
        # Mask: only the GENERATED tokens contribute (positions >= prompt_len-1 in target idx)
        gen_mask = torch.zeros_like(policy_lp)
        gen_mask[:, prompt_len - 1:] = 1.0
        gen_mask = gen_mask * (sequences[:, 1:] != pad_id).float()

        adv = advantages.unsqueeze(-1)  # (K, 1) broadcast over T-1
        s1 = ratio * adv
        s2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
        policy_loss_per_tok = -torch.minimum(s1, s2)

        denom = gen_mask.sum(-1).clamp(min=1)
        pg_loss = (policy_loss_per_tok * gen_mask).sum(-1) / denom    # (K,)
        kl_loss = (kl_per_tok * gen_mask).sum(-1) / denom            # (K,)

        loss = pg_loss.mean() + beta * kl_loss.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(handles.model.parameters()) + list(handles.latent_head.parameters()), max_norm=1.0
        )
        optimizer.step()
        step += 1

        if step % 5 == 0 or step <= 3:
            logger.print(
                f"[stage4_v2] step={step:>5d} R_mean={rewards.mean().item():.3f} R_max={rewards.max().item():.2f} "
                f"adv_std={advantages.std().item():.3f} pg={pg_loss.mean().item():.4f} "
                f"kl={kl_loss.mean().item():.4f} loss={loss.item():.4f}"
            )
            logger.log(step, {
                "reward_mean": float(rewards.mean()), "reward_max": float(rewards.max()),
                "loss_total": float(loss), "pg_loss": float(pg_loss.mean()),
                "kl_loss": float(kl_loss.mean()),
            })
            if step % 25 == 0:
                mdlog.log_step(output_root, step, {
                    "loss_total": float(loss), "loss_text_ce": float(pg_loss.mean()),
                    "latent_l2": float(kl_loss.mean()), "latent_cos": float(rewards.mean()),
                }, time.time() - train_t0,
                    extra=f"R={rewards.mean().item():.3f} kl={kl_loss.mean().item():.4f}")

        if not args.smoke and (step % scfg["ckpt"]["save_every_n_steps"] == 0 or step == total_steps):
            t0 = time.time()
            backbone_state = {k: v.detach().cpu() for k, v in handles.model.state_dict().items()
                              if k in trainable_names}
            head_state = {k: v.detach().cpu() for k, v in handles.latent_head.state_dict().items()}
            ckpt.save(step=step,
                      model_state={"backbone": backbone_state, "latent_head": head_state},
                      optimizer_state=None, rng_state=None,
                      meta={"stage": "stage4_grpo_v2",
                            "reward_mean": float(rewards.mean()), "kl": float(kl_loss.mean())})
            logger.print(f"[ckpt] saved step {step} in {time.time()-t0:.1f}s | disk={ckpt.disk_usage_mb():.1f} MB")

        # 5h periodic eval
        if (time.time() - last_eval_t) / 3600.0 >= args.eval_interval_hours:
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
                               "stage4_grpo_v2", step, results)
                handles.model.train()
                last_eval_t = time.time()
            except Exception as e:
                logger.print(f"[eval] FAILED: {e}")

    logger.print("[done] stage4_grpo_v2 complete")
    mdlog.log_stage_end(output_root, "stage4_grpo_v2",
                        {"reward_mean": float(rewards.mean()) if 'rewards' in locals() else 0.0,
                         "kl_loss": float(kl_loss.mean()) if 'kl_loss' in locals() else 0.0,
                         "loss": float(loss) if 'loss' in locals() else 0.0},
                        ckpt.latest().name if ckpt.latest() else "")
    logger.close()


if __name__ == "__main__":
    main()
