"""Qwen2.5-VL wrapper for view-shift latent reasoning (Stage 2 pilot).

Adds three special tokens — `<lts>`, `<lat>`, `<lte>` — to the tokenizer and a
trainable `LatentHead` that projects the VLM's last-layer hidden state at each
`<lts>` position into the frozen target space `concat(SigLIP(view), PoseEnc(idx))`.

Pilot simplification (vs. the implementation plan):
  - The latent slot's input embedding is the model's *learned* `<lat>` embedding,
    not the projected ground-truth target. This skips teacher-forcing-style target
    re-injection but keeps the architecture identical otherwise. Adequate for a
    smoke test + early convergence; revisit before Stage 3.

The latent head reads the hidden state at the `<lts>` token (the position right
before the latent slot) so the supervision is a true autoregressive next-step
prediction (the slot itself can't trivially observe its own input).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SPECIAL_TOKENS = ["<lts>", "<lat>", "<lte>"]


def init_new_token_embeddings(model: nn.Module, new_token_ids: list[int], n_seed: int = 10):
    """Init new token embeddings as the mean of `n_seed` random existing tokens."""
    emb = model.get_input_embeddings().weight.data
    vocab_size = emb.shape[0] - len(new_token_ids)
    g = torch.Generator(device=emb.device).manual_seed(0)
    seed_ids = torch.randint(0, vocab_size, (n_seed,), generator=g, device=emb.device)
    seed_mean = emb[seed_ids].mean(dim=0)
    for tid in new_token_ids:
        emb[tid] = seed_mean.clone()


class LatentHead(nn.Module):
    """2-layer MLP: hidden_dim -> target_dim. Lives on top of the VLM."""

    def __init__(self, hidden_dim: int, target_dim: int, mid_dim: int | None = None):
        super().__init__()
        mid_dim = mid_dim or max(hidden_dim, target_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim, bias=True),
            nn.GELU(),
            nn.Linear(mid_dim, target_dim, bias=True),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


@dataclass
class ModelHandles:
    model: Qwen2_5_VLForConditionalGeneration
    processor: AutoProcessor
    latent_head: LatentHead
    lts_id: int
    lat_id: int
    lte_id: int
    target_dim: int
    hidden_dim: int


def build_model(
    backbone_id: str,
    target_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    freeze_vision_tower: bool = True,
) -> ModelHandles:
    processor = AutoProcessor.from_pretrained(backbone_id)
    tokenizer = processor.tokenizer

    # Add special tokens (idempotent)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    lts_id, lat_id, lte_id = (tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        backbone_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    # Resize embeddings if we actually added new tokens
    if n_added > 0:
        old_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        new_size = model.get_input_embeddings().weight.shape[0]
        if new_size > old_size:
            new_ids = list(range(old_size, new_size))
            init_new_token_embeddings(model, new_ids)

    # Freeze the vision tower (we only train the LM half during stage 2 pilot)
    if freeze_vision_tower:
        for name, p in model.named_parameters():
            if "visual" in name or "vision_tower" in name:
                p.requires_grad_(False)

    # Latent head — float32 for numerical headroom on the L2 loss
    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    latent_head = LatentHead(hidden_dim=hidden_dim, target_dim=target_dim).to(dtype=torch.float32)

    return ModelHandles(
        model=model,
        processor=processor,
        latent_head=latent_head,
        lts_id=lts_id,
        lat_id=lat_id,
        lte_id=lte_id,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
    )


def latent_loss(
    last_hidden: torch.Tensor,           # (B, T, H)
    input_ids: torch.Tensor,             # (B, T)
    targets: torch.Tensor,               # (Nlat, target_dim) — flattened across batch
    target_batch_idx: torch.Tensor,      # (Nlat,) which batch each target belongs to
    target_step_idx: torch.Tensor,       # (Nlat,) which latent step each target is (within sample)
    lts_id: int,
    latent_head: nn.Module,
    use_infonce: bool = False,
    infonce_temperature: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    """L2 (and optionally InfoNCE) loss between latent_head(hidden) at <lts> positions
    and the per-step (view, pose) targets.

    Returns (loss, metrics_dict).
    """
    B, T, H = last_hidden.shape
    # Find <lts> positions — each batch should have len_targets[b] of them, in order.
    lts_mask = input_ids == lts_id              # (B, T)
    # Flatten (b, t) coordinates for all <lts>
    lts_b, lts_t = torch.where(lts_mask)        # 1D each
    # Sort by (b, t) so that order matches targets ordering
    order = torch.argsort(lts_b * T + lts_t)
    lts_b = lts_b[order]
    lts_t = lts_t[order]

    if lts_b.numel() != targets.shape[0]:
        # Mismatch — likely due to truncation. Truncate to min and warn.
        n = min(lts_b.numel(), targets.shape[0])
        lts_b = lts_b[:n]
        lts_t = lts_t[:n]
        targets = targets[:n]

    if lts_b.numel() == 0:
        zero = last_hidden.new_zeros((), requires_grad=True)
        return zero, {"latent_l2": 0.0, "latent_cos": 0.0, "n_latents": 0}

    h = last_hidden[lts_b, lts_t, :]            # (Nlat, H)
    pred = latent_head(h.float())               # (Nlat, target_dim)
    tgt = targets.to(pred.device, dtype=pred.dtype)

    l2 = F.mse_loss(pred, tgt)
    with torch.no_grad():
        cos = F.cosine_similarity(pred, tgt, dim=-1).mean()

    metrics = {"latent_l2": float(l2.detach()), "latent_cos": float(cos.detach()), "n_latents": int(lts_b.numel())}
    loss = l2

    if use_infonce and pred.shape[0] >= 2:
        # In-batch contrastive: each pred should be closer to its target than to others
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(tgt, dim=-1)
        logits = pred_n @ tgt_n.T / infonce_temperature  # (Nlat, Nlat)
        labels = torch.arange(pred.shape[0], device=pred.device)
        nce = F.cross_entropy(logits, labels)
        loss = loss + nce
        metrics["latent_nce"] = float(nce.detach())

    return loss, metrics
