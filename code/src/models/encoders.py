"""Frozen target encoders: SigLIP for view embedding + a learned positional encoding
over view-index as a stand-in for true SE(3) pose.

When real ScanNet/SQA3D data lands, swap ViewIdxPoseEncoder for a true PoseEncoder
that maps SE(3) -> R^d via 6D-rotation + Fourier translation features.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModel


class FrozenSigLIPEncoder(nn.Module):
    """Pooled image embedding from a SigLIP model. Frozen — no grad."""

    def __init__(self, model_path: str | Path, dtype=torch.bfloat16):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.embed_dim = self.model.config.vision_config.hidden_size
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, pil_images: list[Image.Image]) -> torch.Tensor:
        """Returns (B, D) pooled image embeddings on CPU dtype-as-set."""
        if len(pil_images) == 0:
            return torch.zeros(0, self.embed_dim, dtype=self.dtype)
        device = next(self.model.parameters()).device
        proc = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device=device, dtype=self.dtype)
        out = self.model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        # SigLIP uses pooler_output (last_hidden_state mean-pooled)
        emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        return emb  # (B, D)


class ViewIdxPoseEncoder(nn.Module):
    """Stand-in for SE(3) pose encoder: a learned embedding over [0, max_views) view-index.
    Frozen after a one-shot init for the pilot — replace with full PoseEnc when real
    SE(3) data is available.
    """

    def __init__(self, max_views: int = 32, dim: int = 64):
        super().__init__()
        self.max_views = max_views
        self.dim = dim
        # Sinusoidal embedding (deterministic, no grad needed)
        pe = torch.zeros(max_views, dim)
        position = torch.arange(0, max_views).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, view_indices: torch.Tensor) -> torch.Tensor:
        """view_indices: (B,) int. returns (B, dim)."""
        return self.pe[view_indices]


class TargetBuilder(nn.Module):
    """Combines SigLIP and ViewIdxPoseEncoder into the per-step latent target
    `concat(SigLIP(view_i), PoseEnc(idx_i))`.
    """

    def __init__(self, siglip: FrozenSigLIPEncoder, pose_enc: ViewIdxPoseEncoder):
        super().__init__()
        self.siglip = siglip
        self.pose_enc = pose_enc
        self.target_dim = siglip.embed_dim + pose_enc.dim

    @torch.no_grad()
    def forward(self, pil_images: list[Image.Image], view_indices: torch.Tensor) -> torch.Tensor:
        """Returns (B, target_dim)."""
        v = self.siglip(pil_images)                              # (B, D_v)
        p = self.pose_enc(view_indices.to(v.device)).to(v.dtype) # (B, D_p)
        return torch.cat([v, p], dim=-1)                         # (B, D_v+D_p)
