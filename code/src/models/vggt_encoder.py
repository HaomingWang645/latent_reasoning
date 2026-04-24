"""Frozen VGGT-1B target encoder.

Implements pathway 1 from `latent reasoning ideas.pdf` p.1: the model's latent
thought tokens at each `<lts>` position are supervised against VGGT's per-view
aggregated features. VGGT (Visual Geometry Grounded Transformer, Facebook 2025)
takes K multi-view RGB and produces per-view patch tokens that encode 3D
geometry-aware semantics — exactly the "semantically rich latent space" the
deck specifies.

Per-view target = mean-pool over the 1374 patches of each view's last-layer
aggregator output → 2048-dim vector.

Frozen — no grad. Lives on whichever device the trainer puts it on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class VGGTTargetBuilder(nn.Module):
    """Wraps VGGT-1B for per-view target extraction.

    Interface (mirrors the SigLIP+PoseEnc TargetBuilder so the trainer can
    swap them with no other code changes):

      forward(pil_images_grouped: list[list[Image]], view_indices: tensor)
        -> (Nlat, target_dim)

    `pil_images_grouped[b]` is the K views of batch sample b (in dataset order).
    `view_indices` is (Nlat,) with the per-sample view index of each output row.
    `target_dim` is fixed at 2048 (VGGT-1B aggregator embed dim).
    """

    INPUT_RES = 518  # VGGT-1B canonical input resolution

    def __init__(self, model_path: str | Path, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained(model_path)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.dtype = dtype
        # VGGT-1B aggregator output: (B, S, 1374, 2048)
        self.embed_dim = 2048
        self.target_dim = self.embed_dim
        # ImageNet normalization (VGGT inherits DINOv2 normalization)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _preprocess(self, pil_images_grouped: list[list[Image.Image]], device) -> torch.Tensor:
        """Returns (B, S_max, 3, 518, 518) bf16 tensor; pads short groups with zeros."""
        B = len(pil_images_grouped)
        S_max = max(len(g) for g in pil_images_grouped)
        out = torch.zeros((B, S_max, 3, self.INPUT_RES, self.INPUT_RES),
                          dtype=torch.float32)
        for b, group in enumerate(pil_images_grouped):
            for s, im in enumerate(group):
                im_r = im.convert("RGB").resize((self.INPUT_RES, self.INPUT_RES), Image.LANCZOS)
                arr = np.asarray(im_r).astype(np.float32) / 255.0  # H,W,3 in [0,1]
                out[b, s] = torch.from_numpy(arr).permute(2, 0, 1)
        out = out.to(device=device, dtype=self.dtype)
        # Normalize
        out = (out - self.mean.to(out.device, out.dtype)) / self.std.to(out.device, out.dtype)
        return out

    @torch.no_grad()
    def forward(
        self,
        pil_images_grouped: list[list[Image.Image]],
        view_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (Nlat, target_dim) per-view features in flat order matching
        view_indices' batch_idx ordering."""
        device = next(self.model.parameters()).device
        if not pil_images_grouped:
            return torch.zeros((0, self.target_dim), dtype=self.dtype, device=device)

        x = self._preprocess(pil_images_grouped, device)  # (B, S_max, 3, 518, 518)
        # VGGT.aggregator returns (list_of_layer_outputs, ps_idx)
        layer_outputs, _ = self.model.aggregator(x)
        # Take last layer: (B, S_max, P=1374, D=2048)
        last = layer_outputs[-1]
        # Mean-pool over patches → (B, S_max, D)
        per_view_feat = last.mean(dim=2)
        # Gather per-view features in trace-builder order
        # view_indices is (Nlat,) of view-within-sample indices; need batch idx too.
        # Caller is expected to flatten in (b0_v0, b0_v1, ..., b1_v0, ...) order, so
        # we just unstack per-batch and re-flatten using the actual lengths of pil_images_grouped.
        out = []
        for b, group in enumerate(pil_images_grouped):
            for s in range(len(group)):
                out.append(per_view_feat[b, s])
        return torch.stack(out, dim=0)  # (Nlat, D)
