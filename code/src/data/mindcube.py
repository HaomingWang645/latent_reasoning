"""MindCube dataset loader for Stage 2 latent grounding.

MindCube examples have multi-view images with directional/spatial questions.
We treat the K view images of an example as the per-step view targets — the
trace = [view_0, view_1, ..., view_{K-1}] in dataset order.

Targets:
  - SigLIP+pose (legacy view-shift pathway): computed live in the trainer
  - VGGT (pathway 1, primary): pre-computed once by `scripts.precompute_vggt`,
    loaded from `vggt_cache_dir/<sample_id>.npy` shape (K, 2048)

Records without a cache file are filtered out at __init__ when `vggt_cache_dir`
is set, so every __getitem__ returns a sample with cached targets available.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class MindCubeSample:
    sample_id: str
    question: str
    images: list[Image.Image]
    image_paths: list[str]
    gt_answer: str
    n_views: int
    vggt_features: np.ndarray | None = None   # (K, 2048) fp16 if cached


class MindCubeDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        max_views: int = 4,
        min_views: int = 2,
        vggt_cache_dir: str | Path | None = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.max_views = max_views
        self.min_views = min_views
        self.vggt_cache_dir = Path(vggt_cache_dir) if vggt_cache_dir else None

        self.records = []
        with self.jsonl_path.open() as f:
            for line in f:
                rec = json.loads(line)
                imgs = rec.get("images", [])
                if not (min_views <= len(imgs) <= max_views):
                    continue
                if self.vggt_cache_dir is not None:
                    sid = rec.get("id")
                    if sid is None or not (self.vggt_cache_dir / f"{sid}.npy").exists():
                        continue
                self.records.append(rec)
        if not self.records:
            raise RuntimeError(f"No usable records in {jsonl_path}"
                               + (f" (no VGGT cache hit in {vggt_cache_dir})" if vggt_cache_dir else ""))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MindCubeSample:
        rec = self.records[idx]
        sid = rec.get("id", f"idx_{idx}")
        imgs, paths = [], []
        for rel in rec["images"]:
            p = self.image_root / rel
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(im)
                paths.append(str(p))
            except Exception:
                continue
        if len(imgs) < self.min_views:
            return self[(idx + 1) % len(self)]

        vggt_feats = None
        if self.vggt_cache_dir is not None:
            try:
                vggt_feats = np.load(self.vggt_cache_dir / f"{sid}.npy")  # (K, 2048)
                if vggt_feats.shape[0] != len(imgs):
                    # Mismatch — fall back as if no cache
                    vggt_feats = None
            except Exception:
                vggt_feats = None

        return MindCubeSample(
            sample_id=sid,
            question=rec["question"],
            images=imgs,
            image_paths=paths,
            gt_answer=rec.get("gt_answer", ""),
            n_views=len(imgs),
            vggt_features=vggt_feats,
        )
