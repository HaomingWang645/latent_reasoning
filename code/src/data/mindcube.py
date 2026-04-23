"""MindCube dataset loader for Stage 2 latent grounding.

MindCube examples have multi-view images with directional/spatial questions.
We treat the K view images of an example as the per-step view targets — the
trace = [view_0, view_1, ..., view_{K-1}] in dataset order. The K-th `<lts>`
position's latent target is `concat(SigLIP(view_k), PoseEnc(k))`.

This stands in for the full grounded-detection trace builder until ScanNet/SQA3D
data is available. The architecture is identical; only the source of the per-step
view sequence changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class MindCubeSample:
    sample_id: str
    question: str
    images: list[Image.Image]    # PIL Images in dataset order
    image_paths: list[str]       # for debugging
    gt_answer: str
    n_views: int


class MindCubeDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        max_views: int = 4,
        min_views: int = 2,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.max_views = max_views
        self.min_views = min_views
        self.records = []
        with self.jsonl_path.open() as f:
            for line in f:
                rec = json.loads(line)
                imgs = rec.get("images", [])
                if min_views <= len(imgs) <= max_views:
                    self.records.append(rec)
        if not self.records:
            raise RuntimeError(f"No usable records in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MindCubeSample:
        rec = self.records[idx]
        imgs = []
        paths = []
        for rel in rec["images"]:
            p = self.image_root / rel
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(im)
                paths.append(str(p))
            except Exception as e:
                # Skip broken; if any image broken, return a fallback
                continue
        if len(imgs) < self.min_views:
            # cycle to a valid one
            return self[(idx + 1) % len(self)]
        return MindCubeSample(
            sample_id=rec.get("id", f"idx_{idx}"),
            question=rec["question"],
            images=imgs,
            image_paths=paths,
            gt_answer=rec.get("gt_answer", ""),
            n_views=len(imgs),
        )
