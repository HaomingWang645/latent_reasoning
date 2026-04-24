"""Collate function: pack a MindCube sample into the Qwen2.5-VL chat format
with `<lts><lat><lte>` latent slots before the answer.

For each sample, the assistant turn looks like:

    "I will examine the views in order. <lts><lat><lte> <lts><lat><lte> ... The answer is X."

with one `<lts><lat><lte>` triplet per view. The latent target for the k-th
triplet = `concat(SigLIP(view_k), PoseEnc(k))`, computed eagerly here from the
sample's PIL images.

The latent loss reads from hidden states at <lts> positions (see vlm_wrapper.latent_loss).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoProcessor

from .mindcube import MindCubeSample


def _build_assistant_text(n_views: int, gt_answer: str, lts_lat_lte: str) -> str:
    slots = " ".join([lts_lat_lte] * n_views)
    return f"I will examine the {n_views} views in order. {slots} The answer is {gt_answer}."


def _build_user_text(question: str, n_views: int) -> str:
    img_marker = " ".join(["<image>"] * n_views)  # processor will expand each
    return f"{img_marker}\n{question}"


@dataclass
class CollatedBatch:
    input_ids: torch.Tensor              # (B, T)
    attention_mask: torch.Tensor         # (B, T)
    pixel_values: torch.Tensor | None    # (Σviews_b, C, H, W) if any images
    image_grid_thw: torch.Tensor | None
    labels: torch.Tensor                 # (B, T) for text CE; -100 elsewhere
    target_pil_images: list[Image.Image] # all view images, flattened batch order
    target_view_indices: torch.Tensor    # (Σviews,) per-view in-sample index for PoseEnc
    target_batch_idx: torch.Tensor       # (Σviews,) which batch each target belongs to
    n_views_per_sample: list[int]
    target_features: torch.Tensor | None = None  # (Σviews, target_dim) — cached VGGT, fp16


def make_collate_fn(processor: AutoProcessor, lts_id: int, lat_id: int, lte_id: int):
    lts_lat_lte = "<lts><lat><lte>"
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    def collate(samples: list[MindCubeSample]) -> CollatedBatch:
        # Build one chat message per sample using Qwen-VL's chat template
        all_messages = []
        for s in samples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in s.images],
                        {"type": "text", "text": s.question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": _build_assistant_text(s.n_views, s.gt_answer, lts_lat_lte)},
                    ],
                },
            ]
            all_messages.append(messages)

        # Apply chat template per-sample
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in all_messages
        ]
        all_pil = [img for s in samples for img in s.images]

        # Resize images on input to keep image-token count small (each image
        # at 224x224 -> ~64 image tokens for Qwen2.5-VL). Truncation must NOT
        # cut the expanded image token spans, so we use a generous max_length.
        all_pil_resized = [im.resize((224, 224), Image.LANCZOS) for im in all_pil]
        proc_out = processor(
            text=texts,
            images=all_pil_resized if all_pil_resized else None,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = proc_out["input_ids"]
        attention_mask = proc_out["attention_mask"]
        pixel_values = proc_out.get("pixel_values")
        image_grid_thw = proc_out.get("image_grid_thw")

        # Build labels: text CE on everything except padding & <lat> slots
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        labels[input_ids == lat_id] = -100  # latent slots are not text

        # Per-view target metadata: every view in every sample, in order
        target_pil = []
        target_view_indices = []
        target_batch_idx = []
        n_views_per_sample = []
        target_feat_chunks = []
        any_vggt = False
        for b, s in enumerate(samples):
            n_views_per_sample.append(s.n_views)
            for k, img in enumerate(s.images):
                target_pil.append(img)
                target_view_indices.append(k)
                target_batch_idx.append(b)
            if s.vggt_features is not None:
                any_vggt = True
                target_feat_chunks.append(torch.from_numpy(s.vggt_features[:s.n_views]))

        target_features = None
        if any_vggt and len(target_feat_chunks) == len(samples):
            # (Σviews, target_dim)
            target_features = torch.cat(target_feat_chunks, dim=0)

        return CollatedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            target_pil_images=target_pil,
            target_view_indices=torch.tensor(target_view_indices, dtype=torch.long),
            target_batch_idx=torch.tensor(target_batch_idx, dtype=torch.long),
            n_views_per_sample=n_views_per_sample,
            target_features=target_features,
        )

    return collate
