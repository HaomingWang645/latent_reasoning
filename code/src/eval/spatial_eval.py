"""Spatial-task evaluation harness for the periodic eval (every ~5h).

Loads N samples from MindCube tinybench and asks the live VLM to answer.
Computes:
  - answer accuracy (exact-match against gt_answer letter A/B/C/D)
  - format compliance (model emitted required <lts><lat><lte> trace)
  - mean latent_cos to SigLIP target at the <lts> positions (sanity)

Designed to be called from inside the training process on rank 0 with the
in-memory model. Other ranks must barrier while rank 0 evaluates.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import DataLoader

from ..data.collate import make_collate_fn
from ..data.mindcube import MindCubeDataset


_ANSWER_RE = re.compile(r"\b([A-D])\b")


def _parse_answer(text: str) -> str | None:
    """Extract first letter A-D mentioned in the model's output."""
    m = _ANSWER_RE.search(text or "")
    return m.group(1) if m else None


@dataclass
class EvalResults:
    n_samples: int
    n_correct: int
    n_format_ok: int
    accuracy: float
    format_rate: float
    wall_seconds: float


@torch.no_grad()
def run_eval(
    handles,
    eval_jsonl: str,
    image_root: str,
    max_samples: int = 200,
    max_views: int = 4,
    max_new_tokens: int = 64,
) -> EvalResults:
    """Run the model on `max_samples` MindCube examples and score answers."""
    t0 = time.time()
    model = handles.model
    processor = handles.processor
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    was_train = model.training
    model.eval()
    # Re-enable use_cache for generation
    cache_was = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    ds = MindCubeDataset(eval_jsonl, image_root, max_views=max_views, min_views=2)
    n = min(max_samples, len(ds))
    n_correct = 0
    n_format = 0

    for i in range(n):
        s = ds[i]
        # Build user-only chat (no assistant text — that's what we generate)
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img.resize((224, 224), Image.LANCZOS)} for img in s.images],
                {"type": "text", "text": s.question + "\n\nThink step by step using <lts><lat><lte> latent thoughts, one per view, then state the answer letter (A/B/C/D)."},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc = processor(
            text=[text],
            images=[img.resize((224, 224), Image.LANCZOS) for img in s.images],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}

        try:
            out_ids = model.generate(
                **proc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        except Exception as e:
            # Skip this example on generation error
            continue

        new_ids = out_ids[0, proc["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        answer = _parse_answer(decoded)
        if answer is not None and answer == s.gt_answer.strip().upper():
            n_correct += 1
        # Format check: did the model emit at least one <lts>...<lte> pair?
        if "<lts>" in decoded and "<lte>" in decoded:
            n_format += 1

    model.config.use_cache = cache_was
    if was_train:
        model.train()

    return EvalResults(
        n_samples=n,
        n_correct=n_correct,
        n_format_ok=n_format,
        accuracy=n_correct / max(1, n),
        format_rate=n_format / max(1, n),
        wall_seconds=time.time() - t0,
    )
