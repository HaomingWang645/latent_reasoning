"""MMSI-Bench evaluation harness — out-of-distribution multi-view spatial QA.

MMSI-Bench (Multi-image Spatial Intelligence Benchmark) — Runsen Xu et al., 2025.
1000 questions, 2+ images per question, multiple choice A-D on questions like
motion prediction, object relations, camera pose shifts.

Local parquet at:
    /mnt/data1/workspace/jovyan-huggingface-cache/huggingface/hub/
    datasets--RunsenXu--MMSI-Bench/snapshots/<hash>/MMSI_Bench.parquet

Columns: id, images (ndarray of bytes), question_type, question, answer, thought

This harness is called from stage_trainer.run_post_stage_eval() and writes
results to the shared TRAINING_LOG.md.
"""
from __future__ import annotations

import glob
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


_ANSWER_RE = re.compile(r"\b([A-D])\b")


def _parse_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text or "")
    return m.group(1) if m else None


def _locate_parquet() -> Path | None:
    root = ("/mnt/data1/workspace/jovyan-huggingface-cache/huggingface/hub/"
            "datasets--RunsenXu--MMSI-Bench/snapshots")
    files = glob.glob(f"{root}/*/MMSI_Bench.parquet")
    return Path(files[0]) if files else None


@dataclass
class MMSIResults:
    n_samples: int
    n_correct: int
    n_format_ok: int
    accuracy: float
    format_rate: float
    wall_seconds: float
    per_type_accuracy: dict


def _pil_from_cell(cell) -> Image.Image | None:
    """MMSI-Bench images column is a list of dicts: {"bytes": b"...", "path": "..."} or raw bytes."""
    if cell is None:
        return None
    if isinstance(cell, dict):
        b = cell.get("bytes")
    elif isinstance(cell, (bytes, bytearray)):
        b = bytes(cell)
    elif isinstance(cell, str):
        try:
            return Image.open(cell).convert("RGB")
        except Exception:
            return None
    else:
        return None
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None


@torch.no_grad()
def run_mmsi_eval(
    handles,
    parquet_path: str | Path | None = None,
    max_samples: int = 500,
    max_new_tokens: int = 64,
    max_views: int = 4,
) -> MMSIResults:
    """Evaluate on MMSI-Bench. Returns accuracy + per-question-type breakdown."""
    t0 = time.time()
    if parquet_path is None:
        parquet_path = _locate_parquet()
    if parquet_path is None or not Path(parquet_path).exists():
        raise FileNotFoundError("MMSI-Bench parquet not found")

    df = pd.read_parquet(parquet_path)
    df = df.head(max_samples)

    model = handles.model
    processor = handles.processor
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    was_train = model.training
    model.eval()
    cache_was = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    n_correct = 0; n_format = 0
    per_type_c = {}; per_type_n = {}

    for idx, row in df.iterrows():
        imgs_raw = row["images"]
        if hasattr(imgs_raw, "tolist"):
            imgs_raw = imgs_raw.tolist()
        imgs = [x for x in (_pil_from_cell(c) for c in imgs_raw[:max_views]) if x is not None]
        if len(imgs) < 2:
            continue
        gt = str(row["answer"]).strip().upper()
        qtype = row.get("question_type", "unknown")
        per_type_n[qtype] = per_type_n.get(qtype, 0) + 1

        imgs_resized = [im.resize((224, 224), Image.LANCZOS) for im in imgs]
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": im} for im in imgs_resized],
                {"type": "text",
                 "text": row["question"] + "\n\nThink step by step using <lts><lat><lte> latent thoughts, one per view, then state the answer letter (A/B/C/D)."},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc = processor(
            text=[text], images=imgs_resized,
            return_tensors="pt", padding=True, truncation=False,
        )
        proc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in proc.items()}

        try:
            out = model.generate(
                **proc, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        except Exception:
            continue

        decoded = tokenizer.decode(out[0, proc["input_ids"].shape[1]:], skip_special_tokens=False)
        ans = _parse_answer(decoded)
        if ans is not None and ans == gt:
            n_correct += 1
            per_type_c[qtype] = per_type_c.get(qtype, 0) + 1
        if "<lts>" in decoded and "<lte>" in decoded:
            n_format += 1

    model.config.use_cache = cache_was
    if was_train:
        model.train()

    per_type_acc = {qt: per_type_c.get(qt, 0) / n for qt, n in per_type_n.items()}
    total = sum(per_type_n.values())
    return MMSIResults(
        n_samples=total,
        n_correct=n_correct,
        n_format_ok=n_format,
        accuracy=n_correct / max(1, total),
        format_rate=n_format / max(1, total),
        wall_seconds=time.time() - t0,
        per_type_accuracy=per_type_acc,
    )
