# Latent Spatial Reasoning — Stage 2 Pilot Codebase

Implements the **view-shift latent grounding** stage from
[`latent_spatial_reasoning_implementation_plan.md`](../latent_spatial_reasoning_implementation_plan.md).

## Status

This is a **pilot-stage scaffold + first run** — not the full multi-week pipeline.
What's actually implemented and runnable:

| Piece | Status |
|---|---|
| Qwen2.5-VL-7B backbone + `<lts>`/`<lat>`/`<lte>` tokens + LoRA | implemented, smoke-tested |
| Frozen SigLIP-base target encoder | implemented |
| `ViewIdxPoseEncoder` — sinusoidal stand-in for true SE(3) PoseEnc | implemented |
| MindCube data loader (~10 k multi-view spatial-reasoning examples) | implemented |
| Trace-builder via grounded detection | NOT implemented (needs ScanNet); MindCube uses the dataset's own multi-view ordering as the trace |
| 3-GPU DDP trainer with grad-accum, ckpt save+resume, LR schedule | implemented |
| Stage 2 latent loss (L2 + optional InfoNCE) | implemented |
| Stage 1 / Stage 3 / Stage 4 (GRPO) | NOT implemented |

## Repo layout

```
code/
├── configs/stage2_pilot.yaml      # all knobs
├── src/
│   ├── models/
│   │   ├── encoders.py            # SigLIP + ViewIdxPoseEncoder + TargetBuilder
│   │   └── vlm_wrapper.py         # Qwen2.5-VL + LatentHead + LoRA + new tokens
│   ├── data/
│   │   ├── mindcube.py            # MindCube JSONL loader
│   │   └── collate.py             # <lts><lat><lte> packing into chat template
│   ├── train/
│   │   └── stage2_ground.py       # DDP trainer (entry point)
│   └── utils/
│       ├── checkpoint.py          # save/resume to /mnt/data3, prune old
│       └── logging.py             # JSONL + optional wandb
├── scripts/
│   ├── launch_stage2.sh           # 3-GPU launcher (devices 2,3,5; PCI_BUS_ID order)
│   ├── single_gpu_smoke.sh        # 5-step single-GPU smoke
│   └── inspect_run.py             # pretty-print metrics + ckpt sizes
```

## Storage layout (deliberately off /home)

```
/mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs/
├── ckpts/<run_name>/
│   ├── step_00000250/  step_00000500/  ...   # only keep_last_n=2 retained
│   ├── latest -> step_<latest>
│   └── ...
├── logs/<run_name>/
│   ├── metrics.jsonl            # one line per log_every_n_steps
│   ├── stdout.log               # human-readable trainer prints
│   └── config.json              # frozen config used for this run
└── hf_cache/siglip-base-patch16-224/   # frozen target encoder weights
```

`/home` (88 % full) is never written to. `/mnt/data3` has ~1 TB free; checkpoints
are pruned to `keep_last_n=2` automatically. Each ckpt is the LoRA adapter +
new-token embeddings + latent head + optimizer state — about 4–5 GB each.

## GPU situation (as of 2026-04-23)

Probed with `CUDA_DEVICE_ORDER=PCI_BUS_ID`:

| nvidia-smi index | type | available? |
|---|---|---|
| 0, 1 | H100 NVL (Exclusive_Process) | held by MSPflow |
| 2, 3 | H100 PCIe 80GB | **yes** (used) |
| 4 | H100 NVL 94GB (Exclusive_Process) | held by `ruy45` (mrl env) |
| 5 | H100 NVL 94GB (Exclusive_Process) | **yes** (used) |
| 6, 7 | H100 NVL 94GB (Default) | held by MSPflow at 95–98 % util |

Effective: **3 GPUs (2 PCIe + 1 NVL)** instead of the 4 the user requested. The
launcher sets `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,5` so device
indices match `nvidia-smi`. If GPU 4 frees up, edit `scripts/launch_stage2.sh`.

## Run

```bash
cd /home/haoming/latent_reasoning/code

# 5-step smoke (~30 sec)
bash scripts/single_gpu_smoke.sh

# Full 1500-step pilot (~30 min on 3 GPUs)
bash scripts/launch_stage2.sh

# Resume from latest checkpoint
bash scripts/launch_stage2.sh --resume

# Inspect metrics
python scripts/inspect_run.py
```

## What the pilot proves (and doesn't)

**Proves:**
- The latent-head architecture is correct: `LatentHead(hidden) → R^832` learns to track the SigLIP+pose target via L2.
- Tokenizer extension + new-token init works (`<lts>`/`<lat>`/`<lte>`).
- Multi-GPU DDP + LoRA + gradient checkpointing fits on heterogeneous 80–94 GB H100s.
- Checkpoint mgr writes to `/mnt/data3`, prunes old, supports `--resume`.

**Does NOT prove:**
- The trace-builder (grounded-detection over ScanNet) works — MindCube's
  pre-existing multi-view ordering stands in. Needs ScanNet+SQA3D to test the
  real pipeline.
- Pose grounding works in 6-DoF — `ViewIdxPoseEncoder` is sinusoidal over view
  index, not real `PoseEnc(SE(3))`.
- VSI-Bench / 3DSRBench / SQA3D accuracy gains — eval harness not wired yet.

## Resume / extend recipe

1. **Pick up after pause** — just rerun `bash scripts/launch_stage2.sh --resume`. State (model, LoRA, latent head, optimizer, RNG, step) is restored from the `latest` symlink.
2. **Swap in real ScanNet/SQA3D data** — implement `src/data/trace_builder/` per `latent_spatial_reasoning_implementation_plan.md` § 4.2; replace `MindCubeDataset` in the trainer.
3. **Swap PoseEnc** — implement a real SE(3) → R^d encoder (6-D-rot + Fourier translation) in `src/models/encoders.py`; update `view_idx_pose_dim` in the config to its output dim.
4. **Stage 3 (end-to-end)** — copy `src/train/stage2_ground.py` to `stage3_e2e.py`; drop the latent loss (`loss_lambda_latent: 0`); add scheduled-sampling logic.
5. **Stage 4 (GRPO)** — separate trainer using `verl` or `trl`, loading the Stage-3 ckpt as the policy.

## Known caveats

- **Pilot teacher-forcing is simplified.** The `<lat>` slot's input embedding is the model's *learned* `<lat>` token embedding (not the projected target). This skips one term of the canonical formulation but lets the latent head learn from the true signal at `<lts>` positions. Revisit before Stage 3.
- **No real teacher agent.** Per the conversation that produced the plan, the view-shift idea uses *grounded-detection* trace generation, not a tool-calling LLM agent. The MindCube pilot bypasses both — each example's K dataset views become the K-step trace in dataset order. This is enough to test the *latent-grounding* mechanism, not the *trace selection* mechanism.
- **Transformers 5.1.0** in the `vlm-ex` env is bleeding-edge. If something breaks, fall back to the `madpose` env (transformers 4.57.3) — the Qwen2.5-VL API is identical.
