# Implementation Plan — Latent Spatial Reasoning via Step-Level View-Shift Supervision

Companion to `latent_spatial_reasoning_idea.md`. This document is the execution-level plan: concrete models, data sources, losses, schedules, compute, and milestones.

> **Scope note.** View sequences are generated *deterministically* via grounded detection on the question's object mentions — no tool-calling LLM agent. The tool-calling-agent variant (typed multi-encoder distillation) lives in `tool_trace_latent_reasoning_implementation_plan.md`.

---

## 1. Scope and Success Criteria

**Primary claim to validate.** Pose-conditioned 2D latent thoughts beat (a) text CoT, (b) dense-3D latents (3DThinker-style), and (c) no-CoT baselines on VSI-Bench and 3DSRBench, at matched backbone and matched data.

**Minimum viable result.** +3 absolute points on VSI-Bench overall and non-regression on general VQA (MMMU, MM-Vet) vs. the base VLM.

**Stretch result.** Emitted latent thoughts, when decoded to nearest real views, recover a human-interpretable viewpoint trajectory that matches the trace-builder's reference view sequence ≥ 60 % of the time (trace-IoU metric defined in §8).

---

## 2. Repo Layout

```
latent_reasoning/
├── configs/                       # Hydra configs per stage
│   ├── model/                     #   vlm_backbone, encoders, projectors
│   ├── data/                      #   dataset + trace-builder configs
│   ├── train/                     #   stage1..4 optim/schedule
│   └── eval/                      #   benchmark configs
├── src/
│   ├── models/
│   │   ├── vlm.py                 # backbone wrapper + latent-token heads
│   │   ├── encoders_frozen.py     # VGGT, 2D vision, pose encoder loaders
│   │   ├── projectors.py          # 3 MLP projectors to VLM hidden dim
│   │   └── latent_head.py         # projects VLM hidden -> target latent space
│   ├── data/
│   │   ├── trace_builder/         # grounded-detection view-sequence pipeline
│   │   │   ├── mention_extract.py #   regex + spaCy NP fallback
│   │   │   ├── localize.py        #   Grounded-SAM-3D / OpenMask3D wrapper
│   │   │   └── best_view.py       #   visibility/area/centrality scoring
│   │   ├── trace_filters.py       # coverage + coherence + length filters
│   │   ├── datasets.py            # SQA3D / ScanQA / VSI-train loaders
│   │   └── collate.py             # interleaved text + <lts>/<lte> packing
│   ├── train/
│   │   ├── stage1_align.py
│   │   ├── stage2_ground.py
│   │   ├── stage3_e2e.py
│   │   ├── stage4_grpo.py
│   │   └── losses.py              # L2, InfoNCE, masked CE
│   ├── eval/
│   │   ├── benchmarks.py
│   │   ├── trace_diagnostics.py
│   │   └── ablations.py
│   └── utils/                     # logging, checkpointing, distributed
├── scripts/                       # launchers (slurm / local)
├── tests/                         # unit + smoke tests (fast, no GPU needed)
└── notebooks/                     # trace inspection, failure analysis
```

Framework choices: **PyTorch 2.5 + FSDP2**, HuggingFace Transformers for the backbone, `accelerate` for the smaller alignment runs, `verl` or `trl` for the GRPO stage, `hydra-core` for configs, `wandb` for logging.

---

## 3. Model Selection

| Component | Choice | Rationale |
|---|---|---|
| VLM backbone | **Qwen2.5-VL-7B-Instruct** | Strong multi-image support; matches numbers quoted in the idea doc so the baseline is directly comparable. |
| 3D encoder (frozen) | **VGGT** official checkpoint | Same as 3DThinker; produces per-scene geometry tokens. |
| 2D view encoder (frozen) | **SigLIP2-so400m** | Pose-sensitive-enough features; widely used, 768-d output. Fall back to DINOv2-L if SigLIP2 latents prove too texture-biased in ablations. |
| Pose encoder (frozen) | **Custom small MLP** over (R ∈ SO(3), t ∈ R³) encoded via 6-D rotation + Fourier-feature translation; initialized from a 1-epoch pre-train on ScanNet poses | Lightweight, deterministic, fully frozen after pre-train. |
| Projectors (trainable) | 2-layer MLP, GELU, residual; input-dim → 4096 | Three independent projectors: VGGT→VLM, SigLIP→VLM, Pose→VLM. |
| Latent head (trainable) | 2-layer MLP from VLM hidden → `dim(SigLIP) + dim(Pose)` | Produces the predicted latent that is matched against the frozen target. |

Sequence budget: 32k tokens. A multi-view scene takes ~6k image tokens; latent thoughts add ~1 token each between `<lts>`/`<lte>`; plenty of headroom.

Two new special tokens added to the tokenizer: `<lts>`, `<lte>`. Embedding rows initialized as the mean of 10 random existing tokens (avoids NaN on first step).

---

## 4. Data Pipeline

### 4.1 Source datasets

| Dataset | Use | Size |
|---|---|---|
| ScanNet + ScanQA | Training traces, scene reconstructions | 1.5k scenes, ~40k QA |
| SQA3D | Training traces | 33k QA |
| 3RScan + 3DSSG | Relational reasoning supervision | 1.4k scenes |
| ARKitScenes | Real-world poses for pose-encoder pretraining | 5k scenes |
| VSI-Bench (train split) | If/where a train split is released | — |
| Held-out eval: VSI-Bench val, 3DSRBench, SQA3D test, Real-3DQA | Evaluation only | — |

All scenes are stored as `(rgb_views[], camera_poses[], depth_maps?, mesh?)`. Keep everything on local NVMe; avoid NFS for the inner loop.

### 4.2 Trace builder (grounded-detection view-sequence pipeline)

For each `(question, scene, answer)` triple, produce an ordered list of `(view_i, pose_i)` pairs sufficient to answer the question. Deterministic given a frozen detector — no LLM in the loop.

**Step 1 — Object-mention extraction.**
- Primary: regex match against the union of object vocabularies from ScanNet (~200 classes) + 3DSSG relational nouns.
- Fallback for unmatched nouns: spaCy NP chunker → CLIP text-image score against scene crops, keep mentions with score ≥ 0.25.
- Output: ordered list of mentions `[o_1, ..., o_k]` in question order, deduped, capped at k = 6.

**Step 2 — 3D object localization.** For each `o_i`, run **Grounded-SAM-3D** (or OpenScene / OpenMask3D for ScanNet) over the dataset's RGB views. Per-scene per-object detection results are cached to disk and reused across all questions about that scene.

**Step 3 — Best-view selection.** For each mentioned `o_i`, score every captured view by

```
score(v, o_i) = α · visibility(o_i, v) + β · projected_area(o_i, v) + γ · centrality(o_i, v)
```

with `visibility` from depth-based occlusion checks and `centrality` penalizing edge-of-frame placements. Pick top-1 view; record its known camera pose as `pose_i`. (α, β, γ) = (0.5, 0.3, 0.2) initially; tune on a held-out 1k subset.

**Step 4 — Situation prepend (SQA3D only).** When the dataset gives an agent situation pose, insert it as `(view_0, pose_0)` so the trajectory starts from the agent's vantage.

**Step 5 — Pack as a trace.** Yield `[(view_0, pose_0)?, (view_1, pose_1), ..., (view_k, pose_k)]` along with the dataset's narration / answer text.

Pseudocode:

```
mentions = extract_mentions(question)
detections = [localize_3d(scene, m) for m in mentions]
if any(d.confidence < 0.5 for d in detections): drop
views = [argmax_v score(v, d) for d in detections]
trace = [(scene.situation_pose, scene.situation_view)] if sqa3d else []
trace += list(zip(views, [v.pose for v in views]))
```

### 4.3 Filters

- **Coverage.** Drop questions where any extracted mention fails to localize with detector confidence ≥ 0.5. This is the dominant yield bottleneck.
- **Trace coherence.** For step `i>1`, require cosine-sim between `SigLIP(view_i)` and `SigLIP(view_{i-1})` ∈ [0.3, 0.92]. Out-of-band → consecutive views are incoherent or near-duplicates.
- **Length filter.** Keep traces with 2–6 view steps. <2 is trivial; >6 over-segments and dilutes supervision.

Expected yield: **50–70 %** of questions (higher than an LLM-agent pipeline because the detector doesn't have to *solve* the question, only ground its objects). Target **~120 k kept traces** for Stage 2 — enough based on 3DThinker's sample efficiency.

### 4.4 Trace packing

Each training sample:

```
input_ids  = [sys] [query] [image tokens...] [vggt latents...] [assistant turn tokens]
assistant  = text_1 <lts> L_1 <lte> text_2 <lts> L_2 <lte> ... text_k [answer]
```

Where each `L_i` is one token slot whose label is the trace-builder-produced target
`concat(SigLIP(view_i), PoseEnc(pose_i))`. Loss mask: text tokens → CE;
`<lts>`/`<lte>` → CE (format learning); latent slot → L2 or InfoNCE (below).

---

## 5. Stage 1 — Modality Alignment (3 days, 4×A100)

**Goal:** projectors map frozen encoder outputs into the VLM's input space without wrecking the backbone.

- Trainable: three projectors + the 2 new token embeddings.
- Frozen: VLM backbone, all encoders, latent head.
- Data: standard LLaVA-style caption + VQA (500k samples) **with VGGT tokens appended** for scenes that have them. For non-3D samples, pass a learned null-VGGT embedding.
- Loss: standard next-token CE on text.
- Optim: AdamW, lr 1e-3 for projectors (they start at zero-init residual), 5e-5 for token embeddings, cosine schedule, 1 epoch.
- Exit criterion: MMMU + MM-Vet within 1 point of base VLM — projectors are not harming language capability.

---

## 6. Stage 2 — Latent Thought Grounding (7 days, 8×A100)

**Goal:** teach the VLM to emit latent tokens whose projected embedding matches `(SigLIP(view_i), PoseEnc(pose_i))`.

- Trainable: full VLM backbone + projectors + latent head + 2 new token embeddings.
- Frozen: VGGT, SigLIP2, pose encoder.
- Data: 120 k trace-builder-filtered traces + 30 k general VQA (replay to prevent catastrophic forgetting).
- Losses:
  - `L_text = CE` on text tokens including `<lts>/<lte>`.
  - `L_latent = L2(head(h_i), concat(SigLIP(view_i), PoseEnc(pose_i)))` at each latent slot.
  - Optionally add `L_infonce` with in-batch negatives (other traces' views) once L2 plateaus; weight starts at 0 and ramps to 0.3 after epoch 1.
  - Total: `L = L_text + λ · L_latent`, λ = 1.0, tuned on val.
- Optim: AdamW, lr 2e-5 backbone / 1e-4 projectors+head, cosine, 2 epochs, warmup 2%.
- Precision: bf16 mixed, FSDP2 full-shard.
- Key checks each 500 steps:
  - Teacher-forced latent cosine-sim to target > 0.6 within 500 steps; > 0.8 by end of epoch 1.
  - Text CE on general VQA replay < 1.1× Stage-1 value.

---

## 7. Stage 3 — End-to-End Reasoning (5 days, 8×A100)

**Goal:** remove the latent-token target; the model must freely choose when and where to emit latent thoughts, with only final-answer supervision.

- Trainable: same as Stage 2.
- Loss: `L_text` only, but the `<lts>` / `<lte>` format is still required (format-mismatch penalty via a small auxiliary CE on the structural tokens).
- Data: same traces, but during forward pass the latent-token embeddings are the model's *own* projected latent head outputs, not the trace-builder targets. (This is the transition from supervised to free generation; use scheduled sampling — 80 % supervised → 0 % over 1 epoch.)
- Exit criterion: accuracy on SQA3D val monotonically improving for 3 consecutive 1k-step checkpoints.

---

## 8. Stage 4 — GRPO (4 days, 8×A100)

- Framework: `verl` (preferred) or `trl`'s GRPO.
- Rollouts: 8 per prompt, temperature 0.9, top-p 0.95, max 2 latent thoughts per step allowed.
- Reward:
  - `+1` correct final answer (string match + GPT-4o judge for open-ended).
  - `+0.1` well-formed `<lts>…<lte>` structure.
  - `−0.2` emitted latent with cosine-sim < 0.1 to *any* real view of the scene (prevents latent collapse into noise).
- KL coef: 0.04, clipped at 0.2, 2 epochs.
- Exit criterion: VSI-Bench val accuracy strictly > Stage 3 by ≥1 point with no general-VQA regression > 1 point.

---

## 9. Evaluation Harness

### 9.1 Benchmarks

- **VSI-Bench** (overall + per-task breakdown: distance, counting, size, appearance order, route planning).
- **3DSRBench** (relational).
- **SQA3D** test (situated QA).
- **Real-3DQA**.
- **General VQA regression guards:** MMMU, MM-Vet, MMBench.

### 9.2 Ablations (these are the paper's load-bearing experiments)

| Ablation | What it isolates |
|---|---|
| No 3D VGGT input | Value of 3D conditioning |
| No latent supervision (Stage 2 skipped) | Value of step-level supervision |
| Dense-VGGT target in output (3DThinker-equivalent) | Value of 2D-view target over whole-scene 3D target |
| No pose conditioning (view-only target) | Value of the pose signal specifically |
| Text CoT instead of latent | Re-confirms the accuracy drop noted in the idea |
| L2 vs InfoNCE on latents | Loss-form sensitivity |
| #latent-steps ∈ {1, 2, 4, 6} | Diminishing returns |

### 9.3 Trace-level diagnostics

- **Nearest-view decoding.** For each emitted latent, retrieve the scene view with highest cosine sim under the frozen SigLIP. Report top-1 accuracy against the trace-builder's reference view (chance ≈ 1/#views).
- **Trace-IoU.** Fraction of emitted latents whose nearest real view shares ≥ 50 % of pixels (via re-projected depth) with the trace-builder's reference step view.
- **Pose error.** Angular + translational error between predicted pose and the reference pose, when the model is made to emit an explicit pose token (pose-regression variant).

---

## 10. Design Decisions I'm Pre-Committing To (vs. Open Questions in idea doc)

- **View source = real captured view** for Stages 1–3. Rendered novel views only introduced at Stage 4 as an augmentation. Reason: clean supervision first; mental rotation later.
- **Target pose = from the captured view selected by the best-view scorer**, not sampled from a prior. Reason: the whole point is to ground to *question-relevant* viewpoints, and the scorer already picks those given the mention list. Novel-view targets (rendered from a 3D reconstruction at an arbitrary pose) are deferred to v2.
- **Pose emission:** the VLM predicts a pose *implicitly* inside the latent slot — the latent head output is `[view_emb ; pose_emb]`. No separate pose-regression head in v1. Add one in v2 if pose error is too loose.
- **Loss starts as L2, adds InfoNCE at epoch 2.** L2 alone converges but can be sloppy on fine-grained viewpoint distinctions; the contrastive term sharpens it without the cold-start instability.
- **Latent-step budget = 4** on average at training time (trace-builder outputs filtered to 2–6). Ablate 1/2/4/6 at inference.

---

## 11. Compute Budget

| Stage | Days | GPUs | GPU-hours |
|---|---|---|---|
| Pose-encoder pre-train | 0.5 | 2×A100 | 24 |
| Trace builder pre-compute (Grounded-SAM-3D over ScanNet/SQA3D scenes; cached) | 1 | 2×A100 | 48 |
| Stage 1 alignment | 3 | 4×A100 | 288 |
| Stage 2 grounding | 7 | 8×A100 | 1344 |
| Stage 3 e2e | 5 | 8×A100 | 960 |
| Stage 4 GRPO | 4 | 8×A100 | 768 |
| Evaluation + ablations | 5 | 4×A100 | 480 |
| **Total** | **~25 days** | | **~3.9 k GPU-hours** |

Fits comfortably on an 8×A100 node running ~3.5 weeks, or scale out to 3 nodes for ~10 days. ~400 GPU-hours cheaper than an LLM-agent-based trace pipeline (no VLM-72B inference loop).

---

## 12. Risk Register and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Trace-builder coverage < 50 % (detector misses too many object mentions) | Medium | Lower detection threshold; expand mention vocabulary (Open-vocab DETR fallback); allow partial traces (k = 1) with downweighted loss. If still < 30 %, the question type isn't expressible as "look at object X then Y" — push those questions to the tool-trace pipeline (companion plan). |
| Trace builder picks views that don't actually answer the question (objects visible but reasoning still unsupported) | Medium-High | Per-step decoding diagnostic (§9) flags this — emitted latent matches a real view but the answer is still wrong → root-cause: the view set was insufficient. Expand `k` ceiling; add second-best views for ambiguous mentions. |
| Latent collapse (emitted latent = constant) | Medium | InfoNCE + the GRPO "cosine > 0.1 to some real view" penalty (§8). |
| SigLIP features not pose-sensitive enough | Medium | Switch to DINOv2-L or to a pose-contrastive-finetuned SigLIP2 on ScanNet multi-view pairs. |
| Format regression: model forgets `<lts>/<lte>` | Low | Aux CE on structural tokens, never dropped. |
| General VQA regression after Stage 2 | Medium | Replay 30k general VQA, monitor every 500 steps; revert if MMMU drops >2 points. |
| GRPO reward hacking (emits nonsense latents that still get the answer right) | Medium-High | Latent-real-view cosine reward gate (§8); trace-IoU monitoring. |
| Compute overrun | Medium | Stage-2 can be shortened to 1 epoch if curves plateau; Stage-4 is skippable for the v1 paper — report SFT result only. |

---

## 13. Milestones and Go/No-Go Gates

1. **M1 (week 1):** Trace builder achieves ≥ 50 % kept-trace rate on 5 k ScanQA + SQA3D questions; per-scene detection cache built for ScanNet train. *Gate: if < 30 %, expand mention vocabulary or relax detection threshold; if still < 30 %, the view-shift formulation is mismatched to these questions and we cap scope to the subset it does cover.*
2. **M2 (week 2):** Stage 1 projectors trained; MMMU regression ≤1 pt.
3. **M3 (week 4):** Stage 2 supervised latent cosine-sim ≥ 0.8 against the trace-builder targets; SQA3D val ≥ base VLM + 2 pts.
4. **M4 (week 5):** Stage 3 e2e on VSI-Bench val ≥ base + 2 pts.
5. **M5 (week 6):** Stage 4 GRPO + all ablations complete. *Paper-ready.*

Gates are hard — if M3 doesn't clear, we don't proceed to M4; we redesign Stage 2 (loss form, encoder choice, trace filters).

---

## 14. First Two Weeks — Concrete Next Actions

1. Scaffold repo (§2), land CI with a CPU-only smoke test that loads configs + runs a 1-step forward on a fake batch.
2. Implement the trace builder (§4.2): mention-extractor (regex over ScanNet vocab + spaCy fallback), Grounded-SAM-3D wrapper, best-view scorer. Goal: 1 k traces over ScanQA, eyeball-inspect 20, iterate on coverage.
3. Pre-train the pose encoder (§3) — small, should land in half a day.
4. Scale trace builder to full ScanQA + SQA3D + 3RScan; pre-compute and cache per-scene per-object detections. Target **120 k traces by end of week 1** (deterministic pipeline is fast).
5. Implement trace packer + collate and land a unit test that round-trips a packed sample to loss without exceptions.
6. Kick off Stage 1 on 4×A100 in parallel; remaining GPUs run trace builder.
7. **Land the "random in-scene views" baseline ablation** (recommended in the conversation that produced this doc): replace trace-builder targets with k random scene views per question, run mini Stage 2 on 5 k samples for 2 k steps, compare per-step latent cosine. If random ≥ trace-builder, the supervision pipeline is overkill and worth simplifying further.
