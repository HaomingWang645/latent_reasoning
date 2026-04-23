# Implementation Plan — Tool-Trace Latent Reasoning via Typed Multi-Encoder Distillation

Companion to `tool_trace_latent_reasoning_idea.md` (concept) and `latent_spatial_reasoning_implementation_plan.md` (the parallel view-shift plan, with which this one shares ~60 % of infrastructure). This document is the execution-level plan: concrete encoders, projectors, losses, schedules, compute, milestones, and the head-to-head ablations against CoVT / Pearl / Think3D that the novelty story rests on.

---

## 1. Scope and Success Criteria

**Primary claim to validate.** Typed multi-encoder latent distillation from a 3D-tool-calling teacher beats (a) parallel multi-encoder distillation on the input image (CoVT-style), (b) trajectory-level distillation of the same trace (Pearl-style), (c) text-CoT distillation of the same trace (Think3D-style), and (d) the view-shift-only single-encoder-pair variant (companion idea), at matched backbone, matched teacher trace pool, matched compute.

**Minimum viable result.** +3 absolute points on VSI-Bench overall vs. base VLM **and** ≥ +1 point vs. the strongest of (a)/(b)/(c)/(d). Non-regression > 1 pt on MMMU / MM-Vet.

**Stretch result.** Per-tool-decoded latents (nearest-neighbor in the frozen encoder space) match the teacher's tool output ≥ 60 % of the time across the four most-used tool types (`reconstruct`, `detect`, `orient`, `bev_sketch`).

---

## 2. Repo Layout

Builds on the same scaffold as the view-shift plan; new modules called out:

```
latent_reasoning/
├── configs/
│   ├── model/
│   │   └── tool_trace.yaml            # encoder zoo + per-type heads
│   ├── data/
│   │   └── teacher_agent_full.yaml    # 8-tool toolbox config
│   └── train/{stage1..4}_tool_trace.yaml
├── src/
│   ├── models/
│   │   ├── encoders_frozen.py         # +PointTransformer, SAM-feat, BEVConv, OrientMLP
│   │   ├── projectors.py              # 8 projectors, registered by tool type
│   │   ├── latent_heads.py            # 8 heads, dispatched on emitted type token
│   │   └── routing.py                 # type-token → (head, projector) dispatch
│   ├── data/
│   │   ├── teacher_agent/
│   │   │   ├── tools/                 # one module per tool
│   │   │   ├── trace_runner.py
│   │   │   └── trace_schema.py        # dataclass for typed trace steps
│   │   ├── trace_filters.py           # coherence (per-type) + validity + length
│   │   └── collate.py                 # typed-token packing
│   ├── train/
│   │   ├── stage1_align_tool_trace.py
│   │   ├── stage2_ground_tool_trace.py
│   │   ├── stage3_e2e_tool_trace.py
│   │   ├── stage4_grpo_tool_trace.py
│   │   └── losses.py                  # per-type L2 + InfoNCE; type-CE
│   └── eval/
│       ├── benchmarks.py              # shared with view-shift
│       ├── head_to_head.py            # CoVT-parallel, Pearl-traj, Think3D-text, view-shift baselines
│       └── trace_diagnostics.py       # per-tool decoding, trace-alignment, type entropy
├── scripts/
└── tests/
```

Same framework choices as the view-shift plan: PyTorch 2.5 + FSDP2, HF Transformers backbone, `verl` for GRPO, `hydra-core`, `wandb`.

---

## 3. Model Selection

### 3.1 Backbone

Identical to the view-shift plan: **Qwen2.5-VL-7B-Instruct**, 32 k context, bf16. Same rationale: matched-baseline numbers in the idea doc.

### 3.2 Encoder Zoo (all frozen after their pretrain)

| Tool name | Output type | Frozen encoder | Output dim | Pretrain notes |
|---|---|---|---|---|
| `reconstruct_scene` | dense point cloud | **PointTransformer-V3** (PTv3) on the agent's reconstructed PC | 512 (mean-pooled) | Use public PTv3 ScanNet checkpoint. |
| `detect` | (box, crop) | **SigLIP2-so400m** on the crop ⊕ Fourier features over normalized box | 768 + 64 | Concatenated. |
| `segment` | binary mask | **SAM ViT-H** feature pooled over the mask region | 256 | Use SAM's pre-`mask_decoder` features. |
| `orient` | yaw or 6-D-rot | **OrientMLP** — 6-D-rot continuous rep + axis Fourier features → 2-layer MLP | 128 | Pretrain 1 epoch on ScanNet object orientations. |
| `measure` | scalars | Fourier features over k normalized scalars | 64 | k = 4 channels (distance, angle, area, count). |
| `mental_rotate` | (view, pose) | **SigLIP2-so400m** + **PoseEnc** (shared with view-shift idea) | 768 + 128 | Re-use the view-shift code path verbatim. |
| `bev_sketch` | top-down rasterization | **BEVConv** — small ConvNet (8 layers) on the rasterized BEV image | 256 | Pretrain 1 epoch on auto-rendered BEV from ScanNet. |
| `python` | scalar / string | — (emit as plain text, no latent token) | — | No encoder. |

### 3.3 Trainable Modules

- 8 input-side projectors (2-layer MLPs, GELU, residual; per-type input-dim → 4096).
- 8 latent heads (2-layer MLPs from 4096 → per-type latent dim).
- One `<lts>` token, one `<lte>` token, one `<tool_T>` token per tool type (+ 8 = 11 new tokens). Init = mean of 10 random tokens.

Total new trainable parameters: ~38 M (projectors) + ~28 M (heads) + ~45 K (tokens) ≈ 66 M, on top of the 7 B backbone.

---

## 4. Data Pipeline

### 4.1 Source datasets

Identical to view-shift plan: ScanNet/ScanQA, SQA3D, 3RScan/3DSSG, ARKitScenes (pose-encoder pretrain), VSI-train. All scenes stored as `(rgb_views[], camera_poses[], depth_maps?, mesh?)` on local NVMe.

### 4.2 Teacher agent — 8-tool version

Driver: Qwen2.5-VL-72B (GPT-4o backup). Tool wiring extends the view-shift plan's 6-tool version to all 8:

```
tools = {
  reconstruct_scene(scene_id) -> PointCloud
  detect(view, classes) -> [(box, crop)]
  segment(view, box) -> mask
  orient(view, mask) -> yaw_or_6Drot
  measure(view_a, view_b, mask_a, mask_b) -> {distance, angle, ...}
  mental_rotate(scene_id, pose_delta) -> (rendered_view, pose)
  bev_sketch(scene_id, objects[]) -> bev_image
  python(code) -> scalar_or_string
}
```

Trace step schema (canonical JSON):

```json
{
  "step_idx": 2,
  "tool": "orient",
  "args": {"view": "v3", "mask": "m_chair"},
  "raw_output": <serialized tool-specific output>,
  "encoded_target_dim": 128,
  "narration_text": "I check the chair's facing direction ..."
}
```

### 4.3 Filters (load-bearing — promoted to a named contribution)

Three filters, each ablated separately in §9:

- **Per-type coherence filter** (`trace_filters.coherence`):
  - `reconstruct`: at most one per trace; if more, must be the same scene.
  - `detect` consecutive same-class: box IoU < 0.1 disallowed (degenerate re-detection).
  - `orient` consecutive same-object: angular delta consistent with the narration ("rotated about 90°" → measured Δ ∈ [60°, 120°]); enforced via a regex + tolerance check.
  - `bev_sketch` consecutive: BEV-encoder cosine ∈ [0.3, 0.92].
  - `mental_rotate` consecutive: SigLIP cosine ∈ [0.3, 0.92] (same as view-shift).
  - `segment` mask-area sanity: 50 < area < 0.7·image_area.
- **Validity filter:** final answer matches dataset GT (exact match, or strict GPT-4o judge for open-ended).
- **Length filter:** 2 ≤ #tool steps ≤ 6.

Expected yield: 30–40 %. Target ~120 k kept traces.

### 4.4 Trace packing

```
input_ids = [sys] [query] [image_tokens...] [vggt_tokens...] [assistant_turn_tokens]
assistant = text_1 <lts type=T1> L_1 <lte> text_2 <lts type=T2> L_2 <lte> ... text_k [answer]
```

Loss masks (precomputed in collate):
- text → CE
- `<lts>`, `<lte>`, `<tool_T>` (the type token) → CE (format learning)
- latent slot of type T → L2 (or L2 + InfoNCE) against `Enc_T(tool_out)`
- `python` step narration → CE on text, no latent slot

---

## 5. Stage 1 — Modality Alignment (3 days, 4 × A100)

**Goal:** all 8 input-side projectors map their respective tool outputs into the VLM's input space without breaking language capability.

- Trainable: 8 projectors + 11 new token embeddings.
- Frozen: VLM backbone, all 8 encoders, 8 latent heads.
- Data: 500 k LLaVA-style caption + VQA, with tool outputs *injected as input* whenever the scene supports them. For modalities with no natural caption pretraining set, use synthetic injection (e.g., emit a known PointCloud + ask the VLM to answer a templated question about it).
- Loss: standard next-token CE on text.
- Optim: AdamW; lr 1 e-3 projectors (zero-init residual), 5 e-5 token embeddings; cosine; 1 epoch.
- Exit criterion: MMMU + MM-Vet within 1 pt of base VLM.

---

## 6. Stage 2 — Latent Thought Grounding (8 days, 8 × A100)

**Goal:** the VLM emits a `<tool_T>` type token followed by a latent token whose latent-head output matches the frozen-encoder target for that tool's output.

- Trainable: full VLM backbone + 8 projectors + 8 latent heads + 11 new tokens.
- Frozen: all 8 encoders.
- Data: 120 k filtered teacher traces + 30 k general VQA replay.
- Losses (per type, dispatched on the emitted `<tool_T>` token):
  - `L_text = CE` on text and on `<lts>/<lte>/<tool_T>` tokens.
  - `L_latent = Σ_i L2(head_T(h_i), Enc_T(tool_out_i))` summed over all latent slots.
  - `L_infonce` (per type) added at epoch 2; weight ramped 0 → 0.3 over 5 k steps. Negatives: in-batch other-trace outputs of the *same* type.
  - Total: `L = L_text + λ · L_latent + γ · L_infonce`, λ = 1.0, γ = 0.3.
- Optim: AdamW; lr 2 e-5 backbone, 1 e-4 projectors+heads; cosine; 2 epochs; warmup 2 %.
- Precision: bf16 mixed, FSDP2 full-shard.
- Per-500-step health checks:
  - Per-tool teacher-forced cosine to target > 0.6 by step 1 k; > 0.8 by end of epoch 1.
  - Per-tool latent loss curves visible separately on wandb (one panel per type).
  - Text CE on general VQA replay < 1.1× Stage-1.
  - **Type-prediction accuracy:** the predicted `<tool_T>` token matches the teacher's tool choice ≥ 0.7 by end of epoch 1.

---

## 7. Stage 3 — End-to-End Reasoning (5 days, 8 × A100)

- Trainable: same as Stage 2.
- Loss: `L_text` only, but format CE on `<lts>/<lte>/<tool_T>` is preserved (anti-format-collapse).
- During forward, latent-token embeddings are the model's *own* latent-head outputs; scheduled sampling teacher-forced → free over 1 epoch (80 % → 0 %).
- Exit criterion: SQA3D val monotonically improving for 3 consecutive 1 k-step checkpoints.

---

## 8. Stage 4 — GRPO (4 days, 8 × A100)

- Framework: `verl`.
- Rollouts: 8 / prompt, T = 0.9, top-p = 0.95, max 2 latent tokens per tool step, max 6 tool steps total.
- Reward:
  - `+1` correct final answer (string match + GPT-4o judge for open-ended).
  - `+0.1` well-formed `<tool_T>` `<lts>` … `<lte>` block.
  - `+0.05` per emitted tool whose type matches *some* tool in the teacher's trace for the same query (light supervision; not per-step alignment).
  - `−0.2` emitted latent whose cosine to *any* real tool output of the chosen type in the scene < 0.1 (anti-collapse, per-type version of the view-shift gate).
- KL coef 0.04, clipped 0.2, 2 epochs.
- Exit criterion: VSI-Bench val accuracy strictly > Stage 3 by ≥ 1 pt with no general-VQA regression > 1 pt.

---

## 9. Evaluation Harness

### 9.1 Benchmarks (shared with view-shift plan)

VSI-Bench (overall + per-task), 3DSRBench, SQA3D test, Real-3DQA, plus general-VQA regression guards (MMMU, MM-Vet, MMBench).

### 9.2 Head-to-Head Baselines (the load-bearing experiments)

These are the experiments the novelty story rests on. Each uses **the exact same 120 k teacher traces and the same backbone** to isolate one design axis at a time.

| # | Baseline | Design axis isolated | Closest prior work |
|---|---|---|---|
| H1 | 8 frozen encoders, but run **in parallel on the input image** (no agent chain) | Sequential chaining vs parallel extraction | CoVT (2511.19418) |
| H2 | Same teacher trace, single **JEPA-style encoder over the whole trajectory** | Per-step decoding vs trajectory embedding | Pearl (2604.08065) |
| H3 | Same teacher trace, **text-only** distillation (concat narration steps as CoT) | Latent-space vs text-CoT distillation | Think3D (2601.13029), SpaceTools (2512.04069) |
| H4 | Same pipeline, only `(view, pose)` latents, single shared encoder pair | Typed multi-encoder zoo vs single encoder pair | Companion view-shift idea |
| H5 | Same pipeline, all tools share **one** projector + one head | Tool-typing itself | — |
| H6 | Same pipeline, only the **4 highest-yield tools** (reconstruct, detect, orient, bev_sketch) | Encoder-zoo size | — |

H1–H4 are the must-win comparisons; H5–H6 are sensitivity studies.

### 9.3 Ablations

- Drop one tool at a time (one of: reconstruct / detect / orient / bev_sketch).
- L2 only vs L2 + InfoNCE (per-type).
- Trace-filter ablation: drop coherence; drop validity; drop both.
- # tool steps ∈ {1, 2, 4, 6} at inference.
- Hard typing (discrete `<tool_T>`) vs soft typing (mixture over heads).

### 9.4 Trace-Level Diagnostics

- **Per-tool decoding accuracy.** For each emitted latent of type T, find the nearest neighbor in the set of all real tool-T outputs across the scene; report top-1 match rate against the teacher's actual output. Reported per type.
- **Trace alignment.** Step-wise: fraction of emitted `<tool_T>` tokens matching the teacher's tool choice at the same step.
- **Tool-choice entropy** over training (per epoch), to detect type collapse.
- **Anti-collapse rate.** Fraction of emitted latents with cosine to *any* real tool-T output ≥ 0.1.
- **Per-type loss curves** in wandb (eight panels), kept publicly visible — important for reviewers.

---

## 10. Design Decisions Pre-Committed

- **Hard tool typing.** Discrete `<tool_T>` token before each latent slot routes the latent head and projector. Avoids the soft-mixture failure mode where the model collapses onto one type. Soft typing is an ablation, not the default.
- **Eight tools, not more.** Adding a ninth (e.g., distance-field) doubles engineering and gives diminishing returns; v2 territory.
- **Python steps stay text.** No "scalar latent" in v1. If GRPO discovers it needs one, add in v2.
- **Latent dim variable per type.** No forced common dim; each head outputs its target's natural dim. Simpler than learning a shared projection.
- **Trace filters are explicit code, not LLM-judged.** Every coherence rule is a deterministic function over trace fields. Reviewers should be able to read the filter source in 100 lines.
- **Reuse view-shift infrastructure** for the `mental_rotate` tool. The two ideas share the SigLIP+PoseEnc code path verbatim — single source of truth.

---

## 11. Compute Budget

| Stage | Days | GPUs | GPU-hours |
|---|---|---|---|
| Pre-train: PoseEnc, OrientMLP, BEVConv (parallel) | 0.5 | 4 × A100 | 48 |
| Teacher trace generation (8-tool toolbox) | 6 | 4 × A100 (72B judge) | 576 |
| Stage 1 alignment (8 projectors) | 3 | 4 × A100 | 288 |
| Stage 2 grounding | 8 | 8 × A100 | 1536 |
| Stage 3 e2e | 5 | 8 × A100 | 960 |
| Stage 4 GRPO | 4 | 8 × A100 | 768 |
| **Head-to-head baselines (H1–H6)** | 8 | 8 × A100 | 1536 |
| Ablations + diagnostics | 5 | 4 × A100 | 480 |
| **Total** | **~40 days** | | **~6.2 k GPU-hours** |

About 1.5 × the view-shift plan, almost entirely from (a) the eight head-to-head baselines and (b) larger Stage 2 due to per-type loss heads. Fits on an 8 × A100 node in ~5 weeks, or 3 nodes in ~2 weeks.

---

## 12. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Per-type heads collapse to a few dominant types | Medium-High | Type-CE supervision (Stage 2) + tool-choice-entropy monitoring + GRPO type-match reward (§ 8). |
| Encoder zoo too heterogeneous → unstable Stage 2 | Medium | Per-type loss panels in wandb; per-type lr if needed; revert to L2-only if InfoNCE destabilizes. |
| `bev_sketch` encoder pretrain is weak (no real pretraining corpus) | Medium | Synthesize ScanNet BEVs at 0.5 m/px; if features are too noisy, skip BEV in v1 and keep it in v2. |
| H1 (parallel CoVT-style) wins → core novelty argument breaks | **Critical** | This is the load-bearing comparison. If H1 ≥ ours, paper pivots to "agent-chained beats parallel" framing or merges with view-shift idea. Plan for redesign at M3 gate. |
| H4 (view-shift only) wins → typed multi-encoder zoo is unnecessary | High | If H4 ≥ ours, the view-shift paper is the cleaner story; ship that one and demote this to ablation in the same paper. |
| Teacher yield < 30 % | Medium | Same as view-shift plan: template-synthesize the rule-expressible tools (`measure`, `orient`) directly from ScanNet metadata. |
| Engineering surface (8 encoders × 4 stages) overruns timeline | Medium-High | Cut to H6 (4-encoder zoo) early if Stage-2 step time is > 4 s/it. |
| Teacher / VLM judge introduces label leakage from VLM family | Low | Use GPT-4o for judging (different family from Qwen backbone). |

---

## 13. Milestones and Go/No-Go Gates

1. **M1 (week 1.5):** All 8 tools wired. Teacher agent yields ≥ 30 % kept traces over 5 k queries with all three filters on. *Gate: if < 15 %, drop to 4 tools (H6) and proceed.*
2. **M2 (week 3):** Stage 1 projectors trained for all 8 types; MMMU regression ≤ 1 pt; per-type input injection works (verified via templated VQA).
3. **M3 (week 5):** Stage 2 per-type teacher-forced cosine ≥ 0.8 for ≥ 6 of 8 types; type-prediction accuracy ≥ 0.7; SQA3D val ≥ base VLM + 2 pts. **Critical gate:** if < 5 of 8 types reach 0.8, redesign that type's encoder (likely BEV or segment).
4. **M4 (week 6.5):** Stage 3 e2e on VSI-Bench val ≥ base + 2 pts.
5. **M5 (week 8):** Stage 4 GRPO + H1–H4 head-to-heads complete. **Critical gate:** ours ≥ best-of-{H1, H2, H3, H4} by ≥ 1 pt on VSI-Bench. If not, the paper pivots — see Risk Register.
6. **M6 (week 9):** H5, H6, all ablations, diagnostics. *Paper-ready.*

---

## 14. First Two Weeks — Concrete Next Actions

1. Scaffold tool-trace branch; CI smoke test loads each of the 8 encoders + projectors and runs a 1-step forward on a fake batch of all 8 types.
2. Implement and unit-test the `trace_filters` module (deterministic; full coverage on synthetic traces).
3. Pretrain OrientMLP and BEVConv (1 epoch each on ScanNet; ~½ day).
4. Implement the teacher agent over `reconstruct`, `detect`, `orient`, `bev_sketch` (the 4-tool starter set = H6 baseline). Generate 100 traces, eyeball-inspect 20.
5. Extend teacher to 8 tools; parallelize across 4 GPUs; target 120 k traces by end of week 2.
6. Implement trace packer + collate handling typed latent slots; round-trip unit test (pack → forward → loss without exceptions).
7. Kick off Stage 1 on 4 × A100 in parallel with trace generation on the remaining GPUs.
8. **Decide branch strategy with the view-shift plan:** the two ideas share the `mental_rotate` tool path. Recommended — develop tool-trace as `feature/tool-trace`, periodically rebase view-shift's view+pose code into the shared `src/models/encoders_frozen.py`.
