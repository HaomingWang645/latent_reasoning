# Tool-Trace Latent Reasoning via Typed Multi-Encoder Distillation

Companion to `latent_spatial_reasoning_idea.md` (view-shift supervision). Both ideas distill a 3D-tool-calling teacher agent's trace into the VLM's latent space, but they pick different latent targets:

| Idea | Per-step latent target | Encoder stack |
|---|---|---|
| View-shift (companion) | `concat(SigLIP(view_i), PoseEnc(pose_i))` | One pair for all steps |
| **This doc — tool-trace** | Encoded tool output of step *i*, typed by tool | **Eight encoders, one per tool type** |

The two ideas are orthogonal and could be fused (§ 11).

---

## 1. Motivation

Step-level latent supervision in VLMs currently comes in three shapes:

1. **Single-encoder targets** (Mirage, Latent Sketchpad, LVR, LaViT). Every step is supervised against the same frozen 2D image encoder. The latent space is homogeneous — fine for "imagine one more image," blind to the idea that different reasoning steps yield different *kinds* of geometric evidence.
2. **Parallel multi-encoder targets** (CoVT). Multiple frozen perception experts (DINOv2, DepthAnythingV2, SAM, edges) run *in parallel on the input image*; each becomes its own continuous-token stream. Multi-modal, but not sequential — no notion of "first reconstruct, then orient, then sketch."
3. **Trajectory-level tool traces** (Pearl). A single JEPA-style encoder compresses the whole tool-use trajectory into one embedding. Captures the chain, loses per-step structure.

Separately, text-CoT distillation from 3D-tool-calling agents (SpaceTools, Think3D) keeps the agent chain but throws away the rich intermediate artifacts — the VLM has to re-derive "there is a point cloud shaped like this room" from a textual description of it.

Human 3D reasoning is not any of these. It is a *sequence* of heterogeneous operations — reconstruct → detect → orient → sketch on a mental whiteboard — each emitting a different kind of geometric object. This proposal is the latent analogue of that workflow.

---

## 2. Core Idea

Each step in a teacher-agent trace produces a tool output of some *type* — a point cloud, a bounding box, a mask, an orientation, a scalar measurement, a BEV sketch, a rendered view. Attach one frozen encoder per type. At training time, the VLM autoregressively emits a **typed latent span**:

```
<lts type=reconstruct> L_scene  <lte>   target = PointCloudEnc(tool_out_1)
<lts type=detect>      L_boxes  <lte>   target = RegionEnc(tool_out_2)
<lts type=orient>      L_yaw    <lte>   target = OrientEnc(tool_out_3)
<lts type=bev_sketch>  L_bev    <lte>   target = BEVEnc(tool_out_4)
...
```

Tool type is a discrete token the VLM emits first; it routes which latent head and which projector are used for that slot. The reasoning trace becomes a heterogeneous, typed, autoregressive chain — an explicit latent mirror of a 3D agent's tool calls.

---

## 3. Architecture

**Input to VLM:**
- `[query tokens]`
- `[image tokens]` — raw multi-view RGB
- `[VGGT scene tokens]` — optional, for whole-scene context (same as view-shift idea)

**Output from VLM:** interleaved `[text]` + typed latent spans.

**Encoder zoo (all frozen after their own short pretrain):**

| Tool | Output modality | Frozen encoder | Latent dim |
|---|---|---|---|
| `reconstruct_scene` | Point cloud | PointTransformer-v3 (or VGGT pooled) | 512 |
| `detect` | Box + crop | SigLIP2 on crop + box Fourier features | 768+64 |
| `segment` | Binary mask | SAM ViT feature @ mask centroid | 256 |
| `orient` | SE(3) / yaw vector | 6-D-rotation + axis Fourier-feature MLP | 128 |
| `measure` | Scalar(s) | Fourier features over raw numbers | 64 |
| `mental_rotate` → new view | Rendered RGB + pose | SigLIP2 + PoseEnc (**shared with view-shift idea**) | 768+128 |
| `bev_sketch` | Top-down rasterization | Small ConvNet over BEV | 256 |
| `python` | Text/scalar | (no latent; emit as text) | — |

**Trainable modules:** one 2-layer MLP projector per tool type (upstream of the frozen encoder, for input-side injection of tool outputs when they appear in the conditioning context) and one 2-layer MLP latent head per tool type (downstream of the VLM hidden state, outputting the target-dim latent).

**Special tokens:** `<lts type=T>`, `<lte>`, plus one `<tool_T>` discrete token per tool type. Embeddings initialized as the mean of 10 random existing tokens.

---

## 4. Data — the Teacher Agent

A 3D-tool-calling LLM agent (Qwen2.5-VL-72B or GPT-4o fallback) with the toolbox above solves queries from ScanQA / SQA3D / 3RScan / VSI-train. Each tool invocation emits `(tool_name, tool_args, tool_output)`, which becomes one step in the latent trace.

**Trace filters (load-bearing — positioned as a named data contribution):**

- **Trace coherence (local plausibility):** per-step similarity checks *within each tool type*. Examples: two consecutive BEV sketches must have BEV-encoder cosine sim ∈ [0.3, 0.92]; two consecutive orient vectors must have angular delta consistent with the textual rationale; detect boxes must not teleport across views. Fails → drop the trace.
- **Trace validity (global plausibility):** final answer matches dataset ground truth (exact match or strict GPT-4o-judge rubric for open-ended).
- **Length filter:** 2–6 tool steps.

Expected yield: 30–40 % of attempted queries. Target ~120 k kept traces. The filters are ablated explicitly to show each one pulls weight (§ 8).

---

## 5. How this differs from prior work

| Method | Target space | Structure | Tool teacher? | 3D-grounded encoder zoo? |
|---|---|---|---|---|
| Mirage / LVR / Sketchpad / LaViT | Single 2D image embedding | Per step | No | No |
| CoVT (2511.19418) | 4 × 2D perception experts | **Parallel** on input | No | No |
| Pearl (2604.08065) | Single JEPA trajectory embedding | **Trajectory** | 2D tools | No |
| SpaceTools (2512.04069) / Think3D (2601.13029) | **Text** + rendered images | Chained (text) | **Yes, 3D** | N/A (no latent target) |
| 3DThinker (2510.18632) | Whole-scene VGGT latent | Single block | No | Partial (VGGT only) |
| Companion view-shift doc | `SigLIP + PoseEnc` | Chained (latent) | Yes, 3D | Partial (1 pair) |
| **This work — tool-trace** | **8 × typed 3D encoders** | **Chained (latent)** | **Yes, 3D** | **Yes (PC, mask, orient, BEV, …)** |

The load-bearing novelty axis: **latent targets are the 3D-grounded tool outputs themselves, typed by tool, and chained sequentially as an agent trace** — not parallel feature extraction (CoVT), not a single trajectory embedding (Pearl), not text (SpaceTools/Think3D), not a single encoder pair (view-shift / Mirage / 3DThinker).

---

## 6. Training Stages

1. **Stage 1 — Modality Alignment.** Freeze VLM + all eight encoders. Train only the eight projectors + the new special-token embeddings on a mix of standard VQA and traces where tool outputs are injected as *input* tokens (i.e., ingestion, not generation). Exit: no regression > 1 pt on MMMU / MM-Vet.
2. **Stage 2 — Latent Thought Grounding.** Unfreeze VLM backbone + latent heads. Teacher-force: at each tool step, `L_latent = L2(head_T(h_i), Enc_T(tool_out_i))` with T = emitted tool type. Add InfoNCE with in-batch negatives after epoch 1 (λ 0 → 0.3). Text CE everywhere. Exit: per-tool teacher-forced cosine ≥ 0.8, general VQA CE ≤ 1.1× Stage-1.
3. **Stage 3 — End-to-End Reasoning.** Drop latent-token loss; keep format CE on `<lts type=T>` / `<lte>`. Scheduled sampling: teacher-forced → own latents, 80 % → 0 % over 1 epoch. Exit: SQA3D val strictly improving for three consecutive 1 k-step checkpoints.
4. **Stage 4 — GRPO.** Rewards: `+1` correct answer, `+0.1` well-formed typed-latent block, `−0.2` latent whose cosine to the true tool output of the *chosen* type is < 0.1 (anti-collapse gate). 8 rollouts / prompt, KL 0.04.

---

## 7. Open Design Questions

- **Hard vs soft tool typing.** Discrete type token (hard) vs a learned mixture over all eight heads (soft). Start hard; ablate.
- **Scalar / Python steps.** Do measurement scalars benefit from a learned Fourier-feature encoder, or should the VLM emit them as plain text? Likely the latter for simple arithmetic; the former when a scalar encodes a geometric quantity (e.g., an angle that downstream orient-steps consume).
- **Shared vs per-type projectors.** Tool types are heterogeneous (point cloud ≠ mask ≠ orientation); weight-sharing likely hurts. Verify with a shared-projector ablation.
- **Tool skipping at inference.** If the student emits fewer tools than the teacher used, is the answer still right? A clean diagnostic that the student isn't blindly copying.
- **Encoder-zoo size.** Is 8 encoders necessary or does a reduced zoo (say 4 — PC, region, orient, BEV) capture most of the gain? Ablate.
- **Auxiliary decoder for interpretability.** Optional tiny decoder per tool type that turns an emitted latent back into its tool output (a point cloud, a BEV image). Not required for accuracy; very useful for diagnostics.

---

## 8. Evaluation

**Primary benchmarks.** VSI-Bench (overall + per-subtask), 3DSRBench, SQA3D test, Real-3DQA. General-VQA regression guards: MMMU, MM-Vet, MMBench.

**Head-to-head comparisons — these are the paper's load-bearing experiments, each isolates one specific claim against the closest danger paper:**

| Baseline | What it tests | Closest prior work |
|---|---|---|
| Same encoder zoo, parallel feature extraction on input (no agent chain) | Is the **chained-agent** structure necessary, or is multi-encoder distillation enough? | CoVT |
| Same teacher trace, single JEPA trajectory encoder | Is **per-step decoding** necessary, or is a trajectory embedding enough? | Pearl |
| Same teacher trace, text-only distillation | Is **latent-space** supervision necessary, or is text CoT enough? | SpaceTools / Think3D |
| Same pipeline, only `(view, pose)` latents (view-shift companion) | Is the **typed multi-encoder zoo** better than a single view+pose pair? | Companion idea |
| Same pipeline, all tools share one encoder + projector | Is **tool typing** itself necessary? | — |

**Ablations.** One-tool-out (drop reconstruct / orient / BEV / etc.); L2 vs L2 + InfoNCE; trace-filter ablation (drop coherence, drop validity, drop both); #tool steps ∈ {1,2,4,6}; 8 encoders vs reduced 4-encoder zoo; hard vs soft typing.

**Diagnostics.**
- **Per-tool decoding accuracy:** for each emitted latent, nearest neighbor lookup in the frozen encoder's space; compare to the teacher's tool output.
- **Trace alignment:** step-by-step match of student's emitted tool type vs teacher's.
- **Tool-choice entropy** over training — does the model collapse onto one or two preferred tools?
- **Collapse guard:** fraction of emitted latents whose cosine to *any* real tool output of the chosen type is < 0.1.

---

## 9. Relationship to the Companion View-Shift Idea

- **Shared substrate:** backbone (Qwen2.5-VL-7B), VGGT input tokens, teacher agent, `<lts>/<lte>` delimiters, 4-stage training, trace-filter machinery.
- **Different latent target:** view-shift uses one pair `(SigLIP, PoseEnc)` for every step; tool-trace uses eight typed encoders, one per tool.
- **Reuses `mental_rotate` → View+Pose target:** the one tool whose output is a (view, pose) pair *is* the view-shift signal. In other words, the view-shift idea is a **strict sub-case** of this one, with the toolbox restricted to `mental_rotate`.
- **Natural unified version (Idea C, future):** typed encoder zoo with `mental_rotate` using the full view+pose target of the companion idea. This is the strongest form but also the largest engineering surface — scope as v2.

---

## 10. Summary — What We're Betting On

1. **Typed heterogeneous latent targets** are a better match for 3D reasoning than either a single homogeneous latent target (Mirage / view-shift) or a single trajectory embedding (Pearl).
2. **Sequential agent chaining** beats parallel feature extraction (CoVT) when the answer depends on intermediate artifacts feeding each other.
3. **Latent-space distillation** beats text-CoT distillation (Think3D / SpaceTools) when the tool outputs are geometrically rich.
4. **The trace coherence + validity filter pair is a reportable data contribution**, not boilerplate.

If any of (1)–(3) fails on its head-to-head ablation, the method still has value but the novelty narrative shrinks. (4) is defensible regardless.
