# Latent Spatial Reasoning via Step-Level View-Shift Supervision

## Motivation

Current approaches to spatial reasoning in VLMs fall into three camps, each with a fundamental limitation:

1. **Input-layer 3D injection** (SpatialMLLM, VST-SFT, SpaceR). Feeding depth / point clouds / multi-view into the VLM input and fine-tuning. Gains plateau after ~2M samples; the model takes language shortcuts and suffers attention dilution on dense 3D features.
2. **Text-domain CoT.** Explicit stepwise or multi-view CoT *hurts* spatial accuracy vs. vanilla answering (e.g., Qwen2.5-VL distance comparison drops from 43.4 to 34.2 with multi-view CoT). Thinking in text pulls attention away from relevant image regions.
3. **Latent reasoning with dense 3D supervision** (3DThinker). Reasons in a continuous latent space with frozen-VGGT alignment + GRPO RLVR. Only dense, whole-scene 3D features are used — **no fine-grained, step-level 2D spatial supervision during view shifts**.

Human spatial reasoning is not a single dense 3D blob; it is a sequence of *mental viewpoint changes*. The proposal below operationalizes that.

## Core Idea

Give the model **asymmetric** input and output latent spaces:

| | Encoder (frozen) | Latent role |
|---|---|---|
| **Input context** | 3D VGGT encoder over multi-view RGB | dense, whole-scene geometric conditioning |
| **Output thoughts** | 2D vision encoder + camera-pose encoder over a single extracted view | per-step mental viewpoint the VLM must emit |

At each reasoning step the VLM mentally "moves the camera" — it autoregressively generates a `[2D latent token]` wrapped by `<lts>` and `<lte>`, whose target is the embedding of the actual scene view at a specified camera pose.

The latent reasoning trace becomes:

```
text -> <lts> view@pose_1 <lte> -> text -> <lts> view@pose_2 <lte> -> ... -> answer
```

This forces reasoning *through viewpoints* rather than through a static 3D representation.

## Architecture

```
Input (to VLM):
  [Query tokens] + [Image tokens] + [3D latent tokens from VGGT]

Output (from VLM, interleaved):
  [Text tokens]                   <- standard LM head
  <lts> [2D latent tokens] <lte>  <- supervised against
                                     2D-vision-encoder(extracted_view)
                                   + camera-pose-encoder(extracted_pose)
```

- **Trainable**: VLM backbone only.
- **Frozen**: VGGT (3D input encoder), 2D vision encoder (target for latent thoughts), camera pose encoder.
- **Special tokens**: `<lts>` / `<lte>` delimit a latent thought; the span between them carries the view-conditioned latent.

## Training Stages

1. **Stage 1 — Modality Alignment.** Train projectors so the VLM ingests VGGT 3D latents, 2D view latents, and pose latents in a shared space. LLM backbone frozen.
2. **Stage 2 — Latent Thought Grounding.** Teacher-force: at each reasoning step the model must emit a `[2D latent token]` whose embedding matches the frozen `2D-encoder(view) + pose-encoder(pose)` target. L2 (or contrastive) loss on latent tokens; CE on text tokens.
3. **Stage 3 — End-to-End Reasoning.** Drop latent-token supervision; keep only text-answer CE. The model must *choose* which views to attend to and arrive at the correct answer.
4. **Stage 4 — Reinforcement Learning.** GRPO with reward = answer accuracy (+ format). Latent thoughts are now emergent; RL refines which mental viewpoints help.

## Data: Step-Level Ground-Truth Reasoning Traces

Traces are synthesized by a **3D-tool-calling agent** acting as teacher:

- Toolbox: 3D reconstruction, detection, segmentation, object orientation, measurement, mental rotation, BEV sketch plotting, Python.
- Agent decomposes each query into: reconstruct -> detect target objects -> segment -> extract orientation -> plot BEV -> conclude.
- Each tool-call step yields a `(view, camera_pose)` pair that becomes the ground-truth target for one latent thought token in the trace.
- Trace kept only if:
  - **Trace coherence** (local plausibility): each step is consistent with the previous.
  - **Trace validity** (global validity): the chain leads to the correct final answer.

This gives *structured, mixed-modality, step-level* supervision — filling exactly the gap 3DThinker identified.

## How This Differs From Prior Work

| Method | Input latent | Output latent | Step-level supervision |
|---|---|---|---|
| Think-with-image (ReFocus, Visual-CoT, Zebra-CoT) | 2D | 2D regions / crops | 2D focus only, no geometry |
| Geometrically-Constrained Agent | 2D + tool outputs | text + code | pipeline, not end-to-end |
| 3DThinker | 2D + 3D | dense 3D (VGGT) | whole-scene, not step-level |
| **Ours** | **2D + 3D (VGGT)** | **2D view @ pose** | **per-step, view-shift grounded** |

The distinguishing move: latents at input are dense 3D for context; latents at output are **pose-conditioned 2D views** — one per reasoning step — so the VLM's thought process is an explicit sequence of mental viewpoint changes.

## Open Design Questions

- Is the extracted view a *real captured* view from the dataset, or a *rendered novel* view from the 3D reconstruction? The former is cleaner supervision; the latter allows arbitrary mental-rotation trajectories.
- How is the target camera pose chosen at training time — from the teacher agent's trace, or sampled from a pose prior?
- At inference the model must commit to a pose before emitting the latent. Does the model predict the pose explicitly (a small regression head) or does the pose encoder consume an emitted pose token?
- L2 vs. contrastive (InfoNCE-style) loss on latent tokens: contrastive is more robust but needs negatives.
- How many latent steps per trace before diminishing returns? Budget to be ablated.

## Evaluation

- VSI-Bench / 3DSRBench / SQA3D / Real-3DQA for spatial reasoning accuracy.
- Compositional stress tests: remove linguistic shortcuts to verify the model actually uses geometric latents (cf. SQA3D no-language-cue evaluation).
- Trace-level diagnostics: do emitted 2D latents decode (via the frozen 2D encoder's inverse or a nearest-neighbor lookup) to plausible views of the scene?
- Ablations: no-3D-input, no-latent-supervision (Pathway 1 only), dense-VGGT-target (3DThinker-equivalent), no-pose-conditioning.
