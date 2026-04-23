# Latent Reasoning

Research repo exploring **latent reasoning for vision-language models** — training VLMs to reason in a continuous latent space rather than over textual chains of thought.

Two concrete research directions are developed here:

1. **Latent Spatial Reasoning (view-shift grounding)** — anchor latent reasoning steps to 3D view shifts, supervised step-by-step against a target visual encoder.
2. **Tool-Trace Latent Reasoning** — distill tool-call traces (e.g. detectors, search, Python) into latent reasoning tokens so the VLM can mimic tool use at inference time without invoking tools.

## Layout

```
.
├── latent_spatial_reasoning_idea.md              # Direction 1 — research idea
├── latent_spatial_reasoning_implementation_plan.md  # Direction 1 — full 4-stage plan
├── tool_trace_latent_reasoning_idea.md           # Direction 2 — research idea
├── tool_trace_latent_reasoning_implementation_plan.md # Direction 2 — plan
├── code/                                         # Stage-2 pilot implementation (Direction 1)
│   └── README.md                                 # build/run instructions
├── related papers/                               # reference PDFs (latent CoT, visual reasoning)
├── latent reasoning ideas.pdf                    # seed reading
├── scaling visual spatial reasoning at inference time.pdf
└── The Latent Space- Foundation, Evolution, Mechanism, Ability, and Outlook.pdf
```

## Current status

- Ideas and implementation plans are written up in markdown.
- [code/](code/) contains a **Stage-2 pilot** of the view-shift latent grounding pipeline: Qwen2.5-VL-7B + LoRA + `<lts>`/`<lat>`/`<lte>` tokens + SigLIP target encoder, trained on MindCube. See [code/README.md](code/README.md) for run instructions, GPU setup, and what the pilot does/does not prove.
- Stages 1, 3, 4 (GRPO) are not yet implemented.

## Quick start

```bash
cd code
bash scripts/single_gpu_smoke.sh   # 5-step smoke test
bash scripts/launch_stage2.sh      # full pilot on 3 H100s
```

Details — config knobs, checkpoint layout, resume — live in [code/README.md](code/README.md).
