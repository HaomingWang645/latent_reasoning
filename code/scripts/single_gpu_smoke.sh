#!/usr/bin/env bash
# Single-GPU smoke test on device 2 — fastest path to validate the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
export HF_HOME=${HF_HOME:-/home/haoming/.cache/huggingface}

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python
TORCHRUN=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun

"$TORCHRUN" --standalone --nproc_per_node=1 --master_port=29504 \
    -m src.train.stage2_ground \
    --config configs/stage2_pilot.yaml \
    --smoke "$@"
