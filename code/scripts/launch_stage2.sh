#!/usr/bin/env bash
# Launches Stage 2 view-shift latent grounding on devices 2,3,4,5.
#
# Usage:
#   bash scripts/launch_stage2.sh                 # full pilot run
#   bash scripts/launch_stage2.sh --smoke         # 5-step smoke test
#   bash scripts/launch_stage2.sh --resume        # resume latest checkpoint
#
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # match nvidia-smi indices
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,5}  # GPU 4 held by another user; 2,3 PCIe + 5 NVL
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
export HF_HOME=${HF_HOME:-/home/haoming/.cache/huggingface}
export TRITON_CACHE_DIR=/tmp/triton_haoming
mkdir -p "$TRITON_CACHE_DIR"

# Use a free port for torchrun rendezvous
PORT=${PORT:-29503}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python
TORCHRUN=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun

"$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=$PORT \
    -m src.train.stage2_ground \
    --config configs/stage2_pilot.yaml \
    "$@"
