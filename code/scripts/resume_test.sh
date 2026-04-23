#!/usr/bin/env bash
# Verify resume works end-to-end: capture current step, run --resume for a few
# more steps, confirm step counter advanced monotonically.
#
# Run AFTER at least one checkpoint exists.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/stage2_pilot.yaml
LATEST=$(readlink /mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs/ckpts/stage2_pilot_mindcube_v0/latest 2>/dev/null || echo "")
if [ -z "$LATEST" ]; then
    echo "FAIL: no latest checkpoint to resume from"
    exit 1
fi
echo "Resuming from $LATEST"

# Run 3 more optimizer steps via override
PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python
TORCHRUN=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,5}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"

START_STEP=$(echo "$LATEST" | sed 's/step_0*\([0-9]\+\)/\1/')
TARGET=$((START_STEP + 3))
echo "Will train to step $TARGET (from $START_STEP)"

"$TORCHRUN" --standalone --nproc_per_node=3 --master_port=29505 \
    -m src.train.stage2_ground \
    --config "$CFG" --resume --max_steps_override $TARGET 2>&1 | tail -20

NEW_LATEST=$(readlink /mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs/ckpts/stage2_pilot_mindcube_v0/latest)
NEW_STEP=$(echo "$NEW_LATEST" | sed 's/step_0*\([0-9]\+\)/\1/')
if [ "$NEW_STEP" -gt "$START_STEP" ]; then
    echo
    echo "PASS: resumed from $START_STEP, latest now $NEW_STEP"
else
    echo "FAIL: latest still at $START_STEP (expected > $START_STEP)"
    exit 1
fi
