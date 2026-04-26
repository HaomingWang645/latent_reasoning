#!/usr/bin/env bash
# v4 ablation: λ_latent reduced to 0.3 (Stage 2) and 0.1 (Stage 3, instead of 0).
# 2 GPUs (2, 3 — PCIe). Designed to run ~20 hr.
#
# Hypothesis: v3's λ_latent=1.0 + Stage 3 drop-to-0 caused the LM head to
# overfit to VGGT alignment then catastrophically forget answer text when the
# anchor was removed. Reducing λ_latent and keeping a small anchor in Stage 3
# should preserve answer accuracy.

set -e
cd "$(dirname "$0")/.."

CFG=configs/full_pipeline_v4.yaml

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python
TORCHRUN=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

LOG_ROOT=/mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs
mkdir -p "$LOG_ROOT/logs/orchestrator"
LOG_FILE="$LOG_ROOT/logs/orchestrator/v4_pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "v4 Pipeline log: $LOG_FILE"

run_stage() {
    local stage=$1; local port=$2
    echo "=========================================" | tee -a "$LOG_FILE"
    echo " RUNNING $stage @ $(date)" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    "$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=$port \
        -m src.train.stage_trainer \
        --config "$CFG" --stage "$stage" 2>&1 | tee -a "$LOG_FILE"
}

run_stage4_dist() {
    echo "=========================================" | tee -a "$LOG_FILE"
    echo " RUNNING stage4_grpo_dist (DDP) @ $(date)" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    "$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=29613 \
        -m src.train.stage4_grpo_dist \
        --config "$CFG" \
        --init_run_name full_pipeline_v4_stage3 \
        --ref_run_name  full_pipeline_v4_stage3 \
        --run_suffix _grpo \
        --w_correct 1.0 --w_format 0.05 --format_gates_correct 0 \
        2>&1 | tee -a "$LOG_FILE"
}

run_stage stage1_align 29610
run_stage stage2_ground 29611
run_stage stage3_e2e 29612
run_stage4_dist

echo "=========================================" | tee -a "$LOG_FILE"
echo " v4 PIPELINE COMPLETE @ $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
