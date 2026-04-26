#!/usr/bin/env bash
# Resume v4 pipeline on GPUs 6, 7. Stage 1 done, Stage 2 at step 5000/30000.
# Picks up Stage 2 from latest ckpt, then runs Stage 3, then Stage 4 GRPO.
set -e
cd "$(dirname "$0")/.."

CFG=configs/full_pipeline_v4.yaml

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python
TORCHRUN=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

LOG_ROOT=/mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs
mkdir -p "$LOG_ROOT/logs/orchestrator"
LOG_FILE="$LOG_ROOT/logs/orchestrator/v4_resume_$(date +%Y%m%d_%H%M%S).log"
echo "v4 resume log: $LOG_FILE"

run_stage_resume() {
    local stage=$1; local port=$2
    echo "=========================================" | tee -a "$LOG_FILE"
    echo " RESUMING $stage @ $(date)" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    "$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=$port \
        -m src.train.stage_trainer \
        --config "$CFG" --stage "$stage" --resume 2>&1 | tee -a "$LOG_FILE"
}

run_stage_fresh() {
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
    echo " RUNNING stage4_grpo_dist @ $(date)" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    "$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=29614 \
        -m src.train.stage4_grpo_dist \
        --config "$CFG" \
        --init_run_name full_pipeline_v4_stage3 \
        --ref_run_name  full_pipeline_v4_stage3 \
        --run_suffix _grpo \
        --w_correct 1.0 --w_format 0.05 --format_gates_correct 0 \
        2>&1 | tee -a "$LOG_FILE"
}

# Stage 2 resume from step 5000; Stage 3 + 4 fresh
run_stage_resume stage2_ground 29711
run_stage_fresh stage3_e2e 29712
run_stage4_dist

echo " v4 PIPELINE COMPLETE @ $(date)" | tee -a "$LOG_FILE"
