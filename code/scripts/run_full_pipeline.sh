#!/usr/bin/env bash
# Full 4-stage pipeline orchestrator. Runs sequentially; if any stage fails the
# next ones are NOT executed (set -e). Resumable: re-run with --resume to pick
# up the most recent stage at its latest checkpoint.
#
# Stages 1-3 use multi-GPU (DDP across CUDA_VISIBLE_DEVICES).
# Stage 4 uses single GPU (rollouts cost is the bottleneck, not data parallelism).
#
set -e
cd "$(dirname "$0")/.."

CFG=configs/full_pipeline.yaml
RESUME=""
SKIP_S1=""
SKIP_S2=""
SKIP_S3=""
SKIP_S4=""
while [ $# -gt 0 ]; do
    case "$1" in
        --resume) RESUME="--resume"; shift;;
        --skip-stage1) SKIP_S1="1"; shift;;
        --skip-stage2) SKIP_S2="1"; shift;;
        --skip-stage3) SKIP_S3="1"; shift;;
        --skip-stage4) SKIP_S4="1"; shift;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6}  # 5 H100s
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
export HF_HOME=${HF_HOME:-/home/haoming/.cache/huggingface}

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python
TORCHRUN=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

LOG_ROOT=/mnt/data3/haoming_x_spatial_scratch/latent_reasoning_runs
mkdir -p "$LOG_ROOT/logs/orchestrator"
LOG_FILE="$LOG_ROOT/logs/orchestrator/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Pipeline log: $LOG_FILE"

run_stage() {
    local stage=$1
    local port=$2
    echo "=========================================" | tee -a "$LOG_FILE"
    echo " RUNNING $stage @ $(date)" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    "$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=$port \
        -m src.train.stage_trainer \
        --config "$CFG" --stage "$stage" $RESUME 2>&1 | tee -a "$LOG_FILE"
}

run_stage4_dist() {
    echo "=========================================" | tee -a "$LOG_FILE"
    echo " RUNNING stage4_grpo_dist (DDP) @ $(date)" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    "$TORCHRUN" --standalone --nproc_per_node=$NPROC --master_port=29513 \
        -m src.train.stage4_grpo_dist \
        --config "$CFG" \
        --init_run_name full_pipeline_v3_stage3 \
        --ref_run_name  full_pipeline_v3_stage3 \
        --run_suffix _grpo \
        --w_correct 1.0 --w_format 0.05 --format_gates_correct 0 \
        2>&1 | tee -a "$LOG_FILE"
}

[ -z "$SKIP_S1" ] && run_stage stage1_align 29510
[ -z "$SKIP_S2" ] && run_stage stage2_ground 29511
[ -z "$SKIP_S3" ] && run_stage stage3_e2e 29512
[ -z "$SKIP_S4" ] && run_stage4_dist

echo "=========================================" | tee -a "$LOG_FILE"
echo " PIPELINE COMPLETE @ $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
