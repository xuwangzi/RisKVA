#!/bin/bash

# =================================================================
# RisKVA 子单元风险分析训练脚本 - Qwen2.5-VL-7B-Instruct (内存优化版)
# =================================================================

set -e  # 遇到错误立即退出
set -o pipefail # 任何管道失败都退出，避免误报“训练完成”

# 设置输出目录
DATASET_PATH="datasets/RisKVA/Subunit-Risk_original/metadata.csv"
PRETRAINED_MODEL_PATH="models/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="models/finetuned_models/RisKVA/RisKVA-Qwen2.5-VL-7B-Instruct-sft-subunit-risk-peft"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs/training"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "开始训练 RisKVA 分户检查风险分析模型"
echo "模型: Qwen2.5-VL-7B-Instruct"
echo "时间戳: ${TIMESTAMP}"
echo "输出目录: ${OUTPUT_DIR}"
echo "========================================"

# 启动训练
export CLEANUP_EVERY_N=40 # 每N个batch清理一次内存
accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    src/sft_subunit_risk/train.py \
    --dataset_name "${DATASET_PATH}" \
    --model_name_or_path "${PRETRAINED_MODEL_PATH}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --save_steps 50 \
    --save_total_limit 5 \
    --report_to tensorboard \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_task_type CAUSAL_LM \
    --use_rslora False \
    --use_dora False \
    2>&1 | tee "${LOG_DIR}/train_subunit_risk_7b_peft_${TIMESTAMP}.log"
    # --gradient_checkpointing \  # 注释掉梯度检查点避免与PEFT+量化冲突

echo "========================================"
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "训练日志: ${LOG_DIR}/train_subunit_risk_7b_peft_${TIMESTAMP}.log"
echo "========================================"