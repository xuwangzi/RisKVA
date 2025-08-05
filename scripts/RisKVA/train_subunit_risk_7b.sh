#!/bin/bash

# =================================================================
# RisKVA 子单元风险分析训练脚本 - Qwen2.5-VL-7B-Instruct (内存优化版)
# =================================================================

set -e  # 遇到错误立即退出

# 设置输出目录
DATASET_PATH="datasets/RisKVA/Subunit-Risk_original/metadata.csv"
PRETRAINED_MODEL_PATH="models/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="models/finetuned_models/RisKVA/RisKVA-Qwen2.5-VL-7B-Instruct-sft-subunit-risk"

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
accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    src/sft_subnunit_risk/train.py \
    --dataset_name "${DATASET_PATH}" \
    --model_name_or_path "${PRETRAINED_MODEL_PATH}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 True \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --save_steps 50 \
    --save_total_limit 5 \
    --report_to tensorboard \
    2>&1 | tee "${LOG_DIR}/train_subunit_risk_7b_${TIMESTAMP}.log"

echo "========================================"
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "训练日志: ${LOG_DIR}/train_subunit_risk_7b_${TIMESTAMP}.log"
echo "========================================"