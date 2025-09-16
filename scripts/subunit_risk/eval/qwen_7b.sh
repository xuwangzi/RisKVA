#!/bin/bash

# =================================================================
# RisKVA 子单元风险分析评估脚本
# =================================================================

set -e  # 遇到错误立即退出
set -o pipefail # 任何管道失败都退出，避免误报“训练完成”

# 设置目录
PRETRAINED_MODEL_PATH="models/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct"
EVAL_MODEL_PATH="models/finetuned_models/RisKVA/RisKVA-Subunit-7B_image_augmentation_3_epoch"
DATASET_PATH="datasets/RisKVA/Subunit-Risk_test/metadata.csv"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs/evaluating"

# 创建必要的目录
mkdir -p "${LOG_DIR}"

echo -e "✅ 开始评估\n预训练模型: ${PRETRAINED_MODEL_PATH}\n评估模型: ${EVAL_MODEL_PATH}\n测试数据: ${DATASET_PATH}" | tee -a "${LOG_DIR}/eval_${TIMESTAMP}.log"

# 启动评估
python src/sft_subunit_risk/evaluation.py \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --finetuned_model_path "${EVAL_MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    2>&1 | tee -a "${LOG_DIR}/eval_${TIMESTAMP}.log"

echo -e "✅ 评估完成！\n评估日志: ${LOG_DIR}/eval_${TIMESTAMP}.log" | tee -a "${LOG_DIR}/eval_${TIMESTAMP}.log"