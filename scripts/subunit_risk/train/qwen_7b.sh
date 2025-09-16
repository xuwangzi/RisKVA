#!/bin/bash

# ==============================
# RisKVA 子单元风险分析训练脚本
# ==============================

set -e  # 遇到错误立即退出
set -o pipefail # 任何管道失败都退出，避免误报“训练完成”

# 设置目录
DATASET_PATH="datasets/RisKVA/Subunit-Risk_original/metadata_cleaned.csv"
PRETRAINED_MODEL_PATH="models/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="models/finetuned_models/RisKVA/RisKVA-Qwen2.5-VL-7B-Instruct-sft-subunit-risk"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs/training"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo -e "✅ 开始训练\n模型: ${PRETRAINED_MODEL_PATH}\n数据: ${DATASET_PATH}\n输出: ${OUTPUT_DIR}" | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"

# 启动训练
export CLEANUP_EVERY_N=32 # 每N个batch清理一次内存
accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    src/sft_subunit_risk/train.py \
    --dataset_name "${DATASET_PATH}" \
    --model_name_or_path "${PRETRAINED_MODEL_PATH}" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 True \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --save_steps 50 \
    --save_total_limit 5 \
    --report_to tensorboard \
    2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"  
# todo: 超参数
    # --num_train_epochs 1 \
    # --learning_rate 1e-4 \
    # --lr_scheduler_type cosine \
    # --warmup_ratio 0.03 \

# -e 选项用于使 echo 支持转义字符（如 \n 实现换行），否则 \n 会被当作普通字符输出
echo -e "✅ 训练完成！\n模型保存在: ${OUTPUT_DIR}\n训练日志: ${LOG_DIR}/train_${TIMESTAMP}.log" | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"