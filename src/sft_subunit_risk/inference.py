#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RisKVA 风险评估推理脚本

本脚本用于基于视觉语言模型对房屋分户检查进行风险评估推理。
支持多模态输入（图像+文本），能够分析缺陷并评估风险等级。

主要功能：
- 构建多模态/纯文本消息
- 调用模型生成推理结果
- 解析模型输出为结构化结果
- 提供命令行便捷入口

作者: xuwangzi
创建时间: 2025
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

PRETRAINED_MODEL_PATH="models/pretrained_models/LLaVa/llava-v1.6-mistral-7b-hf"
EVAL_MODEL_PATH="models/finetuned_models/RisKVA/RisKVA-llava-v1.6-mistral-7b-hf-sft-subunit-risk"
DATASET_PATH="datasets/RisKVA/Subunit-Risk_test/metadata.csv"

processor = LlavaNextProcessor.from_pretrained(PRETRAINED_MODEL_PATH)

model = LlavaNextForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
image = Image.open("/root/group-shared/yaww-ai-lesson/xwz/code/RisKVA/datasets/RisKVA/Subunit-Risk_original/images/000024_00_SR-FH-1-20250611-026.jpg")
user_text = (
    "作为房屋分户检查专家，你现在正在进行分户检查分析。"
    f"本次分析提供了1张图像。根据提供的1张图像，"
    "请分析对应的\"缺陷\"、\"风险\"、\"风险等级（原本）\"和\"风险等级（现在）\"，"
    "并提供\"纠正和预防建议\"。\n"
    "请按照以下格式回答：\n"
    "【缺陷】：\n"
    "【风险】：\n"
    "【风险等级（原本）】：\n"
    "【风险等级（现在）】：\n"
    "【纠正和预防建议】：\n"
)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": user_text},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))