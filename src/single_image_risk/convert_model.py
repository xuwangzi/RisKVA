#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型转换脚本：将LoRA权重合并到基础模型中
"""

import os
import argparse
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

def merge_lora_weights(model_path: str, output_path: str):
    """
    合并LoRA权重到基础模型
    
    Args:
        model_path: 训练后的模型路径（包含LoRA权重）
        output_path: 输出合并后模型的路径
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查是否存在adapter_config.json（表示这是LoRA模型）
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("检测到LoRA模型，开始合并权重...")
        
        # 读取adapter配置获取基础模型路径
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_path = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"基础模型: {base_model_path}")
        
        # 加载基础模型
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 加载LoRA模型
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 合并权重
        print("正在合并LoRA权重...")
        merged_model = model.merge_and_unload()
        
        # 保存合并后的模型
        print(f"保存合并后的模型到: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path, safe_serialization=True)
        
        # 复制processor
        processor = AutoProcessor.from_pretrained(base_model_path)
        processor.save_pretrained(output_path)
        
        print("权重合并完成！")
        
    else:
        print("未检测到LoRA配置，直接复制模型...")
        
        # 直接加载并保存模型
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path, safe_serialization=True)
        processor.save_pretrained(output_path)
        
        print("模型复制完成！")

def main():
    parser = argparse.ArgumentParser(description='合并LoRA权重到基础模型')
    parser.add_argument('--model_path', type=str, required=True, help='训练后的模型路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径')
    
    args = parser.parse_args()
    
    merge_lora_weights(args.model_path, args.output_path)

if __name__ == "__main__":
    main() 