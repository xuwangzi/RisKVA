#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本：将风险识别数据集转换为Parquet格式，包含图片数据
支持将图片编码为base64存储在parquet文件中，实现数据的完整打包
"""

import pandas as pd  # type: ignore
import json
import os
import base64
import io
import yaml  # type: ignore
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm  # type: ignore

def load_prompts_config() -> Dict[str, Any]:
    """
    加载提示词配置文件
    
    Returns:
        配置字典
    """
    # 获取配置文件路径（相对于当前脚本）
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent / "configs" / "prompts" / "building_risk_prompts.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")
        return {}

def encode_image_to_base64(image_path: str, max_size: tuple = (1024, 1024), quality: int = 85) -> Optional[str]:
    """
    将图片编码为base64字符串，支持压缩
    
    Args:
        image_path: 图片路径
        max_size: 最大尺寸，用于压缩
        quality: JPEG质量（1-100）
        
    Returns:
        base64编码的字符串，失败返回None
    """
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 转换为RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整尺寸（如果图片过大）
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 保存到内存缓冲区
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            # 编码为base64
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded_string
            
    except Exception as e:
        print(f"编码图片失败 {image_path}: {e}")
        return None

def create_conversation_format(row: pd.Series, image_base64: Optional[str]) -> Dict[str, Any]:
    """
    将单条记录转换为对话格式，使用base64编码的图片
    
    Args:
        row: pandas行数据
        image_base64: base64编码的图片数据
        
    Returns:
        格式化的对话数据
    """
    # 加载提示词配置
    config = load_prompts_config()
    
    # 从配置中获取系统提示
    system_prompt = config.get("system_prompts", {}).get("building_defect_expert", "")
    
    # 从配置中获取用户问题
    user_question = config.get("user_prompts", {}).get("standard_analysis", "")
    
    # 从配置中获取回答模板并格式化
    template = config.get("response_templates", {}).get("standard_format", "")
    assistant_answer = template.format(
        defect_description=row['data_缺陷描述'],
        risk_description=row['data_风险'],
        correction_suggestions=row['data_纠正和预防建议'],
        risk_level_original=row.get('data_原', ''),
        risk_level_now=row.get('data_现', ''),
    )

    # 返回标准的对话格式
    conversation = {
        "id": f"risk_detection_{row['image_id']}",
        "image_base64": image_base64,  # 使用base64编码的图片
        "original_image_path": row['image_path'],  # 保留原始路径信息
        "conversations": [
            {
                "from": "system",
                "value": system_prompt
            },
            {
                "from": "user", 
                "value": f"<image>\n{user_question}"
            },
            {
                "from": "assistant",
                "value": assistant_answer
            }
        ],
        # 保存原始数据的额外字段
        "metadata": {
            "序号": row.get('data_序号', ''),
            "风险": row['data_风险'],
            "纠正和预防建议": row['data_纠正和预防建议'],
            "风险等级": row['data_风险等级'],
            "原状态": row.get('data_原', ''),
            "现状态": row.get('data_现', ''),
            "缺陷描述": row['data_缺陷描述'],
            "来源文件": row.get('source_csv', ''),
            "原始文件名": row.get('original_filename', '')
        }
    }
    
    return conversation

def prepare_sft_data_parquet(csv_path: str, image_base_path: str, output_path: str, 
                           train_ratio: float = 0.8, val_ratio: float = 0.1,
                           image_quality: int = 85, max_image_size: tuple = (1024, 1024)) -> None:
    """
    准备SFT训练数据并保存为Parquet格式
    
    Args:
        csv_path: CSV文件路径
        image_base_path: 图片基础路径
        output_path: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        image_quality: 图片JPEG质量
        max_image_size: 图片最大尺寸
    """
    # 读取CSV数据
    print(f"正在读取数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"总共读取到 {len(df)} 条数据")
    
    # 处理数据并编码图片
    processed_data = []
    missing_images = []
    
    print("正在处理图片数据...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        image_path = os.path.join(image_base_path, row['image_path'])
        
        if os.path.exists(image_path):
            # 编码图片
            image_base64 = encode_image_to_base64(
                image_path, 
                max_size=max_image_size, 
                quality=image_quality
            )
            
            if image_base64:
                # 创建对话格式
                conv = create_conversation_format(row, image_base64)
                processed_data.append(conv)
            else:
                missing_images.append(image_path)
                print(f"图片编码失败: {image_path}")
        else:
            missing_images.append(image_path)
    
    if missing_images:
        print(f"警告: 发现 {len(missing_images)} 个缺失或损坏的图片文件")
        print("前5个问题文件:", missing_images[:5])
    
    print(f"成功处理 {len(processed_data)} 条数据")
    
    # 打乱数据（重要：确保随机性）
    np.random.seed(42)  # 设置随机种子以确保可重现性
    indices = np.random.permutation(len(processed_data))
    processed_data = [processed_data[i] for i in indices]
    
    # 划分数据集
    total_size = len(processed_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size + val_size]
    test_data = processed_data[train_size + val_size:]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存数据集为Parquet格式
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, data in datasets.items():
        if len(data) > 0:
            # 转换为DataFrame
            df_split = pd.DataFrame(data)
            
            # 保存为Parquet
            parquet_file = os.path.join(output_path, f"{split_name}.parquet")
            df_split.to_parquet(parquet_file, compression='snappy', index=False)
            print(f"已保存 {split_name} 数据集到: {parquet_file}")
            
            # 计算文件大小
            file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
            print(f"  文件大小: {file_size_mb:.2f} MB")
    
    # 生成统计信息
    stats = {
        "total_samples": total_size,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "risk_level_distribution": df['data_风险等级'].value_counts().to_dict(),
        "missing_images": len(missing_images),
        "image_processing": {
            "quality": image_quality,
            "max_size": max_image_size,
            "compression": "snappy"
        }
    }
    
    stats_file = os.path.join(output_path, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"数据集统计信息已保存到: {stats_file}")
    print("Parquet数据预处理完成!")

def load_parquet_data(parquet_path: str, decode_images: bool = False) -> pd.DataFrame:
    """
    从Parquet文件加载数据
    
    Args:
        parquet_path: Parquet文件路径
        decode_images: 是否解码图片数据为PIL Image对象
        
    Returns:
        加载的DataFrame
    """
    print(f"正在加载Parquet数据: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if decode_images and 'image_base64' in df.columns:
        print("正在解码图片数据...")
        
        def decode_base64_to_image(base64_str):
            try:
                image_data = base64.b64decode(base64_str)
                image = Image.open(io.BytesIO(image_data))
                return image
            except:
                return None
        
        # 添加解码后的图片列
        df['image_pil'] = df['image_base64'].apply(decode_base64_to_image)
    
    print(f"加载完成，共 {len(df)} 条数据")
    return df

def preview_parquet_data(parquet_path: str, num_samples: int = 3):
    """
    预览Parquet数据内容
    
    Args:
        parquet_path: Parquet文件路径
        num_samples: 预览样本数量
    """
    df = load_parquet_data(parquet_path, decode_images=True)
    
    print(f"\n=== Parquet文件预览: {parquet_path} ===")
    print(f"数据维度: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    for i in range(min(num_samples, len(df))):
        print(f"\n--- 样本 {i+1} ---")
        row = df.iloc[i]
        print(f"ID: {row['id']}")
        print(f"原始图片路径: {row['original_image_path']}")
        
        if 'image_pil' in df.columns and row['image_pil'] is not None:
            img = row['image_pil']
            print(f"图片尺寸: {img.size}")
        
        print(f"对话轮数: {len(row['conversations'])}")
        
        # 显示对话内容（截断显示）
        for j, conv in enumerate(row['conversations']):
            content = conv['value'][:100] + "..." if len(conv['value']) > 100 else conv['value']
            print(f"  {conv['from']}: {content}")

def main():
    parser = argparse.ArgumentParser(description='准备Qwen2.5-VL SFT训练数据（Parquet格式）')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--image_base_path', type=str, required=True, help='图片基础路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--image_quality', type=int, default=85, help='图片JPEG质量(1-100)')
    parser.add_argument('--max_image_size', type=int, nargs=2, default=[1024, 1024], 
                        help='图片最大尺寸 [width height]')
    parser.add_argument('--preview', type=str, help='预览指定的parquet文件')
    
    args = parser.parse_args()
    
    if args.preview:
        # 预览模式
        preview_parquet_data(args.preview)
    else:
        # 处理模式
        prepare_sft_data_parquet(
            csv_path=args.csv_path,
            image_base_path=args.image_base_path,
            output_path=args.output_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            image_quality=args.image_quality,
            max_image_size=tuple(args.max_image_size)
        )

if __name__ == "__main__":
    main() 