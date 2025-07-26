#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集验证脚本
检查生成的数据集格式是否符合Hugging Face标准
"""

import pandas as pd  # type: ignore
import os
import json
from PIL import Image
import argparse
from pathlib import Path

def get_script_directory():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def validate_directory_structure(dataset_dir):
    """验证数据集目录结构"""
    print("🔍 检查目录结构...")
    
    # required_files = ['metadata.csv', 'dataset_info.json', 'README.md']
    required_files = ['metadata.csv', 'dataset_info.json']
    required_dirs = ['images']
    
    issues = []
    
    # 检查必需文件
    for file in required_files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            print(f"  ✅ {file} 存在 ({os.path.getsize(file_path)} bytes)")
        else:
            issues.append(f"❌ 缺少文件: {file}")
    
    # 检查必需目录
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_dir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            image_count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])
            print(f"  ✅ {dir_name}/ 目录存在 ({image_count} 张图片)")
        else:
            issues.append(f"❌ 缺少目录: {dir_name}")
    
    return issues


def validate_metadata_csv(dataset_dir):
    """验证metadata.csv文件"""
    print("\n📊 检查metadata.csv...")
    
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    issues = []
    
    try:
        df = pd.read_csv(metadata_path)
        print(f"  ✅ 成功读取CSV文件，包含 {len(df)} 行数据")
        
        # 检查必需列
        required_columns = ['image_id', 'image_path', 'original_filename']
        for col in required_columns:
            if col in df.columns:
                print(f"  ✅ 包含必需列: {col}")
            else:
                issues.append(f"❌ 缺少必需列: {col}")
        
        # 检查数据类型和完整性
        if 'image_id' in df.columns:
            if df['image_id'].dtype in ['int64', 'int32']:
                print(f"  ✅ image_id 数据类型正确")
            else:
                issues.append(f"❌ image_id 数据类型错误: {df['image_id'].dtype}")
        
        # 检查是否有空值
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            print(f"  ✅ 无空值")
        else:
            print(f"  ⚠️  存在空值: {null_counts[null_counts > 0].to_dict()}")
        
        # 显示列信息
        print(f"  📋 包含列数: {len(df.columns)}")
        print(f"  📋 列名: {list(df.columns)}")
        
        return df, issues
        
    except Exception as e:
        issues.append(f"❌ 读取metadata.csv失败: {e}")
        return None, issues


def validate_images(dataset_dir, df):
    """验证图片文件"""
    print("\n🖼️  检查图片文件...")
    
    images_dir = os.path.join(dataset_dir, 'images')
    issues = []
    
    if df is None:
        issues.append("❌ 无法验证图片：metadata.csv 读取失败")
        return issues
    
    # 检查每张图片
    valid_images = 0
    image_sizes = []
    
    for idx, row in df.iterrows():
        image_path = os.path.join(dataset_dir, row['image_path'])
        
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
                    valid_images += 1
                    if idx < 3:  # 只显示前3张的详细信息
                        print(f"  ✅ {row['original_filename']}: {img.size} {img.format}")
            except Exception as e:
                issues.append(f"❌ 图片损坏: {row['image_path']} - {e}")
        else:
            issues.append(f"❌ 图片文件不存在: {row['image_path']}")
    
    print(f"  📊 有效图片: {valid_images}/{len(df)}")
    
    if image_sizes:
        unique_sizes = list(set(image_sizes))
        print(f"  📐 图片尺寸: {len(unique_sizes)} 种不同尺寸")
        print(f"  📐 尺寸范围: {min(image_sizes)} 到 {max(image_sizes)}")
    
    return issues


def validate_dataset_info(dataset_dir):
    """验证dataset_info.json"""
    print("\n📄 检查dataset_info.json...")
    
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    issues = []
    
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        print(f"  ✅ 成功读取JSON文件")
        
        # 检查必需字段
        required_fields = ['dataset_name', 'total_samples', 'data_format']
        for field in required_fields:
            if field in info:
                print(f"  ✅ 包含字段: {field} = {info[field]}")
            else:
                issues.append(f"❌ 缺少字段: {field}")
        
        return info, issues
        
    except Exception as e:
        issues.append(f"❌ 读取dataset_info.json失败: {e}")
        return None, issues


def test_huggingface_compatibility(dataset_dir):
    """测试Hugging Face兼容性"""
    print("\n🤗 测试Hugging Face兼容性...")
    
    try:
        # 测试pandas读取
        df = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'))
        print(f"  ✅ Pandas兼容: 可以读取 {len(df)} 行数据")
        
        # 测试图片路径
        first_image = os.path.join(dataset_dir, df.iloc[0]['image_path'])
        if os.path.exists(first_image):
            with Image.open(first_image) as img:
                print(f"  ✅ 图片可访问: {img.size} {img.format}")
        
        # 模拟datasets库的使用
        print("  🔧 模拟Hugging Face datasets使用:")
        print("     from datasets import Dataset, Image as HFImage")
        print("     dataset = Dataset.from_pandas(df)")
        print("     dataset = dataset.cast_column('image_path', HFImage())")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 兼容性测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证数据集格式")
    parser.add_argument("dataset_dir", nargs='?', 
                       default=get_script_directory() + "/dataset",
                       help="数据集目录路径")
    
    args = parser.parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    
    print("=" * 80)
    print("🔍 数据集验证工具")
    print("=" * 80)
    print(f"数据集路径: {dataset_dir}")
    print()
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return
    
    all_issues = []
    
    # 1. 验证目录结构
    issues = validate_directory_structure(dataset_dir)
    all_issues.extend(issues)
    
    # 2. 验证metadata.csv
    df, issues = validate_metadata_csv(dataset_dir)
    all_issues.extend(issues)
    
    # 3. 验证图片文件
    issues = validate_images(dataset_dir, df)
    all_issues.extend(issues)
    
    # 4. 验证dataset_info.json
    info, issues = validate_dataset_info(dataset_dir)
    all_issues.extend(issues)
    
    # 5. 测试Hugging Face兼容性
    test_huggingface_compatibility(dataset_dir)
    
    # 总结
    print("\n" + "=" * 80)
    if all_issues:
        print("❌ 验证完成，发现以下问题:")
        for issue in all_issues:
            print(f"  {issue}")
    else:
        print("✅ 验证完成，数据集格式完全正确！")
        print("\n🎯 数据集可以直接用于:")
        print("  • Hugging Face datasets库")
        print("  • PyTorch DataLoader")
        print("  • TensorFlow tf.data")
        print("  • 自定义训练脚本")
    
    print("=" * 80)


if __name__ == "__main__":
    main() 