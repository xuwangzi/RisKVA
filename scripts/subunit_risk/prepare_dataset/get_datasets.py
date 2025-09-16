#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集整理脚本
将input目录中的图片和CSV数据按顺序对应，整理为Hugging Face数据集格式
"""

import pandas as pd  # type: ignore
import os
import sys
import json
from pathlib import Path
import argparse
import shutil
from typing import List, Dict, Tuple
import glob
from PIL import Image
import re


def scan_images(input_dir: str) -> List[str]:
    """
    扫描输入目录中的所有图片文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        图片文件路径列表（已排序）
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
        # 也搜索大写扩展名
        pattern = os.path.join(input_dir, '**', ext.upper())
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    return image_files


def scan_csv_files(input_dir: str) -> List[str]:
    """
    扫描输入目录中的所有CSV文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        CSV文件路径列表（已排序）
    """
    pattern = os.path.join(input_dir, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)
    return sorted(csv_files)


def load_csv_data(csv_files: List[str]) -> pd.DataFrame:
    """
    加载并合并所有CSV文件数据
    
    Args:
        csv_files: CSV文件路径列表
        
    Returns:
        合并后的DataFrame
    """
    all_data = []
    
    for csv_file in csv_files:
        try:
            print(f"正在读取CSV文件: {csv_file}")
            df = pd.read_csv(csv_file, encoding='utf-8')
            # 添加源文件信息
            all_data.append(df)
            print(f"  成功读取 {len(df)} 行数据")
        except Exception as e:
            print(f"  ❌ 读取CSV文件失败: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"总共合并了 {len(combined_df)} 行数据")
    return combined_df


def create_dataset_structure(output_dir: str):
    """
    创建Hugging Face数据集目录结构
    
    Args:
        output_dir: 输出目录路径
    """
    # 创建主要目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "images")).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "data")).mkdir(exist_ok=True)
    
    print(f"创建数据集目录结构: {output_dir}")


def natural_sort_key(text):
    """
    自然排序的键函数，正确处理数字
    例如: file1.jpg, file2.jpg, file10.jpg
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', text)]


def match_images_with_data(image_files: List[str], csv_data: pd.DataFrame) -> List[Dict]:
    """
    将图片与CSV数据按顺序匹配
    
    Args:
        image_files: 图片文件列表
        csv_data: CSV数据DataFrame
        
    Returns:
        匹配后的数据列表
    """
    matched_data = []
    
    # 对图片文件进行自然排序
    image_files = sorted(image_files, key=lambda x: natural_sort_key(os.path.basename(x)))
    
    print(f"开始匹配 {len(image_files)} 张图片与 {len(csv_data)} 行数据...")
    
    image_index = 0  # 当前图片索引
    
    # 遍历每一行CSV数据
    for data_id, row in csv_data.iterrows():
        row_csv_data = row.to_dict()
        
        # 获取当前行需要的图片数量，默认为1
        image_count = int(row_csv_data.get('image_count', 1))
        
        # 检查是否有足够的图片
        if image_index + image_count > len(image_files):
            print(f"  警告: 数据行{data_id}需要{image_count}张图片，但只剩{len(image_files) - image_index}张")
            image_count = len(image_files) - image_index
            if image_count <= 0:
                break
        else:
            print(f"  数据行{data_id}需要{image_count}张图片，找到{len(image_files) - image_index}张")
        
        # 从图片列表中取出对应数量的图片
        image_list = []
        for i in range(image_count):
            if image_index < len(image_files):
                image_path = image_files[image_index]
                image_list.append({
                    'filename': os.path.basename(image_path),
                    'path': image_path,
                    'image_index': i
                })
                image_index += 1
        
        # 创建匹配项
        matched_item = {
            'image_id': data_id,
            'images': image_list,
            'data': row_csv_data,
        }
        matched_data.append(matched_item)
        
        if (data_id + 1) % 10 == 0:
            print(f"  已匹配 {data_id + 1} 个数据项，使用了 {image_index} 张图片...")
    
    print(f"匹配完成: {len(matched_data)} 个数据项，共使用 {image_index} 张图片")
    return matched_data


def copy_images_to_dataset(matched_data: List[Dict], output_dir: str):
    """
    将图片复制到数据集目录
    
    Args:
        matched_data: 匹配后的数据
        output_dir: 输出目录
    """
    images_dir = os.path.join(output_dir, "images")
    
    print("正在复制图片文件...")
    total_images = 0
    
    for item in matched_data:
        dataset_image_paths = []
        
        # 处理图片列表
        for img_idx, image_info in enumerate(item['images']):
            src_path = image_info['path']
            filename = f"{item['image_id']:06d}_{img_idx:02d}_{image_info['filename']}"
            dst_path = os.path.join(images_dir, filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                image_info['dataset_path'] = f"images/{filename}"
                dataset_image_paths.append(f"images/{filename}")
                total_images += 1
            except Exception as e:
                print(f"  ❌ 复制图片失败 {src_path}: {e}")
                image_info['dataset_path'] = None
        
        # 保存所有图片路径
        item['all_dataset_image_paths'] = dataset_image_paths
    
    print(f"图片复制完成: {total_images} 张图片，来自 {len(matched_data)} 个数据项")


def create_metadata_csv(matched_data: List[Dict], output_dir: str):
    """
    创建元数据CSV文件
    
    Args:
        matched_data: 匹配后的数据
        output_dir: 输出目录
    """
    metadata_rows = []
    
    for item in matched_data:
        row = {}
        
        # 先添加CSV数据的所有列
        for key, value in item['data'].items():
            row[key] = value
        
        # 后添加图片数量和图片路径列，放在最后
        row['image_count'] = len(item['images'])
        row['all_image_paths'] = json.dumps(item.get('all_dataset_image_paths', []))
        
        metadata_rows.append(row)
    
    # 创建DataFrame并保存
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False, encoding='utf-8')
    
    print(f"元数据CSV创建完成: {metadata_path}")
    print(f"包含列: {list(metadata_df.columns)}")
    return metadata_path


def create_dataset_info(matched_data: List[Dict], output_dir: str, metadata_path: str):
    """
    创建数据集信息文件
    
    Args:
        matched_data: 匹配后的数据
        output_dir: 输出目录
        metadata_path: 元数据文件路径
    """
    # 统计信息
    total_items = len(matched_data)
    
    # 获取图片尺寸信息
    image_sizes = []
    images_checked = 0
    for item in matched_data:
        if images_checked >= 10:
            break
        for image_info in item['images']:
            if images_checked >= 10:
                break
            try:
                img_path = os.path.join(output_dir, image_info.get('dataset_path', ''))
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        image_sizes.append(img.size)
                    images_checked += 1
            except:
                continue
    
    dataset_info = {
        "dataset_name": "Custom Image-Text Dataset",
        "description": "由图片和CSV数据按顺序匹配生成的数据集",
        "total_samples": total_items,
        "data_format": "image-text pairs",
        "image_format": "various (jpg, png, etc.)",
        "metadata_file": "metadata.csv",
        "images_directory": "images/",
        "sample_image_sizes": image_sizes[:5] if image_sizes else [],
        "created_by": "get_datasets.py script",
        "version": "1.0"
    }
    
    # 保存数据集信息
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"数据集信息文件创建完成: {info_path}")
    return info_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="整理图片和CSV数据为Hugging Face数据集格式")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("-i", "--input", 
                       default=os.path.join(script_dir, "input"),
                       help="输入目录路径 (默认: 脚本同级input目录)")
    parser.add_argument("-o", "--output", 
                       default=os.path.join(script_dir, "dataset"),
                       help="输出数据集目录 (默认: 脚本同级dataset目录)")
    parser.add_argument("--preview", action="store_true",
                       help="仅预览匹配结果，不创建数据集")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    print("=" * 80)
    print("Hugging Face 数据集整理工具")
    print("=" * 80)
    print(f"脚本目录: {script_dir}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"预览模式: {'是' if args.preview else '否'}")
    print()
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        print("请确保input目录存在并包含图片和CSV文件")
        sys.exit(1)
    
    # 扫描图片文件
    print("扫描图片文件...")
    image_files = scan_images(input_dir)
    print(f"发现 {len(image_files)} 张图片")
    
    # 扫描CSV文件
    print("\n扫描CSV文件...")
    csv_files = scan_csv_files(input_dir)
    print(f"发现 {len(csv_files)} 个CSV文件")
    
    if not image_files:
        print("❌ 未找到任何图片文件")
        # sys.exit(1)
    
    if not csv_files:
        print("❌ 未找到任何CSV文件")
        sys.exit(1)
    
    # 加载CSV数据
    print("\n加载CSV数据...")
    csv_data = load_csv_data(csv_files)
    
    if csv_data.empty:
        print("❌ 未能加载任何CSV数据")
        sys.exit(1)
    
    # 匹配图片与数据
    print("\n匹配图片与数据...")
    matched_data = match_images_with_data(image_files, csv_data)
    
    if not matched_data:
        print("❌ 未能匹配任何数据")
        sys.exit(1)
    
    # 显示预览
    print("\n数据匹配预览:")
    print("-" * 50)
    for i, item in enumerate(matched_data[:3]):
        print(f"项目 {i+1}:")
        print(f"  图片数量: {len(item['images'])}")
        for img_idx, img_info in enumerate(item['images']):
            print(f"    图片{img_idx}: {img_info['filename']}")
        print(f"  数据: {dict(list(item['data'].items())[:3])}...")
        print()
    
    if args.preview:
        print("预览完成，退出。")
        return
    
    # 创建数据集
    print("创建数据集...")
    create_dataset_structure(output_dir)
    
    # 复制图片
    copy_images_to_dataset(matched_data, output_dir)
    
    # 创建元数据文件
    metadata_path = create_metadata_csv(matched_data, output_dir)
    
    # 创建数据集信息
    create_dataset_info(matched_data, output_dir, metadata_path)
    
    print("\n" + "=" * 80)
    print("✅ 数据集创建完成！")
    print("=" * 80)
    print(f"数据集位置: {output_dir}")
    print(f"总样本数: {len(matched_data)}")
    print(f"元数据文件: {metadata_path}")
    print("\n可以使用以下命令查看数据集:")
    print(f"  cd {output_dir}")
    print(f"  ls -la")
    print(f"  head -5 metadata.csv")


if __name__ == "__main__":
    main()
