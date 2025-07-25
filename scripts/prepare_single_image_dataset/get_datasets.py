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


def get_script_directory():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))


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
            df['source_csv'] = os.path.basename(csv_file)
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
    
    # 按顺序匹配
    for i, image_path in enumerate(image_files):
        if i < len(csv_data):
            # 获取对应行的数据
            row_data = csv_data.iloc[i].to_dict()
            
            # 创建匹配项
            matched_item = {
                'image_id': i,
                'image_filename': os.path.basename(image_path),
                'image_path': image_path,
                'data': row_data
            }
            matched_data.append(matched_item)
            
            if (i + 1) % 10 == 0:
                print(f"  已匹配 {i + 1} 个项目...")
        else:
            print(f"  警告: 图片 {os.path.basename(image_path)} 没有对应的CSV数据行")
    
    if len(csv_data) > len(image_files):
        print(f"  警告: 有 {len(csv_data) - len(image_files)} 行CSV数据没有对应的图片")
    
    print(f"匹配完成: {len(matched_data)} 个有效数据项")
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
    for item in matched_data:
        src_path = item['image_path']
        # 使用统一的命名格式
        filename = f"{item['image_id']:06d}_{item['image_filename']}"
        dst_path = os.path.join(images_dir, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            # 更新数据中的图片路径
            item['dataset_image_path'] = f"images/{filename}"
        except Exception as e:
            print(f"  ❌ 复制图片失败 {src_path}: {e}")
    
    print(f"图片复制完成: {len(matched_data)} 张图片")


def create_metadata_csv(matched_data: List[Dict], output_dir: str):
    """
    创建元数据CSV文件
    
    Args:
        matched_data: 匹配后的数据
        output_dir: 输出目录
    """
    metadata_rows = []
    
    for item in matched_data:
        row = {
            'image_id': item['image_id'],
            'image_path': item['dataset_image_path'],
            'original_filename': item['image_filename'],
        }
        
        # 添加CSV数据的所有列
        for key, value in item['data'].items():
            if key != 'source_csv':  # 避免重复
                row[f'data_{key}'] = value
        
        # 添加源文件信息
        row['source_csv'] = item['data'].get('source_csv', '')
        
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
    for item in matched_data[:10]:  # 只检查前10张图片作为样本
        try:
            img_path = os.path.join(output_dir, item['dataset_image_path'])
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
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


def create_readme(output_dir: str, total_items: int):
    """
    创建README文件
    
    Args:
        output_dir: 输出目录
        total_items: 数据项总数
    """
    readme_content = f"""# 图像-文本数据集

这是一个由图片和CSV数据按顺序匹配生成的数据集。

## 数据集结构

```
dataset/
├── images/           # 图片文件目录
├── data/            # 额外数据文件目录
├── metadata.csv     # 元数据文件
├── dataset_info.json # 数据集信息
└── README.md        # 说明文件
```

## 基本信息

- **总样本数**: {total_items}
- **数据格式**: 图像-文本对
- **图片格式**: 多种格式 (JPG, PNG等)

## 使用方法

### 使用pandas读取

```python
import pandas as pd
from PIL import Image
import os

# 读取元数据
df = pd.read_csv('metadata.csv')

# 读取第一个样本
row = df.iloc[0]
image_path = row['image_path']
image = Image.open(image_path)

print("图片信息:", image.size)
print("数据信息:", row)
```

### 使用Hugging Face datasets

```python
from datasets import Dataset, Image as HFImage
import pandas as pd

# 读取元数据
df = pd.read_csv('metadata.csv')

# 创建数据集
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column('image_path', HFImage())

print(dataset[0])
```

## 数据格式

每行数据包含：
- `image_id`: 图片ID
- `image_path`: 图片在数据集中的路径
- `original_filename`: 原始文件名
- `data_*`: CSV中的各个数据列
- `source_csv`: 数据来源的CSV文件名

## 注意事项

1. 图片和数据是按照文件的自然排序顺序进行匹配的
2. 如果图片数量与数据行数不匹配，会在控制台输出警告信息
3. 所有图片都被重命名为统一格式: `XXXXXX_原始文件名`

---
由 get_datasets.py 脚本自动生成
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README文件创建完成: {readme_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="整理图片和CSV数据为Hugging Face数据集格式")
    
    # 获取脚本所在目录
    script_dir = get_script_directory()
    
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
        sys.exit(1)
    
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
        print(f"  图片: {item['image_filename']}")
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
    
    # 创建README
    create_readme(output_dir, len(matched_data))
    
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
