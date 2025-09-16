#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集查看器 - 显示所有数据集的前5条数据
"""

import pandas as pd  # type: ignore
from pathlib import Path
import os

def find_csv_file(dataset_dir):
    """在数据集目录中查找CSV文件"""
    dataset_path = Path(dataset_dir)
    
    # 优先查找dataset/metadata.csv
    metadata_csv = dataset_path / "dataset" / "metadata.csv"
    if metadata_csv.exists():
        return metadata_csv
    
    # 查找input目录下的CSV文件
    input_dir = dataset_path / "input"
    if input_dir.exists():
        csv_files = list(input_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]  # 返回第一个找到的CSV文件
    
    return None

def show_dataset_preview(dataset_name, max_rows=3):
    """显示数据集的前几条数据"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    
    csv_file = find_csv_file(dataset_name)
    if not csv_file:
        print("❌ 未找到CSV文件")
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 数据概览: {len(df)} 行, {len(df.columns)} 列")
        print(f"📁 CSV文件: {csv_file}")
        
        print(f"\n📋 列名: {list(df.columns)}")
        
        print(f"\n🔍 前{max_rows}条数据:")
        print("-" * 60)
        
        # 显示前几条数据，限制每列的显示宽度
        preview_df = df.head(max_rows)
        
        # 对于每一行数据，格式化输出
        for idx, row in preview_df.iterrows():
            print(f"\n第{idx+1}条:")
            for col in df.columns:
                value = str(row[col])
                # 限制显示长度，避免输出过长
                if len(value) > 100:
                    value = value[:97] + "..."
                print(f"  {col}: {value}")
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")

def show_dataset_tail(dataset_name, max_rows=3):
    """显示数据集的后几条数据"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name} (后{max_rows}条)")
    print(f"{'='*60}")
    
    csv_file = find_csv_file(dataset_name)
    if not csv_file:
        print("❌ 未找到CSV文件")
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 数据概览: {len(df)} 行, {len(df.columns)} 列")
        print(f"📁 CSV文件: {csv_file}")
        
        print(f"\n📋 列名: {list(df.columns)}")
        
        print(f"\n🔍 后{max_rows}条数据:")
        print("-" * 60)
        
        # 显示后几条数据，限制每列的显示宽度
        tail_df = df.tail(max_rows)
        
        # 对于每一行数据，格式化输出
        for idx, row in tail_df.iterrows():
            print(f"\n第{idx+1}条 (倒数第{len(df)-idx}条):")
            for col in df.columns:
                value = str(row[col])
                # 限制显示长度，避免输出过长
                if len(value) > 100:
                    value = value[:97] + "..."
                print(f"  {col}: {value}")
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")

def main():
    """主函数 - 显示所有数据集的预览"""
    # 要查看的数据集目录
    datasets = [
        "PR-9-20250512_with_image",
        "PR-9-20250512_without_image", 
        "SR-FH-1-20250611_with_image",
        "SR-04-20241125_without_image",
        "SR-FH-1-20250706_without_image"
    ]
    
    print("🔍 数据集预览工具")
    print(f"📁 工作目录: {Path.cwd()}")
    print("\n选择查看模式:")
    print("1. 查看前3条数据")
    print("2. 查看后3条数据")
    print("3. 查看前后各3条数据")
    
    try:
        choice = input("\n请输入选择 (1/2/3，默认为1): ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"  # 如果在脚本中运行，默认选择1
    
    for dataset in datasets:
        if Path(dataset).exists():
            if choice == "1":
                show_dataset_preview(dataset)
            elif choice == "2":
                show_dataset_tail(dataset)
            elif choice == "3":
                show_dataset_preview(dataset)
                show_dataset_tail(dataset)
            else:
                print(f"无效选择，使用默认模式")
                show_dataset_preview(dataset)
        else:
            print(f"\n{'='*60}")
            print(f"数据集: {dataset}")
            print(f"{'='*60}")
            print("❌ 目录不存在")

if __name__ == "__main__":
    main() 