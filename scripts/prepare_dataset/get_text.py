#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量Excel文件转换脚本
将指定目录中的xlsx文件转换为CSV格式
"""

import pandas as pd  # type: ignore
import os
import sys
from pathlib import Path
import numpy as np
import re
import argparse
from typing import List, Tuple
import glob


def clean_text(text):
    """
    清理文本，去除所有空格、缩进、换行符等空白字符
    
    参数:
        text (str): 需要清理的文本
    
    返回:
        str: 清理后的文本
    """
    if pd.isna(text) or text == 'nan':
        return ""
    
    # 转换为字符串
    text = str(text)
    
    # 去除所有类型的空白字符（空格、制表符、换行符等）
    text = re.sub(r'\s+', '', text)
    
    # 去除 'nan' 字符串
    text = text.replace('nan', '')
    
    return text.strip()


def process_excel_data(df, file_name):
    """
    处理Excel数据，合并相关行并重新排列列顺序
    
    参数:
        df (DataFrame): 原始Excel数据
        file_name (str): 输入文件的文件名（不含扩展名）
    
    返回:
        DataFrame: 处理后的数据
    """
    processed_data = []
    
    # 从第2行开始处理数据（索引2，跳过标题行和子标题行）
    i = 2
    
    while i < len(df):
        row = df.iloc[i]
        
        # 检查是否是有效的序号行（不为空且不是NaN）
        序号_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else "" # 文件名、序号
        if 序号_cell:
            序号 = clean_text(row.iloc[0])
            风险 = clean_text(row.iloc[2])
            纠正和预防建议 = clean_text(row.iloc[3])
            风险等级 = clean_text(row.iloc[4])
            
            # 根据Excel结构分析："原"字段是风险等级的原始状态，"现"是整改后的当前状态
            # 从数据看，第4列是风险等级，第5列是当前状态
            # "原"字段设为风险等级，表示发现问题时的原始风险级别
            原 = 风险等级  # 原始风险等级
            现 = clean_text(row.iloc[5])
            
            # 获取缺陷描述（通常在下一行）
            缺陷描述 = ""
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                if pd.notna(next_row.iloc[1]):  # 缺陷描述在第二列
                    缺陷描述 = clean_text(next_row.iloc[1])
            
            processed_data.append({
                'file_id': file_name, # 使用输入文件的文件名
                'defect_index': 序号,
                'defect_description_text': 缺陷描述,
                'risk_detail': 风险,
                'correction_suggestion': 纠正和预防建议,
                'correction_status': "已整改",
                # '风险等级': 风险等级,
                'risk_level_original': 原,
                'risk_level_current': 现,
                'image_count': 1,
            })
            
            # 跳过下一行（缺陷描述行）
            i += 2
        else:
            i += 1
    
    return pd.DataFrame(processed_data)


def excel_to_csv(excel_path, output_dir):
    """
    将 Excel 文件转换为 CSV 格式
    
    参数:
        excel_path (str): Excel 文件的路径
        output_dir (str): 输出目录
    
    返回:
        str: 生成的CSV文件路径，如果失败返回None
    """
    try:
        # 检查 Excel 文件是否存在
        if not os.path.exists(excel_path):
            print(f"错误：找不到文件 {excel_path}")
            return None
        
        print(f"正在处理Excel文件: {excel_path}")
        
        # 读取 Excel 文件
        df = pd.read_excel(excel_path)
        print(f"  原始数据行数: {len(df)} 行，列数: {len(df.columns)} 列")
        
        # 生成文件名
        excel_filename = Path(excel_path).stem
        
        # 处理数据
        print("  正在处理和重新整理数据...")
        processed_df = process_excel_data(df, excel_filename)
        csv_path = os.path.join(output_dir, f"{excel_filename}.csv")
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"  正在保存为 CSV 文件: {csv_path}")
        
        # 保存为 CSV 文件，使用 UTF-8 编码
        processed_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"  ✅ Excel转换成功！")
        print(f"    处理后数据行数: {len(processed_df)} 行")
        print(f"    列顺序: {', '.join(processed_df.columns)}")
        
        return csv_path
        
    except Exception as e:
        print(f"  ❌ Excel转换过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def scan_input_directory(input_dir):
    """
    扫描输入目录，找出所有xlsx文件
    
    参数:
        input_dir (str): 输入目录路径
    
    返回:
        list: xlsx文件列表
    """
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        return []
    
    # 查找xlsx文件
    xlsx_pattern = os.path.join(input_dir, "*.xlsx")
    xlsx_files = glob.glob(xlsx_pattern)
    
    return xlsx_files


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="批量转换xlsx文件为CSV")
    parser.add_argument("-i", "--input", default=os.path.join(script_dir, "input"), 
                       help="输入目录路径 (默认: 脚本同级input目录)")
    parser.add_argument("-o", "--output", default=os.path.join(script_dir, "output"), 
                       help="输出目录路径 (默认: 脚本同级output目录)")
    
    args = parser.parse_args()
    
    # 确保路径处理正确
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    print("=" * 80)
    print("Excel文件转换工具")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 扫描输入目录
    xlsx_files = scan_input_directory(input_dir)
    
    print(f"发现文件:")
    print(f"  Excel文件 (.xlsx): {len(xlsx_files)} 个")
    print()
    
    if not xlsx_files:
        print("未找到任何xlsx文件。")
        return
    
    # 处理Excel文件
    print("开始处理Excel文件...")
    print("-" * 40)
    successful_csv = 0
    for xlsx_file in xlsx_files:
        result = excel_to_csv(xlsx_file, output_dir)
        if result:
            successful_csv += 1
        print()
    
    print(f"Excel处理完成：{successful_csv}/{len(xlsx_files)} 个文件成功转换")
    print()
    
    print("=" * 80)
    print("转换完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main() 