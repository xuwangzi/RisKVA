#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量文件转换脚本
将指定目录中的xlsx文件转换为CSV格式，pdf文件提取图片
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

# 尝试导入PDF处理库
try:
    import fitz  # type: ignore # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("警告：未安装PyMuPDF库，PDF图片提取功能将不可用")
    print("如需PDF功能，请安装：pip install PyMuPDF")


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


def process_excel_data(df):
    """
    处理Excel数据，合并相关行并重新排列列顺序
    
    参数:
        df (DataFrame): 原始Excel数据
    
    返回:
        DataFrame: 处理后的数据
    """
    processed_data = []
    
    # 从第2行开始处理数据（索引2，跳过标题行和子标题行）
    i = 2
    
    while i < len(df):
        row = df.iloc[i]
        
        # 检查是否是有效的序号行（不为空且不是NaN）
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).startswith('SR-'):
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
                'file_id': "SR-FH-1-20250611",
                'defect_index': 序号,
                'defect_description_text': 缺陷描述,
                'risk_detail': 风险,
                'correction_suggestion': 纠正和预防建议,
                'correction_status': "已整改",
                # '风险等级': 风险等级,
                'risk_level_original': 原,
                'risk_level_current': 现,
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
        
        # 处理数据
        print("  正在处理和重新整理数据...")
        processed_df = process_excel_data(df)
        
        # 生成输出文件路径
        excel_filename = Path(excel_path).stem
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


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """
    从PDF文件中按顺序提取所有图片
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录路径
        
    Returns:
        保存的图片文件路径列表
    """
    if not PDF_SUPPORT:
        print(f"  ❌ PDF处理功能不可用，跳过文件: {pdf_path}")
        return []
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开PDF文件
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ❌ 无法打开PDF文件 {pdf_path}: {e}")
        return []
    
    extracted_images = []
    image_count = 0
    
    pdf_filename = Path(pdf_path).stem
    print(f"正在处理PDF文件: {pdf_path}")
    print(f"  总页数：{len(doc)}")
    
    # 遍历每一页
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 获取页面中的图片列表
        image_list = page.get_images()
        
        if image_list:
            print(f"  处理第 {page_num + 1} 页，发现 {len(image_list)} 张图片...")
        
        # 提取每个图片
        for img_index, img in enumerate(image_list):
            # 获取图片引用
            xref = img[0]
            
            try:
                # 提取图片数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # 生成文件名（保持顺序）
                image_count += 1
                filename = f"{pdf_filename}_image_{image_count:04d}_page{page_num + 1:03d}.{image_ext}"
                image_path = os.path.join(output_dir, filename)
                
                # 保存图片
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                extracted_images.append(image_path)
                print(f"    保存图片：{filename} ({len(image_bytes)} bytes)")
                
            except Exception as e:
                print(f"    警告：提取图片失败 (页面{page_num + 1}, 图片{img_index + 1}): {e}")
                continue
    
    doc.close()
    print(f"  ✅ PDF处理完成！共提取 {len(extracted_images)} 张图片")
    return extracted_images


def scan_input_directory(input_dir):
    """
    扫描输入目录，找出所有xlsx和pdf文件
    
    参数:
        input_dir (str): 输入目录路径
    
    返回:
        tuple: (xlsx文件列表, pdf文件列表)
    """
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        return [], []
    
    # 查找xlsx文件
    xlsx_pattern = os.path.join(input_dir, "*.xlsx")
    xlsx_files = glob.glob(xlsx_pattern)
    
    # 查找pdf文件
    pdf_pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    return xlsx_files, pdf_files


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="批量转换xlsx和pdf文件")
    parser.add_argument("-i", "--input", default=os.path.join(script_dir, "input"), 
                       help="输入目录路径 (默认: 脚本同级input目录)")
    parser.add_argument("-o", "--output", default=os.path.join(script_dir, "output"), 
                       help="输出目录路径 (默认: 脚本同级output目录)")
    parser.add_argument("--csv-dir", default=os.path.join(script_dir, "output/csv"), 
                       help="CSV文件输出子目录名 (默认: 脚本同级output目录)")
    parser.add_argument("--images-dir", default=os.path.join(script_dir, "output/images"), 
                       help="图片文件输出子目录名 (默认: 脚本同级output目录)")
    
    args = parser.parse_args()
    
    # 确保路径处理正确
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    csv_output_dir = os.path.join(output_dir, args.csv_dir)
    images_output_dir = os.path.join(output_dir, args.images_dir)
    
    print("=" * 80)
    print("批量文件转换工具")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"CSV输出: {csv_output_dir}")
    print(f"图片输出: {images_output_dir}")
    print()
    
    # 扫描输入目录
    xlsx_files, pdf_files = scan_input_directory(input_dir)
    
    print(f"发现文件:")
    print(f"  Excel文件 (.xlsx): {len(xlsx_files)} 个")
    print(f"  PDF文件 (.pdf): {len(pdf_files)} 个")
    print()
    
    if not xlsx_files and not pdf_files:
        print("未找到任何xlsx或pdf文件。")
        return
    
    # 处理Excel文件
    if xlsx_files:
        print("开始处理Excel文件...")
        print("-" * 40)
        successful_csv = 0
        for xlsx_file in xlsx_files:
            result = excel_to_csv(xlsx_file, csv_output_dir)
            if result:
                successful_csv += 1
            print()
        
        print(f"Excel处理完成：{successful_csv}/{len(xlsx_files)} 个文件成功转换")
        print()
    
    # 处理PDF文件
    if pdf_files and PDF_SUPPORT:
        print("开始处理PDF文件...")
        print("-" * 40)
        total_images = 0
        successful_pdf = 0
        
        for pdf_file in pdf_files:
            images = extract_images_from_pdf(pdf_file, images_output_dir)
            if images:
                successful_pdf += 1
                total_images += len(images)
            print()
        
        print(f"PDF处理完成：{successful_pdf}/{len(pdf_files)} 个文件成功处理")
        print(f"总共提取图片：{total_images} 张")
        print()
    elif pdf_files and not PDF_SUPPORT:
        print("跳过PDF处理：缺少PyMuPDF库")
        print()
    
    print("=" * 80)
    print("批量转换完成！")
    print(f"输出目录: {output_dir}")
    if xlsx_files:
        print(f"CSV文件: {csv_output_dir}")
    if pdf_files and PDF_SUPPORT:
        print(f"图片文件: {images_output_dir}")


if __name__ == "__main__":
    main() 