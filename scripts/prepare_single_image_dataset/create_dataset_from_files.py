#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整数据集创建脚本
从PDF和XLSX文件一键生成Hugging Face数据集

工作流程：
1. 扫描输入目录中的PDF和XLSX文件
2. 将XLSX转换为CSV，PDF提取图片 (使用get_text_image.py功能)
3. 将图片和CSV数据整理为标准数据集格式 (使用get_datasets.py功能)
"""

import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path

# 导入需要的模块功能
try:
    # 导入 get_text_image.py 的功能
    from get_text_image import (
        excel_to_csv, 
        extract_images_from_pdf, 
        scan_input_directory as scan_files,
        clean_text,
        process_excel_data
    )
    
    # 导入 get_datasets.py 的功能  
    from get_datasets import (
        scan_images,
        scan_csv_files,
        load_csv_data,
        match_images_with_data,
        copy_images_to_dataset,
        create_metadata_csv,
        create_dataset_info,
        create_readme,
        create_dataset_structure
    )
    
    print("✅ 成功导入所需模块")
    
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保 get_text_image.py 和 get_datasets.py 在同一目录下")
    sys.exit(1)


def get_script_directory():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))


def step1_convert_files(input_dir: str, temp_dir: str) -> tuple:
    """
    步骤1: 转换原始文件
    将PDF提取图片，XLSX转换为CSV
    
    Args:
        input_dir: 输入目录
        temp_dir: 临时工作目录
        
    Returns:
        tuple: (成功转换的文件数, 转换后的文件路径)
    """
    print("=" * 60)
    print("📁 步骤1: 转换原始文件")
    print("=" * 60)
    
    # 创建临时目录结构
    temp_csv_dir = os.path.join(temp_dir, "csv")
    temp_images_dir = os.path.join(temp_dir, "images")
    os.makedirs(temp_csv_dir, exist_ok=True)
    os.makedirs(temp_images_dir, exist_ok=True)
    
    # 扫描输入文件
    xlsx_files, pdf_files = scan_files(input_dir)
    
    print(f"发现文件:")
    print(f"  📊 Excel文件: {len(xlsx_files)} 个")
    print(f"  📄 PDF文件: {len(pdf_files)} 个")
    print()
    
    if not xlsx_files and not pdf_files:
        print("❌ 未找到任何XLSX或PDF文件")
        return 0, None
    
    successful_conversions = 0
    
    # 处理Excel文件转CSV
    if xlsx_files:
        print("🔄 转换Excel文件...")
        for xlsx_file in xlsx_files:
            result = excel_to_csv(xlsx_file, temp_csv_dir)
            if result:
                successful_conversions += 1
                print(f"  ✅ {os.path.basename(xlsx_file)} -> CSV")
            else:
                print(f"  ❌ {os.path.basename(xlsx_file)} 转换失败")
        print()
    
    # 处理PDF文件提取图片
    if pdf_files:
        print("🖼️  提取PDF图片...")
        for pdf_file in pdf_files:
            images = extract_images_from_pdf(pdf_file, temp_images_dir)
            if images:
                successful_conversions += 1
                print(f"  ✅ {os.path.basename(pdf_file)} -> {len(images)} 张图片")
            else:
                print(f"  ❌ {os.path.basename(pdf_file)} 处理失败")
        print()
    
    print(f"📊 转换统计:")
    print(f"  成功处理: {successful_conversions}/{len(xlsx_files) + len(pdf_files)} 个文件")
    print(f"  CSV文件目录: {temp_csv_dir}")
    print(f"  图片文件目录: {temp_images_dir}")
    
    return successful_conversions, temp_dir


def step2_create_dataset(temp_dir: str, output_dir: str) -> bool:
    """
    步骤2: 创建数据集
    将转换后的图片和CSV整理为标准数据集格式
    
    Args:
        temp_dir: 临时目录（包含转换后的文件）
        output_dir: 最终数据集输出目录
        
    Returns:
        bool: 是否成功创建数据集
    """
    print("\n" + "=" * 60)
    print("🎯 步骤2: 创建标准数据集")
    print("=" * 60)
    
    # 扫描转换后的文件
    temp_csv_dir = os.path.join(temp_dir, "csv")
    temp_images_dir = os.path.join(temp_dir, "images")
    
    # 扫描图片文件
    print("🔍 扫描转换后的文件...")
    image_files = scan_images(temp_images_dir)
    csv_files = scan_csv_files(temp_csv_dir)
    
    print(f"  📷 找到图片: {len(image_files)} 张")
    print(f"  📊 找到CSV: {len(csv_files)} 个")
    
    if not image_files or not csv_files:
        print("❌ 转换后的文件不足，无法创建数据集")
        return False
    
    # 加载CSV数据
    print("\n📖 加载CSV数据...")
    csv_data = load_csv_data(csv_files)
    
    if csv_data.empty:
        print("❌ 无法加载CSV数据")
        return False
    
    # 匹配图片与数据
    print("\n🔗 匹配图片与数据...")
    matched_data = match_images_with_data(image_files, csv_data)
    
    if not matched_data:
        print("❌ 无法匹配图片与数据")
        return False
    
    # 创建数据集结构
    print(f"\n🏗️  创建数据集...")
    create_dataset_structure(output_dir)
    
    # 复制图片到数据集
    copy_images_to_dataset(matched_data, output_dir)
    
    # 创建元数据文件
    metadata_path = create_metadata_csv(matched_data, output_dir)
    
    # 创建数据集信息
    create_dataset_info(matched_data, output_dir, metadata_path)
    
    # 创建README
    create_readme(output_dir, len(matched_data))
    
    print(f"\n📊 数据集统计:")
    print(f"  总样本数: {len(matched_data)}")
    print(f"  元数据文件: {metadata_path}")
    print(f"  数据集位置: {output_dir}")
    
    return True


def cleanup_temp_files(temp_dir: str, keep_temp: bool = False):
    """清理临时文件"""
    if not keep_temp and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 已清理临时文件: {temp_dir}")
        except Exception as e:
            print(f"⚠️  清理临时文件失败: {e}")
    elif keep_temp:
        print(f"📁 保留临时文件: {temp_dir}")


def validate_inputs(input_dir: str) -> bool:
    """验证输入参数"""
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 检查是否有支持的文件
    xlsx_files, pdf_files = scan_files(input_dir)
    if not xlsx_files and not pdf_files:
        print(f"❌ 输入目录中没有找到XLSX或PDF文件: {input_dir}")
        return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从PDF和XLSX文件创建Hugging Face数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                           # 使用默认路径
  %(prog)s -i ./source -o ./my_dataset    # 自定义路径
  %(prog)s --keep-temp               # 保留临时文件用于调试
  %(prog)s --preview                 # 仅预览不创建
        """
    )
    
    # 获取脚本目录
    script_dir = get_script_directory()
    
    parser.add_argument("-i", "--input", 
                       default=os.path.join(script_dir, "input"),
                       help="输入目录路径 (默认: 脚本同级input目录)")
    parser.add_argument("-o", "--output", 
                       default=os.path.join(script_dir, "dataset"),
                       help="输出数据集目录 (默认: 脚本同级dataset目录)")
    parser.add_argument("--temp-dir", 
                       help="临时工作目录 (默认: 系统临时目录)")
    parser.add_argument("--keep-temp", action="store_true",
                       help="保留临时文件(用于调试)")
    parser.add_argument("--preview", action="store_true",
                       help="仅预览转换结果，不创建数据集")
    
    args = parser.parse_args()
    
    # 路径处理
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    if args.temp_dir:
        temp_dir = os.path.abspath(args.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="dataset_creation_")
    
    print("=" * 80)
    print("🚀 一键数据集创建工具")
    print("=" * 80)
    print(f"脚本目录: {script_dir}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"临时目录: {temp_dir}")
    print(f"预览模式: {'是' if args.preview else '否'}")
    print()
    
    try:
        # 验证输入
        if not validate_inputs(input_dir):
            return 1
        
        # 步骤1: 转换文件
        success_count, temp_result = step1_convert_files(input_dir, temp_dir)
        
        if success_count == 0:
            print("❌ 没有成功转换任何文件")
            return 1
        
        if args.preview:
            print("\n👀 预览模式 - 转换完成，不创建最终数据集")
            print(f"转换后的文件保存在: {temp_dir}")
            print("可以检查转换结果，然后使用 --keep-temp 选项重新运行")
            return 0
        
        # 步骤2: 创建数据集
        dataset_success = step2_create_dataset(temp_dir, output_dir)
        
        if not dataset_success:
            print("❌ 数据集创建失败")
            return 1
        
        print("\n" + "=" * 80)
        print("🎉 数据集创建完成！")
        print("=" * 80)
        print(f"📁 数据集位置: {output_dir}")
        print(f"📋 包含文件:")
        print(f"  • metadata.csv    - 元数据文件")
        print(f"  • images/         - 图片目录")
        print(f"  • dataset_info.json - 数据集信息")
        print(f"  • README.md       - 使用说明")
        print()
        print("💡 使用建议:")
        print(f"  cd {output_dir}")
        print(f"  head -3 metadata.csv")
        print(f"  ls images/ | head -5")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 清理临时文件
        cleanup_temp_files(temp_dir, args.keep_temp)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 