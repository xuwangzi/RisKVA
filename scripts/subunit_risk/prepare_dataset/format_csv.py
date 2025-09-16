#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据清理脚本 - 删除数据项中的空格和换行符
"""

import pandas as pd  # type: ignore
import re
from pathlib import Path

def clean_csv_data(input_file, output_file=None):
    """
    清理CSV文件中的空格和换行符
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径，默认为输入文件名加_cleaned后缀
    
    返回:
        bool: 成功返回True，失败返回False
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
        
        # 对所有字符串列进行清理
        for col in df.columns:
            if df[col].dtype == 'object':  # 字符串类型
                df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\s+', '', x) if x != 'nan' else '')
        
        # 生成输出文件名
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        
        # 保存清理后的数据
        print(f"正在保存到: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ 清理完成! 处理了 {len(df)} 行数据")
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

def main():
    """主函数"""
    script_dir = Path(__file__).parent
    default_input = script_dir / "output"
    
    print("CSV数据清理工具")
    print("=" * 40)
    
    # 查找output目录中的CSV文件
    csv_files = list(default_input.glob("*.csv"))
    
    if not csv_files:
        print(f"在 {default_input} 目录中未找到CSV文件")
        return
    
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file.name}")
        clean_csv_data(str(csv_file))

if __name__ == "__main__":
    main()
