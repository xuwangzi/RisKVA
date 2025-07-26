#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简RisKVA数据集加载器 - 获取PIL图片和关键字段
"""

import pandas as pd
from datasets import Dataset, Image as HFImage
from pathlib import Path
from PIL import Image as PILImage

def load_sample(dataset_path="datasets/sft_data/single_image_tiny_247", index=0):
    """加载数据集样本，返回PIL图片和关键字段"""
    # 构建数据集路径
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path / "metadata.csv")
    
    # 修正图片路径为相对于数据集目录
    df['image_path'] = df['image_path'].apply(lambda p: str(dataset_path / p))
    
    # 创建数据集并获取样本
    sample = Dataset.from_pandas(df).cast_column("image_path", HFImage())[index]
    
    # 返回图片和关键字段
    return sample['image_path'], {
        'defect_index': sample.get('defect_index', ''),
        'defect_description_text': sample.get('defect_description_text', ''),
        'risk_detail': sample.get('risk_detail', ''),
        'correction_suggestion': sample.get('correction_suggestion', ''),
        'correction_status': sample.get('correction_status', ''),
        'risk_level_original': sample.get('risk_level_original', ''),
        'risk_level_current': sample.get('risk_level_current', '')
    }

if __name__ == "__main__":
    try:
        image, data = load_sample()
        print("✅ 加载成功")
        print(f"图片: {type(image).__name__} {image.size} {image.mode}")
        print(f"序号: {data['defect_index']} | 风险: {data['risk_level_original']} | 状态: {data['correction_status']}")
        # image.show()  # 取消注释显示图片
    except Exception as e:
        print(f"❌ 失败: {e}") 