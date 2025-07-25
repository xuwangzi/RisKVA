# 风险数据集准备脚本

RisKVA 项目的数据预处理工具，用于将 PDF 和 Excel 文件转换为标准的机器学习数据集。

## 📁 文件说明

```
prepare_risk_dataset/
├── create_dataset_from_files.py  # 🌟 主脚本 - 一键生成数据集
├── get_text_image.py             # PDF和Excel文件转换模块
├── get_datasets.py               # 数据集整理和格式化
├── validate_dataset.py           # 数据集验证工具
├── example_usage.py              # 使用示例
├── input/                        # 输入文件目录
└── dataset/                      # 输出数据集目录
```

## 🚀 快速使用

### 基本用法
```bash
# 进入脚本目录
cd scripts/prepare_risk_dataset

# 将PDF和Excel文件放入input/目录

# 运行转换
python create_dataset_from_files.py

# 验证结果
python validate_dataset.py dataset/
```

### 自定义选项
```bash
# 自定义路径
python create_dataset_from_files.py -i ./my_input -o ./my_output

# 预览模式（不生成最终数据集）
python create_dataset_from_files.py --preview

# 保留临时文件用于调试
python create_dataset_from_files.py --keep-temp
```

## 📋 输入要求

### Excel 文件格式
需要包含以下列：
- 序号 (SR-FH-xxx 格式)
- 风险描述
- 纠正和预防建议  
- 风险等级
- 缺陷描述

### PDF 文件
- 包含与Excel数据对应的图片
- 按页面顺序排列

## 📊 输出格式

生成符合 Hugging Face 标准的数据集：
```
dataset/
├── metadata.csv        # 元数据文件
├── images/            # 图片目录
├── dataset_info.json # 数据集信息
└── README.md          # 数据集说明
```

## 💻 使用示例

### 读取数据集
```python
import pandas as pd
from PIL import Image

# 读取元数据
df = pd.read_csv('dataset/metadata.csv')

# 加载图片和对应数据
sample = df.iloc[0]
image = Image.open(f"dataset/{sample['image_path']}")
risk_level = sample['data_风险等级']
```

### PyTorch 集成
```python
from torch.utils.data import Dataset

class RiskDataset(Dataset):
    def __init__(self, metadata_path, dataset_dir):
        self.df = pd.read_csv(metadata_path)
        self.dataset_dir = dataset_dir
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.dataset_dir}/{row['image_path']}")
        return image, row['data_风险等级']
```

## 🔧 脚本功能

- **`create_dataset_from_files.py`**: 完整的转换流程，从原始文件到数据集
- **`get_text_image.py`**: 文件转换功能，PDF→图片，Excel→CSV
- **`get_datasets.py`**: 数据集整理，图片与文本数据匹配
- **`validate_dataset.py`**: 验证生成数据集的完整性
- **`example_usage.py`**: 各种框架的使用示例

## ⚠️ 注意事项

1. 确保 input/ 目录中有对应的 PDF 和 Excel 文件
2. 图片与数据按文件名顺序自动匹配
3. 生成的数据集兼容 PyTorch、TensorFlow 和 Hugging Face
4. 如果遇到问题，可使用 `--preview` 选项检查转换结果

## 📈 性能说明

- 支持多页 PDF 文件
- 自动处理 Excel 复杂格式
- 生成标准化的数据集格式
- 内存优化的流式处理 