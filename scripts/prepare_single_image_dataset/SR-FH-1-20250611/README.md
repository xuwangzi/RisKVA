# 风险数据集准备脚本

RisKVA 项目的数据预处理工具，用于将 Excel 文件转换为标准的机器学习数据集，同时从Excel中提取文本数据和图片。

## 📁 文件说明

```
prepare_risk_dataset/
├── create_dataset_from_files.py  # 🌟 主脚本 - 一键生成数据集
├── get_text_image.py             # Excel文件转换模块（提取文本和图片）
├── get_datasets.py               # 数据集整理和格式化
├── validate_dataset.py           # 数据集验证工具
├── load_dataset.py               # 简洁的数据集加载器
├── example_usage.py              # 使用示例
├── input/                        # 输入文件目录
└── dataset/                      # 输出数据集目录
```

## 🚀 快速使用

### 基本用法
```bash
# 进入脚本目录
cd scripts/prepare_risk_dataset

# 将Excel文件放入input/目录

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

### 分拆使用

如果需要分步骤处理或单独使用某个功能模块：

#### 1. 文件转换模块 (get_text_image.py)
```bash
# 处理当前目录的input文件夹
python get_text_image.py

# 自定义输入输出路径
python get_text_image.py -i ./原始文件目录 -o ./输出目录

# 自定义CSV和图片输出路径
python get_text_image.py --csv-dir ./csv输出 --images-dir ./图片输出
```

功能：
- Excel文件 (.xlsx) → CSV文件 (提取结构化文本数据)  
- Excel文件 (.xlsx) → 图片文件 (从Excel中提取所有图片)

#### 2. 数据集整理模块 (get_datasets.py)  
```bash
# 整理CSV和图片为标准数据集格式
python get_datasets.py -c ./csv目录 -i ./图片目录 -o ./数据集输出
```

功能：
- 匹配CSV数据与对应图片
- 生成符合机器学习框架的数据集格式
- 创建元数据文件和数据集说明

#### 3. 完整流水线 (create_dataset_from_files.py)
组合以上两个步骤的自动化处理：
```bash
python create_dataset_from_files.py -i ./input -o ./dataset
```

#### 使用场景建议
- **仅需要文件转换**：使用 `get_text_image.py`，适合只需要提取数据或图片的场景
- **已有转换结果**：直接使用 `get_datasets.py` 整理现有的CSV和图片文件  
- **一键式处理**：使用 `create_dataset_from_files.py`，适合完整的数据集制作流程
- **调试或定制**：分步骤使用各模块，便于中间结果检查和参数调整

## 📋 输入要求

- PDF 文件格式

- Excel 文件格式（由 PDF 文件转换得到）

## 📊 输出格式

生成符合 Hugging Face 标准的数据集：
```
dataset/
├── metadata.csv        # 元数据文件
├── images/            # 图片目录
├── dataset_info.json # 数据集信息
└── README.md          # 数据集说明
```

## 🔧 脚本功能

- **`create_dataset_from_files.py`**: 完整的转换流程，从原始文件到数据集
- **`get_text_image.py`**: 文件转换功能，Excel→CSV和图片提取
- **`get_datasets.py`**: 数据集整理，图片与文本数据匹配
- **`validate_dataset.py`**: 验证生成数据集的完整性
- **`load_dataset.py`**: 快速加载数据集样本，返回PIL图片和关键字段
- **`example_usage.py`**: 各种框架的使用示例