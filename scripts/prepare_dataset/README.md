# Prepare Dataset 数据集准备工具

本目录包含用于准备和处理RisKVA数据集的脚本工具。

## 脚本功能

### 1. get_text.py
**功能**: 批量Excel文件转换脚本
- 将指定目录中的xlsx文件转换为CSV格式
- 自动处理Excel数据结构，提取缺陷信息
- 支持动态文件名识别

**使用方法**:
```bash
python3 get_text.py [-i input_dir] [-o output_dir]
```

### 2. format_csv.py
**功能**: CSV数据清理脚本
- 删除CSV数据项中的空格和换行符
- 规范化数据格式，便于后续处理

**使用方法**:
```bash
python3 format_csv.py
```

### 3. get_datasets.py
**功能**: 数据集整理脚本（主要工具）
- 将图片和CSV数据按顺序对应
- 整理为Hugging Face数据集格式
- 支持多种输出格式

**使用方法**:
```bash
python3 get_datasets.py
```

### 4. load_dataset_demo.py
**功能**: 数据集查看器
- 显示所有数据集的前5条数据
- 用于验证数据集处理结果

**使用方法**:
```bash
python3 load_dataset_demo.py
```

### 5. load_dataset_simple.py
**功能**: 简单数据集加载器
- 提供基本的数据集加载功能

### 6. get_images.txt
**功能**: PDF图片提取工具说明
- 提供从PDF文件中提取图片的命令行工具使用方法
- 使用poppler-utils工具包中的pdfimages命令

**使用方法**:
```bash
sudo apt install poppler-utils
pdfimages -all input.pdf output_prefix
```

## 使用流程

1. **Excel转CSV**: 使用 `get_text.py` 将Excel文件转换为CSV
2. **数据清理**: 使用 `format_csv.py` 清理CSV数据格式
3. **DF图片提取**: 使用 `get_images.txt` 中的命令从PDF文件提取图片
4. **数据集生成**: 使用 `get_datasets.py` 生成数据集
5. **结果验证**: 使用 `load_dataset_demo.py` 查看处理结果

- **数据集转换**: 使用 `get_hf_datasets.py` 转换数据集为hf格式

## 依赖环境

- Python 3.7+
- pandas
- PIL/Pillow
- openpyxl（用于Excel文件处理） 