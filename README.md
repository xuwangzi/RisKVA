# Qwen2.5-VL 建筑风险识别 SFT 训练框架

基于Qwen2.5-VL多模态大语言模型的建筑缺陷风险识别系统，支持从图片自动识别建筑缺陷并提供专业的风险评估和修复建议。

## 🚀 特性

- **多模态理解**: 基于Qwen2.5-VL，同时理解图像和文本
- **专业分析**: 针对建筑工程缺陷进行专门优化
- **高效训练**: 支持LoRA微调，显存友好
- **分布式训练**: 支持多GPU并行训练
- **易于部署**: 完整的推理接口，支持批量处理
- **数据格式多样**: 支持JSONL和Parquet两种数据格式
- **数据完整性**: Parquet格式将图片内嵌，避免文件丢失问题

## 📋 环境要求

- Python 3.8+
- CUDA 11.8+ (GPU训练)
- 16GB+ GPU显存 (推荐24GB+)
- PyTorch 2.0+

## 🛠️ 安装

1. 克隆项目：
```bash
git clone <your-repo-url>
cd RisKVA
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置accelerate：
```bash
accelerate config
```

## 📊 数据格式

### 原始CSV数据格式

训练数据应包含以下字段：
- `image_path`: 图片相对路径
- `data_风险`: 风险描述
- `data_风险等级`: 风险等级（轻微/中等/严重）
- `data_纠正和预防建议`: 修复建议
- `data_缺陷描述`: 缺陷详细描述

示例CSV格式：
```csv
image_id,image_path,data_风险,data_纠正和预防建议,data_风险等级,data_缺陷描述
0,images/001.jpg,厨房用水易溅入墙体造成墙体发霉,建议透底处补刷防水涂料,轻微,厨房墙面防水涂料透底
```

### 数据格式说明

项目使用的两种数据格式：

1. 标准的 CSV + 图片文件格式：CSV 存放每张图的元数据

2. 标准的 Parquet 文件格式：

## 🎯 快速开始

### 1. 数据集准备

使用专门的数据集准备工具：

```bash
# 进入数据准备脚本目录
cd scripts/prepare_single_image_dataset

# 将PDF和Excel原始文件放入input/目录
# 运行数据集创建脚本
python create_dataset_from_files.py

# 验证生成的数据集
python validate_dataset.py output/
```

### 2. SFT数据预处理

```bash
# 使用生成的数据集进行SFT数据预处理
python src/single_image_risk/prepare_sft_data.py \
    --csv_path ./scripts/prepare_single_image_dataset/output/metadata.csv \
    --image_base_path ./scripts/prepare_single_image_dataset/output \
    --output_path ./sft_data
```

### 3. 模型训练

```bash
# 使用accelerate进行分布式训练
accelerate launch \
    --config_file configs/deepspeed_zero2.yaml \
    src/single_image_risk/train.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_path ./sft_data \
    --output_dir ./models/risk_detection_qwen25vl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --bf16 \
    --gradient_checkpointing \
    --use_lora True
```

### 4. 模型推理

#### 单张图片推理：
```bash
python src/single_image_risk/inference.py \
    --model_path ./models/risk_detection_qwen25vl \
    --image_path ./test_image.jpg
```

#### 批量推理：
```bash
python src/single_image_risk/inference.py \
    --model_path ./models/risk_detection_qwen25vl \
    --image_list ./image_list.txt \
    --output_file ./results.json
```

#### 模型转换：
```bash
# 合并LoRA权重到基础模型
python src/single_image_risk/convert_model.py \
    --base_model_path ./models/risk_detection_qwen25vl \
    --output_path ./models/merged_model
```

## ⚙️ 训练参数说明

### 关键参数
- `per_device_train_batch_size`: 每个GPU的批大小
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率，建议5e-6
- `num_train_epochs`: 训练轮数
- `use_lora`: 是否使用LoRA微调
- `lora_r`: LoRA秩，影响参数量

### LoRA配置
```bash
--use_lora True \
--lora_r 64 \
--lora_alpha 128 \
--lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
```

## 📁 项目结构

```
RisKVA/
├── src/
│   └── single_image_risk/
│       ├── train.py               # 主训练脚本
│       ├── prepare_sft_data.py    # SFT数据预处理
│       ├── inference.py           # 模型推理脚本
│       └── convert_model.py       # 模型转换脚本
├── scripts/
│   ├── prepare_single_image_dataset/  # 数据集准备工具
│   │   ├── create_dataset_from_files.py  # 主数据集创建脚本
│   │   ├── get_text_image.py            # PDF/Excel转换
│   │   ├── get_datasets.py              # 数据集整理
│   │   ├── validate_dataset.py          # 数据验证
│   │   └── README.md                    # 详细使用说明
│   └── test_single_image_risk/          # 测试脚本
├── configs/
│   ├── deepspeed_zero2.yaml             # DeepSpeed配置
│   └── prompts/                         # 提示词配置
├── references/                          # 参考实现
├── datasets/                            # 数据集目录
├── models/                              # 模型输出目录
├── requirements.txt                     # 依赖包
├── .gitignore                          # Git忽略文件
└── README.md                           # 项目说明
```

## 🗂️ 数据集工作流程

### 完整数据处理流程
1. **原始数据**: PDF文件（图片）+ Excel文件（标注信息）
2. **数据集创建**: 使用 `scripts/prepare_single_image_dataset/` 工具
3. **格式转换**: 生成标准的 CSV + 图片目录结构
4. **SFT预处理**: 转换为训练所需的对话格式
5. **模型训练**: 使用处理后的数据进行微调

### 数据集准备工具详解

`scripts/prepare_single_image_dataset/` 目录包含完整的数据集准备工具链：

- **`create_dataset_from_files.py`**: 主脚本，自动化处理整个数据集创建流程
- **`get_text_image.py`**: 从PDF提取图片，从Excel提取文本标注
- **`get_datasets.py`**: 整理和匹配图片与标注数据
- **`validate_dataset.py`**: 验证生成数据集的完整性和正确性

详细使用说明请参考：`scripts/prepare_single_image_dataset/README.md`

### 数据集验证
```bash
# 验证数据集完整性
cd scripts/prepare_single_image_dataset
python validate_dataset.py output/

# 检查生成的SFT数据
python src/single_image_risk/prepare_sft_data.py --preview_only
```

## 🔧 常见问题

### Q: 显存不够怎么办？
A: 可以尝试：
- 减少`per_device_train_batch_size`
- 增加`gradient_accumulation_steps` 
- 使用DeepSpeed Zero-3
- 开启梯度检查点
- 降低图片质量和尺寸（Parquet格式）

### Q: 如何调整模型输出格式？
A: 修改 `src/single_image_risk/prepare_sft_data.py` 中的 `system_prompt` 和 `assistant_answer` 格式

### Q: 如何添加更多训练数据？
A: 
1. 将新的PDF和Excel文件放入 `scripts/prepare_single_image_dataset/input/` 目录
2. 重新运行 `create_dataset_from_files.py` 脚本
3. 使用新生成的数据集进行SFT预处理和训练

### Q: 如何处理不同格式的原始数据？
A: 
- **PDF文件**: 确保包含清晰的建筑缺陷图片
- **Excel文件**: 必须包含规定的列名（风险描述、风险等级等）
- **图片质量**: 建议使用高分辨率图片以获得更好的训练效果

## 🎨 自定义输出格式

在 `src/single_image_risk/prepare_sft_data.py` 中修改输出格式：

```python
assistant_answer = f"""**风险描述**: {row['data_风险']}
**风险等级**: {row['data_风险等级']}  
**纠正和预防建议**: {row['data_纠正和预防建议']}
**缺陷描述**: {row['data_缺陷描述']}
**位置信息**: {row.get('位置', '未知')}  # 可添加更多字段
"""
```

### 自定义提示词

可以在 `configs/prompts/` 目录中配置自定义的系统提示词和用户提示词模板。

## 📈 训练监控

支持TensorBoard和WandB监控：

```bash
# TensorBoard
tensorboard --logdir ./models/risk_detection_qwen25vl/runs

# WandB (需要登录)
wandb login
```

## 🚀 部署建议

1. **模型合并**: 使用 `src/single_image_risk/convert_model.py` 合并LoRA权重
2. **生产环境**: 使用合并后的模型以提高推理速度
3. **API服务**: 可以基于 FastAPI 封装 `src/single_image_risk/inference.py`
4. **批量处理**: 支持多张图片并行推理
5. **容器化**: 使用 Docker 进行部署，确保环境一致性

## 📝 License

本项目遵循Apache 2.0许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 💡 技术原理

### SFT训练流程
1. **数据预处理**: 将CSV数据转换为对话格式
2. **模型加载**: 加载Qwen2.5-VL预训练模型
3. **LoRA微调**: 使用参数高效微调减少显存占用
4. **多GPU训练**: 使用DeepSpeed进行分布式训练

### 数据格式转换
原始数据 → 对话格式 → Token化 → 模型训练

### 推理过程  
图片输入 → 视觉编码 → 多模态融合 → 文本生成 → 结构化输出
