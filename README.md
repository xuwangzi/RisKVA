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

### 处理后数据格式比较

| 特性 | JSONL格式 | Parquet格式（推荐） |
|------|-----------|-------------------|
| **存储方式** | 图片文件+文本文件分离 | 图片+文本一体化存储 |
| **数据完整性** | 依赖外部图片文件 | 内嵌base64编码图片 |
| **传输便利性** | 需要同时传输多个文件 | 单文件包含所有数据 |
| **压缩效率** | 中等 | 高（snappy压缩） |
| **查询性能** | 一般 | 优秀（列式存储） |
| **随机访问** | 需要遍历 | 支持高效索引 |
| **文件大小** | 较小（文本）+原始图片 | 中等（压缩后图片） |
| **适用场景** | 开发测试 | 生产部署 |

## 🎯 快速开始

### 方案一：JSONL格式（原版）

#### 1. 数据预处理

```bash
python prepare_sft_data.py \
    --csv_path ./datasets/single_image_tiny_247/metadata.csv \
    --image_base_path ./datasets/single_image_tiny_247 \
    --output_path ./sft_data
```

#### 2. 训练模型

```bash
bash run_training.sh
```

### 方案二：Parquet格式（推荐）

#### 1. 数据预处理为Parquet格式

```bash
python prepare_sft_data_parquet.py \
    --csv_path ./datasets/single_image_tiny_247/metadata.csv \
    --image_base_path ./datasets/single_image_tiny_247 \
    --output_path ./datasets/processed/single_image_parquet \
    --image_quality 85 \
    --max_image_size 1024 1024
```

#### 2. 训练模型

```bash
bash run_training_parquet.sh
```

或手动执行：
```bash
accelerate launch \
    --config_file configs/deepspeed_zero2.yaml \
    custom_sft_vlm.py \
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

### 3. 模型推理

#### JSONL格式推理：
单张图片推理：
```bash
python inference.py \
    --model_path ./models/risk_detection_qwen25vl \
    --image_path ./test_image.jpg
```

批量推理：
```bash
python inference.py \
    --model_path ./models/risk_detection_qwen25vl \
    --image_list ./image_list.txt \
    --output_file ./results.json
```

#### Parquet格式推理：
单张图片推理：
```bash
python inference_parquet.py \
    --model_path ./models/risk_detection_qwen25vl_parquet \
    --image_path ./test_image.jpg
```

批量推理（从Parquet文件）：
```bash
python inference_parquet.py \
    --model_path ./models/risk_detection_qwen25vl_parquet \
    --parquet_file ./datasets/processed/single_image_parquet/test.parquet \
    --output_file ./results.json \
    --compare_gt
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
├── prepare_sft_data.py          # 数据预处理脚本
├── custom_sft_vlm.py           # 自定义SFT训练脚本  
├── inference.py                # 推理脚本
├── convert_model.py           # 模型转换脚本
├── run_training.sh            # 训练启动脚本
├── configs/
│   └── deepspeed_zero2.yaml   # DeepSpeed配置
├── datasets/                  # 数据集目录
└── models/                    # 模型输出目录
```

## 🗂️ Parquet格式详细说明

### 优势特点
1. **数据完整性**: 图片以base64编码内嵌，无需担心文件丢失
2. **高效压缩**: 使用snappy压缩，存储效率高
3. **快速加载**: 列式存储，支持高效数据访问
4. **便于传输**: 单文件包含所有数据，易于部署

### 预处理参数说明
```bash
--image_quality 85          # JPEG压缩质量(1-100)
--max_image_size 1024 1024  # 图片最大尺寸[宽 高]
```

### 数据预览
```bash
python prepare_sft_data_parquet.py --preview ./datasets/processed/single_image_parquet/train.parquet
```

### 训练时的额外参数
```bash
--cache_images True         # 是否预先解码图片到内存
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
A: 修改`prepare_sft_data.py`或`prepare_sft_data_parquet.py`中的`system_prompt`和`assistant_answer`格式

### Q: 如何添加更多训练数据？
A: 按照CSV格式准备数据，确保图片路径正确

### Q: Parquet和JSONL格式如何选择？
A: 
- **开发阶段**: 建议使用JSONL格式，方便调试
- **生产环境**: 建议使用Parquet格式，数据完整性好
- **大规模数据**: 必须使用Parquet格式，效率更高

## 🎨 自定义输出格式

在`prepare_sft_data.py`中修改输出格式：

```python
assistant_answer = f"""**风险描述**: {row['data_风险']}
**风险等级**: {row['data_风险等级']}  
**纠正和预防建议**: {row['data_纠正和预防建议']}
**缺陷描述**: {row['data_缺陷描述']}
**位置信息**: {row.get('位置', '未知')}  # 可添加更多字段
"""
```

## 📈 训练监控

支持TensorBoard和WandB监控：

```bash
# TensorBoard
tensorboard --logdir ./models/risk_detection_qwen25vl/runs

# WandB (需要登录)
wandb login
```

## 🚀 部署建议

1. **生产环境**: 使用合并后的模型以提高推理速度
2. **API服务**: 可以基于FastAPI封装推理接口
3. **批量处理**: 使用批量推理脚本处理大量图片

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
