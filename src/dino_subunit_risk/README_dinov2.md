# DINOv2 缺陷分类训练脚本

## 功能说明

此脚本使用 DINOv2 作为特征提取器，添加 MLP 分类头进行缺陷分类任务。

### 主要特性

- 使用 DINOv2 预训练模型作为特征提取器（权重冻结）
- 添加可训练的 MLP 分类头
- 支持多种 DINOv2 模型（small, base, large, giant）
- 自动下载模型到指定目录
- 支持训练、验证和测试
- 自动保存最佳模型
- 生成分类报告和混淆矩阵

## 环境要求

```bash
pip install -r requirements.txt
```

## 数据格式

- CSV 文件包含以下列：
  - `all_image_paths`: JSON 格式的图片路径列表
  - `defect_category_num`: 分类标签（整数）

## 使用方法

### 基本训练

```bash
python diinov2.py
```

### 自定义参数训练

```bash
python diinov2.py \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --model_name dinov2_vitb14 \
    --img_size 224 \
    --save_dir ./checkpoints \
    --freeze_backbone
```

### 参数说明

- `--batch_size`: 批次大小（默认: 32）
- `--epochs`: 训练轮数（默认: 50）
- `--lr`: 学习率（默认: 1e-4）
- `--model_name`: DINOv2 模型名称
  - `dinov2_vits14`: Small 模型（384 维特征）
  - `dinov2_vitb14`: Base 模型（768 维特征，推荐）
  - `dinov2_vitl14`: Large 模型（1024 维特征）
  - `dinov2_vitg14`: Giant 模型（1536 维特征）
- `--img_size`: 输入图像尺寸（默认: 224，可选: 518）
- `--freeze_backbone`: 冻结 DINOv2 权重（默认: True）
- `--test_size`: 测试集比例（默认: 0.2）
- `--val_size`: 验证集比例（默认: 0.1）
- `--save_dir`: 模型保存目录（默认: ./checkpoints）
- `--resume`: 恢复训练的检查点路径
- `--num_workers`: 数据加载器的工作进程数（默认: 4）

## 模型保存

训练过程中会保存：
- `best_model.pth`: 验证集上表现最好的模型
- `checkpoint.pth`: 最新的检查点

模型文件包含：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 标签映射（label_to_idx, idx_to_label）
- 类别数量

## 模型加载

训练完成后，模型会保存在 `--save_dir` 目录下。可以使用以下代码加载模型：

```python
import torch
from diinov2 import DINOv2Classifier

# 加载检查点
checkpoint = torch.load('checkpoints/best_model.pth')

# 创建模型
model = DINOv2Classifier(
    num_classes=checkpoint['num_classes'],
    model_name='dinov2_vitb14',
    freeze_backbone=True
)
model.load_state_dict(checkpoint['model_state_dict'])

# 获取标签映射
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
```

## 输出结果

训练完成后会输出：
- 训练和验证的损失和准确率
- 测试集的分类报告
- 混淆矩阵

## 注意事项

1. 模型会自动下载到 `/home/d3010/code/models` 目录
2. 如果本地存在 dinov2 库（`/home/d3010/code/dinov2`），会优先使用本地库
3. 确保有足够的 GPU 内存（Base 模型约需要 2-4GB）
4. 数据路径在代码中硬编码，如需修改请编辑 `DATA_DIR` 和 `CSV_FILE` 变量

## 故障排除

### 模型下载失败

如果模型下载失败，可以：
1. 手动下载模型权重文件到 `/home/d3010/code/models` 目录
2. 或者使用本地 dinov2 库

### 内存不足

如果遇到内存不足问题：
1. 减小 `--batch_size`
2. 使用较小的模型（如 `dinov2_vits14`）
3. 减小 `--img_size`（使用 224 而不是 518）

### 数据加载错误

如果遇到数据加载错误：
1. 检查 CSV 文件路径是否正确
2. 检查图片路径是否存在
3. 检查 `all_image_paths` 列的格式是否正确（应该是 JSON 格式的列表）

