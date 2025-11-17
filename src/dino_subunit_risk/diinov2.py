"""
DINOv2 图像分类训练脚本（支持类别权重平衡）
使用 DINOv2 作为特征提取器，添加 MLP 分类头进行缺陷分类

# 仅训练分类头（使用类别权重平衡）
python diinov2.py \
  --model_name dinov2_vitg14 \
  --state_dict_path /home/d3010/code/models/dinov2/dinov2_vitg14_pretrain.pth \
  --pass_k 2 \
  --use_class_weights

# 仅训练分类头（不使用类别权重平衡）
python diinov2.py \
  --model_name dinov2_vitg14 \
  --state_dict_path /home/d3010/code/models/dinov2/dinov2_vitg14_pretrain.pth \
  --pass_k 2

python diinov2.py \
  --model_name dinov2_vitg14 \
  --state_dict_path /home/d3010/code/models/dinov2/dinov2_vitg14_pretrain.pth \
  --pass_k 2 \
  --csv_file /home/d3010/code/datasets/Subunit-Risk_all/metadata_with_image_filterbadcase.csv

# 仅评估模型
python diinov2.py \
  --model_name dinov2_vitg14 \
  --state_dict_path /home/d3010/code/models/dinov2/dinov2_vitg14_pretrain.pth \
  --eval_from_checkpoint /home/d3010/code/datasets/Subunit-Risk_all/checkpoints/best_model.pth \
  --pass_k 2

# 训练分类头和DINOv2 backbone（使用类别权重平衡）效果不好
python diinov2.py \
  --model_name dinov2_vitg14 \
  --state_dict_path /home/d3010/code/models/dinov2/dinov2_vitg14_pretrain.pth \
  --freeze_backbone False \
  --pass_k 2 \
  --use_class_weights
"""

import os
import json
import argparse
import ast
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 默认路径（模块级，供类初始化等位置使用；可被命令行参数覆盖）
DEFAULT_DATA_DIR = "/home/d3010/code/datasets/Subunit-Risk_all"
DEFAULT_CSV_FILE = "metadata_with_image.csv"


class DefectDataset(Dataset):
    """缺陷分类数据集"""
    
    def __init__(self, csv_file: str, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        初始化数据集
        
        Args:
            csv_file: CSV 文件路径
            data_dir: 数据根目录
            transform: 图像变换
        """
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # 解析图片路径和标签
        self.samples = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # 收集所有标签
        unique_labels = sorted(self.df['defect_category_num'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        print(f"发现 {self.num_classes} 个类别: {unique_labels}")
        
        # 构建样本列表
        for _, row in self.df.iterrows():
            image_paths_str = row['all_image_paths']
            label = row['defect_category_num']
            label_idx = self.label_to_idx[label]
            
            # 解析 JSON 格式的图片路径列表
            try:
                if isinstance(image_paths_str, str):
                    # 尝试直接解析 JSON
                    try:
                        image_paths = json.loads(image_paths_str)
                    except json.JSONDecodeError:
                        # 如果失败，尝试替换单引号为双引号
                        try:
                            image_paths = json.loads(image_paths_str.replace("'", '"'))
                        except json.JSONDecodeError:
                            # 如果还是失败，尝试使用 ast.literal_eval
                            image_paths = ast.literal_eval(image_paths_str)
                else:
                    image_paths = image_paths_str
                
                # 确保 image_paths 是列表
                if not isinstance(image_paths, list):
                    image_paths = [image_paths]
                
                # 为每个图片路径创建一个样本
                for img_path in image_paths:
                    # 处理路径：如果已经是绝对路径，直接使用；否则拼接数据目录
                    if os.path.isabs(img_path):
                        full_path = img_path
                    else:
                        full_path = os.path.join(self.data_dir, img_path)
                    
                    if os.path.exists(full_path):
                        self.samples.append((full_path, label_idx))
                    else:
                        print(f"警告: 图片文件不存在: {full_path}")
            except Exception as e:
                print(f"解析图片路径时出错: {image_paths_str}, 错误: {e}")
        
        if len(self.samples) == 0:
            raise ValueError("没有找到有效的图片样本！请检查数据路径。")
        
        print(f"数据集大小: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            # 返回黑色图片作为占位符
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换（如果提供了）
        # 注意：如果 transform 为 None，则返回原始 PIL 图像，由外部的 TransformDataset 处理
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class DINOv2Classifier(nn.Module):
    """DINOv2 + MLP 分类器"""
    
    def __init__(self, num_classes: int, model_name: str = "dinov2_vitg14", 
                 freeze_backbone: bool = True, backbone_pickle: Optional[str] = None,
                 state_dict_path: Optional[str] = None, dinov2_repo_dir: Optional[str] = None):
        """
        初始化分类器
        
        Args:
            num_classes: 分类类别数
            model_name: DINOv2 模型名称 ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            freeze_backbone: 是否冻结 DINOv2 权重
            backbone_pickle: 通过 torch.save(model) 完整序列化的 DINOv2 模型路径（优先使用）
            state_dict_path: 官方或本地保存的预训练权重（state_dict 或 checkpoint）的路径
            dinov2_repo_dir: 本地 dinov2 源码目录，用于从源码构建模型结构并加载 state_dict
        """
        super(DINOv2Classifier, self).__init__()
        
        # 优先：直接加载完整序列化的模型对象（无需 dinov2 源码）
        if backbone_pickle is not None:
            print(f"从完整序列化文件加载 DINOv2 模型: {backbone_pickle}")
            if not os.path.exists(backbone_pickle):
                raise FileNotFoundError(f"指定的 backbone_pickle 不存在: {backbone_pickle}")
            try:
                loaded_obj = torch.load(backbone_pickle, map_location="cpu")
                # 情况1：完整模型对象
                if isinstance(loaded_obj, nn.Module):
                    self.backbone = loaded_obj
                    print("成功从 pickle 加载完整模型对象")
                # 情况2：checkpoint / state_dict
                elif isinstance(loaded_obj, dict):
                    # 常见键名判定
                    possible_keys = ["state_dict", "model_state_dict", "model", "backbone", "module"]
                    found_keys = [k for k in possible_keys if k in loaded_obj]
                    raise RuntimeError(
                        "提供的 backbone_pickle 似乎是 checkpoint/state_dict，而不是完整模型对象。\n"
                        f"检测到的键: {found_keys if found_keys else '无明显模型对象键'}\n"
                        "当前已禁用从源码构建模型，无法从 state_dict 还原结构。\n"
                        "请提供通过 torch.save(model) 直接保存的完整模型对象文件，或改用源码加载路径。"
                    )
                else:
                    raise RuntimeError(
                        f"无法识别的 pickle 内容类型: {type(loaded_obj)}。需要 nn.Module 的完整模型对象。"
                    )
            except Exception as e:
                raise RuntimeError(f"加载完整模型对象失败: {e}")
        else:
            # 方案B：从本地源码构建并加载 state_dict/checkpoint
            if state_dict_path is None:
                raise RuntimeError(
                    "未提供 --backbone_pickle，也未提供 --state_dict_path。"
                    "请提供完整模型对象 (--backbone_pickle) 或 state_dict (--state_dict_path + --dinov2_repo_dir)。"
                )
            if not os.path.exists(state_dict_path):
                raise FileNotFoundError(f"state_dict_path 不存在: {state_dict_path}")
            if dinov2_repo_dir is None:
                dinov2_repo_dir = "/home/d3010/code/dinov2"
            if not os.path.exists(dinov2_repo_dir):
                raise FileNotFoundError(f"dinov2_repo_dir 不存在: {dinov2_repo_dir}")
            
            print(f"从源码构建 {model_name} 并加载权重: {state_dict_path}")
            import sys
            if dinov2_repo_dir not in sys.path:
                sys.path.insert(0, dinov2_repo_dir)
            try:
                from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
            except Exception as e:
                raise ImportError(f"无法导入 dinov2 源码（{dinov2_repo_dir}），请确认路径有效。错误: {e}")
            
            model_map = {
                'dinov2_vits14': dinov2_vits14,
                'dinov2_vitb14': dinov2_vitb14,
                'dinov2_vitl14': dinov2_vitl14,
                'dinov2_vitg14': dinov2_vitg14,
            }
            if model_name not in model_map:
                raise ValueError(f"不支持的模型名称: {model_name}，可选: {list(model_map.keys())}")
            
            # 构建模型（不从互联网下载预训练）
            model = model_map[model_name](pretrained=False)
            # 读取权重
            ckpt = torch.load(state_dict_path, map_location="cpu")
            # 兼容不同保存格式
            if isinstance(ckpt, dict):
                # 可能嵌套 key
                for key in ["state_dict", "model_state_dict", "model", "backbone", "module"]:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        ckpt = ckpt[key]
                        break
            if not isinstance(ckpt, dict):
                raise RuntimeError("state_dict_path 文件内容无法识别为权重字典。")
            # 去掉可能的 'module.' 前缀（DataParallel/FSDP）
            def strip_prefix_if_present(state_dict: dict, prefix: str = "module."):
                return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }
            ckpt = strip_prefix_if_present(ckpt, "module.")
            # 加载
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:
                print(f"警告: 有缺失权重: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"警告: 有多余权重: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
            self.backbone = model
            print("已从 state_dict 加载 DINOv2 模型。")
        
        # 冻结 DINOv2 权重
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("DINOv2 backbone 权重已冻结")
        else:
            print("DINOv2 backbone 权重可训练")
        
        # 获取特征维度
        # DINOv2 的特征维度: small=384, base=768, large=1024, giant=1536
        if hasattr(self.backbone, 'embed_dim'):
            feature_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            # 默认值，根据模型名称推断
            dim_map = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536,
            }
            feature_dim = dim_map.get(model_name, 768)
        
        print(f"特征维度: {feature_dim}")
        
        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
        
        Returns:
            分类 logits [batch_size, num_classes]
        """
        # 提取特征 - DINOv2 的 forward_features 返回字典，包含 x_norm_clstoken
        features_dict = self.backbone.forward_features(x)
        
        # 获取 CLS token 的特征
        if isinstance(features_dict, dict):
            features = features_dict['x_norm_clstoken']
        else:
            # 如果返回的是张量，取第一个 token (CLS token)
            features = features_dict[:, 0]
        
        # 分类
        logits = self.classifier(features)
        return logits


def get_transforms(is_training: bool = True, img_size: int = 224):
    """
    获取数据变换
    
    Args:
        is_training: 是否为训练模式
        img_size: 图像尺寸 (DINOv2 可以使用 224 或 518)
    
    Returns:
        transforms.Compose: 数据变换
    """
    # DINOv2 使用 ImageNet 归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': running_loss / len(dataloader),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, device, epoch, pass_k=3):
    """
    验证一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前 epoch
        pass_k: Pass@k 评估中的 k 值（默认: 3）
    
    Returns:
        epoch_loss: epoch 平均损失
        epoch_acc: top-1 准确率
        epoch_pass_k_acc: pass@k 准确率
        all_preds: 所有预测结果
        all_labels: 所有真实标签
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_pass_k = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # 确保 k 不超过类别数
    num_classes = model.classifier[-1].out_features
    k = min(pass_k, num_classes)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 计算 pass@k accuracy
            if k > 0:
                _, topk_pred = torch.topk(outputs.data, k, dim=1)
                # 检查真实标签是否在前 k 个预测中（向量化操作）
                labels_expanded = labels.unsqueeze(1).expand_as(topk_pred)
                correct_pass_k += (topk_pred == labels_expanded).any(dim=1).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': running_loss / len(dataloader),
                'acc': 100 * correct / total,
                f'pass@{k}': 100 * correct_pass_k / total if k > 0 else 0.0
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    epoch_pass_k_acc = 100 * correct_pass_k / total if k > 0 else 0.0
    
    return epoch_loss, epoch_acc, epoch_pass_k_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='DINOv2 缺陷分类训练')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--model_name', type=str, default='dinov2_vitg14', 
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DINOv2 模型名称')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, 
                        help='是否冻结 DINOv2 权重')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸 (224 或 518)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--backbone_pickle', type=str, default=None, 
                        help='通过 torch.save(model) 完整序列化的 DINOv2 模型文件路径（优先使用）')
    parser.add_argument('--state_dict_path', type=str, default=None, 
                        help='DINOv2 预训练权重 state_dict/checkpoint 路径（与 --dinov2_repo_dir 搭配使用）')
    parser.add_argument('--dinov2_repo_dir', type=str, default='/home/d3010/code/dinov2', 
                        help='本地 dinov2 源码目录，用于从源码构建模型结构')
    parser.add_argument('--eval_from_checkpoint', type=str, default=None,
                        help='仅评估：从给定检查点路径加载权重，跳过训练，直接在测试集上评估')
    parser.add_argument('--pass_k', type=int, default=3,
                        help='Pass@k 评估中的 k 值（默认: 3，表示检查真实标签是否在前 k 个预测中）')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='是否使用类别权重平衡损失函数（基于训练集的类别频率计算反频率权重）')

    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, 
                        help=f'数据集根目录（默认: {DEFAULT_DATA_DIR}）')
    parser.add_argument('--csv_file', type=str, default=DEFAULT_CSV_FILE, 
                        help=f'CSV 文件名或完整路径（默认: {DEFAULT_CSV_FILE}，相对于 data_dir）')
    
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir if args.data_dir else DEFAULT_DATA_DIR
    
    # 处理 CSV 文件路径
    if args.csv_file:
        if os.path.isabs(args.csv_file):
            # 如果是绝对路径，直接使用
            CSV_FILE = args.csv_file
        else:
            # 如果是相对路径，相对于 DATA_DIR
            CSV_FILE = os.path.join(DATA_DIR, args.csv_file)
    else:
        # 使用默认文件名
        CSV_FILE = os.path.join(DATA_DIR, DEFAULT_CSV_FILE)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    full_dataset = DefectDataset(
        csv_file=CSV_FILE,
        data_dir=DATA_DIR,
        transform=None  # 变换将在数据加载器中动态应用
    )
    
    # 划分数据集 - 使用分层采样确保每个集合中类别分布一致
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]
    
    # 首先划分训练集和临时集（验证+测试）
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=args.test_size + args.val_size, 
        random_state=42, 
        stratify=labels
    )
    
    # 然后从临时集中划分验证集和测试集
    temp_labels = [full_dataset.samples[i][1] for i in temp_indices]
    val_size_ratio = args.val_size / (args.test_size + args.val_size)
    val_indices, test_indices = train_test_split(
        temp_indices, 
        test_size=1 - val_size_ratio, 
        random_state=42, 
        stratify=temp_labels
    )
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # 为验证集和测试集创建新的数据集实例（使用不同的变换）
    # 注意：Subset 共享底层数据集，所以我们需要在 DataLoader 中处理变换
    # 或者创建包装器来动态设置变换
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
    
    # 获取变换
    train_transform = get_transforms(is_training=True, img_size=args.img_size)
    val_transform = get_transforms(is_training=False, img_size=args.img_size)
    test_transform = get_transforms(is_training=False, img_size=args.img_size)
    
    # 创建带有正确变换的数据集
    train_dataset_wrapped = TransformDataset(train_dataset, train_transform)
    val_dataset_wrapped = TransformDataset(val_dataset, val_transform)
    test_dataset_wrapped = TransformDataset(test_dataset, test_transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 计算类别权重（基于训练集，仅在启用时计算）
    class_weights = None
    if args.use_class_weights:
        print("\n计算类别权重...")
        train_labels = [full_dataset.samples[i][1] for i in train_indices]
        class_counts = [0] * full_dataset.num_classes
        for label_idx in train_labels:
            class_counts[label_idx] += 1
        
        total_samples = sum(class_counts)
        # 计算类别权重（反频率）：total_samples / (num_classes * count)
        class_weights = [total_samples / (full_dataset.num_classes * count) if count > 0 else 1.0 
                         for count in class_counts]
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        print(f"类别样本数: {class_counts}")
        print(f"类别权重: {class_weights.cpu().numpy()}")
    else:
        print("\n未启用类别权重平衡（使用标准交叉熵损失）")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset_wrapped, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset_wrapped, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset_wrapped, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    print("创建模型...")
    print(f"使用数据目录: {DATA_DIR}")
    print(f"使用 CSV 文件: {CSV_FILE}")
    model = DINOv2Classifier(
        num_classes=full_dataset.num_classes,
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        backbone_pickle=args.backbone_pickle,
        state_dict_path=args.state_dict_path,
        dinov2_repo_dir=args.dinov2_repo_dir
    ).to(device)
    
    # 损失函数和优化器
    if args.use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权交叉熵损失函数（类别权重已应用）")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"使用标准交叉熵损失函数")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"恢复训练从: {args.resume}")
        # 兼容 PyTorch 2.6+ 默认 weights_only=True 导致无法反序列化自定义对象的问题
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # 仅评估模式：直接从检查点加载模型并在测试集上评估，跳过训练
    if args.eval_from_checkpoint:
        ckpt_path = args.eval_from_checkpoint
        print(f"仅评估模式：从检查点加载权重并评估: {ckpt_path}")
        # 兼容 PyTorch 2.6+ 的 weights_only=True 默认行为
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # 兼容直接保存的 state_dict
            model.load_state_dict(checkpoint, strict=False)
    
    else:
        # 训练循环
        print("开始训练...")
        for epoch in range(start_epoch, args.epochs):
            # 训练
            train_loss, train_acc, train_preds, train_labels = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # 验证
            val_loss, val_acc, val_pass_k_acc, val_preds, val_labels = validate_epoch(
                model, val_loader, criterion, device, epoch, pass_k=args.pass_k
            )
            
            # 学习率调度
            scheduler.step()
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Pass@{args.pass_k}: {val_pass_k_acc:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'num_classes': full_dataset.num_classes,
                    'label_to_idx': full_dataset.label_to_idx,
                    'idx_to_label': full_dataset.idx_to_label,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
            
            # 保存检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'num_classes': full_dataset.num_classes,
                'label_to_idx': full_dataset.label_to_idx,
                'idx_to_label': full_dataset.idx_to_label,
            }, os.path.join(args.save_dir, 'checkpoint.pth'))
    
    # 测试
    print("\n开始测试...")
    test_loss, test_acc, test_pass_k_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device, args.epochs, pass_k=args.pass_k
    )
    
    print(f"\n测试结果:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Pass@{args.pass_k}: {test_pass_k_acc:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(
        test_labels, test_preds,
        target_names=[f"类别{full_dataset.idx_to_label[i]}" for i in range(full_dataset.num_classes)]
    ))
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()

