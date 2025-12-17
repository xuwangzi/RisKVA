"""
DINOv2 图像分类推理脚本
使用 DINOv2 作为特征提取器，添加 MLP 分类头进行缺陷分类推理

# 推理示例 - 单张图片
python /home/d3010/code/RisKVA/src/dino_subunit_risk/diinov2_inf.py \
  --state_dict_path /home/d3010/code/RisKVA/models/pretrained_models/dinov2/dinov2_vitg14_reg4_pretrain.pth \
  --dinov2_repo_dir /home/d3010/code/RisKVA/src/dino_subunit_risk/dinov2 \
  --checkpoint_path /home/d3010/code/RisKVA/models/finetuned_models/RisKVA-Subunit-dinov2/checkpoints_12_class_81per/best_model.pth \
  --image_paths /home/d3010/code/RisKVA/datasets/Subunit-Risk_v4/PR_data/rebar_PR/rebar_PR_001.jpg \
  /home/d3010/code/RisKVA/datasets/Subunit-Risk_v4/images/000029_00_QILIAN-031.jpg \
  /home/d3010/code/RisKVA/datasets/Subunit-Risk_v4/images/000019_00_QILIAN-021.jpg \
  /home/d3010/code/RisKVA/datasets/Subunit-Risk_v4/images/000016_00_QILIAN-017.jpg
#   钢筋施工及钢材相关问题, 墙面相关问题, 施工工艺缺陷问题, 卫生洁具安装问题

# 推理示例 - 从JSON文件读取图片路径
python /home/d3010/code/RisKVA/src/dino_subunit_risk/diinov2_inf.py \
  --state_dict_path /home/d3010/code/RisKVA/models/pretrained_models/dinov2/dinov2_vitg14_reg4_pretrain.pth \
  --dinov2_repo_dir /home/d3010/code/RisKVA/src/dino_subunit_risk/dinov2 \
  --checkpoint_path /home/d3010/code/RisKVA/models/finetuned_models/RisKVA-Subunit-dinov2/checkpoints_12_class_81per/best_model.pth \
  --image_list_file /home/d3010/code/RisKVA/src/dino_subunit_risk/test_images.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# 默认路径
DEFAULT_DINOV2_REPO_DIR = "/home/d3010/code/RisKVA/src/dino_subunit_risk/dinov2"


class ImageDataset(Dataset):
    """图片路径数据集（用于推理）"""
    
    def __init__(self, image_paths: List[str], transform: Optional[transforms.Compose] = None):
        """
        初始化数据集
        
        Args:
            image_paths: 图片路径列表
            transform: 图像变换
        """
        self.image_paths = image_paths
        self.transform = transform
        
        # 验证图片路径是否存在
        valid_paths = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                valid_paths.append(img_path)
            else:
                print(f"警告: 图片文件不存在: {img_path}")
        
        if len(valid_paths) == 0:
            raise ValueError("没有找到有效的图片文件！请检查图片路径。")
        
        self.image_paths = valid_paths
        print(f"找到 {len(self.image_paths)} 张有效图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            # 返回黑色图片作为占位符
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换（如果提供了）
        if self.transform is not None:
            image = self.transform(image)
        
        return image, img_path


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
                dinov2_repo_dir = DEFAULT_DINOV2_REPO_DIR
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


def get_transforms(img_size: int = 224):
    """
    获取推理时的数据变换
    
    Args:
        img_size: 图像尺寸 (DINOv2 可以使用 224 或 518)
    
    Returns:
        transforms.Compose: 数据变换
    """
    # DINOv2 使用 ImageNet 归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def inference_images(model, dataloader, device, idx_to_label: Dict[int, str], top_k: int = 3):
    """
    对图片进行推理，返回每张图片的分类结果
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        idx_to_label: 类别索引到类别名称的映射
        top_k: 返回top-k预测结果（默认: 3）
    
    Returns:
        results: 每张图片的推理结果列表，每个元素包含：
            - image_path: 图片路径
            - predicted_class_idx: 预测类别索引
            - predicted_class_name: 预测类别名称
            - confidence: 预测置信度（概率）
            - top_k_predictions: top-k预测结果（类别索引、名称、概率）
    """
    model.eval()
    results = []
    
    # 确保 k 不超过类别数
    num_classes = model.classifier[-1].out_features
    k = min(top_k, num_classes)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="推理中")
        for images, image_paths in pbar:
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            predicted_np = predicted.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            # 获取top-k预测
            if k > 0:
                topk_probs, topk_indices = torch.topk(probs, k, dim=1)
                topk_probs_np = topk_probs.cpu().numpy()
                topk_indices_np = topk_indices.cpu().numpy()
            
            # 为每张图片构建结果
            for i in range(len(image_paths)):
                img_path = image_paths[i]
                pred_idx = int(predicted_np[i])
                pred_name = idx_to_label.get(pred_idx, f"类别{pred_idx}")
                confidence = float(probs_np[i][pred_idx])
                
                # top-k预测结果
                top_k_predictions = []
                if k > 0:
                    for j in range(k):
                        topk_idx = int(topk_indices_np[i][j])
                        topk_name = idx_to_label.get(topk_idx, f"类别{topk_idx}")
                        topk_prob = float(topk_probs_np[i][j])
                        top_k_predictions.append({
                            'class_idx': topk_idx,
                            'class_name': topk_name,
                            'probability': topk_prob
                        })
                
                result = {
                    'image_path': img_path,
                    'predicted_class_idx': pred_idx,
                    'predicted_class_name': pred_name,
                    'confidence': confidence,
                    'top_k_predictions': top_k_predictions
                }
                results.append(result)
            
            # 更新进度条
            pbar.set_postfix({'processed': len(results)})
    
    return results


def load_model(checkpoint_path: str, model_name: str = "dinov2_vitg14",
               backbone_pickle: Optional[str] = None,
               state_dict_path: Optional[str] = None,
               dinov2_repo_dir: Optional[str] = None,
               device: Optional[torch.device] = None):
    """
    加载模型和配置
    
    Returns:
        model: 加载的模型
        idx_to_label: 类别索引到类别名称的映射
        num_classes: 类别数
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"加载检查点: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 兼容 PyTorch 2.6+ 的 weights_only=True 默认行为
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点获取模型配置信息
    if isinstance(checkpoint, dict):
        num_classes = checkpoint.get('num_classes', None)
        label_to_idx = checkpoint.get('label_to_idx', None)
        idx_to_label = checkpoint.get('idx_to_label', None)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        # 如果检查点直接是 state_dict
        raise ValueError("检查点必须包含模型配置信息（num_classes, idx_to_label等）")
    
    if num_classes is None or idx_to_label is None:
        raise ValueError("检查点中缺少必要的配置信息（num_classes 或 idx_to_label）")
    
    print(f"类别数: {num_classes}")
    print(f"类别映射: {idx_to_label}")
    
    # 创建模型
    print("创建模型...")
    model = DINOv2Classifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=True,  # 推理时总是冻结
        backbone_pickle=backbone_pickle,
        state_dict_path=state_dict_path,
        dinov2_repo_dir=dinov2_repo_dir
    ).to(device)
    
    # 加载模型权重
    print("加载模型权重...")
    if isinstance(model_state_dict, dict):
        model.load_state_dict(model_state_dict, strict=False)
    else:
        model.load_state_dict(model_state_dict, strict=False)
    print("模型权重加载完成")
    
    return model, idx_to_label, num_classes


def main():
    parser = argparse.ArgumentParser(description='DINOv2 缺陷分类推理')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='模型检查点路径（必需）')
    parser.add_argument('--image_paths', type=str, nargs='+', default=None,
                        help='图片路径列表（可指定多个）')
    parser.add_argument('--image_list_file', type=str, default=None,
                        help='包含图片路径列表的JSON文件路径（格式: {"image_paths": ["path1", "path2", ...]}）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--model_name', type=str, default='dinov2_vitg14', 
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DINOv2 模型名称')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸 (224 或 518)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--backbone_pickle', type=str, default=None, 
                        help='通过 torch.save(model) 完整序列化的 DINOv2 模型文件路径（优先使用）')
    parser.add_argument('--state_dict_path', type=str, default=None, 
                        help='DINOv2 预训练权重 state_dict/checkpoint 路径（与 --dinov2_repo_dir 搭配使用）')
    parser.add_argument('--dinov2_repo_dir', type=str, default=DEFAULT_DINOV2_REPO_DIR, 
                        help='本地 dinov2 源码目录，用于从源码构建模型结构')
    parser.add_argument('--top_k', type=int, default=3,
                        help='返回top-k预测结果（默认: 3）')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出JSON文件路径（保存预测结果，可选）')
    
    args = parser.parse_args()
    
    # 验证输入
    if args.image_paths is None and args.image_list_file is None:
        raise ValueError("必须提供 --image_paths 或 --image_list_file 之一")
    
    if args.image_paths is not None and args.image_list_file is not None:
        raise ValueError("不能同时提供 --image_paths 和 --image_list_file")
    
    # 读取图片路径
    if args.image_list_file:
        print(f"从文件读取图片路径: {args.image_list_file}")
        with open(args.image_list_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'image_paths' in data:
                image_paths = data['image_paths']
            elif isinstance(data, list):
                image_paths = data
            else:
                raise ValueError(f"JSON文件格式错误，应为包含'image_paths'键的字典或图片路径列表")
    else:
        image_paths = args.image_paths
    
    print(f"输入图片数量: {len(image_paths)}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, idx_to_label, num_classes = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        backbone_pickle=args.backbone_pickle,
        state_dict_path=args.state_dict_path,
        dinov2_repo_dir=args.dinov2_repo_dir,
        device=device
    )
    
    # 创建数据集和数据加载器
    transform = get_transforms(img_size=args.img_size)
    dataset = ImageDataset(image_paths=image_paths, transform=transform)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 执行推理
    print("\n开始推理...")
    results = inference_images(
        model, dataloader, device, idx_to_label, top_k=args.top_k
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("推理结果:")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\n图片 {i}: {result['image_path']}")
        print(f"  预测类别: {result['predicted_class_name']} (索引: {result['predicted_class_idx']})")
        print(f"  置信度: {result['confidence']:.4f}")
        if result['top_k_predictions']:
            print(f"  Top-{len(result['top_k_predictions'])} 预测:")
            for j, top_pred in enumerate(result['top_k_predictions'], 1):
                print(f"    {j}. {top_pred['class_name']} (索引: {top_pred['class_idx']}, 概率: {top_pred['probability']:.4f})")
    
    # 保存结果（如果指定了输出文件）
    if args.output_file:
        output_data = {
            'num_images': len(results),
            'results': results
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n推理结果已保存到: {args.output_file}")
    
    print("\n推理完成!")
    
    return results


if __name__ == '__main__':
    main()
