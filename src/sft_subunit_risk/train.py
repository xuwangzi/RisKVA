#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RisKVA 风险评估模型训练脚本

本脚本用于训练基于视觉语言模型的房屋分户检查风险评估模型。
支持多模态输入（图像+文本），能够分析缺陷并评估风险等级。

主要功能：
- 数据预处理和格式转换
- 多模态模型训练
- 内存优化和显存管理
- 分布式训练支持

作者: xuwangzi
创建时间: 2025
"""

# 标准库导入
import gc
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from accelerate import PartialState
import torch
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor, 
)

# TRL 相关导入
from trl import (  # type: ignore
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)



################
# Prepare dataset todo：优化prompt
################
def prepare_dataset(dataset: DatasetDict, dataset_path: str, dataset_train_split: str, templates: Optional[Dict[str, Any]] = None) -> DatasetDict:

    """
    准备训练数据集，将原始数据转换为模型可用的格式。
    
    Args:
        dataset (DatasetDict): 原始数据集
        dataset_path (str): 数据集文件路径，用于定位图像文件
        dataset_train_split (str): 训练集分割名称，通常为'train'
        templates (Optional[Dict[str, Any]]): 可选的模板配置，暂未使用
        
    Returns:
        DatasetDict: 转换后的数据集，包含messages和images字段
        
    Note:
        转换后的数据格式符合TRL SFTTrainer的要求：
        - messages: 对话格式的文本数据
        - images: PIL图像对象列表
    """

    # 🔍 调试
    # print(f"prepare_dataset: {dataset}")
    
    # 转换训练集
    train_data = transform_dataset(dataset[dataset_train_split], dataset_path)
    
    # 创建新的数据集
    new_train_dataset = Dataset.from_dict(train_data)
    
    # 构建新的DatasetDict
    new_dataset = DatasetDict({
        dataset_train_split: new_train_dataset
    })
    
    # print(f"转换完成，新数据集: {new_dataset}") # 🔍 调试
    return new_dataset


# 对数据集进行转换
def transform_dataset(dataset_split: DatasetDict, dataset_path: str):
    """
    转换数据集分割为模型训练格式。
    
    将原始数据集中的每个样本转换为包含对话消息和图像的格式，
    适用于多模态语言模型训练。
    
    Args:
        dataset_split (DatasetDict): 数据集的某个分割（如训练集）
        dataset_path (str): 数据集路径，用于构建图像文件的完整路径
        
    Returns:
        Dict[str, List[Any]]: 转换后的数据，包含：
            - messages: 对话格式的消息列表
            - images: 对应的图像列表
            
    Note:
        为了优化内存使用，函数会在处理过程中主动清理中间变量
        并触发垃圾回收。
    """

    transformed_data = {
        'messages': [],
        'images': []
    }
    
    for idx, sample in enumerate(dataset_split):
        # 原始样本
        formatted = format_sample(sample, dataset_path)
        transformed_data['messages'].append(formatted['messages'])
        transformed_data['images'].append(formatted['images'])

        # # 图像增强
        # # 跳过空白占位图
        # if not (len(formatted['images']) == 1 and all(v == 255 for v in formatted['images'][0].getpixel((0,0)))):
        #     for i in range(2): 
        #         transformed_data['messages'].append(formatted['messages'])
        #         transformed_data['images'].append(augment_image_multi(formatted['images']))
            
        # 清理中间变量，释放内存
        del formatted
        
        # 每处理1000个样本打印一次进度
        if (idx + 1) % 1000 == 0:
            logging.info(f"已处理样本: {idx + 1}/{len(dataset_split)}")
    
    # ♻️ 主动触发垃圾回收
    gc.collect()
    return transformed_data

    
def format_sample(sample: DatasetDict, dataset_path: str):
    """
    将单个数据样本转换为模型训练所需的对话格式。
    
    此函数是数据预处理的核心，将原始的房屋检查数据转换为
    多模态对话格式，包括用户问题和助手回答。
    
    Args:
        sample (Dict[str, Any]): 原始数据样本，包含图像路径和文本字段
        dataset_path (str): 数据集根路径，用于构建图像的完整路径
        
    Returns:
        Dict[str, Any]: 格式化后的样本，包含：
            - messages: 用户和助手的对话消息
            - images: PIL图像对象列表
            
    Note:
        支持有图像和无图像两种情况：
        - 有图像：要求模型基于图像进行分析
        - 无图像：提供缺陷文本描述，要求基于文本分析
    """

    # 1. 加载图片
    # todo: 使用懒加载，改为图片路径
    images = []
    
    # 处理图片路径 - 适配不同的数据格式
    # JSON数组格式的图片路径: ["images/xxx.jpg"] 或 ["img1.jpg", "img2.jpg"]
    all_image_paths = sample['all_image_paths']
    images = get_images_from_paths(all_image_paths, dataset_path)
    
    # 2. 构建对话格式的messages
    # 适配不同字段名称的缺陷描述
    defect_description_text = (sample.get('defect_description_text') or '未知缺陷')
    risk_detail = (sample.get('risk_detail') or '风险详情未提供')
    correction = (sample.get('correction_suggestion') or '建议未提供')
    risk_level_original = (sample.get('risk_level_original') or '风险等级（原本）未提供')
    risk_level_current = (sample.get('risk_level_current') or '风险等级（现在）未提供')
    
    # 构建用户问题（包含文本和图像引用）
    # 区分有图像和无图像的情况
    user_text = "作为房屋分户检查专家，你现在正在进行分户检查分析。"
    image_count = sample.get('image_count', 0)
    if image_count > 0: 
        user_text += f"本次分析提供了{image_count}张图像。根据提供的{image_count}张图像，请分析对应的“缺陷”、“风险”、“风险等级（原本）”和“风险等级（现在）”，并提供“纠正和预防建议”。\n请按照以下格式回答：\n【缺陷】：\n【风险】：\n【风险等级（原本）】：\n【风险等级（现在）】：\n【纠正和预防建议】：\n"
    else:
        user_text += f"本次分析没有提供图像（空白图像作为占位图片，请忽略），但是已知“缺陷”是：{defect_description_text}。根据提供的“缺陷”文本，请分析对应的“缺陷”、“风险”、“风险等级（原本）”和“风险等级（现在）”，并提供“纠正和预防建议”。\n请按照以下格式回答：\n【缺陷】：\n【风险】：\n【风险等级（原本）】：\n【风险等级（现在）】：\n【纠正和预防建议】：\n"
    # 添加用户问题文本
    user_content = [
        {'index': None, 'text': user_text, 'type': 'text'}
    ]
    # 添加每张图片的图像引用
    for i in range(len(images)):
        user_content.append({'index': i, 'text': None, 'type': 'image'})            
    
    # 构建助手回答
    assistant_text = f"【缺陷】：{defect_description_text}\n【风险】：{risk_detail}\n【风险等级（原本）】：{risk_level_original}\n【风险等级（现在）】：{risk_level_current}\n【纠正和预防建议】：{correction}"

    assistant_content = [
        {'index': None, 'text': assistant_text, 'type': 'text'}
    ]
    
    # 构建完整的对话
    messages = [
        {'content': user_content, 'role': 'user'},
        {'content': assistant_content, 'role': 'assistant'}
    ]
    
    # ♻️ 确保及时释放局部变量
    result = {
        'messages': messages,
        'images': images
    }
    
    # ♻️ 清理局部变量
    del messages, images, user_content, assistant_content
    return result
    

def get_images_from_paths(all_image_paths: str, dataset_path: str):
    """
    根据图片路径字符串加载图像文件。
    
    处理JSON格式的图片路径数组，支持相对路径和绝对路径。
    如果没有图片路径，会创建占位图片以保持输入格式一致性。
    
    Args:
        all_image_paths (str): JSON格式的图片路径数组字符串
            例如: '["images/img1.jpg", "images/img2.jpg"]'
        dataset_path (str): 数据集根路径，用于解析相对路径
        
    Returns:
        List[Image.Image]: PIL图像对象列表
        
    Note:
        - 图像会被自动转换为RGB格式
        - 为了节省显存，图像尺寸会被限制为最大1024px
        - 如果没有图片，会创建224x224的白色占位图片
        
    Raises:
        json.JSONDecodeError: 当图片路径不是有效JSON格式时
        FileNotFoundError: 当图片文件不存在时
    """

    images = []
    image_paths = json.loads(all_image_paths)

    # 加载图片为PIL对象
    base_path = Path(dataset_path).parent if dataset_path.endswith('.csv') else Path(dataset_path)
    for image_path in image_paths:
        full_img_path = base_path / image_path
        # 使用with语句确保图像文件句柄被正确关闭
        with Image.open(full_img_path) as img:
            # 创建副本并立即转换为RGB，避免懒加载
            image = img.convert("RGB").copy()
            # ♻️ 限制图像最大尺寸以减少显存占用 (启用1024px解决OOM)
            max_size = 1024  # 大幅减少图像尺寸解决OOM
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            images.append(image)
    
    # 如果没有图片，创建占位图片
    if not images:
        placeholder = Image.new('RGB', (224, 224), color='white') # 使用标准尺寸的白色占位图片
        images.append(placeholder)
        # print("📷 使用占位图片") # 🔍 调试

    return images


def augment_image_multi(images: List[Image.Image]) -> List[Image.Image]:
    """
    输入多张PIL图像，对每张图片随机增强，返回增强后的图片列表（一一对应）。
    Args:
        images (List[PIL.Image]): 输入图像列表
    Returns:
        List[PIL.Image]: 增强后的图像列表
    """
    aug_images = []
    for image in images:
        img_aug = image.copy()
        # 随机水平翻转
        if random.random() < 0.5:
            img_aug = ImageOps.mirror(img_aug)
        # 随机垂直翻转
        if random.random() < 0.2:
            img_aug = ImageOps.flip(img_aug)
        # 随机旋转（90/180/270度）
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img_aug = img_aug.rotate(angle)
        # 随机亮度调整
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img_aug)
            factor = random.uniform(0.7, 1.3)
            img_aug = enhancer.enhance(factor)
        # 随机对比度调整
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img_aug)
            factor = random.uniform(0.7, 1.3)
            img_aug = enhancer.enhance(factor)
        # 随机锐化
        if random.random() < 0.2:
            img_aug = img_aug.filter(ImageFilter.SHARPEN)
        # 随机高斯模糊
        if random.random() < 0.1:
            img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        # 随机缩放（缩放到80%~120%之间，再resize回原尺寸）
        if random.random() < 0.3:
            w, h = img_aug.size
            scale = random.uniform(0.8, 1.2)
            new_w, new_h = int(w * scale), int(h * scale)
            img_aug = img_aug.resize((new_w, new_h), Image.BICUBIC)
            # 再中心裁剪或填充回原尺寸
            if scale >= 1.0:
                # 中心裁剪
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img_aug = img_aug.crop((left, top, left + w, top + h))
            else:
                # 填充
                new_img = Image.new(img_aug.mode, (w, h), (255, 255, 255))
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                new_img.paste(img_aug, (left, top))
                img_aug = new_img
        # 随机裁剪（裁剪后resize回原尺寸）
        if random.random() < 0.2:
            w, h = img_aug.size
            crop_scale = random.uniform(0.85, 1.0)
            crop_w, crop_h = int(w * crop_scale), int(h * crop_scale)
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            img_aug = img_aug.crop((left, top, left + crop_w, top + crop_h))
            img_aug = img_aug.resize((w, h), Image.BICUBIC)
        # 随机色彩抖动（色相/饱和度）
        if random.random() < 0.2:
            img_aug = ImageEnhance.Color(img_aug).enhance(random.uniform(0.7, 1.3))
        # 随机加噪声（简单高斯噪声）
        if random.random() < 0.1:
            import numpy as np
            arr = np.array(img_aug).astype(np.float32)
            noise = np.random.normal(0, 8, arr.shape)
            arr = arr + noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img_aug = Image.fromarray(arr)
        aug_images.append(img_aug)
    return aug_images


################
# Create a data collator to encode text and image pairs
################
def collate_fn(examples):
    """
    数据整理函数，将批次样本转换为模型输入格式。
    
    此函数是训练流程的关键组件，负责：
    1. 应用对话模板并分词化文本
    2. 处理图像数据
    3. 创建训练标签并遮蔽特殊token
    4. 内存管理和显存清理
    
    Args:
        examples (List[Dict[str, Any]]): 批次样本列表，每个样本包含messages和images
        
    Returns:
        Dict[str, torch.Tensor]: 模型训练所需的批次数据，包含：
            - input_ids: 分词后的输入序列
            - attention_mask: 注意力掩码
            - pixel_values: 图像像素值
            - labels: 训练标签（遮蔽了padding和图像token）
            
    Note:
        - 会自动遮蔽padding token和图像token的损失计算
        - 支持可选的周期性内存清理功能
        - 在分布式训练中只在rank 0打印调试信息
    """

    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    
    # ♻️ 清理中间变量，释放内存
    del texts, images, labels

    # ♻️ 可选：仅在主进程中每 N 个 batch 清理一次（通过环境变量 CLEANUP_EVERY_N 控制，默认关闭）
    _perform_optional_cleanup(examples)

    return batch

################
# Clean up memory periodically
################
def _perform_optional_cleanup(examples: List[Dict[str, Any]]) -> None:
    """
    执行可选的周期性内存清理。
    
    通过环境变量CLEANUP_EVERY_N控制清理频率，仅在主进程中执行，
    避免在DataLoader worker进程中进行CUDA清理导致的问题。
    
    Args:
        examples (List[Dict[str, Any]]): 当前批次的样本，用于调试输出
        
    Note:
        - 仅在CLEANUP_EVERY_N > 0时启用
        - 只在主进程中执行（worker_info is None）
        - 只在rank 0进程中打印调试信息
    """

    try:
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
    except Exception:
        worker_info = None

    # 只在主进程中执行清理，但是每个GPU进程都会执行自己的清理
    if worker_info is None:
        global _collate_batch_count
        _collate_batch_count += 1
        
        if CLEANUP_EVERY_N > 0 and (_collate_batch_count % CLEANUP_EVERY_N == 0):
            # 获取进程rank信息
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", "0"))
            except Exception:
                rank = int(os.getenv("RANK", "0"))
            
            device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

            # 只在rank 0打印调试信息
            if rank == 0:
                logging.info(f"[rank={rank} device={device_str}] 🔍 清理前显存状态：")
                os.system("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
                         "--format=csv,noheader,nounits")
                logging.info(f"[rank={rank} device={device_str}] 🔍 清理内存: {_collate_batch_count} 个 batch")
                logging.info(f"[rank={rank} device={device_str}] 第一个样本: {examples[0]}")

            # 执行内存和显存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if rank == 0:
                logging.info(f"[rank={rank} device={device_str}] 🔍 清理后显存状态：")
                os.system("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
                         "--format=csv,noheader,nounits")



if __name__ == "__main__":
    """
    主训练函数。
    
    执行完整的模型训练流程，包括：
    1. 解析命令行参数和配置
    2. 初始化模型、处理器和分词器
    3. 加载和预处理数据集
    4. 配置和启动训练
    5. 保存模型和推送到Hub
    """

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 解析命令行参数
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # ♻️ 可选的周期性清理设置（默认关闭）。设置环境变量 CLEANUP_EVERY_N 为正整数即可启用。
    CLEANUP_EVERY_N = int(os.getenv("CLEANUP_EVERY_N", "0"))
    _collate_batch_count = 0  # 仅在主进程中递增

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map() if quantization_config is not None else None, # 注释掉device_map避免与DeepSpeed冲突
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Dataset
    ################
    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    state = PartialState()
    with state.local_main_process_first():
        if script_args.dataset_name.endswith('.csv'):
            dataset = load_dataset('csv', data_files=f'{script_args.dataset_name}')
            dataset = prepare_dataset(dataset, script_args.dataset_name, script_args.dataset_train_split, templates=None)
            print("✅ load dataset from csv successfully") # 🔍 调试
        else:
            dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    # ♻️ 训练前清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
