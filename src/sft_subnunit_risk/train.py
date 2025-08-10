import torch
import gc  # 添加垃圾回收模块
# gc.collect() # 清理CPU内存
# torch.cuda.empty_cache() # 清理GPU内存
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from transformers import AutoModelForImageTextToText, AutoProcessor, LlavaForConditionalGeneration
from typing import Dict, List, Any, Optional
import json
import os
import logging
from datetime import datetime
from PIL import Image
from pathlib import Path
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
    transformed_data = {
        'messages': [],
        'images': []
    }
    
    for sample in dataset_split:
        formatted = format_sample(sample, dataset_path)
        transformed_data['messages'].append(formatted['messages'])
        transformed_data['images'].append(formatted['images'])
        
        # ♻️ 清理中间变量，释放内存
        del formatted
    
    # ♻️ 主动触发垃圾回收
    gc.collect()
    return transformed_data

    
def format_sample(sample: DatasetDict, dataset_path: str):
    """将单个样本转换为对话格式"""
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
        user_text = f"本次分析提供了{image_count}张图像。根据提供的{image_count}张图像，请分析对应的“缺陷”、“风险”、“风险等级（原本）”和“风险等级（现在）”，并提供“纠正和预防建议”。\n请按照以下格式回答：\n【缺陷】：\n【风险】：\n【风险等级（原本）】：\n【风险等级（现在）】：\n【纠正和预防建议】：\n"
    else:
        user_text = f"本次分析没有提供图像（空白图像作为占位图片，请忽略），但是已知“缺陷”是：{defect_description_text}。根据提供的“缺陷”文本，请分析对应的“缺陷”、“风险”、“风险等级（原本）”和“风险等级（现在）”，并提供“纠正和预防建议”。\n请按照以下格式回答：\n【缺陷】：\n【风险】：\n【风险等级（原本）】：\n【风险等级（现在）】：\n【纠正和预防建议】：\n"
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
    根据图片路径获取图片
    # 处理图片路径 - 适配不同的数据格式
    # JSON数组格式的图片路径: ["images/xxx.jpg"] 或 ["img1.jpg", "img2.jpg"]
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


################
# Create a data collator to encode text and image pairs
################
def collate_fn(examples):
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
    # 避免在 DataLoader worker 进程中进行 CUDA 清理
    try:
        from torch.utils.data import get_worker_info  # 局部导入以避免顶层依赖
        worker_info = get_worker_info()
    except Exception:
        worker_info = None

    if worker_info is None:
        # 主进程
        global _collate_batch_count
        _collate_batch_count += 1
        if CLEANUP_EVERY_N > 0 and (_collate_batch_count % CLEANUP_EVERY_N == 0):
            # 🔍 仅在 rank0 打印显存状态与必要信息
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", "0"))
            except Exception:
                rank = int(os.getenv("RANK", "0"))
            device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

            if rank == 0:
                print(f"\n[rank={rank} device={device_str}] 🔍 清理前显存状态：")
                # 🔍 打印显存状态：GPU_ID, 名称, 利用率%, 已用显存MB, 总显存MB
                os.system("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
                print(f"[rank={rank} device={device_str}] 🔍 清理内存: {_collate_batch_count} 个 batch")
                print(f"[rank={rank} device={device_str}] 第一个example: {examples[0]}")

            # ♻️ 清理内存和显存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if rank == 0:
                print(f"[rank={rank} device={device_str}] 🔍 清理后显存状态：")
                os.system("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")

    return batch



if __name__ == "__main__":
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
