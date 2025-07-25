"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import torch
import yaml  # type: ignore
import os
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class CustomScriptArguments(ScriptArguments):
    """扩展的脚本参数，添加自定义聊天模板配置"""
    prompt_config_path: Optional[str] = field(
        default="configs/prompts/building_risk_prompts.yaml",
        metadata={"help": "Path to the custom prompt configuration file"}
    )
    use_custom_template: bool = field(
        default=True,
        metadata={"help": "Whether to use custom chat template from config file"}
    )


def load_prompt_config(config_path: str) -> dict:
    """加载提示词配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_custom_chat_template(messages: list, prompt_config: dict) -> str:
    """
    使用自定义配置创建聊天模板
    
    Args:
        messages: 原始消息列表
        prompt_config: 提示词配置
    
    Returns:
        格式化后的聊天文本
    """
    # 获取系统提示词
    system_prompt = prompt_config.get('system_prompts', {}).get('building_defect_expert', '')
    
    # 构建对话
    formatted_messages = []
    
    # 添加系统消息
    if system_prompt:
        formatted_messages.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    
    # 处理用户和助手消息
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'user':
            # 如果用户消息为空或只包含图片，使用默认分析请求
            if not content or content.strip() == "":
                content = prompt_config.get('user_prompts', {}).get('standard_analysis', 
                                                                   '请分析这张图片中的建筑工程问题。')
            formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            formatted_messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # 如果最后一个消息不是助手消息，添加助手开始标记
    if messages and messages[-1].get('role') != 'assistant':
        formatted_messages.append("<|im_start|>assistant\n")
    
    return "\n".join(formatted_messages)


if __name__ == "__main__":
    # parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # 加载自定义提示词配置
    prompt_config = None
    if script_args.use_custom_template:
        try:
            prompt_config = load_prompt_config(script_args.prompt_config_path)
            print(f"✅ 成功加载自定义提示词配置: {script_args.prompt_config_path}")
        except Exception as e:
            print(f"⚠️ 加载自定义提示词配置失败: {e}")
            print("使用默认聊天模板")
            script_args.use_custom_template = False

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,  # todo: flash attention
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        if script_args.use_custom_template and prompt_config:
            # 使用自定义聊天模板
            texts = [create_custom_chat_template(example["messages"], prompt_config) for example in examples]
        else:
            # 使用默认聊天模板
            texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        
        images = [example["images"] for example in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
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