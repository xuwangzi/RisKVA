# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" trl's example script
Train Gemma-3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Gemma-3 on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-MMIU-Benchmark \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear
    --attn_implementation eager
"""

""" my script

Train Gemma-3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    src/single_image_risk/train.py \
    --dataset_name datasets/HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path models/pretrained_models/google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft \
    --bf16 True\
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Qwen2.5-VL on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    src/single_image_risk/train.py \
    --dataset_name datasets/HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path models/pretrained_models/Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft \
    --bf16 True\
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Qwen2.5-VL on the datasets/sft_data/single_image_tiny_247 local dataset (single-image).

accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    src/single_image_risk/train.py \
    --dataset_name datasets/sft_data/single_image_tiny_247 \
    --model_name_or_path models/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir models/finetuned_models/RisKVA/Qwen2.5-VL-7B-Instruct_SFT_single_image_tiny_247 \
    --bf16 True\
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager \
    --report_to tensorboard

Tensorboard:
tensorboard --logdir models/finetuned_models/RisKVA
"""

import io
import os
import zipfile

import torch
from datasets import DatasetDict, load_dataset # type: ignore
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl import ( # type: ignore
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# For multi-image example
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


def format_data(samples: dict[str, any]) -> dict[str, list]:
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples


# For multi-image example
def prepare_dataset(dataset: DatasetDict, dataset_name: str, dataset_train_split: str) -> DatasetDict:
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]

    for zip_filename in zip_files:
        zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

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
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    def collate_fn(examples):
        # 转换数据格式：conversations -> messages
        processed_examples = []
        for example in examples:
            # 检查是否有conversations字段（本地数据集）
            if "conversations" in example:
                # 转换conversations格式到messages格式
                messages = []
                for conv in example["conversations"]:
                    role = conv["from"]
                    if role == "system":
                        role = "system"
                    elif role == "user":
                        role = "user" 
                    elif role == "assistant":
                        role = "assistant"
                    
                    content = conv["value"]
                    # 创建标准的messages格式
                    if "<image>" in content:
                        # 包含图片的用户消息
                        text_content = content.replace("<image>", "").strip()
                        messages.append({
                            "role": role,
                            "content": [
                                {"type": "image", "image": None},  # 图片稍后处理
                                {"type": "text", "text": text_content}
                            ]
                        })
                    else:
                        # 纯文本消息
                        messages.append({
                            "role": role,
                            "content": [{"type": "text", "text": content}]
                        })
                
                # 处理图片
                if "image_base64" in example and example["image_base64"]:
                    import base64
                    from io import BytesIO
                    # 解码base64图片
                    image_data = base64.b64decode(example["image_base64"])
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    images = [image]
                else:
                    images = []
                
                processed_examples.append({"messages": messages, "images": images})
            else:
                # 使用原有的格式
                processed_examples.append(example)
        
        # 生成文本
        texts = [
            processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).strip()
            for example in processed_examples
        ]
        
        # 处理图片
        if "images" in processed_examples[0]:  # single-image
            images = [example["images"] for example in processed_examples]
        else:  # multi-image
            images = [process_vision_info(example["messages"]) for example in processed_examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        
        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Try to mask image tokens with different possible token names
        possible_image_tokens = ["boi_token", "image_token", "<image>", "<img>", "<vision_start>", "<|image_start|>"]
        for token_name in possible_image_tokens:
            try:
                if token_name in processor.tokenizer.special_tokens_map:
                    image_token_id = processor.tokenizer.convert_tokens_to_ids(
                        processor.tokenizer.special_tokens_map[token_name]
                    )
                    labels[labels == image_token_id] = -100
                    break
                elif hasattr(processor.tokenizer, 'convert_tokens_to_ids'):
                    # Try direct token conversion
                    image_token_id = processor.tokenizer.convert_tokens_to_ids(token_name)
                    if image_token_id != processor.tokenizer.unk_token_id:
                        labels[labels == image_token_id] = -100
                        break
            except (KeyError, AttributeError):
                continue
        
        # Also try to mask commonly used image token IDs
        possible_image_token_ids = [262144]
        for token_id in possible_image_token_ids:
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch  # Return the prepared batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
        dataset = prepare_dataset(dataset, script_args.dataset_name, script_args.dataset_train_split)

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


if __name__ == "__main__":
    main()
