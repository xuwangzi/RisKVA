import torch
import json
import re
from typing import List, Optional, Dict, Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from pathlib import Path

class RiskAssessmentInference:
    """房屋分户检查风险评估推理类"""
    
    def __init__(self, model_path: str, device: str = "auto", processor_path: str = None):
        """
        初始化推理模型
        
        Args:
            model_path: 模型路径
            device: 设备类型，默认"auto"
            processor_path: processor配置路径，默认与model_path相同
        """
        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        try:
            # 加载模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype="auto", 
                device_map=self.device
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.processor_path)
            
            print(f"模型加载成功：{self.model_path}")
            if self.processor_path != self.model_path:
                print(f"处理器路径：{self.processor_path}")
            
        except Exception as e:
            raise Exception(f"模型加载失败：{str(e)}")
    
    def _build_messages_with_images(self, image_paths: List[str]) -> List[Dict]:
        """
        构建包含图像的消息
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            messages格式的数据
        """
        image_count = len(image_paths)
        user_text = (
            "作为房屋分户检查专家，你现在正在进行分户检查分析。"
            f"本次分析提供了{image_count}张图像。根据提供的{image_count}张图像，"
            "请分析对应的\"缺陷\"、\"风险\"、\"风险等级（原本）\"和\"风险等级（现在）\"，"
            "并提供\"纠正和预防建议\"。\n"
            "请按照以下格式回答：\n"
            "【缺陷】：\n"
            "【风险】：\n"
            "【风险等级（原本）】：\n"
            "【风险等级（现在）】：\n"
            "【纠正和预防建议】：\n"
        )
        
        # 构建content列表
        content = [{"type": "text", "text": user_text}]
        
        # 添加图像
        for image_path in image_paths:
            content.append({
                "type": "image", 
                "image": f"file://{image_path}"
            })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def _build_messages_with_text(self, defect_text: str) -> List[Dict]:
        """
        构建包含缺陷文本的消息
        
        Args:
            defect_text: 缺陷描述文本
            
        Returns:
            messages格式的数据
        """
        user_text = (
            "作为房屋分户检查专家，你现在正在进行分户检查分析。"
            f"本次分析没有提供图像（空白图像作为占位图片，请忽略），但是已知\"缺陷\"是：{defect_text}。"
            "根据提供的\"缺陷\"文本，请分析对应的\"缺陷\"、\"风险\"、\"风险等级（原本）\"和\"风险等级（现在）\"，"
            "并提供\"纠正和预防建议\"。\n"
            "请按照以下格式回答：\n"
            "【缺陷】：\n"
            "【风险】：\n"
            "【风险等级（原本）】：\n"
            "【风险等级（现在）】：\n"
            "【纠正和预防建议】：\n"
        )
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            }
        ]
        
        return messages
    
    def _generate_response(self, messages: List[Dict]) -> str:
        """
        生成模型响应
        
        Args:
            messages: 输入消息
            
        Returns:
            模型生成的文本
        """
        try:
            # 准备推理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # 生成响应
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else ""
            
        except Exception as e:
            raise Exception(f"生成响应失败：{str(e)}")
    
    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        解析模型响应，提取结构化信息
        
        Args:
            response_text: 模型生成的文本
            
        Returns:
            解析后的JSON格式数据
        """
        result = {
            "缺陷": "",
            "风险": "",
            "风险等级（原本）": "",
            "风险等级（现在）": "",
            "纠正和预防建议": ""
        }
        
        # 定义正则表达式模式
        patterns = {
            "缺陷": r"【缺陷】：(.*?)(?=【|$)",
            "风险": r"【风险】：(.*?)(?=【|$)",
            "风险等级（原本）": r"【风险等级（原本）】：(.*?)(?=【|$)",
            "风险等级（现在）": r"【风险等级（现在）】：(.*?)(?=【|$)",
            "纠正和预防建议": r"【纠正和预防建议】：(.*?)(?=【|$)"
        }
        
        # 提取每个字段
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        return result
    
    def inference_with_images(self, image_paths: List[str]) -> Dict[str, str]:
        """
        基于图像进行推理
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            解析后的JSON格式结果
        """
        if not image_paths:
            raise ValueError("图像路径列表不能为空")
        
        # 验证图像路径
        import os
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在：{image_path}")
        
        # 构建消息
        messages = self._build_messages_with_images(image_paths)
        
        # 生成响应
        response_text = self._generate_response(messages)
        
        # 解析响应
        result = self._parse_response(response_text)
        
        return result
    
    def inference_with_text(self, defect_text: str) -> Dict[str, str]:
        """
        基于缺陷文本进行推理
        
        Args:
            defect_text: 缺陷描述文本
            
        Returns:
            解析后的JSON格式结果
        """
        if not defect_text.strip():
            raise ValueError("缺陷文本不能为空")
        
        # 构建消息
        messages = self._build_messages_with_text(defect_text)
        
        # 生成响应
        response_text = self._generate_response(messages)
        
        # 解析响应
        result = self._parse_response(response_text)
        
        return result


def create_inference_model(model_path: str, device: str = "auto", processor_path: str = None) -> RiskAssessmentInference:
    """
    创建推理模型实例
    
    Args:
        model_path: 模型路径
        device: 设备类型
        processor_path: processor配置路径，默认与model_path相同
        
    Returns:
        推理模型实例
    """
    return RiskAssessmentInference(model_path, device, processor_path)


# 便捷函数
def inference_with_images(model_path: str, image_paths: List[str], device: str = "auto", processor_path: str = None) -> Dict[str, str]:
    """
    基于图像进行风险评估推理（便捷函数）
    
    Args:
        model_path: 模型路径
        image_paths: 图像路径列表
        device: 设备类型
        processor_path: processor配置路径，默认与model_path相同
        
    Returns:
        解析后的JSON格式结果
    """
    model = RiskAssessmentInference(model_path, device, processor_path)
    return model.inference_with_images(image_paths)


def inference_with_text(model_path: str, defect_text: str, device: str = "auto", processor_path: str = None) -> Dict[str, str]:
    """
    基于缺陷文本进行风险评估推理（便捷函数）
    
    Args:
        model_path: 模型路径
        defect_text: 缺陷描述文本
        device: 设备类型
        processor_path: processor配置路径，默认与model_path相同
        
    Returns:
        解析后的JSON格式结果
    """
    model = RiskAssessmentInference(model_path, device, processor_path)
    return model.inference_with_text(defect_text)


# 示例用法
if __name__ == "__main__":
    # 示例1：使用便捷函数推理(图像或文本)

    # defect_text = "墙面出现裂缝"
    # result = inference_with_text(model_path, defect_text)
    # print("文本推理结果：")
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    # image_paths = ["path1", "path2", "path3"]
    # result = inference_with_images(model_path, image_paths)
    # print("图像推理结果：")
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    
    """ command: 
    python src/sft_subnunit_risk/inference.py 2>&1 | tee logs/inference_$(date +%Y%m%d_%H%M%S).log
    """

    # 示例2：创建模型实例复用(图像或文本)

    pretrained_model_path = "models/pretrained_models/Qwen/Qwen2.5-VL-3B-Instruct"
    finetuned_model_path = "models/finetuned_models/RisKVA/RisKVA-Qwen2.5-VL-3B-Instruct-sft-subunit-risk"

    dataset_path = "datasets/RisKVA/Subunit-Risk_original/metadata_with_image.csv"
    if dataset_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=f'{dataset_path}')
    else:
        dataset = load_dataset(dataset_path)

    # 预训练模型
    pretrained_model = create_inference_model(pretrained_model_path)
    print("🔍图像推理结果（预训练模型）：")
    for i in range(len(dataset['train'])):
        print(f"🔍第{i}个样本：")
        print(dataset['train'][i])
        # 加载图片
        images = []

        all_image_paths = dataset['train'][i]['all_image_paths']
        image_paths = json.loads(all_image_paths)

        base_path = Path(dataset_path).parent if dataset_path.endswith('.csv') else Path(dataset_path)
        for image_path in image_paths:
            full_img_path = base_path / image_path
            images.append(full_img_path)

        result = pretrained_model.inference_with_images(images)
        print(f"🔍根据第{i}个样本的图像，推理结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 微调模型
    finetuned_model = create_inference_model(finetuned_model_path, processor_path=pretrained_model_path)
    print("🔍图像推理结果（微调模型）：")
    for i in range(len(dataset['train'])):
        print(f"🔍第{i}个样本：")
        print(dataset['train'][i])
        images = []

        all_image_paths = dataset['train'][i]['all_image_paths']
        image_paths = json.loads(all_image_paths)

        base_path = Path(dataset_path).parent if dataset_path.endswith('.csv') else Path(dataset_path)
        for image_path in image_paths:
            full_img_path = base_path / image_path
            images.append(full_img_path)

        result = finetuned_model.inference_with_images(images)
        print(f"🔍根据第{i}个样本的图像，推理结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))