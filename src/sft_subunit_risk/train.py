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

作者: RisKVA Team
创建时间: 2024
"""

# 标准库导入
import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import torch
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from PIL import Image
from transformers import (
    AutoModelForImageTextToText, 
    AutoProcessor, 
    LlavaForConditionalGeneration,
    PreTrainedModel
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

# =============================================================================
# 数据集预处理模块
# =============================================================================

def prepare_dataset(
    dataset: DatasetDict, 
    dataset_path: str, 
    dataset_train_split: str, 
    templates: Optional[Dict[str, Any]] = None
) -> DatasetDict:
    """ todo 添加图像增强处理
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
    logging.info(f"开始准备数据集，训练分割: {dataset_train_split}")
    
    # 转换训练集数据
    train_data = transform_dataset(dataset[dataset_train_split], dataset_path)
    
    # 创建新的数据集对象
    new_train_dataset = Dataset.from_dict(train_data)
    
    # 构建新的DatasetDict
    new_dataset = DatasetDict({
        dataset_train_split: new_train_dataset
    })
    
    logging.info(f"数据集准备完成，样本数量: {len(new_train_dataset)}")
    return new_dataset


def transform_dataset(dataset_split: DatasetDict, dataset_path: str) -> Dict[str, List[Any]]:
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
    logging.info(f"开始转换数据集，样本数量: {len(dataset_split)}")
    
    transformed_data = {
        'messages': [],
        'images': []
    }
    
    for idx, sample in enumerate(dataset_split):
        try:
            formatted = format_sample(sample, dataset_path)
            transformed_data['messages'].append(formatted['messages'])
            transformed_data['images'].append(formatted['images'])
            
            # 清理中间变量，释放内存
            del formatted
            
            # 每处理1000个样本打印一次进度
            if (idx + 1) % 1000 == 0:
                logging.info(f"已处理样本: {idx + 1}/{len(dataset_split)}")
                
        except Exception as e:
            logging.error(f"处理样本 {idx} 时出错: {e}")
            continue
    
    # 主动触发垃圾回收
    gc.collect()
    logging.info("数据集转换完成")
    return transformed_data

    
def format_sample(sample: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:
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
    # 1. 加载图像
    all_image_paths = sample.get('all_image_paths', '[]')
    images = get_images_from_paths(all_image_paths, dataset_path)
    
    # 2. 提取样本字段，提供默认值
    defect_description = sample.get('defect_description_text', '未知缺陷')
    risk_detail = sample.get('risk_detail', '风险详情未提供')
    correction_suggestion = sample.get('correction_suggestion', '建议未提供')
    risk_level_original = sample.get('risk_level_original', '风险等级（原本）未提供')
    risk_level_current = sample.get('risk_level_current', '风险等级（现在）未提供')
    
    # 3. 构建用户问题
    image_count = sample.get('image_count', 0)
    base_instruction = '''作为房屋分户检查专家，你现在正在进行分户检查分析。请依据图像可见证据与项目适用规范进行判断。

【项目阶段与评估范围】
- 阶段：TIS 完工阶段。
- 评估范围：以“施工”为主，涉及部位/专业包括：主体、安装、装饰装修、门窗、隔墙、防水、保温。
- 检查内容类别：质量行为与工程实体为重点。

【分户检查对象与区域】
- 检查区域、部位：楼、地下室；户内、公区楼层、屋面、外墙、地下室等。

【非全装修住宅分户检查要点】
- 室内地面：外观质量、地面高差、细部质量、坡度、踢脚线。
- 室内墙面/顶棚：外观质量、细部质量、渗漏。
- 建筑门窗：猫眼/门铃、门框/门扇/门轴/门吸、门锁/把手、窗框/窗扇（含防脱落装置）、玻璃、密封条、防渗漏、窗把手、窗台施工与防护。
- 护栏栏杆：成品质量、安装质量、安装高度、垂直杆间距。
- 给排水：管道、龙头/角阀、地漏安装与坡度、检修门。
- 室内电气：强电箱外观与回路安装、开关/插座安装与高度、回路控制标识。
- 通风与空调：排气/排烟口、空调预留洞口。
- 其他：烟道外观质量、止逆阀等。

【全装修住宅分户检查要点】
- 地板铺装：安装方式、牢固性、踢脚线。
- 瓷砖/石材：门槛石/窗台石外观与高度、表观质量、铺贴方式、接缝高低与填充、空鼓、细部质量、地面坡度与清洁。
- 涂料/墙纸/软包：基层基面、涂刷质量与效果、异色分界、铺贴质量、细部质量。
- 吊顶/橱柜/扶手：外观完整性、细部质量、厨卫净高、门板/柜身/抽屉与配件、台面水平度、收边收口、踢脚线、窗帘盒接缝、护栏与护手、窗台板安装及表观与接缝处理、门窗套安装与表观。
- 门窗安装：猫眼/门铃、门框/门扇/门窗套、门轴/门吸、门锁/把手、窗框/窗扇（防脱落）、玻璃、密封条、防渗漏、窗把手、窗台施工与防护。
- 卫生洁具：外观/安装、龙头/阀门/角阀、小五金、台盆下水管、密封、淋浴房（表观/安装/挡水条与打胶/密封性/给排水系统）、地漏排水。
- 电气安装：强电箱外观与回路安装、开关面板质量与高度、插座安装与标识、灯具/筒灯安装与回路标识、设备完整性。
- 电器设备：内外机、小家电、设备运行情况。
- 管线安装：新风管道、吊顶内管线路与防水、中央空调内机、冷媒管、给排水。
- 水暖功能：洁具/水槽系统、地漏与地面泛水坡度。

【公共部位检查要点】
- 地下室/底层车库：地面与墙面/顶棚涂饰、管道安装、防渗漏细部（底板/外墙/顶板/穿墙管等）、通风与排风/排烟、支架、控制器与阀门、喷头。
- 楼（电）梯通道与走廊：地面与墙面饰面、楼梯踏步/平台尺寸、疏散外门与通道尺寸、栏杆/扶手尺寸与表面质量、电梯运行。
- 外立面：抹灰与涂饰表观、幕墙安装与板缝注胶、雨棚根部防水与排水坡度/管设置。
- 屋面：防水卷材、细部构造、排汽构造、分格缝与注胶、门窗框缝注胶、排气管/烟风帽、雨水管/雨水斗、栏杆高度与间距、女儿墙坡度。
- 户外安装：配电箱配线/漏保/接地、避雷针/避雷带、室外电气；排水管道接口、管道保温、给水设备噪声。
- 室外绿化及无障碍：土方回填、道路与通道面层、路缘石、隔离栅；无障碍出入口/入口平台/公共走道/候梯厅。

【政策文件】
- 《上海市推进住宅工程质量潜在缺陷保险实施办法》〔2024〕4 号；
- 《上海市住宅工程质量潜在缺陷保险实施细则》〔2025〕2 号；
- 《上海市建设工程质量风险管理机构管理规定》〔2023〕12 号；
- 《上海市建设工程质量风险管理机构管理导则》IDI-TIS-SH-2024-03；
- 《上海市住宅工程质量分户验收管理办法》〔2024〕7 号。

【规范标准（按项目特点取舍增减）】
- 《建筑工程施工质量验收统一标准》GB50300
- 《住宅设计规范》GB50096
- 《民用建筑设计统一标准》GB50352
- 《住宅设计标准》DGJ08-20
- 《建筑抗震设计规范》GB50011
- 《民用建筑供暖通风与空气调节设计规范》GB50736
- 《混凝土结构工程施工质量验收规范》GB50204
- 《地下防水工程质量验收规范》GB50208
- 《住宅室内防水工程技术规范》JGJ298
- 《建筑外墙防水工程技术规程》JGJ/T 235
- 《外墙外保温工程技术规程》JGJ144
- 《铝合金门窗工程技术规范》JGJ214
- 《金属与石材幕墙工程技术规范》JGJ133
- 《屋面工程质量验收规范》GB50207
- 《屋面工程技术规范》GB50345
- 《建筑装饰装修工程质量验收标准》GB50210
- 《建筑电气工程施工质量验收规范》GB50303
- 《智能建筑工程质量验收规范》GB50339
- 《通风与空调工程施工质量验收规范》GB50243
- 《建筑给水排水及采暖工程施工质量验收规范》GB50242
- 《建筑节能工程施工质量验收标准》GB50411

【与保险标的范围相关的缺陷关注重点（用于风险认定提示）】
- 10 年期：整体/局部倒塌；地基不均匀沉降超限；悬挑构件与外墙结构构件坍塌/脱落/影响使用安全裂缝；主体承重结构影响结构安全的变形/裂缝/破损/断裂；法律法规强制性标准规定的其他情形。
- 5 年期：围护结构保温工程；地下/屋面/外墙/室内防水工程。
- 2 年期：装饰装修工程（含外墙饰面、室内二次结构、全/非全装修）；电气管线/给排水管道与设备安装工程；供热与供冷系统安装工程。

【风险等级定义（TIS编码，输出时请使用代码与中文名）】
- A — 正常技术风险（color: blue）
  - 描述：风险等级低；无影响结构/使用安全的质量缺陷；可能有轻微使用功能影响，发生概率极低；基本不造成或仅少量财产损失。
  - 要求：一般风险提醒。
- B — 轻微技术风险（color: yellow）
  - 描述：风险等级较低；无影响结构安全；对使用安全/功能有轻微影响；发生概率低；可能造成轻微财产损失。
  - 要求：风险提示；需引起注意，防止同类问题反复出现，并整改、回复TIS机构。
- C — 中等技术风险（color: orange）
  - 描述：风险等级中等；对结构安全有轻微影响，或对使用安全/功能影响中等；发生概率低或较高；可造成一定财产损失。
  - 要求：风险警示；需落实并采取措施、整改、回复TIS机构。
- D — 严重技术风险（color: red）
  - 描述：风险等级较高；影响结构/使用安全，或事故概率高，严重影响使用功能；可能造成严重财产损失或恶劣社会影响。
  - 要求：严重风险警示；落实并立即采取整改，回复TIS机构。
- E — 严重质量行为风险（color: red）
  - 描述：风险等级较高；违反政府质量管理规定/规范性文件，未履行法定程序或未落实质量管理措施，可能造成较大社会影响的行为。
  - 要求：严重风险警示；落实并立即采取整改，回复TIS机构。
- R — 技术风险保留（color: gray）
  - 描述：检查中未能查见必要工序、材料或控制资料以证明其无技术风险；需进一步资料确认实际风险等级。
  - 要求：持续收集（材料与证据）并复核。
- O — 观察项（color: green）
  - 描述：超出本行业技术知识领域或超过国家/项目地标准规范要求，但基于TIS经验存在物质损坏风险。
  - 要求：持续关注与复查。'''

    format_template = (
        "请按照以下格式回答：\n"
        "【缺陷】：\n"
        "【风险】：\n"
        "【风险等级（原本）】：\n"
        "【风险等级（现在）】：\n"
        "【纠正和预防建议】：\n"
    )
    
    if image_count > 0:
        user_text = (
            f"{base_instruction}\n"
            f"本次分析提供了{image_count}张图像。"
            f"根据提供的{image_count}张图像，请分析对应的\"缺陷\"、\"风险\"、"
            f"\"风险等级（原本）\"和\"风险等级（现在）\"，并提供\"纠正和预防建议\"。\n"
            f"{format_template}"
        )
    else:
        user_text = (
            f"{base_instruction}\n"
            f"本次分析没有提供图像（空白图像作为占位图片，请忽略），"
            f"但是已知\"缺陷\"是：{defect_description}。"
            f"根据提供的\"缺陷\"文本，请分析对应的\"缺陷\"、\"风险\"、"
            f"\"风险等级（原本）\"和\"风险等级（现在）\"，并提供\"纠正和预防建议\"。\n"
            f"{format_template}"
        )
    
    # 4. 构建用户消息内容
    user_content = [{'index': None, 'text': user_text, 'type': 'text'}]
    
    # 添加图像引用
    for i in range(len(images)):
        user_content.append({'index': i, 'text': None, 'type': 'image'})
    
    # 5. 构建助手回答
    assistant_text = (
        f"【缺陷】：{defect_description}\n"
        f"【风险】：{risk_detail}\n"
        f"【风险等级（原本）】：{risk_level_original}\n"
        f"【风险等级（现在）】：{risk_level_current}\n"
        f"【纠正和预防建议】：{correction_suggestion}"
    )
    
    assistant_content = [{'index': None, 'text': assistant_text, 'type': 'text'}]
    
    # 6. 构建完整的对话
    messages = [
        {'content': user_content, 'role': 'user'},
        {'content': assistant_content, 'role': 'assistant'}
    ]
    
    result = {
        'messages': messages,
        'images': images
    }
    
    # 清理局部变量以释放内存
    del messages, images, user_content, assistant_content
    return result
    

def get_images_from_paths(all_image_paths: str, dataset_path: str) -> List[Image.Image]:
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
    
    try:
        image_paths = json.loads(all_image_paths)
    except json.JSONDecodeError:
        logging.warning(f"无法解析图片路径JSON: {all_image_paths}")
        image_paths = []
    
    # 确定基础路径
    base_path = Path(dataset_path).parent if dataset_path.endswith('.csv') else Path(dataset_path)
    
    # 加载每张图片
    for image_path in image_paths:
        try:
            full_img_path = base_path / image_path
            
            # 使用with语句确保图像文件句柄被正确关闭
            with Image.open(full_img_path) as img:
                # 创建副本并转换为RGB格式，避免懒加载问题
                image = img.convert("RGB").copy()
                
                # 限制图像最大尺寸以减少显存占用
                max_size = 1024
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                images.append(image)
                
        except FileNotFoundError:
            logging.error(f"图片文件不存在: {full_img_path}")
            continue
        except Exception as e:
            logging.error(f"加载图片时出错 {image_path}: {e}")
            continue
    
    # 如果没有成功加载任何图片，创建占位图片
    if not images:
        placeholder = Image.new('RGB', (224, 224), color='white') # 使用标准尺寸的白色占位图片
        images.append(placeholder)
        # logging.info("使用白色占位图片")
    
    return images


# =============================================================================
# 数据整理器 (Data Collator)
# =============================================================================

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
    # 1. 应用对话模板并提取文本和图像
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) 
             for example in examples]
    images = [example["images"] for example in examples]

    # 2. 分词化文本并处理图像
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # 3. 创建训练标签
    labels = batch["input_ids"].clone()
    
    # 遮蔽padding token，不参与损失计算
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # 遮蔽图像token，不参与损失计算（模型特定）
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    
    # 4. 清理中间变量，释放内存
    del texts, images, labels

    # 5. 可选的周期性内存清理
    _perform_optional_cleanup(examples)

    return batch


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
                logging.info(f"\n[rank={rank} device={device_str}] 🔍 清理前显存状态：")
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


# =============================================================================
# 全局变量和配置
# =============================================================================

# 周期性内存清理配置（通过环境变量控制）
CLEANUP_EVERY_N = int(os.getenv("CLEANUP_EVERY_N", "0"))
_collate_batch_count = 0  # 仅在主进程中递增

# 全局处理器变量（在主函数中初始化）
processor: Optional[AutoProcessor] = None


# =============================================================================
# 主训练流程
# =============================================================================

def main() -> None:
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
    
    # 配置训练参数
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    
    logging.info("开始初始化模型和处理器...")
    
    # 初始化模型、分词器和处理器
    global processor
    processor, model = _initialize_model_and_processor(model_args)
    
    logging.info("开始加载和预处理数据集...")

    # 只在主进程中计算数据预处理，提高效率
    # 参考: https://github.com/huggingface/trl/pull/1255
    state = PartialState()
    with state.local_main_process_first():
        # 加载和预处理数据集
        dataset = _load_and_prepare_dataset(script_args)
    
    logging.info("开始配置训练器...")
    
    # 训练前内存清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 配置训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    logging.info("开始训练...")
    trainer.train()

    logging.info("训练完成，保存模型...")
    _save_and_upload_model(trainer, training_args, script_args)
    
    logging.info("训练流程全部完成！")


def _initialize_model_and_processor(model_args: ModelConfig) -> Tuple[AutoProcessor, PreTrainedModel]:
    """
    初始化模型和处理器。
    
    Args:
        model_args (ModelConfig): 模型配置参数
        
    Returns:
        Tuple[AutoProcessor, PreTrainedModel]: (processor, model) 处理器和模型对象
        
    Raises:
        ValueError: 当模型配置参数无效时
        OSError: 当模型文件无法加载时
    """
    # 确定数据类型
    torch_dtype = (
        model_args.torch_dtype 
        if model_args.torch_dtype in ["auto", None] 
        else getattr(torch, model_args.torch_dtype)
    )
    
    # 获取量化配置
    quantization_config = get_quantization_config(model_args)
    
    # 构建模型参数
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # 注释掉device_map以避免与DeepSpeed冲突
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    try:
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=model_args.trust_remote_code
        )

        # 加载模型
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=model_args.trust_remote_code, 
            **model_kwargs
        )
        
        logging.info(f"成功加载模型: {model_args.model_name_or_path}")
        return processor, model
        
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise OSError(f"无法加载模型 {model_args.model_name_or_path}: {e}") from e


def _load_and_prepare_dataset(script_args: ScriptArguments) -> DatasetDict:
    """
    加载和预处理数据集。
    
    Args:
        script_args (ScriptArguments): 脚本参数配置
        
    Returns:
        DatasetDict: 预处理后的数据集
        
    Raises:
        FileNotFoundError: 当数据集文件不存在时
        ValueError: 当数据集格式不正确时
    """
    try:
        if script_args.dataset_name.endswith('.csv'):
            # 从CSV文件加载数据集
            if not Path(script_args.dataset_name).exists():
                raise FileNotFoundError(f"数据集文件不存在: {script_args.dataset_name}")
            
            dataset = load_dataset('csv', data_files=script_args.dataset_name)
            dataset = prepare_dataset(
                dataset, 
                script_args.dataset_name, 
                script_args.dataset_train_split, 
                templates=None
            )
            logging.info("成功从CSV文件加载数据集")
        else:
            # 从HuggingFace Hub加载数据集
            dataset = load_dataset(
                script_args.dataset_name, 
                name=script_args.dataset_config
            )
            logging.info("成功从HuggingFace Hub加载数据集")
    
        return dataset
        
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        if isinstance(e, FileNotFoundError):
            raise
        else:
            raise ValueError(f"数据集格式错误或加载失败: {e}") from e


def _save_and_upload_model(
    trainer: SFTTrainer, 
    training_args: SFTConfig, 
    script_args: ScriptArguments
) -> None:
    """
    保存模型并可选地上传到Hub。
    
    Args:
        trainer (SFTTrainer): 训练器对象
        training_args (SFTConfig): 训练配置
        script_args (ScriptArguments): 脚本参数
        
    Raises:
        OSError: 当模型保存失败时
        ConnectionError: 当推送到Hub失败时
    """
    try:
        # 保存模型
        trainer.save_model(training_args.output_dir)
        logging.info(f"模型已保存到: {training_args.output_dir}")
        
        # 可选地推送到Hub
        if training_args.push_to_hub:
            try:
                trainer.push_to_hub(dataset_name=script_args.dataset_name)
                
                # 只在主进程中推送处理器
                if trainer.accelerator.is_main_process:
                    if processor is not None:
                        processor.push_to_hub(training_args.hub_model_id)
                        logging.info(f"模型和处理器已推送到Hub: {training_args.hub_model_id}")
                    else:
                        logging.warning("处理器为None，无法推送到Hub")
                        
            except Exception as e:
                logging.error(f"推送到Hub失败: {e}")
                raise ConnectionError(f"无法推送模型到Hub: {e}") from e
                
    except Exception as e:
        if not isinstance(e, ConnectionError):
            logging.error(f"保存模型失败: {e}")
            raise OSError(f"无法保存模型到 {training_args.output_dir}: {e}") from e
        else:
            raise


if __name__ == "__main__":
    main()
