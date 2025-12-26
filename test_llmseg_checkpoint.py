#!/usr/bin/env python3
"""
测试脚本：加载训练好的 DeepSpeed checkpoint 并在300张标注图片上计算IoU
"""

import argparse
import os
import sys
import json
import random
import gc
import resource
from functools import partial
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import transformers
import cv2

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.robot_arm_dataset import RobotArmDataset
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    dict_to_cuda,
    intersectionAndUnionGPU,
)
from utils.dataset import collate_fn_new


def parse_args():
    parser = argparse.ArgumentParser(description="测试训练好的LLMSeg模型")
    
    # 模型路径
    parser.add_argument("--version", type=str, default="/opt/data/private/model/LISA_Plus_7b",
                       help="LLaVA模型路径")
    parser.add_argument("--vision-tower", type=str, default="/opt/data/private/model/clip-vit-large-patch14",
                       help="CLIP模型路径")
    parser.add_argument("--vision_pretrained", type=str, default="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth",
                       help="SAM模型权重路径")
    
    # Checkpoint路径
    parser.add_argument("--checkpoint", type=str, 
                       default="/opt/data/private/LLMSeg/runs/finetune_llmseg_robot_arm2/ckpt_model",
                       help="DeepSpeed checkpoint目录路径")
    
    # 数据集路径
    parser.add_argument("--dataset_base_dir", type=str, default="/opt/data/private/LLMSeg/dataset/raw_pic",
                       help="原图数据集路径")
    parser.add_argument("--gt_mask_base_dir", type=str, default="/opt/data/private/LLMSeg/dataset/GT_mask",
                       help="GT mask数据集路径")
    parser.add_argument("--sam_masks_base_dir", type=str, default="/opt/data/private/LLMSeg/dataset/sam_candidate",
                       help="SAM候选mask数据集路径")
    
    # 模型参数
    parser.add_argument("--image_size", type=int, default=896, help="图像尺寸")
    parser.add_argument("--model_max_length", type=int, default=512, help="模型最大长度")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"],
                       help="精度类型")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,out_proj",
                       help="LoRA目标模块")
    
    # 测试参数
    parser.add_argument("--threshold", type=float, default=0.5, help="IoU阈值")
    parser.add_argument("--output_file", type=str, default="./test_results_iou.txt",
                       help="结果输出文件路径")
    parser.add_argument("--local_rank", type=int, default=0, help="本地rank")
    
    # 添加其他必要的参数（参考finetune_llmseg_copy2.py）
    parser.add_argument("--conv_type", type=str, default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=0, help="数据加载器工作进程数")
    
    # 加载模式选项
    parser.add_argument("--load_model_only", action="store_true", default=True,
                       help="直接加载模型参数文件（最节省内存，推荐用于测试，默认启用）")
    parser.add_argument("--use_cpu_load", action="store_true", default=False,
                       help="使用CPU加载checkpoint（需要转换ZeRO checkpoint，内存占用大）")
    parser.add_argument("--use_deepspeed_load", action="store_true", default=False,
                       help="使用DeepSpeed引擎加载（需要更多GPU内存）")
    
    # 可视化参数
    parser.add_argument("--test_vis_dir", type=str, default="test_vis",
                       help="测试可视化保存目录（相对于checkpoint目录的父目录）")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="是否保存可视化结果")
    
    return parser.parse_args()


def init_tokenizer(args):
    """初始化tokenizer"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    _ = tokenizer.add_tokens("[SEG]")
    seg_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    args.seg_token_idx = seg_ids[-1]
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )
    return tokenizer


def init_LISA_model(args, tokenizer):
    """初始化LISA模型（参考finetune_llmseg_copy2.py）"""
    import warnings
    import transformers
    from transformers import logging as transformers_logging
    from peft import LoraConfig, get_peft_model
    
    # 设置 transformers 日志级别为 ERROR
    transformers_logging.set_verbosity_error()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        model_args = {
            "train_mask_decoder": False,
            "out_dim": 256,
            "ce_loss_weight": 1.0,
            "align_loss_weight": 1.0,
            "regression_loss_weight": 1.0,
            "align_temperature": 0.05,
            "seg_token_idx": args.seg_token_idx,
            "vision_pretrained": args.vision_pretrained,
            "vision_tower": args.vision_tower,
            "use_mm_start_end": True,
            "mm_vision_tower": args.vision_tower,
        }
        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.half

        model = LISAForCausalLM.from_pretrained(
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
        )
        
        if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower != args.vision_tower:
            old_vision_tower = model.config.mm_vision_tower
            model.config.mm_vision_tower = args.vision_tower
            if hasattr(model.get_model(), 'vision_tower') and old_vision_tower != args.vision_tower:
                delattr(model.get_model(), 'vision_tower')
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 创建 model_args 对象用于 initialize_vision_modules
    class ModelArgs:
        def __init__(self, args_dict):
            for k, v in args_dict.items():
                setattr(self, k, v)
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
            self.pretrain_mm_mlp_adapter = None
    
    model_args_obj = ModelArgs(model_args)
    model.get_model().initialize_vision_modules(model_args_obj)
    vision_tower = model.get_model().get_vision_tower()
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    vision_tower.to(dtype=torch_dtype, device=device)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    # init LoRA
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "lisa_attention_layers",
                                "lisa_final_attn",
                                "lisa_norm_final_attn",
                                "lisa_iou_head",
                                "lisa_embedding_head",
                                "lisa_dino_conv",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs",
                          "lisa_attention_layers", "lisa_final_attn", "lisa_norm_final_attn",
                          "lisa_iou_head", "lisa_embedding_head", "lisa_dino_conv"]
            ]
        ):
            p.requires_grad = True
    
    return model


def build_test_dataset(args, tokenizer):
    """构建测试数据集（使用所有300张图片）"""
    raw_pic_base_dir = args.dataset_base_dir
    gt_mask_base_dir = args.gt_mask_base_dir
    sam_candidate_base_dir = args.sam_masks_base_dir
    
    # 准备多个 JSON 文件路径列表（三个视角）
    json_paths = []
    sam_mask_helpers = {}
    
    for view_name in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
        annotations_path = os.path.join(gt_mask_base_dir, view_name, 'annotations.json')
        if os.path.exists(annotations_path):
            json_paths.append(annotations_path)
            
            # 为每个视角创建SAM mask helper
            sam_mask_dir = os.path.join(sam_candidate_base_dir, view_name)
            if os.path.exists(sam_mask_dir):
                sam_mask_helpers[view_name] = SAM_Mask_Reader_PNG(sam_mask_dir)
            else:
                print(f"Warning: SAM candidate directory not found: {sam_mask_dir}")
    
    # 构建所有样本（不使用train/val split，全部作为测试集）
    # 注意：
    # 1. 从300张原图中，每张原图对应的GT mask中选择一张来测试
    # 2. 过滤掉名字中带有"ring"或"pin"的GT mask（包括ring1、ring2、pin等）
    all_samples = []
    ring_filtered_count = 0
    skipped_images = 0
    processed_images_per_view = {}  # 记录每个视角处理的图片数
    
    for view_name in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
        annotations_path = os.path.join(gt_mask_base_dir, view_name, 'annotations.json')
        if not os.path.exists(annotations_path):
            continue
        with open(annotations_path, "r") as f:
            all_annotations = json.load(f)
        
        # 先按img_name分组，获取所有不同的图片及其GT mask
        img_to_annotations = {}
        for ann in all_annotations:
            img_name = ann['img_name']
            if img_name not in img_to_annotations:
                img_to_annotations[img_name] = []
            img_to_annotations[img_name].append(ann)
        
        # 获取前100张不同的图片（按img_name排序）
        sorted_img_names = sorted(img_to_annotations.keys())[:100]
        processed_images_per_view[view_name] = len(sorted_img_names)
        
        # 对每张图片，从它的GT mask中选择一张（过滤掉包含"ring"和"pin"的）
        for img_name in sorted_img_names:
            ann_list = img_to_annotations[img_name]
            # 过滤掉名字中带有"ring"或"pin"的GT mask
            filtered_anns = [
                ann for ann in ann_list 
                if 'ring' not in ann.get('object', 'object').lower() 
                and 'pin' not in ann.get('object', 'object').lower()
            ]
            
            if not filtered_anns:
                # 如果这张图片的所有GT mask都包含"ring"或"pin"，跳过这张图片
                skipped_images += 1
                ring_filtered_count += len(ann_list)
                continue
            
            # 从过滤后的GT mask中选择第一个（也可以随机选择）
            selected_ann = filtered_anns[0]
            object_name = selected_ann.get('object', 'object')
            gt_path = selected_ann.get('gt_path', '')
            
            image_path = os.path.join(raw_pic_base_dir, view_name, img_name)
            gt_mask_filename = os.path.basename(gt_path.replace('\\', '/'))
            gt_mask_path = os.path.join(gt_mask_base_dir, view_name, "masks", gt_mask_filename)
            question = f"segment the {object_name}"

            all_samples.append({
                'image_path': image_path,
                'gt_mask_path': gt_mask_path,
                'question': question,
                'object': object_name,
                'view_name': view_name,
                'img_name': img_name,
            })
            
            # 统计被过滤掉的ring和pin mask数量
            ring_filtered_count += len(ann_list) - len(filtered_anns)
    
    # 使用 RobotArmDataset（传入所有samples作为测试集）
    test_dataset = RobotArmDataset(
        json_paths=json_paths,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        precision=args.precision,
        image_size=args.image_size,
        raw_pic_base_dir=raw_pic_base_dir,
        gt_mask_base_dir=gt_mask_base_dir,
        sam_candidate_base_dir=sam_candidate_base_dir,
        sam_mask_helpers=sam_mask_helpers,
        max_samples_per_view=100,  # 每个视角100张
        is_train=False,
        samples=all_samples,  # 使用所有样本
        debug_meta=False,
    )
    
    expected_images = 3 * 100  # 三个视角，每个视角100张
    total_processed_images = sum(processed_images_per_view.values())
    
    print(f"测试集加载完成: {len(test_dataset)} 个样本")
    print(f"  - 期望图片数: {expected_images} 张（3个视角 × 100张/视角）")
    print(f"  - 处理的图片数: {total_processed_images} 张")
    for view_name, count in processed_images_per_view.items():
        print(f"    {view_name}: {count} 张")
    print(f"  - 实际测试图片数: {len(test_dataset)} 张（每张原图选择1个GT mask）")
    print(f"  - 已过滤掉 {ring_filtered_count} 个包含'ring'或'pin'的GT mask")
    print(f"  - 跳过了 {skipped_images} 张所有GT mask都包含'ring'或'pin'的图片")
    if len(test_dataset) < expected_images:
        print(f"  ⚠️  警告：实际测试图片数少于期望值，可能因为部分图片的所有GT mask都包含'ring'")
    return test_dataset


def init_deepspeed_config(args):
    """初始化DeepSpeed配置（优化内存使用）"""
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "weight_decay": 0.0,
            },
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "ignore_unused_parameters": True,
            # 启用CPU offload以支持大模型加载（30GB checkpoint无法完全放入23GB GPU）
            # 参数会被offload到CPU，推理时按需加载到GPU
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            # 推理时不需要optimizer，但保留配置以兼容checkpoint格式
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
        },
    }
    return ds_config


def test_model(model_engine, test_loader, args):
    """测试模型并计算IoU"""
    model_engine.eval()
    
    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16
    
    # 存储每张图片的IoU结果
    iou_results = []
    
    for idx, input_dict in enumerate(tqdm.tqdm(test_loader, desc="测试中")):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 获取设备（支持DeepSpeed引擎和普通模型）
        if hasattr(model_engine, 'module'):
            device = next(model_engine.module.parameters()).device
        else:
            device = next(model_engine.parameters()).device
        
        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)
        
        with torch.no_grad():
            input_dict["inference"] = True
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_iou"][0]
        
        # 选择相似度最高的单个mask（与训练脚本一致）
        max_idx = torch.argmax(pred_similarity[0]).item()
        
        sam_segs = input_dict["origin_segs_list"][0]  # (H, W, K)
        gt_mask = output_dict["gt_masks"][0]  # (1, H', W')
        
        # 使用相似度最高的单个mask（避免多个mask合并产生多个分离区域）
        pred_seg = sam_segs[:, :, max_idx]  # (H, W)
        pred_seg = pred_seg.astype(np.uint8)
        
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0)  # (1, H, W)
        # 获取设备（支持DeepSpeed引擎和普通模型）
        if hasattr(model_engine, 'module'):
            device = next(model_engine.module.parameters()).device
        else:
            device = next(model_engine.parameters()).device
        pred_seg = pred_seg.to(device=device)
        gt_mask = gt_mask.to(device=device)
        
        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)
        
        assert pred_seg.shape == gt_mask.shape
        
        # 计算IoU
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )
        
        acc_iou = intersection / (union + 1e-8)
        acc_iou[union == 0] += 1.0  # no-object target
        
        iou_value = acc_iou[0].cpu().item()  # 类别0是掩码区域
        
        # 获取图像信息（从input_dict中提取）
        image_paths = input_dict.get('image_paths', [])
        if image_paths and len(image_paths) > 0:
            image_path = image_paths[0]
            image_name = os.path.basename(image_path)
            # 从路径中提取view_name（例如：/path/to/robot_arm_01/image.png -> robot_arm_01）
            view_name = os.path.basename(os.path.dirname(image_path))
        else:
            image_name = f'image_{idx}'
            view_name = 'unknown'
        
        # 从问题中提取object_name
        questions_list = input_dict.get('questions_list', [])
        if questions_list and len(questions_list) > 0:
            questions = questions_list[0]
            if questions and len(questions) > 0:
                question = questions[0]
                # question格式: "segment the {object_name}"
                if 'segment the' in question:
                    object_name = question.replace('segment the', '').strip()
                else:
                    object_name = 'unknown'
            else:
                object_name = 'unknown'
        else:
            object_name = 'unknown'
        
        iou_results.append({
            'image_name': image_name,
            'view_name': view_name,
            'object_name': object_name,
            'iou': iou_value,
            'selected_mask_index': max_idx,  # 只选择单个mask
            'pred_similarity': pred_similarity[0][max_idx].item() if len(pred_similarity.shape) > 1 else pred_similarity[max_idx].item(),
        })
        
        # 可视化保存（仿照训练脚本中的方法）
        if args.visualize and args.local_rank == 0:
            try:
                # 读取原始图像
                if image_paths and len(image_paths) > 0 and os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    print(f"警告：无法读取图像 {image_path}，跳过可视化")
                    if (idx + 1) % 10 == 0:
                        print(f"已处理 {idx + 1}/{len(test_loader)} 张图片，当前平均IoU: {np.mean([r['iou'] for r in iou_results]):.4f}")
                    continue
                
                # 准备预测mask和GT mask（仿照训练脚本的处理方式）
                pred_mask_np = pred_seg.detach().cpu().numpy()[0]  # (H, W)
                # 处理mask：确保值在0-1范围内
                if pred_mask_np.max() > 1:
                    pred_mask_np = pred_mask_np.astype(np.float32) / 255.0
                else:
                    pred_mask_np = pred_mask_np.astype(np.float32)
                
                gt_mask_np = gt_mask.detach().cpu().numpy()[0]  # (H, W)
                gt_mask_np[gt_mask_np == 255] = 0  # ignored label
                # 处理mask：确保值在0-1范围内
                if gt_mask_np.max() > 1:
                    gt_mask_np = gt_mask_np.astype(np.float32) / 255.0
                else:
                    gt_mask_np = gt_mask_np.astype(np.float32)
                
                # 黑色部分（值为0）是掩码区域
                pred_mask_region = (pred_mask_np == 0)
                gt_mask_region = (gt_mask_np == 0)
                
                # 创建叠加图像：左侧GT（绿色），右侧预测（红色）
                gt_overlay = image.copy()
                gt_overlay[gt_mask_region] = (
                    image[gt_mask_region] * 0.5 + 
                    np.array([0, 255, 0]) * 0.5
                ).astype(np.uint8)
                
                pred_overlay = image.copy()
                pred_overlay[pred_mask_region] = (
                    image[pred_mask_region] * 0.5 + 
                    np.array([255, 0, 0]) * 0.5
                ).astype(np.uint8)
                
                # 拼接图像（左右并排）
                combined_image = np.hstack([gt_overlay, pred_overlay])
                
                # 获取图像宽度（用于文字定位）
                w = image.shape[1]
                
                # 获取mask路径（用于显示）
                gt_mask_path = ""
                segmentation_paths = input_dict.get('segmentation_paths', [])
                if segmentation_paths and len(segmentation_paths) > 0:
                    gt_mask_path = segmentation_paths[0]
                    if len(gt_mask_path) > 60:
                        gt_mask_path = "..." + gt_mask_path[-57:]
                
                pred_mask_path = ""
                candidate_mask_paths = input_dict.get('candidate_mask_paths_list', [])
                if candidate_mask_paths and len(candidate_mask_paths) > 0:
                    candidate_paths = candidate_mask_paths[0]
                    if len(candidate_paths) > max_idx:
                        pred_mask_path = candidate_paths[max_idx]
                        if len(pred_mask_path) > 60:
                            pred_mask_path = "..." + pred_mask_path[-57:]
                
                # 添加文字标题和路径信息（仿照训练脚本）
                title_text = f"Object: {object_name} | IoU: {iou_value:.3f} | Test Sample: {idx + 1}"
                left_text = f"GT (Green): {gt_mask_path}" if gt_mask_path else "GT (Green)"
                right_text = f"Pred (Red): {pred_mask_path}" if pred_mask_path else "Pred (Red)"
                
                # 在图像上添加文字
                text_height = 120 if (gt_mask_path or pred_mask_path) else 80
                combined_image_with_text = np.ones((combined_image.shape[0] + text_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
                combined_image_with_text[text_height:, :] = combined_image
                
                # 添加标题和标签
                cv2.putText(combined_image_with_text, title_text, 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(combined_image_with_text, left_text, 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(combined_image_with_text, right_text, 
                           (w + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # 如果路径太长，换行显示
                if gt_mask_path and len(gt_mask_path) > 60:
                    cv2.putText(combined_image_with_text, gt_mask_path[:60], 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                if pred_mask_path and len(pred_mask_path) > 60:
                    cv2.putText(combined_image_with_text, pred_mask_path[:60], 
                               (w + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # 确定保存目录（相对于checkpoint目录的父目录）
                checkpoint_dir = args.checkpoint
                checkpoint_parent_dir = os.path.dirname(checkpoint_dir)
                save_dir = os.path.join(checkpoint_parent_dir, args.test_vis_dir)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存组合图像（仿照训练脚本的文件名格式）
                image_name_base = os.path.splitext(image_name)[0]  # 移除扩展名
                save_path = os.path.join(save_dir, f"test_sample{idx + 1}_{image_name}")
                combined_image_bgr = cv2.cvtColor(combined_image_with_text, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, combined_image_bgr)
                
                if (idx + 1) % 50 == 0:
                    print(f"已保存 {idx + 1} 张可视化图片到: {save_dir}")
                    
            except Exception as e:
                print(f"保存可视化图片时出错（{image_name}）: {e}")
                import traceback
                traceback.print_exc()
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(test_loader)} 张图片，当前平均IoU: {np.mean([r['iou'] for r in iou_results]):.4f}")
    
    return iou_results


def save_results(iou_results, output_file):
    """保存结果到txt文件"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLMSeg 模型测试结果 - IoU评估\n")
        f.write("=" * 80 + "\n\n")
        
        # 统计信息
        all_ious = [r['iou'] for r in iou_results]
        non_zero_ious = [iou for iou in all_ious if iou > 0]
        zero_count = len(all_ious) - len(non_zero_ious)
        
        # 两种平均IoU
        avg_iou_all = np.mean(all_ious)  # 包含IoU=0的平均IoU
        avg_iou_non_zero = np.mean(non_zero_ious) if len(non_zero_ious) > 0 else 0.0  # 排除IoU=0的平均IoU
        
        std_iou_all = np.std(all_ious)
        std_iou_non_zero = np.std(non_zero_ious) if len(non_zero_ious) > 0 else 0.0
        
        min_iou_all = np.min(all_ious)
        min_iou_non_zero = np.min(non_zero_ious) if len(non_zero_ious) > 0 else 0.0
        max_iou = np.max(all_ious)
        
        f.write(f"总样本数: {len(iou_results)}\n")
        f.write(f"IoU为0的样本数: {zero_count} 张\n")
        f.write(f"有效样本数（IoU > 0）: {len(non_zero_ious)} 张\n")
        f.write("\n")
        f.write(f"【平均IoU（包含IoU=0）】: {avg_iou_all:.4f}\n")
        f.write(f"【平均IoU（排除IoU=0）】: {avg_iou_non_zero:.4f}\n")
        f.write("\n")
        f.write(f"IoU标准差（包含IoU=0）: {std_iou_all:.4f}\n")
        f.write(f"IoU标准差（排除IoU=0）: {std_iou_non_zero:.4f}\n")
        f.write(f"最小IoU（包含IoU=0）: {min_iou_all:.4f}\n")
        f.write(f"最小IoU（排除IoU=0）: {min_iou_non_zero:.4f}\n")
        f.write(f"最大IoU: {max_iou:.4f}\n")
        
        # IoU分布统计（包含所有样本）
        iou_ranges = [
            (0.0, 0.0, "0.0（完全失败）"),
            (0.0, 0.2, "0.0-0.2"),
            (0.2, 0.4, "0.2-0.4"),
            (0.4, 0.6, "0.4-0.6"),
            (0.6, 0.8, "0.6-0.8"),
            (0.8, 1.0, "0.8-1.0"),
        ]
        f.write("\nIoU分布统计（包含所有样本）:\n")
        f.write("-" * 80 + "\n")
        for low, high, label in iou_ranges:
            if low == 0.0 and high == 0.0:
                # 特殊处理：只统计IoU严格等于0的
                count = sum(1 for iou in all_ious if iou == 0.0)
            else:
                count = sum(1 for iou in all_ious if low < iou <= high)
            percentage = count / len(all_ious) * 100 if len(all_ious) > 0 else 0
            f.write(f"{label}: {count} 张 ({percentage:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # IoU小于0.4的图片列表
        low_iou_results = [r for r in iou_results if r['iou'] < 0.4]
        if low_iou_results:
            f.write(f"IoU < 0.4 的图片列表（共 {len(low_iou_results)} 张）:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'序号':<6} {'视角':<15} {'图像名':<20} {'物体':<20} {'IoU':<10}\n")
            f.write("-" * 80 + "\n")
            # 按IoU升序排列（最差的在前）
            low_iou_sorted = sorted(low_iou_results, key=lambda x: x['iou'])
            for idx, result in enumerate(low_iou_sorted, 1):
                f.write(f"{idx:<6} {result['view_name']:<15} {result['image_name']:<20} "
                       f"{result['object_name']:<20} {result['iou']:<10.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        else:
            f.write("IoU < 0.4 的图片: 无\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        
        # 详细结果
        f.write("详细结果（按IoU降序排列）:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'序号':<6} {'视角':<15} {'图像名':<20} {'物体':<20} {'IoU':<10} {'选中mask索引':<12} {'相似度分数':<12}\n")
        f.write("-" * 80 + "\n")
        
        # 按IoU降序排列
        sorted_results = sorted(iou_results, key=lambda x: x['iou'], reverse=True)
        for idx, result in enumerate(sorted_results, 1):
            mask_idx = result.get('selected_mask_index', 'N/A')
            similarity = result.get('pred_similarity', 0.0)
            f.write(f"{idx:<6} {result['view_name']:<15} {result['image_name']:<20} "
                   f"{result['object_name']:<20} {result['iou']:<10.4f} {mask_idx:<12} {similarity:<12.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("按视角统计:\n")
        f.write("-" * 80 + "\n")
        
        # 按视角统计
        view_stats = {}
        for result in iou_results:
            view = result['view_name']
            if view not in view_stats:
                view_stats[view] = []
            view_stats[view].append(result['iou'])
        
        for view, ious in sorted(view_stats.items()):
            avg_view_iou = np.mean(ious)
            f.write(f"{view}: 样本数={len(ious)}, 平均IoU={avg_view_iou:.4f}\n")
    
    print(f"\n结果已保存到: {output_file}")
    print(f"总样本数: {len(iou_results)}")
    print(f"IoU为0的样本数: {zero_count} 张")
    print(f"有效样本数（IoU > 0）: {len(non_zero_ious)} 张")
    print(f"\n【平均IoU（包含IoU=0）】: {avg_iou_all:.4f}")
    print(f"【平均IoU（排除IoU=0）】: {avg_iou_non_zero:.4f}")
    
    # 打印IoU < 0.4的图片统计
    low_iou_results = [r for r in iou_results if r['iou'] < 0.4]
    if low_iou_results:
        print(f"\nIoU < 0.4 的图片（共 {len(low_iou_results)} 张）:")
        print("-" * 80)
        low_iou_sorted = sorted(low_iou_results, key=lambda x: x['iou'])
        for idx, result in enumerate(low_iou_sorted, 1):
            print(f"{idx:3d}. {result['view_name']:<15} {result['image_name']:<20} "
                  f"{result['object_name']:<20} IoU: {result['iou']:.4f}")
    else:
        print("\nIoU < 0.4 的图片: 无")
    


def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 初始化tokenizer
    print("初始化tokenizer...")
    tokenizer = init_tokenizer(args)
    
    # 初始化模型
    print("初始化LISA模型...")
    model = init_LISA_model(args, tokenizer)
    
    # 如果使用GPU加载模式，先将模型移到GPU
    if not args.use_cpu_load and not args.use_deepspeed_load:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.local_rank}")
            print(f"将模型移到GPU: {device}...")
            model = model.to(device)
            print(f"✓ 模型已在GPU上")
            torch.cuda.empty_cache()
    
    # 构建测试数据集
    print("构建测试数据集...")
    test_dataset = build_test_dataset(args, tokenizer)
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn_new,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )
    
    # 加载checkpoint（支持多种模式）
    # 优先级：use_cpu_load > use_deepspeed_load > GPU直接加载 (默认)
    # 方案1：直接加载模型参数到GPU（最节省CPU内存，避免进程内存限制）
    if not args.use_cpu_load and not args.use_deepspeed_load:
        print(f"使用GPU直接加载模式（仅加载模型参数，直接加载到GPU）: {args.checkpoint}")
        print("这将避免CPU内存峰值问题，因为直接加载到GPU显存")
        print("注意：即使CPU内存充足，torch.load加载大文件时也可能触发进程内存限制")
        
        # 读取latest文件获取checkpoint tag
        latest_path = os.path.join(args.checkpoint, 'latest')
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                tag = f.read().strip()
        else:
            raise ValueError(f"找不到latest文件: {latest_path}")
        
        checkpoint_path = os.path.join(args.checkpoint, tag, 'mp_rank_00_model_states.pt')
        print(f"加载模型参数文件: {checkpoint_path}")
        
        # 检查GPU内存
        if not torch.cuda.is_available():
            raise RuntimeError("GPU不可用，无法使用GPU加载模式")
        
        device = torch.device(f"cuda:{args.local_rank}")
        file_size_gb = os.path.getsize(checkpoint_path) / (1024**3)
        
        gpu_mem = torch.cuda.get_device_properties(args.local_rank).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated(args.local_rank) / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved(args.local_rank) / (1024**3)
        gpu_free = gpu_mem - gpu_reserved
        
        print(f"\n{'='*60}")
        print(f"内存诊断信息")
        print(f"{'='*60}")
        print(f"Checkpoint文件大小: {file_size_gb:.2f} GB")
        
        # 检查进程内存限制（ulimit）
        try:
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)  # 虚拟内存限制
            soft_limit_gb = soft_limit / (1024**3) if soft_limit != resource.RLIM_INFINITY else float('inf')
            hard_limit_gb = hard_limit / (1024**3) if hard_limit != resource.RLIM_INFINITY else float('inf')
            
            print(f"\n【进程内存限制（ulimit）】:")
            if soft_limit == resource.RLIM_INFINITY:
                print(f"  虚拟内存软限制: 无限制")
            else:
                print(f"  虚拟内存软限制: {soft_limit_gb:.2f} GB")
            if hard_limit == resource.RLIM_INFINITY:
                print(f"  虚拟内存硬限制: 无限制")
            else:
                print(f"  虚拟内存硬限制: {hard_limit_gb:.2f} GB")
            
            # 检查是否可能受限
            if soft_limit != resource.RLIM_INFINITY:
                estimated_cpu_peak = file_size_gb * 2.5  # torch.load峰值内存估算
                if soft_limit_gb < estimated_cpu_peak:
                    print(f"  ⚠️  警告：进程内存限制可能不足！")
                    print(f"     估算峰值需求: ~{estimated_cpu_peak:.2f} GB")
                    print(f"     进程限制: {soft_limit_gb:.2f} GB")
                    print(f"     建议：增加ulimit -v 或使用GPU直接加载（当前模式）")
        except Exception as e:
            print(f"  无法检查进程内存限制: {e}")
        
        # 检查系统CPU内存
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"\n【系统CPU内存情况】:")
            print(f"  总内存: {mem.total / (1024**3):.2f} GB")
            print(f"  可用内存: {mem.available / (1024**3):.2f} GB")
            print(f"  已使用: {mem.used / (1024**3):.2f} GB")
            print(f"  缓存: {mem.cached / (1024**3):.2f} GB")
        except Exception as e:
            print(f"  无法检查系统内存: {e}")
        
        print(f"\n【GPU内存情况】:")
        print(f"  GPU {args.local_rank}:")
        print(f"    总显存: {gpu_mem:.2f} GB")
        print(f"    已分配: {gpu_allocated:.2f} GB")
        print(f"    已保留: {gpu_reserved:.2f} GB")
        print(f"    可用: {gpu_free:.2f} GB")
        
        # 估算所需GPU显存（文件大小 + 临时空间）
        estimated_need = file_size_gb * 1.5  # GPU加载通常需要更少临时空间
        print(f"\n  估算所需显存: ~{estimated_need:.2f} GB")
        if gpu_free < estimated_need:
            print(f"  ⚠️  警告：可用GPU显存可能不足！")
            print(f"     需要: {estimated_need:.2f} GB, 可用: {gpu_free:.2f} GB")
        else:
            print(f"  ✓ GPU显存充足")
        print(f"{'='*60}\n")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 直接加载到GPU（节省CPU内存）
        print("\n正在加载checkpoint到GPU（这可能需要几分钟，请耐心等待）...")
        print(f"  步骤1/3: 读取checkpoint文件直接到GPU显存...")
        print(f"  注意：直接加载到GPU可以避免CPU内存峰值问题")
        
        try:
            # 直接加载到GPU，避免CPU内存峰值
            print(f"  开始加载到 {device}...")
            print(f"  注意：即使CPU内存充足，torch.load也可能因以下原因失败：")
            print(f"    1. 进程内存限制（ulimit -v）- 检查方法：ulimit -a")
            print(f"    2. 反序列化峰值内存（约为文件大小的2-3倍，30GB文件可能需要75-90GB峰值）")
            print(f"    3. 系统OOM killer（如果其他进程占用大量内存）")
            print(f"  使用GPU直接加载可以避免这些问题...")
            
            # 加载前记录内存状态
            try:
                import psutil
                process = psutil.Process()
                mem_before = process.memory_info()
                mem_before_rss_gb = mem_before.rss / (1024**3)
                mem_before_vms_gb = mem_before.vms / (1024**3)
                print(f"\n  【加载前进程内存】:")
                print(f"    物理内存(RSS): {mem_before_rss_gb:.2f} GB")
                print(f"    虚拟内存(VMS): {mem_before_vms_gb:.2f} GB")
            except Exception as e:
                print(f"  无法获取进程内存信息: {e}")
            
            # 尝试使用mmap模式（如果支持）以减少内存峰值
            print(f"\n  开始读取checkpoint文件（这可能需要几分钟）...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False, mmap=True)
                print(f"  ✓ 使用内存映射模式加载")
            except (TypeError, ValueError) as e:
                # mmap可能不支持某些checkpoint格式，回退到普通加载
                print(f"  内存映射不可用，使用普通加载模式: {e}")
                print(f"  警告：普通加载模式在反序列化时可能需要大量临时内存")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 加载后记录内存状态
            try:
                mem_after = process.memory_info()
                mem_after_rss_gb = mem_after.rss / (1024**3)
                mem_after_vms_gb = mem_after.vms / (1024**3)
                print(f"\n  【加载后进程内存】:")
                print(f"    物理内存(RSS): {mem_after_rss_gb:.2f} GB (增加: {mem_after_rss_gb - mem_before_rss_gb:.2f} GB)")
                print(f"    虚拟内存(VMS): {mem_after_vms_gb:.2f} GB (增加: {mem_after_vms_gb - mem_before_vms_gb:.2f} GB)")
            except Exception:
                pass
            
            print(f"  ✓ Checkpoint文件读取完成（已加载到GPU）")
            
            # 检查GPU内存使用
            gpu_allocated_after = torch.cuda.memory_allocated(args.local_rank) / (1024**3)
            print(f"  加载后GPU显存使用: {gpu_allocated_after:.2f} GB")
            print(f"  显存使用增加: {gpu_allocated_after - gpu_allocated:.2f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                print(f"  ✗ GPU显存不足错误: {e}")
                print(f"  建议：使用DeepSpeed引擎加载（--use_deepspeed_load）")
                raise RuntimeError("GPU显存不足，建议使用DeepSpeed引擎加载")
            else:
                raise
        except MemoryError as e:
            print(f"\n  ✗ CPU内存不足错误: {e}")
            print(f"  可能原因：")
            print(f"    1. 进程内存限制（ulimit -v）不足 - 检查：ulimit -a")
            print(f"    2. 反序列化峰值内存超过限制（30GB文件可能需要75-90GB峰值）")
            print(f"    3. 系统OOM killer杀死了进程（返回码-9）")
            print(f"\n  解决方案：")
            print(f"    1. 在shell脚本中设置：ulimit -v unlimited")
            print(f"    2. 检查系统内存：free -h")
            print(f"    3. 如果GPU显存足够，使用GPU直接加载（当前模式）")
            raise RuntimeError("CPU内存不足，请检查ulimit设置或使用GPU直接加载")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            print(f"  错误类型: {type(e).__name__}")
            import traceback
            print(f"  错误详情:\n{traceback.format_exc()}")
            print(f"\n  如果进程被kill（返回码-9），可能是：")
            print(f"    1. 进程内存限制（ulimit -v）不足")
            print(f"    2. 系统OOM killer（检查：dmesg | tail -20）")
            print(f"    3. 反序列化峰值内存需求过大")
            raise
        
        print(f"  步骤2/3: 提取state_dict...")
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
            print(f"  ✓ 找到'module'键，包含 {len(state_dict)} 个参数")
        else:
            state_dict = checkpoint
            print(f"  ✓ 使用checkpoint本身作为state_dict，包含 {len(state_dict)} 个参数")
        
        # 删除checkpoint引用以释放内存
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  ✓ 已清理checkpoint对象，释放内存")
        
        # 加载权重到模型（模型已经在GPU上）
        print(f"  步骤3/3: 将权重加载到模型...")
        print(f"  模型参数数量: {len(list(model.state_dict().keys()))}")
        print(f"  State dict参数数量: {len(state_dict.keys())}")
        
        # 加载权重到模型（分批加载以节省内存）
        print(f"  正在加载权重（分批处理以节省内存）...")
        
        # 分批加载参数
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        matching_keys = model_keys & state_dict_keys
        
        print(f"  匹配的参数数量: {len(matching_keys)}/{len(model_keys)}")
        
        # 分批加载
        batch_size = 100  # 每批加载100个参数
        matching_keys_list = list(matching_keys)
        
        loaded_count = 0
        for i in range(0, len(matching_keys_list), batch_size):
            batch_keys = matching_keys_list[i:i+batch_size]
            batch_dict = {k: state_dict[k] for k in batch_keys}
            model.load_state_dict(batch_dict, strict=False)
            loaded_count += len(batch_keys)
            if (i // batch_size + 1) % 10 == 0:
                print(f"    已加载 {loaded_count}/{len(matching_keys_list)} 个参数...")
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"  ✓ 权重加载完成 ({loaded_count} 个参数)")
        
        # 检查缺失和意外的参数
        model_state = model.state_dict()
        missing_keys = [k for k in model_keys if k not in state_dict_keys]
        unexpected_keys = [k for k in state_dict_keys if k not in model_keys]
        
        if missing_keys:
            print(f"  警告：缺少以下参数 ({len(missing_keys)} 个): {missing_keys[:10]}...")
        if unexpected_keys:
            print(f"  警告：意外的参数 ({len(unexpected_keys)} 个): {unexpected_keys[:10]}...")
        
        # 清理state_dict以释放内存
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  ✓ 已清理state_dict，释放内存")
        
        # 模型已经在GPU上，不需要移动
        model_engine = model
        model_engine.eval()
        print("模型加载完成！")
    
    elif args.use_cpu_load and not args.use_deepspeed_load:
        print(f"使用CPU模式加载checkpoint: {args.checkpoint}")
        print("⚠️  警告：CPU加载模式需要大量CPU内存（约为checkpoint大小的2-3倍）")
        print("即使系统内存充足，也可能因以下原因失败：")
        print("  1. 进程内存限制（ulimit -v）- 检查方法：ulimit -a")
        print("  2. 反序列化峰值内存需求（60-90GB for 30GB checkpoint）")
        print("  3. 系统OOM killer（如果其他进程占用大量内存）")
        print("建议：优先使用GPU直接加载模式（默认模式）")
        
        # 检查进程内存限制
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if soft_limit != resource.RLIM_INFINITY:
                soft_limit_gb = soft_limit / (1024**3)
                print(f"\n当前进程内存限制: {soft_limit_gb:.2f} GB")
                estimated_need = 90  # 估算峰值需求
                if soft_limit_gb < estimated_need:
                    print(f"⚠️  进程内存限制可能不足（需要~{estimated_need}GB）")
                    print(f"   建议：ulimit -v unlimited 或使用GPU直接加载模式")
        except Exception as e:
            print(f"无法检查进程内存限制: {e}")
        
        # 将模型移到CPU
        model = model.cpu()
        
        # 使用DeepSpeed的工具函数将ZeRO checkpoint转换为fp32 state_dict
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            print("\n正在转换ZeRO checkpoint为fp32 state_dict...")
            print("这可能需要几分钟，并且会消耗大量CPU内存...")
            state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint)
            print("转换完成，正在加载权重到模型...")
            model.load_state_dict(state_dict, strict=False)
            print("权重加载完成")
        except MemoryError as e:
            print(f"\n✗ CPU内存不足错误: {e}")
            print("可能原因：")
            print("  1. 进程内存限制（ulimit -v）不足")
            print("  2. 系统内存被其他进程占用")
            print("  3. 反序列化峰值内存需求超过可用内存")
            print("\n建议解决方案：")
            print("  1. 使用GPU直接加载模式（默认，不指定--use_cpu_load）")
            print("  2. 增加进程内存限制：ulimit -v unlimited")
            print("  3. 检查系统内存使用：free -h")
            raise RuntimeError("CPU内存不足，建议使用GPU直接加载模式")
        except Exception as e:
            print(f"\n✗ CPU加载失败: {e}")
            print("错误类型:", type(e).__name__)
            import traceback
            print("错误详情:\n", traceback.format_exc())
            print("\n建议：使用GPU直接加载模式（默认，不指定--use_cpu_load）")
            raise
        
        # 如果CPU加载成功，将模型移回GPU（如果可用）
        if args.use_cpu_load and torch.cuda.is_available():
            print("将模型移到GPU...")
            device = torch.device(f"cuda:{args.local_rank}")
            model = model.to(device)
        
        # 创建模型引擎（不使用DeepSpeed，直接使用模型）
        model_engine = model
        model_engine.eval()
    else:
        # 使用DeepSpeed引擎模式（启用CPU offload）
        print("="*60)
        print("使用DeepSpeed引擎加载模式（启用CPU offload）")
        print("="*60)
        print("原因：30GB checkpoint无法完全放入23GB GPU显存")
        print("解决方案：启用CPU offload，参数存储在CPU，推理时按需加载到GPU")
        print("")
        
        # 显示内存情况
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(args.local_rank).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(args.local_rank) / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved(args.local_rank) / (1024**3)
            print(f"【当前GPU内存】:")
            print(f"  总显存: {gpu_mem:.2f} GB")
            print(f"  已分配: {gpu_allocated:.2f} GB")
            print(f"  已保留: {gpu_reserved:.2f} GB")
            print(f"  可用: {gpu_mem - gpu_reserved:.2f} GB")
            print("")
        
        print("初始化DeepSpeed引擎（启用CPU offload）...")
        ds_config = init_deepspeed_config(args)
        print("DeepSpeed配置:")
        print(f"  ZeRO Stage: {ds_config['zero_optimization']['stage']}")
        print(f"  CPU Offload: 已启用")
        print(f"  Precision: {args.precision}")
        print("")
        
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        print("✓ DeepSpeed引擎初始化完成")
        
        # 清理内存（在加载checkpoint之前）
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✓ 已清理内存缓存")
        print("")
        
        # 加载checkpoint（优化内存使用）
        print(f"加载checkpoint: {args.checkpoint}")
        print("注意：加载30GB checkpoint可能需要几分钟，请耐心等待...")
        print("DeepSpeed会自动将参数offload到CPU，避免GPU显存不足")
        print("")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 使用分块加载策略：先加载checkpoint到CPU，然后手动设置到DeepSpeed引擎
        print("使用分块加载策略以避免OOM...")
        print("步骤1: 读取checkpoint文件路径...")
        
        # 读取latest文件获取checkpoint tag
        latest_path = os.path.join(args.checkpoint, "latest")
        if os.path.exists(latest_path):
            with open(latest_path, "r") as f:
                checkpoint_tag = f.read().strip()
        else:
            # 查找最新的checkpoint目录
            checkpoint_dirs = [d for d in os.listdir(args.checkpoint) if d.startswith("global_step")]
            if checkpoint_dirs:
                checkpoint_tag = sorted(checkpoint_dirs)[-1]
            else:
                raise FileNotFoundError(f"未找到checkpoint目录: {args.checkpoint}")
        
        checkpoint_dir = os.path.join(args.checkpoint, checkpoint_tag)
        checkpoint_path = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        file_size_gb = os.path.getsize(checkpoint_path) / (1024**3)
        print(f"  Checkpoint文件: {checkpoint_path}")
        print(f"  文件大小: {file_size_gb:.2f} GB")
        print("")
        
        print("步骤2: 分块加载checkpoint到CPU（避免一次性加载30GB）...")
        print("  注意：使用CPU加载，然后逐步设置到DeepSpeed引擎")
        print("")
        
        # 显示系统内存情况
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info()
            mem_before_rss_gb = mem_before.rss / (1024**3)
            mem_before_vms_gb = mem_before.vms / (1024**3)
            
            sys_mem = psutil.virtual_memory()
            sys_total_gb = sys_mem.total / (1024**3)
            sys_available_gb = sys_mem.available / (1024**3)
            
            print(f"  【加载前内存状态】:")
            print(f"    进程物理内存(RSS): {mem_before_rss_gb:.2f} GB")
            print(f"    进程虚拟内存(VMS): {mem_before_vms_gb:.2f} GB")
            print(f"    系统总内存: {sys_total_gb:.2f} GB")
            print(f"    系统可用内存: {sys_available_gb:.2f} GB")
            
            # 估算峰值内存需求
            estimated_peak = file_size_gb * 2.5  # torch.load峰值估算
            print(f"\n  【内存需求估算】:")
            print(f"    Checkpoint文件大小: {file_size_gb:.2f} GB")
            print(f"    估算峰值内存需求: ~{estimated_peak:.2f} GB (文件大小 * 2.5)")
            print(f"    系统可用内存: {sys_available_gb:.2f} GB")
            
            if sys_available_gb < estimated_peak:
                print(f"    ⚠️  警告：可用内存可能不足！")
                print(f"       需要: ~{estimated_peak:.2f} GB, 可用: {sys_available_gb:.2f} GB")
            else:
                print(f"    ✓ 可用内存充足")
            print("")
        except Exception as e:
            print(f"  无法获取内存信息: {e}")
            print("")
        
        # 加载checkpoint到CPU
        print("步骤3: 加载checkpoint到CPU...")
        print("  警告：加载30GB文件可能需要60-90GB峰值内存（反序列化过程）")
        print("  如果进程被kill（返回码-9），可能是以下原因：")
        print("    1. 进程内存限制（ulimit -v）不足")
        print("    2. 系统OOM killer配置过于激进")
        print("    3. 内存碎片化导致无法分配大块连续内存")
        print("")
        
        # 再次检查进程内存限制（在实际加载前）
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if soft_limit != resource.RLIM_INFINITY:
                soft_limit_gb = soft_limit / (1024**3)
                print(f"  【进程内存限制检查】:")
                print(f"    虚拟内存软限制: {soft_limit_gb:.2f} GB")
                estimated_peak = file_size_gb * 2.5
                if soft_limit_gb < estimated_peak:
                    print(f"    ⚠️  警告：进程内存限制可能不足！")
                    print(f"       需要: ~{estimated_peak:.2f} GB, 限制: {soft_limit_gb:.2f} GB")
                    print(f"       建议：在shell脚本中设置 ulimit -v unlimited")
                else:
                    print(f"    ✓ 进程内存限制充足")
                print("")
        except Exception as e:
            print(f"  无法检查进程内存限制: {e}")
            print("")
        
        try:
            # 尝试使用内存映射模式（如果支持）以减少内存峰值
            try:
                print("  尝试使用内存映射模式加载（减少内存峰值）...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
                print("  ✓ 使用内存映射模式加载成功")
            except (TypeError, ValueError, AttributeError) as mmap_error:
                # mmap可能不支持某些checkpoint格式，回退到普通加载
                print(f"  内存映射不可用: {mmap_error}")
                print("  使用普通加载模式（可能需要大量临时内存）...")
                print("  注意：如果进程被kill，请检查系统内存和ulimit设置")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print("  ✓ Checkpoint文件读取完成（已加载到CPU）")
            
            # 提取state_dict
            print("步骤4: 提取state_dict...")
            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint
            
            print(f"  找到 {len(state_dict)} 个参数")
            
            # 删除checkpoint引用以释放内存
            del checkpoint
            gc.collect()
            print("  ✓ 已清理checkpoint对象")
            
        except MemoryError as e:
            print(f"\n  ✗ CPU内存不足错误: {e}")
            print("  可能原因：")
            print("    1. 进程内存限制（ulimit -v）不足 - 检查：ulimit -a")
            print("    2. 反序列化峰值内存需求超过可用内存（60-90GB）")
            print("    3. 系统OOM killer杀死了进程 - 检查：dmesg | tail -20")
            print("    4. 内存碎片化导致无法分配大块连续内存")
            print("\n  建议解决方案：")
            print("    1. 在shell脚本中设置：ulimit -v unlimited")
            print("    2. 检查系统内存：free -h")
            print("    3. 检查是否有其他进程占用大量内存：top -o %MEM")
            print("    4. 检查系统OOM killer日志：dmesg | grep -i oom")
            print("    5. 如果内存充足但仍被kill，尝试重启系统以清理内存碎片")
            raise RuntimeError("CPU内存不足，无法加载checkpoint")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            print("错误详情:\n", traceback.format_exc())
            print("\n  如果进程被kill（返回码-9），请检查：")
            print("    1. 系统内存：free -h")
            print("    2. 进程内存限制：ulimit -a")
            print("    3. 系统OOM killer日志：dmesg | tail -20")
            raise RuntimeError(f"无法加载checkpoint: {e}")
        
        print("")
        
        print("步骤5: 将参数设置到DeepSpeed引擎（分批处理）...")
        # 获取DeepSpeed引擎的模型
        if hasattr(model_engine, 'module'):
            target_model = model_engine.module
        else:
            target_model = model_engine
        
        # 获取模型参数名
        model_keys = set(target_model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        matching_keys = model_keys & state_dict_keys
        
        print(f"  匹配的参数数量: {len(matching_keys)}/{len(model_keys)}")
        print("  分批加载参数到DeepSpeed引擎...")
        
        # 分批加载参数（每批50个，避免内存峰值）
        batch_size = 50
        matching_keys_list = list(matching_keys)
        loaded_count = 0
        
        for i in range(0, len(matching_keys_list), batch_size):
            batch_keys = matching_keys_list[i:i+batch_size]
            batch_dict = {k: state_dict[k] for k in batch_keys}
            
            # 使用load_state_dict加载到DeepSpeed引擎
            target_model.load_state_dict(batch_dict, strict=False)
            
            loaded_count += len(batch_keys)
            if (i // batch_size + 1) % 20 == 0:
                print(f"    已加载 {loaded_count}/{len(matching_keys_list)} 个参数...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"  ✓ 参数加载完成 ({loaded_count} 个参数)")
        
        # 清理state_dict
        del state_dict
        gc.collect()
        print("  ✓ 已清理state_dict")
        
        # 显示加载后的内存情况
        try:
            mem_after = process.memory_info()
            mem_after_rss_gb = mem_after.rss / (1024**3)
            print(f"\n【加载后进程内存】:")
            print(f"  物理内存(RSS): {mem_after_rss_gb:.2f} GB (增加: {mem_after_rss_gb - mem_before_rss_gb:.2f} GB)")
        except Exception:
            pass
        
        if torch.cuda.is_available():
            gpu_allocated_after = torch.cuda.memory_allocated(args.local_rank) / (1024**3)
            gpu_reserved_after = torch.cuda.memory_reserved(args.local_rank) / (1024**3)
            print(f"\n【加载后GPU内存】:")
            print(f"  已分配: {gpu_allocated_after:.2f} GB")
            print(f"  已保留: {gpu_reserved_after:.2f} GB")
            print(f"  说明：参数已通过DeepSpeed offload到CPU，GPU只保留当前需要的部分")
        
        print("\n✓ Checkpoint加载成功！")
    
    # 测试模型
    print("\n开始测试...")
    if args.visualize and args.local_rank == 0:
        checkpoint_dir = args.checkpoint
        checkpoint_parent_dir = os.path.dirname(checkpoint_dir)
        save_dir = os.path.join(checkpoint_parent_dir, args.test_vis_dir)
        print(f"可视化结果将保存到: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        print("")
    
    iou_results = test_model(model_engine, test_loader, args)
    
    # 保存结果
    # 如果启用可视化，将txt结果文件也保存到可视化目录
    if args.visualize and args.local_rank == 0:
        checkpoint_dir = args.checkpoint
        checkpoint_parent_dir = os.path.dirname(checkpoint_dir)
        save_dir = os.path.join(checkpoint_parent_dir, args.test_vis_dir)
        # 将txt结果文件保存到可视化目录
        output_file_name = os.path.basename(args.output_file) if os.path.dirname(args.output_file) else args.output_file
        output_file_path = os.path.join(save_dir, output_file_name)
    else:
        output_file_path = args.output_file
    
    save_results(iou_results, output_file_path)
    
    print("\n测试完成！")
    if args.visualize and args.local_rank == 0:
        checkpoint_dir = args.checkpoint
        checkpoint_parent_dir = os.path.dirname(checkpoint_dir)
        save_dir = os.path.join(checkpoint_parent_dir, args.test_vis_dir)
        print(f"可视化结果已保存到: {save_dir}")
        print(f"共保存 {len(iou_results)} 张图片的可视化结果")
        print(f"测试结果txt文件已保存到: {output_file_path}")


if __name__ == "__main__":
    main()
