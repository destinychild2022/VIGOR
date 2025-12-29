#!/usr/bin/env python3
"""
测试训练好的SAM模型的分割成功率
对每个GT mask，使用SAM生成候选masks，计算IoU，IoU>=0.4算成功
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import colorsys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.segment_anything import sam_model_registry
from model.segment_anything.predictor import SamPredictor
from model.segment_anything.utils.amg import (
    build_all_layer_point_grids,
    calculate_stability_score,
    batched_mask_to_box,
    batch_iterator,
)

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available")


class MaskDecoderWrapper(nn.Module):
    """包装器类，用于正确处理PEFT包装后的mask_decoder"""
    def __init__(self, mask_decoder):
        super().__init__()
        if hasattr(mask_decoder, 'base_model'):
            self.add_module('mask_decoder', mask_decoder.base_model)
            self._peft_model = mask_decoder
        else:
            self.add_module('mask_decoder', mask_decoder)
            self._peft_model = None
    
    def forward(self, 
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool):
        return self._modules['mask_decoder'](
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
    
    def __getattr__(self, name):
        if name.startswith('_') or name == 'mask_decoder':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if 'mask_decoder' in self._modules:
            try:
                return getattr(self._modules['mask_decoder'], name)
            except AttributeError:
                pass
        
        if self._peft_model is not None:
            try:
                return getattr(self._peft_model, name)
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def apply_lora_to_sam(model, lora_config: Dict):
    """对SAM模型应用LoRA"""
    if not PEFT_AVAILABLE:
        print("PEFT不可用，使用全量微调")
        return model
    
    target_modules = lora_config.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'out_proj'])
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=target_modules,
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        bias="none",
    )
    
    if lora_config.get('apply_to_mask_decoder', True):
        print("对mask_decoder应用LoRA...")
        peft_mask_decoder = get_peft_model(model.mask_decoder, peft_config)
        model.mask_decoder = MaskDecoderWrapper(peft_mask_decoder)
        print("LoRA应用成功")
    
    return model


def load_finetuned_sam(checkpoint_path: str, sam_checkpoint: str, use_lora: bool = True):
    """加载训练好的SAM模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载基础SAM模型
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    
    # 加载训练权重
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"\n加载训练权重checkpoint: {checkpoint_path}")
    print(f"  keys: {list(ckpt.keys())}")
    
    # 应用LoRA（如果使用）
    if use_lora and PEFT_AVAILABLE:
        lora_config = {
            'r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
            'apply_to_mask_decoder': True,
            'apply_to_image_encoder': False
        }
        sam = apply_lora_to_sam(sam, lora_config)
    
    # 加载模型权重
    if "model_state_dict" in ckpt:
        missing, unexpected = sam.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        raise ValueError("checkpoint中没有model_state_dict")
    
    # 加载LoRA权重（如果存在）
    if use_lora and PEFT_AVAILABLE and "lora_state_dict" in ckpt:
        print("  加载lora_state_dict...")
        lora_state = ckpt["lora_state_dict"]
        peft_model = getattr(sam.mask_decoder, "_peft_model", None)
        if peft_model is not None:
            missing_l, unexpected_l = peft_model.load_state_dict(lora_state, strict=False)
            print(f"  lora_state_dict: missing={len(missing_l)}, unexpected={len(unexpected_l)}")
    
    # 设置为评估模式
    for p in sam.parameters():
        p.requires_grad = False
    sam.eval()
    
    print("✅ 训练权重加载完成")
    return sam, device


def load_original_sam(sam_checkpoint: str):
    """加载原始SAM模型（未训练的）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载基础SAM模型
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    
    # 设置为评估模式
    for p in sam.parameters():
        p.requires_grad = False
    sam.eval()
    
    print("✅ 原始SAM模型加载完成（未加载训练权重）")
    return sam, device


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask的IoU（优化版本）
    mask格式：0=掩码区域（前景），1=背景
    """
    assert mask1.shape == mask2.shape
    
    # 确保格式统一：0=掩码区域，1=背景
    # 如果mask值大于1，说明是原始格式（0=掩码，255=背景），需要转换
    if mask1.max() > 1:
        mask1_binary = (mask1 == 0).astype(np.uint8)  # 0=掩码区域
    else:
        mask1_binary = mask1.astype(np.uint8)  # 已经是正确格式
    
    if mask2.max() > 1:
        mask2_binary = (mask2 == 0).astype(np.uint8)  # 0=掩码区域
    else:
        mask2_binary = mask2.astype(np.uint8)  # 已经是正确格式
    
    # 计算交集和并集（掩码区域，即值为0的部分）
    # 掩码区域：值为0
    mask1_foreground = (mask1_binary == 0)
    mask2_foreground = (mask2_binary == 0)
    
    intersection = np.logical_and(mask1_foreground, mask2_foreground).sum()
    union = np.logical_or(mask1_foreground, mask2_foreground).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_iou_batch(gt_mask: np.ndarray, candidate_masks: List[np.ndarray]) -> Tuple[float, int]:
    """批量计算GT mask与所有候选masks的IoU，返回最大IoU和最佳mask索引（优化版本）
    
    Args:
        gt_mask: GT mask，格式：0=掩码区域，1=背景，尺寸已匹配
        candidate_masks: 候选masks列表，格式：0=掩码区域，1=背景，尺寸已匹配
    
    Returns:
        (max_iou, best_mask_idx)
    """
    if len(candidate_masks) == 0:
        return 0.0, -1
    
    # 确保GT mask格式正确（已经是正确格式，直接使用）
    gt_foreground = (gt_mask == 0)  # 掩码区域（值为0的部分）
    
    max_iou = 0.0
    best_mask_idx = -1
    
    # 批量计算IoU（向量化操作，假设所有masks尺寸已匹配）
    for idx, candidate_mask in enumerate(candidate_masks):
        # 候选mask已经是正确格式（0=掩码区域，1=背景），且尺寸已匹配
        candidate_foreground = (candidate_mask == 0)  # 掩码区域（值为0的部分）
        
        # 计算IoU（向量化操作）
        intersection = np.logical_and(gt_foreground, candidate_foreground).sum()
        union = np.logical_or(gt_foreground, candidate_foreground).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        if iou > max_iou:
            max_iou = iou
            best_mask_idx = idx
    
    return max_iou, best_mask_idx


def generate_all_candidate_masks(
    predictor: SamPredictor,
    image: np.ndarray,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    max_area_ratio: float = 0.25,
) -> List[np.ndarray]:
    """使用均匀网格点生成所有候选masks（参考test_automatic_mask_generator.py）"""
    orig_size = image.shape[:2]  # (H, W)
    
    # 设置图像
    predictor.set_image(image)
    
    # 生成点网格（归一化坐标 [0, 1]）
    point_grids = build_all_layer_point_grids(points_per_side, 0, 1)
    points_for_image = point_grids[0]  # [N, 2] 归一化坐标
    
    # 转换为像素坐标
    points_scale = np.array(orig_size)[None, ::-1]  # [1, 2] (W, H)
    points_for_image = points_for_image * points_scale  # [N, 2] 像素坐标
    
    # 生成masks
    all_masks = []
    all_scores = []
    all_boxes = []
    
    # 批量处理点
    for (batch_points,) in batch_iterator(points_per_batch, points_for_image):
        # 转换点坐标到模型输入坐标系
        transformed_points = predictor.transform.apply_coords(batch_points, orig_size)
        in_points = torch.as_tensor(transformed_points, device=predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=predictor.device)
        
        # 预测（multimask_output=True会返回3个mask）
        masks, iou_preds, _ = predictor.predict_torch(
            in_points[:, None, :],  # [N, 1, 2]
            in_labels[:, None],      # [N, 1]
            multimask_output=True,
            return_logits=True,
        )
        
        # 展平：从 [N, 3, H, W] 到 [N*3, H, W]
        masks = masks.flatten(0, 1)  # [N*3, H, W]
        iou_preds = iou_preds.flatten(0, 1)  # [N*3]
        
        # 计算稳定性分数
        stability_scores = calculate_stability_score(
            masks,
            predictor.model.mask_threshold,
            1.0,
        )
        
        # 处理NaN值
        stability_scores = torch.where(
            torch.isnan(stability_scores),
            torch.zeros_like(stability_scores),
            stability_scores
        )
        
        # 过滤低质量mask
        # 第一步：过滤IoU
        if pred_iou_thresh > 0.0:
            keep_mask_iou = iou_preds > pred_iou_thresh
            masks = masks[keep_mask_iou]
            iou_preds = iou_preds[keep_mask_iou]
            stability_scores = stability_scores[keep_mask_iou]
        
        # 第二步：过滤稳定性分数
        if stability_score_thresh > 0.0 and len(masks) > 0:
            keep_mask_stability = stability_scores >= stability_score_thresh
            masks = masks[keep_mask_stability]
            iou_preds = iou_preds[keep_mask_stability]
            stability_scores = stability_scores[keep_mask_stability]
        
        # 如果没有mask通过过滤，跳过这个batch
        if len(masks) == 0:
            continue
            
        # 二值化mask
        masks_binary = masks > predictor.model.mask_threshold
        
        # 计算boxes
        boxes = batched_mask_to_box(masks_binary)
        
        # 第三步：过滤面积过大的mask
        if max_area_ratio > 0.0 and max_area_ratio < 1.0:
            image_area = orig_size[0] * orig_size[1]  # H * W
            max_area = image_area * max_area_ratio
            
            # 计算每个mask的面积
            mask_areas = masks_binary.sum(dim=(1, 2))  # [N] 每个mask的像素数
            
            # 过滤面积过大的mask
            keep_mask_area = mask_areas <= max_area
            masks_binary = masks_binary[keep_mask_area]
            boxes = boxes[keep_mask_area]
            iou_preds = iou_preds[keep_mask_area]
            stability_scores = stability_scores[keep_mask_area]
        
        # 存储
        for j in range(len(masks_binary)):
            mask_np = masks_binary[j].cpu().numpy()
            # SAM生成的mask格式：True=掩码区域，False=背景
            # 转换为统一格式：0=掩码区域，1=背景（与GT mask格式一致）
            mask_formatted = (~mask_np).astype(np.uint8)  # True->0, False->1
            all_masks.append(mask_formatted)
            all_scores.append({
                'iou': iou_preds[j].item(),
                'stability': stability_scores[j].item(),
            })
            all_boxes.append(boxes[j].cpu().numpy())
    
    predictor.reset_image()
    
    # NMS去重
    if len(all_boxes) > 0:
        from torchvision.ops.boxes import batched_nms
        boxes_tensor = torch.stack([torch.from_numpy(box) for box in all_boxes])
        scores_tensor = torch.tensor([s['iou'] for s in all_scores])
        keep_indices = batched_nms(
            boxes_tensor.float(),
            scores_tensor,
            torch.zeros(len(boxes_tensor), dtype=torch.long),
            iou_threshold=box_nms_thresh,
        )
        keep_indices = keep_indices.cpu().numpy()
        all_masks = [all_masks[i] for i in keep_indices]
    
    return all_masks


def generate_colored_mask_overlay(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """生成带颜色的mask叠加图像（与test_automatic_mask_generator.py一致）
    
    Args:
        image: RGB图像 [H, W, 3]
        masks: mask列表，每个mask是bool数组（True=掩码区域）
        alpha: 透明度
    """
    overlay = image.copy()
    
    # 生成不同颜色
    num_masks = len(masks)
    colors = []
    for i in range(num_masks):
        hue = i / max(num_masks, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    # 叠加每个mask（与test_automatic_mask_generator.py一致：mask.astype(bool)）
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        # 统一转换为bool格式：True=掩码区域
        if mask.dtype == bool:
            mask_bool = mask
        else:
            # 如果是uint8，0=掩码区域 -> True=掩码区域
            mask_bool = (mask == 0).astype(bool)
        
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
    
    return overlay


def load_candidate_masks_from_dir(masks_dir: str) -> List[np.ndarray]:
    """从已保存的masks目录加载候选masks
    
    Args:
        masks_dir: masks目录路径（包含mask_0000.png, mask_0001.png等）
    
    Returns:
        候选masks列表，格式：0=掩码区域，1=背景
    
    注意：VIGOR数据集中保存的mask格式：
        - 白色（255）= 掩码区域
        - 黑色（0）= 背景
    这与GT mask格式相反（GT mask中黑色是掩码区域，白色是背景）
    """
    if not os.path.exists(masks_dir):
        return []
    
    # 获取所有mask文件
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.startswith('mask_') and f.endswith('.png')])
    
    candidate_masks = []
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # VIGOR数据集保存的mask格式：白色（255）= 掩码区域，黑色（0）= 背景
        # 转换为统一格式：0=掩码区域，1=背景
        # 所以：白色(255)->0(掩码)，黑色(0)->1(背景)
        mask_binary = (mask == 0).astype(np.uint8)  # 黑色(0)->1(背景)，白色(255)->0(掩码)
        candidate_masks.append(mask_binary)
    
    return candidate_masks


def test_vigor_success_rate(
    annotations_file: str,
    dataset_dir: str,
    sam_masks_dir: str,
    sam_masks2_dir: str,
    success_threshold: float = 0.4,
    output_file: str = None,
):
    """测试VIGOR-100K数据集中SAM和SAM2生成的候选掩码成功率
    
    Args:
        annotations_file: VIGOR标注文件路径（all_annotations.json）
        dataset_dir: VIGOR数据集目录（包含图像和masks子目录）
        sam_masks_dir: SAM生成的候选掩码目录（如 sam_masks/）
        sam_masks2_dir: SAM2生成的候选掩码目录（如 sam_masks2/）
        success_threshold: 成功阈值（IoU >= threshold算成功）
        output_file: 结果输出文件路径
    """
    print("=" * 80)
    print("VIGOR-100K Test Set Success Rate Evaluation")
    print("=" * 80)
    print(f"Annotations file: {annotations_file}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"SAM masks dir: {sam_masks_dir}")
    print(f"SAM masks2 dir: {sam_masks2_dir}")
    print(f"Success threshold: IoU >= {success_threshold}")
    print("=" * 80)
    
    # 加载标注文件
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取annotations
    if isinstance(data, dict) and 'annotations' in data:
        annotations = data['annotations']
    elif isinstance(data, list):
        annotations = data
    else:
        raise ValueError(f"Unsupported annotations format in {annotations_file}")
    
    print(f"\n加载了 {len(annotations)} 个标注")
    
    # GT masks目录
    gt_masks_dir = os.path.join(dataset_dir, 'masks')
    
    # 存储所有测试结果
    sam_results = []
    sam2_results = []
    sam_success_count = 0
    sam2_success_count = 0
    total_count = 0
    
    # 按图像分组
    img_to_annotations = {}
    for ann in annotations:
        img_name = ann.get('img_name', '')
        if not img_name:
            continue
        if img_name not in img_to_annotations:
            img_to_annotations[img_name] = []
        img_to_annotations[img_name].append(ann)
    
    print(f"\n共有 {len(img_to_annotations)} 张唯一图像")
    
    # 处理每张图像
    for img_name in tqdm(sorted(img_to_annotations.keys()), desc="Processing images"):
        img_stem = Path(img_name).stem  # 去掉扩展名，如 "10.png" -> "10"
        
        # 加载SAM候选masks
        sam_masks_path = os.path.join(sam_masks_dir, img_stem, 'masks')
        sam_candidate_masks = load_candidate_masks_from_dir(sam_masks_path)
        
        # 加载SAM2候选masks
        sam2_masks_path = os.path.join(sam_masks2_dir, img_stem, 'masks')
        sam2_candidate_masks = load_candidate_masks_from_dir(sam2_masks_path)
        
        # 处理该图像的每个GT mask
        for ann in img_to_annotations[img_name]:
            gt_path_rel = ann.get('gt_path', '')
            if not gt_path_rel:
                continue
            
            object_name = ann.get('object', 'unknown')
            
            # 构建GT mask完整路径
            gt_mask_path = os.path.join(dataset_dir, gt_path_rel.replace('\\', '/'))
            
            # 加载GT mask
            gt_mask = load_gt_mask(gt_mask_path)
            if gt_mask is None:
                continue
            
            total_count += 1
            
            # 测试SAM候选masks
            if len(sam_candidate_masks) > 0:
                # 调整GT mask大小以匹配候选mask
                h, w = sam_candidate_masks[0].shape
                if gt_mask.shape != (h, w):
                    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    gt_mask_resized = (gt_mask_resized > 0).astype(np.uint8)
                else:
                    gt_mask_resized = gt_mask
                
                max_iou, best_mask_idx = compute_iou_batch(gt_mask_resized, sam_candidate_masks)
                is_success = max_iou >= success_threshold
                if is_success:
                    sam_success_count += 1
                
                sam_results.append({
                    'img_name': img_name,
                    'object_name': object_name,
                    'gt_mask_path': gt_mask_path,
                    'max_iou': max_iou,
                    'best_mask_idx': best_mask_idx,
                    'is_success': is_success,
                    'num_candidates': len(sam_candidate_masks),
                })
            else:
                # 没有候选masks，算失败
                sam_results.append({
                    'img_name': img_name,
                    'object_name': object_name,
                    'gt_mask_path': gt_mask_path,
                    'max_iou': 0.0,
                    'best_mask_idx': -1,
                    'is_success': False,
                    'num_candidates': 0,
                })
            
            # 测试SAM2候选masks
            if len(sam2_candidate_masks) > 0:
                # 调整GT mask大小以匹配候选mask
                h, w = sam2_candidate_masks[0].shape
                if gt_mask.shape != (h, w):
                    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    gt_mask_resized = (gt_mask_resized > 0).astype(np.uint8)
                else:
                    gt_mask_resized = gt_mask
                
                max_iou, best_mask_idx = compute_iou_batch(gt_mask_resized, sam2_candidate_masks)
                is_success = max_iou >= success_threshold
                if is_success:
                    sam2_success_count += 1
                
                sam2_results.append({
                    'img_name': img_name,
                    'object_name': object_name,
                    'gt_mask_path': gt_mask_path,
                    'max_iou': max_iou,
                    'best_mask_idx': best_mask_idx,
                    'is_success': is_success,
                    'num_candidates': len(sam2_candidate_masks),
                })
            else:
                # 没有候选masks，算失败
                sam2_results.append({
                    'img_name': img_name,
                    'object_name': object_name,
                    'gt_mask_path': gt_mask_path,
                    'max_iou': 0.0,
                    'best_mask_idx': -1,
                    'is_success': False,
                    'num_candidates': 0,
                })
    
    # 计算成功率
    sam_success_rate = (sam_success_count / total_count * 100) if total_count > 0 else 0.0
    sam2_success_rate = (sam2_success_count / total_count * 100) if total_count > 0 else 0.0
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"测试结果统计")
    print(f"{'='*80}")
    print(f"总GT mask数: {total_count}")
    print(f"\nSAM结果:")
    print(f"  成功数（IoU >= {success_threshold}）: {sam_success_count}")
    print(f"  失败数: {total_count - sam_success_count}")
    print(f"  成功率: {sam_success_rate:.2f}%")
    print(f"  平均IoU: {np.mean([r['max_iou'] for r in sam_results]):.4f}")
    print(f"\nSAM2结果:")
    print(f"  成功数（IoU >= {success_threshold}）: {sam2_success_count}")
    print(f"  失败数: {total_count - sam2_success_count}")
    print(f"  成功率: {sam2_success_rate:.2f}%")
    print(f"  平均IoU: {np.mean([r['max_iou'] for r in sam2_results]):.4f}")
    
    # 打印多阈值成功率统计
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n{'='*90}")
    print("不同IoU阈值下的成功率统计")
    print(f"{'='*90}")
    print(f"{'阈值':<10} {'SAM成功率':<18} {'SAM成功数/总数':<20} {'SAM2成功率':<18} {'SAM2成功数/总数':<20}")
    print("-" * 90)
    
    for threshold in thresholds:
        sam_success_at_threshold = sum(1 for r in sam_results if r['max_iou'] >= threshold)
        sam2_success_at_threshold = sum(1 for r in sam2_results if r['max_iou'] >= threshold)
        sam_rate_at_threshold = (sam_success_at_threshold / total_count * 100) if total_count > 0 else 0.0
        sam2_rate_at_threshold = (sam2_success_at_threshold / total_count * 100) if total_count > 0 else 0.0
        
        print(f"{threshold:<10.1f} {sam_rate_at_threshold:>6.2f}%{'':<10} {sam_success_at_threshold}/{total_count:<15} "
              f"{sam2_rate_at_threshold:>6.2f}%{'':<10} {sam2_success_at_threshold}/{total_count:<15}")
    
    print("=" * 90)
    
    # 保存详细结果到文件
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VIGOR-100K Test Set Success Rate Evaluation\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Annotations file: {annotations_file}\n")
            f.write(f"Dataset dir: {dataset_dir}\n")
            f.write(f"SAM masks dir: {sam_masks_dir}\n")
            f.write(f"SAM masks2 dir: {sam_masks2_dir}\n")
            f.write(f"Success threshold: IoU >= {success_threshold}\n")
            f.write(f"\n总GT mask数: {total_count}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("SAM结果:\n")
            f.write("-" * 80 + "\n")
            f.write(f"成功数（IoU >= {success_threshold}）: {sam_success_count}\n")
            f.write(f"失败数: {total_count - sam_success_count}\n")
            f.write(f"成功率: {sam_success_rate:.2f}%\n")
            f.write(f"平均IoU: {np.mean([r['max_iou'] for r in sam_results]):.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("SAM2结果:\n")
            f.write("-" * 80 + "\n")
            f.write(f"成功数（IoU >= {success_threshold}）: {sam2_success_count}\n")
            f.write(f"失败数: {total_count - sam2_success_count}\n")
            f.write(f"成功率: {sam2_success_rate:.2f}%\n")
            f.write(f"平均IoU: {np.mean([r['max_iou'] for r in sam2_results]):.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 添加多阈值成功率统计
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            f.write("不同IoU阈值下的成功率统计:\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'阈值':<10} {'SAM成功率':<18} {'SAM成功数/总数':<20} {'SAM2成功率':<18} {'SAM2成功数/总数':<20}\n")
            f.write("-" * 90 + "\n")
            
            for threshold in thresholds:
                sam_success_at_threshold = sum(1 for r in sam_results if r['max_iou'] >= threshold)
                sam2_success_at_threshold = sum(1 for r in sam2_results if r['max_iou'] >= threshold)
                sam_rate_at_threshold = (sam_success_at_threshold / total_count * 100) if total_count > 0 else 0.0
                sam2_rate_at_threshold = (sam2_success_at_threshold / total_count * 100) if total_count > 0 else 0.0
                
                f.write(f"{threshold:<10.1f} {sam_rate_at_threshold:>6.2f}%{'':<10} {sam_success_at_threshold}/{total_count:<15} "
                       f"{sam2_rate_at_threshold:>6.2f}%{'':<10} {sam2_success_at_threshold}/{total_count:<15}\n")
            
            f.write("=" * 90 + "\n\n")
            
            # 合并SAM和SAM2结果到一个表格（按SAM IoU降序）
            f.write("SAM vs SAM2 对比结果（按SAM IoU降序）:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'序号':<6} {'图像名':<20} {'物体':<30} {'SAM IoU':<12} {'SAM状态':<12} "
                   f"{'SAM2 IoU':<12} {'SAM2状态':<12}\n")
            f.write("-" * 100 + "\n")
            
            # 合并结果：sam_results和sam2_results应该是对应的（按相同顺序添加）
            merged_results = []
            for i in range(len(sam_results)):
                sam_r = sam_results[i]
                sam2_r = sam2_results[i] if i < len(sam2_results) else {
                    'img_name': sam_r['img_name'],
                    'object_name': sam_r['object_name'],
                    'max_iou': 0.0,
                    'is_success': False,
                    'num_candidates': 0
                }
                merged_results.append({
                    'img_name': sam_r['img_name'],
                    'object_name': sam_r['object_name'],
                    'sam_iou': sam_r['max_iou'],
                    'sam_status': sam_r['is_success'],
                    'sam_candidates': sam_r['num_candidates'],
                    'sam2_iou': sam2_r['max_iou'],
                    'sam2_status': sam2_r['is_success'],
                    'sam2_candidates': sam2_r['num_candidates'],
                })
            
            # 按SAM IoU降序排序
            sorted_merged_results = sorted(merged_results, key=lambda x: x['sam_iou'], reverse=True)
            for idx, result in enumerate(sorted_merged_results, 1):
                sam_status = "成功" if result['sam_status'] else "失败"
                sam2_status = "成功" if result['sam2_status'] else "失败"
                f.write(f"{idx:<6} {result['img_name']:<20} {result['object_name']:<30} "
                       f"{result['sam_iou']:<12.4f} {sam_status:<12} "
                       f"{result['sam2_iou']:<12.4f} {sam2_status:<12}\n")
        
        print(f"\n详细结果已保存到: {output_file}")
    
    return sam_success_rate, sam2_success_rate, sam_results, sam2_results


def load_gt_mask(gt_mask_path: str) -> np.ndarray:
    """加载GT mask
    返回格式：0=掩码区域（前景），1=背景
    
    GT mask文件格式：黑色（值为0）是掩码区域，白色（值为255）是背景
    参考：utils/robot_arm_dataset.py 和 SAM_finetune/finetune_sam_lora_point.py
    """
    if not os.path.exists(gt_mask_path):
        return None
    
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return None
    
    # GT mask格式：黑色（0）是掩码区域，白色（255）是背景
    # 转换为统一格式：0=掩码区域，1=背景
    # 注意：gt_mask是0-255的灰度图，需要正确转换
    # 方法1（与robot_arm_dataset.py一致）：(gt_mask > 0) -> 黑色(0)->0(掩码)，白色(255)->1(背景)
    # 方法2（与finetune_sam_lora_point.py一致）：(gt_mask < 128) -> 黑色(0)->0(掩码)，白色(255)->1(背景)
    # 这里使用方法1，与robot_arm_dataset.py保持一致
    gt_mask_binary = (gt_mask > 0).astype(np.uint8)  # 黑色(0)->0(掩码)，白色(255)->1(背景)
    
    return gt_mask_binary


def test_sam_success_rate(
    sam_model_path: str,
    sam_checkpoint: str,
    annotations_dir: str,
    images_dir: str,
    gt_masks_dir: str,
    success_threshold: float = 0.4,
    use_lora: bool = True,
    output_file: str = None,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    max_area_ratio: float = 0.25,
    vis_output_dir: Optional[str] = None,
    compute_only: bool = False,
    use_original_sam: bool = False,
    vis_output_dir_original: Optional[str] = None,
):
    """测试SAM模型的分割成功率
    
    Args:
        compute_only: 如果为True，仅计算IoU和成功率，从vis_output_dir加载已生成的候选masks
        use_original_sam: 如果为True，使用原始SAM权重（未训练的），否则使用训练好的权重
    """
    
    # 如果只是计算模式，不需要加载模型，也不会写入任何文件
    if not compute_only:
        # 加载SAM模型
        print("=" * 80)
        if use_original_sam:
            print("加载原始SAM模型（未训练的）...")
            sam, device = load_original_sam(sam_checkpoint)
        else:
            print("加载训练好的SAM模型...")
            sam, device = load_finetuned_sam(
                sam_model_path,
                sam_checkpoint,
                use_lora=use_lora
            )
        
        # 创建predictor
        predictor = SamPredictor(sam)
    else:
        predictor = None
        print("=" * 80)
        print("仅计算模式：从已保存的masks目录加载候选masks")
    
    # 创建可视化输出目录（根据是否使用原始SAM选择不同的目录）
    # 注意：在compute_only模式下，actual_vis_output_dir仅用于读取，不会用于写入
    actual_vis_output_dir = vis_output_dir
    if use_original_sam and vis_output_dir_original:
        actual_vis_output_dir = vis_output_dir_original
    
    if actual_vis_output_dir:
        # 仅在非compute_only模式下创建目录（用于保存）
        if not compute_only:
            os.makedirs(actual_vis_output_dir, exist_ok=True)
            if use_original_sam:
                print(f"原始SAM可视化结果将保存到: {actual_vis_output_dir}")
            else:
                print(f"训练好的SAM可视化结果将保存到: {actual_vis_output_dir}")
        else:
            # compute_only模式：仅用于读取，不创建目录，不写入任何文件
            print(f"仅计算模式：从目录加载候选masks（不会修改任何文件）: {actual_vis_output_dir}")
    
    # 存储所有测试结果
    all_results = []
    success_count = 0
    total_count = 0
    
    # 三个视角
    view_names = ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']
    
    for view_name in view_names:
        print(f"\n{'='*80}")
        print(f"处理视角: {view_name}")
        print(f"{'='*80}")
        
        # 加载annotations
        annotations_path = os.path.join(annotations_dir, view_name, 'annotations.json')
        if not os.path.exists(annotations_path):
            print(f"  警告: 未找到annotations文件: {annotations_path}")
            continue
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 按图片分组
        img_to_annotations = {}
        for ann in annotations:
            img_name = ann['img_name']
            if img_name not in img_to_annotations:
                img_to_annotations[img_name] = []
            img_to_annotations[img_name].append(ann)
        
        # 获取前100张唯一图片
        unique_images = sorted(list(img_to_annotations.keys()))[:100]
        
        # 对每张图片进行测试
        for img_name in tqdm(unique_images, desc=f"  {view_name}"):
            image_path = os.path.join(images_dir, view_name, img_name)
            if not os.path.exists(image_path):
                print(f"    警告: 图像不存在: {image_path}")
                continue
            
            # 加载或生成候选masks
            if compute_only:
                # 仅计算模式：从已保存的masks目录加载（不需要读取图像）
                # 使用actual_vis_output_dir（已经根据use_original_sam选择正确的目录）
                image_stem = Path(img_name).stem
                if actual_vis_output_dir:
                    masks_dir = os.path.join(actual_vis_output_dir, view_name, image_stem, "masks")
                else:
                    print(f"    警告: 未指定可视化输出目录，无法加载候选masks ({img_name})，跳过")
                    continue
                
                if not os.path.exists(masks_dir):
                    print(f"    警告: 未找到候选masks ({img_name})，跳过")
                    continue
                
                candidate_masks = load_candidate_masks_from_dir(masks_dir)
                if len(candidate_masks) == 0:
                    print(f"    警告: 未找到候选masks ({img_name})，跳过")
                    continue
                # 仅计算模式不需要读取图像
                image_rgb = None
            else:
                # 正常模式：需要读取图像以生成候选masks
                image = cv2.imread(image_path)
                if image is None:
                    print(f"    警告: 无法读取图像: {image_path}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 正常模式：使用均匀网格点生成所有候选masks（每张图像只生成一次）
                try:
                    print(f"    生成候选masks: {img_name}")
                    candidate_masks = generate_all_candidate_masks(
                        predictor,
                        image_rgb,
                        points_per_side=points_per_side,
                        points_per_batch=points_per_batch,
                        pred_iou_thresh=pred_iou_thresh,
                        stability_score_thresh=stability_score_thresh,
                        box_nms_thresh=box_nms_thresh,
                        max_area_ratio=max_area_ratio,
                    )
                    print(f"    生成了 {len(candidate_masks)} 个候选masks")
                except Exception as e:
                    print(f"    错误: 生成masks失败 ({img_name}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                if len(candidate_masks) == 0:
                    print(f"    警告: 未生成任何候选masks ({img_name})")
                    continue
            
            # 对每个GT mask进行测试
            ann_list = img_to_annotations[img_name]
            for ann in ann_list:
                # 获取GT mask路径
                gt_path = ann.get('gt_path', '')
                if not gt_path:
                    continue
                
                gt_mask_filename = os.path.basename(gt_path.replace('\\', '/'))
                gt_mask_path = os.path.join(gt_masks_dir, view_name, "masks", gt_mask_filename)
                
                # 加载GT mask
                gt_mask = load_gt_mask(gt_mask_path)
                if gt_mask is None:
                    continue
                
                object_name = ann.get('object', 'unknown')
                
                # 过滤掉object名称中包含"ring"或"pin"的样本（不区分大小写）
                object_name_lower = object_name.lower()
                if 'ring' in object_name_lower or 'pin' in object_name_lower:
                    continue  # 跳过这个样本，不纳入计算范围
                
                # 计算GT mask与所有候选masks的IoU（使用批量计算优化）
                # 调整GT mask大小以匹配候选mask（使用第一个候选mask的尺寸作为参考）
                h, w = candidate_masks[0].shape
                if gt_mask.shape != (h, w):
                    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    # resize后需要重新转换格式：0-255灰度图 -> 0=掩码，1=背景
                    gt_mask_resized = (gt_mask_resized > 0).astype(np.uint8)  # 保持格式：0=掩码区域，1=背景
                else:
                    gt_mask_resized = gt_mask
                
                # 使用批量计算函数（内部会处理尺寸匹配）
                max_iou, best_mask_idx = compute_iou_batch(gt_mask_resized, candidate_masks)
                
                # 判断是否成功（IoU >= 0.4）
                is_success = max_iou >= success_threshold
                if is_success:
                    success_count += 1
                total_count += 1
                
                # 保存可视化结果（仅在非compute_only模式下）
                # 参考test_automatic_mask_generator.py的保存逻辑
                # 重要：compute_only模式下绝对不执行任何写入操作，确保不会修改已生成的masks
                if not compute_only and actual_vis_output_dir and best_mask_idx >= 0 and image_rgb is not None:
                    try:
                        # 创建图像特定的输出目录（与test_automatic_mask_generator.py一致）
                        image_stem = Path(img_name).stem
                        image_output_dir = os.path.join(actual_vis_output_dir, view_name, image_stem)
                        os.makedirs(image_output_dir, exist_ok=True)
                        
                        # 保存所有候选masks的overlay（与test_automatic_mask_generator.py一致）
                        # 注意：candidate_masks格式是0=掩码区域，1=背景，需要转换为bool格式
                        masks_for_overlay = []
                        for mask in candidate_masks:
                            # 调整到原图尺寸
                            orig_h, orig_w = image_rgb.shape[:2]
                            if mask.shape != (orig_h, orig_w):
                                mask_resized = cv2.resize(
                                    mask.astype(np.uint8), 
                                    (orig_w, orig_h), 
                                    interpolation=cv2.INTER_NEAREST
                                )
                            else:
                                mask_resized = mask
                            # 转换为bool格式：0=掩码区域 -> True=掩码区域
                            mask_bool = (mask_resized == 0).astype(bool)
                            masks_for_overlay.append(mask_bool)
                        
                        overlay_all = generate_colored_mask_overlay(image_rgb, masks_for_overlay, alpha=0.5)
                        overlay_all_bgr = cv2.cvtColor(overlay_all, cv2.COLOR_RGB2BGR)
                        overlay_path = os.path.join(image_output_dir, "overlay_all_masks.png")
                        cv2.imwrite(overlay_path, overlay_all_bgr)
                        
                        # 保存原图（与test_automatic_mask_generator.py一致）
                        original_path = os.path.join(image_output_dir, "original.png")
                        cv2.imwrite(original_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                        
                        # 保存每个候选mask（与test_automatic_mask_generator.py一致）
                        masks_dir = os.path.join(image_output_dir, "masks")
                        os.makedirs(masks_dir, exist_ok=True)
                        for i, mask in enumerate(candidate_masks):
                            # 调整到原图尺寸
                            orig_h, orig_w = image_rgb.shape[:2]
                            if mask.shape != (orig_h, orig_w):
                                mask_resized = cv2.resize(
                                    mask.astype(np.uint8), 
                                    (orig_w, orig_h), 
                                    interpolation=cv2.INTER_NEAREST
                                )
                            else:
                                mask_resized = mask
                            # 保存mask图像（白色背景，黑色mask）- 与test_automatic_mask_generator.py一致
                            mask_image = np.ones((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8) * 255
                            mask_image[mask_resized == 0] = [0, 0, 0]  # 0=掩码区域 -> 黑色
                            mask_path = os.path.join(masks_dir, f"mask_{i:04d}.png")
                            cv2.imwrite(mask_path, mask_image)
                        
                    except Exception as e:
                        print(f"    警告: 保存可视化失败 ({img_name}, {object_name}): {e}")
                        import traceback
                        traceback.print_exc()
                
                # 记录结果
                all_results.append({
                    'view_name': view_name,
                    'image_name': img_name,
                    'object_name': object_name,
                    'gt_mask_path': gt_mask_path,
                    'max_iou': max_iou,
                    'best_mask_idx': best_mask_idx,
                    'is_success': is_success,
                    'num_candidates': len(candidate_masks),
                })
    
    # 计算成功率
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0.0
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"测试结果统计")
    print(f"{'='*80}")
    print(f"总GT mask数: {total_count}")
    print(f"成功数（IoU >= {success_threshold}）: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_rate:.2f}%")
    
    # 按视角统计
    print(f"\n按视角统计:")
    print(f"{'-'*80}")
    for view_name in view_names:
        view_results = [r for r in all_results if r['view_name'] == view_name]
        if len(view_results) == 0:
            continue
        view_success = sum(1 for r in view_results if r['is_success'])
        view_total = len(view_results)
        view_rate = (view_success / view_total * 100) if view_total > 0 else 0.0
        avg_iou = np.mean([r['max_iou'] for r in view_results])
        print(f"{view_name}: {view_success}/{view_total} ({view_rate:.2f}%), 平均IoU: {avg_iou:.4f}")
    
    # 保存详细结果到文件
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SAM模型分割成功率测试结果\n")
            f.write("=" * 80 + "\n\n")
            if use_original_sam:
                f.write(f"模型类型: 原始SAM（未训练的）\n")
                f.write(f"原始SAM checkpoint: {sam_checkpoint}\n")
            else:
                f.write(f"模型类型: 训练好的SAM\n")
                f.write(f"模型路径: {sam_model_path}\n")
            f.write(f"成功阈值: IoU >= {success_threshold}\n")
            f.write(f"\n总GT mask数: {total_count}\n")
            f.write(f"成功数（IoU >= {success_threshold}）: {success_count}\n")
            f.write(f"失败数: {total_count - success_count}\n")
            f.write(f"成功率: {success_rate:.2f}%\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 按视角统计
            f.write("按视角统计:\n")
            f.write("-" * 80 + "\n")
            for view_name in view_names:
                view_results = [r for r in all_results if r['view_name'] == view_name]
                if len(view_results) == 0:
                    continue
                view_success = sum(1 for r in view_results if r['is_success'])
                view_total = len(view_results)
                view_rate = (view_success / view_total * 100) if view_total > 0 else 0.0
                avg_iou = np.mean([r['max_iou'] for r in view_results])
                f.write(f"{view_name}: {view_success}/{view_total} ({view_rate:.2f}%), 平均IoU: {avg_iou:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 详细结果（按IoU降序）
            f.write("详细结果（按IoU降序）:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'序号':<6} {'视角':<15} {'图像名':<20} {'物体':<20} {'IoU':<10} {'状态':<10}\n")
            f.write("-" * 80 + "\n")
            
            sorted_results = sorted(all_results, key=lambda x: x['max_iou'], reverse=True)
            for idx, result in enumerate(sorted_results, 1):
                status = "成功" if result['is_success'] else "失败"
                f.write(f"{idx:<6} {result['view_name']:<15} {result['image_name']:<20} "
                       f"{result['object_name']:<20} {result['max_iou']:<10.4f} {status:<10}\n")
            
            # 失败案例（IoU < 0.4）
            failed_results = [r for r in all_results if not r['is_success']]
            if failed_results:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"失败案例（IoU < {success_threshold}，共 {len(failed_results)} 个）:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'序号':<6} {'视角':<15} {'图像名':<20} {'物体':<20} {'IoU':<10}\n")
                f.write("-" * 80 + "\n")
                failed_sorted = sorted(failed_results, key=lambda x: x['max_iou'])
                for idx, result in enumerate(failed_sorted, 1):
                    f.write(f"{idx:<6} {result['view_name']:<15} {result['image_name']:<20} "
                           f"{result['object_name']:<20} {result['max_iou']:<10.4f}\n")
        
        print(f"\n详细结果已保存到: {output_file}")
    
    return success_rate, all_results


def main():
    parser = argparse.ArgumentParser(description="测试SAM和SAM2模型的分割成功率")
    parser.add_argument("--dataset_type", type=str, choices=["robot_arm", "vigor"], default="robot_arm",
                       help="数据集类型：robot_arm 或 vigor")
    
    # Robot Arm数据集参数
    parser.add_argument("--sam_model_path", type=str, default=None,
                       help="训练好的SAM模型checkpoint路径（使用--use_original_sam时不需要）")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                       help="原始SAM checkpoint路径")
    parser.add_argument("--annotations_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/GT_mask",
                       help="标注文件目录（包含robot_arm_01/02/03子目录）")
    parser.add_argument("--images_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/raw_pic",
                       help="图像目录（包含robot_arm_01/02/03子目录）")
    parser.add_argument("--gt_masks_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/GT_mask",
                       help="GT mask目录（包含robot_arm_01/02/03/masks子目录）")
    
    # VIGOR数据集参数
    parser.add_argument("--vigor_annotations_file", type=str,
                       default="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/all_annotations.json",
                       help="VIGOR标注文件路径")
    parser.add_argument("--vigor_dataset_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test",
                       help="VIGOR数据集目录（包含图像和masks子目录）")
    parser.add_argument("--sam_masks_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/sam_masks",
                       help="SAM生成的候选掩码目录")
    parser.add_argument("--sam_masks2_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/sam_masks2",
                       help="SAM2生成的候选掩码目录（sam_masks2）")
    
    # 通用参数
    parser.add_argument("--success_threshold", type=float, default=0.4,
                       help="成功阈值（IoU >= threshold算成功）")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="使用LoRA权重")
    parser.add_argument("--output_file", type=str,
                       default="./sam_success_rate_results.txt",
                       help="结果输出文件路径")
    parser.add_argument("--points_per_side", type=int, default=32,
                       help="每边的点数（总点数为points_per_side^2）")
    parser.add_argument("--points_per_batch", type=int, default=64,
                       help="每批处理的点数")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88,
                       help="预测IoU阈值")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95,
                       help="稳定性分数阈值")
    parser.add_argument("--box_nms_thresh", type=float, default=0.7,
                       help="NMS IoU阈值")
    parser.add_argument("--max_area_ratio", type=float, default=0.25,
                       help="最大mask面积比例（相对于图像总面积）")
    parser.add_argument("--vis_output_dir", type=str, default=None,
                       help="可视化结果输出目录（训练好的SAM，如果指定，将保存候选masks和对比图像）")
    parser.add_argument("--vis_output_dir_original", type=str, default=None,
                       help="原始SAM可视化结果输出目录（如果指定，将保存候选masks和对比图像）")
    parser.add_argument("--compute_only", action="store_true",
                       help="仅计算模式：从vis_output_dir加载已生成的候选masks，不重新生成")
    parser.add_argument("--use_original_sam", action="store_true",
                       help="使用原始SAM权重（未训练的）进行测试，用于对比")
    
    args = parser.parse_args()
    
    # VIGOR数据集模式
    if args.dataset_type == "vigor":
        print("=" * 80)
        print("使用VIGOR数据集模式")
        print("=" * 80)
        
        # 运行VIGOR测试
        sam_success_rate, sam2_success_rate, sam_results, sam2_results = test_vigor_success_rate(
            annotations_file=args.vigor_annotations_file,
            dataset_dir=args.vigor_dataset_dir,
            sam_masks_dir=args.sam_masks_dir,
            sam_masks2_dir=args.sam_masks2_dir,
            success_threshold=args.success_threshold,
            output_file=args.output_file,
        )
        
        print(f"\n测试完成！")
        print(f"SAM成功率: {sam_success_rate:.2f}%")
        print(f"SAM2成功率: {sam2_success_rate:.2f}%")
    
    # Robot Arm数据集模式（原有逻辑）
    else:
        # 如果启用仅计算模式，vis_output_dir必须指定
        if args.compute_only and not args.vis_output_dir:
            parser.error("--compute_only模式需要指定--vis_output_dir")
        
        # 如果不是仅计算模式，需要模型路径
        if not args.compute_only:
            if not args.sam_checkpoint:
                parser.error("正常模式需要指定--sam_checkpoint")
            # 如果使用原始SAM，不需要训练权重路径
            if not args.use_original_sam and not args.sam_model_path:
                parser.error("正常模式需要指定--sam_model_path（或使用--use_original_sam使用原始SAM）")
        
        # 运行测试
        success_rate, results = test_sam_success_rate(
            sam_model_path=args.sam_model_path or "",
            sam_checkpoint=args.sam_checkpoint or "",
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            gt_masks_dir=args.gt_masks_dir,
            success_threshold=args.success_threshold,
            use_lora=args.use_lora,
            output_file=args.output_file,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            box_nms_thresh=args.box_nms_thresh,
            max_area_ratio=args.max_area_ratio,
            vis_output_dir=args.vis_output_dir,
            compute_only=args.compute_only,
            use_original_sam=args.use_original_sam,
            vis_output_dir_original=args.vis_output_dir_original,
        )
        
        print(f"\n测试完成！成功率: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
