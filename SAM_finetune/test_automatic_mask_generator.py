#!/usr/bin/env python3
"""
使用训练好的SAM模型进行自动mask生成测试
基于automatic_mask_generator的方法，生成所有mask并保存
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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
    box_xyxy_to_xywh,
    mask_to_rle_pytorch,
    rle_to_mask,
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


def apply_lora_to_sam(model, lora_config: Dict[str, Any]):
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
    
    if lora_config.get('apply_to_image_encoder', False):
        print("对image_encoder应用LoRA...")
        model.image_encoder = get_peft_model(model.image_encoder, peft_config)
    
    return model


def load_finetuned_sam(checkpoint_path: str, sam_checkpoint: str, use_lora: bool = True, load_trained_weights: bool = True):
    """
    加载SAM模型（可以是训练好的或原始的）
    
    Args:
        checkpoint_path: 训练好的checkpoint路径（如果load_trained_weights=False，可以为None）
        sam_checkpoint: 原始SAM checkpoint路径
        use_lora: 是否使用LoRA（仅在加载训练权重时有效）
        load_trained_weights: 是否加载训练权重，如果False则只加载原始SAM权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载基础SAM模型
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    
    if not load_trained_weights:
        # 只使用原始SAM权重
        print(f"\n使用原始SAM权重（未加载训练权重）")
        print(f"  SAM checkpoint: {sam_checkpoint}")
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False
        print("✅ 原始SAM模型加载完成")
        return sam, device
    
    # 加载训练权重
    if checkpoint_path is None:
        raise ValueError("load_trained_weights=True时，checkpoint_path不能为None")
    
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


def generate_colored_mask_overlay(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """生成带颜色的mask叠加图像"""
    overlay = image.copy()
    
    # 生成不同颜色
    num_masks = len(masks)
    colors = []
    for i in range(num_masks):
        hue = i / max(num_masks, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    # 叠加每个mask
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
    
    return overlay


def generate_masks_for_image(
    predictor: SamPredictor,
    image: np.ndarray,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    max_area_ratio: float = 0.25,
) -> Tuple[List[np.ndarray], List[Dict], List[np.ndarray]]:
    """为单张图像生成所有mask"""
    orig_size = image.shape[:2]
    
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
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
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
        
        # 计算稳定性分数（在过滤之前计算，以便调试）
        stability_scores = calculate_stability_score(
            masks,
            predictor.model.mask_threshold,
            1.0,
        )
        
        # 处理NaN值：如果unions为0，stability_score会是NaN，将其设为0
        stability_scores = torch.where(
            torch.isnan(stability_scores),
            torch.zeros_like(stability_scores),
            stability_scores
        )
        
        # 调试信息：打印第一个batch的统计信息（仅第一次）
        if len(all_masks) == 0 and len(masks) > 0:
            print(f"  调试信息: 第一个batch有 {len(masks)} 个mask候选")
            print(f"    IoU预测值范围: [{iou_preds.min().item():.3f}, {iou_preds.max().item():.3f}], 均值: {iou_preds.mean().item():.3f}")
            print(f"    稳定性分数范围: [{stability_scores.min().item():.3f}, {stability_scores.max().item():.3f}], 均值: {stability_scores.mean().item():.3f}")
            print(f"    过滤阈值: IoU>{pred_iou_thresh:.2f}, 稳定性>{stability_score_thresh:.2f}")
        
        # 过滤低质量mask（按照SAM官方实现：先过滤IoU，再过滤稳定性分数）
        # 第一步：过滤IoU
        if pred_iou_thresh > 0.0:
            keep_mask_iou = iou_preds > pred_iou_thresh
            num_before_iou = len(masks)
            masks = masks[keep_mask_iou]
            iou_preds = iou_preds[keep_mask_iou]
            stability_scores = stability_scores[keep_mask_iou]
            num_after_iou = len(masks)
            if len(all_masks) == 0:  # 只在第一个batch打印
                print(f"    IoU过滤: {num_before_iou} -> {num_after_iou} (保留 {num_after_iou/num_before_iou*100:.1f}%)")
        else:
            num_after_iou = len(masks)
        
        # 第二步：过滤稳定性分数
        if stability_score_thresh > 0.0 and len(masks) > 0:
            keep_mask_stability = stability_scores >= stability_score_thresh
            num_before_stability = len(masks)
            masks = masks[keep_mask_stability]
            iou_preds = iou_preds[keep_mask_stability]
            stability_scores = stability_scores[keep_mask_stability]
            num_after_stability = len(masks)
            if len(all_masks) == 0:  # 只在第一个batch打印
                print(f"    稳定性过滤: {num_before_stability} -> {num_after_stability} (保留 {num_after_stability/num_before_stability*100:.1f}%)")
        
        # 如果没有mask通过过滤，跳过这个batch
        if len(masks) == 0:
            continue
            
        # 二值化mask
        masks_binary = masks > predictor.model.mask_threshold
        
        # 计算boxes
        boxes = batched_mask_to_box(masks_binary)
        
        # 第三步：过滤面积过大的mask（超过图像面积的max_area_ratio）
        if max_area_ratio > 0.0 and max_area_ratio < 1.0:
            image_area = orig_size[0] * orig_size[1]  # H * W
            max_area = image_area * max_area_ratio
            
            # 计算每个mask的面积
            mask_areas = masks_binary.sum(dim=(1, 2))  # [N] 每个mask的像素数
            
            # 过滤面积过大的mask
            keep_mask_area = mask_areas <= max_area
            num_before_area = len(masks_binary)
            masks_binary = masks_binary[keep_mask_area]
            boxes = boxes[keep_mask_area]
            iou_preds = iou_preds[keep_mask_area]
            stability_scores = stability_scores[keep_mask_area]
            num_after_area = len(masks_binary)
            if len(all_masks) == 0:  # 只在第一个batch打印
                print(f"    面积过滤: {num_before_area} -> {num_after_area} (保留 {num_after_area/num_before_area*100:.1f}%, 最大面积={max_area_ratio*100:.1f}%)")
        
        # 存储
        for j in range(len(masks_binary)):
            mask_np = masks_binary[j].cpu().numpy()
            all_masks.append(mask_np)
            all_scores.append({
                'iou': iou_preds[j].item(),
                'stability': stability_scores[j].item(),
            })
            all_boxes.append(boxes[j].cpu().numpy())
    
    predictor.reset_image()
    
    # NMS去重
    num_before_nms = len(all_boxes)
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
        all_scores = [all_scores[i] for i in keep_indices]
        all_boxes = [all_boxes[i] for i in keep_indices]
        num_after_nms = len(all_masks)
        print(f"    NMS去重: {num_before_nms} -> {num_after_nms} (保留 {num_after_nms/num_before_nms*100:.1f}%)")
    
    return all_masks, all_scores, all_boxes


def generate_masks_from_affordance_points(
    predictor: SamPredictor,
    image: np.ndarray,
    image_path: str,
    annotations_file: str,
) -> Tuple[List[np.ndarray], List[Dict], List[np.ndarray]]:
    """使用标注中的affordance点生成masks（与训练时一致）"""
    # 加载标注
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 找到当前图像的标注
    image_name = Path(image_path).name
    image_annotations = [ann for ann in annotations if ann.get('img_name') == image_name]
    
    if len(image_annotations) == 0:
        print(f"  警告: 没有找到 {image_name} 的标注，使用空结果")
        return [], [], []
    
    orig_size = image.shape[:2]
    predictor.set_image(image)
    
    all_masks = []
    all_scores = []
    all_boxes = []
    
    # 对每个标注的affordance点进行预测
    for ann in image_annotations:
        affordance_points = ann.get('affordance_points', [])
        if len(affordance_points) == 0:
            continue
        
        # 使用第一个affordance点（与训练时一致）
        first_point = affordance_points[0]
        x, y = first_point[0], first_point[1]
        
        # 转换坐标
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = int(x * orig_size[1])
            y = int(y * orig_size[0])
        else:
            x = int(x)
            y = int(y)
        
        x = np.clip(x, 0, orig_size[1] - 1)
        y = np.clip(y, 0, orig_size[0] - 1)
        
        # 转换到模型输入坐标系（与训练时完全一致）
        # 训练时：points_array = np.array(points_list, dtype=np.float32)  # [1, 2]
        #         transformed_points = self.transform_image.apply_coords(points_array, original_size)
        #         points_torch = torch.as_tensor(transformed_points, dtype=torch.float32)  # [1, 2]
        #         然后添加batch维度: point_coords.unsqueeze(0) -> [1, 1, 2]
        point_coords = np.array([[x, y]], dtype=np.float32)  # [1, 2] - 与训练时一致
        transformed_points = predictor.transform.apply_coords(point_coords, orig_size)  # [1, 2]
        in_points = torch.as_tensor(transformed_points, dtype=torch.float32, device=predictor.device)  # [1, 2]
        in_labels = torch.ones(1, dtype=torch.int, device=predictor.device)  # [1]
        
        # 添加batch维度（与训练时一致）
        # 训练时：point_coords.unsqueeze(0) -> [1, N, 2] = [1, 1, 2]
        in_points_batched = in_points.unsqueeze(0)  # [1, 1, 2] - 与训练时一致
        in_labels_batched = in_labels.unsqueeze(0)  # [1, 1] - 与训练时一致
        
        # 预测（使用multimask_output=False，与训练时一致）
        masks, iou_preds, _ = predictor.predict_torch(
            in_points_batched,  # [1, 1, 2] - 与训练时一致
            in_labels_batched,  # [1, 1] - 与训练时一致
            multimask_output=False,  # 与训练时一致
            return_logits=True,
        )
        
        # masks: [1, 1, H, W], iou_preds: [1, 1]
        mask = masks[0, 0]  # [H, W]
        iou_pred = iou_preds[0, 0].item()
        
        # 二值化
        mask_binary = (mask > predictor.model.mask_threshold).cpu().numpy()
        
        # 计算box
        box = batched_mask_to_box(torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0))[0]
        box_np = box.cpu().numpy()
        
        all_masks.append(mask_binary)
        all_scores.append({
            'iou': iou_pred,
            'stability': 1.0,  # 单个点预测，不需要稳定性分数
        })
        all_boxes.append(box_np)
    
    predictor.reset_image()
    
    return all_masks, all_scores, all_boxes


def generate_masks_from_bbox(
    predictor: SamPredictor,
    image: np.ndarray,
    image_path: str,
    annotations_file: str,
) -> Tuple[List[np.ndarray], List[Dict], List[np.ndarray]]:
    """使用标注中的bbox生成masks（与训练时一致）"""
    # 加载标注
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 找到当前图像的标注
    image_name = Path(image_path).name
    image_annotations = [ann for ann in annotations if ann.get('img_name') == image_name]
    
    if len(image_annotations) == 0:
        print(f"  警告: 没有找到 {image_name} 的标注，使用空结果")
        return [], [], []
    
    orig_size = image.shape[:2]  # (H, W)
    predictor.set_image(image)
    
    all_masks = []
    all_scores = []
    all_boxes = []
    
    # 对每个标注的bbox进行预测
    for ann in image_annotations:
        bbox_ann = ann.get('bbox', None)
        if bbox_ann is None or len(bbox_ann) < 4:
            continue
        
        # 标注中的bbox格式：[x, y, width, height]（归一化，0-1之间）
        # 转换为像素坐标的 [x_min, y_min, x_max, y_max] 格式（与训练时一致）
        x_norm, y_norm, w_norm, h_norm = bbox_ann[0], bbox_ann[1], bbox_ann[2], bbox_ann[3]
        
        # 转换为像素坐标（与训练脚本中的转换方式一致）
        x_min = x_norm * orig_size[1]  # width
        y_min = y_norm * orig_size[0]  # height
        x_max = x_min + w_norm * orig_size[1]
        y_max = y_min + h_norm * orig_size[0]
        
        # 确保坐标在图像范围内
        x_min = np.clip(x_min, 0, orig_size[1] - 1)
        y_min = np.clip(y_min, 0, orig_size[0] - 1)
        x_max = np.clip(x_max, 0, orig_size[1] - 1)
        y_max = np.clip(y_max, 0, orig_size[0] - 1)
        
        # 转换为XYXY格式（predict_torch需要的格式）
        bbox_xyxy = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)  # [1, 4]
        
        # 转换到模型输入坐标系（与训练时一致）
        # 训练时：bbox_torch = self.transform_image.apply_boxes(np.array(bbox).reshape(1, 4), original_size)
        transformed_bbox = predictor.transform.apply_boxes(bbox_xyxy, orig_size)  # [1, 4]
        in_boxes = torch.as_tensor(transformed_bbox, dtype=torch.float32, device=predictor.device)  # [1, 4]
        
        # 预测（使用multimask_output=False，与训练时一致）
        masks, iou_preds, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=in_boxes,  # [1, 4] - XYXY格式，与训练时一致
            multimask_output=False,  # 与训练时一致
            return_logits=True,
        )
        
        # masks: [1, 1, H, W], iou_preds: [1, 1]
        mask = masks[0, 0]  # [H, W]
        iou_pred = iou_preds[0, 0].item()
        
        # 二值化
        mask_binary = (mask > predictor.model.mask_threshold).cpu().numpy()
        
        # 使用预测的bbox（与训练时一致）
        box_np = bbox_xyxy[0]  # [4] - XYXY格式
        
        all_masks.append(mask_binary)
        all_scores.append({
            'iou': iou_pred,
            'stability': 1.0,  # bbox预测，不需要稳定性分数
        })
        all_boxes.append(box_np)
    
    predictor.reset_image()
    
    return all_masks, all_scores, all_boxes


def process_image(
    predictor: SamPredictor,
    image_path: str,
    output_dir: str,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    max_area_ratio: float = 0.25,
    annotations_file: Optional[str] = None,
    use_affordance_points: bool = False,
    use_bbox: bool = False,
):
    """处理单张图像，生成并保存所有mask"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 生成masks
    print(f"  生成masks...")
    
    # 根据模式选择不同的生成方法
    if use_bbox and annotations_file:
        # 使用bbox（与训练时一致）
        masks, scores, boxes = generate_masks_from_bbox(
            predictor,
            image_rgb,
            image_path,
            annotations_file,
        )
    elif use_affordance_points and annotations_file:
        # 使用affordance点
        masks, scores, boxes = generate_masks_from_affordance_points(
            predictor,
            image_rgb,
            image_path,
            annotations_file,
        )
    else:
        # 使用网格点（原始方法）
        masks, scores, boxes = generate_masks_for_image(
            predictor,
            image_rgb,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            max_area_ratio=max_area_ratio,
        )
    
    print(f"  生成了 {len(masks)} 个masks")
    
    # 创建输出目录
    image_name = Path(image_path).stem
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 保存每个mask
    masks_dir = os.path.join(image_output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        # 保存mask图像（白色背景，黑色mask）
        mask_image = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
        mask_image[mask] = [0, 0, 0]
        mask_path = os.path.join(masks_dir, f"mask_{i:04d}_iou{score['iou']:.3f}_stab{score['stability']:.3f}.png")
        cv2.imwrite(mask_path, mask_image)
    
    # 生成并保存合并的mask图像
    if len(masks) > 0:
        overlay = generate_colored_mask_overlay(image_rgb, masks, alpha=0.5)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = os.path.join(image_output_dir, "overlay_all_masks.png")
        cv2.imwrite(overlay_path, overlay_bgr)
        
        # 也保存原图
        original_path = os.path.join(image_output_dir, "original.png")
        cv2.imwrite(original_path, image)
        
        print(f"  保存到: {image_output_dir}")
        print(f"    - {len(masks)} 个单独masks")
        print(f"    - 1 个合并overlay图像")


def main():
    parser = argparse.ArgumentParser(description="使用SAM模型生成所有mask（可以是训练好的或原始的）")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="训练好的checkpoint路径（如果使用--use_original_sam则不需要）")
    parser.add_argument("--sam_checkpoint", type=str, required=True,
                       help="原始SAM checkpoint路径")
    parser.add_argument("--use_original_sam", action="store_true",
                       help="使用原始SAM权重（不加载训练权重），用于对比测试")
    parser.add_argument("--test_images_dir", type=str, 
                       default="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/dataset/picture/robot_arm_02",
                       help="测试图像目录")
    parser.add_argument("--output_dir", type=str,
                       default="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/test_output",
                       help="输出目录")
    parser.add_argument("--points_per_side", type=int, default=32,
                       help="每边的点数（总点数为points_per_side^2）")
    parser.add_argument("--points_per_batch", type=int, default=64,
                       help="每批处理的点数")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.5,
                       help="预测IoU阈值（降低以提高召回率）")
    parser.add_argument("--stability_score_thresh", type=float, default=0.5,
                       help="稳定性分数阈值（降低以提高召回率）")
    parser.add_argument("--box_nms_thresh", type=float, default=0.7,
                       help="NMS IoU阈值")
    parser.add_argument("--max_area_ratio", type=float, default=0.25,
                       help="最大mask面积比例（相对于图像总面积），超过此比例的mask将被过滤，默认0.25（1/4）")
    parser.add_argument("--use_lora", action="store_true",
                       help="使用LoRA权重")
    parser.add_argument("--annotations_file", type=str, default=None,
                       help="标注JSON文件路径（如果使用affordance点或bbox测试）")
    parser.add_argument("--use_affordance_points", action="store_true",
                       help="使用标注中的affordance点进行测试")
    parser.add_argument("--use_bbox", action="store_true",
                       help="使用标注中的bbox进行测试（与训练时一致）")
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.use_original_sam and args.checkpoint is None:
        parser.error("必须提供--checkpoint路径，或使用--use_original_sam使用原始SAM权重")
    
    if args.use_bbox and args.annotations_file is None:
        parser.error("使用--use_bbox时必须提供--annotations_file")
    if args.use_affordance_points and args.annotations_file is None:
        parser.error("使用--use_affordance_points时必须提供--annotations_file")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("=" * 50)
    if args.use_original_sam:
        print("加载原始SAM模型（未加载训练权重）...")
    else:
        print("加载训练好的SAM模型...")
    
    sam, device = load_finetuned_sam(
        args.checkpoint,
        args.sam_checkpoint,
        use_lora=args.use_lora if not args.use_original_sam else False,  # 原始SAM不需要LoRA
        load_trained_weights=not args.use_original_sam
    )
    
    # 创建predictor
    predictor = SamPredictor(sam)
    
    # 获取所有图像文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(args.test_images_dir).glob(f"*{ext}"))
        image_files.extend(Path(args.test_images_dir).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    print(f"\n找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    print("\n" + "=" * 50)
    print("开始处理图像...")
    for image_path in tqdm(image_files, desc="处理图像"):
        print(f"\n处理: {image_path.name}")
        try:
            process_image(
                predictor,
                str(image_path),
                args.output_dir,
                points_per_side=args.points_per_side,
                points_per_batch=args.points_per_batch,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
                box_nms_thresh=args.box_nms_thresh,
                max_area_ratio=args.max_area_ratio,
                annotations_file=args.annotations_file,
                use_affordance_points=args.use_affordance_points,
                use_bbox=args.use_bbox,
            )
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("完成！")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

