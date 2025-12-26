#!/usr/bin/env python3
"""
SAM模型LoRA微调脚本
使用工业零件标注数据集微调SAM模型
支持LoRA微调，显存友好，训练速度快
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import random
from PIL import Image
import colorsys

# SAM imports
# 添加项目根目录到路径，以便导入segment_anything
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # SAM_finetune -> LLMSeg (项目根目录)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 从model.segment_anything导入（因为segment_anything在model目录下）
from model.segment_anything import sam_model_registry
from model.segment_anything.utils.transforms import ResizeLongestSide

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use full fine-tuning")
    

# SwanLab imports
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available, training metrics will not be logged")


class SAMDataset(Dataset):
    """SAM微调数据集"""
    
    def __init__(self, 
                 annotations_file: str,
                 images_dir: str,
                 masks_dir: str,
                 image_size: int = 1024,
                 transform=None,
                 additional_image_dirs: list = None,
                 dataset_configs: list = None):
        """
        初始化数据集
        
        Args:
            annotations_file: 标注JSON文件路径
            images_dir: 主图像目录路径
            masks_dir: mask目录路径
            image_size: 图像尺寸
            transform: 数据增强变换
            additional_image_dirs: 额外的图像目录列表（如果主目录找不到图像，会从这些目录查找）
            dataset_configs: 数据集配置列表，包含每个数据集的名称和图像目录路径
        """
        self.image_size = image_size
        self.transform = transform
        self.transform_image = ResizeLongestSide(image_size)
        
        # 加载标注数据
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.images_dir = images_dir
        self.additional_image_dirs = additional_image_dirs or []
        self.masks_dir = masks_dir
        
        # ✅ 创建数据集名称到图像目录的映射
        self.dataset_image_dirs = {}
        if dataset_configs:
            for config in dataset_configs:
                dataset_name = config.get('name')
                images_dir_path = config.get('images_dir')
                if dataset_name and images_dir_path:
                    self.dataset_image_dirs[dataset_name] = images_dir_path
        
        # 过滤无效的标注
        self.valid_annotations = []
        for ann in self.annotations:
            # ✅ 根据数据集名称查找图像（确保图像和mask来自同一个数据集）
            dataset_name = ann.get('_dataset_name', None)
            img_path = self._find_image(ann['img_name'], dataset_name=dataset_name)
            # gt_path可能是绝对路径或相对路径
            if os.path.isabs(ann.get('gt_path', '')):
                mask_path = ann['gt_path']
            else:
                mask_path = os.path.join(masks_dir, os.path.basename(ann['gt_path']))
            
            if img_path and os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_annotations.append(ann)
        
        print(f"加载了 {len(self.valid_annotations)} 个有效标注（共 {len(self.annotations)} 个）")
    
    def _find_image(self, img_name, dataset_name=None):
        """
        从指定数据集的图像目录中查找图像
        
        Args:
            img_name: 图像文件名
            dataset_name: 数据集名称（如果提供，优先从该数据集的图像目录查找）
        """
        # ✅ 如果提供了dataset_name，优先从该数据集的图像目录查找
        if dataset_name and dataset_name in self.dataset_image_dirs:
            img_path = os.path.join(self.dataset_image_dirs[dataset_name], img_name)
            if os.path.exists(img_path):
                return img_path
        
        # 首先尝试主目录
        img_path = os.path.join(self.images_dir, img_name)
        if os.path.exists(img_path):
            return img_path
        
        # 如果主目录找不到，尝试额外目录
        for additional_dir in self.additional_image_dirs:
            img_path = os.path.join(additional_dir, img_name)
            if os.path.exists(img_path):
                return img_path
        
        return None
    
    def __len__(self):
        return len(self.valid_annotations)
    
    def __getitem__(self, idx):
        ann = self.valid_annotations[idx]
        
        # ✅ 根据数据集名称加载图像（确保图像和mask来自同一个数据集）
        dataset_name = ann.get('_dataset_name', None)
        img_path = self._find_image(ann['img_name'], dataset_name=dataset_name)
        
        if img_path is None:
            raise FileNotFoundError(
                f"找不到图像文件: {ann['img_name']}, "
                f"数据集: {dataset_name}, "
                f"已尝试的目录: {list(self.dataset_image_dirs.values()) if self.dataset_image_dirs else [self.images_dir] + self.additional_image_dirs}"
            )
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)
        
        # 加载mask（gt_path可能是绝对路径）
        if os.path.isabs(ann.get('gt_path', '')):
            mask_path = ann['gt_path']
        else:
            mask_path = os.path.join(self.masks_dir, os.path.basename(ann['gt_path']))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 验证图像和mask尺寸是否一致（不强制对齐，如果尺寸不一致则报错）
        if mask.shape[:2] != original_size:
            raise ValueError(
                f"图像和mask尺寸不一致: 图像={original_size}, mask={mask.shape[:2]}, "
                f"图像路径={img_path}, mask路径={mask_path}"
            )
        
        mask = (mask < 128).astype(np.uint8)  # 转换为二值mask (0/1)
        ground_truth_mask = torch.from_numpy(mask.astype(np.float32))
        
        # ✅ 使用 affordance_points 的第一个点作为 prompt（而不是 bbox）
        affordance_points = ann.get('affordance_points', [])
        
        # ❌ 如果没有 affordance_points，直接报错
        if len(affordance_points) == 0:
            raise ValueError(
                f"标注中缺少 affordance_points 字段: "
                f"图像={img_path}, 标注={ann.get('img_name', 'unknown')}"
            )
        
        # ✅ 只使用第一个点
        first_point = affordance_points[0]
        if len(first_point) < 2:
            raise ValueError(
                f"affordance_points 第一个点格式错误（需要至少2个坐标）: "
                f"图像={img_path}, 点={first_point}"
            )
        
        x, y = first_point[0], first_point[1]
        
        # 如果坐标是归一化的（0-1之间），转换为像素坐标
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = int(x * original_size[1])
            y = int(y * original_size[0])
        else:
            x = int(x)
            y = int(y)
        
        # 确保坐标在图像范围内
        x = np.clip(x, 0, original_size[1] - 1)
        y = np.clip(y, 0, original_size[0] - 1)
        
        # 只使用第一个点，转换为 [N, 2] 格式（N=1）
        points_list = [[x, y]]
        
        # 应用变换到图像
        image_tensor = self.transform_image.apply_image(image)
        transformed_size = image_tensor.shape[:2]  # (H, W) - 变换后的图像尺寸（preprocess之前）
        image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # HWC -> CHW
        
        # 注意：返回 [C, H, W]，不要添加batch维度，SAM的preprocess会处理
        # image_tensor 应该是 [C, H, W]，而不是 [1, C, H, W]
        
        # 转换 points 到模型输入坐标系
        points_array = np.array(points_list, dtype=np.float32)  # [1, 2] - 只有一个点
        transformed_points = self.transform_image.apply_coords(points_array, original_size)
        points_torch = torch.as_tensor(transformed_points, dtype=torch.float32)
        
        # point_labels: 1 表示前景点（affordance point 是前景点）
        point_labels = torch.ones(1, dtype=torch.int)  # 只有一个点
        
        # 返回SAM forward需要的格式
        return {
            'image': image_tensor,  # [C, H, W]
            'original_size': original_size,  # (H, W) - 原始图像尺寸
            'transformed_size': transformed_size,  # (H, W) - 变换后的图像尺寸（preprocess之前，用于正确resize GT mask）
            'points': points_torch,  # [N, 2]
            'point_labels': point_labels,  # [N]
            'ground_truth_mask': ground_truth_mask,  # [H, W] - 原始尺寸的GT mask
            'image_name': ann['img_name'],
            'original_image': image,  # 保存原始图像用于验证集可视化
        }


def dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算Dice损失
    注意: pred_mask应该是logits，函数内部会应用sigmoid
    """
    # 确保pred_mask和gt_mask的形状一致
    # pred_mask可能是[B, 1, H, W]或[B, H, W]（logits）
    # gt_mask可能是[B, 1, H, W]或[B, H, W]（0/1）
    
    # 如果维度不匹配，调整维度
    if pred_mask.dim() == 4 and gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    elif pred_mask.dim() == 3 and gt_mask.dim() == 4:
        pred_mask = pred_mask.unsqueeze(1)
    
    # 如果channel维度不匹配，取第一个channel
    if pred_mask.dim() == 4 and gt_mask.dim() == 4:
        if pred_mask.shape[1] != gt_mask.shape[1]:
            if pred_mask.shape[1] > 1:
                pred_mask = pred_mask[:, 0:1, :, :]
            if gt_mask.shape[1] > 1:
                gt_mask = gt_mask[:, 0:1, :, :]
    
    # 确保batch size一致
    if pred_mask.shape[0] != gt_mask.shape[0]:
        min_batch = min(pred_mask.shape[0], gt_mask.shape[0])
        pred_mask = pred_mask[:min_batch]
        gt_mask = gt_mask[:min_batch]
    
    # 对logits应用sigmoid，转换为概率 [0, 1]
    pred_prob = torch.sigmoid(pred_mask)
    
    # 展平：从[B, 1, H, W]或[B, H, W]到[B, H*W]
    pred_flat = pred_prob.flatten(1)
    gt_flat = gt_mask.flatten(1)
    
    # 确保维度一致
    if pred_flat.shape[0] != gt_flat.shape[0]:
        min_batch = min(pred_flat.shape[0], gt_flat.shape[0])
        pred_flat = pred_flat[:min_batch]
        gt_flat = gt_flat[:min_batch]
    
    intersection = (pred_flat * gt_flat).sum(1)
    pred_sum = pred_flat.sum(1)
    gt_sum = gt_flat.sum(1)
    # 正确的union计算: union = pred_sum + gt_sum - intersection
    union = pred_sum + gt_sum - intersection
    
    # 处理边界情况：如果union为0（两个mask都为空），dice设为1（完全匹配）
    # 这样dice_loss = 0，表示没有损失
    dice = torch.where(
        union > eps,
        (2.0 * intersection + eps) / (union + eps),
        torch.ones_like(intersection)  # 两个mask都为空时，dice=1
    )
    
    dice_loss = 1.0 - dice.mean()
    # 确保dice_loss在合理范围内 [0, 1]
    dice_loss = torch.clamp(dice_loss, min=0.0, max=1.0)
    return dice_loss


def focal_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """计算Focal损失"""
    # 确保pred_mask和gt_mask的形状一致
    if pred_mask.dim() == 4 and gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    elif pred_mask.dim() == 3 and gt_mask.dim() == 4:
        pred_mask = pred_mask.unsqueeze(1)
    
    # 如果channel维度不匹配，取第一个channel
    if pred_mask.dim() == 4 and gt_mask.dim() == 4:
        if pred_mask.shape[1] != gt_mask.shape[1]:
            if pred_mask.shape[1] > 1:
                pred_mask = pred_mask[:, 0:1, :, :]
            if gt_mask.shape[1] > 1:
                gt_mask = gt_mask[:, 0:1, :, :]
    
    # 确保batch size一致
    if pred_mask.shape[0] != gt_mask.shape[0]:
        min_batch = min(pred_mask.shape[0], gt_mask.shape[0])
        pred_mask = pred_mask[:min_batch]
        gt_mask = gt_mask[:min_batch]
    
    pred_flat = pred_mask.flatten(1)
    gt_flat = gt_mask.flatten(1)
    
    # 确保维度一致
    if pred_flat.shape[0] != gt_flat.shape[0]:
        min_batch = min(pred_flat.shape[0], gt_flat.shape[0])
        pred_flat = pred_flat[:min_batch]
        gt_flat = gt_flat[:min_batch]
    
    bce = nn.functional.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction='none')
    p_t = torch.exp(-bce)
    focal = alpha * (1 - p_t) ** gamma * bce
    
    return focal.mean()


def compute_loss(pred_masks: List[torch.Tensor], 
                 gt_mask: torch.Tensor,
                 iou_pred: torch.Tensor = None,
                 loss_weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
    """
    计算总损失
    
    Args:
        pred_masks: 预测的mask列表（多尺度输出）
        gt_mask: 真实mask
        iou_pred: 预测的IoU（可选）
        loss_weights: 损失权重
    """
    if loss_weights is None:
        loss_weights = {
            'dice': 1.0,
            'focal': 1.0,
            'ce': 0.0,
            'iou': 0.0
        }
    
    # 使用最高分辨率的mask计算损失
    pred_mask = pred_masks[0]  # (B, 1, H, W) 或 (B, num_masks, H, W)
    
    # 如果pred_mask有多个mask（multimask_output=True），选择第一个
    if pred_mask.dim() == 4 and pred_mask.shape[1] > 1:
        pred_mask = pred_mask[:, 0:1, :, :]  # 选择第一个mask: (B, 1, H, W)
    
    # 确保pred_mask和gt_mask的batch size一致
    if pred_mask.shape[0] != gt_mask.shape[0]:
        min_batch = min(pred_mask.shape[0], gt_mask.shape[0])
        pred_mask = pred_mask[:min_batch]
        gt_mask = gt_mask[:min_batch]
        # 使用print而不是logger，因为compute_loss可能在没有logger的情况下被调用
        print(f"Warning: Batch size mismatch in compute_loss: pred={pred_mask.shape[0]}, gt={gt_mask.shape[0]}, using {min_batch}")
    
    # 调整gt_mask尺寸到pred_mask尺寸
    if gt_mask.shape[-2:] != pred_mask.shape[-2:]:
        gt_mask_resized = nn.functional.interpolate(
            gt_mask, size=pred_mask.shape[-2:], mode='nearest'
        )
    else:
        gt_mask_resized = gt_mask
    
    # 确保维度一致
    if pred_mask.shape != gt_mask_resized.shape:
        # 如果形状不匹配，尝试调整
        if pred_mask.dim() == 4 and gt_mask_resized.dim() == 3:
            gt_mask_resized = gt_mask_resized.unsqueeze(1)
        elif pred_mask.dim() == 3 and gt_mask_resized.dim() == 4:
            pred_mask = pred_mask.unsqueeze(1)
        elif pred_mask.shape[1] != gt_mask_resized.shape[1]:
            # 如果channel维度不匹配，取第一个channel
            if pred_mask.shape[1] > 1:
                pred_mask = pred_mask[:, 0:1, :, :]
            if gt_mask_resized.shape[1] > 1:
                gt_mask_resized = gt_mask_resized[:, 0:1, :, :]
    
    # Dice损失
    dice = dice_loss(pred_mask, gt_mask_resized) * loss_weights['dice']
    
    # Focal损失
    focal = focal_loss(pred_mask, gt_mask_resized) * loss_weights['focal']
    
    # BCE损失（可选，对边界更敏感）
    if loss_weights['ce'] > 0:
        ce = nn.functional.binary_cross_entropy_with_logits(
            pred_mask, gt_mask_resized
        ) * loss_weights['ce']
    else:
        ce = torch.tensor(0.0, device=pred_mask.device, dtype=pred_mask.dtype)
    
    # IoU损失（可选）
    if loss_weights['iou'] > 0:
        # 计算真实IoU（直接计算1-IoU作为损失，不依赖iou_pred）
        pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
        intersection = (pred_binary * gt_mask_resized).sum(dim=(1, 2, 3))
        union = (pred_binary + gt_mask_resized).sum(dim=(1, 2, 3)) - intersection
        iou_true = (intersection + 1e-7) / (union + 1e-7)
        # IoU损失 = 1 - IoU（越小越好）
        iou_loss = (1.0 - iou_true.mean()) * loss_weights['iou']
        
        # 如果提供了iou_pred，也可以计算MSE损失
        if iou_pred is not None:
            iou_loss_mse = nn.functional.mse_loss(iou_pred.squeeze(), iou_true) * loss_weights['iou']
            iou_loss = 0.5 * iou_loss + 0.5 * iou_loss_mse
    else:
        iou_loss = torch.tensor(0.0, device=pred_mask.device, dtype=pred_mask.dtype)
    
    total_loss = dice + focal + ce + iou_loss
    
    return {
        'total': total_loss,
        'dice': dice,
        'focal': focal,
        'ce': ce,
        'iou': iou_loss
    }


class MaskDecoderWrapper(nn.Module):
    """
    包装器类，用于正确处理PEFT包装后的mask_decoder
    过滤掉PEFT可能传递的错误参数（如input_ids）
    
    参考peft-sam的实现方式，通过包装器确保参数正确传递
    """
    def __init__(self, mask_decoder):
        super().__init__()
        # 如果mask_decoder已经被PEFT包装，直接使用base_model
        if hasattr(mask_decoder, 'base_model'):
            # 将base_model注册为子模块（这样LoRA参数也会被包含）
            self.add_module('mask_decoder', mask_decoder.base_model)
            # 保存PEFT包装器以便访问LoRA参数
            self._peft_model = mask_decoder
        else:
            # 将mask_decoder注册为子模块
            self.add_module('mask_decoder', mask_decoder)
            self._peft_model = None
    
    def forward(self, 
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool):
        """
        转发调用到mask_decoder，只传递SAM需要的参数
        这样可以避免PEFT传递input_ids等错误参数
        """
        # 直接通过_modules访问，避免触发__getattr__
        return self._modules['mask_decoder'](
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
    
    def __getattr__(self, name):
        """转发其他属性访问到mask_decoder"""
        # 避免递归调用 - 排除所有内部属性和已定义的属性
        if name.startswith('_') or name == 'mask_decoder':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 从mask_decoder获取属性
        if 'mask_decoder' in self._modules:
            try:
                return getattr(self._modules['mask_decoder'], name)
            except AttributeError:
                pass
        
        # 如果mask_decoder没有，尝试从peft_model获取（用于访问LoRA参数等）
        if self._peft_model is not None:
            try:
                return getattr(self._peft_model, name)
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def apply_lora_to_sam(model, lora_config: Dict[str, Any]):
    """
    对SAM模型应用LoRA
    
    Args:
        model: SAM模型
        lora_config: LoRA配置
    """
    if not PEFT_AVAILABLE:
        print("PEFT不可用，使用全量微调")
        return model
    
    # 选择要应用LoRA的模块
    # SAM的mask_decoder中的transformer包含q_proj, v_proj, k_proj, out_proj
    target_modules = lora_config.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'out_proj'])
    
    # 创建LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=target_modules,
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        bias="none",
    )
    
    # 应用LoRA到mask_decoder
    if lora_config.get('apply_to_mask_decoder', True):
        print("对mask_decoder应用LoRA...")
        # 使用PEFT包装mask_decoder
        peft_mask_decoder = get_peft_model(model.mask_decoder, peft_config)
        # 使用包装器包装，以正确处理forward调用
        model.mask_decoder = MaskDecoderWrapper(peft_mask_decoder)
        print("LoRA应用成功，已添加包装器以正确处理forward调用")
    
    # 可选：对image_encoder应用LoRA（需要更多显存）
    if lora_config.get('apply_to_image_encoder', False):
        print("对image_encoder应用LoRA...")
        model.image_encoder = get_peft_model(model.image_encoder, peft_config)
    
    return model


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, loss_weights, swanlab_run=None):
    """训练一个epoch - 使用SAM的forward方法（参考Sam_LoRA）"""
    model.train()
    total_loss = 0.0
    loss_dict = {'dice': 0.0, 'focal': 0.0, 'ce': 0.0, 'iou': 0.0}
    
    # 获取实际模型（处理DataParallel情况）
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # 使用monai的DiceCELoss（与Sam_LoRA一致）
    # 注意: squared_pred=False 避免梯度变小，提高训练效果
    try:
        import monai
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=False, reduction='mean')
        use_monai_loss = True
    except ImportError:
        use_monai_loss = False
        print("Warning: monai not available, using custom loss")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # 准备batched_input（SAM forward需要的格式）
        # batch是一个list of dict（因为使用了collate_fn）
        batched_input = []
        for item in batch:
            # ✅ 获取transformed_size（变换后的图像尺寸，在pad之前）
            transformed_size = item.get('transformed_size', None)
            if transformed_size is not None:
                if isinstance(transformed_size, (list, np.ndarray)):
                    transformed_size = tuple(transformed_size)
                elif isinstance(transformed_size, torch.Tensor):
                    transformed_size = tuple(transformed_size.tolist())
            
            # ✅ 使用 points 而不是 boxes
            point_coords = item['points'].to(device).unsqueeze(0)  # [1, N, 2] - 添加batch维度
            point_labels = item['point_labels'].to(device).unsqueeze(0)  # [1, N] - 添加batch维度
            
            batched_input.append({
                'image': item['image'].to(device),
                'original_size': tuple(item['original_size'].numpy()) if isinstance(item['original_size'], torch.Tensor) else item['original_size'],
                'transformed_size': transformed_size,  # ✅ 添加transformed_size字段
                'point_coords': point_coords,
                'point_labels': point_labels,
            })
        
        # 使用混合精度训练
        with autocast():
            # 使用SAM的forward_train方法（允许梯度计算，用于训练）
            outputs = actual_model.forward_train(batched_input=batched_input, multimask_output=False)
            
            # 准备ground truth masks和预测的logits
            # batch是一个list of dict
            stk_gt = torch.stack([item['ground_truth_mask'].to(device) for item in batch], dim=0)  # [B, H_orig, W_orig]
            stk_out = torch.stack([out['low_res_logits'] for out in outputs], dim=0)  # [B, 1, 256, 256]
            
            # 调整stk_out维度：[B, 1, 256, 256] -> [B, 256, 256]
            stk_out = stk_out.squeeze(1)  # [B, 256, 256]
            
            # ✅ 正确调整stk_gt到256x256（匹配原图的处理流程）
            # 原图流程：576x1024 -> pad到1024x1024 -> image_encoder -> mask_decoder输出256x256
            # postprocess_masks流程：256x256 -> 上采样到1024x1024 -> 裁剪到576x1024 -> 上采样到720x1280
            # 所以low_res_logits的256x256在空间上对应的是1024x1024的有效区域（去除padding后是576x1024）
            # GT mask应该：原始尺寸 -> 变换后尺寸(576x1024) -> pad到1024x1024 -> resize到256x256
            stk_gt_resized = []
            for i, (gt_mask, item) in enumerate(zip(stk_gt, batch)):
                # 获取变换后的图像尺寸
                transformed_size = item.get('transformed_size', None)
                if transformed_size is None:
                    # 如果没有transformed_size，从image tensor获取
                    transformed_size = tuple(item['image'].shape[-2:])  # (H, W)
                else:
                    # 确保transformed_size是tuple格式
                    if isinstance(transformed_size, (list, np.ndarray)):
                        transformed_size = tuple(transformed_size)
                    elif isinstance(transformed_size, torch.Tensor):
                        transformed_size = tuple(transformed_size.tolist())
                
                # preprocessed_size 固定为 1024x1024（SAM的输入尺寸）
                preprocessed_size = (1024, 1024)
                
                # 第一步：将GT mask从原始尺寸resize到变换后的图像尺寸（576x1024）
                gt_transformed = nn.functional.interpolate(
                    gt_mask.unsqueeze(0).unsqueeze(0), 
                    size=transformed_size,  # (H_transformed, W_transformed)
                    mode='nearest'
                ).squeeze(0).squeeze(0)  # [H_transformed, W_transformed]
                
                # 第二步：pad到preprocessed_size（1024x1024），匹配preprocess的pad操作
                h, w = gt_transformed.shape
                padh = preprocessed_size[0] - h
                padw = preprocessed_size[1] - w
                gt_padded = torch.nn.functional.pad(
                    gt_transformed.unsqueeze(0),  # [1, H, W]
                    (0, padw, 0, padh), 
                    mode='constant', 
                    value=0
                ).squeeze(0)  # [H_padded, W_padded] = [1024, 1024]
                
                # 第三步：将pad后的GT mask resize到256x256（low_res_logits的尺寸）
                gt_resized = nn.functional.interpolate(
                    gt_padded.unsqueeze(0).unsqueeze(0), 
                    size=(256, 256), 
                    mode='nearest'
                ).squeeze(0).squeeze(0)  # [256, 256]
                
                stk_gt_resized.append(gt_resized)
            stk_gt_resized = torch.stack(stk_gt_resized, dim=0)  # [B, 256, 256]
            # 参考Sam_LoRA: stk_out保持[B, 256, 256]，stk_gt变为[B, 1, 256, 256]
            stk_gt_resized = stk_gt_resized.unsqueeze(1)  # [B, 1, 256, 256]
            # stk_out已经是[B, 256, 256]，不需要再unsqueeze
            
            # 计算损失（在low_res_logits上，参考Sam_LoRA）
            if use_monai_loss:
                # DiceCELoss输入: stk_out [B, 256, 256], stk_gt [B, 1, 256, 256]
                # 注意：DiceCELoss 已经包含了 Dice Loss + CE Loss，所以不需要再单独加 BCE
                loss = seg_loss(stk_out, stk_gt_resized.float())
                
                # ✅ 计算额外的 IoU 损失（增强位置约束，不重复计算CE）
                # IoU 损失（直接约束位置重合度）
                pred_binary = (torch.sigmoid(stk_out) > 0.5).float()
                gt_binary = stk_gt_resized.squeeze(1).float()
                intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
                union = (pred_binary + gt_binary).sum(dim=(1, 2)) - intersection
                iou = (intersection + 1e-7) / (union + 1e-7)
                iou_loss = (1.0 - iou.mean()) * loss_weights.get('iou', 0.5)
                
                # ✅ 总损失 = DiceCELoss (已包含Dice+CE) + IoU Loss
                loss = loss + iou_loss
                
                losses = {
                    'total': loss,
                    'dice': loss - iou_loss,  # DiceCELoss 部分（不包含IoU）
                    'focal': torch.tensor(0.0, device=device),
                    'ce': torch.tensor(0.0, device=device),  # DiceCELoss 内部已包含，不单独计算
                    'iou': iou_loss
                }
            else:
                # 使用自定义损失
                losses = compute_loss([stk_out], stk_gt_resized, None, loss_weights)
                loss = losses['total']
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 累计损失
        total_loss += loss.item()
        for k in loss_dict:
            # 处理可能是Tensor或float的情况
            if isinstance(losses[k], torch.Tensor):
                loss_dict[k] += losses[k].item()
            else:
                loss_dict[k] += float(losses[k])
        
        # 更新进度条
        dice_val = losses['dice'].item() if isinstance(losses['dice'], torch.Tensor) else losses['dice']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{dice_val:.4f}"
        })
        
        # 记录到SwanLab（每10个batch记录一次，避免记录过于频繁）
        if swanlab_run is not None and batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            dice_val = losses['dice'].item() if isinstance(losses['dice'], torch.Tensor) else losses['dice']
            swanlab_run.log({
                'train/batch_loss': loss.item(),
                'train/batch_dice_loss': dice_val,
            }, step=global_step)
    
    avg_loss = total_loss / len(dataloader)
    for k in loss_dict:
        loss_dict[k] /= len(dataloader)
    
    return avg_loss, loss_dict


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


def validate(model, dataloader, device, loss_weights, visualize_dir=None, epoch=None):
    """验证 - 使用SAM的forward方法（参考Sam_LoRA）"""
    model.eval()
    total_loss = 0.0
    loss_dict = {'dice': 0.0, 'focal': 0.0, 'ce': 0.0, 'iou': 0.0}
    
    # 获取实际模型（处理DataParallel情况）
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # 使用monai的DiceCELoss（与Sam_LoRA一致）
    # 注意: squared_pred=False 避免梯度变小，提高训练效果
    try:
        import monai
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=False, reduction='mean')
        use_monai_loss = True
    except ImportError:
        use_monai_loss = False
    
    # 用于可视化的标志
    visualize_done = False
    
    # ✅ 预先选择要可视化的batch索引（从所有batch中随机选择，而不是总是选择第一个）
    if visualize_dir is not None and epoch is not None:
        # 使用epoch作为随机种子的一部分，确保每个epoch选择不同的batch
        random.seed(42 + epoch)
        total_batches = len(dataloader)
        visualize_batch_idx = random.randint(0, total_batches - 1) if total_batches > 0 else 0
        random.seed()  # 重置随机种子，避免影响其他随机操作
    else:
        visualize_batch_idx = -1  # 不进行可视化
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            # 准备batched_input（SAM forward需要的格式）
            batched_input = []
            for i in range(len(batch)):
                # ✅ 获取transformed_size（变换后的图像尺寸，在pad之前）
                transformed_size = batch[i].get('transformed_size', None)
                if transformed_size is not None:
                    if isinstance(transformed_size, (list, np.ndarray)):
                        transformed_size = tuple(transformed_size)
                    elif isinstance(transformed_size, torch.Tensor):
                        transformed_size = tuple(transformed_size.tolist())
                
                # ✅ 使用 points 而不是 boxes
                point_coords = batch[i]['points'].to(device).unsqueeze(0)  # [1, N, 2] - 添加batch维度
                point_labels = batch[i]['point_labels'].to(device).unsqueeze(0)  # [1, N] - 添加batch维度
                
                batched_input.append({
                    'image': batch[i]['image'].to(device),
                    'original_size': tuple(batch[i]['original_size'].numpy()) if isinstance(batch[i]['original_size'], torch.Tensor) else batch[i]['original_size'],
                    'transformed_size': transformed_size,  # ✅ 添加transformed_size字段
                    'point_coords': point_coords,
                    'point_labels': point_labels,
                })
            
            with autocast():
                # 验证时使用forward_train（允许梯度计算，虽然验证时不需要，但保持代码一致性）
                # 注意：验证函数已经在torch.no_grad()上下文中，所以不会实际计算梯度
                outputs = actual_model.forward_train(batched_input=batched_input, multimask_output=False)
                
                # 准备ground truth masks和预测的logits
                # batch是一个list of dict
                stk_gt = torch.stack([item['ground_truth_mask'].to(device) for item in batch], dim=0)  # [B, H_orig, W_orig]
                stk_out = torch.stack([out['low_res_logits'] for out in outputs], dim=0)  # [B, 1, 256, 256]
                
                # 调整stk_out维度：[B, 1, 256, 256] -> [B, 256, 256]
                stk_out = stk_out.squeeze(1)  # [B, 256, 256]
                
                # ✅ 正确调整stk_gt到256x256（匹配原图的处理流程）
                # 原图流程：576x1024 -> pad到1024x1024 -> image_encoder -> mask_decoder输出256x256
                # postprocess_masks流程：256x256 -> 上采样到1024x1024 -> 裁剪到576x1024 -> 上采样到720x1280
                # 所以low_res_logits的256x256在空间上对应的是1024x1024的有效区域（去除padding后是576x1024）
                # GT mask应该：原始尺寸 -> 变换后尺寸(576x1024) -> pad到1024x1024 -> resize到256x256
                stk_gt_resized = []
                for i, (gt_mask, item) in enumerate(zip(stk_gt, batch)):
                    # 获取变换后的图像尺寸
                    transformed_size = item.get('transformed_size', None)
                    if transformed_size is None:
                        # 如果没有transformed_size，从image tensor获取
                        transformed_size = tuple(item['image'].shape[-2:])  # (H, W)
                    else:
                        # 确保transformed_size是tuple格式
                        if isinstance(transformed_size, (list, np.ndarray)):
                            transformed_size = tuple(transformed_size)
                        elif isinstance(transformed_size, torch.Tensor):
                            transformed_size = tuple(transformed_size.tolist())
                    
                    # preprocessed_size 固定为 1024x1024（SAM的输入尺寸）
                    preprocessed_size = (1024, 1024)
                    
                    # 第一步：将GT mask从原始尺寸resize到变换后的图像尺寸（576x1024）
                    gt_transformed = nn.functional.interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0), 
                        size=transformed_size,  # (H_transformed, W_transformed)
                        mode='nearest'
                    ).squeeze(0).squeeze(0)  # [H_transformed, W_transformed]
                    
                    # 第二步：pad到preprocessed_size（1024x1024），匹配preprocess的pad操作
                    h, w = gt_transformed.shape
                    padh = preprocessed_size[0] - h
                    padw = preprocessed_size[1] - w
                    gt_padded = torch.nn.functional.pad(
                        gt_transformed.unsqueeze(0),  # [1, H, W]
                        (0, padw, 0, padh), 
                        mode='constant', 
                        value=0
                    ).squeeze(0)  # [H_padded, W_padded] = [1024, 1024]
                    
                    # 第三步：将pad后的GT mask resize到256x256（low_res_logits的尺寸）
                    gt_resized = nn.functional.interpolate(
                        gt_padded.unsqueeze(0).unsqueeze(0), 
                        size=(256, 256), 
                        mode='nearest'
                    ).squeeze(0).squeeze(0)  # [256, 256]
                    
                    stk_gt_resized.append(gt_resized)
                stk_gt_resized = torch.stack(stk_gt_resized, dim=0)  # [B, 256, 256]
                # 参考Sam_LoRA: stk_out保持[B, 256, 256]，stk_gt变为[B, 1, 256, 256]
                stk_gt_resized = stk_gt_resized.unsqueeze(1)  # [B, 1, 256, 256]
                # stk_out已经是[B, 256, 256]，不需要再unsqueeze
                
                # 计算损失（在low_res_logits上，参考Sam_LoRA）
                if use_monai_loss:
                    # DiceCELoss输入: stk_out [B, 256, 256], stk_gt [B, 1, 256, 256]
                    # 注意：DiceCELoss 已经包含了 Dice Loss + CE Loss，所以不需要再单独加 BCE
                    loss = seg_loss(stk_out, stk_gt_resized.float())
                    
                    # ✅ 计算额外的 IoU 损失（增强位置约束，不重复计算CE）
                    # IoU 损失（直接约束位置重合度）
                    pred_binary = (torch.sigmoid(stk_out) > 0.5).float()
                    gt_binary = stk_gt_resized.squeeze(1).float()
                    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
                    union = (pred_binary + gt_binary).sum(dim=(1, 2)) - intersection
                    iou = (intersection + 1e-7) / (union + 1e-7)
                    iou_loss = (1.0 - iou.mean()) * loss_weights.get('iou', 0.5)
                    
                    # ✅ 总损失 = DiceCELoss (已包含Dice+CE) + IoU Loss
                    loss = loss + iou_loss
                    
                    losses = {
                        'total': loss,
                        'dice': loss - iou_loss,  # DiceCELoss 部分（不包含IoU）
                        'focal': torch.tensor(0.0, device=device),
                        'ce': torch.tensor(0.0, device=device),  # DiceCELoss 内部已包含，不单独计算
                        'iou': iou_loss
                    }
                else:
                    # 使用自定义损失
                    losses = compute_loss([stk_out], stk_gt_resized, None, loss_weights)
                    loss = losses['total']
                
                # 可视化：每个epoch随机选择一个batch中的一张图像（合并同一张图片的所有mask）
                if visualize_dir is not None and epoch is not None and not visualize_done and batch_idx == visualize_batch_idx:
                    try:
                        # 随机选择一个样本（而不是固定选择第一个）
                        random_idx = random.randint(0, len(batch) - 1)
                        selected_item = batch[random_idx]
                        original_image = selected_item.get('original_image')
                        image_name = selected_item.get('image_name', 'unknown')
                        
                        if original_image is not None:
                            # 收集同一张图片的所有mask（同一张图片可能有多个标注/物体）
                            masks_list = []
                            
                            # 遍历batch中所有属于同一张图片的样本
                            for i, item in enumerate(batch):
                                if item.get('image_name') == image_name:
                                    # 获取对应的预测mask（全分辨率）
                                    pred_mask = outputs[i]['masks']  # [B, C, H, W] 或 [C, H, W]
                                    
                                    # 处理mask维度
                                    if pred_mask.dim() == 4:
                                        pred_mask = pred_mask[0, 0]  # 取第一个batch，第一个mask
                                    elif pred_mask.dim() == 3:
                                        pred_mask = pred_mask[0]  # 取第一个mask
                                    
                                    # 转换为numpy
                                    pred_mask_np = pred_mask.cpu().numpy().astype(bool)
                                    
                                    # 确保图像和mask尺寸一致
                                    if original_image.shape[:2] != pred_mask_np.shape[:2]:
                                        # 调整mask尺寸到原始图像尺寸
                                        pred_mask_np = cv2.resize(
                                            pred_mask_np.astype(np.uint8),
                                            (original_image.shape[1], original_image.shape[0]),
                                            interpolation=cv2.INTER_NEAREST
                                        ).astype(bool)
                                    
                                    masks_list.append(pred_mask_np)
                            
                            # 如果找到了mask，生成可视化
                            if len(masks_list) > 0:
                                overlay = generate_colored_mask_overlay(original_image, masks_list, alpha=0.5)
                                
                                # 保存可视化结果
                                os.makedirs(visualize_dir, exist_ok=True)
                                vis_path = os.path.join(visualize_dir, f"epoch_{epoch:03d}_{image_name}.png")
                                
                                # 转换为BGR格式保存
                                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(vis_path, overlay_bgr)
                                
                                print(f"\n  验证集可视化已保存: {vis_path} (包含 {len(masks_list)} 个mask)")
                                visualize_done = True
                            else:
                                print(f"\n  警告: 未找到图像 {image_name} 的mask")
                    except Exception as e:
                        print(f"\n  验证集可视化失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            total_loss += loss.item()
            for k in loss_dict:
                # 处理可能是Tensor或float的情况
                if isinstance(losses[k], torch.Tensor):
                    loss_dict[k] += losses[k].item()
                else:
                    loss_dict[k] += float(losses[k])
    
    avg_loss = total_loss / len(dataloader)
    for k in loss_dict:
        loss_dict[k] /= len(dataloader)
    
    return avg_loss, loss_dict


def main():
    parser = argparse.ArgumentParser(description="SAM模型LoRA微调")
    parser.add_argument("--sam_checkpoint", type=str, required=True,
                       help="SAM模型checkpoint路径")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="数据集目录（包含annotations.json和masks目录）")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="图像目录路径")
    parser.add_argument("--output_dir", type=str, default="./output/sam_finetuned",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备类型")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="权重衰减")
    parser.add_argument("--use_lora", action="store_true",
                       help="使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,out_proj",
                       help="LoRA目标模块（逗号分隔）")
    parser.add_argument("--apply_to_image_encoder", action="store_true",
                       help="对image_encoder也应用LoRA（需要更多显存）")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="验证集比例")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--save_every", type=int, default=5,
                       help="每N个epoch保存一次")
    parser.add_argument("--swanlab_api_key", type=str, default=None,
                       help="SwanLab API key")
    parser.add_argument("--swanlab_project", type=str, default="SAM-Finetune",
                       help="SwanLab项目名称")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None,
                       help="SwanLab实验名称")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检测可用GPU数量（在SwanLab初始化之前，因为需要用到）
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    # 初始化SwanLab
    swanlab_run = None
    if SWANLAB_AVAILABLE:
        try:
            # 设置API key（如果提供）
            if args.swanlab_api_key:
                os.environ['SWANLAB_API_KEY'] = args.swanlab_api_key
            
            # 生成实验名称
            experiment_name = args.swanlab_experiment_name
            if experiment_name is None:
                from datetime import datetime
                experiment_name = f"SAM-LoRA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # 初始化SwanLab
            swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=experiment_name,
                config={
                    'sam_checkpoint': args.sam_checkpoint,
                    'dataset_dir': args.dataset_dir,
                    'images_dir': args.images_dir,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'use_lora': args.use_lora,
                    'lora_r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'lora_dropout': args.lora_dropout,
                    'lora_target_modules': args.lora_target_modules,
                    'val_split': args.val_split,
                    'num_gpus': num_gpus,
                }
            )
            print(f"SwanLab初始化成功，实验名称: {experiment_name}")
        except Exception as e:
            print(f"SwanLab初始化失败: {e}，将继续训练但不记录指标")
            swanlab_run = None
    else:
        print("SwanLab不可用，训练指标将不会记录")
    
    # 如果指定了device且是cuda，检查是否使用多GPU
    use_multi_gpu = False
    if args.device.startswith('cuda') and num_gpus > 1:
        use_multi_gpu = True
        print(f"使用多GPU训练: {num_gpus} 个GPU")
        # 使用第一个GPU作为主设备
        device = torch.device('cuda:0')
    else:
        device = torch.device(args.device)
    
    # 加载SAM模型
    print("加载SAM模型...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    
    # 设置哪些部分需要训练
    # 通常只训练mask_decoder，image_encoder保持冻结
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True
    
    # 应用LoRA（如果启用）
    if args.use_lora and PEFT_AVAILABLE:
        print("应用LoRA微调...")
        lora_config = {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'target_modules': args.lora_target_modules.split(','),
            'apply_to_mask_decoder': True,
            'apply_to_image_encoder': args.apply_to_image_encoder
        }
        sam = apply_lora_to_sam(sam, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in sam.parameters())
        print(f"可训练参数: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    else:
        print("使用全量微调（仅mask_decoder）")
        trainable_params = sum(p.numel() for p in sam.mask_decoder.parameters())
        total_params = sum(p.numel() for p in sam.parameters())
        print(f"可训练参数: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    # 合并三个数据集（robot_arm_01, robot_arm_02, robot_arm_03）
    # 每个数据集都有：图像目录、mask目录，可能还有annotations.json
    datasets_config = []
    # args.images_dir应该是picture目录，直接使用它而不是os.path.dirname
    base_images_dir = args.images_dir
    base_masks_dir = "/opt/data/private/LLMSeg/dataset/GT_mask"
    
    for dataset_name in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
        images_dir = os.path.join(base_images_dir, dataset_name)
        masks_dir = os.path.join(base_masks_dir, dataset_name, "masks")
        # annotations.json在每个数据集的mask目录下
        annotations_file = os.path.join(base_masks_dir, dataset_name, "annotations.json")
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            config = {
                'name': dataset_name,
                'images_dir': images_dir,
                'masks_dir': masks_dir,
                'annotations_file': annotations_file if os.path.exists(annotations_file) else None
            }
            datasets_config.append(config)
            print(f"找到数据集 {dataset_name}: 图像={images_dir}, mask={masks_dir}, 标注文件={'存在' if config['annotations_file'] else '不存在（将从mask文件名生成）'}")
    
    # 合并所有数据集的标注
    all_annotations = []
    for config in datasets_config:
        if config['annotations_file']:
            # 如果有标注文件，直接加载
            with open(config['annotations_file'], 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                # 更新gt_path以指向正确的mask目录
                for ann in annotations:
                    # gt_path可能是相对路径（如 robot_arm_01\masks\1_seat_mask.png）或文件名
                    gt_path = ann.get('gt_path', '')
                    # 统一处理路径分隔符
                    gt_path = gt_path.replace('\\', '/')
                    # 提取mask文件名
                    mask_filename = os.path.basename(gt_path)
                    # 使用绝对路径
                    ann['gt_path'] = os.path.join(config['masks_dir'], mask_filename).replace('\\', '/')
                    ann['_dataset_name'] = config['name']
                all_annotations.extend(annotations)
        else:
            # 如果没有标注文件，从mask文件名生成
            mask_files = [f for f in os.listdir(config['masks_dir']) if f.endswith(('.png', '.jpg'))]
            for mask_file in mask_files:
                # mask文件名格式: {image_number}_{object_name}_mask.png
                # 例如: 1_core_mask.png -> img_name=1.png, object=core
                parts = mask_file.replace('_mask.png', '').replace('_mask.jpg', '').split('_')
                if len(parts) >= 2:
                    img_number = parts[0]
                    object_name = '_'.join(parts[1:])
                    # 尝试找到对应的图像文件
                    img_name = f"{img_number}.png"
                    img_path = os.path.join(config['images_dir'], img_name)
                    if not os.path.exists(img_path):
                        img_name = f"{img_number}.jpg"
                        img_path = os.path.join(config['images_dir'], img_name)
                    
                    if os.path.exists(img_path):
                        # 加载图像获取尺寸
                        import cv2
                        img = cv2.imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            ann = {
                                'img_name': img_name,
                                'bbox': [0, 0, 0, 0],  # 将从mask计算
                                'object': object_name,
                                'action': 'grasp',  # 默认值
                                'gt_path': os.path.join(config['masks_dir'], mask_file).replace('\\', '/'),
                                'points': [],
                                'affordance_points': [],
                                'height': h,
                                'width': w,
                                '_dataset_name': config['name']
                            }
                            all_annotations.append(ann)
    
    print(f"\n合并后的总标注数: {len(all_annotations)}")
    
    # 确保输出目录存在（用于保存临时标注文件）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建临时标注文件用于数据集加载
    temp_annotations_file = os.path.join(args.output_dir, "merged_annotations.json")
    with open(temp_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    print(f"临时合并标注文件: {temp_annotations_file}")
    
    # 创建数据集（使用合并后的标注和所有图像目录）
    all_image_dirs = [config['images_dir'] for config in datasets_config]
    main_images_dir = all_image_dirs[0] if all_image_dirs else args.images_dir
    additional_image_dirs = all_image_dirs[1:] if len(all_image_dirs) > 1 else []
    
    # 使用第一个数据集的mask目录作为主目录（实际会从标注中的gt_path读取）
    main_masks_dir = datasets_config[0]['masks_dir'] if datasets_config else os.path.join(args.dataset_dir, "masks")

    
    
    print("创建合并数据集...")
    full_dataset = SAMDataset(
        annotations_file=temp_annotations_file,
        images_dir=main_images_dir,
        masks_dir=main_masks_dir,
        additional_image_dirs=additional_image_dirs,
        dataset_configs=datasets_config  # ✅ 传入数据集配置，用于根据数据集名称查找图像
    )
    
    # 划分训练集和验证集（按图像划分，避免数据泄漏）
    # ✅ 由于一张图像可能对应多个标注，需要按图像名称和数据集名称划分
    # 确保一张图像只从它自己归属的数据集中找mask
    import random
    from collections import defaultdict
    
    # ✅ 按图像名称和数据集名称分组标注
    annotations_by_image = defaultdict(list)
    for idx, ann in enumerate(full_dataset.valid_annotations):
        img_name = ann['img_name']
        dataset_name = ann.get('_dataset_name', 'unknown')
        # ✅ 使用 (img_name, dataset_name) 作为分组键
        group_key = (img_name, dataset_name)
        annotations_by_image[group_key].append(idx)
    
    # 获取所有唯一图像（包含数据集信息）
    unique_images = list(annotations_by_image.keys())
    random.seed(42)
    random.shuffle(unique_images)
    
    # 按图像划分训练集和验证集
    num_val_images = int(len(unique_images) * args.val_split)
    val_images = set(unique_images[:num_val_images])
    train_images = set(unique_images[num_val_images:])
    
    # 根据图像分组，确定训练集和验证集的标注索引
    train_indices = []
    val_indices = []
    for group_key, indices in annotations_by_image.items():
        if group_key in val_images:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"总图像数: {len(unique_images)}")
    print(f"训练集图像: {len(train_images)}, 标注数: {len(train_indices)}")
    print(f"验证集图像: {len(val_images)}, 标注数: {len(val_indices)}")
    
    # 如果使用多GPU，调整batch_size
    effective_batch_size = args.batch_size
    if use_multi_gpu:
        # 使用DataParallel包装模型
        print("使用DataParallel进行多GPU训练...")
        sam = nn.DataParallel(sam)
        # 每个GPU的batch_size保持不变，总batch_size = batch_size * num_gpus
        effective_batch_size = args.batch_size * num_gpus
        print(f"总batch_size: {effective_batch_size} (每个GPU: {args.batch_size})")
    
    # 创建collate_fn（参考Sam_LoRA，返回list而不是stacked tensor）
    def collate_fn(batch):
        """返回list of dict，而不是stacked tensor"""
        return list(batch)
    
    # ✅ 创建按图像分组的BatchSampler（确保一个batch包含一张图像的所有标注）
    def create_image_grouped_batches(dataset, batch_size, shuffle=True):
        """
        创建按图像分组的batch列表
        确保一个batch包含一张图像的所有标注
        如果一张图像的标注数量超过batch_size，会分割成多个batch
        
        注意：返回的索引是相对于dataset的索引（如果是Subset，则是Subset内的索引）
        """
        # 获取数据集的实际数据（如果是Subset，需要访问底层数据集）
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices  # Subset的索引映射
            # 创建从原始索引到Subset索引的映射
            idx_map = {orig_idx: sub_idx for sub_idx, orig_idx in enumerate(subset_indices)}
        else:
            base_dataset = dataset
            subset_indices = list(range(len(dataset)))
            idx_map = {idx: idx for idx in range(len(dataset))}
        
        # ✅ 按图像名称和数据集名称分组标注索引（使用原始索引）
        # 确保一张图像只从它自己归属的数据集中找mask
        annotations_by_image = defaultdict(list)
        for orig_idx in subset_indices:
            # 获取对应的标注（使用原始索引）
            ann = base_dataset.valid_annotations[orig_idx]
            img_name = ann['img_name']
            dataset_name = ann.get('_dataset_name', 'unknown')
            # ✅ 使用 (img_name, dataset_name) 作为分组键
            group_key = (img_name, dataset_name)
            annotations_by_image[group_key].append(orig_idx)
        
        # # ✅ 调试：打印前几张图像对应的mask信息
        # print("\n=== 调试：检查图像分组情况 ===")
        # sample_count = 0
        # for (img_name, dataset_name), orig_indices in list(annotations_by_image.items())[:10]:
        #     gt_paths = []
        #     for orig_idx in orig_indices:
        #         ann = base_dataset.valid_annotations[orig_idx]
        #         gt_paths.append(ann.get('gt_path', 'unknown'))
        #     print(f"图像: {img_name} [{dataset_name}], 标注数量: {len(orig_indices)}, gt_paths:")
        #     for i, gt_path in enumerate(gt_paths, 1):
        #         print(f"  {i}. {gt_path}")
        #     sample_count += 1
        # print(f"（仅显示前 {min(sample_count, 10)} 张图像）\n")
        
        # 创建batch列表：每个batch包含一张图像的所有标注
        # 但需要转换为Subset内的索引
        batches = []
        image_names = list(annotations_by_image.keys())
        
        if shuffle:
            random.shuffle(image_names)
        
        for (img_name, dataset_name) in image_names:
            orig_indices_for_image = annotations_by_image[(img_name, dataset_name)]
            # 转换为Subset内的索引
            subset_indices_for_image = [idx_map[orig_idx] for orig_idx in orig_indices_for_image]
            
            # ✅ 如果一张图像的标注数量超过batch_size，分割成多个batch
            if len(subset_indices_for_image) <= batch_size:
                # 一个batch包含一张图像的所有标注
                batches.append(subset_indices_for_image)
            else:
                # 如果超过batch_size，分割成多个batch
                for i in range(0, len(subset_indices_for_image), batch_size):
                    batches.append(subset_indices_for_image[i:i+batch_size])
        
        return batches
    
    # 创建训练集的batch列表（传入batch_size参数）
    train_batches = create_image_grouped_batches(train_dataset, args.batch_size, shuffle=True)
    # 创建验证集的batch列表（传入batch_size参数）
    val_batches = create_image_grouped_batches(val_dataset, args.batch_size, shuffle=False)
    
    print(f"\n训练集batch数: {len(train_batches)} (每个batch包含一张图像的所有标注，如果超过batch_size={args.batch_size}则分割)")
    print(f"验证集batch数: {len(val_batches)} (每个batch包含一张图像的所有标注，如果超过batch_size={args.batch_size}则分割)")
    # 打印一些统计信息
    train_batch_sizes = [len(b) for b in train_batches]
    val_batch_sizes = [len(b) for b in val_batches]
    print(f"训练集batch大小范围: min={min(train_batch_sizes)}, max={max(train_batch_sizes)}, avg={np.mean(train_batch_sizes):.1f}")
    print(f"验证集batch大小范围: min={min(val_batch_sizes)}, max={max(val_batch_sizes)}, avg={np.mean(val_batch_sizes):.1f}")
    
    # ✅ 创建自定义的BatchSampler，直接返回预定义的batches
    # BatchSampler需要继承torch.utils.data.Sampler，但返回的是batch的索引列表
    class ImageGroupedBatchSampler:
        """按图像分组的BatchSampler，确保一个batch包含一张图像的所有标注"""
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            for batch_indices in self.batches:
                yield batch_indices
        
        def __len__(self):
            return len(self.batches)
    
    train_batch_sampler = ImageGroupedBatchSampler(train_batches)
    val_batch_sampler = ImageGroupedBatchSampler(val_batches)
    
    # 创建数据加载器（使用自定义batch_sampler）
    # 注意：使用batch_sampler时，不能同时指定batch_size和sampler
    # 但是DataLoader的batch_sampler参数需要的是torch.utils.data.Sampler类型
    # 我们需要直接传递batches，所以使用一个包装类
    class BatchSamplerWrapper:
        """包装BatchSampler以兼容DataLoader"""
        def __init__(self, batch_sampler):
            self.batch_sampler = batch_sampler
        
        def __iter__(self):
            return iter(self.batch_sampler)
        
        def __len__(self):
            return len(self.batch_sampler)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BatchSamplerWrapper(train_batch_sampler),
        num_workers=0,  # 暂时设为0，避免多进程问题
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=BatchSamplerWrapper(val_batch_sampler),
        num_workers=0,  # 暂时设为0，避免多进程问题
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 损失权重（增加边界损失和BCE损失以改善边界学习，启用IoU损失以增强位置约束）
    loss_weights = {
        'dice': 1.0,
        'focal': 1.0,
        'ce': 0.5,  # ✅ 启用BCE损失，对位置更敏感
        'iou': 0.5  # ✅ 启用IoU损失，直接约束位置重合度
    }
    
    # 训练循环
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_loss_dict = train_epoch(
            sam, train_loader, optimizer, scaler, device, epoch, loss_weights, swanlab_run
        )
        
        # 验证（带可视化）
        visualize_dir = os.path.join(args.output_dir, "val_vis")
        val_loss, val_loss_dict = validate(
            sam, val_loader, device, loss_weights, visualize_dir=visualize_dir, epoch=epoch
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f"\n训练损失: {train_loss:.4f} (dice: {train_loss_dict['dice']:.4f}, "
              f"ce: {train_loss_dict['ce']:.4f}, iou: {train_loss_dict['iou']:.4f})")
        print(f"验证损失: {val_loss:.4f} (dice: {val_loss_dict['dice']:.4f}, "
              f"ce: {val_loss_dict['ce']:.4f}, iou: {val_loss_dict['iou']:.4f})")
        print(f"学习率: {current_lr:.6f}")
        
        # 记录到SwanLab
        if swanlab_run is not None:
            swanlab_run.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/dice_loss': train_loss_dict['dice'],
                # 'train/focal_loss': train_loss_dict['focal'],
                'train/ce_loss': train_loss_dict['ce'],
                'train/iou_loss': train_loss_dict['iou'],
                'val/loss': val_loss,
                'val/dice_loss': val_loss_dict['dice'],
                # 'val/focal_loss': val_loss_dict['focal'],
                'val/ce_loss': val_loss_dict['ce'],
                'val/iou_loss': val_loss_dict['iou'],
                'learning_rate': current_lr,
            }, step=epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 确保输出目录存在
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            # 如果使用DataParallel，需要获取module
            model_to_save = sam.module if use_multi_gpu else sam
            
            # 获取模型状态字典
            model_state_dict = model_to_save.state_dict()
            
            # 如果使用LoRA，需要额外保存LoRA权重
            lora_state_dict = None
            if args.use_lora and PEFT_AVAILABLE:
                # 从mask_decoder中提取LoRA权重
                actual_model = model_to_save
                if hasattr(actual_model.mask_decoder, '_peft_model'):
                    peft_model = actual_model.mask_decoder._peft_model
                    # PEFT的get_peft_state_dict只返回LoRA适配器权重
                    try:
                        from peft import get_peft_state_dict
                        lora_state_dict = get_peft_state_dict(peft_model)
                        print(f"提取LoRA权重: {len(lora_state_dict)} 个LoRA参数")
                    except:
                        # 如果get_peft_state_dict不可用，尝试直接获取
                        if hasattr(peft_model, 'get_peft_state_dict'):
                            lora_state_dict = peft_model.get_peft_state_dict()
                        else:
                            # 从state_dict中筛选LoRA权重
                            peft_state = peft_model.state_dict()
                            lora_state_dict = {k: v for k, v in peft_state.items() if 'lora' in k.lower()}
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            
            # 如果使用LoRA，添加LoRA权重
            if lora_state_dict is not None:
                checkpoint['lora_state_dict'] = lora_state_dict
                print(f"保存LoRA权重: {len(lora_state_dict)} 个参数")
            
            torch.save(checkpoint, checkpoint_path)
            print(f"保存最佳模型到: {checkpoint_path}")
        
        # 定期保存
        if epoch % args.save_every == 0:
            # 确保输出目录存在
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            model_to_save = sam.module if use_multi_gpu else sam
            
            # 获取模型状态字典
            model_state_dict = model_to_save.state_dict()
            
            # 如果使用LoRA，需要额外保存LoRA权重
            lora_state_dict = None
            if args.use_lora and PEFT_AVAILABLE:
                # 从mask_decoder中提取LoRA权重
                actual_model = model_to_save
                if hasattr(actual_model.mask_decoder, '_peft_model'):
                    peft_model = actual_model.mask_decoder._peft_model
                    try:
                        from peft import get_peft_state_dict
                        lora_state_dict = get_peft_state_dict(peft_model)
                    except:
                        if hasattr(peft_model, 'get_peft_state_dict'):
                            lora_state_dict = peft_model.get_peft_state_dict()
                        else:
                            peft_state = peft_model.state_dict()
                            lora_state_dict = {k: v for k, v in peft_state.items() if 'lora' in k.lower()}
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            
            # 如果使用LoRA，添加LoRA权重
            if lora_state_dict is not None:
                checkpoint['lora_state_dict'] = lora_state_dict
            
            torch.save(checkpoint, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {args.output_dir}")
    
    # 结束SwanLab记录
    if swanlab_run is not None:
        swanlab_run.finish()
        print("SwanLab记录已结束")


if __name__ == "__main__":
    main()

