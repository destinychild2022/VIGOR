#!/usr/bin/env python3
"""调试IoU计算，检查mask格式转换是否正确"""

import cv2
import numpy as np
import os
from pathlib import Path

# 测试一个具体的GT mask和候选mask
view_name = "robot_arm_03"
img_name = "1.png"
gt_mask_filename = "seat_1.png"  # 假设这是GT mask文件名

# 路径
gt_masks_dir = "/opt/data/private/LLMSeg/dataset/GT_mask"
vis_output_dir = "/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/test_vis"

# 加载GT mask
gt_mask_path = os.path.join(gt_masks_dir, view_name, "masks", gt_mask_filename)
print(f"GT mask路径: {gt_mask_path}")

if os.path.exists(gt_mask_path):
    gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"GT mask原始格式:")
    print(f"  shape: {gt_mask_raw.shape}")
    print(f"  dtype: {gt_mask_raw.dtype}")
    print(f"  min: {gt_mask_raw.min()}, max: {gt_mask_raw.max()}")
    print(f"  唯一值: {np.unique(gt_mask_raw)}")
    print(f"  值为0的像素数: {(gt_mask_raw == 0).sum()}")
    print(f"  值为255的像素数: {(gt_mask_raw == 255).sum()}")
    
    # 转换格式
    gt_mask_binary = (gt_mask_raw > 0).astype(np.uint8)
    print(f"\nGT mask转换后:")
    print(f"  dtype: {gt_mask_binary.dtype}")
    print(f"  唯一值: {np.unique(gt_mask_binary)}")
    print(f"  值为0的像素数（掩码区域）: {(gt_mask_binary == 0).sum()}")
    print(f"  值为1的像素数（背景）: {(gt_mask_binary == 1).sum()}")
else:
    print(f"GT mask不存在: {gt_mask_path}")
    # 尝试找第一个存在的GT mask
    gt_mask_dir = os.path.join(gt_masks_dir, view_name, "masks")
    if os.path.exists(gt_mask_dir):
        gt_files = [f for f in os.listdir(gt_mask_dir) if f.endswith('.png')]
        if len(gt_files) > 0:
            gt_mask_filename = gt_files[0]
            gt_mask_path = os.path.join(gt_mask_dir, gt_mask_filename)
            print(f"\n使用第一个找到的GT mask: {gt_mask_filename}")
            gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"GT mask原始格式:")
            print(f"  shape: {gt_mask_raw.shape}")
            print(f"  dtype: {gt_mask_raw.dtype}")
            print(f"  min: {gt_mask_raw.min()}, max: {gt_mask_raw.max()}")
            print(f"  唯一值: {np.unique(gt_mask_raw)}")
            print(f"  值为0的像素数: {(gt_mask_raw == 0).sum()}")
            print(f"  值为255的像素数: {(gt_mask_raw == 255).sum()}")
            
            gt_mask_binary = (gt_mask_raw > 0).astype(np.uint8)
            print(f"\nGT mask转换后:")
            print(f"  dtype: {gt_mask_binary.dtype}")
            print(f"  唯一值: {np.unique(gt_mask_binary)}")
            print(f"  值为0的像素数（掩码区域）: {(gt_mask_binary == 0).sum()}")
            print(f"  值为1的像素数（背景）: {(gt_mask_binary == 1).sum()}")

# 加载候选mask
image_stem = Path(img_name).stem
masks_dir = os.path.join(vis_output_dir, view_name, image_stem, "masks")
print(f"\n候选masks目录: {masks_dir}")

if os.path.exists(masks_dir):
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.startswith('mask_') and f.endswith('.png')])
    print(f"找到 {len(mask_files)} 个候选masks")
    
    if len(mask_files) > 0:
        # 加载第一个候选mask
        mask_path = os.path.join(masks_dir, mask_files[0])
        print(f"\n加载第一个候选mask: {mask_files[0]}")
        candidate_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"候选mask原始格式:")
        print(f"  shape: {candidate_mask_raw.shape}")
        print(f"  dtype: {candidate_mask_raw.dtype}")
        print(f"  min: {candidate_mask_raw.min()}, max: {candidate_mask_raw.max()}")
        print(f"  唯一值: {np.unique(candidate_mask_raw)}")
        print(f"  值为0的像素数: {(candidate_mask_raw == 0).sum()}")
        print(f"  值为255的像素数: {(candidate_mask_raw == 255).sum()}")
        
        # 转换格式（当前代码的逻辑）
        candidate_mask_binary = (candidate_mask_raw == 0).astype(np.uint8)
        print(f"\n候选mask转换后（当前逻辑）:")
        print(f"  dtype: {candidate_mask_binary.dtype}")
        print(f"  唯一值: {np.unique(candidate_mask_binary)}")
        print(f"  值为0的像素数（掩码区域）: {(candidate_mask_binary == 0).sum()}")
        print(f"  值为1的像素数（背景）: {(candidate_mask_binary == 1).sum()}")
        
        # 计算IoU
        if 'gt_mask_binary' in locals():
            # 调整尺寸
            if gt_mask_binary.shape != candidate_mask_binary.shape:
                print(f"\n调整GT mask尺寸: {gt_mask_binary.shape} -> {candidate_mask_binary.shape}")
                gt_resized = cv2.resize(gt_mask_binary.astype(np.uint8), 
                                       (candidate_mask_binary.shape[1], candidate_mask_binary.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                gt_resized = (gt_resized > 0).astype(np.uint8)
            else:
                gt_resized = gt_mask_binary
            
            # 计算IoU
            mask1_foreground = (gt_resized == 0)
            mask2_foreground = (candidate_mask_binary == 0)
            
            intersection = np.logical_and(mask1_foreground, mask2_foreground).sum()
            union = np.logical_or(mask1_foreground, mask2_foreground).sum()
            
            print(f"\nIoU计算:")
            print(f"  GT掩码区域像素数: {mask1_foreground.sum()}")
            print(f"  候选掩码区域像素数: {mask2_foreground.sum()}")
            print(f"  交集: {intersection}")
            print(f"  并集: {union}")
            if union > 0:
                iou = intersection / union
                print(f"  IoU: {iou:.4f}")
            else:
                print(f"  并集为0，IoU: 1.0（如果交集也为0）或0.0")
else:
    print(f"候选masks目录不存在: {masks_dir}")

