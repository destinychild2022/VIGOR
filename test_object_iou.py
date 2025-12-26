#!/usr/bin/env python3
"""
按object名称测试IoU脚本
对于每个object名称（如"seat"），查找所有对应的GT mask，与候选mask计算IoU
"""

import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import colorsys
import re


def load_gt_mask(gt_mask_path: str) -> np.ndarray:
    """加载GT mask
    返回格式：0=掩码区域（前景），1=背景
    """
    if not os.path.exists(gt_mask_path):
        return None
    
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return None
    
    # GT mask格式：黑色（0）是掩码区域，白色（255）是背景
    # 转换为统一格式：0=掩码区域，1=背景
    gt_mask_binary = (gt_mask > 0).astype(np.uint8)
    
    return gt_mask_binary


def load_candidate_masks_from_dir(masks_dir: str) -> list:
    """从已保存的masks目录加载候选masks
    
    Args:
        masks_dir: masks目录路径（包含mask_0000.png, mask_0001.png等）
    
    Returns:
        候选masks列表，格式：0=掩码区域，1=背景
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
        
        # 保存的mask格式：白色背景（255），黑色mask（0）
        # 转换为统一格式：0=掩码区域，1=背景
        mask_binary = (mask > 0).astype(np.uint8)
        candidate_masks.append(mask_binary)
    
    return candidate_masks


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask的IoU
    
    Args:
        mask1, mask2: 格式为0=掩码区域，1=背景
    
    Returns:
        IoU值（0-1之间）
    """
    # 调整尺寸以匹配
    if mask1.shape != mask2.shape:
        h, w = mask2.shape
        mask1 = cv2.resize(mask1.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask1 = (mask1 > 0).astype(np.uint8)  # 保持格式
    
    # 计算交集和并集
    # 掩码区域 = 0，背景 = 1
    # 交集：两个mask都是掩码区域（都是0）
    intersection = np.logical_and(mask1 == 0, mask2 == 0).sum()
    # 并集：至少一个是掩码区域（至少一个是0）
    union = np.logical_or(mask1 == 0, mask2 == 0).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)


def compute_iou_batch(gt_mask: np.ndarray, candidate_masks: list) -> tuple:
    """计算GT mask与所有候选masks的IoU，返回最大IoU和最佳mask索引
    
    Args:
        gt_mask: GT mask，格式：0=掩码区域，1=背景
        candidate_masks: 候选masks列表，格式：0=掩码区域，1=背景
    
    Returns:
        (max_iou, best_mask_idx)
    """
    if len(candidate_masks) == 0:
        return 0.0, -1
    
    max_iou = 0.0
    best_mask_idx = -1
    
    # 调整GT mask尺寸以匹配候选mask（使用第一个候选mask的尺寸作为参考）
    h, w = candidate_masks[0].shape
    if gt_mask.shape != (h, w):
        gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        gt_mask_resized = (gt_mask_resized > 0).astype(np.uint8)
    else:
        gt_mask_resized = gt_mask
    
    # 计算每个候选mask的IoU
    for i, candidate_mask in enumerate(candidate_masks):
        iou = compute_iou(gt_mask_resized, candidate_mask)
        if iou > max_iou:
            max_iou = iou
            best_mask_idx = i
    
    return max_iou, best_mask_idx


def extract_base_object_name(object_name: str) -> str:
    """提取object的基础名称（去掉数字后缀，合并相关变体）
    
    合并规则：
    1. 去掉末尾数字：seat1 -> seat, adjusting screw2 -> adjusting screw
    2. 合并adjusting相关：adjusting screw, adjusting screw1, ad -> adjusting
    3. a和ad都归为adjusting（如果ad存在）
    
    例如：
    - "seat" -> "seat"
    - "seat1" -> "seat"
    - "seat2" -> "seat"
    - "adjusting screw" -> "adjusting"
    - "adjusting screw1" -> "adjusting"
    - "ad" -> "adjusting"
    - "a" -> "adjusting" (如果ad存在，否则保持为"a")
    """
    original = object_name.strip()
    base_name = original.lower()
    
    # 移除末尾的数字（如 seat1 -> seat, adjusting screw2 -> adjusting screw）
    base_name = re.sub(r'\s*\d+$', '', base_name)
    
    # 特殊处理：adjusting相关的都归为"adjusting"
    # 包括：adjusting screw, adjusting screw1, ad, a（如果ad存在）
    if base_name.startswith('adjusting') or base_name == 'ad':
        return 'adjusting'
    
    # 如果名称是"a"，检查是否有"ad"存在，如果有则归为"adjusting"
    # 注意：这个需要在收集完所有名称后才能确定，所以先返回"a"
    # 后续在收集完所有名称后统一处理
    if base_name == 'a':
        return 'a'  # 暂时返回"a"，后续统一处理
    
    return base_name


def generate_colored_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """将mask以指定颜色叠加在原图上
    
    Args:
        image: 原图 (RGB格式)
        mask: mask (0=掩码区域，1=背景)
        color: 叠加颜色 (RGB格式)
        alpha: 透明度 (0-1)
    
    Returns:
        叠加后的图像 (RGB格式)
    """
    overlay = image.copy()
    mask_region = (mask == 0)  # 掩码区域
    
    # 在mask区域叠加颜色
    overlay[mask_region] = (
        image[mask_region] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay


def test_object_iou(
    annotations_dir: str,
    gt_masks_dir: str,
    candidate_mask_dir: str,
    images_dir: str,
    output_file: str = None,
    vis_output_dir: str = None,
    view_names: list = None,
):
    """按object名称测试IoU
    
    Args:
        annotations_dir: annotations.json所在目录
        gt_masks_dir: GT mask目录
        candidate_mask_dir: 候选mask目录（test_vis_sam_origin）
        output_file: 结果输出文件路径
        view_names: 视角名称列表
    """
    if view_names is None:
        view_names = ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']
    
    # 收集所有object名称和对应的GT mask信息
    # 使用基础名称作为key（seat, seat1, seat2都归为seat）
    object_to_masks = defaultdict(list)  # {base_object_name: [(view_name, img_name, gt_path, original_object_name), ...]}
    base_to_original = defaultdict(set)  # {base_object_name: {original_name1, original_name2, ...}}
    
    print("=" * 80)
    print("收集GT mask信息...")
    print("=" * 80)
    
    for view_name in view_names:
        annotations_path = os.path.join(annotations_dir, view_name, 'annotations.json')
        if not os.path.exists(annotations_path):
            print(f"警告: 找不到annotations.json: {annotations_path}")
            continue
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        for ann in annotations:
            original_object_name = ann.get('object', 'unknown')
            img_name = ann.get('img_name', '')
            gt_path = ann.get('gt_path', '')
            
            # 过滤掉ring和pin（包括ring1, ring2等）
            object_name_lower = original_object_name.lower()
            if 'ring' in object_name_lower or 'pin' in object_name_lower:
                continue
            
            # 提取基础名称（去掉数字后缀）
            base_object_name = extract_base_object_name(original_object_name)
            
            # 记录基础名称到原始名称的映射
            base_to_original[base_object_name].add(original_object_name)
            
            # 使用基础名称作为key
            object_to_masks[base_object_name].append({
                'view_name': view_name,
                'img_name': img_name,
                'gt_path': gt_path,
                'object_name': original_object_name,  # 保留原始名称
                'base_object_name': base_object_name,  # 基础名称
            })
    
    # 后处理：合并"a"和"adjusting"
    # 如果同时存在"a"和"adjusting"（或"ad"），将"a"合并到"adjusting"
    if 'a' in base_to_original and 'adjusting' in base_to_original:
        # 将"a"的所有masks合并到"adjusting"
        if 'a' in object_to_masks:
            object_to_masks['adjusting'].extend(object_to_masks['a'])
            base_to_original['adjusting'].update(base_to_original['a'])
            del object_to_masks['a']
            del base_to_original['a']
    elif 'a' in base_to_original and 'ad' in base_to_original:
        # 如果只有"a"和"ad"，都归为"adjusting"
        if 'a' in object_to_masks:
            if 'adjusting' not in object_to_masks:
                object_to_masks['adjusting'] = []
                base_to_original['adjusting'] = set()
            object_to_masks['adjusting'].extend(object_to_masks['a'])
            base_to_original['adjusting'].update(base_to_original['a'])
            del object_to_masks['a']
            del base_to_original['a']
        if 'ad' in object_to_masks:
            if 'adjusting' not in object_to_masks:
                object_to_masks['adjusting'] = []
                base_to_original['adjusting'] = set()
            object_to_masks['adjusting'].extend(object_to_masks['ad'])
            base_to_original['adjusting'].update(base_to_original['ad'])
            del object_to_masks['ad']
            del base_to_original['ad']
    elif 'ad' in base_to_original:
        # 如果只有"ad"，重命名为"adjusting"
        if 'ad' in object_to_masks:
            object_to_masks['adjusting'] = object_to_masks['ad']
            base_to_original['adjusting'] = base_to_original['ad']
            del object_to_masks['ad']
            del base_to_original['ad']
    
    # 获取所有基础object名称并排序
    all_base_objects = sorted(object_to_masks.keys())
    print(f"\n找到 {len(all_base_objects)} 个不同的object类别（基础名称）:")
    for base_obj in all_base_objects:
        original_names = sorted(base_to_original[base_obj])
        original_names_str = ', '.join(original_names) if len(original_names) <= 3 else ', '.join(original_names[:3]) + f' ... (共{len(original_names)}个)'
        print(f"  - {base_obj}: {len(object_to_masks[base_obj])} 个GT mask (包含: {original_names_str})")
    
    print("\n" + "=" * 80)
    print("开始计算IoU...")
    print("=" * 80)
    
    # 为每个基础object名称计算IoU
    results = {}
    
    # 创建可视化输出目录
    if vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)
    
    for base_object_name in tqdm(all_base_objects, desc="处理object"):
        masks_info = object_to_masks[base_object_name]
        ious = []
        details = []
        
        # 为每个基础object名称创建可视化目录
        object_vis_dir = None
        if vis_output_dir:
            # 清理基础object名称，用于目录名（移除特殊字符）
            safe_object_name = base_object_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            object_vis_dir = os.path.join(vis_output_dir, safe_object_name)
            os.makedirs(object_vis_dir, exist_ok=True)
        
        for mask_info in tqdm(masks_info, desc=f"  {base_object_name}", leave=False):
            view_name = mask_info['view_name']
            img_name = mask_info['img_name']
            gt_path = mask_info['gt_path']
            
            # 获取GT mask路径
            gt_mask_filename = os.path.basename(gt_path.replace('\\', '/'))
            gt_mask_path = os.path.join(gt_masks_dir, view_name, "masks", gt_mask_filename)
            
            # 加载GT mask
            gt_mask = load_gt_mask(gt_mask_path)
            if gt_mask is None:
                continue
            
            # 获取候选mask目录
            image_stem = Path(img_name).stem
            candidate_masks_dir = os.path.join(candidate_mask_dir, view_name, image_stem, "masks")
            
            # 加载候选masks
            candidate_masks = load_candidate_masks_from_dir(candidate_masks_dir)
            if len(candidate_masks) == 0:
                continue
            
            # 计算最大IoU
            max_iou, best_mask_idx = compute_iou_batch(gt_mask, candidate_masks)
            
            ious.append(max_iou)
            details.append({
                'view_name': view_name,
                'img_name': img_name,
                'original_object_name': mask_info['object_name'],  # 原始名称
                'max_iou': max_iou,
                'best_mask_idx': best_mask_idx,
            })
            
            # 保存可视化结果
            if object_vis_dir and best_mask_idx >= 0:
                try:
                    # 读取原图
                    image_path = os.path.join(images_dir, view_name, img_name)
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            # 调整GT mask和最佳候选mask到原图尺寸
                            h, w = image_rgb.shape[:2]
                            gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            gt_mask_resized = (gt_mask_resized > 0).astype(np.uint8)
                            
                            best_mask = candidate_masks[best_mask_idx]
                            best_mask_resized = cv2.resize(best_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            best_mask_resized = (best_mask_resized > 0).astype(np.uint8)
                            
                            # 生成GT mask叠加图（绿色）
                            gt_overlay = generate_colored_mask_overlay(image_rgb, gt_mask_resized, color=(0, 255, 0), alpha=0.5)
                            
                            # 生成最佳候选mask叠加图（红色）
                            pred_overlay = generate_colored_mask_overlay(image_rgb, best_mask_resized, color=(255, 0, 0), alpha=0.5)
                            
                            # 创建对比图：左侧原图+GT，右侧原图+预测
                            h_img, w_img = image_rgb.shape[:2]
                            combined = np.zeros((h_img, w_img * 2, 3), dtype=np.uint8)
                            combined[:, :w_img] = gt_overlay
                            combined[:, w_img:] = pred_overlay
                            
                            # 添加文字标注
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.8
                            thickness = 2
                            cv2.putText(combined, f"GT (IoU: {max_iou:.3f})", 
                                       (10, 30), font, font_scale, (255, 255, 255), thickness)
                            cv2.putText(combined, f"Pred (mask_{best_mask_idx:04d})", 
                                       (w_img + 10, 30), font, font_scale, (255, 255, 255), thickness)
                            
                            # 保存可视化结果
                            vis_filename = f"{view_name}_{image_stem}_iou{max_iou:.3f}.png"
                            vis_path = os.path.join(object_vis_dir, vis_filename)
                            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(vis_path, combined_bgr)
                except Exception as e:
                    print(f"警告: 保存可视化失败 ({view_name}/{img_name}): {e}")
        
        if len(ious) > 0:
            results[base_object_name] = {
                'count': len(ious),
                'mean_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'min_iou': np.min(ious),
                'max_iou': np.max(ious),
                'original_names': sorted(base_to_original[base_object_name]),  # 包含的原始名称列表
                'details': details,
            }
    
    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("按Object名称的IoU测试结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"候选mask目录: {candidate_mask_dir}\n")
            f.write(f"测试的object类别数量: {len(results)}\n")
            f.write("\n说明: 带数字后缀的object名称（如seat1, seat2）已归为基础名称（seat）\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 按平均IoU排序
            sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_iou'], reverse=True)
            
            f.write("汇总统计（按平均IoU降序）:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Object基础名称':<25} {'数量':<8} {'平均IoU':<12} {'标准差':<12} {'最小IoU':<12} {'最大IoU':<12} {'包含的原始名称'}\n")
            f.write("-" * 80 + "\n")
            
            for base_object_name, stats in sorted_results:
                original_names_str = ', '.join(stats['original_names'])
                f.write(f"{base_object_name:<25} {stats['count']:<8} "
                       f"{stats['mean_iou']:<12.4f} {stats['std_iou']:<12.4f} "
                       f"{stats['min_iou']:<12.4f} {stats['max_iou']:<12.4f} {original_names_str}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 详细结果
            for base_object_name, stats in sorted_results:
                f.write(f"\nObject类别: {base_object_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"包含的原始名称: {', '.join(stats['original_names'])}\n")
                f.write(f"数量: {stats['count']}\n")
                f.write(f"平均IoU: {stats['mean_iou']:.4f}\n")
                f.write(f"标准差: {stats['std_iou']:.4f}\n")
                f.write(f"最小IoU: {stats['min_iou']:.4f}\n")
                f.write(f"最大IoU: {stats['max_iou']:.4f}\n")
                f.write("\n详细结果（按IoU降序）:\n")
                f.write(f"{'视角':<15} {'图像名':<20} {'原始Object名称':<25} {'IoU':<10} {'最佳mask索引':<15}\n")
                f.write("-" * 80 + "\n")
                
                details_sorted = sorted(stats['details'], key=lambda x: x['max_iou'], reverse=True)
                for detail in details_sorted:
                    f.write(f"{detail['view_name']:<15} {detail['img_name']:<20} "
                           f"{detail['original_object_name']:<25} "
                           f"{detail['max_iou']:<10.4f} {detail['best_mask_idx']:<15}\n")
        
        print(f"\n详细结果已保存到: {output_file}")
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("汇总统计（按平均IoU降序）:")
    print("=" * 80)
    print(f"{'Object基础名称':<25} {'数量':<8} {'平均IoU':<12} {'标准差':<12} {'最小IoU':<12} {'最大IoU':<12}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_iou'], reverse=True)
    for base_object_name, stats in sorted_results:
        print(f"{base_object_name:<25} {stats['count']:<8} "
              f"{stats['mean_iou']:<12.4f} {stats['std_iou']:<12.4f} "
              f"{stats['min_iou']:<12.4f} {stats['max_iou']:<12.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="按object名称测试IoU")
    parser.add_argument("--annotations_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/GT_mask",
                       help="annotations.json所在目录")
    parser.add_argument("--gt_masks_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/GT_mask",
                       help="GT mask目录")
    parser.add_argument("--images_dir", type=str,
                       default="/opt/data/private/LLMSeg/dataset/raw_pic",
                       help="原图目录")
    parser.add_argument("--candidate_mask_dir", type=str,
                       default="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/test_vis_sam_origin",
                       help="候选mask目录（test_vis_sam_origin）")
    parser.add_argument("--output_file", type=str,
                       default="./test_object_iou_results.txt",
                       help="结果输出文件路径")
    parser.add_argument("--vis_output_dir", type=str,
                       default=None,
                       help="可视化结果输出目录（按object名称分目录保存）")
    parser.add_argument("--view_names", type=str, nargs="+",
                       default=['robot_arm_01', 'robot_arm_02', 'robot_arm_03'],
                       help="视角名称列表")
    
    args = parser.parse_args()
    
    test_object_iou(
        annotations_dir=args.annotations_dir,
        gt_masks_dir=args.gt_masks_dir,
        candidate_mask_dir=args.candidate_mask_dir,
        images_dir=args.images_dir,
        output_file=args.output_file,
        vis_output_dir=args.vis_output_dir,
        view_names=args.view_names,
    )


if __name__ == "__main__":
    main()
