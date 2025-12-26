#!/usr/bin/env python3
"""
可视化候选mask脚本
将指定图片的候选mask（从test_vis_sam_origin）以红色叠加在原图上
每个mask单独生成一个叠加图片
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob


def load_candidate_mask(mask_path):
    """
    加载候选mask
    注意：mask中黑色（值为0）是掩码区域，白色（值为255）是背景
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取mask文件: {mask_path}")
    return mask


def visualize_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    将mask以指定颜色叠加在原图上
    
    Args:
        image: 原图 (BGR格式)
        mask: mask (灰度图，0=mask区域，255=背景)
        color: 叠加颜色 (RGB格式)
        alpha: 透明度 (0-1)
    
    Returns:
        叠加后的图像 (RGB格式)
    """
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # mask中黑色（值为0）是掩码区域，白色（值为255）是背景
    # 转换为布尔掩码：True表示mask区域
    mask_region = (mask == 0)
    
    # 调整mask大小以匹配图像
    h, w = image_rgb.shape[:2]
    if mask.shape != (h, w):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_region = (mask_resized == 0)
    
    # 创建叠加图像
    overlay = image_rgb.copy()
    
    # 在mask区域叠加颜色（红色）
    overlay[mask_region] = (
        image_rgb[mask_region] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="可视化候选mask")
    parser.add_argument("--view", default="robot_arm_02", type=str, help="视角名称")
    parser.add_argument("--image_id", default="75", type=str, help="图片ID（不含扩展名）")
    parser.add_argument("--raw_pic_dir", default="/opt/data/private/LLMSeg/dataset/raw_pic", type=str, help="原图目录")
    parser.add_argument("--candidate_mask_dir", 
                       default="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/test_vis_sam_origin",
                       type=str, help="候选mask目录")
    parser.add_argument("--output_dir", default="/opt/data/private/LLMSeg/vis_output", type=str, help="输出目录")
    parser.add_argument("--alpha", default=0.5, type=float, help="透明度 (0-1)")
    parser.add_argument("--color", nargs=3, type=int, default=[255, 0, 0], 
                       help="叠加颜色 (RGB格式，默认红色: 255 0 0)")
    
    args = parser.parse_args()
    
    # 路径设置
    view_name = args.view
    image_id = args.image_id
    
    # 原图路径（需要找到对应的图片文件名）
    raw_pic_view_dir = os.path.join(args.raw_pic_dir, view_name)
    
    # 候选mask目录
    candidate_mask_view_dir = os.path.join(args.candidate_mask_dir, view_name, image_id)
    masks_dir = os.path.join(candidate_mask_view_dir, "masks")
    
    # 输出目录
    output_view_dir = os.path.join(args.output_dir, view_name, image_id)
    os.makedirs(output_view_dir, exist_ok=True)
    
    # 检查目录是否存在
    if not os.path.exists(raw_pic_view_dir):
        print(f"错误: 找不到原图目录: {raw_pic_view_dir}")
        return
    
    if not os.path.exists(masks_dir):
        print(f"错误: 找不到候选mask目录: {masks_dir}")
        return
    
    # 查找对应的原图文件（可能是 69.png 或 69.jpg 等）
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_path = None
    for ext in image_extensions:
        potential_path = os.path.join(raw_pic_view_dir, f"{image_id}{ext}")
        if os.path.exists(potential_path):
            image_path = potential_path
            break
    
    if image_path is None:
        # 如果找不到精确匹配，尝试查找包含image_id的文件
        pattern = os.path.join(raw_pic_view_dir, f"{image_id}.*")
        matches = glob.glob(pattern)
        if matches:
            image_path = matches[0]
        else:
            print(f"错误: 找不到原图文件（尝试: {image_id}.png, {image_id}.jpg等）")
            print(f"  在目录: {raw_pic_view_dir}")
            return
    
    print(f"原图路径: {image_path}")
    print(f"候选mask目录: {masks_dir}")
    print(f"输出目录: {output_view_dir}")
    
    # 读取原图
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取原图: {image_path}")
        return
    
    # 查找所有候选mask文件
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.png")))
    
    if len(mask_files) == 0:
        print(f"错误: 在目录 {masks_dir} 中找不到候选mask文件")
        return
    
    print(f"\n找到 {len(mask_files)} 个候选mask")
    
    # 叠加颜色（RGB格式）
    overlay_color = tuple(args.color)
    
    # 处理每个mask
    for idx, mask_path in enumerate(mask_files):
        try:
            # 加载mask
            mask = load_candidate_mask(mask_path)
            
            # 叠加在原图上
            overlay = visualize_mask(image, mask, color=overlay_color, alpha=args.alpha)
            
            # 获取mask文件名（如 mask_0000.png）
            mask_filename = os.path.basename(mask_path)
            # 生成输出文件名（如 mask_0000_overlay.png）
            output_filename = mask_filename.replace('.png', '_overlay.png')
            output_path = os.path.join(output_view_dir, output_filename)
            
            # 保存结果（转换为BGR格式）
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, overlay_bgr)
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(mask_files)} 个mask...")
                
        except Exception as e:
            print(f"警告: 处理 {mask_path} 时出错: {e}")
            continue
    
    print(f"\n✅ 完成！共生成 {len(mask_files)} 个叠加图片")
    print(f"✅ 结果保存在: {output_view_dir}")


if __name__ == "__main__":
    main()
