#!/usr/bin/env python3
"""
将指定序号的mask叠加图片按照九宫格形式合并
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def crop_to_square(image, target_size=None):
    """将图片裁剪为正方形（从长边中间裁剪）
    
    Args:
        image: 输入图片 (H, W, 3)
        target_size: 目标尺寸（如果为None，使用较短的边作为尺寸）
    
    Returns:
        裁剪后的正方形图片
    """
    h, w = image.shape[:2]
    
    if target_size is None:
        target_size = min(h, w)  # 使用较短的边作为正方形边长
    
    # 确定裁剪区域
    if w > h:
        # 横向图片：从宽度中间裁剪出与高度相等的宽度
        crop_size = h
        x_start = (w - crop_size) // 2
        x_end = x_start + crop_size
        cropped = image[:, x_start:x_end]
    elif h > w:
        # 纵向图片：从高度中间裁剪出与宽度相等的高度
        crop_size = w
        y_start = (h - crop_size) // 2
        y_end = y_start + crop_size
        cropped = image[y_start:y_end, :]
    else:
        # 已经是正方形
        cropped = image
    
    # 如果裁剪后的尺寸不等于目标尺寸，进行缩放
    if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
        cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return cropped


def merge_images_grid(image_paths, output_path, grid_size=(3, 3), square_size=None):
    """将多张图片按照网格形式合并
    
    Args:
        image_paths: 图片路径列表（应该包含9张图片）
        output_path: 输出路径
        grid_size: 网格大小 (rows, cols)，默认(3, 3)
        square_size: 每张图片的正方形尺寸（如果为None，自动计算）
    """
    if len(image_paths) != grid_size[0] * grid_size[1]:
        raise ValueError(f"需要 {grid_size[0] * grid_size[1]} 张图片，但提供了 {len(image_paths)} 张")
    
    images = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在: {img_path}")
            # 创建黑色占位图
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片: {img_path}")
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    # 确定每张图片的正方形尺寸
    if square_size is None:
        # 使用所有图片中较短的边的最小值作为基准（确保所有图片都能裁剪）
        min_size = min(min(img.shape[:2]) for img in images)
        square_size = min_size
    
    # 将所有图片裁剪为相同大小的正方形
    square_images = []
    for img in images:
        square_img = crop_to_square(img, target_size=square_size)
        square_images.append(square_img)
    
    # 创建网格
    rows, cols = grid_size
    grid_image = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)
    
    # 将图片放入网格
    for idx, square_img in enumerate(square_images):
        row = idx // cols
        col = idx % cols
        y_start = row * square_size
        y_end = y_start + square_size
        x_start = col * square_size
        x_end = x_start + square_size
        grid_image[y_start:y_end, x_start:x_end] = square_img
    
    # 保存结果
    grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, grid_image_bgr)
    print(f"✅ 已保存合并图片: {output_path}")
    print(f"   网格大小: {grid_size[0]}x{grid_size[1]}")
    print(f"   每张图片尺寸: {square_size}x{square_size}")
    print(f"   总尺寸: {grid_image.shape[1]}x{grid_image.shape[0]}")


def main():
    parser = argparse.ArgumentParser(description="将指定序号的mask叠加图片按照九宫格形式合并")
    parser.add_argument("--input_dir", type=str,
                       default="/opt/data/private/LLMSeg/vis_output/robot_arm_02/75",
                       help="输入图片目录")
    parser.add_argument("--indices", type=int, nargs="+",
                       default=[5, 9, 10, 11, 15, 19, 20, 23, 30],
                       help="要合并的图片序号列表（9个）")
    parser.add_argument("--output_path", type=str,
                       default="/opt/data/private/LLMSeg/vis_output/robot_arm_02/75/grid_merged.png",
                       help="输出图片路径")
    parser.add_argument("--square_size", type=int, default=None,
                       help="每张图片的正方形尺寸（如果为None，自动计算）")
    
    args = parser.parse_args()
    
    if len(args.indices) != 9:
        print(f"错误: 需要9张图片，但提供了 {len(args.indices)} 个序号")
        return
    
    # 构建图片路径列表
    image_paths = []
    for idx in args.indices:
        mask_filename = f"mask_{idx:04d}_overlay.png"
        mask_path = os.path.join(args.input_dir, mask_filename)
        image_paths.append(mask_path)
        print(f"  {idx:2d}: {mask_filename}")
    
    print(f"\n合并 {len(image_paths)} 张图片...")
    
    # 合并图片
    merge_images_grid(
        image_paths=image_paths,
        output_path=args.output_path,
        grid_size=(3, 3),
        square_size=args.square_size
    )


if __name__ == "__main__":
    main()

