#!/usr/bin/env python3
"""
可视化 GT mask 脚本
将指定图片的 GT mask 以不同颜色叠加在原图上
"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path


def load_gt_mask(mask_path):
    """
    加载 GT mask
    注意：mask中黑色（值为0）和红色点是掩码区域，白色（值为255）是背景
    """
    # 读取为彩色图像（以便识别红色点）
    mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_color is None:
        raise ValueError(f"无法读取mask文件: {mask_path}")
    
    # 转换为RGB格式
    mask_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    
    # 创建二值mask：黑色区域（值为0）或红色点（R值高，G和B值低）
    h, w = mask_rgb.shape[:2]
    mask_binary = np.zeros((h, w), dtype=np.uint8)
    
    # 识别黑色区域（RGB值都接近0）
    black_region = np.all(mask_rgb < 50, axis=2)
    
    # 识别红色点（R值高，G和B值低）
    # 红色点：R > 200, G < 100, B < 100
    red_region = (mask_rgb[:, :, 0] > 200) & (mask_rgb[:, :, 1] < 100) & (mask_rgb[:, :, 2] < 100)
    
    # 合并黑色区域和红色点作为mask区域
    mask_region = black_region | red_region
    
    # mask区域设为0（黑色），背景设为255（白色）
    mask_binary[mask_region] = 0
    mask_binary[~mask_region] = 255
    
    return mask_binary


def visualize_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    将mask以指定颜色叠加在原图上
    
    Args:
        image: 原图 (BGR格式)
        mask: GT mask (灰度图，0=mask区域，255=背景)
        color: 叠加颜色 (BGR格式)
        alpha: 透明度 (0-1)
    
    Returns:
        叠加后的图像
    """
    # 转换为RGB格式（便于显示）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # mask中黑色（值为0）是掩码区域，白色（值为255）是背景
    # 转换为布尔掩码：True表示mask区域
    mask_region = (mask == 0)
    
    # 创建叠加图像
    overlay = image_rgb.copy()
    
    # 在mask区域叠加颜色
    overlay[mask_region] = (
        image_rgb[mask_region] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay


def visualize_multiple_masks(image, masks_list, colors=None, alpha=0.5):
    """
    将多个mask用不同颜色叠加在同一张原图上
    
    Args:
        image: 原图 (BGR格式)
        masks_list: GT mask列表 (灰度图列表，每个mask中0=mask区域，255=背景)
        colors: 颜色列表，每个颜色对应一个mask
        alpha: 透明度
    
    Returns:
        叠加后的图像 (RGB格式)
    """
    if colors is None:
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 红色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 黄色
        ]
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    
    # 确保颜色数量足够
    if len(colors) < len(masks_list):
        # 如果颜色不够，重复使用
        colors = colors * ((len(masks_list) // len(colors)) + 1)
    
    # 逐个叠加mask
    for i, mask in enumerate(masks_list):
        # 调整mask大小以匹配图像
        h, w = image_rgb.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # mask中黑色（值为0）是掩码区域
        mask_region = (mask == 0)
        color = colors[i % len(colors)]
        
        # 在mask区域叠加颜色
        overlay[mask_region] = (
            overlay[mask_region] * (1 - alpha) + 
            np.array(color) * alpha
        ).astype(np.uint8)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="可视化 GT mask")
    parser.add_argument("--view", default="robot_arm_02", type=str, help="视角名称")
    parser.add_argument("--image_id", default="69", type=str, help="图片ID")
    parser.add_argument("--dataset_dir", default="/opt/data/private/LLMSeg/dataset", type=str)
    parser.add_argument("--output_dir", default="./vis_output", type=str, help="输出目录")
    parser.add_argument("--alpha", default=0.5, type=float, help="透明度 (0-1)")
    parser.add_argument("--colors", nargs="+", default=None, help="颜色列表 (RGB格式，如: 0 255 0 表示绿色)")
    
    args = parser.parse_args()
    
    # 路径设置
    raw_pic_dir = os.path.join(args.dataset_dir, "raw_pic", args.view)
    gt_mask_dir = os.path.join(args.dataset_dir, "GT_mask", args.view)
    annotations_path = os.path.join(gt_mask_dir, "annotations.json")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取annotations.json
    if not os.path.exists(annotations_path):
        print(f"错误: 找不到annotations.json文件: {annotations_path}")
        return
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # 查找指定图片的标注
    image_items = [item for item in annotations if args.image_id in item.get('img_name', '')]
    
    if len(image_items) == 0:
        print(f"错误: 找不到图片ID为 {args.image_id} 的标注")
        return
    
    print(f"找到 {len(image_items)} 个标注项")
    
    # 获取图片名称（所有标注项应该对应同一张图片）
    if len(image_items) == 0:
        print(f"错误: 找不到图片ID为 {args.image_id} 的标注")
        return
    
    img_name = image_items[0]['img_name']
    image_path = os.path.join(raw_pic_dir, img_name)
    
    if not os.path.exists(image_path):
        print(f"错误: 找不到原图: {image_path}")
        return
    
    print(f"\n处理图片: {img_name}")
    print(f"  原图: {image_path}")
    print(f"  找到 {len(image_items)} 个GT mask")
    
    # 读取原图
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取原图: {image_path}")
        return
    
    # 读取所有GT mask
    masks_list = []
    mask_names = []
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 红色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋红色
        (0, 255, 255),  # 黄色
    ]
    color_names = ["绿色", "红色", "蓝色", "青色", "洋红色", "黄色"]
    
    for idx, item in enumerate(image_items):
        gt_path = item['gt_path'].replace('\\', os.sep)
        gt_mask_path = os.path.join(args.dataset_dir, "GT_mask", gt_path)
        
        if not os.path.exists(gt_mask_path):
            print(f"警告: 找不到GT mask: {gt_mask_path}")
            continue
        
        mask = load_gt_mask(gt_mask_path)
        masks_list.append(mask)
        
        # 提取mask名称（从路径中）
        mask_name = os.path.basename(gt_mask_path).replace('.png', '').replace(f'{args.image_id}_', '')
        mask_names.append(mask_name)
        print(f"  Mask {idx+1}: {mask_name} ({color_names[idx % len(color_names)]})")
    
    if len(masks_list) == 0:
        print("错误: 没有找到有效的GT mask")
        return
    
    # 将所有mask用不同颜色叠加在一张图上
    overlay = visualize_multiple_masks(image, masks_list, colors=colors[:len(masks_list)], alpha=args.alpha)
    
    # 创建对比图：左侧原图，右侧叠加图
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # 添加文字说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # 计算图例高度（需要和原图标题高度一致）
    legend_height = 50 + len(masks_list) * 30
    total_height = h + legend_height
    
    # 在叠加图上添加图例
    overlay_with_legend = np.ones((total_height, w, 3), dtype=np.uint8) * 255
    overlay_with_legend[:h, :] = overlay
    
    # 添加标题
    cv2.putText(overlay_with_legend, f"GT Masks Overlay ({len(masks_list)} masks)", 
                (10, h + 30), font, font_scale, (0, 0, 0), thickness)
    
    # 添加图例
    for idx, (mask_name, color) in enumerate(zip(mask_names, colors[:len(masks_list)])):
        y_pos = h + 60 + idx * 30
        # 颜色块
        cv2.rectangle(overlay_with_legend, (10, y_pos - 15), (30, y_pos + 5), color, -1)
        # 文字说明
        text = f"{mask_name} ({color_names[idx]})"
        cv2.putText(overlay_with_legend, text, (35, y_pos), font, 0.6, (0, 0, 0), 1)
    
    # 在原图上添加标题（高度要和叠加图一致）
    image_with_title = np.ones((total_height, w, 3), dtype=np.uint8) * 255
    image_with_title[:h, :] = image_rgb
    cv2.putText(image_with_title, "Original Image", 
                (10, h + 35), font, font_scale, (0, 0, 0), thickness)
    
    # 拼接：左侧原图，右侧叠加图（带图例）
    combined = np.hstack([image_with_title, overlay_with_legend])
    
    # 保存结果
    output_path = os.path.join(
        args.output_dir,
        f"{args.view}_{args.image_id}_all_masks_overlay.png"
    )
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_bgr)
    print(f"\n✅ 保存结果: {output_path}")
    
    print(f"\n✅ 可视化完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

