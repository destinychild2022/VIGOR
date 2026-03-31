#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import zmq

# 确保能搜到 model/
VIGOR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, VIGOR_ROOT)

def init_sam(checkpoint_path, device="cuda"):
    from model.segment_anything import sam_model_registry
    from model.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

    print(f"  -> 正在加载 SAM (ViT-H) 到 {device}...")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
        box_nms_thresh=0.7,
        points_per_side=32,
    )
    print("  ✅ SAM 初始化完成")
    return mask_generator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='5557')
    parser.add_argument('--checkpoint', type=str, default='/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # 初始化模型
    mask_generator = init_sam(args.checkpoint, args.device)

    # 启动 ZMQ 服务
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"\n[SAM Service] 就绪, 监听端口: {args.port}")

    try:
        while True:
            msg = socket.recv_pyobj()
            rgb = msg['rgb'] # (H,W,3) uint8
            print(f"  -> 收到图像: {rgb.shape}")

            with torch.no_grad():
                masks_raw = mask_generator.generate(rgb)
            
            # 转换为 VIGOR 约定的 0/1 binary 掩码
            masks_binary = []
            for m in masks_raw:
                seg = m['segmentation'] # bool
                binary = (~seg).astype(np.uint8) # 0=前景, 1=背景
                masks_binary.append(binary)
            
            print(f"  -> 生成了 {len(masks_binary)} 个候选掩码")
            socket.send_pyobj({'masks': masks_binary})
            
    except KeyboardInterrupt:
        print("\n-> 停止服务")
    finally:
        socket.close()

if __name__ == "__main__":
    main()
