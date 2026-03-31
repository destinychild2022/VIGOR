#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GraspNet 推理服务
=================
运行环境: graspnet conda 环境
          conda activate graspnet

职责:
  - 常驻监听, 接收 affordance mask + RGB + Depth + 内参
  - 在 affordance 区域上用 GraspNet 预测抓取位姿
  - 返回最佳抓取位姿 (相机坐标系)

用法:
  conda activate graspnet
  python graspnet_service.py \
      --checkpoint /opt/data/private/LLMSeg/graspnet-baseline/logs/checkpoint-rs.tar \
      --port 5556
"""

import os
import sys
import argparse
import numpy as np
import torch
import zmq
from PIL import Image, ImageDraw

# ============================================================================
#  路径设置
# ============================================================================
sys.path.append('..')
sys.path.append('../models')
sys.path.append('../utils')

from models.graspnet import GraspNet, pred_decode
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


# ============================================================================
#  GraspNet 推理
# ============================================================================
def init_graspnet(checkpoint_path, device="cuda"):
    """初始化 GraspNet 模型"""
    net = GraspNet(
        input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False
    )
    net.to(device)
    net.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    print(f"  ✅ GraspNet 加载完成: {checkpoint_path}")
    return net


def predict_grasp(net, rgb, depth, affordance_mask, intrinsic, device="cuda",
                  vis_dir=None):
    """
    在 affordance 区域上预测抓取位姿。

    Args:
        rgb:              (H, W, 3) uint8
        depth:            (H, W) float
        affordance_mask:  (H, W) uint8, 0=前景(affordance), 1=背景
        intrinsic:        (3, 3) float

    Returns:
        dict: {status, translation, rotation, width, score}
              translation/rotation 都在相机坐标系下
    """
    H, W = depth.shape

    # 1. 创建点云
    camera = CameraInfo(W, H, intrinsic[0, 0], intrinsic[1, 1],
                        intrinsic[0, 2], intrinsic[1, 2], 1.0)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud = cloud.reshape(-1, 3)

    # 2. 基础过滤
    depth_flat = depth.reshape(-1)
    valid = np.isfinite(depth_flat) & (depth_flat > 0.1) & (depth_flat < 2.0)

    # 3. affordance 区域过滤: 只保留前景 (mask==0) 的点
    aff_flat = affordance_mask.reshape(-1)
    aff_valid = (aff_flat == 0)

    # 组合: 深度有效 & 在 affordance 区域
    combined_mask = valid & aff_valid
    cloud_proc = cloud[combined_mask]

    n_aff = aff_valid.sum()
    n_combined = combined_mask.sum()
    print(f"  -> Affordance 区域像素: {n_aff}, 有效3D点: {n_combined}")

    if n_combined < 100:
        print("  [Warning] affordance 区域点云过少, 回退到全场景点云")
        cloud_proc = cloud[valid]

    if len(cloud_proc) < 50:
        return {'status': 'FAIL', 'message': '有效点云不足'}

    # 4. 下采样
    if len(cloud_proc) > 20000:
        idx = np.random.choice(len(cloud_proc), 20000, replace=False)
        cloud_proc = cloud_proc[idx]

    # 5. GraspNet 推理
    cloud_tensor = torch.from_numpy(
        cloud_proc[np.newaxis].astype(np.float32)
    ).to(device)

    with torch.no_grad():
        end_points = net({'point_clouds': cloud_tensor})
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    gg.nms()
    gg.sort_by_score()

    if len(gg) == 0:
        return {'status': 'FAIL', 'message': 'GraspNet 未生成位姿'}

    # 6. 保存可视化 (如果指定了 vis_dir)
    if vis_dir:
        save_grasp_vis(rgb, affordance_mask, gg, intrinsic, vis_dir)

    # 7. 返回最佳抓取 (相机坐标系)
    best = gg[0]
    return {
        'status':      'SUCCESS',
        'translation': best.translation.tolist(),      # (3,) 相机系
        'rotation':    best.rotation_matrix.tolist(),   # (3,3) 相机系
        'width':       float(best.width),
        'score':       float(best.score),
        'num_grasps':  len(gg),
    }


def save_grasp_vis(rgb, affordance_mask, gg, intrinsic, vis_dir):
    """保存抓取预测投影可视化"""
    os.makedirs(vis_dir, exist_ok=True)

    vis_img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(vis_img)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    top_k = min(10, len(gg))
    for i in range(top_k):
        t = gg[i].translation
        if t[2] <= 0:
            continue
        u = t[0] * fx / t[2] + cx
        v = t[1] * fy / t[2] + cy
        if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
            r = 8 if i == 0 else 3
            color = 'red' if i == 0 else 'blue'
            draw.ellipse([u - r, v - r, u + r, v + r],
                         fill=color if i == 0 else None, outline=color)
            if i == 0:
                draw.text((u + 10, v), f"Best: {gg[i].score:.3f}", fill='red')

    vis_img.save(os.path.join(vis_dir, 'graspnet_projection.png'))

    # affordance overlay
    overlay = rgb.copy()
    fg = (affordance_mask == 0)
    # 将 affordance 区域染成绿色，并叠加上抓取点
    overlay[fg] = (overlay[fg] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(vis_dir, 'graspnet_affordance_overlay.png'))

    print(f"  -> 可视化图像已更新至: {vis_dir}")


# ============================================================================
#  主程序
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="GraspNet 推理服务")
    p.add_argument('--checkpoint', type=str,
                   default='/opt/data/private/LLMSeg/graspnet-baseline/logs/checkpoint-rs.tar',
                   help='GraspNet checkpoint 路径')
    p.add_argument('--port', type=str, default='5556',
                   help='ZMQ 监听端口')
    p.add_argument('--vis_dir', type=str,
                   default='/opt/data/private/LLMSeg/graspnet-baseline/vigor_grasp_vis',
                   help='可视化输出目录')
    p.add_argument('--device', type=str, default='cuda:1',
                   help='推理使用的设备 (位如 cuda:0, cuda:1)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(">>> GraspNet 推理服务 <<<")
    print("=" * 80)

    # 初始化模型
    net = init_graspnet(args.checkpoint, device)

    # 启动 ZMQ 服务
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"\n-> GraspNet Service 就绪, 监听 port={args.port}")
    print("   等待 vigor_client_main.py 发送请求...\n")

    request_count = 0
    try:
        while True:
            # 阻塞等待请求
            msg = socket.recv_pyobj()
            request_count += 1

            print(f"\n[Request #{request_count}] 收到推理请求")

            rgb              = msg['rgb']               # (H, W, 3)
            depth            = msg['depth']             # (H, W)
            affordance_mask  = msg['affordance_mask']   # (H, W), 0=fg, 1=bg
            intrinsic        = msg['intrinsic']         # (3, 3)

            H, W = depth.shape
            fg_ratio = (affordance_mask == 0).sum() / (H * W) * 100
            print(f"  -> 图像尺寸: {W}x{H}")
            print(f"  -> Affordance 覆盖率: {fg_ratio:.1f}%")

            # 执行推理
            result = predict_grasp(
                net, rgb, depth, affordance_mask, intrinsic,
                device=device, vis_dir=args.vis_dir
            )

            if result['status'] == 'SUCCESS':
                print(f"  -> 生成 {result['num_grasps']} 个候选位姿")
                print(f"  -> 最佳分数: {result['score']:.4f}")
                print(f"  -> 最佳位置(cam): {result['translation']}")
            else:
                print(f"  -> ❌ 失败: {result.get('message', '')}")

            # 返回结果
            socket.send_pyobj(result)

    except KeyboardInterrupt:
        print(f"\n-> 服务关闭 (共处理 {request_count} 个请求)")
    finally:
        socket.close()


if __name__ == "__main__":
    main()
