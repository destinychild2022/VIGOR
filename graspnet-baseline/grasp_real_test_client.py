#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GraspNet 实时决策客户端 (增加预测位姿投影可视化)
=========================================
1. 可视化原始 RGB 和 归一化深度。
2. 将预测出的抓取中心投影回图像上，绘制标记并保存。
"""

import os
import sys
import numpy as np
import torch
import zmq
import argparse
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import cv2

# 添加项目依赖路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup

# ============================================================================
#  工具函数
# ============================================================================
def project_points(points_3d, intrinsic):
    """
    将 3D 坐标投影到 2D 像素坐标
    points_3d: (N, 3) 在相机坐标系下的坐标
    intrinsic: (3, 3) 内参矩阵
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = (points_3d[:, 0] * fx / points_3d[:, 2]) + cx
    v = (points_3d[:, 1] * fy / points_3d[:, 2]) + cy
    
    return np.stack([u, v], axis=1)

def save_visualizations(rgb, depth, gg, intrinsic, save_dir):
    """
    保存 2D 预测结果图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 基础图保存
    Image.fromarray(rgb).save(os.path.join(save_dir, 'current_rgb.png'))
    
    # 2. 预测点位投影
    if len(gg) > 0:
        # 准备画布
        vis_img = Image.fromarray(rgb.copy())
        draw = ImageDraw.Draw(vis_img)
        
        # 提取前 20 个高分位姿的中心点 (相机坐标系)
        # graspnetAPI 的 Grasp 对象属性：translation 是中心点
        top_k = min(20, len(gg))
        centers_3d = np.array([gg[i].translation for i in range(top_k)])
        
        # 投影到像素坐标
        pixel_coords = project_points(centers_3d, intrinsic)
        
        for i in range(top_k):
            u, v = pixel_coords[i]
            # 检查坐标是否在图像范围内
            if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                if i == 0:
                    # 最佳抓取：大红点 + 标注分数
                    r = 8
                    draw.ellipse([u-r, v-r, u+r, v+r], fill='red', outline='white')
                    draw.text((u+10, v), f"Best Score: {gg[i].score:.3f}", fill='red')
                else:
                    # 其他候选位姿：小蓝圈
                    r = 3
                    draw.ellipse([u-r, v-r, u+r, v+r], outline='blue')
        
        vis_img.save(os.path.join(save_dir, 'grasp_prediction.png'))
    
    # 3. 深度图保存
    depth_vis = depth.copy()
    depth_vis[~np.isfinite(depth_vis)] = 0
    d_min, d_max = depth_vis.min(), depth_vis.max()
    if d_max > d_min:
        depth_norm = (depth_vis - d_min) / (d_max - d_min) * 255
        Image.fromarray(depth_norm.astype(np.uint8)).save(os.path.join(save_dir, 'current_depth.png'))

    print(f"-> 预测结果已输出至: {os.path.join(save_dir, 'grasp_prediction.png')}")

# ============================================================================
#  主程序
# ============================================================================
def transform_to_world(translation, rotation, cam_pos, cam_quat):
    # 仿真器相机位姿 (OpenGL Camera -> World)
    cam_rot_mat = R.from_quat(cam_quat).as_matrix()
    
    # 坐标系转换矩阵 (OpenCV Camera -> OpenGL Camera)
    # OpenCV: X右, Y下, Z前 (GraspNet 使用)
    # OpenGL: X右, Y上, Z后 (OmniGibson 相机帧)
    # 需要绕 X 轴旋转 180 度
    R_cv2gl = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    # 先将 GraspNet 预测的位姿转换到 OpenGL 相机坐标系下
    trans_gl = R_cv2gl @ translation
    rot_gl = R_cv2gl @ rotation
    
    # 再借助相机的世界位姿，将点转换到世界坐标系
    world_pos = cam_rot_mat @ trans_gl + cam_pos
    world_rot_mat = cam_rot_mat @ rot_gl
    
    return world_pos, world_rot_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='219.223.182.106')
    parser.add_argument('--checkpoint', type=str, default='/opt/data/private/LLMSeg/graspnet-baseline/logs/checkpoint-rs.tar')
    parser.add_argument('--vis_dir', type=str, default='/opt/data/private/LLMSeg/graspnet-baseline/rgbd')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    net.to(device); net.eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("-> 系统就绪")

    context = zmq.Context()
    socket = context.socket(zmq.REQ); socket.connect(f"tcp://{args.server_ip}:5555")

    try:
        while True:
            input("\n[Ready] 按回车开始单次抓取实验...")
            socket.send_pyobj({'type': 'GET_OBS'})
            obs = socket.recv_pyobj()
            rgb, depth, intrinsic = obs['rgb'], obs['depth'], obs['intrinsic']
            cam_pos, cam_quat = obs['cam_pos'], obs['cam_quat']
            
            H, W = depth.shape
            camera = CameraInfo(W, H, intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2], 1.0)
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True).reshape(-1, 3)
            mask = np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0.1) & (depth.reshape(-1) < 2.0)
            cloud_proc = cloud[mask]
            
            if len(cloud_proc) > 20000:
                cloud_proc = cloud_proc[np.random.choice(len(cloud_proc), 20000, replace=False)]
            cloud_sampled = torch.from_numpy(cloud_proc[np.newaxis].astype(np.float32)).to(device)

            with torch.no_grad():
                end_points = net({'point_clouds': cloud_sampled})
                grasp_preds = pred_decode(end_points)
            
            gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
            gg.nms(); gg.sort_by_score()

            # 保存可视化 (带投影点)
            save_visualizations(rgb, depth, gg, intrinsic, args.vis_dir)

            if len(gg) > 0:
                best = gg[0]
                # 转换为世界坐标系
                world_pos, world_rot = transform_to_world(best.translation, best.rotation_matrix, cam_pos, cam_quat)
                
                # 打印详细位姿信息供调试
                world_euler = R.from_matrix(world_rot).as_euler('xyz', degrees=True)
                print(f"\n[Grasp Prediction]")
                print(f"-> World Translation: {world_pos}")
                print(f"-> World Euler (deg): {world_euler}")
                print(f"-> Internal Width: {best.width:.4f}")

                socket.send_pyobj({
                    'type': 'EXECUTE', 'translation': world_pos.tolist(),
                    'rotation': world_rot.tolist(), 'width': float(best.width)
                })
                print(f"-> Server Response: {socket.recv_pyobj()['status']}")

    except KeyboardInterrupt: pass
    finally: socket.close()

if __name__ == "__main__":
    main()
