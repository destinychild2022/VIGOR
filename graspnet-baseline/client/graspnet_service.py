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
from scipy.spatial.transform import Rotation as R

# ============================================================================
#  路径设置
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRASPNET_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (
    GRASPNET_ROOT,
    os.path.join(GRASPNET_ROOT, "models"),
    os.path.join(GRASPNET_ROOT, "utils"),
    os.path.join(GRASPNET_ROOT, "pointnet2"),
    os.path.join(GRASPNET_ROOT, "graspnetAPI"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.graspnet import GraspNet, pred_decode
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup


R_CV2GL = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

AFFORDANCE_CENTER_WEIGHT = 0.3
AFFORDANCE_FINGER_WEIGHT = 0.7
FINAL_GRASPNET_WEIGHT = 0.2
FINAL_AFFORDANCE_WEIGHT = 0.8


def convert_range_depth_to_z_depth(depth, intrinsic):
    """Convert ray/range depth to optical-axis z-depth for pinhole backprojection."""
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    xmap, ymap = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x_norm = (xmap - cx) / fx
    y_norm = (ymap - cy) / fy
    return depth / np.sqrt(1.0 + x_norm * x_norm + y_norm * y_norm)


def normalize_depth_for_backprojection(depth, intrinsic, depth_key=None):
    """Return z-depth expected by GraspNet's create_point_cloud_from_depth_image."""
    key = str(depth_key or "depth_linear").lower()
    if key in {"depth", "range", "ray", "ray_depth", "distance_to_camera"}:
        return convert_range_depth_to_z_depth(depth, intrinsic), f"{key}->z_depth"
    return np.asarray(depth, dtype=np.float32), key


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


def camera_points_to_world(points_cam, cam_pos, cam_quat):
    """Convert OpenCV camera-frame points to simulator world-frame points."""
    cam_rot_mat = R.from_quat(np.asarray(cam_quat, dtype=np.float64)).as_matrix()
    points_gl = (R_CV2GL @ np.asarray(points_cam, dtype=np.float64).T).T
    return (cam_rot_mat @ points_gl.T).T + np.asarray(cam_pos, dtype=np.float64)


def normalize_to_unit_interval(values, neutral=0.5):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.full(values.shape, neutral, dtype=np.float64)
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax - vmin < 1e-8:
        return np.full(values.shape, neutral, dtype=np.float64)
    norm = (values - vmin) / (vmax - vmin)
    norm[~finite_mask] = neutral
    return np.clip(norm, 0.0, 1.0)


def project_points_to_image(points_cam, intrinsic, image_shape):
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    uv = np.zeros((len(points_cam), 2), dtype=np.int32)
    if valid.any():
        u = np.round(points_cam[valid, 0] * fx / z[valid] + cx).astype(np.int32)
        v = np.round(points_cam[valid, 1] * fy / z[valid] + cy).astype(np.int32)
        uv_valid = np.stack([u, v], axis=1)
        uv[valid] = uv_valid
        h, w = image_shape[:2]
        valid_idx = np.flatnonzero(valid)
        inside = (
            (uv_valid[:, 0] >= 0) & (uv_valid[:, 0] < w) &
            (uv_valid[:, 1] >= 0) & (uv_valid[:, 1] < h)
        )
        valid[valid_idx] &= inside
    return uv, valid


def sample_heatmap_values_at_points(points_cam, heatmap, intrinsic):
    if heatmap is None:
        return 0.0
    uv, valid = project_points_to_image(points_cam, intrinsic, heatmap.shape)
    if not valid.any():
        return 0.0
    uv_valid = uv[valid]
    values = heatmap[uv_valid[:, 1], uv_valid[:, 0]]
    if values.size == 0:
        return 0.0
    return float(np.clip(values.mean(), 0.0, 1.0))


def build_grasp_finger_sample_points_cam(
    grasp,
    finger_length=0.06,
    width_scale=1.0,
    depth_scale=1.0,
    height_scale=1.0,
    max_grasp_width=None,
    num_depth_samples=3,
    num_height_samples=3,
):
    width, depth, height = get_effective_grasp_dims(
        grasp,
        width_scale=width_scale,
        depth_scale=depth_scale,
        height_scale=height_scale,
        max_grasp_width=max_grasp_width,
    )
    half_w = width / 2.0
    half_h = height / 2.0
    depth_start = max(depth - max(min(finger_length, depth), 0.01), 0.0)
    x_samples = np.linspace(depth_start, depth, num=max(num_depth_samples, 2))
    z_samples = np.linspace(-half_h, half_h, num=max(num_height_samples, 2))

    local_points = []
    for x in x_samples:
        for z in z_samples:
            local_points.append([x, -half_w, z])
            local_points.append([x, half_w, z])

    local_points = np.asarray(local_points, dtype=np.float64)
    return (grasp.rotation_matrix @ local_points.T).T + grasp.translation


def rerank_grasps_with_affordance(
    gg,
    affordance_mask,
    intrinsic,
    finger_length=0.06,
    width_scale=1.0,
    depth_scale=1.0,
    height_scale=1.0,
    max_grasp_width=None,
):
    """Rerank final filtered grasps using the binary affordance mask as heatmap."""
    info = {
        'enabled': False,
        'center_scores': np.zeros((len(gg),), dtype=np.float64),
        'finger_scores': np.zeros((len(gg),), dtype=np.float64),
        'affordance_scores': np.zeros((len(gg),), dtype=np.float64),
        'graspnet_scores': np.asarray(gg.scores, dtype=np.float64).copy(),
        'final_scores': np.asarray(gg.scores, dtype=np.float64).copy(),
    }
    if affordance_mask is None or len(gg) == 0:
        return gg, info

    heatmap = np.asarray(affordance_mask == 0, dtype=np.float32)
    if heatmap.size == 0 or float(heatmap.max()) <= 0.0:
        return gg, info

    center_scores = np.zeros((len(gg),), dtype=np.float64)
    finger_scores = np.zeros((len(gg),), dtype=np.float64)
    for i in range(len(gg)):
        center_scores[i] = sample_heatmap_values_at_points(
            gg.translations[i:i + 1],
            heatmap=heatmap,
            intrinsic=intrinsic,
        )
        finger_points = build_grasp_finger_sample_points_cam(
            gg[i],
            finger_length=finger_length,
            width_scale=width_scale,
            depth_scale=depth_scale,
            height_scale=height_scale,
            max_grasp_width=max_grasp_width,
        )
        finger_scores[i] = sample_heatmap_values_at_points(
            finger_points,
            heatmap=heatmap,
            intrinsic=intrinsic,
        )

    affordance_raw = (
        AFFORDANCE_CENTER_WEIGHT * center_scores +
        AFFORDANCE_FINGER_WEIGHT * finger_scores
    )
    graspnet_scores = np.asarray(gg.scores, dtype=np.float64).copy()
    graspnet_norm = normalize_to_unit_interval(graspnet_scores)
    affordance_norm = normalize_to_unit_interval(affordance_raw)
    final_scores = (
        FINAL_GRASPNET_WEIGHT * graspnet_norm +
        FINAL_AFFORDANCE_WEIGHT * affordance_norm
    )
    order = np.argsort(final_scores)[::-1]

    info = {
        'enabled': True,
        'center_scores': center_scores[order],
        'finger_scores': finger_scores[order],
        'affordance_scores': affordance_norm[order],
        'graspnet_scores': graspnet_scores[order],
        'final_scores': final_scores[order],
    }
    return gg[order], info


def box_vertices(x_min, x_max, y_min, y_max, z_min, z_max):
    return np.array([
        [x_min, y_min, z_min], [x_min, y_min, z_max],
        [x_min, y_max, z_min], [x_min, y_max, z_max],
        [x_max, y_min, z_min], [x_max, y_min, z_max],
        [x_max, y_max, z_min], [x_max, y_max, z_max],
    ], dtype=np.float64)


def get_effective_grasp_dims(
    grasp,
    width_scale=1.0,
    depth_scale=1.0,
    height_scale=1.0,
    max_grasp_width=None,
):
    width = max(float(grasp.width) * float(width_scale), 1e-4)
    if max_grasp_width is not None:
        width = min(width, float(max_grasp_width))
    depth = float(grasp.depth) * float(depth_scale)
    height = max(float(grasp.height) * float(height_scale), 0.004)
    return width, depth, height


def grasp_gripper_vertices_cam(
    grasp,
    finger_width=0.01,
    finger_length=0.06,
    width_scale=1.0,
    depth_scale=1.0,
    height_scale=1.0,
    max_grasp_width=None,
):
    """Approximate the GraspNet gripper geometry used by collision detection."""
    width, depth, height = get_effective_grasp_dims(
        grasp,
        width_scale=width_scale,
        depth_scale=depth_scale,
        height_scale=height_scale,
        max_grasp_width=max_grasp_width,
    )
    half_w = width / 2.0
    half_h = height / 2.0

    local_vertices = np.concatenate([
        box_vertices(
            depth - finger_length, depth,
            -half_w - finger_width, -half_w,
            -half_h, half_h,
        ),
        box_vertices(
            depth - finger_length, depth,
            half_w, half_w + finger_width,
            -half_h, half_h,
        ),
        box_vertices(
            depth - finger_length - finger_width, depth - finger_length,
            -half_w - finger_width, half_w + finger_width,
            -half_h, half_h,
        ),
    ], axis=0)

    return (grasp.rotation_matrix @ local_vertices.T).T + grasp.translation


def filter_table_safe_grasps(
    gg, cam_pos, cam_quat, table_z, safety_margin,
    finger_width=0.01, finger_length=0.06,
    width_scale=1.0, depth_scale=1.0, height_scale=1.0,
    max_grasp_width=None,
):
    """Keep grasps whose simplified gripper geometry stays above the table."""
    if cam_pos is None or cam_quat is None:
        print("  [Warning] 未收到相机外参，无法执行桌面安全检查，本次按无可用位姿处理")
        return gg[:0], len(gg), None

    safe = []
    min_z_values = []
    min_allowed_z = float(table_z) + float(safety_margin)
    for i in range(len(gg)):
        vertices_cam = grasp_gripper_vertices_cam(
            gg[i],
            finger_width=finger_width,
            finger_length=finger_length,
            width_scale=width_scale,
            depth_scale=depth_scale,
            height_scale=height_scale,
            max_grasp_width=max_grasp_width,
        )
        vertices_world = camera_points_to_world(vertices_cam, cam_pos, cam_quat)
        min_z = float(vertices_world[:, 2].min())
        min_z_values.append(min_z)
        safe.append(min_z >= min_allowed_z)

    safe = np.asarray(safe, dtype=bool)
    min_z_values = np.asarray(min_z_values, dtype=np.float64)
    filtered = gg[safe]
    rejected = int((~safe).sum())
    best_min_z = float(min_z_values[safe][0]) if safe.any() else None
    return filtered, rejected, best_min_z


def build_collision_relaxation_trials(collision_thresh, collision_approach_dist):
    """Return progressively looser collision-filter settings."""
    base_thresh = float(collision_thresh)
    base_approach = float(collision_approach_dist)
    trials = [
        ("base", base_thresh, base_approach),
        ("relaxed", max(base_thresh, 0.05), min(base_approach, 0.03)),
        ("very_relaxed", max(base_thresh, 0.12), min(base_approach, 0.02)),
        ("ultra_relaxed", max(base_thresh, 0.25), min(base_approach, 0.01)),
    ]

    unique_trials = []
    seen = set()
    for label, thresh, approach in trials:
        key = (round(thresh, 6), round(approach, 6))
        if key in seen:
            continue
        seen.add(key)
        unique_trials.append((label, thresh, approach))
    return unique_trials


def filter_collision_safe_grasps(
    gg,
    cloud_scene,
    collision_thresh=0.08,
    voxel_size=0.01,
    collision_approach_dist=0.02,
    gripper_finger_width=0.01,
    gripper_finger_length=0.06,
    grasp_width_scale=1.0,
    grasp_depth_scale=1.0,
    grasp_height_scale=1.0,
    max_grasp_width=None,
    mode_label='enabled',
):
    """Apply collision filtering once with the provided settings."""
    info = {
        'mode': 'disabled',
        'collision_thresh': None,
        'collision_approach_dist': None,
        'survivors': len(gg),
        'rejected': 0,
    }
    if collision_thresh <= 0 or len(cloud_scene) == 0 or len(gg) == 0:
        return gg, 0, info

    detector = ModelFreeCollisionDetector(
        cloud_scene.astype(np.float32),
        voxel_size=voxel_size,
        finger_width=gripper_finger_width,
        finger_length=gripper_finger_length,
        width_scale=grasp_width_scale,
        depth_scale=grasp_depth_scale,
        height_scale=grasp_height_scale,
        max_grasp_width=max_grasp_width,
    )
    collision_mask = detector.detect(
        gg,
        approach_dist=collision_approach_dist,
        collision_thresh=collision_thresh,
    )
    rejected = int(collision_mask.sum())
    filtered = gg[~collision_mask]
    info = {
        'mode': str(mode_label),
        'collision_thresh': float(collision_thresh),
        'collision_approach_dist': float(collision_approach_dist),
        'survivors': len(filtered),
        'rejected': rejected,
    }
    return filtered, rejected, info


def select_grasps_with_relaxed_collision(
    gg,
    cloud_scene,
    cam_pos,
    cam_quat,
    collision_thresh=0.08,
    voxel_size=0.01,
    collision_approach_dist=0.02,
    table_z=0.40,
    table_safety_margin=0.01,
    gripper_finger_width=0.01,
    gripper_finger_length=0.06,
    grasp_width_scale=1.0,
    grasp_depth_scale=1.0,
    grasp_height_scale=1.0,
    max_grasp_width=None,
):
    """Relax collision filtering only when it still leads to zero table-safe grasps."""
    if collision_thresh <= 0 or len(cloud_scene) == 0:
        collision_trials = [('disabled', None, None)]
    else:
        collision_trials = build_collision_relaxation_trials(
            collision_thresh=collision_thresh,
            collision_approach_dist=collision_approach_dist,
        )
        collision_trials.append(('disabled', None, None))

    last_info = {
        'mode': 'disabled',
        'collision_thresh': None,
        'collision_approach_dist': None,
        'num_collision_rejected': 0,
        'num_table_rejected': 0,
        'best_min_z': None,
    }
    base_attempt = None
    for label, trial_thresh, trial_approach in collision_trials:
        if label == 'disabled':
            trial_grasps = gg
            num_collision_rejected = 0
            collision_info = {
                'mode': 'disabled',
                'collision_thresh': None,
                'collision_approach_dist': None,
                'survivors': len(trial_grasps),
                'rejected': 0,
            }
            print(f"  -> 碰撞过滤[{label}]: 跳过，保留 {len(trial_grasps)} 个候选")
        else:
            trial_grasps, num_collision_rejected, collision_info = filter_collision_safe_grasps(
                gg,
                cloud_scene=cloud_scene,
                collision_thresh=trial_thresh,
                voxel_size=voxel_size,
                collision_approach_dist=trial_approach,
                gripper_finger_width=gripper_finger_width,
                gripper_finger_length=gripper_finger_length,
                grasp_width_scale=grasp_width_scale,
                grasp_depth_scale=grasp_depth_scale,
                grasp_height_scale=grasp_height_scale,
                max_grasp_width=max_grasp_width,
                mode_label=label,
            )
            print(
                f"  -> 碰撞过滤[{label}]: thresh={trial_thresh:.3f}, "
                f"approach={trial_approach:.3f}, {len(gg)} -> {len(trial_grasps)}"
            )

        if base_attempt is None:
            base_attempt = collision_info

        if len(trial_grasps) == 0:
            print(f"  [Warning] 碰撞过滤[{label}] 后无候选，继续放松碰撞过滤")
            last_info = {
                'mode': collision_info['mode'],
                'collision_thresh': collision_info['collision_thresh'],
                'collision_approach_dist': collision_info['collision_approach_dist'],
                'num_collision_rejected': num_collision_rejected,
                'num_table_rejected': 0,
                'best_min_z': None,
            }
            continue

        table_grasps, num_table_rejected, best_min_z = filter_table_safe_grasps(
            trial_grasps,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            table_z=table_z,
            safety_margin=table_safety_margin,
            finger_width=gripper_finger_width,
            finger_length=gripper_finger_length,
            width_scale=grasp_width_scale,
            depth_scale=grasp_depth_scale,
            height_scale=grasp_height_scale,
            max_grasp_width=max_grasp_width,
        )
        print(
            f"  -> 桌面安全过滤[{label}]: {len(trial_grasps)} -> {len(table_grasps)} "
            f"(剔除 {num_table_rejected}), "
            f"best_min_z={best_min_z if best_min_z is not None else 'None'}"
        )

        trial_info = {
            'mode': collision_info['mode'],
            'collision_thresh': collision_info['collision_thresh'],
            'collision_approach_dist': collision_info['collision_approach_dist'],
            'num_collision_rejected': num_collision_rejected,
            'num_table_rejected': num_table_rejected,
            'best_min_z': best_min_z,
        }
        if len(table_grasps) > 0:
            if base_attempt is not None and label != 'base':
                print(
                    "  [Warning] 基础碰撞过滤在桌面高度过滤后无可用位姿，"
                    f"自动放宽到 [{label}]"
                )
            return table_grasps, trial_info

        print(f"  [Warning] 组合过滤[{label}] 后仍无可用位姿，继续放松碰撞过滤")
        last_info = trial_info

    return gg[:0], last_info


def predict_grasp(net, rgb, depth, affordance_mask, intrinsic, device="cuda",
                  vis_dir=None, cam_pos=None, cam_quat=None,
                  collision_thresh=0.08, voxel_size=0.01,
                  collision_approach_dist=0.02, table_z=0.40,
                  table_safety_margin=0.01, min_mask_points=100,
                  gripper_finger_width=0.01, gripper_finger_length=0.06,
                  grasp_width_scale=1.0, grasp_depth_scale=1.0,
                  grasp_height_scale=1.0, max_grasp_width=None):
    """
    在整场景有效点云上预测抓取位姿。
    如果提供 affordance mask，则只用于最终过滤后的 grasp rerank。

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
    cloud_organized = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud = cloud_organized.reshape(-1, 3)

    # 2. 基础过滤
    depth_flat = depth.reshape(-1)
    valid = np.isfinite(depth_flat) & (depth_flat > 0.1) & (depth_flat < 2.0)

    cloud_scene = cloud[valid]
    cloud_proc = cloud_scene
    used_full_scene_for_prediction = True

    if affordance_mask is not None:
        aff_flat = affordance_mask.reshape(-1)
        n_aff = int((aff_flat == 0).sum())
        print(f"  -> Affordance 区域像素: {n_aff}")
        print("  -> 局部目标区域: disabled_whole_scene")
    else:
        print("  -> 未提供 affordance mask，最终 rerank 将跳过")

    print(f"  -> GraspNet 输入点云: 全场景有效点 {len(cloud_proc)}")

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

    num_raw = len(gg)

    # 6. 先按当前碰撞档位过滤，再执行桌面高度过滤；若桌面过滤后为空，就继续放松碰撞过滤。
    gg, filter_info = select_grasps_with_relaxed_collision(
        gg,
        cloud_scene=cloud_scene,
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        collision_thresh=collision_thresh,
        voxel_size=voxel_size,
        collision_approach_dist=collision_approach_dist,
        table_z=table_z,
        table_safety_margin=table_safety_margin,
        gripper_finger_width=gripper_finger_width,
        gripper_finger_length=gripper_finger_length,
        grasp_width_scale=grasp_width_scale,
        grasp_depth_scale=grasp_depth_scale,
        grasp_height_scale=grasp_height_scale,
        max_grasp_width=max_grasp_width,
    )
    num_collision_rejected = int(filter_info['num_collision_rejected'])
    num_table_rejected = int(filter_info['num_table_rejected'])
    best_min_z = filter_info['best_min_z']
    print(
        f"  -> 组合过滤结果: {num_raw} -> {len(gg)} "
        f"(collision_mode={filter_info['mode']}, "
        f"collision_rejected={num_collision_rejected})"
    )
    print(
        f"  -> 末次桌面过滤: 剔除 {num_table_rejected}, "
        f"best_min_z={best_min_z if best_min_z is not None else 'None'}"
    )

    if len(gg) == 0:
        return {'status': 'FAIL', 'message': '桌面安全过滤后无可用位姿'}

    gg, rerank_info = rerank_grasps_with_affordance(
        gg,
        affordance_mask=affordance_mask,
        intrinsic=intrinsic,
        finger_length=gripper_finger_length,
        width_scale=grasp_width_scale,
        depth_scale=grasp_depth_scale,
        height_scale=grasp_height_scale,
        max_grasp_width=max_grasp_width,
    )
    if rerank_info['enabled']:
        print(
            "  -> affordance rerank: "
            f"best graspnet={rerank_info['graspnet_scores'][0]:.4f}, "
            f"aff={rerank_info['affordance_scores'][0]:.4f}, "
            f"final={rerank_info['final_scores'][0]:.4f}"
        )

    # 8. 保存可视化 (如果指定了 vis_dir)
    if vis_dir:
        save_grasp_vis(
            rgb,
            affordance_mask,
            gg,
            intrinsic,
            vis_dir,
            rerank_info=rerank_info,
        )

    # 9. 返回最佳抓取 (相机坐标系)
    best = gg[0]
    return {
        'status':      'SUCCESS',
        'translation': best.translation.tolist(),      # (3,) 相机系
        'rotation':    best.rotation_matrix.tolist(),   # (3,3) 相机系
        'width':       float(best.width),
        'score':       float(best.score),
        'graspnet_score': float(rerank_info['graspnet_scores'][0]) if rerank_info['enabled'] else float(best.score),
        'affordance_score': float(rerank_info['affordance_scores'][0]) if rerank_info['enabled'] else None,
        'final_score': float(rerank_info['final_scores'][0]) if rerank_info['enabled'] else float(best.score),
        'center_score': float(rerank_info['center_scores'][0]) if rerank_info['enabled'] else None,
        'finger_score': float(rerank_info['finger_scores'][0]) if rerank_info['enabled'] else None,
        'num_grasps':  len(gg),
        'num_raw_grasps': num_raw,
        'num_collision_rejected': num_collision_rejected,
        'collision_filter_mode': filter_info['mode'],
        'collision_filter_thresh': filter_info['collision_thresh'],
        'collision_filter_approach_dist': filter_info['collision_approach_dist'],
        'num_table_rejected': num_table_rejected,
        'best_gripper_min_world_z': best_min_z,
        'used_full_scene_for_prediction': used_full_scene_for_prediction,
    }


def save_grasp_vis(rgb, affordance_mask, gg, intrinsic, vis_dir, rerank_info=None):
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
                if rerank_info and rerank_info.get('enabled'):
                    draw.text(
                        (u + 10, v),
                        f"Best F:{rerank_info['final_scores'][0]:.3f}",
                        fill='red',
                    )
                else:
                    draw.text((u + 10, v), f"Best: {gg[i].score:.3f}", fill='red')

    vis_img.save(os.path.join(vis_dir, 'graspnet_projection.png'))

    for stale_name in ('graspnet_affordance_overlay.png', 'graspnet_local_target_overlay.png'):
        stale_path = os.path.join(vis_dir, stale_name)
        if os.path.exists(stale_path):
            try:
                os.remove(stale_path)
            except OSError:
                pass

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
    p.add_argument('--collision_thresh', type=float, default=0.08,
                   help='全场景点云碰撞过滤阈值；<=0 时关闭')
    p.add_argument('--voxel_size', type=float, default=0.01,
                   help='碰撞检测点云体素下采样大小')
    p.add_argument('--collision_approach_dist', type=float, default=0.02,
                   help='碰撞检测中预抓取接近段长度')
    p.add_argument('--table_z', type=float, default=0.40,
                   help='仿真世界系桌面高度')
    p.add_argument('--table_safety_margin', type=float, default=0.01,
                   help='夹爪最低点相对桌面的安全余量')
    p.add_argument('--min_mask_points', type=int, default=100,
                   help='mask 内最少有效点数，低于该值不回退全场景估计')
    p.add_argument('--gripper_finger_width', type=float, default=0.01,
                   help='用于碰撞/桌面过滤的单指厚度')
    p.add_argument('--gripper_finger_length', type=float, default=0.06,
                   help='用于碰撞/桌面过滤的单指长度')
    p.add_argument('--grasp_width_scale', type=float, default=1.0,
                   help='对 GraspNet grasp.width 的缩放系数')
    p.add_argument('--grasp_depth_scale', type=float, default=1.0,
                   help='对 GraspNet grasp.depth 的缩放系数')
    p.add_argument('--grasp_height_scale', type=float, default=1.0,
                   help='对 GraspNet grasp.height 的缩放系数')
    p.add_argument('--max_grasp_width', type=float, default=0.0,
                   help='真实夹爪最大开口；<=0 表示不裁剪 grasp.width')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    max_grasp_width = None if args.max_grasp_width <= 0 else args.max_grasp_width

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
    print(
        "   简化夹爪参数:"
        f" finger_width={args.gripper_finger_width},"
        f" finger_length={args.gripper_finger_length},"
        f" width_scale={args.grasp_width_scale},"
        f" depth_scale={args.grasp_depth_scale},"
        f" height_scale={args.grasp_height_scale},"
        f" max_width={max_grasp_width}"
    )

    request_count = 0
    try:
        while True:
            # 阻塞等待请求
            msg = socket.recv_pyobj()
            request_count += 1

            print(f"\n[Request #{request_count}] 收到推理请求")

            rgb              = msg['rgb']               # (H, W, 3)
            depth            = msg['depth']             # (H, W)
            depth_key        = msg.get('depth_key', 'depth_linear')
            affordance_mask  = msg['affordance_mask']   # (H, W), 0=fg, 1=bg
            intrinsic        = msg['intrinsic']         # (3, 3)
            cam_pos          = msg.get('cam_pos')
            cam_quat         = msg.get('cam_quat')
            depth, depth_key = normalize_depth_for_backprojection(depth, intrinsic, depth_key)

            H, W = depth.shape
            fg_ratio = (affordance_mask == 0).sum() / (H * W) * 100
            print(f"  -> 图像尺寸: {W}x{H}")
            print(f"  -> Depth key: {depth_key}")
            print(f"  -> Affordance 覆盖率: {fg_ratio:.1f}%")

            # 执行推理
            result = predict_grasp(
                net, rgb, depth, affordance_mask, intrinsic,
                device=device, vis_dir=args.vis_dir,
                cam_pos=cam_pos, cam_quat=cam_quat,
                collision_thresh=args.collision_thresh,
                voxel_size=args.voxel_size,
                collision_approach_dist=args.collision_approach_dist,
                table_z=args.table_z,
                table_safety_margin=args.table_safety_margin,
                min_mask_points=args.min_mask_points,
                gripper_finger_width=args.gripper_finger_width,
                gripper_finger_length=args.gripper_finger_length,
                grasp_width_scale=args.grasp_width_scale,
                grasp_depth_scale=args.grasp_depth_scale,
                grasp_height_scale=args.grasp_height_scale,
                max_grasp_width=max_grasp_width,
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
