#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VIGOR 主控客户端 (SAM + VIGOR affordance)
==========================================
运行环境: .venv (uv 环境)
          source /opt/data/private/LLMSeg/.venv/bin/activate

职责:
  1. 从 OmniGibson Server 获取 RGB + Depth
  2. SAM (ViT-H) 生成候选掩码
  3. VIGOR (LLMSeg) 根据语言指令选择 affordance 掩码
  4. 将 affordance mask + RGB + Depth 发给 GraspNet Service
  5. 收到抓取位姿后转发给 OmniGibson Server

用法:
  source /opt/data/private/LLMSeg/.venv/bin/activate
  python vigor_client_main.py \
      --server_ip 219.223.182.106 \
      --instruction "pick up the bolt"
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import zmq
import cv2
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

# ============================================================================
#  路径设置: 确保 VIGOR 项目的 model/ 和 utils/ 可以被导入
# ============================================================================
# 脚本在 [LLMSeg]/graspnet-baseline/client/ 下，而 model/ 在 [LLMSeg]/ 下
# 所以我们需要向上跳两级找到 [LLMSeg] 根目录
VIGOR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, VIGOR_ROOT)


# ============================================================================
#  Step 2: SAM 候选掩码生成
# ============================================================================
def generate_sam_masks(sam_socket, rgb_np):
    """
    通过 ZMQ 向远程服务请求 SAM 候选掩码。
    """
    print(f"  -> 正在请求远程 SAM 服务生成掩码...")
    sam_socket.send_pyobj({'rgb': rgb_np})
    response = sam_socket.recv_pyobj()
    masks_binary = response['masks']
    print(f"  -> 远程 SAM 返回了 {len(masks_binary)} 个候选掩码")
    return masks_binary


# ============================================================================
#  Step 3: VIGOR (LLMSeg) affordance 推理
# ============================================================================
def init_vigor(args, device="cuda"):
    """
    初始化 VIGOR (LLMSeg / LISA) 模型
    参考: test_llmseg_vigor.py 中 load_model()
    """
    import warnings
    warnings.filterwarnings("ignore")

    from transformers import AutoTokenizer, CLIPImageProcessor
    from model.LISA import LISAForCausalLM
    from model.llava import conversation as conversation_lib
    from model.segment_anything.utils.transforms import ResizeLongestSide
    from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

    print("  加载 VIGOR (LLMSeg) 模型...")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.vigor_version,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )

    # --- 精度 ---
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    # --- 加载基础模型 ---
    # 使用极致节约内存加载模式 (device_map 将让模型尽可能直读显存)
    # 我们将模型指定到你选定的 device_vigor 上
    model = LISAForCausalLM.from_pretrained(
        args.vigor_version,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        train_mask_decoder=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_idx,
        out_dim=256,
        vision_pretrained=args.vision_pretrained,
        use_mm_start_end=True,
        mm_vision_tower=args.vision_tower,
        device_map={"": device}, # 修正变量名为 device
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 初始化视觉模块 ---
    class _VisionArgs:
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        pretrain_mm_mlp_adapter = None
        vision_tower = args.vision_tower
    model.get_model().initialize_vision_modules(_VisionArgs())
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=device)
    model.get_model().initialize_lisa_modules(model.get_model().config)
    model.resize_token_embeddings(len(tokenizer))

    # --- LoRA ---
    from peft import LoraConfig, get_peft_model
    import torch.nn as nn

    def find_linear_layers(mdl, targets):
        names = set()
        exclude = ["visual_model", "vision_tower", "mm_projector",
                    "text_hidden_fcs", "lisa_attention_layers",
                    "lisa_final_attn", "lisa_norm_final_attn",
                    "lisa_iou_head", "lisa_embedding_head", "lisa_dino_conv"]
        for n, m in mdl.named_modules():
            if isinstance(m, nn.Linear) and all(x not in n for x in exclude) \
               and any(x in n for x in targets):
                names.add(n)
        return sorted(names)

    lora_targets = find_linear_layers(model, ["q_proj", "v_proj"])
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=lora_targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # --- 加载微调权重 ---
    if args.vigor_checkpoint and os.path.exists(args.vigor_checkpoint):
        _load_vigor_checkpoint(model, args.vigor_checkpoint)

    model = model.to(dtype=torch_dtype, device=device)
    model.eval()
    print("  ✅ VIGOR 模型加载完成")

    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(896)

    return model, tokenizer, clip_processor, transform, seg_token_idx


def _load_vigor_checkpoint(model, ckpt_path):
    """从 DeepSpeed checkpoint 或 LoRA adapter 加载权重"""
    from peft import PeftModel

    lora_path = os.path.join(ckpt_path, "lora_adapter")
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  ✅ LoRA adapter 加载: {lora_path}")
        return

    # DeepSpeed checkpoint
    step_dirs = [d for d in os.listdir(ckpt_path)
                 if os.path.isdir(os.path.join(ckpt_path, d))
                 and d.startswith("global_step")]
    if not step_dirs:
        print(f"  [Warning] 未找到 checkpoint: {ckpt_path}")
        return

    step_dirs.sort(key=lambda x: int(x.replace("global_step", "")))
    step_dir = step_dirs[-1]
    mp_path = os.path.join(ckpt_path, step_dir, "mp_rank_00_model_states.pt")
    if os.path.exists(mp_path):
        state = torch.load(mp_path, map_location="cpu")
        if "module" in state:
            missing, unexpected = model.load_state_dict(state["module"], strict=False)
            print(f"  ✅ 已加载微调权重 (Step: {step_dir})")
            print(f"     路径: {mp_path}")
            print(f"     (Missing: {len(missing)}, Unexpected: {len(unexpected)})")


def vigor_predict_affordance(
    model, tokenizer, clip_processor, transform,
    rgb_np, instruction, sam_masks_binary, precision="bf16"
):
    """
    使用 VIGOR 模型从 SAM 候选中选择最佳 affordance 掩码。

    Args:
        rgb_np:           (H, W, 3) uint8, RGB
        instruction:      str, 语言指令
        sam_masks_binary: list of (H, W) uint8, 0=前景 1=背景

    Returns:
        best_mask: (H, W) uint8, 0=前景 1=背景
        best_idx:  int
    """
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.utils import dict_to_cuda

    DEFAULT_IMAGE_TOKEN = "<image>"
    ori_size = rgb_np.shape[:2]
    image_size = 896

    if len(sam_masks_binary) == 0:
        print("  [Warning] SAM 没有生成掩码, 返回全背景")
        return np.ones(ori_size, dtype=np.uint8), -1

    # --- 准备 SAM 候选 segs ---
    K = len(sam_masks_binary)
    segs_origin = np.stack(sam_masks_binary, axis=2)  # (H, W, K)

    # resize + pad
    segs_resized_list = []
    for k in range(K):
        mk = segs_origin[:, :, k]
        mk_u8 = (mk * 255).astype(np.uint8)
        mk_resized = transform.apply_image(mk_u8).astype(np.float32) / 255.0
        segs_resized_list.append(mk_resized)

    rh, rw = segs_resized_list[0].shape
    segs_resized = np.stack(segs_resized_list, axis=2)
    padh, padw = image_size - rh, image_size - rw
    segs_square = np.pad(segs_resized, ((0, padh), (0, padw), (0, 0)),
                         mode="constant", constant_values=1)

    segs_tensor = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous()
    segs = F.interpolate(
        segs_tensor.unsqueeze(0), size=(256, 256),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    segs = segs.to(torch_dtype)

    # --- CLIP 图像 ---
    image_clip = clip_processor.preprocess(rgb_np, return_tensors="pt")["pixel_values"][0]

    # --- SAM 图像预处理 ---
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std  = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    image_resized = transform.apply_image(rgb_np)
    resize = image_resized.shape[:2]
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float()
    image_tensor = (image_tensor - pixel_mean) / pixel_std
    h, w = image_tensor.shape[-2:]
    image_tensor = F.pad(image_tensor, (0, image_size - w, 0, image_size - h))

    # --- 构建对话 ---
    question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], "[SEG]")
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dummy_ious = np.zeros((1, K))
    dummy_iops = np.zeros((1, K))

    input_dict = {
        "images":           image_tensor.unsqueeze(0),
        "images_clip":      image_clip.unsqueeze(0),
        "input_ids":        input_ids,
        "labels":           labels,
        "attention_masks":  attention_mask,
        "offset":           torch.tensor([0, 1]),
        "masks_list":       [torch.zeros(1, ori_size[0], ori_size[1])],
        "label_list":       [torch.ones(ori_size[0], ori_size[1]) * 255],
        "resize_list":      [resize],
        "sam_segs_list":    [segs],
        "sam_ious_list":    [dummy_ious],
        "sam_iops_list":    [dummy_iops],
        "origin_segs_list": [segs_origin],
        "inference":        True,
    }

    device = next(model.parameters()).device
    input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)

    with torch.no_grad():
        output_dict = model(**input_dict)

    pred_similarity = output_dict["pred_similarity"][0]  # (1, K)
    best_idx = torch.argmax(pred_similarity).item()
    best_mask = segs_origin[:, :, best_idx]

    print(f"  -> VIGOR 选择了掩码 #{best_idx} (score={pred_similarity[0, best_idx]:.4f})")
    return best_mask, best_idx


# ============================================================================
#  坐标转换: 相机系 → 世界系
# ============================================================================
def transform_to_world(translation, rotation, cam_pos, cam_quat):
    """将 GraspNet 预测 (OpenCV 相机系) 转到世界系"""
    cam_rot_mat = R.from_quat(cam_quat).as_matrix()
    R_cv2gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    trans_gl = R_cv2gl @ translation
    rot_gl   = R_cv2gl @ rotation
    world_pos = cam_rot_mat @ trans_gl + cam_pos
    world_rot = cam_rot_mat @ rot_gl
    return world_pos, world_rot


# ============================================================================
#  可视化保存
# ============================================================================
def save_debug_images(rgb, depth, affordance_mask, save_dir):
    """保存调试可视化图像"""
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(rgb).save(os.path.join(save_dir, 'rgb.png'))

    # affordance overlay (红色)
    overlay = rgb.copy()
    fg = (affordance_mask == 0)
    overlay[fg] = (overlay[fg] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(save_dir, 'affordance_overlay.png'))

    # 深度图
    d_vis = depth.copy()
    d_vis[~np.isfinite(d_vis)] = 0
    d_min, d_max = d_vis.min(), d_vis.max()
    if d_max > d_min:
        d_norm = ((d_vis - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        Image.fromarray(d_norm).save(os.path.join(save_dir, 'depth.png'))

    print(f"  -> 调试图像已保存至: {save_dir}")


# ============================================================================
#  主程序
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="VIGOR 主控客户端 (SAM + VIGOR)")

    # 连接 - OmniGibson Server
    p.add_argument('--server_ip', type=str, default='219.223.182.106',
                   help='OmniGibson Server IP')
    p.add_argument('--server_port', type=str, default='5555')

    # 连接 - GraspNet Service (物理显卡 2 所在的服务器)
    p.add_argument('--graspnet_ip', type=str, default='localhost',
                   help='GraspNet Service 所在的 IP (跨服务器时修改)')
    p.add_argument('--graspnet_port', type=str, default='5556',
                   help='本地 GraspNet Service 端口')

    # 连接 - SAM Service (物理显卡 2 所在的服务器)
    p.add_argument('--sam_ip', type=str, default='localhost',
                   help='SAM Service 所在的 IP (跨服务器时修改)')
    p.add_argument('--sam_port', type=str, default='5557',
                   help='本地 SAM Service 端口')

    # 语言指令
    p.add_argument('--instruction', type=str, default=None,
                   help='抓取语言指令, 如 "pick up the bolt"')
    p.add_argument('--instruction_idx', type=int, default=0,
                   help='手动模式下使用样本 instructions 中的第几条指令 (0-indexed)')

    # VIGOR
    p.add_argument('--vigor_version', type=str,
                   default='/opt/data/private/model/LISA_Plus_7b')
    p.add_argument('--vigor_checkpoint', type=str,
                   default='/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple/ckpt_model')
    p.add_argument('--vision_tower', type=str,
                   default='/opt/data/private/model/clip-vit-large-patch14')
    p.add_argument('--vision_pretrained', type=str,
                   default='/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth')

    # 显存与精度
    p.add_argument('--vigor_device', type=str, default='cuda:0', help='LISA/VIGOR 所在显卡')
    p.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'bf16', 'fp16'])

    # 数据集
    p.add_argument('--dataset_path', type=str,
                   default='/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/open_vocab_grasp_hard.json',
                   help='测试数据集 JSON 路径')
    p.add_argument('--scene_id', type=str, default=None,
                   help='手动模式下指定 JSON 中的 scene 编号，如 7、10940')
    p.add_argument('--object_idx', type=int, default=0,
                   help='手动模式下指定 scene 内第几个物体/sample (0-indexed)')
    p.add_argument('--auto', action='store_true',
                   help='自动遍历 JSON 中每个 sample 的每条 instruction')
    p.add_argument('--max_attempts', type=int, default=8,
                   help='每条 instruction 最多连续尝试多少步')
    p.add_argument('--max_scene_id', type=int, default=None,
                   help='自动模式只遍历 scene id <= 该值的 samples；None 表示直到 JSON 结束')
    p.add_argument('--results_path', type=str, default=None,
                   help='自动模式结果 JSON 保存路径；为空则只打印统计')
    p.add_argument('--record_video', action='store_true',
                   help='自动模式下为每条 instruction 在 server 端录制一段视频')
    p.add_argument('--server_video_dir', type=str,
                   default='/home/harrison/workspace/BEHAVIOR-1K/test/vigor_instruction_videos',
                   help='视频保存目录；该路径在 OmniGibson server 端解析')
    p.add_argument('--video_fps', type=int, default=12,
                   help='server 端保存视频的 FPS')
    p.add_argument('--video_frame_stride', type=int, default=2,
                   help='server 端每隔多少个仿真 step 写一帧，越大越省空间')

    # 输出
    p.add_argument('--vis_dir', type=str,
                   default='/opt/data/private/LLMSeg/graspnet-baseline/vigor_grasp_vis')

    return p.parse_args()


def load_dataset_samples(json_path):
    """读取 VIGOR JSON。"""
    if not json_path or not os.path.exists(json_path):
        print(f"  [Error] 数据集不存在: {json_path}")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    print(f"  -> 数据集加载成功: {json_path} (samples={len(samples)})")
    return samples


def _safe_instructions(sample):
    instructions = sample.get('instructions', [])
    return instructions if isinstance(instructions, list) else []


def _scene_id_as_int(sample):
    try:
        return int(str(sample.get('scene')))
    except (TypeError, ValueError):
        return None


def select_manual_sample(samples, args):
    """手动模式：按 scene_id + scene 内 object_idx 选择。"""
    if not samples:
        return None, None

    if args.scene_id is None:
        raise ValueError("手动模式必须指定 --scene_id，不再支持 JSON 全局 sample index")

    scene_id = str(args.scene_id)
    scene_items = [
        (idx, sample) for idx, sample in enumerate(samples)
        if str(sample.get('scene')) == scene_id
    ]
    if not scene_items:
        raise ValueError(f"JSON 中找不到 scene={scene_id}")
    if args.object_idx < 0 or args.object_idx >= len(scene_items):
        names = [s.get('object', 'Unknown') for _, s in scene_items]
        raise ValueError(
            f"scene={scene_id} 的 object_idx={args.object_idx} 超出范围 "
            f"(0..{len(scene_items)-1}); objects={names}"
        )
    return scene_items[args.object_idx]


def _normalize_name(value):
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _singularize_token(token):
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 1:
        return token[:-1]
    return token


def _singularize_name(value):
    return " ".join(_singularize_token(tok) for tok in _normalize_name(value).split())


def _names_match(a, b):
    if a is None or b is None:
        return None
    na, nb = _normalize_name(a), _normalize_name(b)
    if not na or not nb:
        return None
    sa, sb = _singularize_name(a), _singularize_name(b)
    return (
        na == nb or na in nb or nb in na or
        sa == sb or sa in sb or sb in sa
    )


def parse_execute_success(response, gt_object=None):
    """
    解析 OmniGibson Server 的 EXECUTE 返回。
    Server 只返回实际锁定到的 grasped_object；
    是否成功由客户端用 JSON 中的 gt_object 判断。
    """
    if not isinstance(response, dict):
        return False, "invalid response"

    grasped_object = response.get('grasped_object')
    if gt_object and grasped_object:
        object_match = _names_match(grasped_object, gt_object)
        if object_match is False:
            return False, f"object mismatch: got={grasped_object}, gt={gt_object}"
        return True, f"matched gt_object={gt_object}"

    return False, "no grasped object returned"


def load_scene(server_socket, scene_id):
    """请求 server 加载指定场景。"""
    if scene_id is None:
        return False
    print(f"\n[Scene] 请求 Server 加载场景: {scene_id}")
    server_socket.send_pyobj({'type': 'LOAD_SCENE', 'scene_id': str(scene_id)})
    res = server_socket.recv_pyobj()
    status = str(res.get('status', ''))
    print(f"  -> Server 结果: {status}")
    return "SUCCESS" in status.upper()


def start_instruction_video(args, server_socket, sample, instruction_idx):
    if not (args.auto and args.record_video):
        return None
    video_meta = {
        'scene_id': sample.get('scene'),
        'object_name': sample.get('object', 'object'),
        'instruction_idx': instruction_idx,
    }
    server_socket.send_pyobj({
        'type': 'START_VIDEO',
        'video_dir': args.server_video_dir,
        'video_meta': video_meta,
        'video_fps': args.video_fps,
        'video_frame_stride': args.video_frame_stride,
    })
    res = server_socket.recv_pyobj()
    print(f"  -> Server 视频录制: {res.get('status')}, path={res.get('video_path')}")
    return res.get('video_path')


def stop_instruction_video(args, server_socket):
    if not (args.auto and args.record_video):
        return None
    server_socket.send_pyobj({'type': 'STOP_VIDEO'})
    res = server_socket.recv_pyobj()
    print(f"  -> Server 视频结束: {res.get('status')}, path={res.get('video_path')}")
    return res.get('video_path')


def run_one_attempt(
    args, server_socket, graspnet_socket, sam_socket,
    vigor_model, tokenizer, clip_proc, transform,
    instruction, sample, attempt_idx,
):
    """完成一次观察-分割-抓取-执行闭环。返回是否成功和 server response。"""
    eval_object = sample.get('object') if sample else None

    print(f"\n--- Attempt {attempt_idx}/{args.max_attempts}: 获取仿真观测 ---")
    server_socket.send_pyobj({'type': 'GET_OBS'})
    obs = server_socket.recv_pyobj()
    rgb       = obs['rgb']
    depth     = obs['depth']
    depth_key = obs.get('depth_key', 'depth_linear')
    intrinsic = obs['intrinsic']
    cam_pos   = obs['cam_pos']
    cam_quat  = obs['cam_quat']

    print("\n--- SAM 生成候选掩码 ---")
    sam_masks = generate_sam_masks(sam_socket, rgb)
    if len(sam_masks) == 0:
        torch.cuda.empty_cache()
        return False, {'status': 'FAIL', 'message': 'SAM 未生成候选掩码'}

    print("\n--- VIGOR 选择 affordance ---")
    affordance_mask, _ = vigor_predict_affordance(
        vigor_model, tokenizer, clip_proc, transform,
        rgb, instruction, sam_masks, args.precision
    )
    save_debug_images(rgb, depth, affordance_mask, args.vis_dir)

    print("\n--- 发送给 GraspNet Service ---")
    graspnet_socket.send_pyobj({
        'rgb':             rgb,
        'depth':           depth,
        'depth_key':       depth_key,
        'affordance_mask': affordance_mask,
        'intrinsic':       intrinsic,
        'cam_pos':         cam_pos,
        'cam_quat':        cam_quat,
    })
    grasp_result = graspnet_socket.recv_pyobj()
    if grasp_result['status'] != 'SUCCESS':
        print(f"  -> GraspNet 失败: {grasp_result.get('message', '')}")
        torch.cuda.empty_cache()
        return False, grasp_result

    cam_translation = np.array(grasp_result['translation'])
    cam_rotation    = np.array(grasp_result['rotation'])
    grasp_width     = grasp_result['width']
    grasp_score     = grasp_result['score']

    world_pos, world_rot = transform_to_world(
        cam_translation, cam_rotation, cam_pos, cam_quat
    )
    world_euler = R.from_matrix(world_rot).as_euler('xyz', degrees=True)
    print(f"  -> World Position:  {world_pos}")
    print(f"  -> World Euler(°):  {world_euler}")
    print(f"  -> Gripper Width:   {grasp_width:.4f}")
    print(f"  -> Grasp Score:     {grasp_score:.4f}")

    execute_msg = {
        'type': 'EXECUTE',
        'translation': world_pos.tolist(),
        'rotation': world_rot.tolist(),
        'width': float(grasp_width),
    }

    print("\n--- 发送抓取指令给 Server ---")
    server_socket.send_pyobj(execute_msg)
    response = server_socket.recv_pyobj()
    success, reason = parse_execute_success(response, eval_object)
    print(f"  -> Server grasped_object: {response.get('grasped_object')}")
    print(f"  -> Success: {success} ({reason})")
    torch.cuda.empty_cache()
    return success, response


def run_instruction_trial(
    args, server_socket, graspnet_socket, sam_socket,
    vigor_model, tokenizer, clip_proc, transform,
    sample_idx, sample, instruction_idx, instruction,
):
    """对一条 instruction 最多连续尝试 args.max_attempts 步。"""
    scene_id = sample.get('scene')
    object_name = sample.get('object', 'Unknown')
    gt_object = object_name

    print("\n" + "=" * 80)
    print(
        f"[Trial] sample={sample_idx}, scene={scene_id}, object={object_name}, "
        f"gt_object={gt_object}, instruction_idx={instruction_idx}"
    )
    print(f"        instruction: \"{instruction}\"")

    success = False
    response = None
    steps = 0
    video_path = start_instruction_video(args, server_socket, sample, instruction_idx)
    try:
        for attempt_idx in range(1, args.max_attempts + 1):
            steps = attempt_idx
            success, response = run_one_attempt(
                args, server_socket, graspnet_socket, sam_socket,
                vigor_model, tokenizer, clip_proc, transform,
                instruction, sample, attempt_idx,
            )
            if success:
                break
    finally:
        stopped_video_path = stop_instruction_video(args, server_socket)
        if stopped_video_path:
            video_path = stopped_video_path

    print(f"[Trial Done] success={success}, steps={steps}")
    return {
        'sample_idx': sample_idx,
        'scene': scene_id,
        'object': object_name,
        'gt_object': gt_object,
        'instruction_idx': instruction_idx,
        'instruction': instruction,
        'success': success,
        'steps': steps,
        'video_path': video_path,
        'response': response,
    }


def summarize_results(results):
    total = len(results)
    success_results = [r for r in results if r['success']]
    success_count = len(success_results)
    total_steps = sum(r['steps'] for r in results)
    success_steps = sum(r['steps'] for r in success_results)

    return {
        'total_instructions': total,
        'success_count': success_count,
        'success_rate': success_count / total if total else 0.0,
        'avg_steps': total_steps / total if total else 0.0,
        'avg_success_steps': success_steps / success_count if success_count else 0.0,
    }


def print_summary(summary):
    print("\n" + "=" * 80)
    print("[Summary]")
    print(f"  Total instructions: {summary['total_instructions']}")
    print(f"  Success count:      {summary['success_count']}")
    print(f"  Success rate:       {summary['success_rate']:.4f}")
    print(f"  Avg steps:          {summary['avg_steps']:.4f}")
    print(f"  Avg success steps:  {summary['avg_success_steps']:.4f}")
    print("=" * 80)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_results(results_path, results, summary):
    if not results_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(
            _json_safe({'summary': summary, 'results': results}),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  -> 结果已保存: {results_path}")


def main():
    args = parse_args()
    if args.max_attempts <= 0:
        raise ValueError("--max_attempts 必须大于 0")

    samples = load_dataset_samples(args.dataset_path) if args.dataset_path else []
    if args.dataset_path and not samples:
        return

    device_vigor = torch.device(args.vigor_device)

    print("\n" + "=" * 80)
    print(">>> VIGOR 主控客户端 (Dual-GPU Mode) <<<")
    print(f"    VIGOR Device: {args.vigor_device}")
    print(f"    Mode: {'AUTO' if args.auto else 'MANUAL'}")
    print(f"    Max Attempts: {args.max_attempts}")
    print("=" * 80)

    print("\n[1/1] 初始化 VIGOR (LLMSeg)...")
    vigor_model, tokenizer, clip_proc, transform, _ = init_vigor(args, device_vigor)

    ctx = zmq.Context()
    server_socket = ctx.socket(zmq.REQ)
    server_socket.connect(f"tcp://{args.server_ip}:{args.server_port}")
    print(f"\n-> 已连接 OmniGibson Server ({args.server_ip}:{args.server_port})")

    graspnet_socket = ctx.socket(zmq.REQ)
    graspnet_socket.connect(f"tcp://{args.graspnet_ip}:{args.graspnet_port}")
    print(f"-> 已连接 GraspNet Service ({args.graspnet_ip}:{args.graspnet_port})")

    sam_socket = ctx.socket(zmq.REQ)
    sam_socket.connect(f"tcp://{args.sam_ip}:{args.sam_port}")
    print(f"-> 已连接 SAM Service ({args.sam_ip}:{args.sam_port})")

    try:
        if args.auto:
            results = []
            for sample_idx, sample in enumerate(samples):
                scene_id_int = _scene_id_as_int(sample)
                if args.max_scene_id is not None:
                    if scene_id_int is None:
                        print(f"  [Skip] sample={sample_idx} scene id 无法转成整数: {sample.get('scene')}")
                        continue
                    if scene_id_int > args.max_scene_id:
                        continue

                instructions = _safe_instructions(sample)
                if not instructions:
                    print(f"  [Skip] sample={sample_idx} 没有 instructions")
                    continue

                for instruction_idx, instruction in enumerate(instructions):
                    # 每条 instruction 独立评估；重新加载场景，避免上一次抓取改变场景状态。
                    if not load_scene(server_socket, sample.get('scene')):
                        results.append({
                            'sample_idx': sample_idx,
                            'scene': sample.get('scene'),
                            'object': sample.get('object', 'Unknown'),
                            'gt_object': sample.get('gt_object', sample.get('object', 'Unknown')),
                            'instruction_idx': instruction_idx,
                            'instruction': instruction,
                            'success': False,
                            'steps': 0,
                            'response': {'status': 'LOAD_SCENE_FAIL'},
                        })
                        continue

                    result = run_instruction_trial(
                        args, server_socket, graspnet_socket, sam_socket,
                        vigor_model, tokenizer, clip_proc, transform,
                        sample_idx, sample, instruction_idx, instruction,
                    )
                    results.append(result)

            summary = summarize_results(results)
            print_summary(summary)
            save_results(args.results_path, results, summary)
        else:
            if samples:
                sample_idx, sample = select_manual_sample(samples, args)
                instructions = _safe_instructions(sample)
                if not instructions and not args.instruction:
                    raise ValueError(f"sample={sample_idx} 没有可用 instruction")
                if args.instruction:
                    instruction_list = [args.instruction]
                    start_idx = 0
                else:
                    if args.instruction_idx < 0 or args.instruction_idx >= len(instructions):
                        raise ValueError(
                            f"instruction_idx={args.instruction_idx} 超出范围 "
                            f"(0..{len(instructions)-1})"
                        )
                    instruction_list = instructions
                    start_idx = args.instruction_idx
            else:
                sample_idx = -1
                sample = {'scene': None, 'object': 'object', 'gt_object': None}
                instruction_list = [args.instruction if args.instruction else "pick up the object"]
                start_idx = 0

            all_results = []
            current_idx = start_idx
            attempt_idx = 1

            # 加载初始场景
            if sample.get('scene') is not None:
                if not load_scene(server_socket, sample.get('scene')):
                    print("  -> 场景加载失败，退出")
                    current_idx = len(instruction_list)

            while current_idx < len(instruction_list):
                instruction_idx = current_idx if not args.instruction else -1
                instruction = instruction_list[current_idx]

                print("\n[Manual Preview]")
                print(f"  sample_idx:       {sample_idx}")
                print(f"  scene:            {sample.get('scene')}")
                print(f"  object:           {sample.get('object')}")
                print(f"  gt_object:        {sample.get('gt_object')}")
                print(f"  instruction_idx:  {instruction_idx}  ({current_idx + 1}/{len(instruction_list)} 条指令)")
                print(f"  instruction:      \"{instruction}\"")
                print(f"  attempt:          {attempt_idx}/{args.max_attempts}")
                user_input = input("[Ready] 回车执行一次 / n 下一条指令 / e 退出: ").strip().lower()
                if user_input == 'e':
                    break
                if user_input == 'n':
                    current_idx += 1
                    attempt_idx = 1
                    if current_idx < len(instruction_list) and sample.get('scene') is not None:
                        load_scene(server_socket, sample.get('scene'))
                    continue

                success, response = run_one_attempt(
                    args, server_socket, graspnet_socket, sam_socket,
                    vigor_model, tokenizer, clip_proc, transform,
                    instruction, sample, attempt_idx,
                )
                result_entry = {
                    'sample_idx': sample_idx,
                    'scene': sample.get('scene'),
                    'object': sample.get('object'),
                    'gt_object': sample.get('object'),
                    'instruction_idx': instruction_idx,
                    'instruction': instruction,
                    'success': success,
                    'steps': attempt_idx,
                    'response': response,
                }
                all_results.append(result_entry)
                print(f"  -> {'success' if success else 'fail'} (attempt {attempt_idx})")

                if success or attempt_idx >= args.max_attempts:
                    print_summary(summarize_results([result_entry]))
                    current_idx += 1
                    attempt_idx = 1
                    if current_idx < len(instruction_list) and sample.get('scene') is not None:
                        load_scene(server_socket, sample.get('scene'))
                else:
                    attempt_idx += 1

            if all_results:
                print("\n[All Results Summary]")
                print_summary(summarize_results(all_results))

    except KeyboardInterrupt:
        print("\n-> 用户中断")
    finally:
        server_socket.close()
        graspnet_socket.close()
        sam_socket.close()


if __name__ == "__main__":
    main()