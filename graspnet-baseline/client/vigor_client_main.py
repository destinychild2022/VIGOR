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

    # SAM
    p.add_argument('--sam_checkpoint', type=str,
                   default='/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth')

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
    p.add_argument('--scene_idx', type=int, default=0,
                   help='运行数据集中的第几个样本 (0-indexed)')

    # 输出
    p.add_argument('--vis_dir', type=str,
                   default='/opt/data/private/LLMSeg/graspnet-baseline/vigor_grasp_vis')

    return p.parse_args()


def load_dataset_instruction(json_path, idx):
    """从 JSON 数据集中加载指令"""
    import json
    if not os.path.exists(json_path):
        print(f"  [Error] 数据集不存在: {json_path}")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    if idx >= len(samples):
        print(f"  [Error] 索引 {idx} 超出范围 (max={len(samples)-1})")
        return None
    
    sample = samples[idx]
    # 取第一个指令
    instr = sample['instructions'][0]
    scene_id = sample.get('scene', 'Unknown')
    obj_name = sample.get('object', 'Unknown')
    
    print(f"  -> 从数据集加载成功: Index={idx}, Scene={scene_id}, Object={obj_name}")
    return instr, scene_id


def main():
    args = parse_args()
    device_vigor = torch.device(args.vigor_device)

    print("\n" + "=" * 80)
    print(">>> VIGOR 主控客户端 (Dual-GPU Mode) <<<")
    print(f"    VIGOR Device: {args.vigor_device}")
    print("=" * 80)

    # --- 初始化模型 ---
    print("\n[1/1] 初始化 VIGOR (LLMSeg)...")
    vigor_model, tokenizer, clip_proc, transform, seg_idx = init_vigor(args, device_vigor)

    # --- 连接 OmniGibson Server ---
    ctx = zmq.Context()
    server_socket = ctx.socket(zmq.REQ)
    server_socket.connect(f"tcp://{args.server_ip}:{args.server_port}")
    print(f"\n-> 已连接 OmniGibson Server ({args.server_ip}:{args.server_port})")

    # --- 连接 GraspNet Service ---
    graspnet_socket = ctx.socket(zmq.REQ)
    graspnet_socket.connect(f"tcp://{args.graspnet_ip}:{args.graspnet_port}")
    print(f"-> 已连接 GraspNet Service ({args.graspnet_ip}:{args.graspnet_port})")

    # --- 连接 SAM Service ---
    sam_socket = ctx.socket(zmq.REQ)
    sam_socket.connect(f"tcp://{args.sam_ip}:{args.sam_port}")
    print(f"-> 已连接 SAM Service ({args.sam_ip}:{args.sam_port})")

    print("\n-> 所有模型就绪, 可以开始抓取实验")
    print("   ⚠️  请确保 graspnet_service.py 已在 graspnet conda 环境中启动!\n")

    current_idx = args.scene_idx
    last_scene_id = None
    try:
        while True:
            # ==============================================================
            # --- 0. 预先解析任务详情 ---
            # ==============================================================
            instruction = None
            scene_id = None
            if args.dataset_path:
                instruction, scene_id = load_dataset_instruction(args.dataset_path, current_idx)
            
            if instruction is None:
                instruction = args.instruction if args.instruction else "pick up the object"
            
            print(f"\n[任务预览] Index: {current_idx} | 场景 ID: {scene_id} | 指令: \"{instruction}\"")
            
            # --- 主交互提示 ---
            user_input = input(f"[Ready] 准备开始 (回车: 执行 / q: 重试上一个 / e: 退出): ").strip().lower()

            if user_input == 'e':
                print("  -> 退出实验循环")
                break
                
            if user_input == 'q':
                # 回退：回到上一个 index
                # 注意：如果本来就是 0，就保持 0
                current_idx = max(0, current_idx - 1)
                print(f"  << [动作] 正在回退到 Index={current_idx}")
                continue

            # ==============================================================
            # --- 1. 场景同步 (如果需要) ---
            # ==============================================================
            if scene_id is not None and scene_id != last_scene_id:
                print(f"\n[动作] 正在请求 Server 加载新场景: {scene_id}...")
                server_socket.send_pyobj({'type': 'LOAD_SCENE', 'scene_id': scene_id})
                res = server_socket.recv_pyobj()
                print(f"  -> Server 结果: {res['status']}")
                
                if "SUCCESS" in res['status']:
                    last_scene_id = scene_id
                    # 场景切换比较久，在这里加一个二次确认
                    print(f"\n[确认] 场景 {scene_id} 已加载完毕。")
                    confirm = input("   按回车继续，或输入 q 取消本次进度重来: ").strip().lower()
                    if confirm == 'q':
                        print(f"  << [动作] 本次 index={current_idx} 已取消。")
                        continue
                else:
                    print(f"  ❌ Server 换场失败，请检查 Server 端或 JSON 配置。")
                    current_idx = max(0, current_idx - 1)
                    continue
            
            print(f"  语言指令: \"{instruction}\"")
            
            # ==============================================================
            #  Step 1: 从 OmniGibson Server 获取 RGBD
            # ==============================================================
            print("\n--- Step 1: 获取仿真观测 ---")
            server_socket.send_pyobj({'type': 'GET_OBS'})
            obs = server_socket.recv_pyobj()
            rgb       = obs['rgb']         # (H, W, 3) uint8
            depth     = obs['depth']       # (H, W) float
            intrinsic = obs['intrinsic']   # (3, 3)
            cam_pos   = obs['cam_pos']     # (3,)
            cam_quat  = obs['cam_quat']    # (4,)

            # ==============================================================
            #  Step 2: SAM 生成候选掩码 (运行在独立服务中)
            # ==============================================================
            print("\n--- Step 2: SAM 生成候选掩码 ---")
            sam_masks = generate_sam_masks(sam_socket, rgb)

            if len(sam_masks) == 0:
                print("  ❌ SAM 未能生成候选掩码, 跳过")
                continue

            # ==============================================================
            #  Step 3: VIGOR 选择 affordance 掩码 (运行在 GPU 0)
            # ==============================================================
            print("\n--- Step 3: VIGOR 选择 affordance ---")
            
            affordance_mask, best_idx = vigor_predict_affordance(
                vigor_model, tokenizer, clip_proc, transform,
                rgb, instruction, sam_masks, args.precision
            )

            # 保存调试图像
            save_debug_images(rgb, depth, affordance_mask, args.vis_dir)

            # ==============================================================
            #  Step 4: 发送给 GraspNet Service
            # ==============================================================
            print("\n--- Step 4: 发送给 GraspNet Service ---")
            graspnet_socket.send_pyobj({
                'rgb':             rgb,
                'depth':           depth,
                'affordance_mask': affordance_mask,
                'intrinsic':       intrinsic,
            })
            grasp_result = graspnet_socket.recv_pyobj()

            if grasp_result['status'] != 'SUCCESS':
                print(f"  ❌ GraspNet 失败: {grasp_result.get('message', '')}")
                continue

            # GraspNet 返回的是相机坐标系下的位姿
            cam_translation = np.array(grasp_result['translation'])
            cam_rotation    = np.array(grasp_result['rotation'])
            grasp_width     = grasp_result['width']
            grasp_score     = grasp_result['score']

            # 转换到世界坐标系
            world_pos, world_rot = transform_to_world(
                cam_translation, cam_rotation, cam_pos, cam_quat
            )

            world_euler = R.from_matrix(world_rot).as_euler('xyz', degrees=True)
            print(f"  -> World Position:  {world_pos}")
            print(f"  -> World Euler(°):  {world_euler}")
            print(f"  -> Gripper Width:   {grasp_width:.4f}")
            print(f"  -> Grasp Score:     {grasp_score:.4f}")

            # ==============================================================
            #  Step 5: 发送抓取指令给 OmniGibson Server
            # ==============================================================
            print("\n--- Step 5: 发送抓取指令给 Server ---")
            server_socket.send_pyobj({
                'type': 'EXECUTE',
                'translation': world_pos.tolist(),
                'rotation': world_rot.tolist(),
                'width': float(grasp_width)
            })
            response = server_socket.recv_pyobj()
            print(f"  -> Server 响应: {response['status']}")
            print("=" * 60 + "\n")

            # --- 自动跳转下一个样本 ---
            current_idx += 1
            del obs, rgb, depth, sam_masks, affordance_mask, grasp_result
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n-> 用户中断")
    finally:
        server_socket.close()
        graspnet_socket.close()


if __name__ == "__main__":
    main()
