"""
LLMSeg (LISA-based) 单张图片推理脚本

使用 finetune_llmseg_vigor_simple.py 训练的模型进行单张图片推理

【重要】LLMSeg 的推理逻辑与 GLOVER 不同：
- GLOVER: 直接生成像素级 mask
- LLMSeg: 从 SAM 候选 mask 中选择相似度最高的

因此推理时需要：
1. 提供 SAM 候选 mask 目录
2. 模型计算每个候选 mask 与指令的相似度
3. 选择相似度最高的候选 mask 作为预测结果

使用方法:
    python infer_single_llmseg.py --image /path/to/image.png --instruction "segment the cup" --sam_masks_dir /path/to/sam_masks
"""

import argparse
import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG


DEFAULT_IMAGE_TOKEN = "<image>"


def parse_args():
    parser = argparse.ArgumentParser(description="LLMSeg 单张图片推理")
    
    # 模型路径
    parser.add_argument("--version", default="/opt/data/private/model/LISA_Plus_7b",
                        type=str, help="LISA 基础模型路径")
    parser.add_argument("--checkpoint", default="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple/ckpt_model",
                        type=str, help="微调后的 checkpoint 路径")
    parser.add_argument("--vision_tower", default="/opt/data/private/model/clip-vit-large-patch14",
                        type=str, help="CLIP 模型路径")
    parser.add_argument("--vision_pretrained", default="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth",
                        type=str, help="SAM ViT-H 权重路径")
    
    # 输入
    parser.add_argument("--image", required=True, type=str, help="输入图片路径")
    parser.add_argument("--instruction", required=True, type=str, help="分割指令")
    parser.add_argument("--sam_masks_dir", required=True, type=str, 
                        help="SAM 候选 mask 目录 (必需)")
    parser.add_argument("--gt_mask", default=None, type=str,
                        help="GT mask 路径 (可选，用于对比可视化)")
    
    # 输出
    parser.add_argument("--output_dir", default="./infer_output", type=str, help="输出目录")
    
    # 模型配置
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"], help="推理精度")
    parser.add_argument("--image_size", default=896, type=int, help="图像尺寸")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    
    return parser.parse_args()


def load_model(args):
    """加载 LLMSeg 模型"""
    print("\n" + "=" * 60)
    print("  加载模型...")
    print("=" * 60)
    
    # 创建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    
    # 确定精度
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    # 加载模型
    model_args = {
        "train_mask_decoder": False,
        "out_dim": 256,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "mm_vision_tower": args.vision_tower,
    }
    
    print(f"  基础模型: {args.version}")
    # 注意：使用 low_cpu_mem_usage=False 避免创建 meta tensor
    model = LISAForCausalLM.from_pretrained(
        args.version, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=False,
        **model_args
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 初始化视觉模块
    class ModelArgs:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
            self.pretrain_mm_mlp_adapter = None
            self.vision_tower = args.vision_tower
    
    model_args_obj = ModelArgs()
    model.get_model().initialize_vision_modules(model_args_obj)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device="cuda")
    model.get_model().initialize_lisa_modules(model.get_model().config)
    
    model.resize_token_embeddings(len(tokenizer))
    
    # 加载微调权重
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  微调权重目录: {args.checkpoint}")
        
        # 检查 meta_log 文件获取训练信息
        checkpoint_parent = os.path.dirname(args.checkpoint)
        try:
            meta_files = [f for f in os.listdir(checkpoint_parent) if f.startswith("meta_log_")]
            if meta_files:
                meta_file = meta_files[-1]
                print(f"  📋 找到训练元信息: {meta_file}")
                import re
                match = re.search(r'giou([\d.]+)_ciou([\d.]+)', meta_file)
                if match:
                    print(f"     gIoU: {match.group(1)}, cIoU: {match.group(2)}")
        except Exception:
            pass
        
        lora_path = os.path.join(args.checkpoint, "lora_adapter")
        if not os.path.exists(lora_path):
            print(f"  [信息] 尝试从 DeepSpeed checkpoint 加载...")
            
            step_dirs = [d for d in os.listdir(args.checkpoint) 
                        if os.path.isdir(os.path.join(args.checkpoint, d)) and d.startswith("global_step")]
            
            if step_dirs:
                step_dirs.sort(key=lambda x: int(x.replace("global_step", "")))
                latest_step_dir = step_dirs[-1]
                global_step = int(latest_step_dir.replace("global_step", ""))
                
                item_path = os.path.join(args.checkpoint, latest_step_dir)
                mp_rank_path = os.path.join(item_path, "mp_rank_00_model_states.pt")
                
                if os.path.exists(mp_rank_path):
                    print(f"  📁 找到 DeepSpeed checkpoint: {latest_step_dir}")
                    print(f"     Global Step: {global_step}")
                    
                    state_dict = torch.load(mp_rank_path, map_location="cpu")
                    
                    if "epoch" in state_dict:
                        print(f"     Epoch: {state_dict['epoch']}")
                    if "global_steps" in state_dict:
                        print(f"     Saved Global Steps: {state_dict['global_steps']}")
                    
                    if "module" in state_dict:
                        module_state = state_dict["module"]
                        print(f"     参数数量: {len(module_state)}")
                        model.load_state_dict(module_state, strict=False)
                        print(f"  ✅ 从 DeepSpeed checkpoint 加载成功")
            else:
                print(f"  [警告] 未找到 global_step 目录")
        else:
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"  ✅ LoRA 适配器加载成功: {lora_path}")
    else:
        print(f"  [警告] 未找到微调权重: {args.checkpoint}")
    
    # 将整个模型转换为指定精度并移到 GPU
    # 这样可以确保所有组件（DINOv2, lisa_dino_conv, lisa_attention_layers 等）都使用相同精度
    model = model.to(dtype=torch_dtype, device="cuda")
    model.eval()
    print(f"  ✅ 模型精度已转换为: {torch_dtype}")
    
    print(f"  ✅ 模型加载完成")
    
    # CLIP 图像处理器
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    
    # SAM transform
    transform = ResizeLongestSide(args.image_size)
    
    return model, tokenizer, clip_image_processor, transform, seg_token_idx


def preprocess_image(image_np, transform, image_size):
    """图像预处理"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    image_resized = transform.apply_image(image_np)
    resize = image_resized.shape[:2]
    
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float()
    image_tensor = (image_tensor - pixel_mean) / pixel_std
    
    h, w = image_tensor.shape[-2:]
    padh = image_size - h
    padw = image_size - w
    image_tensor = F.pad(image_tensor, (0, padw, 0, padh))
    
    return image_tensor, resize


def prepare_sam_masks(sam_mask_helper, img_name, image_size, transform, precision):
    """准备 SAM 候选 mask"""
    segs_dict = sam_mask_helper.extract_sam_segs(img_name)
    
    segs_origin = segs_dict["segs_origin"]  # (H, W, K)
    segs_square = segs_dict["segs_square"]
    
    if segs_origin is None or segs_origin.shape[2] == 0:
        print(f"  [警告] 没有找到候选 mask: {img_name}")
        return None, None
    
    # 使用 ResizeLongestSide 处理候选 mask
    K = segs_origin.shape[2]
    segs_resized_list = []
    for k in range(K):
        mask_k = segs_origin[:, :, k]
        mask_k_uint8 = (mask_k * 255).astype(np.uint8)
        mask_k_resized = transform.apply_image(mask_k_uint8)
        mask_k_resized = (mask_k_resized.astype(np.float32) / 255.0)
        segs_resized_list.append(mask_k_resized)
    
    resized_h, resized_w = segs_resized_list[0].shape
    segs_resized = np.stack(segs_resized_list, axis=2)
    
    # Pad 到正方形
    h2, w2, _ = segs_resized.shape
    padh = image_size - h2
    padw = image_size - w2
    segs_square = np.pad(
        segs_resized,
        ((0, padh), (0, padw), (0, 0)),
        mode="constant",
        constant_values=1,
    )
    
    # 转换为 tensor 并插值到 256x256
    segs_tensor = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous()
    segs = F.interpolate(
        segs_tensor.unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    
    # 转换精度
    if precision == "bf16":
        segs = segs.bfloat16()
    elif precision == "fp16":
        segs = segs.half()
    
    return segs, segs_origin


def build_input_dict(
    image_np, image_clip, image_tensor, resize, 
    segs, segs_origin, instruction, tokenizer, args
):
    """构建模型输入字典"""
    # 构建对话
    question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
    
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], "[SEG]")
    prompt = conv.get_prompt()
    
    # 占空 IoU
    K = segs.shape[0]
    dummy_ious = np.zeros((1, K))
    dummy_iops = np.zeros((1, K))
    
    ori_size = image_np.shape[:2]
    
    input_dict = {
        "images": image_tensor.unsqueeze(0),
        "images_clip": image_clip.unsqueeze(0),
        "input_ids": None,  # 将在下面处理
        "labels": None,
        "attention_masks": None,
        "offset": torch.tensor([0, 1]),
        "masks_list": [torch.zeros(1, ori_size[0], ori_size[1])],
        "label_list": [torch.ones(ori_size[0], ori_size[1]) * 255],
        "resize_list": [resize],
        "sam_segs_list": [segs],
        "sam_ious_list": [dummy_ious],
        "sam_iops_list": [dummy_iops],
        "origin_segs_list": [segs_origin],
        "inference": True,
    }
    
    # Tokenize
    from model.llava.mm_utils import tokenizer_image_token
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0)
    
    # 处理 attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # 处理 labels
    labels = input_ids.clone()
    
    input_dict["input_ids"] = input_ids
    input_dict["labels"] = labels
    input_dict["attention_masks"] = attention_mask
    
    return input_dict


def visualize_result(image_np, pred_mask, gt_mask, output_dir, image_path, instruction, pred_mask_idx=None, similarity_score=None):
    """可视化结果 - 左侧 GT mask（绿色），右侧预测 mask（红色）"""
    h, w = image_np.shape[:2]
    
    # 调整 pred_mask 到原图大小
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 调整 gt_mask 到原图大小
    if gt_mask is not None and gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 归一化 mask 到 0-1 范围
    if pred_mask.max() > 1:
        pred_mask = pred_mask.astype(np.float32) / 255.0
    if gt_mask is not None and gt_mask.max() > 1:
        gt_mask = gt_mask.astype(np.float32) / 255.0
    
    # VIGOR mask: 0 = 前景 (掩码区域), 非0 = 背景
    pred_mask_region = (pred_mask == 0)
    gt_mask_region = (gt_mask == 0) if gt_mask is not None else np.zeros((h, w), dtype=bool)
    
    # 创建 GT 叠加图（绿色）
    gt_overlay = image_np.copy()
    if gt_mask is not None:
        gt_overlay[gt_mask_region] = (
            image_np[gt_mask_region] * 0.5 + 
            np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)
    
    # 创建预测叠加图（红色）
    pred_overlay = image_np.copy()
    pred_overlay[pred_mask_region] = (
        image_np[pred_mask_region] * 0.5 + 
        np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)
    
    # 拼接图像：左侧 GT，右侧预测
    combined_image = np.hstack([gt_overlay, pred_overlay])
    
    # 计算 IoU
    if gt_mask is not None:
        intersection = np.logical_and(pred_mask_region, gt_mask_region).sum()
        union = np.logical_or(pred_mask_region, gt_mask_region).sum()
        iou = intersection / (union + 1e-8)
    else:
        iou = 0.0
    
    # 添加文字标题
    title_text = f"Instruction: {instruction[:60]}..." if len(instruction) > 60 else f"Instruction: {instruction}"
    info_text = f"IoU: {iou:.4f}"
    if pred_mask_idx is not None:
        info_text += f" | Pred Mask Idx: {pred_mask_idx}"
    if similarity_score is not None:
        info_text += f" | Similarity: {similarity_score:.4f}"
    
    # 在图像上添加文字
    text_height = 80
    combined_image_with_text = np.ones((combined_image.shape[0] + text_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
    combined_image_with_text[text_height:, :] = combined_image
    
    # 绘制文字
    cv2.putText(combined_image_with_text, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(combined_image_with_text, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(combined_image_with_text, "GT (Green)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
    cv2.putText(combined_image_with_text, "Pred (Red)", (w + 10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 保存结果
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    mask_path = os.path.join(output_dir, f"{image_name}_{timestamp}_mask.png")
    combined_path = os.path.join(output_dir, f"{image_name}_{timestamp}_combined.png")
    
    # 保存 mask (0=前景, 255=背景)
    cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
    
    # 保存拼接图
    combined_bgr = cv2.cvtColor(combined_image_with_text, cv2.COLOR_RGB2BGR)
    cv2.imwrite(combined_path, combined_bgr)
    
    print(f"\n  ✅ 结果已保存:")
    print(f"     掩码: {mask_path}")
    print(f"     对比图: {combined_path}")
    
    # 打印掩码统计
    pred_fg_area = pred_mask_region.sum()
    gt_fg_area = gt_mask_region.sum() if gt_mask is not None else 0
    total_area = h * w
    print(f"\n  掩码统计:")
    print(f"     预测前景像素: {pred_fg_area} / {total_area} ({100*pred_fg_area/total_area:.2f}%)")
    if gt_mask is not None:
        print(f"     GT前景像素: {gt_fg_area} / {total_area} ({100*gt_fg_area/total_area:.2f}%)")
    print(f"     IoU: {iou:.4f}")


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查 SAM masks 目录
    if not os.path.exists(args.sam_masks_dir):
        print(f"  [错误] SAM 候选 mask 目录不存在: {args.sam_masks_dir}")
        print(f"  请使用 --sam_masks_dir 指定正确的目录")
        return
    
    # 创建 SAM mask helper
    sam_mask_helper = SAM_Mask_Reader_PNG(args.sam_masks_dir)
    transform = ResizeLongestSide(args.image_size)
    
    # 加载模型
    model, tokenizer, clip_image_processor, transform, seg_token_idx = load_model(args)
    
    # 读取图片
    print(f"\n  输入图片: {args.image}")
    print(f"  指令: {args.instruction}")
    print(f"  SAM masks: {args.sam_masks_dir}")
    
    image_np = cv2.imread(args.image)
    if image_np is None:
        print(f"  [Error] 无法读取图片: {args.image}")
        return
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    ori_size = image_np.shape[:2]
    
    # CLIP 图像处理
    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
    
    # SAM 图像处理
    image_tensor, resize = preprocess_image(image_np, transform, args.image_size)
    
    # 获取图片文件名用于查找 SAM mask
    img_name = os.path.basename(args.image)
    
    # 准备 SAM 候选 mask
    segs, segs_origin = prepare_sam_masks(
        sam_mask_helper, img_name, args.image_size, transform, args.precision
    )
    
    if segs is None:
        print(f"  [错误] 未找到对应的 SAM 候选 mask")
        return
    
    print(f"  候选 mask 数量: {segs.shape[0]}")
    
    # 构建输入
    input_dict = build_input_dict(
        image_np, image_clip, image_tensor, resize,
        segs, segs_origin, args.instruction, tokenizer, args
    )
    
    # 确定精度
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    # 移动到 GPU
    device = next(model.parameters()).device
    input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)
    
    # 推理
    print("\n  正在推理...")
    start_time = time.time()
    
    with torch.no_grad():
        output_dict = model(**input_dict)
    
    elapsed = time.time() - start_time
    print(f"  推理耗时: {elapsed:.2f}s")
    
    # 获取预测结果
    pred_similarity = output_dict["pred_similarity"][0]  # (1, K)
    max_idx = torch.argmax(pred_similarity).item()
    similarity_score = pred_similarity[0, max_idx].item()
    
    print(f"  选择的候选 mask 索引: {max_idx}")
    print(f"  相似度分数: {similarity_score:.4f}")
    
    # 获取预测 mask
    sam_segs = segs_origin  # (H, W, K)
    pred_mask = sam_segs[:, :, max_idx]  # (H, W)
    
    # 读取 GT mask（如果提供）
    gt_mask = None
    if args.gt_mask and os.path.exists(args.gt_mask):
        gt_mask = cv2.imread(args.gt_mask, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            print(f"  GT mask: {args.gt_mask}")
    
    # 可视化结果
    visualize_result(
        image_np, pred_mask, gt_mask, args.output_dir, args.image, args.instruction,
        pred_mask_idx=max_idx, similarity_score=similarity_score
    )


if __name__ == "__main__":
    main()
