"""
LLMSeg (LISA-based) VIGOR-100K 测试脚本

使用 finetune_llmseg_vigor_simple.py 训练的模型进行测试
计算 IC-IoU 和 ICR 指标，结果格式与 test_vigor.py 一致

【重要】LLMSeg 的推理逻辑与 GLOVER 不同：
- GLOVER: 直接生成像素级 mask
- LLMSeg: 从 SAM 候选 mask 中选择相似度最高的

使用方法:
    python test_llmseg_vigor.py --checkpoint /path/to/ckpt_model --sam_masks_dir /path/to/sam_masks
"""

import argparse
import os
import sys
import json
import time
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from peft import PeftModel
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

DEFAULT_AUTODL_TMP_DIR = os.environ.get("AUTODL_TMP_DIR", os.path.join("..", "root", "autodl-tmp"))
DEFAULT_MODEL_DIR = os.environ.get("MODEL_BASE_DIR", os.path.join(DEFAULT_AUTODL_TMP_DIR, "model"))
DEFAULT_VIGOR_DATA_DIR = os.environ.get("VIGOR_DATA_DIR", os.path.join(DEFAULT_AUTODL_TMP_DIR, "VIGOR-100K_new"))

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG


DEFAULT_IMAGE_TOKEN = "<image>"


def parse_args(args):
    parser = argparse.ArgumentParser(description="LLMSeg VIGOR-100K 测试脚本")
    
    # 模型路径
    parser.add_argument("--version", default=os.path.join(DEFAULT_MODEL_DIR, "LISA_Plus_7b"),
                        type=str, help="LISA 基础模型路径")
    parser.add_argument("--checkpoint", default=os.path.join(DEFAULT_AUTODL_TMP_DIR, "runs", "finetune_llmseg_vigor_simple-object", "ckpt_model"),
                        type=str, help="微调后的 checkpoint 路径")
    parser.add_argument("--vision_tower", default=os.path.join(DEFAULT_MODEL_DIR, "clip-vit-large-patch14"),
                        type=str, help="CLIP 模型路径")
    parser.add_argument("--vision_pretrained", default=os.path.join(DEFAULT_MODEL_DIR, "SAM-vit-h", "sam_vit_h_4b8939.pth"),
                        type=str, help="SAM ViT-H 权重路径")
    
    # 数据集路径
    parser.add_argument("--data_dir", default=os.path.join(DEFAULT_VIGOR_DATA_DIR, "test"),
                        type=str, help="测试数据集路径")
    parser.add_argument("--sam_masks_dir", default=os.path.join(DEFAULT_AUTODL_TMP_DIR, "test_mask", "sam_masks"), type=str,
                        help="SAM 候选 mask 目录")
    parser.add_argument("--depth_min", default=float(os.environ.get("VIGOR_DEPTH_MIN", "0.6")), type=float,
                        help="Depth normalization lower bound, same meaning as VIGOR_DEPTH_MIN in training")
    parser.add_argument("--depth_max", default=float(os.environ.get("VIGOR_DEPTH_MAX", "1.85")), type=float,
                        help="Depth normalization upper bound, same meaning as VIGOR_DEPTH_MAX in training")
    
    # 输出路径
    parser.add_argument("--output_dir", default=os.path.join(DEFAULT_AUTODL_TMP_DIR, "test_results"),
                        type=str, help="结果保存目录")
    
    # 模型配置
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"], help="推理精度")
    parser.add_argument("--image_size", default=896, type=int, help="图像尺寸")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    parser.add_argument("--device", default="cuda:0", type=str, help="使用的 GPU 设备")
    
    # LoRA 配置（需要与训练时一致）
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str,
                        help="LoRA target modules")
    
    # ICR 阈值配置
    parser.add_argument("--icr_thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        type=str, help="ICR 阈值列表，逗号分隔")
    
    # 可视化配置
    parser.add_argument("--vis_dir", default=os.path.join(DEFAULT_AUTODL_TMP_DIR, "test_vis_output"), type=str,
                        help="可视化输出目录")
    parser.add_argument("--save_vis", action="store_true", default=False,
                        help="是否保存可视化图片")
    
    # 调试
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max_samples", default=None, type=int,
                        help="最大测试样本数（用于调试）")
    parser.add_argument("--workers", default=4, type=int,
                        help="测试 DataLoader worker 数量")
    parser.add_argument("--iou_selection_only", action="store_true", default=False,
                        help="Use validate() style pred_similarity argmax mask selection; if false, use validate_threshold() style pred_iou threshold selection")
    parser.add_argument("--iou_threshold", default=0.5, type=float,
                        help="pred_iou threshold used when --iou_selection_only is not set")
    parser.add_argument("--split", default="both", type=str,
                        choices=["easy", "hard", "both"], help="测试的数据子集")
    
    return parser.parse_args(args)



def load_samples(data_dir: str, difficulty: str) -> List[Dict]:
    """加载测试样本"""
    json_candidates = [
        os.path.join(data_dir, f"open_vocab_grasp_{difficulty}_new_1.json"),
        os.path.join(data_dir, f"open_vocab_grasp_{difficulty}.json"),
    ]
    json_file = next((path for path in json_candidates if os.path.exists(path)), None)
    
    if json_file is None:
        print(f"  [警告] 文件不存在: {json_candidates[0]} 或 {json_candidates[1]}")
        return []
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "samples" in data:
        raw_samples = data["samples"]
    elif isinstance(data, list):
        raw_samples = data
    else:
        return []
    
    samples = []
    for sample in raw_samples:
        # 从 gt_mask_path 提取图片编号
        gt_mask_path_rel = sample.get('gt_mask_path', '')
        if not gt_mask_path_rel:
            continue
        
        mask_filename = os.path.basename(gt_mask_path_rel)
        match = re.match(r'^(\d+)_', mask_filename)
        if not match:
            continue
        
        img_num = match.group(1)
        img_name = f"{img_num}.png"
        
        image_path = os.path.join(data_dir, img_name)
        gt_mask_paths = [p.strip() for p in gt_mask_path_rel.split(',')]
        gt_mask_path = ",".join(os.path.join(data_dir, p) for p in gt_mask_paths)
        depth_path = os.path.join(data_dir, "depth", f"{img_num}.npy")
        
        gt_object = sample.get('gt_object', sample.get('object', ''))
        instructions = sample.get('instructions', [])
        
        if not os.path.exists(image_path):
            continue
        
        samples.append({
            'image_path': image_path,
            'gt_mask_path': gt_mask_path,
            'depth_path': depth_path,
            'gt_object': gt_object,
            'instructions': instructions,
            'difficulty': difficulty,
            'img_name': img_name,
        })
    
    return samples


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算两个二值 mask 的 IoU"""
    if pred_mask is None:
        return 0.0
    
    # VIGOR mask 语义: 0 = 前景, 1 = 背景
    pred_fg = (pred_mask == 0).astype(np.uint8)
    gt_fg = (gt_mask == 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def compute_icr_score(iou_pairs: List[float], threshold: float) -> int:
    """计算 ICR 分数"""
    return sum(1 for iou in iou_pairs if iou >= threshold)


def save_visualization(
    image_np: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    vis_dir: str,
    sample_info: dict,
    instruction_idx: int,
    iou: float,
    obj_count: int = 1
):
    """保存可视化结果 - 左侧 GT mask（绿色），右侧预测 mask（红色）
    
    Args:
        image_np: RGB 格式的原始图像
        pred_mask: 预测掩码
        gt_mask: GT 掩码
        vis_dir: 可视化输出目录
        sample_info: 包含 gt_object, difficulty, img_name 等信息
        instruction_idx: 指令索引 (0, 1, 2)
        iou: IoU 值
        obj_count: 同一图片中同一物体名称的实例计数
    """
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
    
    # 获取样本信息
    gt_object = sample_info.get('gt_object', 'Unknown')
    difficulty = sample_info.get('difficulty', 'unknown')
    instruction = sample_info.get('instructions', [''])[instruction_idx] if instruction_idx < len(sample_info.get('instructions', [])) else ''
    img_name = sample_info.get('img_name', 'unknown')
    
    # 添加文字标题区域
    text_height = 100
    combined_with_text = np.ones((combined_image.shape[0] + text_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
    combined_with_text[text_height:, :] = combined_image
    
    # 绘制文字
    # 第一行：GT Object 名称
    title_text = f"GT Object: {gt_object} | Difficulty: {difficulty}"
    cv2.putText(combined_with_text, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 第二行：指令（截断显示）
    instr_display = instruction[:80] + "..." if len(instruction) > 80 else instruction
    cv2.putText(combined_with_text, f"Instruction: {instr_display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 第三行：IoU 和标签说明
    info_text = f"IoU: {iou:.4f}"
    cv2.putText(combined_with_text, info_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(combined_with_text, "GT (Green)", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
    cv2.putText(combined_with_text, "Pred (Red)", (350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 构建保存路径，使用指令特定的命名
    # 格式: {difficulty}/{img_name}_{gt_object}_instr{idx}_iou{:.3f}.png
    difficulty_dir = os.path.join(vis_dir, difficulty)
    os.makedirs(difficulty_dir, exist_ok=True)
    
    # 清理 gt_object 名称中可能存在的特殊字符（用于文件名）
    gt_object_clean = gt_object.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_filename = f"{img_name}_{gt_object_clean}_{obj_count}_instr{instruction_idx}_iou{iou:.3f}.png"

    output_path = os.path.join(difficulty_dir, output_filename)
    
    # 保存拼接图
    combined_bgr = cv2.cvtColor(combined_with_text, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_bgr)
    
    return output_path


def load_model(args):
    """加载 LLMSeg (LISA) 模型
    
    关键：训练时使用了 LoRA，需要在加载权重前先初始化 LoRA 结构
    """
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
    model = LISAForCausalLM.from_pretrained(
        args.version, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=False,  # 避免 meta tensor 问题
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
    vision_tower.to(dtype=torch_dtype, device=args.device)
    model.get_model().initialize_lisa_modules(model.get_model().config)
    
    model.resize_token_embeddings(len(tokenizer))
    
    # ========== LoRA 初始化 ==========
    # 训练时使用了 LoRA，测试时也需要初始化同样的 LoRA 结构
    lora_r = getattr(args, 'lora_r', 8)  # 默认值与训练脚本一致
    if lora_r > 0:
        print(f"\n  📋 初始化 LoRA (r={lora_r})...")
        from peft import LoraConfig, get_peft_model
        
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "lisa_attention_layers",
                                "lisa_final_attn",
                                "lisa_norm_final_attn",
                                "lisa_iou_head",
                                "lisa_embedding_head",
                                "lisa_dino_conv",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        
        lora_target_modules = getattr(args, 'lora_target_modules', 'q_proj,v_proj').split(",")
        lora_alpha = getattr(args, 'lora_alpha', 16)
        lora_dropout = getattr(args, 'lora_dropout', 0.05)
        
        lora_target = find_linear_layers(model, lora_target_modules)
        print(f"     LoRA target modules: {len(lora_target)} layers")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"  ✅ LoRA 初始化完成")
    
    # 加载微调权重
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n  微调权重目录: {args.checkpoint}")
        
        # 检查 meta_log 文件
        checkpoint_parent = os.path.dirname(args.checkpoint)
        try:
            meta_files = [f for f in os.listdir(checkpoint_parent) if f.startswith("meta_log_")]
            if meta_files:
                meta_file = meta_files[-1]
                print(f"  📋 找到训练元信息: {meta_file}")
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
                    
                    if "module" in state_dict:
                        module_state = state_dict["module"]
                        print(f"     Checkpoint 参数数量: {len(module_state)}")
                        
                        # 打印当前模型的参数名称样例
                        model_param_names = list(model.state_dict().keys())
                        print(f"     当前模型参数数量: {len(model_param_names)}")
                        
                        # 打印几个 checkpoint 中的参数名称样例
                        ckpt_keys = list(module_state.keys())
                        print(f"\n     Checkpoint 参数名称样例 (前10个):")
                        for k in ckpt_keys[:10]:
                            print(f"       - {k}")
                        
                        print(f"\n     模型参数名称样例 (前10个):")
                        for k in model_param_names[:10]:
                            print(f"       - {k}")
                        
                        # 检查关键模块是否在 checkpoint 中
                        key_modules = ['lisa_dino_conv', 'lisa_attention_layers', 'lisa_embedding_head', 'lora']
                        print(f"\n     检查关键模块:")
                        for km in key_modules:
                            found = [k for k in ckpt_keys if km in k]
                            print(f"       {km}: 找到 {len(found)} 个参数")
                        
                        # 加载权重
                        missing_keys, unexpected_keys = model.load_state_dict(module_state, strict=False)
                        
                        print(f"\n     加载结果:")
                        print(f"       Missing keys: {len(missing_keys)}")
                        print(f"       Unexpected keys: {len(unexpected_keys)}")
                        
                        if len(missing_keys) > 0:
                            print(f"\n     Missing keys 样例 (前10个):")
                            for k in missing_keys[:10]:
                                print(f"       - {k}")
                        
                        if len(unexpected_keys) > 0:
                            print(f"\n     Unexpected keys 样例 (前10个):")
                            for k in unexpected_keys[:10]:
                                print(f"       - {k}")
                        
                        # 检查 LISA 模块是否正确加载
                        lisa_loaded = len([k for k in ckpt_keys if 'lisa_' in k and k not in unexpected_keys])
                        print(f"\n     LISA 模块加载: {lisa_loaded} 个参数")
                        
                        print(f"\n  ✅ 从 DeepSpeed checkpoint 加载成功")
                    else:
                        print(f"  [警告] checkpoint 中没有 'module' 键")
                        print(f"     可用的键: {list(state_dict.keys())}")
            else:
                print(f"  [警告] 未找到 global_step 目录")
        else:
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"  ✅ LoRA 适配器加载成功: {lora_path}")
    else:
        print(f"  [警告] 未找到微调权重: {args.checkpoint}")
    
    # 将整个模型转换为指定精度并移到设备
    model = model.to(dtype=torch_dtype, device=args.device)
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
    
    if segs_origin is None or len(segs_origin.shape) < 3 or segs_origin.shape[2] == 0:
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


def prepare_depth_map(depth_path: str, ori_size, image_size: int, transform, depth_min: float, depth_max: float):
    if depth_path and os.path.exists(depth_path):
        depth_raw = np.load(depth_path).astype(np.float32)
        if depth_raw.shape[:2] != ori_size:
            depth_raw = cv2.resize(depth_raw, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"Warning: depth not found: {depth_path}, using zeros")
        depth_raw = np.zeros(ori_size, dtype=np.float32)

    depth_raw = np.nan_to_num(depth_raw, nan=depth_max, posinf=depth_max, neginf=depth_min)
    denom = max(depth_max - depth_min, 1e-6)
    depth_norm = np.clip((depth_raw - depth_min) / denom, 0.0, 1.0).astype(np.float32)
    depth_target_hw = transform.get_preprocess_shape(ori_size[0], ori_size[1], image_size)
    depth_resized = cv2.resize(depth_norm, (depth_target_hw[1], depth_target_hw[0]), interpolation=cv2.INTER_LINEAR)
    padh_depth = image_size - depth_resized.shape[0]
    padw_depth = image_size - depth_resized.shape[1]
    depth_square = np.pad(depth_resized, ((0, padh_depth), (0, padw_depth)), mode="constant", constant_values=0)
    depth_256 = F.interpolate(
        torch.from_numpy(depth_square).unsqueeze(0).unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0).contiguous()
    return depth_256


class VIGORTestDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        sam_mask_helper,
        clip_image_processor,
        transform,
        image_size: int,
        precision: str,
        depth_min: float,
        depth_max: float,
    ):
        self.samples = samples
        self.sam_mask_helper = sam_mask_helper
        self.clip_image_processor = clip_image_processor
        self.transform = transform
        self.image_size = image_size
        self.precision = precision
        self.depth_min = depth_min
        self.depth_max = depth_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image_path = sample['image_path']
            gt_mask_path = sample['gt_mask_path']
            img_name = sample['img_name']
            depth_path = sample.get('depth_path')

            image_np = cv2.imread(image_path)
            if image_np is None:
                return {"error": f"Failed to load image: {image_path}", "sample": sample}
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            ori_size = image_np.shape[:2]

            gt_masks = []
            for p in [x.strip() for x in gt_mask_path.split(',') if x.strip()]:
                mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    gt_masks.append(mask)
            if not gt_masks:
                return {"error": f"Failed to load GT mask: {gt_mask_path}", "sample": sample}

            segs, segs_origin = prepare_sam_masks(
                self.sam_mask_helper, img_name, self.image_size, self.transform, self.precision
            )
            image_clip = self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
            image_tensor, resize = preprocess_image(image_np, self.transform, self.image_size)
            depth_256 = prepare_depth_map(
                depth_path, ori_size, self.image_size, self.transform, self.depth_min, self.depth_max
            )

            return {
                "sample": sample,
                "image_np": image_np,
                "gt_masks": gt_masks,
                "image_tensor": image_tensor,
                "image_clip": image_clip,
                "resize": resize,
                "segs": segs,
                "segs_origin": segs_origin,
                "depth_256": depth_256,
                "ori_size": ori_size,
            }
        except Exception as e:
            return {"error": repr(e), "sample": sample}


def collate_single_item(batch):
    return batch[0]


def predict_mask_llmseg_prepared(model, tokenizer, prepared: Dict, instruction: str, args):
    segs = prepared.get("segs")
    segs_origin = prepared.get("segs_origin")
    ori_size = prepared["ori_size"]
    if segs is None or segs_origin is None:
        return np.ones(ori_size, dtype=np.float32)

    question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], "[SEG]")
    prompt = conv.get_prompt()

    from model.llava.mm_utils import tokenizer_image_token
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    k = segs.shape[0]
    dummy_ious = np.zeros((1, k))
    dummy_iops = np.zeros((1, k))

    input_dict = {
        "images": prepared["image_tensor"].unsqueeze(0),
        "images_clip": prepared["image_clip"].unsqueeze(0),
        "input_ids": input_ids,
        "labels": labels,
        "attention_masks": attention_mask,
        "offset": torch.tensor([0, 1]),
        "masks_list": [torch.zeros(1, ori_size[0], ori_size[1])],
        "label_list": [torch.ones(ori_size[0], ori_size[1]) * 255],
        "resize_list": [prepared["resize"]],
        "sam_segs_list": [segs],
        "sam_ious_list": [dummy_ious],
        "sam_iops_list": [dummy_iops],
        "origin_segs_list": [segs_origin],
        "depths": [prepared["depth_256"]],
        "inference": True,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    device = next(model.parameters()).device
    input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)

    with torch.no_grad():
        output_dict = model(**input_dict)

    if args.iou_selection_only:
        pred_similarity = output_dict["pred_similarity"][0]
        max_idx = torch.argmax(pred_similarity).item()
        return segs_origin[:, :, max_idx]

    pred_iou = output_dict["pred_iou"][0]
    max_ids = []
    for i in range(pred_iou.shape[1]):
        if pred_iou[0][i] > args.iou_threshold:
            max_ids.append(i)

    pred_mask = np.ones_like(segs_origin[:, :, 0])
    for i in max_ids:
        pred_mask = np.minimum(pred_mask, segs_origin[:, :, i])
    return pred_mask.astype(np.uint8)


def evaluate_prepared_sample(model, tokenizer, prepared: Dict, args, obj_count: int = 1) -> Dict:
    if prepared.get("error"):
        if args.debug:
            sample = prepared.get("sample", {})
            print(f"  [Error] {sample.get('image_path', 'N/A')}: {prepared['error']}")
        return None

    sample = prepared["sample"]
    instructions = sample['instructions']
    gt_masks = prepared["gt_masks"]
    image_np = prepared["image_np"]
    original_h, original_w = image_np.shape[:2]

    pred_masks = []
    ic_ious = []
    for instr_idx, instruction in enumerate(instructions[:3]):
        pred_mask = predict_mask_llmseg_prepared(model, tokenizer, prepared, instruction, args)
        resized_pred_masks = []
        iou_candidates = []
        for gt_mask in gt_masks:
            pred_for_gt = pred_mask
            if pred_for_gt.shape != gt_mask.shape:
                pred_for_gt = cv2.resize(
                    pred_for_gt.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            resized_pred_masks.append(pred_for_gt)
            iou_candidates.append(compute_iou(pred_for_gt, gt_mask))

        best_gt_idx = int(np.argmax(iou_candidates)) if iou_candidates else 0
        pred_mask = resized_pred_masks[best_gt_idx] if resized_pred_masks else pred_mask
        gt_mask_for_vis = gt_masks[best_gt_idx] if gt_masks else np.zeros((original_h, original_w), dtype=np.uint8)

        pred_masks.append(pred_mask)
        iou = iou_candidates[best_gt_idx] if iou_candidates else 0.0
        ic_ious.append(iou)

        if args.save_vis:
            sample_info = {
                'gt_object': sample['gt_object'],
                'difficulty': sample['difficulty'],
                'instructions': instructions,
                'img_name': sample['img_name'],
            }
            save_visualization(
                image_np=image_np,
                pred_mask=pred_mask,
                gt_mask=gt_mask_for_vis,
                vis_dir=args.vis_dir,
                sample_info=sample_info,
                instruction_idx=instr_idx,
                iou=iou,
                obj_count=obj_count,
            )

    avg_ic_iou = np.mean(ic_ious) if ic_ious else 0.0
    iou_pairs = []
    if len(pred_masks) >= 3:
        iou_pairs.append(compute_iou(pred_masks[0], pred_masks[1]))
        iou_pairs.append(compute_iou(pred_masks[0], pred_masks[2]))
        iou_pairs.append(compute_iou(pred_masks[1], pred_masks[2]))
    elif len(pred_masks) == 2:
        iou_pairs.append(compute_iou(pred_masks[0], pred_masks[1]))

    return {
        'image_path': sample['image_path'],
        'gt_mask_path': sample['gt_mask_path'],
        'gt_object': sample['gt_object'],
        'difficulty': sample['difficulty'],
        'ic_ious': ic_ious,
        'avg_ic_iou': avg_ic_iou,
        'iou_pairs': iou_pairs,
        'num_instructions': len(instructions[:3]),
    }


def run_split_with_loader(split_name: str, samples: List[Dict], model, tokenizer, sam_mask_helper, clip_image_processor, transform, args):
    dataset = VIGORTestDataset(
        samples=samples,
        sam_mask_helper=sam_mask_helper,
        clip_image_processor=clip_image_processor,
        transform=transform,
        image_size=args.image_size,
        precision=args.precision,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_single_item,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(**loader_kwargs)

    result_list = []
    seen_counts = defaultdict(int)
    for prepared in tqdm(loader, desc=split_name.capitalize()):
        sample = prepared.get("sample", {})
        img_name = sample.get('img_name', '')
        obj_name = sample.get('gt_object', '')
        seen_counts[(img_name, obj_name)] += 1
        curr_count = seen_counts[(img_name, obj_name)]
        result = evaluate_prepared_sample(model, tokenizer, prepared, args, obj_count=curr_count)
        if result:
            result_list.append(result)
    return result_list


def main(args):
    args = parse_args(args)
    
    # 解析 ICR 阈值
    icr_thresholds = [float(t) for t in args.icr_thresholds.split(",")]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建可视化输出目录
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"\n  可视化输出目录: {args.vis_dir}")
    
    # 检查 SAM masks 目录
    if not os.path.exists(args.sam_masks_dir):
        print(f"  [错误] SAM 候选 mask 目录不存在: {args.sam_masks_dir}")
        return
    
    # 创建 SAM mask helper
    sam_mask_helper = SAM_Mask_Reader_PNG(args.sam_masks_dir)
    transform = ResizeLongestSide(args.image_size)
    
    # 加载模型
    model, tokenizer, clip_image_processor, transform, seg_token_idx = load_model(args)
    
    # 加载测试数据
    print("=" * 60)
    print("  加载测试数据...")
    print("=" * 60)
    
    print(f"  测试模式 (Split): {args.split}")
    
    easy_samples = []
    hard_samples = []
    
    if args.split in ["easy", "both"]:
        easy_samples = load_samples(args.data_dir, "easy")
    if args.split in ["hard", "both"]:
        hard_samples = load_samples(args.data_dir, "hard")
    
    if args.max_samples:
        easy_samples = easy_samples[:args.max_samples]
        hard_samples = hard_samples[:args.max_samples]
    
    print(f"  Easy 样本数: {len(easy_samples)}")
    print(f"  Hard 样本数: {len(hard_samples)}")
    print(f"  SAM 候选 mask 目录: {args.sam_masks_dir}")
    selection_mode = "pred_similarity_argmax" if args.iou_selection_only else f"pred_iou_threshold_{args.iou_threshold}"
    print(f"  Depth归一化: [{args.depth_min}, {args.depth_max}]")
    print(f"  Mask选择方式: {selection_mode}")
    
    # 存储结果
    results = {
        'easy': [],
        'hard': [],
    }
    
    print(f"  DataLoader workers: {args.workers}")

    # 测试 Easy 样本
    print("\n" + "=" * 60)
    print("  测试 Easy 样本")
    print("=" * 60)
    results['easy'] = run_split_with_loader(
        "easy", easy_samples, model, tokenizer, sam_mask_helper, clip_image_processor, transform, args
    )

    # 测试 Hard 样本
    print("\n" + "=" * 60)
    print("  测试 Hard 样本")
    print("=" * 60)
    results['hard'] = run_split_with_loader(
        "hard", hard_samples, model, tokenizer, sam_mask_helper, clip_image_processor, transform, args
    )

    # 计算统计指标
    print("\n" + "=" * 60)
    print("  计算统计指标")
    print("=" * 60)
    
    def compute_metrics(result_list: List[Dict], icr_thresholds: List[float]) -> Dict:
        if not result_list:
            return {
                'ic_iou': 0.0,
                'count': 0,
                'icr_scores': {t: 0.0 for t in icr_thresholds},
            }
        
        ic_ious = [r['avg_ic_iou'] for r in result_list]
        avg_ic_iou = np.mean(ic_ious)
        
        icr_scores = {t: [] for t in icr_thresholds}
        for r in result_list:
            if len(r['iou_pairs']) >= 3:
                for t in icr_thresholds:
                    score = compute_icr_score(r['iou_pairs'], t)
                    icr_scores[t].append(score)
        
        avg_icr_scores = {}
        for t in icr_thresholds:
            if icr_scores[t]:
                avg_icr_scores[t] = np.mean(icr_scores[t])
            else:
                avg_icr_scores[t] = 0.0
        
        return {
            'ic_iou': avg_ic_iou,
            'count': len(result_list),
            'icr_scores': avg_icr_scores,
        }
    
    easy_metrics = compute_metrics(results['easy'], icr_thresholds)
    hard_metrics = compute_metrics(results['hard'], icr_thresholds)
    
    # 计算总体平均值
    total_count = easy_metrics['count'] + hard_metrics['count']
    if total_count > 0:
        avg_ic_iou = (
            easy_metrics['ic_iou'] * easy_metrics['count'] +
            hard_metrics['ic_iou'] * hard_metrics['count']
        ) / total_count
        
        avg_icr_scores = {}
        for t in icr_thresholds:
            avg_icr_scores[t] = (
                easy_metrics['icr_scores'][t] * easy_metrics['count'] +
                hard_metrics['icr_scores'][t] * hard_metrics['count']
            ) / total_count
    else:
        avg_ic_iou = 0.0
        avg_icr_scores = {t: 0.0 for t in icr_thresholds}
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"llmseg_vigor_test_results_{timestamp}.txt")
    
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  LLMSeg VIGOR-100K 测试结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"模型路径: {args.version}\n")
        f.write(f"微调权重: {args.checkpoint}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"SAM masks: {args.sam_masks_dir}\n")
        f.write(f"Depth range: [{args.depth_min}, {args.depth_max}]\n")
        f.write(f"DataLoader workers: {args.workers}\n")
        f.write(f"Mask selection: {selection_mode}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 汇总表格
        f.write("=" * 80 + "\n")
        f.write("  汇总结果表格\n")
        f.write("=" * 80 + "\n\n")
        
        header = "| Category | Methods | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg |"
        for t in icr_thresholds:
            header += f" ICR@{t:.1f} Avg |"
        f.write(header + "\n")
        
        row = f"| Offline  | LLMSeg  | {easy_metrics['ic_iou']:.4f}      | {hard_metrics['ic_iou']:.4f}      | {avg_ic_iou:.4f}     |"
        for t in icr_thresholds:
            row += f" {avg_icr_scores[t]:.4f}      |"
        f.write(row + "\n\n")
        
        # 详细结果
        f.write("\n" + "=" * 80 + "\n")
        f.write("  总体 IC-IoU 和 ICR@0.X\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总样本数: {total_count}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n\n")
        
        f.write("IC-IoU:\n")
        f.write(f"  Easy:    {easy_metrics['ic_iou']:.4f}\n")
        f.write(f"  Hard:    {hard_metrics['ic_iou']:.4f}\n")
        f.write(f"  Average: {avg_ic_iou:.4f}\n\n")
        
        f.write("ICR@0.X:\n")
        for t in icr_thresholds:
            f.write(f"  ICR@{t:.1f}:\n")
            f.write(f"    Easy:    {easy_metrics['icr_scores'][t]:.4f}\n")
            f.write(f"    Hard:    {hard_metrics['icr_scores'][t]:.4f}\n")
            f.write(f"    Average: {avg_icr_scores[t]:.4f}\n")
        
        # 详细样本结果
        f.write("\n" + "=" * 80 + "\n")
        f.write("  Easy 样本详细结果\n")
        f.write("=" * 80 + "\n\n")
        
        for i, r in enumerate(results['easy']):
            f.write(f"[Easy Sample {i+1}]\n")
            f.write(f"  Image: {r['image_path']}\n")
            f.write(f"  GT Mask: {r['gt_mask_path']}\n")
            f.write(f"  Object: {r['gt_object']}\n")
            f.write(f"  IC-IoU (per instruction): {[f'{x:.4f}' for x in r['ic_ious']]}\n")
            f.write(f"  Avg IC-IoU: {r['avg_ic_iou']:.4f}\n")
            f.write(f"  IoU pairs: {[f'{x:.4f}' for x in r['iou_pairs']]}\n")
            for t in icr_thresholds:
                score = compute_icr_score(r['iou_pairs'], t)
                f.write(f"  ICR@{t:.1f}: {score}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("  Hard 样本详细结果\n")
        f.write("=" * 80 + "\n\n")
        
        for i, r in enumerate(results['hard']):
            f.write(f"[Hard Sample {i+1}]\n")
            f.write(f"  Image: {r['image_path']}\n")
            f.write(f"  GT Mask: {r['gt_mask_path']}\n")
            f.write(f"  Object: {r['gt_object']}\n")
            f.write(f"  IC-IoU (per instruction): {[f'{x:.4f}' for x in r['ic_ious']]}\n")
            f.write(f"  Avg IC-IoU: {r['avg_ic_iou']:.4f}\n")
            f.write(f"  IoU pairs: {[f'{x:.4f}' for x in r['iou_pairs']]}\n")
            for t in icr_thresholds:
                score = compute_icr_score(r['iou_pairs'], t)
                f.write(f"  ICR@{t:.1f}: {score}\n")
            f.write("\n")
    
    print(f"\n结果已保存到: {result_file}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("  测试完成 - 结果汇总")
    print("=" * 60)
    print(f"总样本数: {total_count}")
    print(f"IC-IoU: Easy={easy_metrics['ic_iou']:.4f}, Hard={hard_metrics['ic_iou']:.4f}, Avg={avg_ic_iou:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
