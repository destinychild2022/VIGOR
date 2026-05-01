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

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG


DEFAULT_IMAGE_TOKEN = "<image>"


def parse_args(args):
    parser = argparse.ArgumentParser(description="LLMSeg VIGOR-100K 测试脚本")
    
    # 模型路径
    parser.add_argument("--version", default="/opt/data/private/model/LISA_Plus_7b",
                        type=str, help="LISA 基础模型路径")
    parser.add_argument("--checkpoint", default="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple/ckpt_model",
                        type=str, help="微调后的 checkpoint 路径")
    parser.add_argument("--vision_tower", default="/opt/data/private/model/clip-vit-large-patch14",
                        type=str, help="CLIP 模型路径")
    parser.add_argument("--vision_pretrained", default="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth",
                        type=str, help="SAM ViT-H 权重路径")
    
    # 数据集路径
    parser.add_argument("--data_dir", default="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test",
                        type=str, help="测试数据集路径")
    parser.add_argument("--sam_masks_dir", required=True, type=str,
                        help="SAM 候选 mask 目录 (必需)")
    
    # 输出路径
    parser.add_argument("--output_dir", default="./result",
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
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,out_proj", type=str,
                        help="LoRA target modules")
    
    # ICR 阈值配置
    parser.add_argument("--icr_thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        type=str, help="ICR 阈值列表，逗号分隔")
    
    # 可视化配置
    parser.add_argument("--vis_dir", default="./vis_output", type=str,
                        help="可视化输出目录")
    parser.add_argument("--save_vis", action="store_true", default=False,
                        help="是否保存可视化图片")
    
    # 调试
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max_samples", default=None, type=int,
                        help="最大测试样本数（用于调试）")
    parser.add_argument("--split", default="both", type=str,
                        choices=["easy", "hard", "both"], help="测试的数据子集")
    parser.add_argument("--workers", default=12, type=int,
                        help="测试 DataLoader worker 数量")
    
    return parser.parse_args(args)



def load_samples(data_dir: str, difficulty: str) -> List[Dict]:
    """加载测试样本 (适配多目标解析)"""
    # 优先尝试加载 _new_1.json 高质量过滤版本，如果不存在则加载原版
    json_file = os.path.join(data_dir, f"open_vocab_grasp_{difficulty}_new_1.json")
    if not os.path.exists(json_file):
        json_file = os.path.join(data_dir, f"open_vocab_grasp_{difficulty}.json")
    
    if not os.path.exists(json_file):
        print(f"  [警告] 文件不存在: {json_file}")
        return []
    
    print(f"  [信息] 加载数据文件: {json_file}")
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
        # 支持解析逗号分隔的多路径
        gt_mask_path_raw = sample.get('gt_mask_path', '')
        if not gt_mask_path_raw:
            continue
        
        # 兼容多目标逻辑
        gt_mask_paths_rel = [p.strip() for p in gt_mask_path_raw.split(',') if p.strip()]
        if not gt_mask_paths_rel:
            continue
        
        # 获取图片编号（取第一个掩码文件名即可）
        first_mask_filename = os.path.basename(gt_mask_paths_rel[0])
        match = re.match(r'^(\d+)_', first_mask_filename)
        if not match:
            continue
        
        img_num = match.group(1)
        img_name = f"{img_num}.png"
        image_path = os.path.join(data_dir, img_name)
        
        # 转化为绝对路径列表
        gt_mask_paths = [os.path.join(data_dir, p) for p in gt_mask_paths_rel]
        
        gt_object = sample.get('gt_object', sample.get('object', ''))
        instructions = sample.get('instructions', [])
        
        if not os.path.exists(image_path):
            continue
        
        samples.append({
            'image_path': image_path,
            'gt_mask_paths': gt_mask_paths, # 返回列表
            'gt_object': gt_object,
            'instructions': instructions,
            'difficulty': difficulty,
            'img_name': img_name,
        })
    
    return samples


def numpy_to_torch(array: np.ndarray, torch_dtype=None) -> torch.Tensor:
    """Convert numpy arrays to torch tensors without torch's NumPy C-API bridge."""
    dtype_pairs = {
        np.dtype("uint8"): (torch.uint8, np.uint8),
        np.dtype("int32"): (torch.int32, np.int32),
        np.dtype("int64"): (torch.int64, np.int64),
        np.dtype("float16"): (torch.float16, np.float16),
        np.dtype("float32"): (torch.float32, np.float32),
        np.dtype("float64"): (torch.float64, np.float64),
        np.dtype("bool"): (torch.bool, np.bool_),
    }
    reverse_dtype = {v[0]: v[1] for v in dtype_pairs.values()}

    if torch_dtype is not None:
        np_dtype = reverse_dtype.get(torch_dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported target torch dtype: {torch_dtype}")
        array = array.astype(np_dtype, copy=False)

    array = np.ascontiguousarray(array)
    mapped = dtype_pairs.get(array.dtype)
    if mapped is None:
        raise TypeError(f"Unsupported numpy dtype: {array.dtype}")

    dtype = torch_dtype if torch_dtype is not None else mapped[0]
    return torch.frombuffer(array, dtype=dtype).reshape(array.shape)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算两个二值 mask 的 IoU"""
    if pred_mask is None or gt_mask is None:
        return 0.0
    
    # VIGOR mask 语义: 0 = 前景, 1 = 背景
    pred_fg = (pred_mask == 0)
    gt_fg = (gt_mask == 0)
    
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def compute_iou_max(pred_mask: np.ndarray, gt_masks: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """多目标择优：计算预测掩码与多个 GT 掩码中的最大 IoU"""
    if not gt_masks:
        return 0.0, None
    
    max_iou = -1.0
    best_gt = gt_masks[0]
    
    for gt in gt_masks:
        iou = compute_iou(pred_mask, gt)
        if iou > max_iou:
            max_iou = iou
            best_gt = gt
            
    return max_iou, best_gt


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
    
    image_tensor = numpy_to_torch(image_resized, torch.uint8).permute(2, 0, 1).contiguous().float()
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
    segs_tensor = numpy_to_torch(segs_square, torch.float32).permute(2, 0, 1).contiguous()
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


class VIGORTestDataset(Dataset):
    """测试阶段预加载单个 sample 的静态视觉输入。"""

    def __init__(
        self,
        samples: List[Dict],
        clip_image_processor,
        transform,
        sam_mask_helper,
        image_size: int,
        precision: str,
    ):
        self.samples = samples
        self.clip_image_processor = clip_image_processor
        self.transform = transform
        self.sam_mask_helper = sam_mask_helper
        self.image_size = image_size
        self.precision = precision

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image_path = sample['image_path']
            image_np = cv2.imread(image_path)
            if image_np is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            gt_masks = []
            for gp in sample['gt_mask_paths']:
                m = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    gt_masks.append(m)
            if not gt_masks:
                raise ValueError(f"No valid GT masks for: {image_path}")

            segs, segs_origin = prepare_sam_masks(
                self.sam_mask_helper,
                sample['img_name'],
                self.image_size,
                self.transform,
                self.precision,
            )
            image_clip_np = self.clip_image_processor.preprocess(
                image_np, return_tensors="np"
            )["pixel_values"][0]
            image_clip = numpy_to_torch(image_clip_np, torch.float32)
            image_tensor, resize = preprocess_image(
                image_np, self.transform, self.image_size
            )

            return {
                "sample": sample,
                "image_np": image_np,
                "gt_masks": gt_masks,
                "segs": segs,
                "segs_origin": segs_origin,
                "image_clip": image_clip,
                "image_tensor": image_tensor,
                "resize": resize,
                "ori_size": image_np.shape[:2],
                "error": None,
            }
        except Exception as e:
            return {
                "sample": sample,
                "error": repr(e),
            }


def collate_single_sample(batch):
    return batch[0]


def build_test_loader(samples, clip_image_processor, transform, sam_mask_helper, args):
    dataset = VIGORTestDataset(
        samples=samples,
        clip_image_processor=clip_image_processor,
        transform=transform,
        sam_mask_helper=sam_mask_helper,
        image_size=args.image_size,
        precision=args.precision,
    )
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "collate_fn": collate_single_sample,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **loader_kwargs)


def predict_mask_llmseg(
    model, tokenizer, prepared_sample: Dict, instruction, args
):
    """LLMSeg 预测掩码 - 从 SAM 候选中选择"""
    ori_size = prepared_sample["ori_size"]
    segs = prepared_sample["segs"]
    segs_origin = prepared_sample["segs_origin"]
    
    if segs is None:
        return np.ones(ori_size, dtype=np.float32)  # 返回全背景
    
    image_clip = prepared_sample["image_clip"]
    image_tensor = prepared_sample["image_tensor"]
    resize = prepared_sample["resize"]
    
    # 构建对话
    question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], "[SEG]")
    prompt = conv.get_prompt()
    
    # Tokenize
    from model.llava.mm_utils import tokenizer_image_token
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # 占空 IoU
    K = segs.shape[0]
    dummy_ious = np.zeros((1, K))
    dummy_iops = np.zeros((1, K))
    
    input_dict = {
        "images": image_tensor.unsqueeze(0),
        "images_clip": image_clip.unsqueeze(0),
        "input_ids": input_ids,
        "labels": labels,
        "attention_masks": attention_mask,
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
    with torch.no_grad():
        output_dict = model(**input_dict)
    
    # 获取预测结果
    pred_similarity = output_dict["pred_similarity"][0]  # (1, K)
    max_idx = torch.argmax(pred_similarity).item()
    
    # 获取预测 mask
    pred_mask = segs_origin[:, :, max_idx]  # (H, W)
    
    return pred_mask


def evaluate_sample(
    model, tokenizer, prepared_sample: Dict, args, obj_count: int = 1
) -> Dict:
    """评估单个样本 (支持多目标 Max-IoU)"""
    sample = prepared_sample["sample"]
    image_path = sample['image_path']
    gt_mask_paths = sample['gt_mask_paths'] # 这是一个列表
    instructions = sample['instructions']
    img_name = sample['img_name']
    
    # 设置当前图片名
    args._current_img_name = img_name

    image_np = prepared_sample["image_np"]
    gt_masks = prepared_sample["gt_masks"]
    if not gt_masks:
        return None
    
    original_size = image_np.shape[:2]
    original_h, original_w = original_size
    
    # 对每条指令进行推理
    pred_masks = []
    ic_ious = []
    
    for instr_idx, instruction in enumerate(instructions[:3]):
        pred_mask = predict_mask_llmseg(
            model, tokenizer, prepared_sample, instruction, args
        )
        
        # 调整大小
        if pred_mask.shape != (original_h, original_w):
            pred_mask = cv2.resize(pred_mask.astype(np.float32), 
                                   (original_w, original_h),
                                   interpolation=cv2.INTER_NEAREST)
        
        pred_masks.append(pred_mask)
        
        # 多目标择优逻辑 (Max-IoU)
        iou, best_gt = compute_iou_max(pred_mask, gt_masks)
        ic_ious.append(iou)
        
        # 保存可视化结果 (展示最匹配的那个 GT)
        if args.save_vis:
            sample_info = {
                'gt_object': sample['gt_object'],
                'difficulty': sample['difficulty'],
                'instructions': instructions,
                'img_name': img_name,
            }
            save_visualization(
                image_np=image_np,
                pred_mask=pred_mask,
                gt_mask=best_gt,
                vis_dir=args.vis_dir,
                sample_info=sample_info,
                instruction_idx=instr_idx,
                iou=iou,
                obj_count=obj_count
            )
    
    # 计算平均 IC-IoU
    avg_ic_iou = np.mean(ic_ious) if ic_ious else 0.0
    
    # 计算 ICR (pred 之间的 IoU)
    iou_pairs = []
    if len(pred_masks) >= 3:
        iou_pairs.append(compute_iou(pred_masks[0], pred_masks[1]))
        iou_pairs.append(compute_iou(pred_masks[0], pred_masks[2]))
        iou_pairs.append(compute_iou(pred_masks[1], pred_masks[2]))
    elif len(pred_masks) == 2:
        iou_pairs.append(compute_iou(pred_masks[0], pred_masks[1]))
    
    return {
        'image_path': image_path,
        'gt_mask_path': gt_mask_paths[0], # 返回第一个作为代表
        'gt_object': sample['gt_object'],
        'difficulty': sample['difficulty'],
        'ic_ious': ic_ious,
        'avg_ic_iou': avg_ic_iou,
        'iou_pairs': iou_pairs,
        'num_instructions': len(instructions[:3]),
    }



def main(args):
    args = parse_args(args)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
    
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
    print(f"  DataLoader workers: {args.workers}")
    
    # 存储结果
    results = {
        'easy': [],
        'hard': [],
    }
    skipped_errors = {
        'easy': [],
        'hard': [],
    }
    
    # 测试 Easy 样本
    print("\n" + "=" * 60)
    print("  测试 Easy 样本")
    print("=" * 60)
    easy_seen_counts = defaultdict(int)
    easy_loader = build_test_loader(
        easy_samples, clip_image_processor, transform, sam_mask_helper, args
    )
    for prepared_sample in tqdm(easy_loader, desc="Easy"):
        sample = prepared_sample.get("sample", {})
        try:
            if prepared_sample.get("error"):
                raise RuntimeError(prepared_sample["error"])
            img_name = sample['img_name']
            obj_name = sample['gt_object']
            easy_seen_counts[(img_name, obj_name)] += 1
            curr_count = easy_seen_counts[(img_name, obj_name)]
            
            result = evaluate_sample(
                model, tokenizer, prepared_sample, args, obj_count=curr_count
            )
            if result:
                results['easy'].append(result)
        except Exception as e:
            skipped_errors['easy'].append((sample.get('image_path', 'unknown'), repr(e)))
            if len(skipped_errors['easy']) <= 5:
                print(f"  [Easy Error] {sample.get('image_path', 'unknown')}: {e}")
            if args.debug:
                print(f"  [Error] {sample['image_path']}: {e}")
    
    # 测试 Hard 样本
    print("\n" + "=" * 60)
    print("  测试 Hard 样本")
    print("=" * 60)
    hard_seen_counts = defaultdict(int)
    hard_loader = build_test_loader(
        hard_samples, clip_image_processor, transform, sam_mask_helper, args
    )
    for prepared_sample in tqdm(hard_loader, desc="Hard"):
        sample = prepared_sample.get("sample", {})
        try:
            if prepared_sample.get("error"):
                raise RuntimeError(prepared_sample["error"])
            img_name = sample['img_name']
            obj_name = sample['gt_object']
            hard_seen_counts[(img_name, obj_name)] += 1
            curr_count = hard_seen_counts[(img_name, obj_name)]
            
            result = evaluate_sample(
                model, tokenizer, prepared_sample, args, obj_count=curr_count
            )
            if result:
                results['hard'].append(result)
        except Exception as e:
            skipped_errors['hard'].append((sample.get('image_path', 'unknown'), repr(e)))
            if len(skipped_errors['hard']) <= 5:
                print(f"  [Hard Error] {sample.get('image_path', 'unknown')}: {e}")
            if args.debug:
                print(f"  [Error] {sample['image_path']}: {e}")
    
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
        f.write(f"  Hard: {hard_metrics['count']}\n")
        f.write(f"跳过样本数: Easy={len(skipped_errors['easy'])}, Hard={len(skipped_errors['hard'])}\n")
        if skipped_errors['easy'] or skipped_errors['hard']:
            f.write("跳过样本错误示例:\n")
            for split_name in ['easy', 'hard']:
                for path, err in skipped_errors[split_name][:5]:
                    f.write(f"  [{split_name}] {path}: {err}\n")
        f.write("\n")
        
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
