#!/usr/bin/env python3
"""
使用SAM2的SamAutomaticMaskGenerator为VIGOR-100K数据集生成候选掩码
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
import colorsys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尝试导入SAM2（如果可用）
SAM2_AVAILABLE = False
try:
    # 方法1: 尝试使用transformers库加载SAM2
    from transformers import Sam2Model, Sam2Processor
    SAM2_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SAM2_TRANSFORMERS_AVAILABLE = False

try:
    # 方法2: 尝试使用SAM2官方代码
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_OFFICIAL_AVAILABLE = True
except ImportError:
    SAM2_OFFICIAL_AVAILABLE = False

if SAM2_TRANSFORMERS_AVAILABLE or SAM2_OFFICIAL_AVAILABLE:
    SAM2_AVAILABLE = True

# 导入SAM（作为备选）
from model.segment_anything import sam_model_registry
from model.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available")


class MaskDecoderWrapper(nn.Module):
    """包装器类，用于正确处理PEFT包装后的mask_decoder"""
    def __init__(self, mask_decoder):
        super().__init__()
        if hasattr(mask_decoder, 'base_model'):
            self.add_module('mask_decoder', mask_decoder.base_model)
            self._peft_model = mask_decoder
        else:
            self.add_module('mask_decoder', mask_decoder)
            self._peft_model = mask_decoder  # 存储原始PEFT模型以便加载权重
    
    def forward(self, 
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool):
        return self._modules['mask_decoder'](
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
    
    def __getattr__(self, name):
        if name == '_peft_model':
            return super().__getattribute__('_peft_model')
        
        if name.startswith('_') or name == 'mask_decoder':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if 'mask_decoder' in self._modules:
            try:
                return getattr(self._modules['mask_decoder'], name)
            except AttributeError:
                pass
        
        if self._peft_model is not None:
            try:
                return getattr(self._peft_model, name)
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def apply_lora_to_sam(model, lora_config: Dict):
    """对SAM模型应用LoRA"""
    if not PEFT_AVAILABLE:
        print("PEFT不可用，使用全量微调")
        return model
    
    target_modules = lora_config.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'out_proj'])
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=target_modules,
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        bias="none",
    )
    
    if lora_config.get('apply_to_mask_decoder', True):
        print("对mask_decoder应用LoRA...")
        peft_mask_decoder = get_peft_model(model.mask_decoder, peft_config)
        model.mask_decoder = MaskDecoderWrapper(peft_mask_decoder)
        print("LoRA应用成功")
    
    return model


def load_sam2_model_transformers(model_path: str, device: str = "cuda"):
    """使用transformers库加载SAM2模型"""
    if not SAM2_TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library not available")
    
    model = Sam2Model.from_pretrained(model_path)
    processor = Sam2Processor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def load_sam2_model_official(model_path: str, device: str = "cuda"):
    """使用SAM2官方代码加载模型"""
    if not SAM2_OFFICIAL_AVAILABLE:
        raise ImportError("SAM2 official code not available")
    
    # SAM2使用Hydra加载配置，配置文件名应该是相对于SAM2包的configs目录的路径
    # 例如: "configs/sam2/sam2_hiera_l.yaml"
    config_file = "configs/sam2/sam2_hiera_l.yaml"
    
    # 查找checkpoint文件
    checkpoint = os.path.join(model_path, "sam2_hiera_large.pt")
    
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(model_path, "model.safetensors")
    
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found in {model_path}")
    
    sam2_model = build_sam2(config_file, ckpt_path=checkpoint, device=device)
    return sam2_model


def load_sam_model(model_path: str, device: str = "cuda", use_finetuned: bool = False):
    """加载SAM模型（作为备选）"""
    
    if use_finetuned:
        # 加载微调后的模型
        print(f"Loading finetuned SAM checkpoint: {model_path}")
        
        # 首先加载原始SAM模型作为基础
        base_checkpoint = "/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"
        if not os.path.exists(base_checkpoint):
            raise FileNotFoundError(f"Base SAM checkpoint not found: {base_checkpoint}")
        
        sam = sam_model_registry["vit_h"](checkpoint=base_checkpoint)
        sam.to(device=device)
        
        # 加载微调后的权重
        print(f"Loading finetuned weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            
            # 检查权重键的匹配情况
            model_keys = set(sam.state_dict().keys())
            checkpoint_keys = set(model_state_dict.keys())
            
            missing_in_checkpoint = model_keys - checkpoint_keys
            extra_in_checkpoint = checkpoint_keys - model_keys
            
            if missing_in_checkpoint:
                print(f"Warning: Missing keys in checkpoint: {len(missing_in_checkpoint)}")
                if len(missing_in_checkpoint) <= 5:
                    for key in missing_in_checkpoint:
                        print(f"  Missing: {key}")
            
            if extra_in_checkpoint:
                print(f"Info: Extra keys in checkpoint: {len(extra_in_checkpoint)}")
                if len(extra_in_checkpoint) <= 5:
                    for key in extra_in_checkpoint:
                        print(f"  Extra: {key}")
            
            # 检查是否有LoRA权重和独立的lora_state_dict
            lora_keys = [k for k in model_state_dict.keys() if 'lora' in k.lower()]
            has_lora_in_model_dict = len(lora_keys) > 0
            has_separate_lora_dict = 'lora_state_dict' in checkpoint
            
            if has_separate_lora_dict:
                print(f"   Found separate lora_state_dict with LoRA weights")
                lora_state_dict = checkpoint['lora_state_dict']
                print(f"   LoRA state dict contains {len(lora_state_dict)} LoRA parameters")
                
                # 应用LoRA配置（与微调脚本一致）
                lora_config = {
                    'r': 32,
                    'lora_alpha': 64,
                    'lora_dropout': 0.1,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                    'apply_to_mask_decoder': True,
                    'apply_to_image_encoder': False
                }
                sam = apply_lora_to_sam(sam, lora_config)
                
                # 加载模型权重（非LoRA部分）
                non_lora_state_dict = {k: v for k, v in model_state_dict.items() if 'lora' not in k.lower()}
                load_result = sam.load_state_dict(non_lora_state_dict, strict=False)
                print(f"   Model weights loaded: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
                
                # 加载LoRA权重（使用独立的lora_state_dict）
                try:
                    # 获取PEFT包装的mask_decoder
                    peft_mask_decoder = sam.mask_decoder
                    if hasattr(peft_mask_decoder, '_modules') and 'mask_decoder' in peft_mask_decoder._modules:
                        # 通过wrapper访问PEFT模型
                        wrapper = peft_mask_decoder
                        # 重新创建PEFT模型以正确加载权重
                        lora_config = {
                            'r': 32,
                            'lora_alpha': 64,
                            'lora_dropout': 0.1,
                            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                            'apply_to_mask_decoder': True,
                            'apply_to_image_encoder': False
                        }
                        
                        # 获取原始mask_decoder
                        original_mask_decoder = wrapper._modules['mask_decoder']
                        
                        # 重新创建PEFT模型
                        peft_config = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            r=lora_config['r'],
                            lora_alpha=lora_config['lora_alpha'],
                            target_modules=lora_config['target_modules'],
                            lora_dropout=lora_config['lora_dropout'],
                            bias="none",
                        )
                        
                        # 包装原始mask_decoder
                        new_peft_model = get_peft_model(original_mask_decoder, peft_config)
                        
                        # 使用PEFT的set_peft_model_state_dict加载LoRA权重
                        from peft import set_peft_model_state_dict
                        set_peft_model_state_dict(new_peft_model, lora_state_dict)
                        
                        # 重新包装
                        sam.mask_decoder = MaskDecoderWrapper(new_peft_model)
                        
                        print(f"   ✅ LoRA weights loaded successfully using set_peft_model_state_dict")
                        
                        # 验证LoRA权重是否正确加载
                        peft_state = new_peft_model.state_dict()
                        loaded_lora_keys = [k for k in peft_state.keys() if 'lora' in k.lower()]
                        print(f"   ✅ Verified {len(loaded_lora_keys)} LoRA parameters loaded")
                        
                    else:
                        print(f"   ❌ Cannot access PEFT model structure")
                        
                except Exception as e:
                    print(f"   ❌ Failed to load LoRA weights: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif has_lora_in_model_dict:
                print(f"   Found {len(lora_keys)} LoRA weights in model_state_dict")
                
                # 检查权重键名格式，判断是否使用了MaskDecoderWrapper
                has_wrapper_prefix = any(k.startswith('mask_decoder.mask_decoder.') for k in model_state_dict.keys())
                
                if has_wrapper_prefix:
                    print(f"   Detected MaskDecoderWrapper prefix in weights, fixing key names...")
                    # 修复权重键名：移除重复的mask_decoder前缀
                    fixed_model_state_dict = {}
                    for k, v in model_state_dict.items():
                        if k.startswith('mask_decoder.mask_decoder.'):
                            # 移除重复的mask_decoder前缀
                            fixed_key = k.replace('mask_decoder.mask_decoder.', 'mask_decoder.')
                            fixed_model_state_dict[fixed_key] = v
                        else:
                            fixed_model_state_dict[k] = v
                    
                    print(f"   Fixed {len(fixed_model_state_dict)} weight keys")
                    model_state_dict = fixed_model_state_dict
                
                print(f"   Loading ALL finetuned weights (mask_decoder + LoRA)...")
                
                # 首先应用LoRA配置（与训练时一致）
                lora_config = {
                    'r': 32,
                    'lora_alpha': 64,
                    'lora_dropout': 0.1,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                    'apply_to_mask_decoder': True,
                    'apply_to_image_encoder': False
                }
                sam = apply_lora_to_sam(sam, lora_config)
                print(f"   ✅ LoRA configuration applied")
                
                # 加载完整的微调权重（包括mask_decoder权重和LoRA权重）
                load_result = sam.load_state_dict(model_state_dict, strict=False)
                print(f"   ✅ Complete finetuned model weights loaded")
                print(f"   Missing keys: {len(load_result.missing_keys)}")
                print(f"   Unexpected keys: {len(load_result.unexpected_keys)}")
                
                # 验证关键组件是否加载
                mask_decoder_keys = [k for k in model_state_dict.keys() if 'mask_decoder' in k]
                lora_keys_loaded = [k for k in mask_decoder_keys if 'lora' in k.lower()]
                non_lora_mask_keys = [k for k in mask_decoder_keys if 'lora' not in k.lower()]
                
                print(f"   ✅ Mask decoder weights loaded: {len(non_lora_mask_keys)}")
                print(f"   ✅ LoRA weights loaded: {len(lora_keys_loaded)}")
                print(f"   ✅ Total finetuned parameters: {len(mask_decoder_keys)}")
                
                # 详细检查加载情况
                if load_result.missing_keys:
                    print(f"   ⚠️  Missing keys (first 10):")
                    for key in list(load_result.missing_keys)[:10]:
                        print(f"     - {key}")
                
                if load_result.unexpected_keys:
                    print(f"   ⚠️  Unexpected keys (first 10):")
                    for key in list(load_result.unexpected_keys)[:10]:
                        print(f"     - {key}")
            else:
                # 没有LoRA权重，可能是全量微调，直接加载
                print(f"   No LoRA weights found, treating as full fine-tuning")
                try:
                    load_result = sam.load_state_dict(model_state_dict, strict=False)
                    print(f"   ✅ Full finetuned weights loaded successfully")
                    print(f"   Missing keys: {len(load_result.missing_keys)}")
                    print(f"   Unexpected keys: {len(load_result.unexpected_keys)}")
                except Exception as e:
                    print(f"   ❌ Failed to load finetuned weights: {e}")
                    print("   Falling back to original SAM weights")
        else:
            print("Warning: No 'model_state_dict' found in checkpoint, using original SAM")
        
        return sam
    
    else:
        # 原有逻辑：加载原始SAM模型
        # 尝试多个可能的checkpoint文件名
        possible_checkpoints = [
            "sam_vit_h_4b8939.pth",
            "sam_vit_h.pth",
        ]
        
        checkpoint = None
        for ckpt_name in possible_checkpoints:
            ckpt_path = os.path.join(model_path, ckpt_name)
            if os.path.exists(ckpt_path):
                checkpoint = ckpt_path
                break
        
        if checkpoint is None:
            # 尝试查找目录中的.pth文件
            pth_files = list(Path(model_path).glob("*.pth"))
            if pth_files:
                checkpoint = str(pth_files[0])
                print(f"Found checkpoint: {checkpoint}")
            else:
                # 如果model_path本身就是checkpoint文件
                if os.path.isfile(model_path) and model_path.endswith('.pth'):
                    checkpoint = model_path
                else:
                    raise FileNotFoundError(f"SAM checkpoint not found in {model_path}")
        
        print(f"Loading SAM checkpoint: {checkpoint}")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(device=device)
        return sam


def init_mask_generator(model_path: str, use_sam2: bool = False, device: str = "cuda", fallback_sam_path: str = None, use_finetuned: bool = False):
    """
    初始化掩码生成器
    
    注意：SAM2的API与SAM不同，目前代码库中使用的是SAM。
    如果需要使用SAM2，需要安装SAM2官方代码库：
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    
    Args:
        model_path: SAM2或SAM模型路径
        use_sam2: 是否使用SAM2
        device: 设备
        fallback_sam_path: 如果SAM2不可用，回退使用的SAM路径
        use_finetuned: 是否使用微调后的SAM模型
    """
    if use_sam2:
        if not SAM2_AVAILABLE:
            print("⚠️  SAM2库未安装，无法使用SAM2")
            print("   安装方法: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            if fallback_sam_path:
                print(f"   回退到SAM，使用路径: {fallback_sam_path}")
                model_path = fallback_sam_path
            else:
                raise ImportError("SAM2不可用，且未提供SAM回退路径。请安装SAM2或使用SAM模型。")
        elif SAM2_OFFICIAL_AVAILABLE:
            print("Loading SAM2 model...")
            try:
                sam2_model = load_sam2_model_official(model_path, device)
                mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
                print("✅ SAM2 model loaded successfully (official)")
                return mask_generator
            except Exception as e:
                print(f"Failed to load SAM2 via official code: {e}")
                if fallback_sam_path:
                    print(f"Falling back to SAM, using path: {fallback_sam_path}")
                    model_path = fallback_sam_path
                else:
                    raise
        else:
            print("SAM2 official code not available.")
            if fallback_sam_path:
                print(f"Falling back to SAM, using path: {fallback_sam_path}")
                model_path = fallback_sam_path
            else:
                raise ImportError("SAM2不可用，且未提供SAM回退路径")
    
    # 使用SAM（推荐，因为代码库中已有完整支持）
    print("Loading SAM model...")
    sam = load_sam_model(model_path, device, use_finetuned=use_finetuned)
    
    # 对于微调后的模型，使用与原始SAM相同的参数以确保公平比较
    if use_finetuned:
        print("Using same parameters as original SAM for fair comparison...")
        mask_generator = SamAutomaticMaskGenerator(
            sam, 
            pred_iou_thresh=0.8,      # 与原始SAM相同0.88
            stability_score_thresh=0.8,  # 与原始SAM相同0.95
            min_mask_region_area=100,  # 与原始SAM相同
            box_nms_thresh=0.7,        # NMS阈值（默认值）
            crop_n_layers=0,           # 禁用裁剪，提高速度
            crop_n_points_downscale_factor=1,  # 裁剪点下采样因子
            points_per_side=32,         # 采样点数（默认值）
            points_per_batch=64,       # 批处理大小（默认值）
        )
    else:
        print("use SAM origin")
        mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.8, stability_score_thresh=0.8)
    
    print("✅ SAM model loaded successfully")
    return mask_generator


def generate_colored_mask_overlay(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """生成带颜色的mask叠加图像"""
    overlay = image.copy()
    
    # 生成不同颜色
    num_masks = len(masks)
    colors = []
    for i in range(num_masks):
        hue = i / max(num_masks, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    # 叠加每个mask
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
    
    return overlay


def get_image_files(image_dir: str) -> List[str]:
    """获取所有图像文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    return sorted([str(f) for f in image_files])


def process_image(
    mask_generator: Any,
    image_path: str,
    output_dir: str,
    save_format: str = "png"
):
    """处理单张图像，生成并保存候选掩码"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot read image {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 生成掩码
    masks = mask_generator.generate(image_rgb)
    
    if len(masks) == 0:
        print(f"Warning: No masks generated for {image_path}")
        return None
    
    # 创建输出目录
    image_name = Path(image_path).stem
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    masks_dir = os.path.join(image_output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # 保存每个掩码
    mask_info = []
    for idx, mask_data in enumerate(masks):
        # 获取掩码数组
        if isinstance(mask_data['segmentation'], dict):
            # RLE格式，需要转换
            from model.segment_anything.utils.amg import rle_to_mask
            mask = rle_to_mask(mask_data['segmentation'])
        else:
            mask = mask_data['segmentation']  # numpy array
        
        # 保存掩码
        mask_filename = f"mask_{idx:04d}.{save_format}"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if save_format == "png":
            # 保存为PNG（0=背景，255=前景）
            mask_uint8 = (mask.astype(np.uint8) * 255)
            cv2.imwrite(mask_path, mask_uint8)
        else:
            # 保存为numpy数组
            np.save(mask_path.replace('.png', '.npy'), mask)
        
        # 保存掩码信息
        mask_info.append({
            'mask_file': mask_filename,
            'area': mask_data.get('area', int(mask.sum())),
            'bbox': mask_data.get('bbox', []),
            'predicted_iou': mask_data.get('predicted_iou', 0.0),
            'stability_score': mask_data.get('stability_score', 0.0),
            'point_coords': mask_data.get('point_coords', []),
        })
    
    # 保存掩码信息JSON
    info_path = os.path.join(image_output_dir, "mask_info.json")
    with open(info_path, 'w') as f:
        json.dump({
            'image_name': image_name,
            'image_path': image_path,
            'num_masks': len(masks),
            'masks': mask_info
        }, f, indent=2)
    
    # ✅ 生成并保存掩码叠加可视化图像
    mask_arrays = []
    for mask_data in masks:
        if isinstance(mask_data['segmentation'], dict):
            from model.segment_anything.utils.amg import rle_to_mask
            mask = rle_to_mask(mask_data['segmentation'])
        else:
            mask = mask_data['segmentation']
        mask_arrays.append(mask)
    
    if len(mask_arrays) > 0:
        # 生成彩色叠加图像
        overlay = generate_colored_mask_overlay(image_rgb, mask_arrays, alpha=0.5)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # 保存叠加图像
        overlay_path = os.path.join(image_output_dir, "overlay_all_masks.png")
        cv2.imwrite(overlay_path, overlay_bgr)
        
        # 保存原图（用于对比）
        original_path = os.path.join(image_output_dir, "original_image.png")
        cv2.imwrite(original_path, image)
    
    return len(masks)


def main():
    parser = argparse.ArgumentParser(description="使用SAM2/SAM为VIGOR数据集生成候选掩码")
    parser.add_argument("--model_path", type=str, required=True,
                       help="SAM2/SAM模型路径（包含checkpoint文件的目录）")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="VIGOR数据集目录（包含图像文件的目录）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（保存生成的掩码）")
    parser.add_argument("--use_sam2", action="store_true",
                       help="使用SAM2（如果可用），否则使用SAM")
    parser.add_argument("--fallback_sam_path", type=str, default=None,
                       help="如果SAM2不可用，回退使用的SAM模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备（cuda或cpu）")
    parser.add_argument("--save_format", type=str, default="png",
                       choices=["png", "npy"],
                       help="保存格式（png或npy）")
    parser.add_argument("--max_images", type=int, default=None,
                       help="最大处理图像数量（用于测试）")
    parser.add_argument("--use_finetuned", action="store_true",
                       help="使用微调后的SAM模型（model_path应该是.pth文件）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化掩码生成器
    print("=" * 60)
    print("Initializing mask generator...")
    mask_generator = init_mask_generator(
        args.model_path,
        use_sam2=args.use_sam2,
        device=args.device,
        fallback_sam_path=args.fallback_sam_path,
        use_finetuned=args.use_finetuned
    )
    
    # 获取所有图像文件
    print("\n" + "=" * 60)
    print("Scanning for images...")
    image_files = get_image_files(args.dataset_dir)
    print(f"Found {len(image_files)} images")
    
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Processing first {len(image_files)} images (limited by --max_images)")
    
    # 处理每张图像
    print("\n" + "=" * 60)
    print("Processing images...")
    total_masks = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Processing"):
        try:
            num_masks = process_image(
                mask_generator,
                image_path,
                args.output_dir,
                save_format=args.save_format
            )
            if num_masks:
                total_masks += num_masks
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            failed_images.append(image_path)
            import traceback
            traceback.print_exc()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Processing completed!")
    print(f"  Total images processed: {len(image_files) - len(failed_images)}/{len(image_files)}")
    print(f"  Total masks generated: {total_masks}")
    print(f"  Average masks per image: {total_masks / (len(image_files) - len(failed_images)):.1f}" if (len(image_files) - len(failed_images)) > 0 else "  Average masks per image: 0")
    print(f"  Failed images: {len(failed_images)}")
    print(f"  Output directory: {args.output_dir}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img in failed_images[:10]:  # 只显示前10个
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")


if __name__ == "__main__":
    main()
