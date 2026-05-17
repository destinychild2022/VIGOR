"""
LLMSeg (LISA-based) VIGOR-100K test script.

The model selects from SAM candidate masks. This test entry mirrors validation:
- one dataset item is one VIGOR sample with up to three instructions;
- image/depth/SAM candidates are loaded once per sample by DataLoader workers;
- output mask can be selected by max pred_similarity or by the validation
  pred_iou-threshold merge path;
- when a sample has multiple GT masks, metrics use the GT with the best IoU.
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda


DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_LABEL = 255


def parse_args(args):
    parser = argparse.ArgumentParser(description="LLMSeg VIGOR-100K 测试脚本")

    parser.add_argument("--version", default="/root/autodl-tmp/model/LISA_Plus_7b", type=str)
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/runs/finetune_llmseg_vigor_simple-spatial-1/ckpt_model/best",
        type=str,
        help="单个 checkpoint 目录，例如 ckpt_model/best 或 ckpt_model/epoch_10",
    )
    parser.add_argument("--checkpoint_tag", default=None, type=str, help="结果文件/可视化子目录使用的权重名")
    parser.add_argument("--vision_tower", default="/root/autodl-tmp/model/clip-vit-large-patch14", type=str)
    parser.add_argument(
        "--vision_pretrained",
        default="/root/autodl-tmp/model/SAM-vit-h/sam_vit_h_4b8939.pth",
        type=str,
    )

    parser.add_argument("--data_dir", default="/root/autodl-tmp/VIGOR-100K_new/test", type=str)
    parser.add_argument("--vigor_easy_json_file", default="open_vocab_grasp_easy_new_1.json", type=str, help="VIGOR easy JSON filename under data_dir")
    parser.add_argument("--vigor_hard_json_file", default="open_vocab_grasp_hard_new_1.json", type=str, help="VIGOR hard JSON filename under data_dir")
    parser.add_argument("--sam_masks_dir", required=True, type=str)
    parser.add_argument("--depth_dir", default=None, type=str, help="默认使用 data_dir/depth")
    parser.add_argument("--allow_missing_depth", action="store_true", default=False)

    parser.add_argument("--output_dir", default="/root/autodl-tmp/result", type=str)
    parser.add_argument("--vis_dir", default="/root/autodl-tmp/vis_output", type=str)
    parser.add_argument("--save_vis", action="store_true", default=False)

    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"], type=str)
    parser.add_argument("--image_size", default=896, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,out_proj", type=str)

    parser.add_argument("--icr_thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9", type=str)
    parser.add_argument("--split", default="both", choices=["easy", "hard", "both"], type=str)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--max_instructions", default=3, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument(
        "--mask_selection_mode",
        default="similarity",
        choices=["similarity", "iou"],
        help="similarity: 选 pred_similarity 最大候选；iou: 合并 pred_iou > iou_threshold 的候选",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.5,
        type=float,
        help="MASK_SELECTION_MODE=iou 时使用的 pred_iou 阈值，训练验证默认 0.5",
    )

    parsed = parser.parse_args(args)
    if parsed.checkpoint_tag is None:
        parsed.checkpoint_tag = os.path.basename(os.path.normpath(parsed.checkpoint)) or "checkpoint"
    return parsed


def torch_dtype_from_precision(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.half
    return torch.float32


def load_samples(data_dir: str, difficulty: str, easy_json_file: str, hard_json_file: str) -> List[Dict]:
    json_name = easy_json_file if difficulty == "easy" else hard_json_file
    json_file = os.path.join(data_dir, json_name)
    if not os.path.exists(json_file):
        print(f"  [警告] {difficulty} JSON 文件不存在: {json_file}")
        return []

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    if not isinstance(raw_samples, list):
        return []

    samples = []
    for sample in raw_samples:
        gt_mask_path_rel = sample.get("gt_mask_path", "")
        if not gt_mask_path_rel:
            continue

        rel_paths = [p.strip() for p in gt_mask_path_rel.split(",") if p.strip()]
        if not rel_paths:
            continue

        mask_filename = os.path.basename(rel_paths[0])
        match = re.match(r"^(\d+)_", mask_filename)
        if not match:
            continue

        img_num = match.group(1)
        img_name = f"{img_num}.png"
        image_path = os.path.join(data_dir, img_name)
        gt_mask_paths = [
            p if os.path.isabs(p) else os.path.join(data_dir, p)
            for p in rel_paths
        ]

        if not os.path.exists(image_path):
            continue

        samples.append(
            {
                "image_path": image_path,
                "gt_mask_paths": gt_mask_paths,
                "gt_mask_path": ",".join(gt_mask_paths),
                "gt_object": sample.get("gt_object", sample.get("object", "")),
                "instructions": sample.get("instructions", []),
                "difficulty": difficulty,
                "img_name": img_name,
                "scene": sample.get("scene", ""),
            }
        )
    return samples


def load_camera_intrinsic(data_dir: str, depth_dir: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    split = os.path.basename(os.path.normpath(data_dir))
    base_dir = os.path.dirname(os.path.normpath(data_dir))
    camera_paths = [
        os.path.join(depth_dir, "camera.json"),
        os.path.join(base_dir, split, "depth", "camera.json"),
        os.path.join(base_dir, "train", "depth", "camera.json"),
    ]
    for camera_path in camera_paths:
        if not os.path.exists(camera_path):
            continue
        with open(camera_path, "r", encoding="utf-8") as f:
            camera = json.load(f)
        intr = camera.get("intrinsics", {})
        if all(k in intr for k in ("fx", "fy", "cx", "cy")):
            intrinsic = np.array(
                [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        else:
            intrinsic = np.array(intr.get("intrinsic", np.eye(3)), dtype=np.float32)
        size = (int(camera.get("height", 0)), int(camera.get("width", 0)))
        return intrinsic, size
    return None, None


def preprocess_image(image_np: np.ndarray, transform: ResizeLongestSide, image_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    image_resized = transform.apply_image(image_np)
    resize = image_resized.shape[:2]
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float()
    image_tensor = (image_tensor - pixel_mean) / pixel_std
    h, w = image_tensor.shape[-2:]
    image_tensor = F.pad(image_tensor, (0, image_size - w, 0, image_size - h))
    return image_tensor, resize


def load_depth_and_intrinsic(
    img_name: str,
    resize: Tuple[int, int],
    depth_dir: str,
    camera_intrinsic: Optional[np.ndarray],
    camera_size: Optional[Tuple[int, int]],
    image_size: int,
    allow_missing_depth: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if camera_intrinsic is None:
        if allow_missing_depth:
            return None, None
        raise FileNotFoundError(f"未找到 camera.json，已检查 depth_dir={depth_dir}")

    depth_path = os.path.join(depth_dir, os.path.splitext(img_name)[0] + ".npy")
    if not os.path.exists(depth_path):
        if allow_missing_depth:
            return None, None
        raise FileNotFoundError(f"未找到 depth 文件: {depth_path}")

    depth = np.load(depth_path)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Unexpected depth shape for {depth_path}: {depth.shape}")
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)

    resized_h, resized_w = resize
    depth_resized = cv2.resize(depth, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padh = image_size - resized_h
    padw = image_size - resized_w
    if padh < 0 or padw < 0:
        raise ValueError(f"depth_resized larger than image_size={image_size}: {(resized_h, resized_w)}")
    depth_square = np.pad(depth_resized, ((0, padh), (0, padw)), mode="constant", constant_values=0.0)

    cam_h, cam_w = camera_size if camera_size is not None else depth.shape
    if cam_h <= 0 or cam_w <= 0:
        cam_h, cam_w = depth.shape
    intrinsic = camera_intrinsic.copy()
    intrinsic[0, :] *= resized_w / float(cam_w)
    intrinsic[1, :] *= resized_h / float(cam_h)
    return torch.from_numpy(depth_square).unsqueeze(0).float(), torch.from_numpy(intrinsic).float()


def prepare_sam_masks(
    sam_mask_helper: SAM_Mask_Reader_PNG,
    img_name: str,
    image_size: int,
    transform: ResizeLongestSide,
    precision: str,
    ori_size: Tuple[int, int],
) -> Tuple[torch.Tensor, np.ndarray]:
    try:
        segs_dict = sam_mask_helper.extract_sam_segs(img_name)
        segs_origin = segs_dict["segs_origin"]
    except Exception:
        h, w = ori_size
        segs_origin = np.ones((h, w, 1), dtype=np.uint8)

    if segs_origin is None or segs_origin.ndim != 3 or segs_origin.shape[2] == 0:
        h, w = ori_size
        segs_origin = np.ones((h, w, 1), dtype=np.uint8)

    segs_resized_list = []
    for k in range(segs_origin.shape[2]):
        mask_k_uint8 = (segs_origin[:, :, k] * 255).astype(np.uint8)
        mask_k_resized = transform.apply_image(mask_k_uint8).astype(np.float32) / 255.0
        segs_resized_list.append(mask_k_resized)

    segs_resized = np.stack(segs_resized_list, axis=2)
    h2, w2, _ = segs_resized.shape
    if h2 > image_size or w2 > image_size:
        raise ValueError(f"segs_resized larger than image_size={image_size}: {(h2, w2)}")
    segs_square = np.pad(
        segs_resized,
        ((0, image_size - h2), (0, image_size - w2), (0, 0)),
        mode="constant",
        constant_values=1,
    )
    segs = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous()
    segs = F.interpolate(segs.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
    return segs.to(torch_dtype_from_precision(precision)), segs_origin


class VIGORTestDataset(Dataset):
    def __init__(self, samples: List[Dict], args, clip_image_processor, transform):
        self.samples = samples
        self.args = args
        self.clip_image_processor = clip_image_processor
        self.transform = transform
        self.sam_mask_helper = SAM_Mask_Reader_PNG(args.sam_masks_dir)
        self.depth_dir = args.depth_dir or os.path.join(args.data_dir, "depth")
        self.camera_intrinsic, self.camera_size = load_camera_intrinsic(args.data_dir, self.depth_dir)

    def __len__(self):
        return len(self.samples)

    def _conversation(self, instruction: str) -> str:
        question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
        conv = conversation_lib.conv_templates[self.args.conv_type].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], "[SEG]")
        return conv.get_prompt()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_np = cv2.imread(sample["image_path"])
        if image_np is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        ori_size = image_np.shape[:2]

        image_clip = self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        image_tensor, resize = preprocess_image(image_np, self.transform, self.args.image_size)
        depth, intrinsic = load_depth_and_intrinsic(
            sample["img_name"],
            resize,
            self.depth_dir,
            self.camera_intrinsic,
            self.camera_size,
            self.args.image_size,
            self.args.allow_missing_depth,
        )
        segs, segs_origin = prepare_sam_masks(
            self.sam_mask_helper,
            sample["img_name"],
            self.args.image_size,
            self.transform,
            self.args.precision,
            ori_size,
        )

        gt_masks = []
        for gt_path in sample["gt_mask_paths"]:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_masks.append((gt_mask > 0).astype(np.float32))
        if not gt_masks:
            gt_masks = [np.ones(ori_size, dtype=np.float32)]
        gt_mask_tensor = torch.from_numpy(np.stack(gt_masks, axis=0)).float()

        instructions = sample.get("instructions", [])[: self.args.max_instructions]
        if not instructions:
            instructions = [f"segment {sample.get('gt_object', 'object')}"]
        conversations = [self._conversation(instruction) for instruction in instructions]

        return {
            "image_path": sample["image_path"],
            "images": image_tensor,
            "images_clip": image_clip,
            "depth": depth,
            "intrinsic": intrinsic,
            "conversations": conversations,
            "masks": gt_mask_tensor,
            "label": torch.ones(ori_size[0], ori_size[1]) * IGNORE_LABEL,
            "resize": resize,
            "segs": segs,
            "segs_origin": segs_origin,
            "ious": [np.zeros((1, segs.shape[0]), dtype=np.float32)],
            "iops": [np.zeros((1, segs.shape[0]), dtype=np.float32)],
            "inference": True,
            "meta": {
                **sample,
                "image_np": image_np,
                "gt_masks": gt_masks,
                "instructions": instructions,
            },
        }


def collate_vigor_test(batch, tokenizer, conv_type="llava_v1", use_mm_start_end=True):
    conversations = []
    offset = [0]
    for item in batch:
        conversations.extend(item["conversations"])
        offset.append(len(conversations))

    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversations = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversations]

    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    output = {
        "image_paths": [item["image_path"] for item in batch],
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "images_clip": torch.stack([item["images_clip"] for item in batch], dim=0),
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_masks": attention_masks,
        "masks_list": [item["masks"] for item in batch],
        "label_list": [item["label"] for item in batch],
        "resize_list": [item["resize"] for item in batch],
        "offset": torch.LongTensor(offset),
        "sam_segs_list": [item["segs"] for item in batch],
        "sam_ious_list": [item["ious"] for item in batch],
        "sam_iops_list": [item["iops"] for item in batch],
        "origin_segs_list": [item["segs_origin"] for item in batch],
        "inference": True,
        "meta_list": [item["meta"] for item in batch],
        "conversation_list": conversations,
    }

    if all(item["depth"] is not None and item["intrinsic"] is not None for item in batch):
        output["depths"] = torch.stack([item["depth"] for item in batch], dim=0)
        output["intrinsics"] = torch.stack([item["intrinsic"] for item in batch], dim=0)
    return output


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if pred_mask is None or gt_mask is None:
        return 0.0
    pred_fg = (pred_mask == 0)
    gt_fg = (gt_mask == 0)
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    return 0.0 if union == 0 else float(intersection) / float(union)


def select_best_gt(pred_mask: np.ndarray, gt_masks: List[np.ndarray]) -> Tuple[np.ndarray, float, int]:
    best_iou = -1.0
    best_idx = 0
    for idx, gt_mask in enumerate(gt_masks):
        iou = compute_iou(pred_mask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return gt_masks[best_idx], max(best_iou, 0.0), best_idx


def compute_icr_score(iou_pairs: List[float], threshold: float) -> int:
    return sum(1 for iou in iou_pairs if iou >= threshold)


def save_visualization(
    image_np: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    vis_dir: str,
    sample_info: dict,
    instruction_idx: int,
    iou: float,
    obj_count: int = 1,
):
    h, w = image_np.shape[:2]
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    pred_region = pred_mask == 0
    gt_region = gt_mask == 0
    gt_overlay = image_np.copy()
    pred_overlay = image_np.copy()
    gt_overlay[gt_region] = (image_np[gt_region] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    pred_overlay[pred_region] = (image_np[pred_region] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    combined = np.hstack([gt_overlay, pred_overlay])

    text_height = 100
    out = np.ones((combined.shape[0] + text_height, combined.shape[1], 3), dtype=np.uint8) * 255
    out[text_height:, :] = combined

    gt_object = sample_info.get("gt_object", "Unknown")
    difficulty = sample_info.get("difficulty", "unknown")
    instructions = sample_info.get("instructions", [])
    instruction = instructions[instruction_idx] if instruction_idx < len(instructions) else ""
    img_name = sample_info.get("img_name", "unknown")
    mode = sample_info.get("mask_selection_mode", "similarity")

    cv2.putText(out, f"GT Object: {gt_object} | Difficulty: {difficulty} | Select: {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    instr_display = instruction[:80] + "..." if len(instruction) > 80 else instruction
    cv2.putText(out, f"Instruction: {instr_display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(out, f"IoU: {iou:.4f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(out, "GT (Green)", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
    cv2.putText(out, "Pred (Red)", (350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    difficulty_dir = os.path.join(vis_dir, difficulty)
    os.makedirs(difficulty_dir, exist_ok=True)
    gt_object_clean = gt_object.replace("/", "_").replace("\\", "_").replace(" ", "_")
    output_filename = f"{img_name}_{gt_object_clean}_{obj_count}_instr{instruction_idx}_iou{iou:.3f}.png"
    output_path = os.path.join(difficulty_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return output_path


def load_model(args):
    print("\n" + "=" * 60)
    print("  加载模型...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    torch_dtype = torch_dtype_from_precision(args.precision)
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
        low_cpu_mem_usage=False,
        local_files_only=True,
        **model_args,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    class ModelArgs:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = "patch"
            self.pretrain_mm_mlp_adapter = None
            self.vision_tower = args.vision_tower

    model.get_model().initialize_vision_modules(ModelArgs())
    model.get_model().get_vision_tower().to(dtype=torch_dtype, device=args.device)
    model.get_model().initialize_lisa_modules(model.get_model().config)
    model.resize_token_embeddings(len(tokenizer))

    if args.lora_r > 0:
        print(f"\n  初始化 LoRA (r={args.lora_r})...")
        from peft import LoraConfig, get_peft_model

        def find_linear_layers(model_obj, lora_target_modules):
            cls = torch.nn.Linear
            excluded = [
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
            names = set()
            for name, module in model_obj.named_modules():
                if isinstance(module, cls) and all(x not in name for x in excluded) and any(x in name for x in lora_target_modules):
                    names.add(name)
            return sorted(names)

        lora_target = find_linear_layers(model, args.lora_target_modules.split(","))
        print(f"     LoRA target modules: {len(lora_target)} layers")
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=lora_target,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n  微调权重目录: {args.checkpoint}")
        lora_path = os.path.join(args.checkpoint, "lora_adapter")
        if os.path.exists(lora_path):
            model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)
            print(f"  LoRA 适配器加载成功: {lora_path}")
        else:
            step_dirs = [
                d
                for d in os.listdir(args.checkpoint)
                if os.path.isdir(os.path.join(args.checkpoint, d)) and d.startswith("global_step")
            ]
            if not step_dirs:
                print("  [警告] 未找到 global_step 目录")
            else:
                step_dirs.sort(key=lambda x: int(x.replace("global_step", "")))
                latest_step_dir = step_dirs[-1]
                mp_rank_path = os.path.join(args.checkpoint, latest_step_dir, "mp_rank_00_model_states.pt")
                if not os.path.exists(mp_rank_path):
                    print(f"  [警告] 未找到模型状态: {mp_rank_path}")
                else:
                    state_dict = torch.load(mp_rank_path, map_location="cpu")
                    module_state = state_dict.get("module", state_dict)
                    missing_keys, unexpected_keys = model.load_state_dict(module_state, strict=False)
                    print(f"  DeepSpeed checkpoint: {latest_step_dir}")
                    print(f"  Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"  [警告] 未找到微调权重: {args.checkpoint}")

    model = model.to(dtype=torch_dtype, device=args.device)
    model.eval()
    print(f"  模型精度: {torch_dtype}")

    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, local_files_only=True)
    transform = ResizeLongestSide(args.image_size)
    return model, tokenizer, clip_image_processor, transform


def _score_row(output_dict, key: str, batch_idx: int, instruction_idx: int) -> torch.Tensor:
    scores = output_dict[key][batch_idx].detach().float().cpu()
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    row_idx = min(instruction_idx, scores.shape[0] - 1)
    return scores[row_idx]


def select_prediction_mask(
    output_dict,
    segs_origin: np.ndarray,
    batch_idx: int,
    instruction_idx: int,
    args,
) -> Tuple[np.ndarray, List[int], float]:
    if args.mask_selection_mode == "similarity":
        scores = _score_row(output_dict, "pred_similarity", batch_idx, instruction_idx)
        max_idx = int(torch.argmax(scores).item())
        return segs_origin[:, :, max_idx], [max_idx], float(scores[max_idx].item())

    pred_iou = _score_row(output_dict, "pred_iou", batch_idx, instruction_idx)
    selected_ids = [int(i) for i in torch.nonzero(pred_iou > args.iou_threshold, as_tuple=False).flatten().tolist()]
    pred_seg = np.ones_like(segs_origin[:, :, 0])
    for idx in selected_ids:
        pred_seg = np.minimum(pred_seg, segs_origin[:, :, idx])
    score = float(pred_iou[selected_ids].max().item()) if selected_ids else 0.0
    return pred_seg.astype(np.uint8), selected_ids, score


def evaluate_loader(model, loader, args, split_name: str) -> List[Dict]:
    results = []
    seen_counts = defaultdict(int)
    torch_dtype = torch_dtype_from_precision(args.precision)

    for input_dict in tqdm(loader, desc=split_name.capitalize()):
        meta_list = input_dict.pop("meta_list")
        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=next(model.parameters()).device)
        input_dict["inference"] = True

        with torch.no_grad():
            output_dict = model(**input_dict)

        for batch_idx, meta in enumerate(meta_list):
            img_name = meta["img_name"]
            obj_name = meta["gt_object"]
            seen_counts[(img_name, obj_name)] += 1
            obj_count = seen_counts[(img_name, obj_name)]

            image_np = meta["image_np"]
            h, w = image_np.shape[:2]
            segs_origin = input_dict["origin_segs_list"][batch_idx]
            gt_masks = meta["gt_masks"]

            pred_masks = []
            ic_ious = []
            selected_gt_indices = []
            selected_candidate_indices = []

            for instr_idx, _instruction in enumerate(meta["instructions"]):
                pred_mask, selected_ids, score = select_prediction_mask(
                    output_dict,
                    segs_origin,
                    batch_idx,
                    instr_idx,
                    args,
                )
                if pred_mask.shape != (h, w):
                    pred_mask = cv2.resize(pred_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

                best_gt, iou, best_gt_idx = select_best_gt(pred_mask, gt_masks)
                pred_masks.append(pred_mask)
                ic_ious.append(iou)
                selected_gt_indices.append(best_gt_idx)
                selected_candidate_indices.append(selected_ids)

                if args.debug:
                    print(
                        f"[{split_name}] {img_name} instr={instr_idx} "
                        f"select={args.mask_selection_mode} cand={selected_ids} score={score:.6f} "
                        f"best_gt={best_gt_idx} iou={iou:.4f}",
                        flush=True,
                    )

                if args.save_vis:
                    vis_info = {**meta, "mask_selection_mode": args.mask_selection_mode}
                    save_visualization(
                        image_np=image_np,
                        pred_mask=pred_mask,
                        gt_mask=best_gt,
                        vis_dir=args.vis_dir,
                        sample_info=vis_info,
                        instruction_idx=instr_idx,
                        iou=iou,
                        obj_count=obj_count,
                    )

            iou_pairs = []
            if len(pred_masks) >= 3:
                iou_pairs.extend(
                    [
                        compute_iou(pred_masks[0], pred_masks[1]),
                        compute_iou(pred_masks[0], pred_masks[2]),
                        compute_iou(pred_masks[1], pred_masks[2]),
                    ]
                )
            elif len(pred_masks) == 2:
                iou_pairs.append(compute_iou(pred_masks[0], pred_masks[1]))

            results.append(
                {
                    "image_path": meta["image_path"],
                    "gt_mask_path": meta["gt_mask_path"],
                    "gt_object": meta["gt_object"],
                    "difficulty": meta["difficulty"],
                    "ic_ious": ic_ious,
                    "avg_ic_iou": float(np.mean(ic_ious)) if ic_ious else 0.0,
                    "iou_pairs": iou_pairs,
                    "num_instructions": len(meta["instructions"]),
                    "selected_gt_indices": selected_gt_indices,
                    "selected_candidate_indices": selected_candidate_indices,
                }
            )
    return results


def compute_metrics(result_list: List[Dict], icr_thresholds: List[float]) -> Dict:
    if not result_list:
        return {"ic_iou": 0.0, "count": 0, "icr_scores": {t: 0.0 for t in icr_thresholds}}

    avg_ic_iou = float(np.mean([r["avg_ic_iou"] for r in result_list]))
    icr_scores = {t: [] for t in icr_thresholds}
    for r in result_list:
        if len(r["iou_pairs"]) >= 3:
            for t in icr_thresholds:
                icr_scores[t].append(compute_icr_score(r["iou_pairs"], t))

    return {
        "ic_iou": avg_ic_iou,
        "count": len(result_list),
        "icr_scores": {t: float(np.mean(icr_scores[t])) if icr_scores[t] else 0.0 for t in icr_thresholds},
    }


def _instruction_values(result_list: List[Dict], instruction_indices: List[int]) -> Tuple[List[float], int]:
    values = []
    sample_count = 0
    for result in result_list:
        ic_ious = result.get("ic_ious", [])
        selected = [float(ic_ious[idx]) for idx in instruction_indices if idx < len(ic_ious)]
        if selected:
            sample_count += 1
            values.extend(selected)
    return values, sample_count


def compute_hard_instruction_metrics(hard_results: List[Dict]) -> Dict:
    first2_values, first2_samples = _instruction_values(hard_results, [0, 1])
    third_values, third_samples = _instruction_values(hard_results, [2])
    return {
        "first2_avg": float(np.mean(first2_values)) if first2_values else 0.0,
        "first2_count": len(first2_values),
        "first2_sample_count": first2_samples,
        "third_avg": float(np.mean(third_values)) if third_values else 0.0,
        "third_count": len(third_values),
        "third_sample_count": third_samples,
    }


def write_results(args, results: Dict[str, List[Dict]], icr_thresholds: List[float]):
    easy_metrics = compute_metrics(results["easy"], icr_thresholds)
    hard_metrics = compute_metrics(results["hard"], icr_thresholds)
    hard_instr_metrics = compute_hard_instruction_metrics(results["hard"])
    total_count = easy_metrics["count"] + hard_metrics["count"]
    if total_count > 0:
        avg_ic_iou = (
            easy_metrics["ic_iou"] * easy_metrics["count"] + hard_metrics["ic_iou"] * hard_metrics["count"]
        ) / total_count
        avg_icr_scores = {
            t: (
                easy_metrics["icr_scores"][t] * easy_metrics["count"]
                + hard_metrics["icr_scores"][t] * hard_metrics["count"]
            )
            / total_count
            for t in icr_thresholds
        }
    else:
        avg_ic_iou = 0.0
        avg_icr_scores = {t: 0.0 for t in icr_thresholds}

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir,
        f"llmseg_vigor_{args.checkpoint_tag}_{args.mask_selection_mode}_{timestamp}.txt",
    )

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  LLMSeg VIGOR-100K 测试结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"模型路径: {args.version}\n")
        f.write(f"微调权重: {args.checkpoint}\n")
        f.write(f"权重标签: {args.checkpoint_tag}\n")
        f.write(f"候选选择: {args.mask_selection_mode}\n")
        f.write(f"IoU 选择阈值: {args.iou_threshold}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"Depth 目录: {args.depth_dir or os.path.join(args.data_dir, 'depth')}\n")
        f.write(f"SAM masks: {args.sam_masks_dir}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        header = "| Category | Methods | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg |"
        for t in icr_thresholds:
            header += f" ICR@{t:.1f} Avg |"
        f.write(header + "\n")
        row = f"| Offline  | LLMSeg  | {easy_metrics['ic_iou']:.4f}      | {hard_metrics['ic_iou']:.4f}      | {avg_ic_iou:.4f}     |"
        for t in icr_thresholds:
            row += f" {avg_icr_scores[t]:.4f}      |"
        f.write(row + "\n\n")

        f.write(f"总样本数: {total_count}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n\n")
        f.write(f"IC-IoU Easy: {easy_metrics['ic_iou']:.4f}\n")
        f.write(f"IC-IoU Hard: {hard_metrics['ic_iou']:.4f}\n")
        f.write(f"IC-IoU Hard 前2条指令 Avg: {hard_instr_metrics['first2_avg']:.4f}\n")
        f.write(f"IC-IoU Hard 第3条指令 Avg: {hard_instr_metrics['third_avg']:.4f}\n")
        f.write(f"IC-IoU Avg:  {avg_ic_iou:.4f}\n\n")
        f.write("ICR@0.X:\n")
        for t in icr_thresholds:
            f.write(f"  ICR@{t:.1f}: Easy={easy_metrics['icr_scores'][t]:.4f}, Hard={hard_metrics['icr_scores'][t]:.4f}, Avg={avg_icr_scores[t]:.4f}\n")

        for difficulty in ["easy", "hard"]:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"  {difficulty.capitalize()} 样本详细结果\n")
            f.write("=" * 80 + "\n\n")
            for i, r in enumerate(results[difficulty]):
                f.write(f"[{difficulty.capitalize()} Sample {i + 1}]\n")
                f.write(f"  Image: {r['image_path']}\n")
                f.write(f"  GT Mask: {r['gt_mask_path']}\n")
                f.write(f"  Object: {r['gt_object']}\n")
                f.write(f"  Selected candidates: {r['selected_candidate_indices']}\n")
                f.write(f"  Selected GT indices: {r['selected_gt_indices']}\n")
                f.write(f"  IC-IoU (per instruction): {[f'{x:.4f}' for x in r['ic_ious']]}\n")
                f.write(f"  Avg IC-IoU: {r['avg_ic_iou']:.4f}\n")
                f.write(f"  IoU pairs: {[f'{x:.4f}' for x in r['iou_pairs']]}\n")
                for t in icr_thresholds:
                    f.write(f"  ICR@{t:.1f}: {compute_icr_score(r['iou_pairs'], t)}\n")
                f.write("\n")

    print(f"\n结果已保存到: {result_file}")
    print("\n" + "=" * 60)
    print("  测试完成 - 结果汇总")
    print("=" * 60)
    print(f"权重: {args.checkpoint_tag} | 选择策略: {args.mask_selection_mode}")
    print(f"总样本数: {total_count}")
    print(f"IC-IoU: Easy={easy_metrics['ic_iou']:.4f}, Hard={hard_metrics['ic_iou']:.4f}, Avg={avg_ic_iou:.4f}")


def main(cli_args):
    args = parse_args(cli_args)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(args.device.split(":")[-1]))
        except Exception:
            pass

    icr_thresholds = [float(t) for t in args.icr_thresholds.split(",")]
    if args.save_vis:
        args.vis_dir = os.path.join(args.vis_dir, args.checkpoint_tag, args.mask_selection_mode)
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"\n  可视化输出目录: {args.vis_dir}")
    if not os.path.exists(args.sam_masks_dir):
        raise FileNotFoundError(f"SAM 候选 mask 目录不存在: {args.sam_masks_dir}")

    model, tokenizer, clip_image_processor, transform = load_model(args)

    print("=" * 60)
    print("  加载测试数据...")
    print("=" * 60)
    easy_samples = load_samples(args.data_dir, "easy", args.vigor_easy_json_file, args.vigor_hard_json_file) if args.split in ["easy", "both"] else []
    hard_samples = load_samples(args.data_dir, "hard", args.vigor_easy_json_file, args.vigor_hard_json_file) if args.split in ["hard", "both"] else []
    if args.max_samples is not None:
        easy_samples = easy_samples[: args.max_samples]
        hard_samples = hard_samples[: args.max_samples]
    print(f"  测试模式: {args.split}")
    print(f"  Easy JSON: {args.vigor_easy_json_file}")
    print(f"  Hard JSON: {args.vigor_hard_json_file}")
    print(f"  Easy 样本数: {len(easy_samples)}")
    print(f"  Hard 样本数: {len(hard_samples)}")
    print(f"  SAM 候选 mask 目录: {args.sam_masks_dir}")
    print(f"  候选选择策略: {args.mask_selection_mode}")
    print(f"  DataLoader workers: {args.workers}")

    def make_loader(samples):
        dataset = VIGORTestDataset(samples, args, clip_image_processor, transform)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
            collate_fn=lambda batch: collate_vigor_test(
                batch,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
            ),
        )

    results = {"easy": [], "hard": []}
    if easy_samples:
        print("\n" + "=" * 60)
        print("  测试 Easy 样本")
        print("=" * 60)
        results["easy"] = evaluate_loader(model, make_loader(easy_samples), args, "easy")
    if hard_samples:
        print("\n" + "=" * 60)
        print("  测试 Hard 样本")
        print("=" * 60)
        results["hard"] = evaluate_loader(model, make_loader(hard_samples), args, "hard")

    write_results(args, results, icr_thresholds)


if __name__ == "__main__":
    main(sys.argv[1:])
