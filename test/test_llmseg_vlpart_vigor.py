#!/usr/bin/env python
"""Two-stage VIGOR evaluation: LLMSeg mask -> region RGB -> VLPart mask.

This file intentionally lives next to the existing test scripts and imports
their helpers without modifying them.
"""

import argparse
import gc
import importlib.util
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLPART_ROOT = PROJECT_ROOT / "VLPart"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_llmseg_module():
    sys.path.insert(0, str(PROJECT_ROOT))
    return load_module("llmseg_vigor_base", PROJECT_ROOT / "test" / "test_llmseg_vigor.py")


def load_vlpart_module():
    sys.path.insert(0, str(VLPART_ROOT))
    sys.path.insert(0, str(VLPART_ROOT / "demo"))
    return load_module("vlpart_vigor_base", VLPART_ROOT / "tools" / "test_vigor_vlpart.py")


def parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB value as R,G,B, got: {value}")
    rgb = tuple(int(item) for item in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"RGB values must be in [0, 255], got: {value}")
    return rgb


def has_existing_outputs(path: str) -> bool:
    output_path = Path(path)
    if not output_path.exists():
        return False
    if output_path.is_file():
        return True
    try:
        next(output_path.iterdir())
        return True
    except StopIteration:
        return False


def timestamped_sibling(path: str, timestamp: str) -> str:
    output_path = Path(path)
    candidate = output_path.with_name(f"{output_path.name}_{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = output_path.with_name(f"{output_path.name}_{timestamp}_{suffix}")
        suffix += 1
    return str(candidate)


def resolve_run_output_dirs(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.save_vis and has_existing_outputs(args.vis_dir):
        old_dir = args.vis_dir
        args.vis_dir = timestamped_sibling(args.vis_dir, timestamp)
        print(f"  Existing visualizations found; writing this run to: {args.vis_dir} (was {old_dir})")
    if args.save_pred_masks and has_existing_outputs(args.vlpart_pred_masks_dir):
        old_dir = args.vlpart_pred_masks_dir
        args.vlpart_pred_masks_dir = timestamped_sibling(args.vlpart_pred_masks_dir, timestamp)
        print(f"  Existing VLPart pred masks found; writing this run to: {args.vlpart_pred_masks_dir} (was {old_dir})")


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("._")
    return (text or "sample")[:max_len]


def to_vigor_mask(mask: np.ndarray) -> np.ndarray:
    """Return uint8 mask using VIGOR convention: 0 foreground, 255 background."""
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return np.where(mask, 0, 255).astype(np.uint8)
    if mask.max(initial=0) <= 1:
        return np.where(mask <= 0.5, 0, 255).astype(np.uint8)
    return np.where(mask < 128, 0, 255).astype(np.uint8)


def project_mask_to_region_rgb(
    image_rgb: np.ndarray,
    vigor_mask: np.ndarray,
    background_rgb: Tuple[int, int, int],
) -> np.ndarray:
    if vigor_mask.shape != image_rgb.shape[:2]:
        h, w = image_rgb.shape[:2]
        vigor_mask = cv2.resize(vigor_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    fg = vigor_mask == 0
    background = np.zeros_like(image_rgb, dtype=np.uint8)
    background[:, :] = np.array(background_rgb, dtype=np.uint8)
    region_rgb = background
    region_rgb[fg] = image_rgb[fg]
    return region_rgb


def overlay_vigor_mask(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    overlay = image_rgb.copy()
    if mask is None:
        return overlay

    h, w = image_rgb.shape[:2]
    vigor_mask = to_vigor_mask(mask)
    if vigor_mask.shape != (h, w):
        vigor_mask = cv2.resize(vigor_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    fg = vigor_mask == 0
    if fg.any():
        color = np.array(color_rgb, dtype=np.float32)
        overlay[fg] = (
            image_rgb[fg].astype(np.float32) * (1.0 - alpha) + color * alpha
        ).astype(np.uint8)
    return overlay


def load_gt_masks(mask_paths: List[str], image_shape: Tuple[int, int]) -> List[np.ndarray]:
    gt_masks = []
    target_h, target_w = image_shape
    for mask_path in mask_paths:
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Failed to read GT mask: {mask_path}")
        if gt.shape != (target_h, target_w):
            gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        gt_masks.append(gt)
    return gt_masks


def output_stem(sample: Dict, obj_count: int, instruction_idx: int) -> str:
    img_stem = Path(sample.get("img_name", sample.get("image_path", "image"))).stem
    obj = safe_name(sample.get("gt_object", "object"))
    return f"{safe_name(img_stem)}_{obj}_{obj_count}_instr{instruction_idx}"


def compute_icr_score(iou_pairs: List[float], threshold: float) -> int:
    return sum(1 for iou in iou_pairs if iou >= threshold)


def compute_metrics(result_list: List[Dict], icr_thresholds: List[float]) -> Dict:
    if not result_list:
        return {
            "ic_iou": 0.0,
            "count": 0,
            "icr_scores": {t: 0.0 for t in icr_thresholds},
        }

    ic_ious = [result["avg_ic_iou"] for result in result_list]
    icr_scores = {t: [] for t in icr_thresholds}
    for result in result_list:
        if len(result["iou_pairs"]) >= 3:
            for threshold in icr_thresholds:
                icr_scores[threshold].append(
                    compute_icr_score(result["iou_pairs"], threshold)
                )

    return {
        "ic_iou": float(np.mean(ic_ious)),
        "count": len(result_list),
        "icr_scores": {
            threshold: float(np.mean(scores)) if scores else 0.0
            for threshold, scores in icr_scores.items()
        },
    }


def weighted_average(easy_metrics: Dict, hard_metrics: Dict, key: str) -> float:
    total_count = easy_metrics["count"] + hard_metrics["count"]
    if total_count == 0:
        return 0.0
    return (
        easy_metrics[key] * easy_metrics["count"]
        + hard_metrics[key] * hard_metrics["count"]
    ) / total_count


def compute_instruction_subset_metrics(result_list: List[Dict], start: int, end: int) -> Dict:
    values = []
    sample_count = 0
    for result in result_list:
        subset = result.get("ic_ious", [])[start:end]
        if subset:
            sample_count += 1
            values.extend(subset)
    return {
        "ic_iou": float(np.mean(values)) if values else 0.0,
        "sample_count": sample_count,
        "instruction_count": len(values),
    }


def normalize_mask_panel(mask: np.ndarray, foreground_rgb: Tuple[int, int, int]) -> np.ndarray:
    vigor_mask = to_vigor_mask(mask)
    panel = np.full((vigor_mask.shape[0], vigor_mask.shape[1], 3), 255, dtype=np.uint8)
    panel[vigor_mask == 0] = np.array(foreground_rgb, dtype=np.uint8)
    return panel


def save_gt_pred_mask_comparison(
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    vis_dir: str,
    sample_info: Dict,
    instruction_idx: int,
    iou: float,
    obj_count: int = 1,
) -> str:
    """Save one original-layout GT vs VLPart comparison overlaid on RGB."""
    h, w = image_rgb.shape[:2]
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask is not None and gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_overlay = overlay_vigor_mask(image_rgb, gt_mask, (0, 255, 0), alpha=0.5)
    pred_overlay = overlay_vigor_mask(image_rgb, pred_mask, (255, 0, 0), alpha=0.5)
    combined_image = np.hstack([gt_overlay, pred_overlay])

    gt_object = sample_info.get("gt_object", "Unknown")
    difficulty = sample_info.get("difficulty", "unknown")
    instructions = sample_info.get("instructions", [])
    instruction = instructions[instruction_idx] if instruction_idx < len(instructions) else ""
    img_name = sample_info.get("img_name", "unknown")

    text_height = 100
    combined_with_text = np.ones(
        (combined_image.shape[0] + text_height, combined_image.shape[1], 3),
        dtype=np.uint8,
    ) * 255
    combined_with_text[text_height:, :] = combined_image

    title_text = f"GT Object: {gt_object} | Difficulty: {difficulty}"
    cv2.putText(combined_with_text, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    instr_display = instruction[:80] + "..." if len(instruction) > 80 else instruction
    cv2.putText(combined_with_text, f"Instruction: {instr_display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(combined_with_text, f"IoU: {iou:.4f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(combined_with_text, "GT overlay (Green)", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
    cv2.putText(combined_with_text, "VLPart overlay (Red)", (390, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    difficulty_dir = Path(vis_dir) / difficulty
    difficulty_dir.mkdir(parents=True, exist_ok=True)
    gt_object_clean = safe_name(gt_object, max_len=80)
    output_filename = f"{img_name}_{gt_object_clean}_{obj_count}_instr{instruction_idx}_iou{iou:.3f}.png"
    output_path = difficulty_dir / output_filename

    cv2.imwrite(str(output_path), cv2.cvtColor(combined_with_text, cv2.COLOR_RGB2BGR))
    return str(output_path)


def write_results(
    args: argparse.Namespace,
    results: Dict[str, List[Dict]],
    skipped_errors: Dict[str, List[Tuple[str, str]]],
) -> str:
    icr_thresholds = [float(t) for t in args.icr_thresholds.split(",")]
    easy_metrics = compute_metrics(results["easy"], icr_thresholds)
    hard_metrics = compute_metrics(results["hard"], icr_thresholds)
    hard_first2_metrics = compute_instruction_subset_metrics(results["hard"], 0, 2)
    hard_third_metrics = compute_instruction_subset_metrics(results["hard"], 2, 3)
    total_count = easy_metrics["count"] + hard_metrics["count"]
    avg_ic_iou = weighted_average(easy_metrics, hard_metrics, "ic_iou")

    avg_icr_scores = {}
    for threshold in icr_thresholds:
        if total_count > 0:
            avg_icr_scores[threshold] = (
                easy_metrics["icr_scores"][threshold] * easy_metrics["count"]
                + hard_metrics["icr_scores"][threshold] * hard_metrics["count"]
            ) / total_count
        else:
            avg_icr_scores[threshold] = 0.0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = str(Path(args.output_dir) / f"llmseg_vlpart_vigor_test_results_{timestamp}.txt")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  LLMSeg+VLPart VIGOR-100K 测试结果\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"模型路径: {args.version}\n")
        f.write(f"微调权重: {args.checkpoint}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"SAM masks: {args.sam_masks_dir}\n")
        f.write(f"VLPart config: {args.vlpart_config_file}\n")
        f.write(f"VLPart weights: {args.vlpart_weights}\n")
        f.write(f"VLPart vocabulary: {args.vlpart_vocabulary}\n")
        f.write("VLPart input mode: region_rgb\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("  汇总结果表格\n")
        f.write("=" * 80 + "\n\n")

        header = "| Category | Methods | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg |"
        for threshold in icr_thresholds:
            header += f" ICR@{threshold:.1f} Avg |"
        f.write(header + "\n")

        row = (
            f"| Offline  | LLMSeg+VLPart | {easy_metrics['ic_iou']:.4f}      | "
            f"{hard_metrics['ic_iou']:.4f}      | {avg_ic_iou:.4f}     |"
        )
        for threshold in icr_thresholds:
            row += f" {avg_icr_scores[threshold]:.4f}      |"
        f.write(row + "\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  总体 IC-IoU 和 ICR@0.X\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总样本数: {total_count}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n")
        f.write(
            f"跳过样本数: Easy={len(skipped_errors['easy'])}, "
            f"Hard={len(skipped_errors['hard'])}\n"
        )
        if skipped_errors["easy"] or skipped_errors["hard"]:
            f.write("跳过样本错误示例:\n")
            for split_name in ["easy", "hard"]:
                for path, err in skipped_errors[split_name][:5]:
                    f.write(f"  [{split_name}] {path}: {err}\n")
        f.write("\n")

        f.write("IC-IoU:\n")
        f.write(f"  Easy:    {easy_metrics['ic_iou']:.4f}\n")
        f.write(f"  Hard:    {hard_metrics['ic_iou']:.4f}\n")
        f.write(f"  Average: {avg_ic_iou:.4f}\n\n")

        f.write("Hard instruction split IC-IoU:\n")
        f.write(
            f"  Hard First2: {hard_first2_metrics['ic_iou']:.4f} "
            f"({hard_first2_metrics['instruction_count']} instructions / "
            f"{hard_first2_metrics['sample_count']} samples)\n"
        )
        f.write(
            f"  Hard Third:  {hard_third_metrics['ic_iou']:.4f} "
            f"({hard_third_metrics['instruction_count']} instructions / "
            f"{hard_third_metrics['sample_count']} samples)\n\n"
        )

        f.write("ICR@0.X:\n")
        for threshold in icr_thresholds:
            f.write(f"  ICR@{threshold:.1f}:\n")
            f.write(f"    Easy:    {easy_metrics['icr_scores'][threshold]:.4f}\n")
            f.write(f"    Hard:    {hard_metrics['icr_scores'][threshold]:.4f}\n")
            f.write(f"    Average: {avg_icr_scores[threshold]:.4f}\n")

        for split_name, split_title in [("easy", "Easy"), ("hard", "Hard")]:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"  {split_title} 样本详细结果\n")
            f.write("=" * 80 + "\n\n")
            for index, result in enumerate(results[split_name], start=1):
                f.write(f"[{split_title} Sample {index}]\n")
                f.write(f"  Image: {result['image_path']}\n")
                f.write(f"  GT Mask: {result['gt_mask_path']}\n")
                f.write(f"  Object: {result['gt_object']}\n")
                f.write(f"  IC-IoU (per instruction): {[f'{x:.4f}' for x in result['ic_ious']]}\n")
                if split_name == "hard":
                    hard_first2 = result["ic_ious"][:2]
                    hard_third = result["ic_ious"][2:3]
                    f.write(f"  Hard First2 IC-IoU: {[f'{x:.4f}' for x in hard_first2]}\n")
                    f.write(f"  Hard Third IC-IoU: {[f'{x:.4f}' for x in hard_third]}\n")
                f.write(f"  Avg IC-IoU: {result['avg_ic_iou']:.4f}\n")
                f.write(f"  IoU pairs: {[f'{x:.4f}' for x in result['iou_pairs']]}\n")
                for threshold in icr_thresholds:
                    score = compute_icr_score(result["iou_pairs"], threshold)
                    f.write(f"  ICR@{threshold:.1f}: {score}\n")
                f.write("\n")

    return result_file


def run_success_rate_analysis(result_file: str) -> str:
    analyzer = load_module(
        "ic_iou_success_rate_analyzer",
        PROJECT_ROOT / "test" / "analyze_ic_iou_success_rate.py",
    )
    output_file = result_file.replace(".txt", "_ic_iou_success_rate.txt")
    analyzer.analyze_results(result_file, output_file)
    return output_file


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLMSeg -> region RGB -> VLPart VIGOR evaluator"
    )

    parser.add_argument("--version", default="/opt/data/private/model/LISA_Plus_7b")
    parser.add_argument(
        "--checkpoint",
        default="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple-newdata/ckpt_model/epoch_20",
    )
    parser.add_argument("--vision_tower", default="/opt/data/private/model/clip-vit-large-patch14")
    parser.add_argument(
        "--vision_pretrained",
        default="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--data_dir", default="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test")
    parser.add_argument("--easy_json_file", default="open_vocab_grasp_easy_object_2.json")
    parser.add_argument("--hard_json_file", default="open_vocab_grasp_hard_object_2.json")
    parser.add_argument(
        "--sam_masks_dir",
        default="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test_mask/sam_masks3",
    )

    parser.add_argument("--output_dir", default="./result_llmseg_vlpart")
    parser.add_argument("--vis_dir", default="./vis_output_llmseg_vlpart")
    parser.add_argument("--region_rgb_dir", default="./result_llmseg_vlpart/region_rgb")
    parser.add_argument("--llmseg_masks_dir", default="./result_llmseg_vlpart/llmseg_masks")
    parser.add_argument("--vlpart_pred_masks_dir", default="./result_llmseg_vlpart/vlpart_pred_masks")
    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--save_pred_masks", action="store_true", default=True)
    parser.add_argument("--save_region_rgb", action="store_true", default=True)
    parser.add_argument("--skip_success_rate_analysis", action="store_true", default=False)
    parser.add_argument("--region_background_rgb", default="0,0,0")
    parser.add_argument("--reuse_region_inputs", dest="reuse_region_inputs", action="store_true", default=True)
    parser.add_argument("--no_reuse_region_inputs", dest="reuse_region_inputs", action="store_false")

    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=896, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--icr_thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--split", default="both", choices=["easy", "hard", "both"])
    parser.add_argument("--workers", default=12, type=int)

    parser.add_argument("--vlpart_config_file", default="configs/vigor/swinbase_vigor_easy_stage1.yaml")
    parser.add_argument(
        "--vlpart_weights",
        default="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_easy_stage1_bs16_lr4e-5/model_final.pth",
    )
    parser.add_argument("--vlpart_device", default="cuda:0")
    parser.add_argument("--vlpart_confidence_threshold", default=0.05, type=float)
    parser.add_argument("--vlpart_mask_selection", default="top1", choices=["top1", "union"])
    parser.add_argument("--vlpart_prediction_cache_size", default=256, type=int)
    parser.add_argument(
        "--vlpart_vocabulary",
        default="custom",
        choices=[
            "pascal_part",
            "partimagenet",
            "paco",
            "voc",
            "coco",
            "lvis",
            "pascal_part_voc",
            "lvis_paco",
            "custom",
        ],
    )
    parser.add_argument(
        "--vlpart_custom_vocabulary",
        default="cylindrical side surface,hexagonal side face,flat side surface,whole object",
    )
    parser.add_argument(
        "--vlpart_opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Additional VLPart config options in KEY VALUE pairs. Keep this last.",
    )
    return parser.parse_args(argv)


def load_split_samples(llmseg, args: argparse.Namespace) -> Dict[str, List[Dict]]:
    samples = {"easy": [], "hard": []}
    if args.split in ["easy", "both"]:
        samples["easy"] = llmseg.load_samples(args.data_dir, args.easy_json_file, "easy")
    if args.split in ["hard", "both"]:
        samples["hard"] = llmseg.load_samples(args.data_dir, args.hard_json_file, "hard")
    if args.max_samples is not None:
        samples["easy"] = samples["easy"][: args.max_samples]
        samples["hard"] = samples["hard"][: args.max_samples]
    return samples


def read_rgb_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def maybe_load_cached_region_inputs(
    llmseg,
    args: argparse.Namespace,
) -> Optional[Tuple[Dict[str, List[Dict]], Dict[str, List[Tuple[str, str]]]]]:
    if not args.reuse_region_inputs:
        return None

    region_dir = Path(args.region_rgb_dir)
    mask_dir = Path(args.llmseg_masks_dir)
    if not (region_dir.exists() and mask_dir.exists()):
        return None

    print("\n" + "=" * 60)
    print("  Existing region_rgb and llmseg_masks found; checking cache")
    print("=" * 60)
    samples = load_split_samples(llmseg, args)
    items = {"easy": [], "hard": []}
    skipped_errors = {"easy": [], "hard": []}
    missing = []

    for split_name in ["easy", "hard"]:
        seen_counts = defaultdict(int)
        for sample in samples[split_name]:
            try:
                img_name = sample["img_name"]
                obj_name = sample["gt_object"]
                seen_counts[(img_name, obj_name)] += 1
                obj_count = seen_counts[(img_name, obj_name)]
                item = {
                    "sample": sample,
                    "obj_count": obj_count,
                    "image_shape": None,
                    "records": [],
                }

                for instruction_idx, instruction in enumerate(sample.get("instructions", [])[:3]):
                    stem = output_stem(sample, obj_count, instruction_idx)
                    mask_path = Path(args.llmseg_masks_dir) / split_name / f"{stem}.png"
                    region_path = Path(args.region_rgb_dir) / split_name / f"{stem}.png"
                    if not mask_path.is_file():
                        missing.append(str(mask_path))
                        continue
                    if not region_path.is_file():
                        missing.append(str(region_path))
                        continue

                    item["records"].append(
                        {
                            "instruction_idx": instruction_idx,
                            "instruction": instruction,
                            "llmseg_mask_path": str(mask_path),
                            "region_path": str(region_path),
                            "stem": stem,
                        }
                    )

                if item["records"]:
                    items[split_name].append(item)
            except Exception as exc:
                image_path = sample.get("image_path", "unknown")
                skipped_errors[split_name].append((image_path, repr(exc)))
                if len(skipped_errors[split_name]) <= 5 or args.debug:
                    print(f"  [Cache {split_name} Error] {image_path}: {exc}")

    if missing:
        print(f"  Cache is incomplete ({len(missing)} missing files); regenerating Stage 1.")
        for missing_path in missing[:5]:
            print(f"    missing: {missing_path}")
        return None

    easy_count = len(items["easy"])
    hard_count = len(items["hard"])
    print(f"  Reusing cached inputs: Easy={easy_count}, Hard={hard_count}")
    print(f"  VLPart input mode: region RGB ({args.region_rgb_dir})")
    return items, skipped_errors


def generate_region_inputs(llmseg, args: argparse.Namespace) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Tuple[str, str]]]]:
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    background_rgb = parse_rgb(args.region_background_rgb)
    Path(args.region_rgb_dir).mkdir(parents=True, exist_ok=True)
    Path(args.llmseg_masks_dir).mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)

    if not Path(args.sam_masks_dir).exists():
        raise FileNotFoundError(f"SAM masks dir does not exist: {args.sam_masks_dir}")

    sam_mask_helper = llmseg.SAM_Mask_Reader_PNG(args.sam_masks_dir)
    model, tokenizer, clip_image_processor, transform, _ = llmseg.load_model(args)
    samples = load_split_samples(llmseg, args)
    easy_count = len(samples["easy"])
    hard_count = len(samples["hard"])

    print(f"  Easy samples: {easy_count}")
    print(f"  Hard samples: {hard_count}")
    print(f"  Region RGB dir: {args.region_rgb_dir}")
    print(f"  LLMSeg mask dir: {args.llmseg_masks_dir}")

    items = {"easy": [], "hard": []}
    skipped_errors = {"easy": [], "hard": []}

    try:
        for split_name in ["easy", "hard"]:
            split_samples = samples[split_name]
            if not split_samples:
                continue

            print("\n" + "=" * 60)
            print(f"  Stage 1: LLMSeg masks -> region RGB ({split_name})")
            print("=" * 60)
            seen_counts = defaultdict(int)
            loader = llmseg.build_test_loader(
                split_samples,
                clip_image_processor,
                transform,
                sam_mask_helper,
                args,
            )

            for prepared_sample in tqdm(loader, desc=f"LLMSeg {split_name.capitalize()}"):
                sample = prepared_sample.get("sample", {})
                try:
                    if prepared_sample.get("error"):
                        raise RuntimeError(prepared_sample["error"])

                    img_name = sample["img_name"]
                    obj_name = sample["gt_object"]
                    seen_counts[(img_name, obj_name)] += 1
                    obj_count = seen_counts[(img_name, obj_name)]

                    image_rgb = prepared_sample["image_np"]
                    h, w = image_rgb.shape[:2]
                    item = {
                        "sample": sample,
                        "obj_count": obj_count,
                        "image_shape": (h, w),
                        "records": [],
                    }

                    for instruction_idx, instruction in enumerate(sample.get("instructions", [])[:3]):
                        pred_mask = llmseg.predict_mask_llmseg(
                            model, tokenizer, prepared_sample, instruction, args
                        )
                        if pred_mask.shape != (h, w):
                            pred_mask = cv2.resize(
                                pred_mask.astype(np.float32),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        vigor_mask = to_vigor_mask(pred_mask)
                        stem = output_stem(sample, obj_count, instruction_idx)

                        mask_path = Path(args.llmseg_masks_dir) / split_name / f"{stem}.png"
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(mask_path), vigor_mask)

                        region_rgb = project_mask_to_region_rgb(
                            image_rgb=image_rgb,
                            vigor_mask=vigor_mask,
                            background_rgb=background_rgb,
                        )
                        region_path = Path(args.region_rgb_dir) / split_name / f"{stem}.png"
                        if args.save_region_rgb:
                            region_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(region_path), cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR))
                        else:
                            raise ValueError("--save_region_rgb must remain enabled for VLPart file input")

                        item["records"].append(
                            {
                                "instruction_idx": instruction_idx,
                                "instruction": instruction,
                                "llmseg_mask_path": str(mask_path),
                                "region_path": str(region_path),
                                "stem": stem,
                            }
                        )

                    if item["records"]:
                        items[split_name].append(item)
                except Exception as exc:
                    image_path = sample.get("image_path", "unknown")
                    skipped_errors[split_name].append((image_path, repr(exc)))
                    if len(skipped_errors[split_name]) <= 5 or args.debug:
                        print(f"  [Stage1 {split_name} Error] {image_path}: {exc}")
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return items, skipped_errors


def normalize_custom_vocabulary(text: str) -> str:
    return ",".join(item.strip() for item in text.split(",") if item.strip())


def make_vlpart_args(args: argparse.Namespace) -> SimpleNamespace:
    custom_vocabulary = normalize_custom_vocabulary(args.vlpart_custom_vocabulary)
    if args.vlpart_vocabulary == "custom" and not custom_vocabulary:
        raise ValueError("--vlpart_custom_vocabulary must not be empty")
    return SimpleNamespace(
        config_file=args.vlpart_config_file,
        weights=args.vlpart_weights,
        device=args.vlpart_device,
        confidence_threshold=args.vlpart_confidence_threshold,
        vocabulary=args.vlpart_vocabulary,
        custom_vocabulary=custom_vocabulary,
        mask_selection=args.vlpart_mask_selection,
        opts=args.vlpart_opts,
    )


def evaluate_with_vlpart(
    llmseg,
    vlpart_eval,
    items: Dict[str, List[Dict]],
    skipped_errors: Dict[str, List[Tuple[str, str]]],
    args: argparse.Namespace,
) -> Dict[str, List[Dict]]:
    if args.vlpart_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.vlpart_device))

    if args.save_pred_masks:
        Path(args.vlpart_pred_masks_dir).mkdir(parents=True, exist_ok=True)

    vlpart_args = make_vlpart_args(args)
    cfg = vlpart_eval.setup_cfg(vlpart_args)
    demo = vlpart_eval.VisualizationDemo(cfg, vlpart_args)
    pred_cache = vlpart_eval.PredictionCache(args.vlpart_prediction_cache_size)
    results = {"easy": [], "hard": []}

    for split_name in ["easy", "hard"]:
        split_items = items[split_name]
        if not split_items:
            continue

        print("\n" + "=" * 60)
        print(f"  Stage 2: VLPart region RGB inference ({split_name})")
        print("=" * 60)

        for item in tqdm(split_items, desc=f"VLPart {split_name.capitalize()}"):
            sample = item["sample"]
            try:
                image_rgb = read_rgb_image(sample["image_path"])
                h, w = image_rgb.shape[:2]
                gt_masks = load_gt_masks(sample["gt_mask_paths"], (h, w))

                pred_masks = []
                ic_ious = []

                for record in item["records"]:
                    region_path = record["region_path"]
                    cached = pred_cache.get(region_path)
                    if cached is None:
                        pred_fg, pred_info, _ = vlpart_eval.predict_one_image(
                            demo,
                            region_path,
                            args.vlpart_mask_selection,
                        )
                        pred_cache.put(region_path, (pred_fg, pred_info))
                    else:
                        pred_fg, pred_info = cached

                    final_mask = vlpart_eval.selected_mask_to_vigor_mask(pred_fg)
                    if final_mask.shape != (h, w):
                        final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    pred_masks.append(final_mask)
                    iou, best_gt = llmseg.compute_iou_max(final_mask, gt_masks)
                    ic_ious.append(iou)

                    if args.save_pred_masks:
                        pred_mask_path = (
                            Path(args.vlpart_pred_masks_dir)
                            / split_name
                            / f"{record['stem']}.png"
                        )
                        pred_mask_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(pred_mask_path), final_mask)

                    if args.save_vis:
                        sample_info = {
                            "gt_object": sample["gt_object"],
                            "difficulty": sample["difficulty"],
                            "instructions": sample.get("instructions", []),
                            "img_name": sample["img_name"],
                        }
                        save_gt_pred_mask_comparison(
                            image_rgb=image_rgb,
                            pred_mask=final_mask,
                            gt_mask=best_gt,
                            vis_dir=args.vis_dir,
                            sample_info=sample_info,
                            instruction_idx=record["instruction_idx"],
                            iou=iou,
                            obj_count=item["obj_count"],
                        )

                    if args.debug:
                        cls_name = pred_info.get("pred_class_name", "")
                        score = pred_info.get("pred_score", 0.0)
                        print(
                            f"    {record['stem']}: IoU={iou:.4f}, "
                            f"class={cls_name}, score={score:.4f}"
                        )

                iou_pairs = []
                if len(pred_masks) >= 3:
                    iou_pairs.append(llmseg.compute_iou(pred_masks[0], pred_masks[1]))
                    iou_pairs.append(llmseg.compute_iou(pred_masks[0], pred_masks[2]))
                    iou_pairs.append(llmseg.compute_iou(pred_masks[1], pred_masks[2]))
                elif len(pred_masks) == 2:
                    iou_pairs.append(llmseg.compute_iou(pred_masks[0], pred_masks[1]))

                results[split_name].append(
                    {
                        "image_path": sample["image_path"],
                        "gt_mask_path": sample["gt_mask_paths"][0],
                        "gt_object": sample["gt_object"],
                        "difficulty": sample["difficulty"],
                        "ic_ious": ic_ious,
                        "avg_ic_iou": float(np.mean(ic_ious)) if ic_ious else 0.0,
                        "iou_pairs": iou_pairs,
                        "num_instructions": len(ic_ious),
                    }
                )
            except Exception as exc:
                skipped_errors[split_name].append((sample.get("image_path", "unknown"), repr(exc)))
                if len(skipped_errors[split_name]) <= 5 or args.debug:
                    print(f"  [Stage2 {split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    return results


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    resolve_run_output_dirs(args)

    print("=" * 72)
    print("  LLMSeg -> region RGB -> VLPart VIGOR evaluation")
    print("=" * 72)
    print(f"Data dir: {args.data_dir}")
    print(f"LLMSeg checkpoint: {args.checkpoint}")
    print(f"VLPart weights: {args.vlpart_weights}")
    print(f"Split: {args.split}")
    print(f"Output dir: {args.output_dir}")
    print(f"Visualization dir: {args.vis_dir}")
    print(f"VLPart pred masks dir: {args.vlpart_pred_masks_dir}")
    print(f"VLPart input mode: region RGB ({args.region_rgb_dir})")
    print("=" * 72)

    llmseg = load_llmseg_module()
    cached_inputs = maybe_load_cached_region_inputs(llmseg, args)
    if cached_inputs is None:
        items, skipped_errors = generate_region_inputs(llmseg, args)
    else:
        items, skipped_errors = cached_inputs

    vlpart_eval = load_vlpart_module()
    results = evaluate_with_vlpart(llmseg, vlpart_eval, items, skipped_errors, args)
    result_file = write_results(args, results, skipped_errors)
    success_rate_file = ""
    if not args.skip_success_rate_analysis:
        success_rate_file = run_success_rate_analysis(result_file)

    icr_thresholds = [float(t) for t in args.icr_thresholds.split(",")]
    easy_metrics = compute_metrics(results["easy"], icr_thresholds)
    hard_metrics = compute_metrics(results["hard"], icr_thresholds)
    avg_iou = weighted_average(easy_metrics, hard_metrics, "ic_iou")

    print("\n" + "=" * 60)
    print("  LLMSeg+VLPart evaluation finished")
    print("=" * 60)
    print(f"Result file: {result_file}")
    print(
        f"IC-IoU: Easy={easy_metrics['ic_iou']:.4f}, "
        f"Hard={hard_metrics['ic_iou']:.4f}, Avg={avg_iou:.4f}"
    )
    if success_rate_file:
        print(f"Success-rate file: {success_rate_file}")


if __name__ == "__main__":
    main()
