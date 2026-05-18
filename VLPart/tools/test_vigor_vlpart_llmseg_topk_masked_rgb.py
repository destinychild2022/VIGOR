import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


VLPART_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VLPART_ROOT.parent
sys.path.insert(0, str(VLPART_ROOT))
sys.path.insert(0, str(VLPART_ROOT / "demo"))

from detectron2.data.detection_utils import read_image  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from predictor import VisualizationDemo  # noqa: E402
from test_vigor_vlpart import (  # noqa: E402
    PredictionCache,
    choose_single_mask,
    compute_iou_from_foreground,
    compute_metrics,
    load_gt_masks,
    normalize_custom_vocabulary,
    resolve_data_path,
    resolve_repo_path,
    safe_name,
    save_prediction_overlay,
    save_raw_pred_mask,
    setup_cfg,
    split_path_list,
    weighted_average,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "VLPart VIGOR evaluation with LLMSeg top-K object masks as masked RGB inputs"
    )
    parser.add_argument("--config-file", required=True, help="VLPart config file.")
    parser.add_argument("--weights", required=True, help="VLPart checkpoint path.")
    parser.add_argument(
        "--data_dir",
        default="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test",
        help="Dataset root that contains scene images and JSON files.",
    )
    parser.add_argument(
        "--easy_json_file",
        default="open_vocab_grasp_easy_object_mix.json",
        help="Easy split JSON filename under data_dir.",
    )
    parser.add_argument(
        "--hard_json_file",
        default="open_vocab_grasp_hard_object_mix.json",
        help="Hard split JSON filename under data_dir.",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["easy", "hard", "both"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--llmseg_topk_masks_dir",
        required=True,
        help="Directory containing topK/easy and topK/hard LLMSeg object masks.",
    )
    parser.add_argument(
        "--topk_mask_k",
        type=int,
        default=5,
        help="Use LLMSeg top-1..top-K object masks for each instruction.",
    )
    parser.add_argument("--output_dir", default="result_vlpart_topk_masked_rgb")
    parser.add_argument("--vis_dir", default="vis_vlpart_topk_masked_rgb")
    parser.add_argument("--pred_masks_dir", default="pred_masks_vlpart_topk_masked_rgb")
    parser.add_argument("--masked_rgb_dir", default="masked_rgb_vlpart_topk")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_pred_masks", action="store_true")
    parser.add_argument(
        "--save_masked_rgb",
        action="store_true",
        help="Save the selected masked RGB candidate for each instruction.",
    )
    parser.add_argument(
        "--vocabulary",
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
        "--custom_vocabulary",
        default="cylindrical side surface,hexagonal side face,flat side surface,whole object",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument(
        "--mask_selection",
        default="top1",
        choices=["top1", "union"],
        help="How to collapse VLPart instances into one evaluated mask.",
    )
    parser.add_argument("--ssr_threshold", type=float, default=0.5)
    parser.add_argument("--icr_threshold", type=float, default=0.7)
    parser.add_argument(
        "--object_hit_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for deciding whether top-K contains the correct object.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction_cache_size", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--opts",
        help="Additional config options in KEY VALUE pairs.",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def resolve_existing_or_repo_path(path: str) -> str:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    cwd_candidate = Path.cwd() / expanded
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str(REPO_ROOT / expanded)


def sanitize_filename_token(value) -> str:
    token = str(value) if value is not None else "unknown"
    token = token.replace("/", "_").replace("\\", "_").replace(" ", "_")
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def scene_name_from_sample(sample: Dict) -> str:
    scene = str(sample.get("scene", "")).strip()
    if scene:
        return scene

    for key in ("gt_mask_path", "gt_object_mask_path", "gt_object_path"):
        value = sample.get(key, "")
        paths = split_path_list(value)
        if not paths:
            continue
        match = re.match(r"^(\d+)_", os.path.basename(paths[0]))
        if match:
            return match.group(1)
    return ""


def load_samples_per_instruction(data_dir: str, json_file_name: str, difficulty: str) -> List[Dict]:
    json_file = Path(data_dir) / json_file_name
    if not json_file.exists():
        print(f"  [Warning] JSON not found: {json_file}")
        return []

    print(f"  [Info] Loading {difficulty} JSON: {json_file}")
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        raw_samples = data["samples"]
    elif isinstance(data, list):
        raw_samples = data
    else:
        raise ValueError(f"Unsupported JSON format: {json_file}")

    samples = []
    seen_counts = defaultdict(int)
    for raw_index, sample in enumerate(raw_samples):
        scene = scene_name_from_sample(sample)
        gt_mask_path_raw = sample.get("gt_mask_path", "")
        gt_object_mask_path_raw = sample.get("gt_object_mask_path", "")
        if not scene or not gt_mask_path_raw:
            continue

        image_path = resolve_data_path(data_dir, f"{scene}.png")
        gt_mask_paths = [
            resolve_data_path(data_dir, path) for path in split_path_list(gt_mask_path_raw)
        ]
        gt_object_mask_paths = [
            resolve_data_path(data_dir, path)
            for path in split_path_list(gt_object_mask_path_raw)
        ]
        if not os.path.exists(image_path) or not gt_mask_paths:
            continue

        img_name = os.path.basename(image_path)
        gt_object = sample.get("gt_object", sample.get("object", ""))
        seen_counts[(img_name, gt_object)] += 1
        obj_count = seen_counts[(img_name, gt_object)]
        instructions = sample.get("instructions", []) or [""]

        for instruction_index, instruction in enumerate(instructions):
            samples.append(
                {
                    "sample_index": raw_index,
                    "instruction_index": instruction_index,
                    "instruction": instruction,
                    "scene": scene,
                    "image_path": image_path,
                    "gt_mask_paths": gt_mask_paths,
                    "gt_mask_path": gt_mask_paths[0],
                    "gt_object_mask_paths": gt_object_mask_paths,
                    "gt_object_mask_path": gt_object_mask_paths[0] if gt_object_mask_paths else "",
                    "gt_object": gt_object,
                    "object": sample.get("object", ""),
                    "difficulty": difficulty,
                    "img_name": img_name,
                    "obj_count": obj_count,
                }
            )
    return samples


def resolve_topk_mask_paths(sample: Dict, args: argparse.Namespace) -> List[Tuple[int, str]]:
    topk_root = Path(args.llmseg_topk_masks_dir)
    difficulty = sample["difficulty"]
    img_name = sample["img_name"]
    obj_name = sanitize_filename_token(sample.get("gt_object", "unknown"))
    obj_count = int(sample.get("obj_count", 1))
    instruction_index = int(sample.get("instruction_index", 0))

    mask_paths = []
    for rank in range(1, args.topk_mask_k + 1):
        rank_dir = topk_root / f"top{rank}" / difficulty
        prefix = (
            f"{img_name}_{obj_name}_{obj_count}_instr{instruction_index}_"
            f"top{rank}_"
        )
        matches = sorted(rank_dir.glob(prefix + "*.png"))
        if matches:
            mask_paths.append((rank, str(matches[0])))
    return mask_paths


def read_llmseg_object_mask(mask_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read LLMSeg top-K mask: {mask_path}")

    target_h, target_w = image_shape
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask == 0


def make_masked_rgb(image_bgr: np.ndarray, object_fg: np.ndarray) -> np.ndarray:
    if object_fg.shape != image_bgr.shape[:2]:
        h, w = image_bgr.shape[:2]
        object_fg = cv2.resize(
            object_fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    masked = np.zeros_like(image_bgr)
    masked[object_fg] = image_bgr[object_fg]
    return masked


def predict_masked_rgb_from_object_fg(
    demo: VisualizationDemo,
    image_bgr: np.ndarray,
    object_fg: np.ndarray,
    mask_selection: str,
):
    masked_bgr = make_masked_rgb(image_bgr, object_fg)
    predictions = demo.predictor(masked_bgr)
    pred_fg, pred_info = choose_single_mask(
        predictions=predictions,
        metadata=demo.metadata,
        image_shape=image_bgr.shape[:2],
        mask_selection=mask_selection,
    )
    return pred_fg, pred_info, masked_bgr


def predict_masked_rgb(
    demo: VisualizationDemo,
    image_bgr: np.ndarray,
    object_mask_path: str,
    mask_selection: str,
):
    object_fg = read_llmseg_object_mask(object_mask_path, image_bgr.shape[:2])
    return predict_masked_rgb_from_object_fg(demo, image_bgr, object_fg, mask_selection)


def compute_best_object_iou(
    object_fg: np.ndarray,
    gt_object_masks: List[np.ndarray],
) -> float:
    if not gt_object_masks:
        return 0.0
    return max(compute_iou_from_foreground(object_fg, gt) for gt in gt_object_masks)


def evaluate_sample_candidates(
    sample: Dict,
    demo: VisualizationDemo,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
) -> Dict:
    image_path = sample["image_path"]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    topk_mask_paths = resolve_topk_mask_paths(sample, args)
    if not topk_mask_paths:
        raise FileNotFoundError(
            "No LLMSeg top-K masks found for "
            f"{sample['img_name']} {sample.get('gt_object', '')} "
            f"obj{sample.get('obj_count', 1)} instr{sample.get('instruction_index', 0)}"
        )

    image_bgr = read_image(image_path, format="BGR")
    gt_masks = load_gt_masks(sample["gt_mask_paths"], image_bgr.shape[:2])
    gt_object_mask_paths = sample.get("gt_object_mask_paths", [])
    gt_object_masks = (
        load_gt_masks(gt_object_mask_paths, image_bgr.shape[:2])
        if gt_object_mask_paths
        else []
    )
    object_hit_available = bool(gt_object_masks)
    best_topk_object_iou = 0.0
    best_topk_object_rank = -1
    best_topk_object_mask_path = ""
    candidates = []

    for rank, topk_mask_path in topk_mask_paths:
        object_fg = read_llmseg_object_mask(topk_mask_path, image_bgr.shape[:2])
        object_iou = compute_best_object_iou(object_fg, gt_object_masks)
        if object_iou > best_topk_object_iou:
            best_topk_object_iou = object_iou
            best_topk_object_rank = rank
            best_topk_object_mask_path = topk_mask_path
        candidates.append((rank, topk_mask_path, object_fg, object_iou))

    topk_object_hit = (
        object_hit_available
        and best_topk_object_iou >= args.object_hit_iou_threshold
    )
    best = None

    for rank, topk_mask_path, object_fg, object_iou in candidates:
        cache_key = f"{image_path}|{topk_mask_path}|{args.mask_selection}"
        cached = pred_cache.get(cache_key)
        if cached is None:
            pred_fg, pred_info, masked_bgr = predict_masked_rgb_from_object_fg(
                demo, image_bgr, object_fg, args.mask_selection
            )
            pred_cache.put(cache_key, (pred_fg, pred_info, masked_bgr))
        else:
            pred_fg, pred_info, masked_bgr = cached

        for gt_mask_path, gt_mask in zip(sample["gt_mask_paths"], gt_masks):
            iou = compute_iou_from_foreground(pred_fg, gt_mask)
            if best is None or iou > best["iou"]:
                best = {
                    "iou": iou,
                    "image_path": image_path,
                    "image_bgr": image_bgr,
                    "masked_bgr": masked_bgr,
                    "pred_fg": pred_fg,
                    "pred_info": pred_info,
                    "gt_mask": gt_mask,
                    "gt_mask_path": gt_mask_path,
                    "llmseg_topk_rank": rank,
                    "llmseg_topk_mask_path": topk_mask_path,
                    "selected_topk_object_iou": object_iou,
                    "best_topk_object_iou": best_topk_object_iou,
                    "best_topk_object_rank": best_topk_object_rank,
                    "best_topk_object_mask_path": best_topk_object_mask_path,
                    "topk_object_hit": topk_object_hit,
                    "topk_object_hit_available": object_hit_available,
                    "candidate_topk_count": len(topk_mask_paths),
                }

    if best is None:
        raise RuntimeError("No valid LLMSeg top-K/VLPart candidates")
    return best


def output_stem(sample: Dict, ordinal: int) -> str:
    base = Path(sample["image_path"]).stem
    obj = safe_name(sample.get("gt_object", "object"))
    instruction_index = int(sample.get("instruction_index", 0))
    obj_count = int(sample.get("obj_count", 1))
    return f"{ordinal:06d}_{safe_name(base)}_{obj}_{obj_count}_ins{instruction_index:02d}"


def raw_sample_key(sample: Dict) -> Tuple[str, int, str]:
    return (
        str(sample.get("difficulty", "")),
        int(sample.get("sample_index", -1)),
        str(sample.get("scene", "")),
    )


def compute_foreground_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    fg_a = mask_a.astype(bool)
    fg_b = mask_b.astype(bool)
    if fg_a.shape != fg_b.shape:
        h, w = fg_b.shape
        fg_a = cv2.resize(
            fg_a.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    intersection = np.logical_and(fg_a, fg_b).sum()
    union = np.logical_or(fg_a, fg_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_icr_score(iou_pairs: List[float], threshold: float) -> int:
    return sum(1 for iou in iou_pairs if iou >= threshold)


def compute_icr_metrics(icr_records: List[Dict], threshold: float) -> Dict:
    scores = [
        compute_icr_score(record.get("iou_pairs", []), threshold)
        for record in icr_records
        if len(record.get("iou_pairs", [])) >= 3
    ]
    if not scores:
        return {"icr": 0.0, "count": 0, "threshold": threshold}
    return {
        "icr": float(np.mean(scores)),
        "count": len(scores),
        "threshold": threshold,
    }


def filter_instruction_group(results: List[Dict], group: str) -> List[Dict]:
    if group == "first2":
        return [r for r in results if int(r.get("instruction_index", -1)) in (0, 1)]
    if group == "third":
        return [r for r in results if int(r.get("instruction_index", -1)) == 2]
    raise ValueError(f"Unknown instruction group: {group}")


def compute_split_summary(easy_results: List[Dict], hard_results: List[Dict], threshold: float) -> Dict:
    easy_metrics = compute_metrics(easy_results, threshold)
    hard_metrics = compute_metrics(hard_results, threshold)
    return {
        "easy": easy_metrics,
        "hard": hard_metrics,
        "avg_ic_iou": weighted_average(easy_metrics, hard_metrics, "ic_iou"),
        "avg_ssr": weighted_average(easy_metrics, hard_metrics, "ssr"),
        "total_count": easy_metrics["count"] + hard_metrics["count"],
        "total_success": easy_metrics["success_count"] + hard_metrics["success_count"],
    }


def compute_failure_source_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    success = sum(1 for result in results if result.get("success", False))
    fail = total - success
    object_hit_available = sum(
        1 for result in results if result.get("topk_object_hit_available", False)
    )
    topk_object_hit = sum(1 for result in results if result.get("topk_object_hit", False))
    hit_but_affordance_fail = sum(
        1
        for result in results
        if result.get("failure_source") == "topk_object_hit_but_affordance_fail"
    )
    object_miss = sum(
        1 for result in results if result.get("failure_source") == "topk_object_miss"
    )
    unknown = sum(
        1 for result in results if result.get("failure_source") == "unknown_object_gt"
    )

    return {
        "total": total,
        "success": success,
        "fail": fail,
        "object_hit_available": object_hit_available,
        "topk_object_hit": topk_object_hit,
        "hit_but_affordance_fail": hit_but_affordance_fail,
        "object_miss": object_miss,
        "unknown": unknown,
    }


def merge_failure_source_metrics(*metrics_items: Dict) -> Dict:
    keys = [
        "total",
        "success",
        "fail",
        "object_hit_available",
        "topk_object_hit",
        "hit_but_affordance_fail",
        "object_miss",
        "unknown",
    ]
    return {key: sum(item.get(key, 0) for item in metrics_items) for key in keys}


def count_with_rate(count: int, total: int) -> str:
    rate = (float(count) / float(total)) if total > 0 else 0.0
    return f"{count} ({rate:.4f})"


def write_failure_source_row(f, title: str, metrics: Dict) -> None:
    f.write(
        f"| {title} | {metrics['total']} | {metrics['success']} | {metrics['fail']} | "
        f"{count_with_rate(metrics['topk_object_hit'], metrics['object_hit_available'])} | "
        f"{count_with_rate(metrics['hit_but_affordance_fail'], metrics['fail'])} | "
        f"{count_with_rate(metrics['object_miss'], metrics['fail'])} | "
        f"{count_with_rate(metrics['unknown'], metrics['fail'])} |\n"
    )


def save_selected_masked_rgb(masked_bgr: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, masked_bgr)


def evaluate_split(
    split_name: str,
    samples: List[Dict],
    demo: VisualizationDemo,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
):
    results = []
    skipped = []
    icr_records = []
    current_key = None
    current_predictions = []

    def flush_icr_record():
        nonlocal current_key, current_predictions
        if current_key is None:
            return
        first_three = {}
        for instruction_index, pred_mask in current_predictions:
            if instruction_index in (0, 1, 2) and instruction_index not in first_three:
                first_three[instruction_index] = pred_mask
        if len(first_three) >= 3:
            iou_pairs = [
                compute_foreground_iou(first_three[0], first_three[1]),
                compute_foreground_iou(first_three[0], first_three[2]),
                compute_foreground_iou(first_three[1], first_three[2]),
            ]
            icr_records.append({"sample_key": current_key, "iou_pairs": iou_pairs})
        current_predictions = []

    for ordinal, sample in enumerate(tqdm(samples, desc=split_name.capitalize()), start=1):
        sample_key = raw_sample_key(sample)
        if current_key is None:
            current_key = sample_key
        elif sample_key != current_key:
            flush_icr_record()
            current_key = sample_key

        try:
            best = evaluate_sample_candidates(sample, demo, args, pred_cache)
            iou = best["iou"]
            success = iou >= args.ssr_threshold
            instruction_index = int(sample.get("instruction_index", 0))
            if success:
                failure_source = "success"
            elif best["topk_object_hit_available"]:
                if best["topk_object_hit"]:
                    failure_source = "topk_object_hit_but_affordance_fail"
                else:
                    failure_source = "topk_object_miss"
            else:
                failure_source = "unknown_object_gt"

            selected_sample = dict(sample)
            selected_sample["image_path"] = best["image_path"]
            stem = output_stem(selected_sample, ordinal)
            vis_path = ""
            pred_mask_path = ""
            masked_rgb_path = ""

            if args.save_vis:
                vis_path = str(Path(args.vis_dir) / split_name / f"{stem}_iou{iou:.3f}.png")
                save_prediction_overlay(
                    image_bgr=best["image_bgr"],
                    pred_fg=best["pred_fg"],
                    gt_mask=best["gt_mask"],
                    output_path=vis_path,
                    sample=selected_sample,
                    iou=iou,
                    pred_info=best["pred_info"],
                )
            if args.save_pred_masks:
                pred_mask_path = str(Path(args.pred_masks_dir) / split_name / f"{stem}.png")
                save_raw_pred_mask(best["pred_fg"], pred_mask_path)
            if args.save_masked_rgb:
                masked_rgb_path = str(
                    Path(args.masked_rgb_dir)
                    / split_name
                    / f"{stem}_top{best['llmseg_topk_rank']}.png"
                )
                save_selected_masked_rgb(best["masked_bgr"], masked_rgb_path)

            results.append(
                {
                    "image_path": best["image_path"],
                    "gt_mask_path": best["gt_mask_path"],
                    "gt_object_mask_path": sample.get("gt_object_mask_path", ""),
                    "llmseg_topk_mask_path": best["llmseg_topk_mask_path"],
                    "llmseg_topk_rank": best["llmseg_topk_rank"],
                    "selected_topk_object_iou": best["selected_topk_object_iou"],
                    "best_topk_object_iou": best["best_topk_object_iou"],
                    "best_topk_object_rank": best["best_topk_object_rank"],
                    "best_topk_object_mask_path": best["best_topk_object_mask_path"],
                    "topk_object_hit": best["topk_object_hit"],
                    "topk_object_hit_available": best["topk_object_hit_available"],
                    "gt_object": sample["gt_object"],
                    "sample_index": sample.get("sample_index", -1),
                    "scene": sample.get("scene", ""),
                    "instruction": sample.get("instruction", ""),
                    "instruction_index": instruction_index,
                    "obj_count": sample.get("obj_count", 1),
                    "difficulty": sample["difficulty"],
                    "iou": iou,
                    "ic_ious": [iou],
                    "avg_ic_iou": iou,
                    "success": success,
                    "failure_source": failure_source,
                    "pred_class": best["pred_info"].get("pred_class", -1),
                    "pred_class_name": best["pred_info"].get("pred_class_name", ""),
                    "pred_score": best["pred_info"].get("pred_score", 0.0),
                    "num_instances": best["pred_info"].get("num_instances", 0),
                    "vis_path": vis_path,
                    "pred_mask_path": pred_mask_path,
                    "masked_rgb_path": masked_rgb_path,
                    "candidate_topk_count": best["candidate_topk_count"],
                    "candidate_gt_count": len(sample["gt_mask_paths"]),
                }
            )
            if instruction_index in (0, 1, 2):
                current_predictions.append((instruction_index, best["pred_fg"].copy()))
        except Exception as exc:
            skipped.append((sample.get("image_path", "unknown"), repr(exc)))
            if len(skipped) <= 5 or args.debug:
                print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    flush_icr_record()
    return results, skipped, icr_records


def write_instruction_group_row(f, title: str, summary: Dict) -> None:
    easy_metrics = summary["easy"]
    hard_metrics = summary["hard"]
    f.write(
        f"| {title} | {easy_metrics['ic_iou']:.4f} | "
        f"{hard_metrics['ic_iou']:.4f} | {summary['avg_ic_iou']:.4f} | "
        f"{easy_metrics['ssr']:.4f} | {hard_metrics['ssr']:.4f} | "
        f"{summary['avg_ssr']:.4f} |\n"
    )


def write_sample_result(f, split_title: str, index: int, result: Dict) -> None:
    f.write(f"[{split_title} Instruction Sample {index}]\n")
    f.write(f"  Image: {result['image_path']}\n")
    f.write(f"  GT Affordance Mask: {result['gt_mask_path']}\n")
    if result.get("gt_object_mask_path"):
        f.write(f"  GT Object Mask: {result['gt_object_mask_path']}\n")
    f.write(f"  LLMSeg TopK Mask: top{result['llmseg_topk_rank']} {result['llmseg_topk_mask_path']}\n")
    f.write(
        f"  Best TopK Object IoU: {result.get('best_topk_object_iou', 0.0):.4f} "
        f"(top{result.get('best_topk_object_rank', -1)})\n"
    )
    f.write(
        f"  Selected TopK Object IoU: {result.get('selected_topk_object_iou', 0.0):.4f}\n"
    )
    f.write(f"  TopK Object Hit: {int(bool(result.get('topk_object_hit', False)))}\n")
    f.write(f"  Object: {result['gt_object']} (obj_count={result.get('obj_count', 1)})\n")
    f.write(f"  Instruction {result['instruction_index']}: {result.get('instruction', '')}\n")
    f.write(f"  IC-IoU: {result['avg_ic_iou']:.4f}\n")
    f.write(f"  SSR Success: {int(result['success'])}\n")
    f.write(f"  Failure Source: {result.get('failure_source', '')}\n")
    f.write(f"  Pred Class: {result['pred_class_name']} ({result['pred_class']})\n")
    f.write(f"  Pred Score: {result['pred_score']:.4f}\n")
    f.write(f"  Detected Instances: {result['num_instances']}\n")
    f.write(f"  TopK candidates found: {result['candidate_topk_count']}\n")
    if result.get("pred_mask_path"):
        f.write(f"  Pred Mask: {result['pred_mask_path']}\n")
    if result.get("masked_rgb_path"):
        f.write(f"  Masked RGB: {result['masked_rgb_path']}\n")
    if result.get("vis_path"):
        f.write(f"  Visualization: {result['vis_path']}\n")
    f.write("\n")


def write_results(
    args: argparse.Namespace,
    easy_results: List[Dict],
    hard_results: List[Dict],
    skipped_errors: Dict[str, List[Tuple[str, str]]],
    easy_icr_records: List[Dict],
    hard_icr_records: List[Dict],
) -> str:
    all_summary = compute_split_summary(easy_results, hard_results, args.ssr_threshold)
    first2_summary = compute_split_summary(
        filter_instruction_group(easy_results, "first2"),
        filter_instruction_group(hard_results, "first2"),
        args.ssr_threshold,
    )
    third_summary = compute_split_summary(
        filter_instruction_group(easy_results, "third"),
        filter_instruction_group(hard_results, "third"),
        args.ssr_threshold,
    )
    easy_metrics = all_summary["easy"]
    hard_metrics = all_summary["hard"]
    easy_icr_metrics = compute_icr_metrics(easy_icr_records, args.icr_threshold)
    hard_icr_metrics = compute_icr_metrics(hard_icr_records, args.icr_threshold)
    avg_icr = weighted_average(easy_icr_metrics, hard_icr_metrics, "icr")
    total_icr_count = easy_icr_metrics["count"] + hard_icr_metrics["count"]
    easy_failure_metrics = compute_failure_source_metrics(easy_results)
    hard_failure_metrics = compute_failure_source_metrics(hard_results)
    all_failure_metrics = merge_failure_source_metrics(
        easy_failure_metrics, hard_failure_metrics
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = str(
        Path(args.output_dir) / f"vlpart_llmseg_topk_masked_rgb_results_{timestamp}.txt"
    )

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart VIGOR-100K LLMSeg Top-K Masked RGB Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model config: {args.config_file}\n")
        f.write(f"Model weights: {args.weights}\n")
        f.write(f"Data dir: {args.data_dir}\n")
        f.write(f"LLMSeg top-K masks: {args.llmseg_topk_masks_dir}\n")
        f.write(f"Top-K K: {args.topk_mask_k}\n")
        f.write("Input image: scene RGB masked by each LLMSeg object mask\n")
        f.write("Selection: oracle Max-IoU against gt_mask_path among top-K VLPart outputs\n")
        f.write(f"Vocabulary: {args.custom_vocabulary if args.vocabulary == 'custom' else args.vocabulary}\n")
        f.write(f"Mask selection: {args.mask_selection}\n")
        f.write(f"Confidence threshold: {args.confidence_threshold}\n")
        f.write(f"SSR threshold: {args.ssr_threshold}\n")
        f.write(f"ICR threshold: {args.icr_threshold}\n")
        f.write(f"Top-K object hit IoU threshold: {args.object_hit_iou_threshold}\n\n")

        f.write(
            "| Method | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            f"SSR@{args.ssr_threshold:.1f} Easy | "
            f"SSR@{args.ssr_threshold:.1f} Hard | "
            f"SSR@{args.ssr_threshold:.1f} Avg | "
            f"ICR@{args.icr_threshold:.1f} Easy | "
            f"ICR@{args.icr_threshold:.1f} Hard | ICR@{args.icr_threshold:.1f} Avg |\n"
        )
        f.write(
            f"| VLPart+LLMSegTopKMaskedRGB | {easy_metrics['ic_iou']:.4f} | "
            f"{hard_metrics['ic_iou']:.4f} | {all_summary['avg_ic_iou']:.4f} | "
            f"{easy_metrics['ssr']:.4f} | {hard_metrics['ssr']:.4f} | "
            f"{all_summary['avg_ssr']:.4f} | {easy_icr_metrics['icr']:.4f} | "
            f"{hard_icr_metrics['icr']:.4f} | {avg_icr:.4f} |\n\n"
        )

        f.write(f"Total instruction samples: {all_summary['total_count']}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n")
        f.write(f"Success count: {all_summary['total_success']} / {all_summary['total_count']}\n")
        f.write(f"Skipped samples: Easy={len(skipped_errors['easy'])}, Hard={len(skipped_errors['hard'])}\n\n")

        f.write("=" * 80 + "\n")
        f.write("  Top-K Oracle Cascade failure sources\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Definitions: Top-K object hit means any LLMSeg top-K object mask reaches "
            f"IoU >= {args.object_hit_iou_threshold} against gt_object_mask_path. "
            "Affordance fail means final affordance IoU is below SSR threshold.\n\n"
        )
        f.write(
            "| Split | Total | Success | Fail | Top-K Object Hit | "
            "Top-K Object Hit but Affordance Fail | Top-K Object Miss | Unknown |\n"
        )
        write_failure_source_row(f, "Easy", easy_failure_metrics)
        write_failure_source_row(f, "Hard", hard_failure_metrics)
        write_failure_source_row(f, "All", all_failure_metrics)
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("  Instruction group IC-IoU and SSR\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "| Instruction Group | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            f"SSR@{args.ssr_threshold:.1f} Easy | SSR@{args.ssr_threshold:.1f} Hard | "
            f"SSR@{args.ssr_threshold:.1f} Avg |\n"
        )
        write_instruction_group_row(f, "All instructions", all_summary)
        write_instruction_group_row(f, "First 2 instructions", first2_summary)
        write_instruction_group_row(f, "Third instruction", third_summary)
        f.write("\n")

        f.write(f"ICR@{args.icr_threshold:.1f} (3 pairwise predictions per raw sample):\n")
        f.write(
            f"  Easy:    {easy_icr_metrics['icr']:.4f} "
            f"({easy_icr_metrics['count']} samples)\n"
        )
        f.write(
            f"  Hard:    {hard_icr_metrics['icr']:.4f} "
            f"({hard_icr_metrics['count']} samples)\n"
        )
        f.write(f"  Average: {avg_icr:.4f} ({total_icr_count} samples)\n\n")

        for split_name in ["easy", "hard"]:
            if skipped_errors[split_name]:
                f.write(f"[{split_name} skipped examples]\n")
                for path, err in skipped_errors[split_name][:5]:
                    f.write(f"  {path}: {err}\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("  Easy instruction details\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(easy_results, start=1):
            write_sample_result(f, "Easy", i, result)

        f.write("=" * 80 + "\n")
        f.write("  Hard instruction details\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(hard_results, start=1):
            write_sample_result(f, "Hard", i, result)

    return result_file


def main() -> None:
    args = parse_args()
    if args.topk_mask_k <= 0:
        raise ValueError(f"--topk_mask_k must be > 0, got {args.topk_mask_k}")

    args.config_file = resolve_repo_path(args.config_file)
    args.weights = resolve_repo_path(args.weights)
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.llmseg_topk_masks_dir = resolve_existing_or_repo_path(args.llmseg_topk_masks_dir)

    if args.vocabulary == "custom":
        args.custom_vocabulary = normalize_custom_vocabulary(args.custom_vocabulary)
        if not args.custom_vocabulary:
            raise ValueError("--custom_vocabulary must not be empty when --vocabulary custom")

    if not os.path.isdir(args.llmseg_topk_masks_dir):
        raise FileNotFoundError(f"LLMSeg top-K mask dir not found: {args.llmseg_topk_masks_dir}")

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
    if args.save_pred_masks:
        Path(args.pred_masks_dir).mkdir(parents=True, exist_ok=True)
    if args.save_masked_rgb:
        Path(args.masked_rgb_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Loading VLPart")
    print("=" * 60)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

    print("=" * 60)
    print("  Loading VIGOR instruction samples")
    print("=" * 60)
    easy_samples = []
    hard_samples = []
    if args.split in ["easy", "both"]:
        easy_samples = load_samples_per_instruction(args.data_dir, args.easy_json_file, "easy")
    if args.split in ["hard", "both"]:
        hard_samples = load_samples_per_instruction(args.data_dir, args.hard_json_file, "hard")
    if args.max_samples is not None:
        easy_samples = easy_samples[: args.max_samples]
        hard_samples = hard_samples[: args.max_samples]

    print(f"  Easy instruction samples: {len(easy_samples)}")
    print(f"  Hard instruction samples: {len(hard_samples)}")
    print(f"  LLMSeg top-K masks: {args.llmseg_topk_masks_dir}")
    print(f"  Top-K K: {args.topk_mask_k}")
    print(f"  Vocabulary: {args.vocabulary}")
    if args.vocabulary == "custom":
        print(f"  Custom vocabulary: {args.custom_vocabulary}")
    print("  Input: scene RGB masked by each LLMSeg object mask")
    print("  Selection: best IoU against gt_mask_path among top-K candidates")
    print(f"  Mask selection: {args.mask_selection}")
    print(f"  SSR threshold: {args.ssr_threshold}")
    print(f"  ICR threshold: {args.icr_threshold}")
    print(f"  Top-K object hit IoU threshold: {args.object_hit_iou_threshold}")

    pred_cache = PredictionCache(args.prediction_cache_size)
    results = {"easy": [], "hard": []}
    skipped_errors = {"easy": [], "hard": []}
    icr_records = {"easy": [], "hard": []}

    if easy_samples:
        print("\n" + "=" * 60)
        print("  Testing Easy instruction samples")
        print("=" * 60)
        results["easy"], skipped_errors["easy"], icr_records["easy"] = evaluate_split(
            "easy", easy_samples, demo, args, pred_cache
        )

    if hard_samples:
        print("\n" + "=" * 60)
        print("  Testing Hard instruction samples")
        print("=" * 60)
        results["hard"], skipped_errors["hard"], icr_records["hard"] = evaluate_split(
            "hard", hard_samples, demo, args, pred_cache
        )

    result_file = write_results(
        args=args,
        easy_results=results["easy"],
        hard_results=results["hard"],
        skipped_errors=skipped_errors,
        easy_icr_records=icr_records["easy"],
        hard_icr_records=icr_records["hard"],
    )

    all_summary = compute_split_summary(results["easy"], results["hard"], args.ssr_threshold)
    first2_summary = compute_split_summary(
        filter_instruction_group(results["easy"], "first2"),
        filter_instruction_group(results["hard"], "first2"),
        args.ssr_threshold,
    )
    third_summary = compute_split_summary(
        filter_instruction_group(results["easy"], "third"),
        filter_instruction_group(results["hard"], "third"),
        args.ssr_threshold,
    )
    easy_metrics = all_summary["easy"]
    hard_metrics = all_summary["hard"]
    easy_icr_metrics = compute_icr_metrics(icr_records["easy"], args.icr_threshold)
    hard_icr_metrics = compute_icr_metrics(icr_records["hard"], args.icr_threshold)
    avg_icr = weighted_average(easy_icr_metrics, hard_icr_metrics, "icr")
    easy_failure_metrics = compute_failure_source_metrics(results["easy"])
    hard_failure_metrics = compute_failure_source_metrics(results["hard"])
    all_failure_metrics = merge_failure_source_metrics(
        easy_failure_metrics, hard_failure_metrics
    )

    print("\n" + "=" * 60)
    print("  VLPart LLMSeg Top-K Masked RGB evaluation finished")
    print("=" * 60)
    print(f"Result file: {result_file}")
    print(
        f"IC-IoU: Easy={easy_metrics['ic_iou']:.4f}, "
        f"Hard={hard_metrics['ic_iou']:.4f}, Avg={all_summary['avg_ic_iou']:.4f}"
    )
    print(
        f"SSR@{args.ssr_threshold:.1f}: Easy={easy_metrics['ssr']:.4f}, "
        f"Hard={hard_metrics['ssr']:.4f}, Avg={all_summary['avg_ssr']:.4f}"
    )
    print(
        f"First2 IC-IoU/SSR: Easy={first2_summary['easy']['ic_iou']:.4f}/"
        f"{first2_summary['easy']['ssr']:.4f}, Hard={first2_summary['hard']['ic_iou']:.4f}/"
        f"{first2_summary['hard']['ssr']:.4f}, Avg={first2_summary['avg_ic_iou']:.4f}/"
        f"{first2_summary['avg_ssr']:.4f}"
    )
    print(
        f"Third IC-IoU/SSR: Easy={third_summary['easy']['ic_iou']:.4f}/"
        f"{third_summary['easy']['ssr']:.4f}, Hard={third_summary['hard']['ic_iou']:.4f}/"
        f"{third_summary['hard']['ssr']:.4f}, Avg={third_summary['avg_ic_iou']:.4f}/"
        f"{third_summary['avg_ssr']:.4f}"
    )
    print(
        f"ICR@{args.icr_threshold:.1f}: Easy={easy_icr_metrics['icr']:.4f}, "
        f"Hard={hard_icr_metrics['icr']:.4f}, Avg={avg_icr:.4f}"
    )
    print(
        "Failure sources: "
        f"Top-K object hit but affordance fail="
        f"{all_failure_metrics['hit_but_affordance_fail']}, "
        f"Top-K object miss={all_failure_metrics['object_miss']}, "
        f"Unknown={all_failure_metrics['unknown']}"
    )


if __name__ == "__main__":
    main()
