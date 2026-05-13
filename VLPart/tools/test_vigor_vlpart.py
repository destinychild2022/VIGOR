import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
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
sys.path.insert(0, str(VLPART_ROOT))
sys.path.insert(0, str(VLPART_ROOT / "demo"))

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from vlpart.config import add_vlpart_config
from predictor import VisualizationDemo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Batch VIGOR evaluation with VLPart")

    parser.add_argument(
        "--config-file",
        default="configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml",
        help="VLPart config file.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="VLPart checkpoint path.",
    )
    parser.add_argument(
        "--data_dir",
        default="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test",
        help="Dataset root that contains the easy/hard JSON files.",
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
        "--output_dir",
        default="result_vlpart",
        help="Directory for txt results.",
    )
    parser.add_argument(
        "--vis_dir",
        default="vis_output_vlpart",
        help="Directory for prediction-mask overlays.",
    )
    parser.add_argument(
        "--pred_masks_dir",
        default="pred_masks_vlpart",
        help="Directory for one raw predicted mask per sample.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save the selected VLPart mask overlaid on gt_object_path image.",
    )
    parser.add_argument(
        "--save_pred_masks",
        action="store_true",
        help="Save one raw VIGOR-style predicted mask per sample, 0 foreground and 255 background.",
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
        help="VLPart vocabulary name.",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="cylindrical side surface,hexagonal side face,flat side surface,whole object",
        help="Comma-separated custom vocabulary for VLPart.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Minimum instance confidence threshold.",
    )
    parser.add_argument(
        "--mask_selection",
        default="top1",
        choices=["top1", "union"],
        help="How to collapse VLPart instances into one evaluated mask.",
    )
    parser.add_argument(
        "--ssr_threshold",
        type=float,
        default=0.5,
        help="SSR success threshold. SSR counts samples with IoU >= threshold.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Debug limit per split.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for VLPart.",
    )
    parser.add_argument(
        "--prediction_cache_size",
        type=int,
        default=256,
        help="LRU cache size for repeated gt_object_path predictions. Set 0 to disable.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra error details.",
    )
    parser.add_argument(
        "--opts",
        help="Additional config options in KEY VALUE pairs.",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def resolve_repo_path(path: str) -> str:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(VLPART_ROOT / expanded)


def resolve_data_path(data_dir: str, path: str) -> str:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(Path(data_dir) / expanded)


def normalize_custom_vocabulary(text: str) -> str:
    return ",".join(item.strip() for item in text.split(",") if item.strip())


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_vlpart_config(cfg)
    cfg.merge_from_file(resolve_repo_path(args.config_file))
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.MODEL.WEIGHTS = resolve_repo_path(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def split_path_list(path_text: str) -> List[str]:
    return [item.strip() for item in path_text.split(",") if item.strip()]


def load_samples(data_dir: str, json_file_name: str, difficulty: str) -> List[Dict]:
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
    for idx, sample in enumerate(raw_samples):
        gt_object_path_raw = sample.get("gt_object_path", "")
        gt_mask_path_raw = sample.get("gt_mask_path", "")
        if not gt_object_path_raw or not gt_mask_path_raw:
            continue

        gt_object_paths = split_path_list(gt_object_path_raw)
        image_paths = [
            resolve_data_path(data_dir, path)
            for path in gt_object_paths
        ]
        gt_mask_paths = [
            resolve_data_path(data_dir, path)
            for path in split_path_list(gt_mask_path_raw)
        ]
        if not image_paths or not gt_mask_paths:
            continue

        image_path = image_paths[0]

        samples.append(
            {
                "sample_index": idx,
                "image_path": image_path,
                "image_paths": image_paths,
                "gt_mask_paths": gt_mask_paths,
                "gt_mask_path": gt_mask_paths[0] if gt_mask_paths else "",
                "gt_object_path": gt_object_path_raw,
                "gt_object_paths": gt_object_paths,
                "gt_object": sample.get("gt_object", sample.get("object", "")),
                "object": sample.get("object", ""),
                "instructions": sample.get("instructions", []),
                "difficulty": difficulty,
                "img_name": os.path.basename(image_path),
            }
        )
    return samples


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


def compute_iou_from_foreground(pred_fg: np.ndarray, gt_mask: np.ndarray) -> float:
    gt_fg = gt_mask == 0
    pred_fg = pred_fg.astype(bool)
    if pred_fg.shape != gt_fg.shape:
        h, w = gt_fg.shape
        pred_fg = cv2.resize(
            pred_fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_iou_max(pred_fg: np.ndarray, gt_masks: List[np.ndarray]) -> Tuple[float, Optional[np.ndarray]]:
    best_iou = 0.0
    best_gt = gt_masks[0] if gt_masks else None
    for gt in gt_masks:
        iou = compute_iou_from_foreground(pred_fg, gt)
        if iou >= best_iou:
            best_iou = iou
            best_gt = gt
    return best_iou, best_gt


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = text.strip("._")
    return (text or "sample")[:max_len]


def selected_mask_to_vigor_mask(pred_fg: np.ndarray) -> np.ndarray:
    return np.where(pred_fg, 0, 255).astype(np.uint8)


def choose_single_mask(
    predictions: Dict,
    metadata,
    image_shape: Tuple[int, int],
    mask_selection: str,
) -> Tuple[np.ndarray, Dict]:
    h, w = image_shape
    empty = np.zeros((h, w), dtype=bool)
    info = {
        "num_instances": 0,
        "pred_score": 0.0,
        "pred_class": -1,
        "pred_class_name": "",
    }

    instances = predictions.get("instances")
    if instances is None or len(instances) == 0 or not instances.has("pred_masks"):
        return empty, info

    instances = instances.to("cpu")
    masks = instances.pred_masks.numpy().astype(bool)
    scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances))
    classes = (
        instances.pred_classes.numpy()
        if instances.has("pred_classes")
        else np.full(len(instances), -1, dtype=np.int64)
    )

    info["num_instances"] = int(len(instances))
    if len(masks) == 0:
        return empty, info

    if mask_selection == "union":
        pred_fg = np.any(masks, axis=0)
        best_idx = int(np.argmax(scores))
    else:
        best_idx = int(np.argmax(scores))
        pred_fg = masks[best_idx]

    if pred_fg.shape != (h, w):
        pred_fg = cv2.resize(
            pred_fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    class_idx = int(classes[best_idx])
    class_names = metadata.get("thing_classes", [])
    if 0 <= class_idx < len(class_names):
        class_name = class_names[class_idx]
    else:
        class_name = str(class_idx) if class_idx >= 0 else ""

    info.update(
        {
            "pred_score": float(scores[best_idx]),
            "pred_class": class_idx,
            "pred_class_name": class_name,
        }
    )
    return pred_fg, info


def save_prediction_overlay(
    image_bgr: np.ndarray,
    pred_fg: np.ndarray,
    gt_mask: Optional[np.ndarray],
    output_path: str,
    sample: Dict,
    iou: float,
    pred_info: Dict,
    alpha: float = 0.45,
) -> None:
    # VLPart demo/visualizer.py also uses alpha-blended mask overlays; here
    # we draw only the selected one-mask affordance prediction.
    image = image_bgr.copy()
    if pred_fg.shape != image.shape[:2]:
        h, w = image.shape[:2]
        pred_fg = cv2.resize(
            pred_fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    overlay = image.copy()
    overlay[pred_fg] = (0, 0, 255)
    image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    if gt_mask is not None:
        gt_fg = (gt_mask == 0).astype(np.uint8)
        contours, _ = cv2.findContours(gt_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    title = (
        f"{sample['difficulty']} | {sample['gt_object']} | "
        f"IoU {iou:.4f} | {pred_info.get('pred_class_name', '')} "
        f"{pred_info.get('pred_score', 0.0):.3f}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (255, 255, 255), thickness=-1)
    cv2.putText(
        image,
        title[:180],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def save_raw_pred_mask(pred_fg: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, selected_mask_to_vigor_mask(pred_fg))


class PredictionCache:
    def __init__(self, max_size: int):
        self.max_size = max(0, int(max_size))
        self.cache = OrderedDict()

    def get(self, key: str):
        if self.max_size <= 0 or key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: str, value):
        if self.max_size <= 0:
            return
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


def predict_one_image(
    demo: VisualizationDemo,
    image_path: str,
    mask_selection: str,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = read_image(image_path, format="BGR")
    predictions = demo.predictor(image_bgr)
    pred_fg, pred_info = choose_single_mask(
        predictions=predictions,
        metadata=demo.metadata,
        image_shape=image_bgr.shape[:2],
        mask_selection=mask_selection,
    )
    return pred_fg, pred_info, image_bgr


def evaluate_sample_candidates(
    sample: Dict,
    demo: VisualizationDemo,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
) -> Dict:
    """Evaluate every gt_object_path against every GT mask and keep Max-IoU."""
    best = None
    image_paths = sample.get("image_paths") or [sample["image_path"]]

    for image_path in image_paths:
        cached = pred_cache.get(image_path)
        if cached is None:
            pred_fg, pred_info, image_bgr = predict_one_image(
                demo, image_path, args.mask_selection
            )
            pred_cache.put(image_path, (pred_fg, pred_info, image_bgr))
        else:
            pred_fg, pred_info, image_bgr = cached

        gt_masks = load_gt_masks(sample["gt_mask_paths"], image_bgr.shape[:2])
        for gt_mask_path, gt_mask in zip(sample["gt_mask_paths"], gt_masks):
            iou = compute_iou_from_foreground(pred_fg, gt_mask)
            if best is None or iou > best["iou"]:
                best = {
                    "iou": iou,
                    "image_path": image_path,
                    "image_bgr": image_bgr,
                    "pred_fg": pred_fg,
                    "pred_info": pred_info,
                    "gt_mask": gt_mask,
                    "gt_mask_path": gt_mask_path,
                }

    if best is None:
        raise RuntimeError("No valid gt_object_path/gt_mask_path candidates")
    return best


def output_stem(sample: Dict, ordinal: int) -> str:
    base = Path(sample["image_path"]).stem
    obj = safe_name(sample.get("gt_object", "object"))
    return f"{ordinal:06d}_{safe_name(base)}_{obj}"


def evaluate_split(
    split_name: str,
    samples: List[Dict],
    demo: VisualizationDemo,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    results = []
    skipped = []

    for ordinal, sample in enumerate(tqdm(samples, desc=split_name.capitalize()), start=1):
        try:
            best = evaluate_sample_candidates(sample, demo, args, pred_cache)
            iou = best["iou"]
            success = iou >= args.ssr_threshold
            image_path = best["image_path"]
            image_bgr = best["image_bgr"]
            pred_fg = best["pred_fg"]
            pred_info = best["pred_info"]
            best_gt = best["gt_mask"]
            gt_mask_path = best["gt_mask_path"]

            selected_sample = dict(sample)
            selected_sample["image_path"] = image_path
            stem = output_stem(selected_sample, ordinal)
            vis_path = ""
            pred_mask_path = ""
            if args.save_vis:
                vis_path = str(Path(args.vis_dir) / split_name / f"{stem}_iou{iou:.3f}.png")
                save_prediction_overlay(
                    image_bgr=image_bgr,
                    pred_fg=pred_fg,
                    gt_mask=best_gt,
                    output_path=vis_path,
                    sample=selected_sample,
                    iou=iou,
                    pred_info=pred_info,
                )
            if args.save_pred_masks:
                pred_mask_path = str(Path(args.pred_masks_dir) / split_name / f"{stem}.png")
                save_raw_pred_mask(pred_fg, pred_mask_path)

            results.append(
                {
                    "image_path": image_path,
                    "gt_mask_path": gt_mask_path,
                    "gt_object": sample["gt_object"],
                    "difficulty": sample["difficulty"],
                    "iou": iou,
                    "ic_ious": [iou],
                    "avg_ic_iou": iou,
                    "success": success,
                    "pred_class": pred_info.get("pred_class", -1),
                    "pred_class_name": pred_info.get("pred_class_name", ""),
                    "pred_score": pred_info.get("pred_score", 0.0),
                    "num_instances": pred_info.get("num_instances", 0),
                    "vis_path": vis_path,
                    "pred_mask_path": pred_mask_path,
                    "candidate_image_count": len(sample.get("image_paths") or [sample["image_path"]]),
                    "candidate_gt_count": len(sample["gt_mask_paths"]),
                }
            )
        except Exception as exc:
            skipped.append((sample.get("image_path", "unknown"), repr(exc)))
            if len(skipped) <= 5 or args.debug:
                print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    return results, skipped


def compute_metrics(results: List[Dict], ssr_threshold: float) -> Dict:
    if not results:
        return {
            "ic_iou": 0.0,
            "count": 0,
            "success_count": 0,
            "ssr": 0.0,
            "ssr_threshold": ssr_threshold,
        }

    ious = [item["avg_ic_iou"] for item in results]
    success_count = sum(1 for item in results if item["avg_ic_iou"] >= ssr_threshold)
    return {
        "ic_iou": float(np.mean(ious)),
        "count": len(results),
        "success_count": success_count,
        "ssr": success_count / len(results),
        "ssr_threshold": ssr_threshold,
    }


def weighted_average(easy_metrics: Dict, hard_metrics: Dict, key: str) -> float:
    total = easy_metrics["count"] + hard_metrics["count"]
    if total == 0:
        return 0.0
    return (
        easy_metrics[key] * easy_metrics["count"]
        + hard_metrics[key] * hard_metrics["count"]
    ) / total


def write_results(
    args: argparse.Namespace,
    easy_results: List[Dict],
    hard_results: List[Dict],
    skipped_errors: Dict[str, List[Tuple[str, str]]],
) -> str:
    easy_metrics = compute_metrics(easy_results, args.ssr_threshold)
    hard_metrics = compute_metrics(hard_results, args.ssr_threshold)
    total_count = easy_metrics["count"] + hard_metrics["count"]
    avg_ic_iou = weighted_average(easy_metrics, hard_metrics, "ic_iou")
    avg_ssr = weighted_average(easy_metrics, hard_metrics, "ssr")
    total_success = easy_metrics["success_count"] + hard_metrics["success_count"]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = str(Path(args.output_dir) / f"vlpart_vigor_test_results_{timestamp}.txt")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart VIGOR-100K 测试结果\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"模型路径: {args.config_file}\n")
        f.write(f"微调权重: {args.weights}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write("SAM masks: N/A (VLPart direct instance masks)\n")
        f.write(f"Vocabulary: {args.vocabulary}\n")
        f.write(f"Custom vocabulary: {args.custom_vocabulary}\n")
        f.write(f"Mask selection: {args.mask_selection}\n")
        f.write(f"Confidence threshold: {args.confidence_threshold}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("  汇总结果表格\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "| Category | Methods | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            "SSR@0.5 Easy | SSR@0.5 Hard | SSR@0.5 Avg |\n"
        )
        f.write(
            f"| Offline  | VLPart  | {easy_metrics['ic_iou']:.4f}      | "
            f"{hard_metrics['ic_iou']:.4f}      | {avg_ic_iou:.4f}     | "
            f"{easy_metrics['ssr']:.4f}       | {hard_metrics['ssr']:.4f}       | "
            f"{avg_ssr:.4f}      |\n\n"
        )

        f.write("\n" + "=" * 80 + "\n")
        f.write("  总体 IC-IoU 和 SSR@0.5\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总样本数: {total_count}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n")
        f.write(f"跳过样本数: Easy={len(skipped_errors['easy'])}, Hard={len(skipped_errors['hard'])}\n")
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

        f.write(f"SSR@{args.ssr_threshold:.1f}:\n")
        f.write(
            f"  Easy:    {easy_metrics['ssr']:.4f} "
            f"({easy_metrics['success_count']} / {easy_metrics['count']})\n"
        )
        f.write(
            f"  Hard:    {hard_metrics['ssr']:.4f} "
            f"({hard_metrics['success_count']} / {hard_metrics['count']})\n"
        )
        f.write(f"  Average: {avg_ssr:.4f} ({total_success} / {total_count})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  Easy 样本详细结果\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(easy_results, start=1):
            write_sample_result(f, "Easy", i, result)

        f.write("\n" + "=" * 80 + "\n")
        f.write("  Hard 样本详细结果\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(hard_results, start=1):
            write_sample_result(f, "Hard", i, result)

    return result_file


def write_sample_result(f, split_title: str, index: int, result: Dict) -> None:
    f.write(f"[{split_title} Sample {index}]\n")
    f.write(f"  Image: {result['image_path']}\n")
    f.write(f"  GT Mask: {result['gt_mask_path']}\n")
    f.write(f"  Object: {result['gt_object']}\n")
    f.write(f"  IC-IoU (per instruction): {[f'{x:.4f}' for x in result['ic_ious']]}\n")
    f.write(f"  Avg IC-IoU: {result['avg_ic_iou']:.4f}\n")
    f.write(f"  SSR Success: {int(result['success'])}\n")
    f.write(f"  Pred Class: {result['pred_class_name']} ({result['pred_class']})\n")
    f.write(f"  Pred Score: {result['pred_score']:.4f}\n")
    f.write(f"  Detected Instances: {result['num_instances']}\n")
    if result.get("pred_mask_path"):
        f.write(f"  Pred Mask: {result['pred_mask_path']}\n")
    if result.get("vis_path"):
        f.write(f"  Visualization: {result['vis_path']}\n")
    f.write("\n")


def main() -> None:
    args = parse_args()
    args.config_file = resolve_repo_path(args.config_file)
    args.weights = resolve_repo_path(args.weights)
    args.data_dir = str(Path(args.data_dir).expanduser())
    if args.vocabulary == "custom":
        args.custom_vocabulary = normalize_custom_vocabulary(args.custom_vocabulary)
        if not args.custom_vocabulary:
            raise ValueError("--custom_vocabulary must not be empty when --vocabulary custom")

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

    print("=" * 60)
    print("  Loading VLPart")
    print("=" * 60)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

    print("=" * 60)
    print("  Loading VIGOR samples")
    print("=" * 60)
    easy_samples = []
    hard_samples = []
    if args.split in ["easy", "both"]:
        easy_samples = load_samples(args.data_dir, args.easy_json_file, "easy")
    if args.split in ["hard", "both"]:
        hard_samples = load_samples(args.data_dir, args.hard_json_file, "hard")
    if args.max_samples is not None:
        easy_samples = easy_samples[: args.max_samples]
        hard_samples = hard_samples[: args.max_samples]

    print(f"  Easy samples: {len(easy_samples)}")
    print(f"  Hard samples: {len(hard_samples)}")
    print(f"  Vocabulary: {args.vocabulary}")
    if args.vocabulary == "custom":
        print(f"  Custom vocabulary: {args.custom_vocabulary}")
    print(f"  Mask selection: {args.mask_selection}")
    print(f"  SSR threshold: {args.ssr_threshold}")

    pred_cache = PredictionCache(args.prediction_cache_size)
    results = {"easy": [], "hard": []}
    skipped_errors = {"easy": [], "hard": []}

    if easy_samples:
        print("\n" + "=" * 60)
        print("  Testing Easy samples")
        print("=" * 60)
        results["easy"], skipped_errors["easy"] = evaluate_split(
            "easy", easy_samples, demo, args, pred_cache
        )

    if hard_samples:
        print("\n" + "=" * 60)
        print("  Testing Hard samples")
        print("=" * 60)
        results["hard"], skipped_errors["hard"] = evaluate_split(
            "hard", hard_samples, demo, args, pred_cache
        )

    result_file = write_results(
        args=args,
        easy_results=results["easy"],
        hard_results=results["hard"],
        skipped_errors=skipped_errors,
    )

    easy_metrics = compute_metrics(results["easy"], args.ssr_threshold)
    hard_metrics = compute_metrics(results["hard"], args.ssr_threshold)
    avg_iou = weighted_average(easy_metrics, hard_metrics, "ic_iou")
    avg_ssr = weighted_average(easy_metrics, hard_metrics, "ssr")

    print("\n" + "=" * 60)
    print("  VLPart VIGOR evaluation finished")
    print("=" * 60)
    print(f"Result file: {result_file}")
    print(
        f"IC-IoU: Easy={easy_metrics['ic_iou']:.4f}, "
        f"Hard={hard_metrics['ic_iou']:.4f}, Avg={avg_iou:.4f}"
    )
    print(
        f"SSR@{args.ssr_threshold:.1f}: Easy={easy_metrics['ssr']:.4f}, "
        f"Hard={hard_metrics['ssr']:.4f}, Avg={avg_ssr:.4f}"
    )


if __name__ == "__main__":
    main()
