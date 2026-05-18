import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from detectron2.data.detection_utils import read_image  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from predictor import VisualizationDemo  # noqa: E402
from test_vigor_vlpart import (  # noqa: E402
    PredictionCache,
    compute_iou_from_foreground,
    compute_metrics,
    load_gt_masks,
    normalize_custom_vocabulary,
    resolve_repo_path,
    save_prediction_overlay,
    save_raw_pred_mask,
    setup_cfg,
    weighted_average,
)
from test_vigor_vlpart_llmseg_topk_masked_rgb import (  # noqa: E402
    compute_foreground_iou,
    compute_icr_metrics,
    compute_split_summary,
    filter_instruction_group,
    load_samples_per_instruction,
    make_masked_rgb,
    output_stem,
    raw_sample_key,
    read_llmseg_object_mask,
    resolve_existing_or_repo_path,
    resolve_topk_mask_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "VLPart-score selection on LLMSeg top-K masked RGB candidates"
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--easy_json_file", default="open_vocab_grasp_easy_object_mix.json")
    parser.add_argument("--hard_json_file", default="open_vocab_grasp_hard_object_mix.json")
    parser.add_argument("--split", default="both", choices=["easy", "hard", "both"])
    parser.add_argument("--llmseg_topk_masks_dir", required=True)
    parser.add_argument("--topk_mask_k", type=int, default=3)
    parser.add_argument("--output_dir", default="vlpart_vigor_outputs/llmseg_topk_vlpart_score/results")
    parser.add_argument("--vis_dir", default="vlpart_vigor_outputs/llmseg_topk_vlpart_score/visualizations")
    parser.add_argument("--pred_masks_dir", default="vlpart_vigor_outputs/llmseg_topk_vlpart_score/pred_masks")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_pred_masks", action="store_true")
    parser.add_argument("--vocabulary", default="custom")
    parser.add_argument(
        "--custom_vocabulary",
        default="cylindrical side surface,hexagonal side face,flat side surface,whole object",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--ssr_threshold", type=float, default=0.5)
    parser.add_argument("--icr_threshold", type=float, default=0.7)
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


def instances_to_score_candidates(predictions: Dict, metadata, image_shape: Tuple[int, int]):
    h, w = image_shape
    instances = predictions.get("instances")
    if instances is None or len(instances) == 0 or not instances.has("pred_masks"):
        return []

    instances = instances.to("cpu")
    masks = instances.pred_masks.numpy().astype(bool)
    scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances))
    classes = (
        instances.pred_classes.numpy()
        if instances.has("pred_classes")
        else np.full(len(instances), -1, dtype=np.int64)
    )
    class_names = metadata.get("thing_classes", [])
    candidates = []
    for idx, mask in enumerate(masks):
        pred_fg = mask
        if pred_fg.shape != (h, w):
            pred_fg = cv2.resize(
                pred_fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        class_idx = int(classes[idx])
        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = str(class_idx) if class_idx >= 0 else ""
        candidates.append(
            {
                "pred_fg": pred_fg,
                "pred_score": float(scores[idx]),
                "pred_class": class_idx,
                "pred_class_name": class_name,
                "vlpart_instance_index": int(idx),
            }
        )
    return candidates


def evaluate_sample_by_vlpart_score(
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
    best = None
    candidate_count = 0

    for rank, topk_mask_path in topk_mask_paths:
        cache_key = f"{image_path}|{topk_mask_path}|all_vlpart_score_candidates"
        cached = pred_cache.get(cache_key)
        if cached is None:
            object_fg = read_llmseg_object_mask(topk_mask_path, image_bgr.shape[:2])
            masked_bgr = make_masked_rgb(image_bgr, object_fg)
            predictions = demo.predictor(masked_bgr)
            candidates = instances_to_score_candidates(
                predictions, demo.metadata, image_bgr.shape[:2]
            )
            pred_cache.put(cache_key, (candidates, masked_bgr))
        else:
            candidates, masked_bgr = cached

        if not candidates:
            candidates = [
                {
                    "pred_fg": np.zeros(image_bgr.shape[:2], dtype=bool),
                    "pred_score": 0.0,
                    "pred_class": -1,
                    "pred_class_name": "",
                    "vlpart_instance_index": -1,
                }
            ]

        for candidate in candidates:
            candidate_count += 1
            if best is None or candidate["pred_score"] > best["pred_score"]:
                best = {
                    "image_path": image_path,
                    "image_bgr": image_bgr,
                    "masked_bgr": masked_bgr,
                    "pred_fg": candidate["pred_fg"],
                    "pred_score": candidate["pred_score"],
                    "pred_class": candidate["pred_class"],
                    "pred_class_name": candidate["pred_class_name"],
                    "vlpart_instance_index": candidate["vlpart_instance_index"],
                    "llmseg_topk_rank": rank,
                    "llmseg_topk_mask_path": topk_mask_path,
                }

    if best is None:
        raise RuntimeError("No valid VLPart score candidates")

    selected_iou = 0.0
    selected_gt = gt_masks[0] if gt_masks else None
    selected_gt_path = sample["gt_mask_paths"][0] if sample["gt_mask_paths"] else ""
    for gt_mask_path, gt_mask in zip(sample["gt_mask_paths"], gt_masks):
        iou = compute_iou_from_foreground(best["pred_fg"], gt_mask)
        if iou >= selected_iou:
            selected_iou = iou
            selected_gt = gt_mask
            selected_gt_path = gt_mask_path

    best.update(
        {
            "iou": selected_iou,
            "gt_mask": selected_gt,
            "gt_mask_path": selected_gt_path,
            "candidate_count": candidate_count,
        }
    )
    return best


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
            icr_records.append(
                {
                    "sample_key": current_key,
                    "iou_pairs": [
                        compute_foreground_iou(first_three[0], first_three[1]),
                        compute_foreground_iou(first_three[0], first_three[2]),
                        compute_foreground_iou(first_three[1], first_three[2]),
                    ],
                }
            )
        current_predictions = []

    for ordinal, sample in enumerate(tqdm(samples, desc=split_name.capitalize()), start=1):
        sample_key = raw_sample_key(sample)
        if current_key is None:
            current_key = sample_key
        elif sample_key != current_key:
            flush_icr_record()
            current_key = sample_key

        try:
            best = evaluate_sample_by_vlpart_score(sample, demo, args, pred_cache)
            iou = best["iou"]
            success = iou >= args.ssr_threshold
            instruction_index = int(sample.get("instruction_index", 0))
            selected_sample = dict(sample)
            stem = output_stem(selected_sample, ordinal)
            pred_mask_path = ""
            vis_path = ""

            pred_info = {
                "pred_score": best["pred_score"],
                "pred_class": best["pred_class"],
                "pred_class_name": best["pred_class_name"],
                "num_instances": best["candidate_count"],
            }

            if args.save_pred_masks:
                pred_mask_path = str(Path(args.pred_masks_dir) / split_name / f"{stem}.png")
                save_raw_pred_mask(best["pred_fg"], pred_mask_path)
            if args.save_vis:
                vis_path = str(Path(args.vis_dir) / split_name / f"{stem}_iou{iou:.3f}.png")
                save_prediction_overlay(
                    image_bgr=best["image_bgr"],
                    pred_fg=best["pred_fg"],
                    gt_mask=best["gt_mask"],
                    output_path=vis_path,
                    sample=selected_sample,
                    iou=iou,
                    pred_info=pred_info,
                )

            results.append(
                {
                    "image_path": best["image_path"],
                    "gt_mask_path": best["gt_mask_path"],
                    "gt_object": sample["gt_object"],
                    "sample_index": sample.get("sample_index", -1),
                    "scene": sample.get("scene", ""),
                    "instruction": sample.get("instruction", ""),
                    "instruction_index": instruction_index,
                    "difficulty": sample["difficulty"],
                    "iou": iou,
                    "ic_ious": [iou],
                    "avg_ic_iou": iou,
                    "success": success,
                    "pred_class": best["pred_class"],
                    "pred_class_name": best["pred_class_name"],
                    "pred_score": best["pred_score"],
                    "vlpart_instance_index": best["vlpart_instance_index"],
                    "llmseg_topk_rank": best["llmseg_topk_rank"],
                    "llmseg_topk_mask_path": best["llmseg_topk_mask_path"],
                    "candidate_count": best["candidate_count"],
                    "pred_mask_path": pred_mask_path,
                    "vis_path": vis_path,
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
    f.write(f"  Object: {result['gt_object']}\n")
    f.write(f"  Instruction {result['instruction_index']}: {result.get('instruction', '')}\n")
    f.write(f"  IC-IoU: {result['avg_ic_iou']:.4f}\n")
    f.write(f"  SSR Success: {int(result['success'])}\n")
    f.write(f"  VLPart Score: {result['pred_score']:.4f}\n")
    f.write(f"  Pred Class: {result['pred_class_name']} ({result['pred_class']})\n")
    f.write(f"  Selected TopK Rank: {result['llmseg_topk_rank']}\n")
    f.write(f"  VLPart Instance Index: {result['vlpart_instance_index']}\n")
    f.write(f"  Candidate Count: {result['candidate_count']}\n")
    f.write(f"  LLMSeg TopK Mask: {result['llmseg_topk_mask_path']}\n")
    if result.get("pred_mask_path"):
        f.write(f"  Pred Mask: {result['pred_mask_path']}\n")
    if result.get("vis_path"):
        f.write(f"  Visualization: {result['vis_path']}\n")
    f.write("\n")


def write_results(args, easy_results, hard_results, skipped_errors, easy_icr_records, hard_icr_records):
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
    easy_icr = compute_icr_metrics(easy_icr_records, args.icr_threshold)
    hard_icr = compute_icr_metrics(hard_icr_records, args.icr_threshold)
    avg_icr = weighted_average(easy_icr, hard_icr, "icr")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_file = str(
        Path(args.output_dir) / f"vlpart_llmseg_topk_vlpart_score_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart VIGOR-100K LLMSeg Top-K VLPart-Score Selection Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model config: {args.config_file}\n")
        f.write(f"Model weights: {args.weights}\n")
        f.write(f"Data dir: {args.data_dir}\n")
        f.write(f"LLMSeg top-K masks: {args.llmseg_topk_masks_dir}\n")
        f.write(f"Top-K K: {args.topk_mask_k}\n")
        f.write("Selection: highest VLPart instance score across all top-K masked RGB outputs\n")
        f.write("GT affordance masks are used only for evaluation, not candidate selection.\n\n")

        f.write(
            "| Method | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            f"SSR@{args.ssr_threshold:.1f} Easy | SSR@{args.ssr_threshold:.1f} Hard | "
            f"SSR@{args.ssr_threshold:.1f} Avg | ICR@{args.icr_threshold:.1f} Easy | "
            f"ICR@{args.icr_threshold:.1f} Hard | ICR@{args.icr_threshold:.1f} Avg |\n"
        )
        f.write(
            f"| VLPartScore | {easy_metrics['ic_iou']:.4f} | "
            f"{hard_metrics['ic_iou']:.4f} | {all_summary['avg_ic_iou']:.4f} | "
            f"{easy_metrics['ssr']:.4f} | {hard_metrics['ssr']:.4f} | "
            f"{all_summary['avg_ssr']:.4f} | {easy_icr['icr']:.4f} | "
            f"{hard_icr['icr']:.4f} | {avg_icr:.4f} |\n\n"
        )

        f.write(f"Total instruction samples: {all_summary['total_count']}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n")
        f.write(f"Skipped samples: Easy={len(skipped_errors['easy'])}, Hard={len(skipped_errors['hard'])}\n\n")

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

        f.write(f"ICR@{args.icr_threshold:.1f}:\n")
        f.write(f"  Easy:    {easy_icr['icr']:.4f} ({easy_icr['count']} samples)\n")
        f.write(f"  Hard:    {hard_icr['icr']:.4f} ({hard_icr['count']} samples)\n")
        f.write(f"  Average: {avg_icr:.4f}\n\n")

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


def main():
    args = parse_args()
    if args.topk_mask_k <= 0:
        raise ValueError("--topk_mask_k must be > 0")

    args.config_file = resolve_repo_path(args.config_file)
    args.weights = resolve_repo_path(args.weights)
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.llmseg_topk_masks_dir = resolve_existing_or_repo_path(args.llmseg_topk_masks_dir)
    if args.vocabulary == "custom":
        args.custom_vocabulary = normalize_custom_vocabulary(args.custom_vocabulary)

    setup_logger(name="fvcore")
    setup_logger().info("Arguments: " + str(args))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    if args.save_vis:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
    if args.save_pred_masks:
        Path(args.pred_masks_dir).mkdir(parents=True, exist_ok=True)

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)
    easy_samples = []
    hard_samples = []
    if args.split in ["easy", "both"]:
        easy_samples = load_samples_per_instruction(args.data_dir, args.easy_json_file, "easy")
    if args.split in ["hard", "both"]:
        hard_samples = load_samples_per_instruction(args.data_dir, args.hard_json_file, "hard")
    if args.max_samples is not None:
        easy_samples = easy_samples[: args.max_samples]
        hard_samples = hard_samples[: args.max_samples]

    pred_cache = PredictionCache(args.prediction_cache_size)
    results = {"easy": [], "hard": []}
    skipped = {"easy": [], "hard": []}
    icr_records = {"easy": [], "hard": []}

    if easy_samples:
        results["easy"], skipped["easy"], icr_records["easy"] = evaluate_split(
            "easy", easy_samples, demo, args, pred_cache
        )
    if hard_samples:
        results["hard"], skipped["hard"], icr_records["hard"] = evaluate_split(
            "hard", hard_samples, demo, args, pred_cache
        )

    result_file = write_results(
        args,
        results["easy"],
        results["hard"],
        skipped,
        icr_records["easy"],
        icr_records["hard"],
    )
    summary = compute_split_summary(results["easy"], results["hard"], args.ssr_threshold)
    print(f"Result file: {result_file}")
    print(
        f"IC-IoU: Easy={summary['easy']['ic_iou']:.4f}, "
        f"Hard={summary['hard']['ic_iou']:.4f}, Avg={summary['avg_ic_iou']:.4f}"
    )
    print(
        f"SSR@{args.ssr_threshold:.1f}: Easy={summary['easy']['ssr']:.4f}, "
        f"Hard={summary['hard']['ssr']:.4f}, Avg={summary['avg_ssr']:.4f}"
    )


if __name__ == "__main__":
    main()
