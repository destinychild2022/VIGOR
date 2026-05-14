import json
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

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from predictor import (  # noqa: E402
    BUILDIN_CLASSIFIER,
    BUILDIN_METADATA_PATH,
    get_clip_embeddings,
    reset_cls_test,
)
from test_vigor_vlpart import (  # noqa: E402
    PredictionCache,
    choose_single_mask,
    compute_iou_from_foreground,
    compute_metrics,
    load_gt_masks,
    normalize_custom_vocabulary,
    parse_args,
    resolve_cfg_repo_paths,
    resolve_data_path,
    resolve_repo_path,
    safe_name,
    save_prediction_overlay,
    save_raw_pred_mask,
    split_path_list,
    weighted_average,
)
from vlpart.config import add_vlpart_config  # noqa: E402
from vlpart.config_object_prior import add_object_prior_config  # noqa: E402

# Register experiment-only model classes.
import vlpart.modeling.meta_arch.vlm_rcnn_object_prior  # noqa: F401,E402
import vlpart.modeling.roi_heads.object_prior_roi_heads  # noqa: F401,E402


def read_vigor_prior_mask(mask_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    prior = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if prior is None:
        raise FileNotFoundError(f"Failed to read GT object prior mask: {mask_path}")
    if prior.shape != (h, w):
        prior = cv2.resize(prior, (w, h), interpolation=cv2.INTER_NEAREST)
    return prior


class ObjectPriorPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = (
            MetadataCatalog.get(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) else None
        )

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, object_prior_mask_path):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            prior_mask = read_vigor_prior_mask(object_prior_mask_path, (height, width))

            transform = self.aug.get_transform(original_image)
            image = transform.apply_image(original_image)
            prior_mask = transform.apply_segmentation(prior_mask)
            object_prior = (prior_mask == 0).astype("float32")[None, :, :]

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            object_prior = torch.as_tensor(np.ascontiguousarray(object_prior))
            inputs = {
                "image": image,
                "object_prior": object_prior,
                "height": height,
                "width": width,
            }
            return self.model([inputs])[0]


class VisualizationDemoObjectPrior:
    def __init__(self, cfg, args=None):
        if args is None:
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH["pascal_part"])
            classifier = BUILDIN_CLASSIFIER["pascal_part"]
        elif args.vocabulary == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(",")
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        elif args.vocabulary == "pascal_part_voc":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(",")
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        elif args.vocabulary == "lvis_paco":
            self.metadata = MetadataCatalog.get("__unused")
            lvis_thing_classes = MetadataCatalog.get(BUILDIN_METADATA_PATH["lvis"]).thing_classes
            paco_thing_classes = MetadataCatalog.get(BUILDIN_METADATA_PATH["paco"]).thing_classes[
                75:
            ]
            self.metadata.thing_classes = lvis_thing_classes + paco_thing_classes
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        self.predictor = ObjectPriorPredictor(cfg)
        self.cfg = cfg
        reset_cls_test(self.predictor.model, classifier)


def setup_cfg(args):
    cfg = get_cfg()
    add_vlpart_config(cfg)
    add_object_prior_config(cfg)
    cfg.merge_from_file(resolve_repo_path(args.config_file))
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.defrost()
    resolve_cfg_repo_paths(cfg)
    cfg.MODEL.WEIGHTS = resolve_repo_path(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.OBJECT_PRIOR.ENABLED = True
    cfg.MODEL.OBJECT_PRIOR.MODE = "gt"
    cfg.freeze()
    return cfg


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
    for idx, sample in enumerate(raw_samples):
        gt_object_path_raw = sample.get("gt_object_path", "")
        gt_mask_path_raw = sample.get("gt_mask_path", "")
        gt_object_mask_path_raw = sample.get("gt_object_mask_path", "")
        scene = str(sample.get("scene", "")).strip()
        if not scene or not gt_mask_path_raw or not gt_object_mask_path_raw:
            continue

        image_path = resolve_data_path(data_dir, f"{scene}.png")
        image_paths = [image_path]
        gt_mask_paths = [
            resolve_data_path(data_dir, path) for path in split_path_list(gt_mask_path_raw)
        ][:1]
        object_prior_paths = [
            resolve_data_path(data_dir, path)
            for path in split_path_list(gt_object_mask_path_raw)
        ][:1]
        if not image_paths or not gt_mask_paths or not object_prior_paths:
            continue

        instructions = sample.get("instructions", []) or [""]
        for instruction_index, instruction in enumerate(instructions):
            samples.append(
                {
                    "sample_index": idx,
                    "instruction_index": instruction_index,
                    "instruction": instruction,
                    "scene": scene,
                    "image_path": image_path,
                    "image_paths": image_paths,
                    "gt_mask_paths": gt_mask_paths,
                    "gt_mask_path": gt_mask_paths[0],
                    "object_prior_mask_path": object_prior_paths[0],
                    "object_prior_mask_paths": object_prior_paths,
                    "gt_object_mask_path": object_prior_paths[0],
                    "gt_object_mask_paths": object_prior_paths,
                    "gt_object": sample.get("gt_object", sample.get("object", "")),
                    "object": sample.get("object", ""),
                    "difficulty": difficulty,
                    "img_name": os.path.basename(image_paths[0]),
                }
            )
    return samples


def predict_one_image(
    demo: VisualizationDemoObjectPrior,
    image_path: str,
    object_prior_mask_path: str,
    mask_selection: str,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(object_prior_mask_path):
        raise FileNotFoundError(f"GT object prior mask not found: {object_prior_mask_path}")

    image_bgr = read_image(image_path, format="BGR")
    predictions = demo.predictor(image_bgr, object_prior_mask_path)
    pred_fg, pred_info = choose_single_mask(
        predictions=predictions,
        metadata=demo.metadata,
        image_shape=image_bgr.shape[:2],
        mask_selection=mask_selection,
    )
    return pred_fg, pred_info, image_bgr


def evaluate_sample_candidates(
    sample: Dict,
    demo: VisualizationDemoObjectPrior,
    args,
    pred_cache: PredictionCache,
) -> Dict:
    best = None
    image_paths = sample.get("image_paths") or [sample["image_path"]]
    object_prior_mask_path = sample["object_prior_mask_path"]

    for image_path in image_paths:
        cache_key = f"{image_path}|{object_prior_mask_path}"
        cached = pred_cache.get(cache_key)
        if cached is None:
            pred_fg, pred_info, image_bgr = predict_one_image(
                demo, image_path, object_prior_mask_path, args.mask_selection
            )
            pred_cache.put(cache_key, (pred_fg, pred_info, image_bgr))
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
                    "object_prior_mask_path": object_prior_mask_path,
                }

    if best is None:
        raise RuntimeError("No valid gt_object_path/gt_mask_path candidates")
    return best


def output_stem(sample: Dict, ordinal: int) -> str:
    base = Path(sample["image_path"]).stem
    obj = safe_name(sample.get("gt_object", "object"))
    instruction_index = int(sample.get("instruction_index", 0))
    return f"{ordinal:06d}_{safe_name(base)}_{obj}_ins{instruction_index:02d}"


def evaluate_split(
    split_name: str,
    samples: List[Dict],
    demo: VisualizationDemoObjectPrior,
    args,
    pred_cache: PredictionCache,
):
    results = []
    skipped = []

    for ordinal, sample in enumerate(tqdm(samples, desc=split_name.capitalize()), start=1):
        try:
            best = evaluate_sample_candidates(sample, demo, args, pred_cache)
            iou = best["iou"]
            success = iou >= args.ssr_threshold
            image_bgr = best["image_bgr"]
            pred_fg = best["pred_fg"]
            pred_info = best["pred_info"]

            selected_sample = dict(sample)
            selected_sample["image_path"] = best["image_path"]
            stem = output_stem(selected_sample, ordinal)
            vis_path = ""
            pred_mask_path = ""
            if args.save_vis:
                vis_path = str(Path(args.vis_dir) / split_name / f"{stem}_iou{iou:.3f}.png")
                save_prediction_overlay(
                    image_bgr=image_bgr,
                    pred_fg=pred_fg,
                    gt_mask=best["gt_mask"],
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
                    "image_path": best["image_path"],
                    "gt_mask_path": best["gt_mask_path"],
                    "object_prior_mask_path": best["object_prior_mask_path"],
                    "gt_object": sample["gt_object"],
                    "instruction": sample.get("instruction", ""),
                    "instruction_index": sample.get("instruction_index", 0),
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


def write_sample_result(f, split_title: str, index: int, result: Dict) -> None:
    f.write(f"[{split_title} Instruction Sample {index}]\n")
    f.write(f"  Image: {result['image_path']}\n")
    f.write(f"  GT Affordance Mask: {result['gt_mask_path']}\n")
    f.write(f"  GT Object Prior: {result['object_prior_mask_path']}\n")
    f.write(f"  Object: {result['gt_object']}\n")
    f.write(f"  Instruction {result['instruction_index']}: {result.get('instruction', '')}\n")
    f.write(f"  IC-IoU: {result['avg_ic_iou']:.4f}\n")
    f.write(f"  SSR Success: {int(result['success'])}\n")
    f.write(f"  Pred Class: {result['pred_class_name']} ({result['pred_class']})\n")
    f.write(f"  Pred Score: {result['pred_score']:.4f}\n")
    f.write(f"  Detected Instances: {result['num_instances']}\n")
    if result.get("pred_mask_path"):
        f.write(f"  Pred Mask: {result['pred_mask_path']}\n")
    if result.get("vis_path"):
        f.write(f"  Visualization: {result['vis_path']}\n")
    f.write("\n")


def write_results(args, easy_results, hard_results, skipped_errors):
    easy_metrics = compute_metrics(easy_results, args.ssr_threshold)
    hard_metrics = compute_metrics(hard_results, args.ssr_threshold)
    total_count = easy_metrics["count"] + hard_metrics["count"]
    avg_ic_iou = weighted_average(easy_metrics, hard_metrics, "ic_iou")
    avg_ssr = weighted_average(easy_metrics, hard_metrics, "ssr")
    total_success = easy_metrics["success_count"] + hard_metrics["success_count"]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = str(Path(args.output_dir) / f"vlpart_gt_object_prior_results_{timestamp}.txt")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart VIGOR-100K GT Object Prior 测试结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"模型配置: {args.config_file}\n")
        f.write(f"模型权重: {args.weights}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write("Object prior: GT object mask tensor, not multiplied into RGB\n")
        f.write("Instruction text: not fed into VLPart; each instruction counted once\n")
        f.write(f"Vocabulary: {args.custom_vocabulary if args.vocabulary == 'custom' else args.vocabulary}\n")
        f.write(f"Mask selection: {args.mask_selection}\n")
        f.write(f"Confidence threshold: {args.confidence_threshold}\n\n")

        f.write(
            "| Method | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            "SSR Easy | SSR Hard | SSR Avg |\n"
        )
        f.write(
            f"| VLPart+GTObjectPrior | {easy_metrics['ic_iou']:.4f} | "
            f"{hard_metrics['ic_iou']:.4f} | {avg_ic_iou:.4f} | "
            f"{easy_metrics['ssr']:.4f} | {hard_metrics['ssr']:.4f} | {avg_ssr:.4f} |\n\n"
        )

        f.write(f"总 instruction 样本数: {total_count}\n")
        f.write(f"  Easy: {easy_metrics['count']}\n")
        f.write(f"  Hard: {hard_metrics['count']}\n")
        f.write(f"成功数: {total_success} / {total_count}\n")
        f.write(f"跳过样本数: Easy={len(skipped_errors['easy'])}, Hard={len(skipped_errors['hard'])}\n\n")

        for split_name in ["easy", "hard"]:
            if skipped_errors[split_name]:
                f.write(f"[{split_name} skipped examples]\n")
                for path, err in skipped_errors[split_name][:5]:
                    f.write(f"  {path}: {err}\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("  Easy instruction 样本详细结果\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(easy_results, start=1):
            write_sample_result(f, "Easy", i, result)

        f.write("=" * 80 + "\n")
        f.write("  Hard instruction 样本详细结果\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(hard_results, start=1):
            write_sample_result(f, "Hard", i, result)

    return result_file


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
    print("  Loading VLPart GT Object Prior")
    print("=" * 60)
    cfg = setup_cfg(args)
    demo = VisualizationDemoObjectPrior(cfg, args)

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
    print(f"  Vocabulary: {args.vocabulary}")
    if args.vocabulary == "custom":
        print(f"  Custom vocabulary: {args.custom_vocabulary}")
    print("  Object prior: GT object mask tensor")
    print("  Instruction text input: disabled")
    print(f"  Mask selection: {args.mask_selection}")
    print(f"  SSR threshold: {args.ssr_threshold}")

    pred_cache = PredictionCache(args.prediction_cache_size)
    results = {"easy": [], "hard": []}
    skipped_errors = {"easy": [], "hard": []}

    if easy_samples:
        print("\n" + "=" * 60)
        print("  Testing Easy instruction samples")
        print("=" * 60)
        results["easy"], skipped_errors["easy"] = evaluate_split(
            "easy", easy_samples, demo, args, pred_cache
        )

    if hard_samples:
        print("\n" + "=" * 60)
        print("  Testing Hard instruction samples")
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
    print("  VLPart GT Object Prior evaluation finished")
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
