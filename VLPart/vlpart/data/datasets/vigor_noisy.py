# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import fcntl
import hashlib
import json
import logging
import os
import pickle
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


DEFAULT_VIGOR_ROOT = "/opt/data/private/LLMSeg/dataset/VIGOR-100K_new"
DEFAULT_MAPPING_PATH = "configs/vigor/vigor_easy_object_to_vocabulary.json"
DEFAULT_CACHE_VERSION = "vigor_noisy_dataset_cache_v1"
DEFAULT_TOPK_MASKS_DIR = "/opt/data/private/LLMSeg/vis_output_object_topk_trainset/topk_masks"
DEFAULT_VOCABULARY = [
    "cylindrical side surface",
    "hexagonal side face",
    "flat side surface",
    "whole object",
]

logger = logging.getLogger("detectron2.vigor_noisy")

_OPTIONS = {
    "enabled": True,
    "noisy_ratio": 0.5,
    "topk_masks_dir": DEFAULT_TOPK_MASKS_DIR,
    "topk_rank": 1,
    "difficulty": "easy",
    "instruction_index": 0,
    "object_iou_threshold": 0.5,
    "aff_coverage_threshold": 0.7,
    "random_seed": 42,
}


def set_vigor_noisy_options(
    *,
    enabled=True,
    noisy_ratio=0.5,
    topk_masks_dir=DEFAULT_TOPK_MASKS_DIR,
    topk_rank=1,
    difficulty="easy",
    instruction_index=0,
    object_iou_threshold=0.5,
    aff_coverage_threshold=0.7,
    random_seed=42,
):
    _OPTIONS.update(
        {
            "enabled": bool(enabled),
            "noisy_ratio": float(noisy_ratio),
            "topk_masks_dir": str(topk_masks_dir),
            "topk_rank": int(topk_rank),
            "difficulty": str(difficulty),
            "instruction_index": int(instruction_index),
            "object_iou_threshold": float(object_iou_threshold),
            "aff_coverage_threshold": float(aff_coverage_threshold),
            "random_seed": int(random_seed),
        }
    )


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _resolve_path(root, path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


def _split_paths(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_samples(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported VIGOR JSON format: {json_file}")


def _load_mapping(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocabulary = data.get("vocabulary", DEFAULT_VOCABULARY)
    object_to_vocabulary = data["object_to_vocabulary"]
    vocab_to_id = {name: idx for idx, name in enumerate(vocabulary)}
    return vocabulary, object_to_vocabulary, vocab_to_id


def _file_signature(path):
    path = Path(path)
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def _rank_dir_signature(rank_dir):
    rank_dir = Path(rank_dir)
    if not rank_dir.exists():
        return f"{rank_dir.resolve()}:missing"
    stat = rank_dir.stat()
    return f"{rank_dir.resolve()}:{stat.st_mtime_ns}"


def _cache_path(dataset_name, json_file, data_root, mapping_file, options):
    cache_dir = Path(os.getenv("VIGOR_CACHE_DIR", _repo_root() / "datasets" / "cache" / "vigor"))
    rank_dir = (
        Path(options["topk_masks_dir"])
        / f"top{int(options['topk_rank'])}"
        / str(options["difficulty"])
    )
    key_parts = [
        DEFAULT_CACHE_VERSION,
        dataset_name or "vigor_noisy",
        str(Path(data_root).resolve()),
        _file_signature(json_file),
        _file_signature(mapping_file),
        _rank_dir_signature(rank_dir),
        f"enabled={options['enabled']}",
        f"ratio={float(options['noisy_ratio']):.6f}",
        f"rank={int(options['topk_rank'])}",
        f"difficulty={options['difficulty']}",
        f"instr={int(options['instruction_index'])}",
        f"object_iou>{float(options['object_iou_threshold']):.6f}",
        f"aff_cov>{float(options['aff_coverage_threshold']):.6f}",
        f"seed={int(options['random_seed'])}",
    ]
    digest = hashlib.md5("|".join(key_parts).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{dataset_name or 'vigor_noisy'}_{digest}.pkl"


def _load_cache(cache_file):
    if not cache_file.exists():
        return None
    with cache_file.open("rb") as f:
        dataset_dicts = pickle.load(f)
    logger.info("Loaded VIGOR noisy dataset cache: %s (%d samples)", cache_file, len(dataset_dicts))
    return dataset_dicts


def _write_cache(cache_file, dataset_dicts):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
    with tmp_file.open("wb") as f:
        pickle.dump(dataset_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_file, cache_file)
    logger.info("Wrote VIGOR noisy dataset cache: %s (%d samples)", cache_file, len(dataset_dicts))


def _mask_to_annotation(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read VIGOR mask: {mask_path}")

    fg = np.asfortranarray((mask == 0).astype(np.uint8))
    if fg.sum() == 0:
        raise ValueError(f"Empty foreground mask: {mask_path}")

    rle = mask_util.encode(fg)
    rle["counts"] = rle["counts"].decode("ascii")
    bbox = mask_util.toBbox(rle).tolist()
    return rle, bbox, int(mask.shape[0]), int(mask.shape[1])


def _read_foreground(mask_path, image_shape=None):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if image_shape is not None and mask.shape != tuple(image_shape):
        h, w = image_shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask == 0


def _compute_iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _compute_coverage(pred, aff_gt):
    denom = aff_gt.sum()
    if denom == 0:
        return 0.0
    return float(np.logical_and(pred, aff_gt).sum() / denom)


def _sanitize_filename_token(value):
    token = str(value) if value is not None else "unknown"
    token = token.replace("/", "_").replace("\\", "_").replace(" ", "_")
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _scene_name_from_sample(sample):
    scene = str(sample.get("scene", "")).strip()
    if scene:
        return scene

    for key in ("gt_mask_path", "gt_object_mask_path", "gt_object_path"):
        paths = _split_paths(sample.get(key, ""))
        if not paths:
            continue
        match = re.match(r"^(\d+)_", os.path.basename(paths[0]))
        if match:
            return match.group(1)
    return ""


def _build_topk_index(options):
    rank = int(options["topk_rank"])
    rank_dir = Path(options["topk_masks_dir"]) / f"top{rank}" / str(options["difficulty"])
    marker = f"_top{rank}_"
    index = {}
    if not rank_dir.is_dir():
        logger.warning("LLM-Seg top-K mask directory not found: %s", rank_dir)
        return index

    start = time.perf_counter()
    scanned = 0
    for entry in os.scandir(rank_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if not name.endswith(".png"):
            continue
        marker_pos = name.find(marker)
        if marker_pos < 0:
            continue
        prefix = name[: marker_pos + len(marker)]
        index.setdefault(prefix, entry.path)
        scanned += 1
        if scanned % 50000 == 0:
            logger.info("Indexed %d LLM-Seg top-K masks from %s", scanned, rank_dir)

    logger.info(
        "Indexed %d LLM-Seg top-K masks from %s in %.1fs",
        len(index),
        rank_dir,
        time.perf_counter() - start,
    )
    return index


def _build_noisy_prefix(sample, obj_count, options):
    rank = int(options["topk_rank"])
    instruction_index = int(options["instruction_index"])
    scene = _scene_name_from_sample(sample)
    if not scene:
        return ""
    img_name = f"{scene}.png"
    obj_name = _sanitize_filename_token(sample.get("gt_object", sample.get("object", "unknown")))
    return f"{img_name}_{obj_name}_{obj_count}_instr{instruction_index}_top{rank}_"


def load_vigor_noisy_json(json_file, data_root, mapping_file, dataset_name=None):
    options = copy.deepcopy(_OPTIONS)
    cache_file = _cache_path(dataset_name, json_file, data_root, mapping_file, options)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

        dataset_dicts = _build_vigor_noisy_json(json_file, data_root, mapping_file, options, dataset_name)
        _write_cache(cache_file, dataset_dicts)
        return dataset_dicts


def _build_vigor_noisy_json(json_file, data_root, mapping_file, options, dataset_name=None):
    vocabulary, object_to_vocabulary, vocab_to_id = _load_mapping(mapping_file)
    samples = _load_samples(json_file)
    topk_index = _build_topk_index(options) if options["enabled"] else {}
    records = []
    valid_noisy = []
    seen_counts = {}
    missing_topk = 0
    rejected_object_iou = 0
    rejected_aff_coverage = 0
    log_period = max(1, int(os.getenv("VIGOR_LOAD_LOG_PERIOD", "1000")))
    start_time = time.perf_counter()

    logger.info(
        "Building VIGOR noisy dataset %s from %s (%d samples), noisy_ratio=%.3f",
        dataset_name or "<unnamed>",
        json_file,
        len(samples),
        float(options["noisy_ratio"]),
    )

    for idx, sample in enumerate(samples):
        gt_object = sample.get("gt_object", sample.get("object", ""))
        if gt_object not in object_to_vocabulary:
            raise KeyError(f"gt_object '{gt_object}' is missing from mapping file: {mapping_file}")

        image_paths = _split_paths(sample.get("gt_object_path", ""))
        aff_mask_paths = _split_paths(sample.get("gt_mask_path", ""))
        object_mask_paths = _split_paths(sample.get("gt_object_mask_path", ""))
        if len(image_paths) != 1 or len(aff_mask_paths) != 1:
            raise ValueError(
                "vigor_easy_train expects one GT object image and one affordance mask per sample. "
                f"Got image_paths={image_paths}, aff_mask_paths={aff_mask_paths}"
            )

        scene = _scene_name_from_sample(sample)
        if not scene:
            raise ValueError(f"Failed to infer scene id for sample index {idx}")

        clean_image_path = _resolve_path(data_root, image_paths[0])
        scene_image_path = _resolve_path(data_root, f"{scene}.png")
        aff_mask_path = _resolve_path(data_root, aff_mask_paths[0])
        object_mask_path = _resolve_path(data_root, object_mask_paths[0]) if object_mask_paths else ""
        if not os.path.exists(clean_image_path):
            raise FileNotFoundError(f"GT object masked RGB not found: {clean_image_path}")
        if not os.path.exists(scene_image_path):
            raise FileNotFoundError(f"Scene image not found: {scene_image_path}")

        segmentation, bbox, height, width = _mask_to_annotation(aff_mask_path)
        vocabulary_name = object_to_vocabulary[gt_object]
        category_id = vocab_to_id[vocabulary_name]

        img_name = f"{scene}.png"
        count_key = (img_name, gt_object)
        seen_counts[count_key] = seen_counts.get(count_key, 0) + 1
        obj_count = seen_counts[count_key]

        record = {
            "file_name": clean_image_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "vigor_scene": scene,
            "vigor_scene_file_name": scene_image_path,
            "vigor_gt_object": gt_object,
            "vigor_input_type": "gt_object",
            "gt_affordance_mask_path": aff_mask_path,
            "gt_object_mask_path": object_mask_path,
            "annotations": [
                {
                    "iscrowd": 0,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": category_id,
                    "segmentation": segmentation,
                }
            ],
        }

        if options["enabled"] and object_mask_path:
            prefix = _build_noisy_prefix(sample, obj_count, options)
            pred_mask_path = topk_index.get(prefix)
            if pred_mask_path:
                try:
                    pred_fg = _read_foreground(pred_mask_path, (height, width))
                    object_fg = _read_foreground(object_mask_path, (height, width))
                    aff_fg = _read_foreground(aff_mask_path, (height, width))
                    object_iou = _compute_iou(pred_fg, object_fg)
                    aff_coverage = _compute_coverage(pred_fg, aff_fg)
                except Exception as exc:
                    logger.warning("Failed to validate noisy mask %s: %s", pred_mask_path, exc)
                else:
                    if object_iou <= float(options["object_iou_threshold"]):
                        rejected_object_iou += 1
                    elif aff_coverage <= float(options["aff_coverage_threshold"]):
                        rejected_aff_coverage += 1
                    else:
                        valid_noisy.append((idx, pred_mask_path, object_iou, aff_coverage))
            else:
                missing_topk += 1

        records.append(record)
        if (idx + 1) % log_period == 0 or idx + 1 == len(samples):
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Built VIGOR noisy dataset base records: %d / %d samples (%.1f samples/s)",
                idx + 1,
                len(samples),
                (idx + 1) / max(elapsed, 1e-6),
            )

    if options["enabled"]:
        target_noisy = int(round(len(records) * min(max(float(options["noisy_ratio"]), 0.0), 1.0)))
    else:
        target_noisy = 0
    rng = random.Random(int(options["random_seed"]))
    rng.shuffle(valid_noisy)
    selected_noisy = valid_noisy[:target_noisy]

    for idx, pred_mask_path, object_iou, aff_coverage in selected_noisy:
        records[idx]["vigor_input_type"] = "llmseg_pred"
        records[idx]["vigor_llmseg_object_mask_path"] = pred_mask_path
        records[idx]["vigor_llmseg_object_iou"] = object_iou
        records[idx]["vigor_llmseg_aff_coverage"] = aff_coverage

    if target_noisy and len(valid_noisy) < target_noisy:
        logger.warning(
            "Only %d valid noisy masks are available for target noisy count %d; "
            "actual noisy ratio is %.4f",
            len(valid_noisy),
            target_noisy,
            len(selected_noisy) / max(len(records), 1),
        )

    logger.info(
        "VIGOR noisy dataset %s: total=%d, selected_noisy=%d, clean=%d, "
        "valid_noisy=%d, missing_topk=%d, rejected_object_iou=%d, rejected_aff_coverage=%d",
        dataset_name or "<unnamed>",
        len(records),
        len(selected_noisy),
        len(records) - len(selected_noisy),
        len(valid_noisy),
        missing_topk,
        rejected_object_iou,
        rejected_aff_coverage,
    )
    return records


def register_vigor_noisy_instances(name, json_file, data_root, mapping_file):
    DatasetCatalog.register(
        name,
        lambda: load_vigor_noisy_json(json_file, data_root, mapping_file, name),
    )
    vocabulary, _, _ = _load_mapping(mapping_file)
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=data_root,
        mapping_file=mapping_file,
        evaluator_type="coco",
        thing_classes=vocabulary,
        thing_dataset_id_to_contiguous_id={idx: idx for idx in range(len(vocabulary))},
    )


def register_all_vigor_noisy():
    repo_root = _repo_root()
    mapping_file = str(repo_root / DEFAULT_MAPPING_PATH)
    root = os.getenv("VIGOR_DATASETS", DEFAULT_VIGOR_ROOT)

    register_vigor_noisy_instances(
        "vigor_easy_train_llmseg_noisy",
        os.path.join(root, "train", "open_vocab_grasp_easy_object_mix.json"),
        os.path.join(root, "train"),
        mapping_file,
    )


register_all_vigor_noisy()
