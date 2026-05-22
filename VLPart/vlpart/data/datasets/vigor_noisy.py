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
DEFAULT_CACHE_VERSION = "vigor_gt_nearby_noisy_dataset_cache_v1"
DEFAULT_VOCABULARY = [
    "cylindrical side surface",
    "hexagonal side face",
    "flat side surface",
    "whole object",
]

logger = logging.getLogger("detectron2.vigor_noisy")

_OPTIONS = {
    "enabled": True,
    # Extra noisy duplicate samples relative to the full clean easy set.
    "noisy_ratio": 0.2,
    "nearby_topk": 1,
    "max_center_distance": 150.0,
    "random_seed": 42,
}


def set_vigor_noisy_options(
    *,
    enabled=True,
    noisy_ratio=0.2,
    nearby_topk=1,
    max_center_distance=150.0,
    random_seed=42,
):
    _OPTIONS.update(
        {
            "enabled": bool(enabled),
            "noisy_ratio": float(noisy_ratio),
            "nearby_topk": int(nearby_topk),
            "max_center_distance": float(max_center_distance),
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


def _cache_path(dataset_name, json_file, data_root, mapping_file, options):
    cache_dir = Path(os.getenv("VIGOR_CACHE_DIR", _repo_root() / "datasets" / "cache" / "vigor"))
    key_parts = [
        DEFAULT_CACHE_VERSION,
        dataset_name or "vigor_noisy",
        str(Path(data_root).resolve()),
        _file_signature(json_file),
        _file_signature(mapping_file),
        "enabled={}".format(options["enabled"]),
        "extra_ratio={:.6f}".format(float(options["noisy_ratio"])),
        "nearby_topk={}".format(int(options["nearby_topk"])),
        "max_center_distance={:.3f}".format(float(options["max_center_distance"])),
        "seed={}".format(int(options["random_seed"])),
    ]
    digest = hashlib.md5("|".join(key_parts).encode("utf-8")).hexdigest()[:12]
    return cache_dir / "{}_{}.pkl".format(dataset_name or "vigor_noisy", digest)


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


def _object_mask_info(mask_path, image_shape):
    fg = _read_foreground(mask_path, image_shape)
    ys, xs = np.where(fg)
    if len(xs) == 0:
        raise ValueError(f"Empty object foreground mask: {mask_path}")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
    center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    return {
        "bbox": bbox,
        "center": center,
        "area": int(fg.sum()),
    }


def _squared_distance(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def _ensure_scene_object_info(scene_candidates):
    valid = []
    for item in scene_candidates:
        if "center" not in item or "bbox" not in item:
            try:
                object_info = _object_mask_info(item["object_mask_path"], item["image_shape"])
            except Exception as exc:
                logger.warning("Failed to read GT object mask %s for nearby-noisy candidate: %s", item["object_mask_path"], exc)
                continue
            item["bbox"] = object_info["bbox"]
            item["center"] = object_info["center"]
            item["area"] = object_info["area"]
        valid.append(item)
    return valid


def _select_nearby_partner(target, candidates_by_scene, rng, nearby_topk, max_center_distance):
    scene_candidates = _ensure_scene_object_info(candidates_by_scene.get(target["scene"], []))
    if "center" not in target:
        return None, None, "missing_target_mask"

    pool = []
    for item in scene_candidates:
        if item["sample_index"] == target["sample_index"]:
            continue
        distance = float(np.sqrt(_squared_distance(target["center"], item["center"])))
        pool.append((item, distance))
    if not pool:
        return None, None, "without_partner"

    pool.sort(key=lambda item_and_distance: item_and_distance[1])
    max_distance = float(max_center_distance)
    if max_distance > 0:
        pool = [item_and_distance for item_and_distance in pool if item_and_distance[1] <= max_distance]
        if not pool:
            return None, None, "too_far"

    k = max(1, min(int(nearby_topk), len(pool)))
    partner, distance = rng.choice(pool[:k])
    return partner, distance, "ok"


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
    records = []
    candidates = []
    log_period = max(1, int(os.getenv("VIGOR_LOAD_LOG_PERIOD", "1000")))
    start_time = time.perf_counter()

    logger.info(
        "Building VIGOR GT+nearby noisy dataset %s from %s (%d clean samples), extra_noisy_ratio=%.3f, nearby_topk=%d, max_center_distance=%.1f",
        dataset_name or "<unnamed>",
        json_file,
        len(samples),
        float(options["noisy_ratio"]),
        int(options["nearby_topk"]),
        float(options["max_center_distance"]),
    )

    for idx, sample in enumerate(samples):
        gt_object = sample.get("gt_object", sample.get("object", ""))
        if gt_object not in object_to_vocabulary:
            raise KeyError(f"gt_object {gt_object!r} is missing from mapping file: {mapping_file}")

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
        records.append(record)

        if options["enabled"] and object_mask_path:
            candidates.append(
                {
                    "sample_index": idx,
                    "record_index": len(records) - 1,
                    "scene": scene,
                    "gt_object": gt_object,
                    "object_image_path": clean_image_path,
                    "object_mask_path": object_mask_path,
                    "image_shape": (height, width),
                }
            )

        if (idx + 1) % log_period == 0 or idx + 1 == len(samples):
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Built VIGOR clean records: %d / %d samples (%.1f samples/s)",
                idx + 1,
                len(samples),
                (idx + 1) / max(elapsed, 1e-6),
            )

    candidates_by_scene = {}
    for candidate in candidates:
        candidates_by_scene.setdefault(candidate["scene"], []).append(candidate)

    if options["enabled"]:
        extra_ratio = max(float(options["noisy_ratio"]), 0.0)
        target_noisy = int(round(len(records) * extra_ratio))
    else:
        target_noisy = 0

    eligible_targets = [
        candidate
        for candidate in candidates
        if len(candidates_by_scene.get(candidate["scene"], [])) > 1
    ]
    rng = random.Random(int(options["random_seed"]))
    rng.shuffle(eligible_targets)

    clean_count = len(records)
    noisy_records = []
    skipped_without_partner = 0
    skipped_too_far = 0
    skipped_missing_target_mask = 0
    for target in eligible_targets:
        if len(noisy_records) >= target_noisy:
            break

        partner, center_distance, skip_reason = _select_nearby_partner(
            target,
            candidates_by_scene,
            rng,
            int(options["nearby_topk"]),
            float(options["max_center_distance"]),
        )
        if partner is None:
            if skip_reason == "too_far":
                skipped_too_far += 1
            elif skip_reason == "missing_target_mask":
                skipped_missing_target_mask += 1
            else:
                skipped_without_partner += 1
            continue

        noisy_record = copy.deepcopy(records[target["record_index"]])
        noisy_record.update(
            {
                "image_id": clean_count + len(noisy_records),
                "vigor_input_type": "gt_nearby_object_mix",
                "vigor_noisy_source_index": target["sample_index"],
                "vigor_secondary_sample_index": partner["sample_index"],
                "vigor_secondary_gt_object": partner["gt_object"],
                "vigor_secondary_object_mask_path": partner["object_mask_path"],
                "vigor_secondary_object_image_path": partner["object_image_path"],
                "vigor_secondary_bbox": partner["bbox"],
                "vigor_secondary_center": partner["center"],
                "vigor_secondary_center_distance": center_distance,
                "vigor_nearby_topk": int(options["nearby_topk"]),
                "vigor_max_center_distance": float(options["max_center_distance"]),
            }
        )
        noisy_records.append(noisy_record)

    records.extend(noisy_records)

    if target_noisy and len(noisy_records) < target_noisy:
        logger.warning(
            "Only %d usable GT nearby-noisy records are available for requested noisy count %d",
            len(noisy_records),
            target_noisy,
        )

    logger.info(
        "VIGOR GT+nearby noisy dataset %s: clean=%d, requested_noisy=%d, added_noisy=%d, total=%d, "
        "eligible_targets=%d, skipped_without_partner=%d, skipped_too_far=%d, skipped_missing_target_mask=%d, nearby_topk=%d, max_center_distance=%.1f",
        dataset_name or "<unnamed>",
        clean_count,
        target_noisy,
        len(noisy_records),
        len(records),
        len(eligible_targets),
        skipped_without_partner,
        skipped_too_far,
        skipped_missing_target_mask,
        int(options["nearby_topk"]),
        float(options["max_center_distance"]),
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
