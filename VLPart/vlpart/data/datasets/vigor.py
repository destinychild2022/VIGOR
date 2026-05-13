# Copyright (c) Facebook, Inc. and its affiliates.
import json
import hashlib
import logging
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import fcntl
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


DEFAULT_VIGOR_ROOT = "/opt/data/private/LLMSeg/dataset/VIGOR-100K_new"
DEFAULT_MAPPING_PATH = "configs/vigor/vigor_easy_object_to_vocabulary.json"
DEFAULT_CACHE_VERSION = "vigor_dataset_cache_v1"
DEFAULT_VOCABULARY = [
    "cylindrical side surface",
    "hexagonal side face",
    "flat side surface",
    "whole object",
]

logger = logging.getLogger("detectron2.vigor")


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _resolve_path(root, path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


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


def _cache_path(dataset_name, json_file, data_root, mapping_file):
    cache_dir = Path(os.getenv("VIGOR_CACHE_DIR", _repo_root() / "datasets" / "cache" / "vigor"))
    stat_parts = []
    for path in [json_file, mapping_file]:
        path = Path(path)
        stat = path.stat()
        stat_parts.append(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}")
    key = "|".join([DEFAULT_CACHE_VERSION, dataset_name or "vigor", str(Path(data_root).resolve()), *stat_parts])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{dataset_name or 'vigor'}_{digest}.pkl"


def _load_cache(cache_file):
    if not cache_file.exists():
        return None
    with cache_file.open("rb") as f:
        dataset_dicts = pickle.load(f)
    logger.info("Loaded VIGOR dataset cache: %s (%d samples)", cache_file, len(dataset_dicts))
    return dataset_dicts


def _write_cache(cache_file, dataset_dicts):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
    with tmp_file.open("wb") as f:
        pickle.dump(dataset_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_file, cache_file)
    logger.info("Wrote VIGOR dataset cache: %s (%d samples)", cache_file, len(dataset_dicts))


def _mask_to_annotation(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read VIGOR mask: {mask_path}")

    # VIGOR masks use 0 for foreground and non-zero for background.
    fg = np.asfortranarray((mask == 0).astype(np.uint8))
    if fg.sum() == 0:
        raise ValueError(f"Empty foreground mask: {mask_path}")

    rle = mask_util.encode(fg)
    rle["counts"] = rle["counts"].decode("ascii")
    bbox = mask_util.toBbox(rle).tolist()  # XYWH_ABS
    return rle, bbox, int(mask.shape[0]), int(mask.shape[1])


def load_vigor_json(json_file, data_root, mapping_file, dataset_name=None):
    cache_file = _cache_path(dataset_name, json_file, data_root, mapping_file)
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

        dataset_dicts = _build_vigor_json(json_file, data_root, mapping_file, dataset_name)
        _write_cache(cache_file, dataset_dicts)
        return dataset_dicts


def _build_vigor_json(json_file, data_root, mapping_file, dataset_name=None):
    vocabulary, object_to_vocabulary, vocab_to_id = _load_mapping(mapping_file)
    samples = _load_samples(json_file)
    dataset_dicts = []
    log_period = max(1, int(os.getenv("VIGOR_LOAD_LOG_PERIOD", "1000")))
    start_time = time.perf_counter()
    logger.info(
        "Building VIGOR dataset %s from %s (%d samples)",
        dataset_name or "<unnamed>",
        json_file,
        len(samples),
    )

    for idx, sample in enumerate(samples):
        gt_object = sample.get("gt_object", sample.get("object", ""))
        if gt_object not in object_to_vocabulary:
            raise KeyError(
                f"gt_object '{gt_object}' is missing from mapping file: {mapping_file}"
            )

        image_paths = [p.strip() for p in sample.get("gt_object_path", "").split(",") if p.strip()]
        mask_paths = [p.strip() for p in sample.get("gt_mask_path", "").split(",") if p.strip()]
        if len(image_paths) != 1 or len(mask_paths) != 1:
            raise ValueError(
                "vigor_easy_train expects one image and one mask per sample. "
                f"Got image_paths={image_paths}, mask_paths={mask_paths}"
            )

        image_path = _resolve_path(data_root, image_paths[0])
        mask_path = _resolve_path(data_root, mask_paths[0])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        segmentation, bbox, height, width = _mask_to_annotation(mask_path)
        vocabulary_name = object_to_vocabulary[gt_object]
        category_id = vocab_to_id[vocabulary_name]

        dataset_dicts.append(
            {
                "file_name": image_path,
                "image_id": idx,
                "height": height,
                "width": width,
                "vigor_gt_object": gt_object,
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
        )
        if (idx + 1) % log_period == 0 or idx + 1 == len(samples):
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Built VIGOR dataset %s: %d / %d samples (%.1f samples/s)",
                dataset_name or "<unnamed>",
                idx + 1,
                len(samples),
                (idx + 1) / max(elapsed, 1e-6),
            )

    return dataset_dicts


def register_vigor_instances(name, json_file, data_root, mapping_file):
    DatasetCatalog.register(
        name,
        lambda: load_vigor_json(json_file, data_root, mapping_file, name),
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


def register_all_vigor():
    repo_root = _repo_root()
    mapping_file = str(repo_root / DEFAULT_MAPPING_PATH)
    root = os.getenv("VIGOR_DATASETS", DEFAULT_VIGOR_ROOT)

    register_vigor_instances(
        "vigor_easy_train",
        os.path.join(root, "train", "open_vocab_grasp_easy_object_mix.json"),
        os.path.join(root, "train"),
        mapping_file,
    )
    register_vigor_instances(
        "vigor_easy_test",
        os.path.join(root, "test", "open_vocab_grasp_easy_object_mix.json"),
        os.path.join(root, "test"),
        mapping_file,
    )


register_all_vigor()
