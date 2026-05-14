# Copyright (c) Facebook, Inc. and its affiliates.
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path

import cv2
import fcntl
import numpy as np
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


DEFAULT_VIGOR_ROOT = "/opt/data/private/LLMSeg/dataset/VIGOR-100K_new"
DEFAULT_MAPPING_PATH = "configs/vigor/vigor_easy_object_to_vocabulary.json"
DEFAULT_CACHE_VERSION = "vigor_gt_object_prior_cache_v2_scene_first"
DEFAULT_VOCABULARY = [
    "cylindrical side surface",
    "hexagonal side face",
    "flat side surface",
    "whole object",
]

logger = logging.getLogger("detectron2.vigor_gt_object_prior")


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _resolve_path(root, path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


def _split_paths(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _split_objects(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


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
    json_files = json_file if isinstance(json_file, (list, tuple)) else [json_file]
    stat_parts = []
    for path in [*json_files, mapping_file]:
        path = Path(path)
        stat = path.stat()
        stat_parts.append(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}")
    key = "|".join(
        [
            DEFAULT_CACHE_VERSION,
            dataset_name or "vigor_gt_object_prior",
            str(Path(data_root).resolve()),
            *stat_parts,
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{dataset_name or 'vigor_gt_object_prior'}_{digest}.pkl"


def _load_cache(cache_file):
    if not cache_file.exists():
        return None
    with cache_file.open("rb") as f:
        dataset_dicts = pickle.load(f)
    logger.info("Loaded VIGOR GT object-prior cache: %s (%d samples)", cache_file, len(dataset_dicts))
    return dataset_dicts


def _write_cache(cache_file, dataset_dicts):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
    with tmp_file.open("wb") as f:
        pickle.dump(dataset_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_file, cache_file)
    logger.info("Wrote VIGOR GT object-prior cache: %s (%d samples)", cache_file, len(dataset_dicts))


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
    bbox = mask_util.toBbox(rle).tolist()
    return rle, bbox, int(mask.shape[0]), int(mask.shape[1])


def load_vigor_gt_object_prior_json(json_file, data_root, mapping_file, dataset_name=None):
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

        dataset_dicts = _build_vigor_gt_object_prior_json(
            json_file, data_root, mapping_file, dataset_name
        )
        _write_cache(cache_file, dataset_dicts)
        return dataset_dicts


def load_vigor_gt_object_prior_jsons(json_files, data_root, mapping_file, dataset_name=None):
    json_files = list(json_files)
    cache_file = _cache_path(dataset_name, json_files, data_root, mapping_file)
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

        dataset_dicts = _build_vigor_gt_object_prior_jsons(
            json_files, data_root, mapping_file, dataset_name
        )
        _write_cache(cache_file, dataset_dicts)
        return dataset_dicts


def _infer_split_name(json_file):
    name = Path(json_file).name.lower()
    if "easy" in name:
        return "easy"
    if "hard" in name:
        return "hard"
    return Path(json_file).stem


def _build_vigor_gt_object_prior_jsons(json_files, data_root, mapping_file, dataset_name=None):
    dataset_dicts = []
    next_image_id = 0
    for json_file in json_files:
        partial = _build_vigor_gt_object_prior_json(
            json_file,
            data_root,
            mapping_file,
            dataset_name=dataset_name,
            image_id_start=next_image_id,
            split_name=_infer_split_name(json_file),
        )
        dataset_dicts.extend(partial)
        next_image_id += len(partial)
    logger.info(
        "Built combined VIGOR GT object-prior dataset %s from %d JSON files: %d instruction samples",
        dataset_name or "<unnamed>",
        len(json_files),
        len(dataset_dicts),
    )
    return dataset_dicts


def _build_vigor_gt_object_prior_json(
    json_file,
    data_root,
    mapping_file,
    dataset_name=None,
    image_id_start=0,
    split_name=None,
):
    vocabulary, object_to_vocabulary, vocab_to_id = _load_mapping(mapping_file)
    samples = _load_samples(json_file)
    dataset_dicts = []
    log_period = max(1, int(os.getenv("VIGOR_LOAD_LOG_PERIOD", "1000")))
    start_time = time.perf_counter()
    logger.info(
        "Building VIGOR GT object-prior dataset %s from %s (%d JSON samples)",
        dataset_name or "<unnamed>",
        json_file,
        len(samples),
    )

    image_id = image_id_start
    split_name = split_name or _infer_split_name(json_file)
    for idx, sample in enumerate(samples):
        gt_object = sample.get("gt_object", sample.get("object", ""))
        gt_objects = _split_objects(gt_object)
        if not gt_objects:
            raise ValueError(f"Empty gt_object in sample {idx} from {json_file}")
        scene = str(sample.get("scene", "")).strip()
        if not scene:
            raise ValueError(f"Missing scene id in sample {idx} from {json_file}")
        mask_paths = _split_paths(sample.get("gt_mask_path", ""))
        object_mask_paths = _split_paths(sample.get("gt_object_mask_path", ""))
        if not mask_paths or not object_mask_paths:
            raise ValueError(
                "VIGOR GT object-prior training expects non-empty affordance mask and object "
                f"mask path lists. Got mask_paths={mask_paths}, object_mask_paths={object_mask_paths}"
            )
        gt_object_first = gt_objects[0]
        if gt_object_first not in object_to_vocabulary:
            raise KeyError(
                f"gt_object component '{gt_object_first}' is missing from mapping file: {mapping_file}"
            )

        image_path = _resolve_path(data_root, f"{scene}.png")
        mask_path = _resolve_path(data_root, mask_paths[0])
        object_mask_path = _resolve_path(data_root, object_mask_paths[0])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(object_mask_path):
            raise FileNotFoundError(f"GT object mask not found: {object_mask_path}")

        segmentation, bbox, height, width = _mask_to_annotation(mask_path)
        vocabulary_name = object_to_vocabulary[gt_object_first]
        category_id = vocab_to_id[vocabulary_name]
        annotation = {
            "iscrowd": 0,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": category_id,
            "segmentation": segmentation,
        }
        instructions = sample.get("instructions", [])
        if not instructions:
            instructions = [""]

        for instruction_index, instruction in enumerate(instructions):
            dataset_dicts.append(
                {
                    "file_name": image_path,
                    "image_id": image_id,
                    "height": height,
                    "width": width,
                    "vigor_scene": scene,
                    "vigor_gt_object": gt_object,
                    "instruction": instruction,
                    "instruction_index": instruction_index,
                    "source_sample_index": idx,
                    "source_json": str(json_file),
                    "vigor_split": split_name,
                    "vigor_gt_object_first": gt_object_first,
                    "gt_object_mask_path": object_mask_path,
                    "gt_object_mask_paths": [object_mask_path],
                    "object_prior_mask_path": object_mask_path,
                    "object_prior_mask_paths": [object_mask_path],
                    "annotations": [annotation],
                }
            )
            image_id += 1

        if (idx + 1) % log_period == 0 or idx + 1 == len(samples):
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Built VIGOR GT object-prior dataset %s: %d / %d JSON samples, %d instructions "
                "(%.1f instructions/s)",
                dataset_name or "<unnamed>",
                idx + 1,
                len(samples),
                len(dataset_dicts),
                len(dataset_dicts) / max(elapsed, 1e-6),
            )

    return dataset_dicts


def register_vigor_gt_object_prior_instances(name, json_file, data_root, mapping_file):
    DatasetCatalog.register(
        name,
        lambda: load_vigor_gt_object_prior_json(json_file, data_root, mapping_file, name),
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


def register_vigor_gt_object_prior_multi_json_instances(name, json_files, data_root, mapping_file):
    json_files = tuple(json_files)
    DatasetCatalog.register(
        name,
        lambda: load_vigor_gt_object_prior_jsons(json_files, data_root, mapping_file, name),
    )
    vocabulary, _, _ = _load_mapping(mapping_file)
    MetadataCatalog.get(name).set(
        json_file=list(json_files),
        image_root=data_root,
        mapping_file=mapping_file,
        evaluator_type="coco",
        thing_classes=vocabulary,
        thing_dataset_id_to_contiguous_id={idx: idx for idx in range(len(vocabulary))},
    )


def register_all_vigor_gt_object_prior():
    repo_root = _repo_root()
    mapping_file = str(repo_root / DEFAULT_MAPPING_PATH)
    root = os.getenv("VIGOR_DATASETS", DEFAULT_VIGOR_ROOT)

    register_vigor_gt_object_prior_instances(
        "vigor_easy_train_gt_object_prior",
        os.path.join(root, "train", "open_vocab_grasp_easy_object_mix.json"),
        os.path.join(root, "train"),
        mapping_file,
    )
    register_vigor_gt_object_prior_instances(
        "vigor_hard_train_gt_object_prior",
        os.path.join(root, "train", "open_vocab_grasp_hard_object_mix.json"),
        os.path.join(root, "train"),
        mapping_file,
    )
    register_vigor_gt_object_prior_multi_json_instances(
        "vigor_easy_hard_train_gt_object_prior",
        [
            os.path.join(root, "train", "open_vocab_grasp_easy_object_mix.json"),
            os.path.join(root, "train", "open_vocab_grasp_hard_object_mix.json"),
        ],
        os.path.join(root, "train"),
        mapping_file,
    )
    register_vigor_gt_object_prior_instances(
        "vigor_easy_test_gt_object_prior",
        os.path.join(root, "test", "open_vocab_grasp_easy_object_mix.json"),
        os.path.join(root, "test"),
        mapping_file,
    )
    register_vigor_gt_object_prior_instances(
        "vigor_hard_test_gt_object_prior",
        os.path.join(root, "test", "open_vocab_grasp_hard_object_mix.json"),
        os.path.join(root, "test"),
        mapping_file,
    )


register_all_vigor_gt_object_prior()
