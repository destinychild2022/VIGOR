import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import swanlab
    SWANLAB_AVAILABLE = True
    SWANLAB_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    swanlab = None
    SWANLAB_AVAILABLE = False
    SWANLAB_IMPORT_ERROR = exc

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
    compute_iou_from_foreground,
    load_gt_masks,
    normalize_custom_vocabulary,
    resolve_repo_path,
    save_prediction_overlay,
    save_raw_pred_mask,
    selected_mask_to_vigor_mask,
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


FEATURE_NAMES = [
    "pred_similarity",
    "pred_iou",
    "vlpart_score",
    "containment_score",
]

CANDIDATE_FEATURE_CACHE_VERSION = 1
CANDIDATE_FEATURE_SHARD_CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train/test a VLPart top-K affordance reranker")
    parser.add_argument("--mode", choices=["train", "test", "extract_train_cache"], required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--easy_json_file", default="open_vocab_grasp_easy_object_mix.json")
    parser.add_argument("--hard_json_file", default="open_vocab_grasp_hard_object_mix.json")
    parser.add_argument("--split", default="both", choices=["easy", "hard", "both"])
    parser.add_argument("--llmseg_topk_masks_dir", required=True)
    parser.add_argument("--topk_mask_k", type=int, default=5)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", default="vlpart_topk_reranker/results")
    parser.add_argument("--vis_dir", default="vlpart_topk_reranker/visualizations")
    parser.add_argument("--pred_masks_dir", default="vlpart_topk_reranker/pred_masks")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_pred_masks", action="store_true")

    parser.add_argument("--vocabulary", default="custom")
    parser.add_argument(
        "--custom_vocabulary",
        default="cylindrical side surface,hexagonal side face,flat side surface,whole object",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--mask_selection", default="top1", choices=["top1", "union"])
    parser.add_argument("--ssr_threshold", type=float, default=0.5)
    parser.add_argument("--icr_threshold", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction_cache_size", type=int, default=128)
    parser.add_argument("--candidate_feature_cache_dir", default="")
    parser.add_argument("--skip_icr", action="store_true")
    parser.add_argument("--extract_shard_index", type=int, default=0)
    parser.add_argument("--extract_num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--target_temperature", type=float, default=0.10)
    parser.add_argument("--ranking_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab_enabled", action="store_true")
    parser.add_argument("--swanlab_project", default="VLPart")
    parser.add_argument("--swanlab_exp_name", default="vigor_topk_reranker")

    parser.add_argument(
        "--opts",
        help="Additional VLPart config options in KEY VALUE pairs.",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


class RerankerMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_llmseg_feature_from_filename(mask_path: str) -> Tuple[float, float, bool]:
    name = Path(mask_path).name
    score_match = re.search(r"_score(-?\d+(?:\.\d+)?)", name)
    pred_similarity = float(score_match.group(1)) if score_match else 0.0

    # Do not use the existing "_iou..." suffix: in current saved top-K masks it
    # is GT object IoU for recall reporting, not LLMSeg's predicted IoU.
    pred_iou_match = re.search(r"_(?:prediou|pred_iou)(-?\d+(?:\.\d+)?)", name)
    pred_iou_available = pred_iou_match is not None
    pred_iou = float(pred_iou_match.group(1)) if pred_iou_available else 0.0
    return pred_similarity, pred_iou, pred_iou_available


def instances_to_candidates(predictions: Dict, metadata, image_shape: Tuple[int, int]):
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
        class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else str(class_idx)
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


def containment_score(affordance_fg: np.ndarray, object_fg: np.ndarray) -> float:
    if affordance_fg.shape != object_fg.shape:
        h, w = object_fg.shape
        affordance_fg = cv2.resize(
            affordance_fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    area = int(affordance_fg.sum())
    if area == 0:
        return 0.0
    return float(np.logical_and(affordance_fg, object_fg).sum()) / float(area)


def best_quality(pred_fg: np.ndarray, gt_masks: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    best_iou = 0.0
    best_gt = gt_masks[0] if gt_masks else None
    for gt_mask in gt_masks:
        iou = compute_iou_from_foreground(pred_fg, gt_mask)
        if iou >= best_iou:
            best_iou = iou
            best_gt = gt_mask
    return best_iou, best_gt



def stable_json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def path_fingerprint(path: str) -> Dict:
    try:
        stat = os.stat(path)
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def json_path_for_split(args: argparse.Namespace, split_name: str) -> str:
    json_name = args.easy_json_file if split_name == "easy" else args.hard_json_file
    return str(Path(args.data_dir) / json_name)


def candidate_feature_cache_signature(
    sample: Dict,
    args: argparse.Namespace,
    split_name: str,
    topk_mask_paths: List[Tuple[int, str]],
) -> Tuple[str, Dict]:
    payload = {
        "version": CANDIDATE_FEATURE_CACHE_VERSION,
        "feature_names": FEATURE_NAMES,
        "split": split_name,
        "json_file": path_fingerprint(json_path_for_split(args, split_name)),
        "context": {
            "config_file": path_fingerprint(args.config_file),
            "weights": path_fingerprint(args.weights),
            "data_dir": args.data_dir,
            "llmseg_topk_masks_dir": args.llmseg_topk_masks_dir,
            "topk_mask_k": int(args.topk_mask_k),
            "vocabulary": args.vocabulary,
            "custom_vocabulary": args.custom_vocabulary,
            "confidence_threshold": float(args.confidence_threshold),
        },
        "sample": {
            "sample_index": sample.get("sample_index", -1),
            "instruction_index": sample.get("instruction_index", 0),
            "instruction": sample.get("instruction", ""),
            "scene": sample.get("scene", ""),
            "difficulty": sample.get("difficulty", ""),
            "gt_object": sample.get("gt_object", ""),
            "object": sample.get("object", ""),
            "obj_count": sample.get("obj_count", 1),
            "img_name": sample.get("img_name", ""),
            "image_path": path_fingerprint(sample.get("image_path", "")),
            "gt_mask_paths": [
                path_fingerprint(path) for path in sample.get("gt_mask_paths", [])
            ],
        },
        "topk_mask_paths": [
            {"rank": int(rank), "mask": path_fingerprint(mask_path)}
            for rank, mask_path in topk_mask_paths
        ],
    }
    digest = hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return digest, payload


def safe_cache_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def candidate_feature_cache_path(
    args: argparse.Namespace,
    split_name: str,
    sample: Dict,
    digest: str,
) -> Path:
    sample_index = safe_cache_int(sample.get("sample_index", -1))
    instruction_index = safe_cache_int(sample.get("instruction_index", 0), 0)
    name = f"{sample_index:08d}_instr{instruction_index}_{digest[:16]}.pt"
    return Path(args.candidate_feature_cache_dir) / split_name / name


def numeric_candidate_row(row: Dict) -> Dict:
    keys = [
        "features",
        "target",
        "llmseg_topk_rank",
        "llmseg_topk_mask_path",
        "pred_iou_available",
        "pred_similarity",
        "pred_iou",
        "vlpart_score",
        "containment_score",
        "pred_class",
        "pred_class_name",
        "vlpart_instance_index",
    ]
    cached = {}
    for key in keys:
        value = row.get(key)
        if isinstance(value, np.generic):
            value = value.item()
        if key == "features" and value is not None:
            value = [float(item) for item in value]
        elif isinstance(value, float):
            value = float(value)
        elif isinstance(value, int):
            value = int(value)
        cached[key] = value
    return cached


def save_candidate_feature_cache(
    cache_path: Path,
    digest: str,
    signature_payload: Dict,
    group: Dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CANDIDATE_FEATURE_CACHE_VERSION,
        "feature_names": FEATURE_NAMES,
        "signature_digest": digest,
        "signature": signature_payload,
        "features": group["features"].astype(np.float32, copy=False),
        "targets": group["targets"].astype(np.float32, copy=False),
        "candidates": [numeric_candidate_row(row) for row in group["candidates"]],
        "pred_iou_missing": int(group.get("pred_iou_missing", 0)),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def load_candidate_feature_cache(cache_path: Path, digest: str, sample: Dict) -> Optional[Dict]:
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"  [Warning] Failed to read feature cache {cache_path}: {exc}")
        return None
    if payload.get("version") != CANDIDATE_FEATURE_CACHE_VERSION:
        return None
    if payload.get("feature_names") != FEATURE_NAMES:
        return None
    if payload.get("signature_digest") != digest:
        return None
    return {
        "sample": sample,
        "features": np.asarray(payload["features"], dtype=np.float32),
        "targets": np.asarray(payload["targets"], dtype=np.float32),
        "candidates": payload["candidates"],
        "image_bgr": None,
        "pred_iou_missing": int(payload.get("pred_iou_missing", 0)),
        "candidate_feature_cache_hit": True,
    }



def numeric_group_for_cache(group: Dict, sample_digest: str, split_name: str) -> Dict:
    return {
        "sample_digest": sample_digest,
        "split": split_name,
        "sample": group.get("sample", {}),
        "features": group["features"].astype(np.float32, copy=False),
        "targets": group["targets"].astype(np.float32, copy=False),
        "candidates": [numeric_candidate_row(row) for row in group["candidates"]],
        "pred_iou_missing": int(group.get("pred_iou_missing", 0)),
    }


def group_from_numeric_cache(entry: Dict) -> Dict:
    return {
        "sample": entry.get("sample", {}),
        "features": np.asarray(entry["features"], dtype=np.float32),
        "targets": np.asarray(entry["targets"], dtype=np.float32),
        "candidates": entry["candidates"],
        "image_bgr": None,
        "pred_iou_missing": int(entry.get("pred_iou_missing", 0)),
        "candidate_feature_cache_hit": True,
    }


def candidate_feature_shard_cache_path(
    args: argparse.Namespace,
    split_name: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        Path(args.candidate_feature_cache_dir)
        / "shards"
        / f"{split_name}_shard{shard_index:03d}_of{num_shards:03d}.pt"
    )


def sample_identity_for_shard_cache(sample: Dict) -> Dict:
    return {
        "sample_index": sample.get("sample_index", -1),
        "instruction_index": sample.get("instruction_index", 0),
        "instruction": sample.get("instruction", ""),
        "scene": sample.get("scene", ""),
        "difficulty": sample.get("difficulty", ""),
        "gt_object": sample.get("gt_object", ""),
        "object": sample.get("object", ""),
        "obj_count": sample.get("obj_count", 1),
        "img_name": sample.get("img_name", ""),
        "image_path": sample.get("image_path", ""),
        "gt_mask_paths": list(sample.get("gt_mask_paths", [])),
    }


def build_shard_signature(
    args: argparse.Namespace,
    split_name: str,
    samples: List[Dict],
) -> Tuple[str, List[Tuple[Dict, str]]]:
    entries = []
    sample_digests = []
    for sample in samples:
        sample_identity = sample_identity_for_shard_cache(sample)
        digest = hashlib.sha256(stable_json_dumps(sample_identity).encode("utf-8")).hexdigest()
        entries.append((sample, digest))
        sample_digests.append(digest)
    payload = {
        "version": CANDIDATE_FEATURE_SHARD_CACHE_VERSION,
        "format": "vlpart_topk_reranker_numeric_shard",
        "feature_names": FEATURE_NAMES,
        "split": split_name,
        "json_file": path_fingerprint(json_path_for_split(args, split_name)),
        "context": {
            "config_file": path_fingerprint(args.config_file),
            "weights": path_fingerprint(args.weights),
            "data_dir": args.data_dir,
            "llmseg_topk_masks_dir": path_fingerprint(args.llmseg_topk_masks_dir),
            "topk_mask_k": int(args.topk_mask_k),
            "vocabulary": args.vocabulary,
            "custom_vocabulary": args.custom_vocabulary,
            "confidence_threshold": float(args.confidence_threshold),
            "max_samples": args.max_samples,
        },
        "num_samples": len(samples),
        "sample_digests": sample_digests,
    }
    shard_digest = hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return shard_digest, entries


def load_candidate_feature_shard_cache(
    shard_path: Path,
    expected_digest: str,
    expected_count: int,
) -> Optional[List[Dict]]:
    if not shard_path.exists():
        return None
    try:
        payload = torch.load(shard_path, map_location="cpu")
    except Exception as exc:
        print(f"  [Warning] Failed to read shard cache {shard_path}: {exc}")
        return None
    if payload.get("version") != CANDIDATE_FEATURE_SHARD_CACHE_VERSION:
        return None
    if payload.get("feature_names") != FEATURE_NAMES:
        return None
    if payload.get("shard_digest") != expected_digest:
        return None
    groups = payload.get("groups", [])
    if len(groups) != expected_count:
        return None
    return [group_from_numeric_cache(entry) for entry in groups]


def save_candidate_feature_shard_cache(
    shard_path: Path,
    shard_digest: str,
    split_name: str,
    shard_index: int,
    num_shards: int,
    groups: List[Dict],
    stats: Dict,
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CANDIDATE_FEATURE_SHARD_CACHE_VERSION,
        "format": "vlpart_topk_reranker_numeric_shard",
        "feature_names": FEATURE_NAMES,
        "shard_digest": shard_digest,
        "split": split_name,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": dict(stats),
        "groups": groups,
    }
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, shard_path)


def extract_or_load_feature_shard(
    args: argparse.Namespace,
    split_name: str,
    samples: List[Dict],
    shard_index: int,
    num_shards: int,
    device: torch.device,
    allow_extract: bool,
) -> Tuple[List[Dict], Dict]:
    shard_samples_for_split = shard_samples(samples, shard_index, num_shards)
    stats = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "groups": 0,
        "skipped": 0,
    }
    if not shard_samples_for_split:
        return [], stats

    print(
        f"  [{split_name} shard {shard_index + 1}/{num_shards}] "
        f"prepare signature for {len(shard_samples_for_split)} samples",
        flush=True,
    )
    shard_digest, entries = build_shard_signature(args, split_name, shard_samples_for_split)
    shard_path = candidate_feature_shard_cache_path(args, split_name, shard_index, num_shards)
    cached_groups = load_candidate_feature_shard_cache(
        shard_path, shard_digest, expected_count=len(entries)
    )
    if cached_groups is not None:
        stats["hits"] = len(cached_groups)
        stats["groups"] = len(cached_groups)
        print(f"  [{split_name} shard {shard_index + 1}/{num_shards}] cache hit: {shard_path}", flush=True)
        return cached_groups, stats

    stats["misses"] = len(entries)
    if not allow_extract:
        raise FileNotFoundError(
            f"Shard cache miss for {split_name} shard {shard_index + 1}/{num_shards}: {shard_path}"
        )

    pred_cache = PredictionCache(args.prediction_cache_size)
    demo_ref = {"demo": None}
    cached_entries = []
    skipped = []
    desc = f"Cache {split_name} shard {shard_index + 1}/{num_shards}"
    for sample, sample_digest in tqdm(entries, desc=desc):
        try:
            group = extract_candidate_group(
                sample,
                get_vlpart_demo(demo_ref, args),
                args,
                pred_cache,
                keep_masks=False,
            )
            cached_entries.append(numeric_group_for_cache(group, sample_digest, split_name))
        except Exception as exc:
            skipped.append((sample.get("image_path", "unknown"), repr(exc)))
            if len(skipped) <= 5 or args.debug:
                print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    stats["skipped"] = len(skipped)
    stats["groups"] = len(cached_entries)
    if skipped:
        raise RuntimeError(f"Feature shard extraction skipped {len(skipped)} samples")
    stats["writes"] = len(cached_entries)
    save_candidate_feature_shard_cache(
        shard_path,
        shard_digest,
        split_name,
        shard_index,
        num_shards,
        cached_entries,
        stats,
    )
    print(f"  [{split_name} shard {shard_index + 1}/{num_shards}] wrote shard cache: {shard_path}", flush=True)
    return [], stats


def load_all_feature_shards(args: argparse.Namespace, samples_by_split: Dict[str, List[Dict]]) -> Tuple[List[Dict], Dict]:
    groups = []
    total_stats = {"hits": 0, "misses": 0, "writes": 0, "groups": 0, "skipped": 0}
    num_shards = max(1, int(args.extract_num_shards))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for split_name, samples in samples_by_split.items():
        if not samples:
            continue
        for shard_index in range(num_shards):
            shard_groups, stats = extract_or_load_feature_shard(
                args,
                split_name,
                samples,
                shard_index,
                num_shards,
                device,
                allow_extract=False,
            )
            groups.extend(shard_groups)
            for key in total_stats:
                total_stats[key] += int(stats.get(key, 0))
    return groups, total_stats

def get_vlpart_demo(demo_ref: Dict, args: argparse.Namespace):
    if demo_ref.get("demo") is None:
        demo_ref["demo"] = build_vlpart(args)
    return demo_ref["demo"]


def load_or_extract_candidate_group(
    sample: Dict,
    demo_ref: Dict,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
    keep_masks: bool,
    split_name: str,
    cache_stats: Optional[Dict[str, int]] = None,
) -> Dict:
    cache_dir = getattr(args, "candidate_feature_cache_dir", "")
    if cache_dir and not keep_masks:
        topk_mask_paths = resolve_topk_mask_paths(sample, args)
        if not topk_mask_paths:
            raise FileNotFoundError(
                f"No LLMSeg top-K masks found for {sample['img_name']} "
                f"{sample.get('gt_object', '')} obj{sample.get('obj_count', 1)} "
                f"instr{sample.get('instruction_index', 0)}"
            )
        digest, signature_payload = candidate_feature_cache_signature(
            sample, args, split_name, topk_mask_paths
        )
        cache_path = candidate_feature_cache_path(args, split_name, sample, digest)
        cached_group = load_candidate_feature_cache(cache_path, digest, sample)
        if cached_group is not None:
            if cache_stats is not None:
                cache_stats["hits"] = cache_stats.get("hits", 0) + 1
            return cached_group
        if cache_stats is not None:
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
        group = extract_candidate_group(
            sample, get_vlpart_demo(demo_ref, args), args, pred_cache, keep_masks=False
        )
        save_candidate_feature_cache(cache_path, digest, signature_payload, group)
        if cache_stats is not None:
            cache_stats["writes"] = cache_stats.get("writes", 0) + 1
        return group

    return extract_candidate_group(
        sample, get_vlpart_demo(demo_ref, args), args, pred_cache, keep_masks=keep_masks
    )

def extract_candidate_group(
    sample: Dict,
    demo: VisualizationDemo,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
    keep_masks: bool,
) -> Dict:
    image_path = sample["image_path"]
    image_bgr = read_image(image_path, format="BGR")
    gt_masks = load_gt_masks(sample["gt_mask_paths"], image_bgr.shape[:2])
    topk_mask_paths = resolve_topk_mask_paths(sample, args)
    if not topk_mask_paths:
        raise FileNotFoundError(
            f"No LLMSeg top-K masks found for {sample['img_name']} "
            f"{sample.get('gt_object', '')} obj{sample.get('obj_count', 1)} "
            f"instr{sample.get('instruction_index', 0)}"
        )

    rows = []
    pred_iou_missing = 0
    for rank, topk_mask_path in topk_mask_paths:
        object_fg = read_llmseg_object_mask(topk_mask_path, image_bgr.shape[:2])
        pred_similarity, pred_iou, pred_iou_available = parse_llmseg_feature_from_filename(
            topk_mask_path
        )
        if not pred_iou_available:
            pred_iou_missing += 1

        cache_key = f"{image_path}|{topk_mask_path}|all_instances"
        cached = pred_cache.get(cache_key)
        if cached is None:
            masked_bgr = make_masked_rgb(image_bgr, object_fg)
            predictions = demo.predictor(masked_bgr)
            vlpart_candidates = instances_to_candidates(
                predictions, demo.metadata, image_bgr.shape[:2]
            )
            if not vlpart_candidates:
                empty = np.zeros(image_bgr.shape[:2], dtype=bool)
                vlpart_candidates = [
                    {
                        "pred_fg": empty,
                        "pred_score": 0.0,
                        "pred_class": -1,
                        "pred_class_name": "",
                        "vlpart_instance_index": -1,
                    }
                ]
            pred_cache.put(cache_key, vlpart_candidates)
        else:
            vlpart_candidates = cached

        for candidate in vlpart_candidates:
            pred_fg = candidate["pred_fg"]
            quality, best_gt = best_quality(pred_fg, gt_masks)
            contain = containment_score(pred_fg, object_fg)
            row = {
                "features": [
                    pred_similarity,
                    pred_iou,
                    candidate["pred_score"],
                    contain,
                ],
                "target": quality,
                "best_gt": best_gt if keep_masks else None,
                "llmseg_topk_rank": rank,
                "llmseg_topk_mask_path": topk_mask_path,
                "pred_iou_available": pred_iou_available,
                "pred_similarity": pred_similarity,
                "pred_iou": pred_iou,
                "vlpart_score": candidate["pred_score"],
                "containment_score": contain,
                "pred_class": candidate["pred_class"],
                "pred_class_name": candidate["pred_class_name"],
                "vlpart_instance_index": candidate["vlpart_instance_index"],
            }
            if keep_masks:
                row["pred_fg"] = pred_fg
            rows.append(row)

    features = np.asarray([row["features"] for row in rows], dtype=np.float32)
    targets = np.asarray([row["target"] for row in rows], dtype=np.float32)
    return {
        "sample": sample,
        "features": features,
        "targets": targets,
        "candidates": rows,
        "image_bgr": image_bgr if keep_masks else None,
        "pred_iou_missing": pred_iou_missing,
    }


def load_instruction_samples(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    samples = {"easy": [], "hard": []}
    if args.split in ["easy", "both"]:
        samples["easy"] = load_samples_per_instruction(args.data_dir, args.easy_json_file, "easy")
    if args.split in ["hard", "both"]:
        samples["hard"] = load_samples_per_instruction(args.data_dir, args.hard_json_file, "hard")
    if args.max_samples is not None:
        samples["easy"] = samples["easy"][: args.max_samples]
        samples["hard"] = samples["hard"][: args.max_samples]
    return samples


def normalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std.clamp_min(1e-6)


def reranker_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    target_dist = F.softmax(targets / max(args.target_temperature, 1e-6), dim=0)
    log_pred_dist = F.log_softmax(logits / max(args.ranking_temperature, 1e-6), dim=0)
    kl = F.kl_div(log_pred_dist, target_dist, reduction="batchmean")
    mse = F.mse_loss(torch.sigmoid(logits), targets)
    return args.kl_weight * kl + args.mse_weight * mse


def evaluate_reranker_groups(
    model: RerankerMLP,
    groups: List[Dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Dict:
    if not groups:
        return {"selected_quality": 0.0, "oracle_quality": 0.0, "reg_mse": 0.0}
    selected = []
    oracle = []
    mses = []
    model.eval()
    with torch.no_grad():
        for group in groups:
            x = torch.as_tensor(group["features"], dtype=torch.float32, device=device)
            y = torch.as_tensor(group["targets"], dtype=torch.float32, device=device)
            logits = model(normalize_features(x, mean, std))
            idx = int(torch.argmax(logits).item())
            selected.append(float(y[idx].item()))
            oracle.append(float(torch.max(y).item()))
            mses.append(float(F.mse_loss(torch.sigmoid(logits), y).item()))
    return {
        "selected_quality": float(np.mean(selected)),
        "oracle_quality": float(np.mean(oracle)),
        "reg_mse": float(np.mean(mses)),
    }


def build_vlpart(args: argparse.Namespace):
    cfg = setup_cfg(args)
    return VisualizationDemo(cfg, args)


def prepare_common_args(args: argparse.Namespace) -> None:
    args.config_file = resolve_repo_path(args.config_file)
    args.weights = resolve_repo_path(args.weights)
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.llmseg_topk_masks_dir = resolve_existing_or_repo_path(args.llmseg_topk_masks_dir)
    if getattr(args, "candidate_feature_cache_dir", ""):
        args.candidate_feature_cache_dir = str(Path(args.candidate_feature_cache_dir).expanduser())
    if args.vocabulary == "custom":
        args.custom_vocabulary = normalize_custom_vocabulary(args.custom_vocabulary)
    if args.topk_mask_k <= 0:
        raise ValueError("--topk_mask_k must be > 0")


def init_swanlab(args: argparse.Namespace):
    if not args.swanlab_enabled:
        return None
    if not SWANLAB_AVAILABLE or swanlab is None:
        print(f"  [Warning] SwanLab requested but unavailable: {SWANLAB_IMPORT_ERROR}")
        return None
    try:
        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_exp_name,
            config={
                "mode": args.mode,
                "data_dir": args.data_dir,
                "easy_json_file": args.easy_json_file,
                "hard_json_file": args.hard_json_file,
                "split": args.split,
                "llmseg_topk_masks_dir": args.llmseg_topk_masks_dir,
                "topk_mask_k": args.topk_mask_k,
                "vlpart_config": args.config_file,
                "vlpart_weights": args.weights,
                "feature_names": FEATURE_NAMES,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "kl_weight": args.kl_weight,
                "mse_weight": args.mse_weight,
                "target_temperature": args.target_temperature,
                "ranking_temperature": args.ranking_temperature,
                "seed": args.seed,
            },
        )
        print(
            f"  SwanLab enabled: project={args.swanlab_project}, "
            f"experiment={args.swanlab_exp_name}"
        )
        return run
    except Exception as exc:
        print(f"  [Warning] SwanLab init failed: {exc}")
        return None


def swanlab_log(run, metrics: Dict, step: int) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:
        print(f"  [Warning] SwanLab log failed at step {step}: {exc}")


def swanlab_finish(run) -> None:
    if run is None or not hasattr(run, "finish"):
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"  [Warning] SwanLab finish failed: {exc}")



def shard_samples(samples: List[Dict], shard_index: int, num_shards: int) -> List[Dict]:
    if num_shards <= 1:
        return samples
    return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_index]


def extract_feature_cache(args: argparse.Namespace) -> None:
    if not args.candidate_feature_cache_dir:
        raise ValueError("--candidate_feature_cache_dir is required for extract_train_cache")
    if args.extract_num_shards <= 0:
        raise ValueError("--extract_num_shards must be > 0")
    if not (0 <= args.extract_shard_index < args.extract_num_shards):
        raise ValueError("--extract_shard_index must be in [0, extract_num_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    samples_by_split = load_instruction_samples(args)
    total_stats = {"hits": 0, "misses": 0, "writes": 0, "groups": 0, "skipped": 0}

    print(
        f"Feature shard extraction "
        f"{args.extract_shard_index + 1}/{args.extract_num_shards} on {device}"
    )
    print(f"Feature shard cache dir: {args.candidate_feature_cache_dir}")

    for split_name, samples in samples_by_split.items():
        if not samples:
            continue
        _groups, stats = extract_or_load_feature_shard(
            args,
            split_name,
            samples,
            args.extract_shard_index,
            args.extract_num_shards,
            device,
            allow_extract=True,
        )
        for key in total_stats:
            total_stats[key] += int(stats.get(key, 0))

    print(
        "Feature shard extraction finished: "
        f"groups={total_stats['groups']}, skipped={total_stats['skipped']}, "
        f"hits={total_stats['hits']}, misses={total_stats['misses']}, "
        f"writes={total_stats['writes']}"
    )

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    samples_by_split = load_instruction_samples(args)
    skipped = []
    if args.candidate_feature_cache_dir:
        groups, cache_stats = load_all_feature_shards(args, samples_by_split)
        print(
            "  [train feature shard cache] "
            f"hits={cache_stats['hits']} misses={cache_stats['misses']} "
            f"writes={cache_stats['writes']} groups={cache_stats['groups']} "
            f"dir={args.candidate_feature_cache_dir}"
        )
    else:
        pred_cache = PredictionCache(args.prediction_cache_size)
        demo_ref = {"demo": None}
        groups = []
        cache_stats = {"hits": 0, "misses": 0, "writes": 0, "groups": 0, "skipped": 0}
        for split_name, samples in samples_by_split.items():
            if not samples:
                continue
            for sample in tqdm(samples, desc=f"Extract {split_name}"):
                try:
                    group = extract_candidate_group(
                        sample, get_vlpart_demo(demo_ref, args), args, pred_cache, keep_masks=False
                    )
                    groups.append(group)
                except Exception as exc:
                    skipped.append((sample.get("image_path", "unknown"), repr(exc)))
                    if len(skipped) <= 5 or args.debug:
                        print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    pred_iou_missing = sum(int(group.get("pred_iou_missing", 0)) for group in groups)

    if not groups:
        raise RuntimeError("No reranker training groups were extracted")

    random.shuffle(groups)
    train_groups = groups

    all_train_features = np.concatenate([group["features"] for group in train_groups], axis=0)
    mean_np = all_train_features.mean(axis=0).astype(np.float32)
    std_np = all_train_features.std(axis=0).astype(np.float32)
    std_np[std_np < 1e-6] = 1.0
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    model = RerankerMLP(input_dim=4, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    swanlab_run = init_swanlab(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / f"topk_reranker_train_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("VLPart top-K reranker training\n")
        log_f.write(f"groups: train={len(train_groups)}, skipped={len(skipped)}\n")
        log_f.write(f"feature_names: {FEATURE_NAMES}\n")
        log_f.write(f"feature_mean: {mean_np.tolist()}\n")
        log_f.write(f"feature_std: {std_np.tolist()}\n")
        log_f.write(f"pred_iou_missing_candidates: {pred_iou_missing}\n")
        if args.candidate_feature_cache_dir:
            log_f.write(f"feature_cache_dir: {args.candidate_feature_cache_dir}\n")
            log_f.write(
                "feature_cache_stats: "
                f"hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                f"writes={cache_stats['writes']}\n"
            )
        log_f.write("\n")

        if pred_iou_missing:
            print(
                "  [Warning] Some top-K masks do not expose LLMSeg pred_iou; "
                "their pred_iou feature is set to 0.0."
            )

        for epoch in range(1, args.epochs + 1):
            model.train()
            random.shuffle(train_groups)
            losses = []
            kl_losses = []
            mse_losses = []
            for group in train_groups:
                x = torch.as_tensor(group["features"], dtype=torch.float32, device=device)
                y = torch.as_tensor(group["targets"], dtype=torch.float32, device=device)
                logits = model(normalize_features(x, mean, std))
                loss = reranker_loss(logits, y, args)
                target_dist = F.softmax(y / max(args.target_temperature, 1e-6), dim=0)
                log_pred_dist = F.log_softmax(
                    logits / max(args.ranking_temperature, 1e-6), dim=0
                )
                kl_loss = F.kl_div(log_pred_dist, target_dist, reduction="batchmean")
                mse_loss = F.mse_loss(torch.sigmoid(logits), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                kl_losses.append(float(kl_loss.item()))
                mse_losses.append(float(mse_loss.item()))

            train_eval = evaluate_reranker_groups(model, train_groups, mean, std, device)
            line = (
                f"epoch {epoch:03d} loss={np.mean(losses):.6f} "
                f"kl={np.mean(kl_losses):.6f} "
                f"mse={np.mean(mse_losses):.6f} "
                f"train_selected={train_eval['selected_quality']:.4f} "
                f"train_oracle={train_eval['oracle_quality']:.4f} "
                f"train_mse={train_eval['reg_mse']:.6f}"
            )
            print(line)
            log_f.write(line + "\n")
            swanlab_log(
                swanlab_run,
                {
                    "train/loss": float(np.mean(losses)),
                    "train/kl_loss": float(np.mean(kl_losses)),
                    "train/mse_loss": float(np.mean(mse_losses)),
                    "train/selected_quality": train_eval["selected_quality"],
                    "train/oracle_quality": train_eval["oracle_quality"],
                    "train/reg_mse": train_eval["reg_mse"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

    ckpt = {
        "model_state": model.state_dict(),
        "feature_mean": mean_np.tolist(),
        "feature_std": std_np.tolist(),
        "feature_names": FEATURE_NAMES,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "args": vars(args),
    }
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.checkpoint_path)
    print(f"Saved reranker checkpoint: {args.checkpoint_path}")
    print(f"Training log: {log_path}")
    swanlab_finish(swanlab_run)

def load_reranker_checkpoint(args: argparse.Namespace, device: torch.device):
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model = RerankerMLP(
        input_dim=4,
        hidden_dim=int(ckpt.get("hidden_dim", args.hidden_dim)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ckpt["feature_std"], dtype=torch.float32, device=device)
    return model, mean, std


def pick_candidate(
    model: RerankerMLP,
    mean: torch.Tensor,
    std: torch.Tensor,
    group: Dict,
    device: torch.device,
) -> Tuple[int, float]:
    x = torch.as_tensor(group["features"], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(normalize_features(x, mean, std))
        idx = int(torch.argmax(logits).item())
        score = float(torch.sigmoid(logits[idx]).item())
    return idx, score


def evaluate_split(
    split_name: str,
    samples: List[Dict],
    demo_ref: Dict,
    model: RerankerMLP,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    pred_cache: PredictionCache,
):
    results = []
    skipped = []
    icr_records = []
    current_key = None
    current_predictions = []
    collect_icr = not args.skip_icr
    keep_masks = bool(args.save_vis or args.save_pred_masks or collect_icr)
    cache_stats = {"hits": 0, "misses": 0, "writes": 0}

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

    for ordinal, sample in enumerate(tqdm(samples, desc=f"Rerank {split_name}"), start=1):
        if collect_icr:
            sample_key = raw_sample_key(sample)
            if current_key is None:
                current_key = sample_key
            elif sample_key != current_key:
                flush_icr_record()
                current_key = sample_key

        try:
            group = load_or_extract_candidate_group(
                sample, demo_ref, args, pred_cache, keep_masks, split_name, cache_stats
            )
            best_idx, reranker_score = pick_candidate(model, mean, std, group, device)
            candidate = group["candidates"][best_idx]
            pred_fg = candidate.get("pred_fg")
            iou = float(candidate["target"])
            success = iou >= args.ssr_threshold
            instruction_index = int(sample.get("instruction_index", 0))

            selected_sample = dict(sample)
            stem = output_stem(selected_sample, ordinal)
            pred_mask_path = ""
            vis_path = ""
            pred_info = {
                "pred_score": candidate["vlpart_score"],
                "pred_class": candidate["pred_class"],
                "pred_class_name": candidate["pred_class_name"],
                "num_instances": len(group["candidates"]),
            }

            if args.save_pred_masks:
                if pred_fg is None:
                    raise RuntimeError("Prediction masks are unavailable from numeric feature cache")
                pred_mask_path = str(Path(args.pred_masks_dir) / split_name / f"{stem}.png")
                save_raw_pred_mask(pred_fg, pred_mask_path)
            if args.save_vis:
                if pred_fg is None:
                    raise RuntimeError("Prediction masks are unavailable from numeric feature cache")
                vis_path = str(Path(args.vis_dir) / split_name / f"{stem}_iou{iou:.3f}.png")
                save_prediction_overlay(
                    image_bgr=group["image_bgr"],
                    pred_fg=pred_fg,
                    gt_mask=candidate["best_gt"],
                    output_path=vis_path,
                    sample=selected_sample,
                    iou=iou,
                    pred_info=pred_info,
                )

            results.append(
                {
                    "image_path": sample["image_path"],
                    "gt_mask_path": sample["gt_mask_path"],
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
                    "reranker_score": reranker_score,
                    "target_quality": iou,
                    "llmseg_topk_rank": candidate["llmseg_topk_rank"],
                    "llmseg_topk_mask_path": candidate["llmseg_topk_mask_path"],
                    "vlpart_instance_index": candidate["vlpart_instance_index"],
                    "features": candidate["features"],
                    "pred_class": candidate["pred_class"],
                    "pred_class_name": candidate["pred_class_name"],
                    "pred_score": candidate["vlpart_score"],
                    "num_candidates": len(group["candidates"]),
                    "pred_mask_path": pred_mask_path,
                    "vis_path": vis_path,
                }
            )
            if collect_icr and instruction_index in (0, 1, 2) and pred_fg is not None:
                current_predictions.append((instruction_index, pred_fg.copy()))
        except Exception as exc:
            skipped.append((sample.get("image_path", "unknown"), repr(exc)))
            if len(skipped) <= 5 or args.debug:
                print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    if collect_icr:
        flush_icr_record()
    if args.candidate_feature_cache_dir and not keep_masks:
        print(
            f"  [{split_name} feature cache] "
            f"hits={cache_stats['hits']} misses={cache_stats['misses']} "
            f"writes={cache_stats['writes']} dir={args.candidate_feature_cache_dir}"
        )
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


def write_test_results(args, results, skipped_errors, icr_records) -> str:
    easy_results = results["easy"]
    hard_results = results["hard"]
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
    if args.skip_icr:
        easy_icr_text = "N/A"
        hard_icr_text = "N/A"
        avg_icr_text = "N/A"
    else:
        easy_icr = compute_icr_metrics(icr_records["easy"], args.icr_threshold)
        hard_icr = compute_icr_metrics(icr_records["hard"], args.icr_threshold)
        avg_icr = weighted_average(easy_icr, hard_icr, "icr")
        easy_icr_text = f"{easy_icr['icr']:.4f}"
        hard_icr_text = f"{hard_icr['icr']:.4f}"
        avg_icr_text = f"{avg_icr:.4f}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_file = str(
        Path(args.output_dir) / f"vlpart_topk_reranker_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart Top-K Reranker VIGOR Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Reranker checkpoint: {args.checkpoint_path}\n")
        f.write(f"VLPart config: {args.config_file}\n")
        f.write(f"VLPart weights: {args.weights}\n")
        f.write(f"LLMSeg top-K masks: {args.llmseg_topk_masks_dir}\n")
        f.write(f"Top-K K: {args.topk_mask_k}\n")
        f.write(f"Feature names: {FEATURE_NAMES}\n")
        f.write("Selection: highest reranker score, no GT oracle at inference\n\n")

        f.write(
            "| Method | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            f"SSR@{args.ssr_threshold:.1f} Easy | SSR@{args.ssr_threshold:.1f} Hard | "
            f"SSR@{args.ssr_threshold:.1f} Avg | ICR@{args.icr_threshold:.1f} Easy | "
            f"ICR@{args.icr_threshold:.1f} Hard | ICR@{args.icr_threshold:.1f} Avg |\n"
        )
        f.write(
            f"| VLPart+TopKReranker | {easy_metrics['ic_iou']:.4f} | "
            f"{hard_metrics['ic_iou']:.4f} | {all_summary['avg_ic_iou']:.4f} | "
            f"{easy_metrics['ssr']:.4f} | {hard_metrics['ssr']:.4f} | "
            f"{all_summary['avg_ssr']:.4f} | {easy_icr_text} | "
            f"{hard_icr_text} | {avg_icr_text} |\n\n"
        )
        if args.skip_icr:
            f.write("ICR skipped because --skip_icr was enabled; numeric feature cache does not store masks.\n\n")

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

        f.write(f"Skipped samples: Easy={len(skipped_errors['easy'])}, Hard={len(skipped_errors['hard'])}\n")
        for split_name in ["easy", "hard"]:
            if skipped_errors[split_name]:
                f.write(f"[{split_name} skipped examples]\n")
                for path, err in skipped_errors[split_name][:5]:
                    f.write(f"  {path}: {err}\n")
                f.write("\n")

        for split_name, split_results in [("Easy", easy_results), ("Hard", hard_results)]:
            f.write("=" * 80 + "\n")
            f.write(f"  {split_name} instruction details\n")
            f.write("=" * 80 + "\n\n")
            for idx, result in enumerate(split_results, start=1):
                f.write(f"[{split_name} Instruction Sample {idx}]\n")
                f.write(f"  Image: {result['image_path']}\n")
                f.write(f"  GT Mask: {result['gt_mask_path']}\n")
                f.write(f"  Object: {result['gt_object']}\n")
                f.write(f"  Instruction {result['instruction_index']}: {result.get('instruction', '')}\n")
                f.write(f"  IC-IoU: {result['avg_ic_iou']:.4f}\n")
                f.write(f"  SSR Success: {int(result['success'])}\n")
                f.write(f"  Reranker Score: {result['reranker_score']:.4f}\n")
                f.write(f"  Selected TopK Rank: {result['llmseg_topk_rank']}\n")
                f.write(f"  VLPart Instance Index: {result['vlpart_instance_index']}\n")
                f.write(f"  Features {FEATURE_NAMES}: {[f'{x:.4f}' for x in result['features']]}\n")
                f.write(f"  Candidate Count: {result['num_candidates']}\n")
                if result.get("pred_mask_path"):
                    f.write(f"  Pred Mask: {result['pred_mask_path']}\n")
                if result.get("vis_path"):
                    f.write(f"  Visualization: {result['vis_path']}\n")
                f.write("\n")
    return result_file


def test(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
    model, mean, std = load_reranker_checkpoint(args, device)
    demo_ref = {"demo": None}
    samples_by_split = load_instruction_samples(args)
    pred_cache = PredictionCache(args.prediction_cache_size)

    results = {"easy": [], "hard": []}
    skipped = {"easy": [], "hard": []}
    icr_records = {"easy": [], "hard": []}

    if args.save_vis:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
    if args.save_pred_masks:
        Path(args.pred_masks_dir).mkdir(parents=True, exist_ok=True)

    for split_name in ["easy", "hard"]:
        if not samples_by_split[split_name]:
            continue
        results[split_name], skipped[split_name], icr_records[split_name] = evaluate_split(
            split_name,
            samples_by_split[split_name],
            demo_ref,
            model,
            mean,
            std,
            args,
            device,
            pred_cache,
        )

    result_file = write_test_results(args, results, skipped, icr_records)
    all_summary = compute_split_summary(results["easy"], results["hard"], args.ssr_threshold)
    print(f"Result file: {result_file}")
    print(
        f"IC-IoU: Easy={all_summary['easy']['ic_iou']:.4f}, "
        f"Hard={all_summary['hard']['ic_iou']:.4f}, Avg={all_summary['avg_ic_iou']:.4f}"
    )
    print(
        f"SSR@{args.ssr_threshold:.1f}: Easy={all_summary['easy']['ssr']:.4f}, "
        f"Hard={all_summary['hard']['ssr']:.4f}, Avg={all_summary['avg_ssr']:.4f}"
    )
    if args.skip_icr:
        print("ICR: skipped (--skip_icr; numeric feature cache does not store masks)")


def main() -> None:
    args = parse_args()
    prepare_common_args(args)
    setup_logger(name="fvcore")
    setup_logger().info("Arguments: " + str(args))
    if args.mode == "train":
        train(args)
    elif args.mode == "extract_train_cache":
        extract_feature_cache(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
