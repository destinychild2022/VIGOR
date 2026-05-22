import argparse
import csv
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
FORMULA_SCORER_FEATURE_NAMES = ["pred_similarity", "pred_iou", "vlpart_score"]

CANDIDATE_FEATURE_CACHE_VERSION = 1
CANDIDATE_FEATURE_SHARD_CACHE_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train/test a VLPart top-K affordance reranker")
    parser.add_argument("--mode", choices=["train", "test", "extract_train_cache", "analyze_features", "train_formula_scorer", "test_formula_alpha_sweep"], required=True)
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
    parser.add_argument("--save_feature_rows", action="store_true")
    parser.add_argument(
        "--ranker_feature",
        action="append",
        choices=FEATURE_NAMES,
        default=None,
        help=(
            "Feature used as reranker input. Repeat to select multiple features. "
            f"Default: all features {FEATURE_NAMES}."
        ),
    )

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
    parser.add_argument("--feature_cache_chunk_size", type=int, default=1000)
    parser.add_argument("--balance_train_easy_hard", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--extra_mlp_layer", action="store_true")
    parser.add_argument(
        "--extra_hidden_dim",
        type=int,
        default=0,
        help="Hidden dimension of the optional extra MLP layer; 0 means reuse --hidden_dim.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--target_temperature", type=float, default=0.10)
    parser.add_argument("--ranking_temperature", type=float, default=1.0)
    parser.add_argument("--formula_alpha_init", type=float, default=0.5)
    parser.add_argument("--formula_alpha_start", type=float, default=0.0)
    parser.add_argument("--formula_alpha_end", type=float, default=1.0)
    parser.add_argument("--formula_alpha_step", type=float, default=0.1)
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
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        extra_mlp_layer: bool = False,
        extra_hidden_dim: int = 0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        extra_hidden_dim = int(extra_hidden_dim) if int(extra_hidden_dim) > 0 else hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        if extra_mlp_layer:
            layers.extend(
                [
                    nn.Linear(hidden_dim, extra_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            output_dim = extra_hidden_dim
        else:
            output_dim = hidden_dim
        layers.append(nn.Linear(output_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class FormulaAlphaScorer(nn.Module):
    def __init__(self, alpha_init: float = 0.5):
        super().__init__()
        alpha = float(np.clip(alpha_init, 1e-4, 1.0 - 1e-4))
        raw_alpha = np.log(alpha / (1.0 - alpha))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def alpha_value(self) -> float:
        return float(self.alpha().detach().cpu().item())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pred_similarity = features[:, 0]
        pred_iou = features[:, 1]
        vlpart_score = features[:, 2]
        alpha = self.alpha()
        return (alpha * pred_similarity + (1.0 - alpha) * pred_iou) * vlpart_score


def resolve_ranker_feature_names(args: argparse.Namespace) -> List[str]:
    selected = getattr(args, "ranker_feature", None)
    if not selected:
        return list(FEATURE_NAMES)
    names = []
    for name in selected:
        if name not in FEATURE_NAMES:
            raise ValueError(f"Unknown ranker feature: {name}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one --ranker_feature must be selected")
    return names


def ranker_feature_indices(feature_names: List[str]) -> List[int]:
    return [FEATURE_NAMES.index(name) for name in feature_names]


def select_reranker_feature_matrix(features, feature_indices: List[int]) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected candidate features to be 2-D, got shape={arr.shape}")
    if arr.shape[1] < len(FEATURE_NAMES):
        raise ValueError(
            f"Expected cached candidate features to contain {len(FEATURE_NAMES)} columns "
            f"{FEATURE_NAMES}, got shape={arr.shape}"
        )
    return arr[:, feature_indices]


def group_feature_tensor(group: Dict, feature_indices: List[int], device: torch.device) -> torch.Tensor:
    selected = select_reranker_feature_matrix(group["features"], feature_indices)
    return torch.as_tensor(selected, dtype=torch.float32, device=device)


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


def candidate_feature_shard_chunk_dir(
    args: argparse.Namespace,
    split_name: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        Path(args.candidate_feature_cache_dir)
        / "chunks"
        / f"{split_name}_shard{shard_index:03d}_of{num_shards:03d}"
    )


def candidate_feature_shard_chunk_cache_path(
    args: argparse.Namespace,
    split_name: str,
    shard_index: int,
    num_shards: int,
    chunk_index: int,
    entry_start: int,
    entry_count: int,
) -> Path:
    return (
        candidate_feature_shard_chunk_dir(args, split_name, shard_index, num_shards)
        / f"chunk{chunk_index:05d}_start{entry_start:06d}_count{entry_count:04d}.pt"
    )


def build_feature_chunk_digest(
    shard_digest: str,
    chunk_entries: List[Tuple[Dict, str]],
    entry_start: int,
) -> str:
    payload = {
        "version": CANDIDATE_FEATURE_SHARD_CACHE_VERSION,
        "format": "vlpart_topk_reranker_numeric_chunk",
        "entry_start": int(entry_start),
        "entry_count": len(chunk_entries),
        "sample_digests": [sample_digest for _sample, sample_digest in chunk_entries],
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def load_numeric_chunk_payload(chunk_path: Path) -> Optional[Dict]:
    try:
        payload = torch.load(chunk_path, map_location="cpu")
    except Exception as exc:
        print(f"  [Warning] Failed to read chunk cache {chunk_path}: {exc}", flush=True)
        return None
    if payload.get("version") != CANDIDATE_FEATURE_SHARD_CACHE_VERSION:
        return None
    if payload.get("feature_names") != FEATURE_NAMES:
        return None
    groups = payload.get("groups", [])
    if not isinstance(groups, list):
        return None
    return payload


def load_candidate_feature_shard_chunk_subset_cache(
    chunk_dir: Path,
    expected_sample_digests: List[str],
) -> Optional[List[Dict]]:
    if not chunk_dir.is_dir() or not expected_sample_digests:
        return None

    needed = set(expected_sample_digests)
    found: Dict[str, Dict] = {}
    for candidate_path in sorted(chunk_dir.glob("*.pt")):
        if candidate_path.name.endswith(".tmp"):
            continue
        payload = load_numeric_chunk_payload(candidate_path)
        if payload is None:
            continue
        groups = payload.get("groups", [])
        touched = False
        for entry in groups:
            sample_digest = entry.get("sample_digest", "")
            if sample_digest in needed and sample_digest not in found:
                found[sample_digest] = entry
                touched = True
        if touched and len(found) == len(needed):
            ordered = [found[digest] for digest in expected_sample_digests]
            print(
                f"  [chunk cache subset] reused {len(ordered)} groups from existing chunks in {chunk_dir}",
                flush=True,
            )
            return [group_from_numeric_cache(entry) for entry in ordered]
    return None


def load_candidate_feature_shard_chunk_cache(
    chunk_path: Path,
    expected_digest: str,
    expected_count: int,
    expected_sample_digests: Optional[List[str]] = None,
) -> Optional[List[Dict]]:
    if chunk_path.exists():
        payload = load_numeric_chunk_payload(chunk_path)
        if payload is not None:
            groups = payload.get("groups", [])
            if len(groups) == expected_count:
                if payload.get("chunk_digest") == expected_digest:
                    return [group_from_numeric_cache(entry) for entry in groups]
                cached_sample_digests = [entry.get("sample_digest", "") for entry in groups]
                if expected_sample_digests is not None and cached_sample_digests == expected_sample_digests:
                    print(
                        f"  [chunk cache compat] accepted by sample digests: {chunk_path}",
                        flush=True,
                    )
                    return [group_from_numeric_cache(entry) for entry in groups]

    if expected_sample_digests is not None:
        return load_candidate_feature_shard_chunk_subset_cache(
            chunk_path.parent,
            expected_sample_digests,
        )
    return None

def save_candidate_feature_shard_chunk_cache(
    chunk_path: Path,
    chunk_digest: str,
    shard_digest: str,
    split_name: str,
    shard_index: int,
    num_shards: int,
    chunk_index: int,
    entry_start: int,
    groups: List[Dict],
    stats: Dict,
) -> None:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CANDIDATE_FEATURE_SHARD_CACHE_VERSION,
        "format": "vlpart_topk_reranker_numeric_chunk",
        "feature_names": FEATURE_NAMES,
        "chunk_digest": chunk_digest,
        "shard_digest": shard_digest,
        "split": split_name,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "chunk_index": int(chunk_index),
        "entry_start": int(entry_start),
        "entry_count": len(groups),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": dict(stats),
        "groups": groups,
    }
    tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, chunk_path)


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

    # Backward compatibility for old one-file shard caches. New extraction writes chunks.
    shard_path = candidate_feature_shard_cache_path(args, split_name, shard_index, num_shards)
    cached_groups = load_candidate_feature_shard_cache(
        shard_path, shard_digest, expected_count=len(entries)
    )
    if cached_groups is not None:
        stats["hits"] = len(cached_groups)
        stats["groups"] = len(cached_groups)
        print(
            f"  [{split_name} shard {shard_index + 1}/{num_shards}] cache hit: {shard_path}",
            flush=True,
        )
        return ([] if allow_extract else cached_groups), stats

    chunk_size = max(1, int(getattr(args, "feature_cache_chunk_size", 1000) or 1000))
    total_chunks = (len(entries) + chunk_size - 1) // chunk_size
    print(
        f"  [{split_name} shard {shard_index + 1}/{num_shards}] "
        f"chunk cache: size={chunk_size}, chunks={total_chunks}",
        flush=True,
    )

    all_groups = []
    pred_cache = None
    demo_ref = {"demo": None}

    for chunk_index, entry_start in enumerate(range(0, len(entries), chunk_size)):
        entry_end = min(entry_start + chunk_size, len(entries))
        chunk_entries = entries[entry_start:entry_end]
        chunk_digest = build_feature_chunk_digest(shard_digest, chunk_entries, entry_start)
        chunk_path = candidate_feature_shard_chunk_cache_path(
            args,
            split_name,
            shard_index,
            num_shards,
            chunk_index,
            entry_start,
            len(chunk_entries),
        )
        cached_chunk = load_candidate_feature_shard_chunk_cache(
            chunk_path,
            chunk_digest,
            expected_count=len(chunk_entries),
            expected_sample_digests=[sample_digest for _sample, sample_digest in chunk_entries],
        )
        if cached_chunk is not None:
            stats["hits"] += len(cached_chunk)
            stats["groups"] += len(cached_chunk)
            if not allow_extract:
                all_groups.extend(cached_chunk)
            print(
                f"  [{split_name} shard {shard_index + 1}/{num_shards}] "
                f"chunk {chunk_index + 1}/{total_chunks} cache hit: {chunk_path.name}",
                flush=True,
            )
            continue

        if not allow_extract:
            raise FileNotFoundError(
                f"Chunk cache miss for {split_name} shard {shard_index + 1}/{num_shards} "
                f"chunk {chunk_index + 1}/{total_chunks}: {chunk_path}"
            )

        stats["misses"] += len(chunk_entries)
        if pred_cache is None:
            pred_cache = PredictionCache(args.prediction_cache_size)

        cached_entries = []
        skipped = []
        desc = (
            f"Cache {split_name} shard {shard_index + 1}/{num_shards} "
            f"chunk {chunk_index + 1}/{total_chunks}"
        )
        for sample, sample_digest in tqdm(chunk_entries, desc=desc):
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
                    print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}", flush=True)

        stats["skipped"] += len(skipped)
        if skipped:
            raise RuntimeError(
                f"Feature chunk extraction skipped {len(skipped)} samples in "
                f"{split_name} shard {shard_index + 1}/{num_shards} "
                f"chunk {chunk_index + 1}/{total_chunks}. Completed previous chunks remain cached."
            )

        stats["writes"] += len(cached_entries)
        stats["groups"] += len(cached_entries)
        save_candidate_feature_shard_chunk_cache(
            chunk_path,
            chunk_digest,
            shard_digest,
            split_name,
            shard_index,
            num_shards,
            chunk_index,
            entry_start,
            cached_entries,
            stats,
        )
        print(
            f"  [{split_name} shard {shard_index + 1}/{num_shards}] "
            f"wrote chunk {chunk_index + 1}/{total_chunks}: {chunk_path}",
            flush=True,
        )

    return ([] if allow_extract else all_groups), stats

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


def balance_train_easy_hard_samples(
    args: argparse.Namespace,
    samples: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    if not getattr(args, "balance_train_easy_hard", False):
        return samples
    if getattr(args, "mode", "") not in {"train", "train_formula_scorer", "extract_train_cache"}:
        return samples
    if args.split != "both":
        print(
            "  [train balance Warning] --balance_train_easy_hard requires --split both; "
            f"current split={args.split}, keeping loaded samples unchanged.",
            flush=True,
        )
        return samples

    easy_samples = samples.get("easy", [])
    hard_samples = samples.get("hard", [])
    if not easy_samples or not hard_samples:
        print(
            "  [train balance Warning] easy or hard split is empty; "
            f"easy={len(easy_samples)}, hard={len(hard_samples)}.",
            flush=True,
        )
        return samples

    balanced = dict(samples)
    if len(easy_samples) > len(hard_samples):
        balanced["easy"] = easy_samples[: len(hard_samples)]
        print(
            "  [train balance] using all hard and first easy samples: "
            f"easy {len(easy_samples)} -> {len(balanced['easy'])}, "
            f"hard {len(hard_samples)} -> {len(hard_samples)}",
            flush=True,
        )
    elif len(easy_samples) < len(hard_samples):
        print(
            "  [train balance Warning] hard has more samples than easy; keeping full hard "
            f"and all easy, ratio cannot be 1:1. easy={len(easy_samples)}, hard={len(hard_samples)}",
            flush=True,
        )
    else:
        print(
            f"  [train balance] easy/hard already 1:1: easy={len(easy_samples)}, hard={len(hard_samples)}",
            flush=True,
        )
    return balanced


def load_instruction_samples(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    samples = {"easy": [], "hard": []}
    if args.split in ["easy", "both"]:
        samples["easy"] = load_samples_per_instruction(args.data_dir, args.easy_json_file, "easy")
    if args.split in ["hard", "both"]:
        samples["hard"] = load_samples_per_instruction(args.data_dir, args.hard_json_file, "hard")
    if args.max_samples is not None:
        samples["easy"] = samples["easy"][: args.max_samples]
        samples["hard"] = samples["hard"][: args.max_samples]
    samples = balance_train_easy_hard_samples(args, samples)
    if getattr(args, "mode", "") in {"train", "train_formula_scorer", "extract_train_cache"}:
        print(
            "  [train samples] "
            f"easy={len(samples.get('easy', []))}, hard={len(samples.get('hard', []))}, "
            f"total={len(samples.get('easy', [])) + len(samples.get('hard', []))}",
            flush=True,
        )
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


def one_hot_top_iou_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if scores.numel() == 0:
        raise ValueError("Cannot compute one-hot loss for an empty candidate group")
    target_index = torch.argmax(targets).view(1)
    logits = (scores / max(args.ranking_temperature, 1e-6)).view(1, -1)
    return F.cross_entropy(logits, target_index)


def evaluate_reranker_groups(
    model: RerankerMLP,
    groups: List[Dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    feature_indices: List[int],
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
            x = group_feature_tensor(group, feature_indices, device)
            y = torch.as_tensor(group["targets"], dtype=torch.float32, device=device)
            logits = model(x)
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
    args.ranker_feature_names = resolve_ranker_feature_names(args)
    args.ranker_feature_indices = ranker_feature_indices(args.ranker_feature_names)


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
                "ranker_feature_names": getattr(args, "ranker_feature_names", FEATURE_NAMES),
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hidden_dim": args.hidden_dim,
                "extra_mlp_layer": args.extra_mlp_layer,
                "extra_hidden_dim": args.extra_hidden_dim,
                "dropout": args.dropout,
                "kl_weight": args.kl_weight,
                "mse_weight": args.mse_weight,
                "target_temperature": args.target_temperature,
                "ranking_temperature": args.ranking_temperature,
                "formula_alpha_init": args.formula_alpha_init,
                "seed": args.seed,
                "balance_train_easy_hard": args.balance_train_easy_hard,
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

    ranker_feature_names = args.ranker_feature_names
    ranker_feature_indices = args.ranker_feature_indices
    print(f"  [ranker features] {ranker_feature_names}")
    all_train_features = np.concatenate(
        [select_reranker_feature_matrix(group["features"], ranker_feature_indices) for group in train_groups],
        axis=0,
    )
    # Use raw feature values exactly as stored in the cache. Identity stats are
    # still saved for checkpoint compatibility with the shared test code.
    mean_np = np.zeros(all_train_features.shape[1], dtype=np.float32)
    std_np = np.ones(all_train_features.shape[1], dtype=np.float32)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    extra_hidden_dim = int(args.extra_hidden_dim) if int(args.extra_hidden_dim) > 0 else int(args.hidden_dim)
    print(
        "  [ranker architecture] "
        f"input_dim={len(ranker_feature_names)} hidden_dim={args.hidden_dim} "
        f"extra_mlp_layer={args.extra_mlp_layer} extra_hidden_dim={extra_hidden_dim} output_dim=1"
    )
    model = RerankerMLP(
        input_dim=len(ranker_feature_names),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        extra_mlp_layer=args.extra_mlp_layer,
        extra_hidden_dim=extra_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    swanlab_run = init_swanlab(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / f"topk_reranker_train_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("VLPart top-K reranker training\n")
        log_f.write(f"groups: train={len(train_groups)}, skipped={len(skipped)}\n")
        log_f.write(
            f"train_samples: easy={len(samples_by_split.get('easy', []))}, "
            f"hard={len(samples_by_split.get('hard', []))}\n"
        )
        log_f.write("optimizer_update_batch_groups: 1\n")
        log_f.write(f"cache_feature_names: {FEATURE_NAMES}\n")
        log_f.write(f"ranker_feature_names: {ranker_feature_names}\n")
        log_f.write("feature_input: raw cache values, no normalization\n")
        log_f.write("loss: one-hot cross entropy over all candidates in each instruction group\n")
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
            for group in train_groups:
                x = group_feature_tensor(group, ranker_feature_indices, device)
                y = torch.as_tensor(group["targets"], dtype=torch.float32, device=device)
                logits = model(x)
                loss = one_hot_top_iou_loss(logits, y, args)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            train_eval = evaluate_reranker_groups(
                model, train_groups, mean, std, ranker_feature_indices, device
            )
            line = (
                f"epoch {epoch:03d} one_hot_ce={np.mean(losses):.6f} "
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
                    "train/one_hot_ce_loss": float(np.mean(losses)),
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
        "feature_names": ranker_feature_names,
        "ranker_feature_names": ranker_feature_names,
        "all_feature_names": FEATURE_NAMES,
        "feature_input": "raw_no_normalization",
        "loss_type": "one_hot_top_iou_cross_entropy",
        "hidden_dim": args.hidden_dim,
        "extra_mlp_layer": bool(args.extra_mlp_layer),
        "extra_hidden_dim": int(extra_hidden_dim),
        "dropout": args.dropout,
        "args": vars(args),
    }
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.checkpoint_path)
    print(f"Saved reranker checkpoint: {args.checkpoint_path}")
    print(f"Training log: {log_path}")
    swanlab_finish(swanlab_run)


def train_formula_scorer(args: argparse.Namespace) -> None:
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
            "  [formula scorer train feature shard cache] "
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
                        sample,
                        get_vlpart_demo(demo_ref, args),
                        args,
                        pred_cache,
                        keep_masks=False,
                    )
                    groups.append(group)
                except Exception as exc:
                    skipped.append((sample.get("image_path", "unknown"), repr(exc)))
                    if len(skipped) <= 5 or args.debug:
                        print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")

    pred_iou_missing = sum(int(group.get("pred_iou_missing", 0)) for group in groups)
    if not groups:
        raise RuntimeError("No formula scorer training groups were extracted")

    random.shuffle(groups)
    train_groups = groups
    formula_feature_names = list(FORMULA_SCORER_FEATURE_NAMES)
    formula_feature_indices = ranker_feature_indices(formula_feature_names)
    args.ranker_feature_names = formula_feature_names
    args.ranker_feature_indices = formula_feature_indices
    mean = torch.zeros(len(formula_feature_names), dtype=torch.float32, device=device)
    std = torch.ones(len(formula_feature_names), dtype=torch.float32, device=device)

    model = FormulaAlphaScorer(alpha_init=args.formula_alpha_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    swanlab_run = init_swanlab(args)

    print(
        "  [formula scorer] score = "
        "(alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score"
    )
    print(f"  [formula scorer] init alpha={model.alpha_value():.6f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / f"topk_formula_scorer_train_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("VLPart top-K formula scorer training\n")
        log_f.write(
            "formula: (alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score\n"
        )
        log_f.write(f"formula_feature_names: {formula_feature_names}\n")
        log_f.write(f"groups: train={len(train_groups)}, skipped={len(skipped)}\n")
        log_f.write(
            f"train_samples: easy={len(samples_by_split.get('easy', []))}, "
            f"hard={len(samples_by_split.get('hard', []))}\n"
        )
        log_f.write("optimizer_update_batch_groups: 1\n")
        log_f.write(f"alpha_init: {args.formula_alpha_init}\n")
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
            for group in train_groups:
                x = group_feature_tensor(group, formula_feature_indices, device)
                y = torch.as_tensor(group["targets"], dtype=torch.float32, device=device)
                scores = model(x)
                loss = one_hot_top_iou_loss(scores, y, args)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            train_eval = evaluate_reranker_groups(
                model, train_groups, mean, std, formula_feature_indices, device
            )
            alpha_value = model.alpha_value()
            line = (
                f"epoch {epoch:03d} one_hot_ce={np.mean(losses):.6f} "
                f"alpha={alpha_value:.6f} "
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
                    "train/one_hot_ce_loss": float(np.mean(losses)),
                    "train/selected_quality": train_eval["selected_quality"],
                    "train/oracle_quality": train_eval["oracle_quality"],
                    "train/reg_mse": train_eval["reg_mse"],
                    "train/alpha": alpha_value,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

    ckpt = {
        "scorer_type": "formula_alpha",
        "model_state": model.state_dict(),
        "alpha": float(np.float32(model.alpha_value())),
        "alpha_fp32": model.alpha().detach().cpu().to(torch.float32),
        "formula": "(alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score",
        "feature_names": formula_feature_names,
        "ranker_feature_names": formula_feature_names,
        "all_feature_names": FEATURE_NAMES,
        "feature_mean": [0.0 for _ in formula_feature_names],
        "feature_std": [1.0 for _ in formula_feature_names],
        "args": vars(args),
    }
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.checkpoint_path)
    print(f"Saved formula scorer checkpoint: {args.checkpoint_path}")
    print(f"Final alpha: {model.alpha_value():.6f}")
    print(f"Training log: {log_path}")
    swanlab_finish(swanlab_run)

def load_reranker_checkpoint(args: argparse.Namespace, device: torch.device):
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    scorer_type = ckpt.get("scorer_type", "mlp_reranker")
    if scorer_type == "formula_alpha":
        formula_feature_names = list(ckpt.get("feature_names", FORMULA_SCORER_FEATURE_NAMES))
        if formula_feature_names != FORMULA_SCORER_FEATURE_NAMES:
            raise ValueError(
                "Formula scorer checkpoint must use feature order "
                f"{FORMULA_SCORER_FEATURE_NAMES}, got {formula_feature_names}"
            )
        alpha_init = ckpt.get("alpha_fp32", ckpt.get("alpha", 0.5))
        if isinstance(alpha_init, torch.Tensor):
            alpha_init = float(alpha_init.to(dtype=torch.float32).cpu().item())
        model = FormulaAlphaScorer(alpha_init=float(np.float32(alpha_init))).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        feature_indices = ranker_feature_indices(formula_feature_names)
        mean = torch.zeros(len(formula_feature_names), dtype=torch.float32, device=device)
        std = torch.ones(len(formula_feature_names), dtype=torch.float32, device=device)
        args.ranker_feature_names = formula_feature_names
        args.ranker_feature_indices = feature_indices
        print(
            "  [formula scorer checkpoint] "
            f"alpha={model.alpha_value():.6f}; "
            "score=(alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score"
        )
        return model, mean, std, feature_indices, formula_feature_names

    ranker_feature_names = ckpt.get("ranker_feature_names", ckpt.get("feature_names", FEATURE_NAMES))
    if not ranker_feature_names:
        ranker_feature_names = list(FEATURE_NAMES)
    ranker_feature_names = list(ranker_feature_names)
    requested_feature_names = resolve_ranker_feature_names(args)
    if getattr(args, "ranker_feature", None) and requested_feature_names != ranker_feature_names:
        print(
            "  [Warning] Ignoring requested --ranker_feature values during test; "
            f"checkpoint was trained with {ranker_feature_names}."
        )
    feature_indices = ranker_feature_indices(ranker_feature_names)
    model = RerankerMLP(
        input_dim=len(ranker_feature_names),
        hidden_dim=int(ckpt.get("hidden_dim", args.hidden_dim)),
        dropout=float(ckpt.get("dropout", 0.0)),
        extra_mlp_layer=bool(ckpt.get("extra_mlp_layer", False)),
        extra_hidden_dim=int(ckpt.get("extra_hidden_dim", ckpt.get("hidden_dim", args.hidden_dim))),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ckpt["feature_std"], dtype=torch.float32, device=device)
    if mean.numel() != len(ranker_feature_names) or std.numel() != len(ranker_feature_names):
        raise ValueError(
            "Checkpoint feature stats do not match ranker feature names: "
            f"names={ranker_feature_names}, mean_shape={tuple(mean.shape)}, std_shape={tuple(std.shape)}"
        )
    args.ranker_feature_names = ranker_feature_names
    args.ranker_feature_indices = feature_indices
    return model, mean, std, feature_indices, ranker_feature_names


def pick_candidate(
    model: RerankerMLP,
    mean: torch.Tensor,
    std: torch.Tensor,
    feature_indices: List[int],
    group: Dict,
    device: torch.device,
) -> Tuple[int, float]:
    x = group_feature_tensor(group, feature_indices, device)
    with torch.no_grad():
        logits = model(x)
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
    feature_indices: List[int],
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
            best_idx, reranker_score = pick_candidate(model, mean, std, feature_indices, group, device)
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
        f.write(f"Cache feature names: {FEATURE_NAMES}\n")
        f.write(
            "Ranker input feature names: "
            + str(getattr(args, "ranker_feature_names", FEATURE_NAMES))
            + "\n"
        )
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
                f.write(f"  Candidate features {FEATURE_NAMES}: {[f'{x:.4f}' for x in result['features']]}\n")
                f.write(f"  Candidate Count: {result['num_candidates']}\n")
                if result.get("pred_mask_path"):
                    f.write(f"  Pred Mask: {result['pred_mask_path']}\n")
                if result.get("vis_path"):
                    f.write(f"  Visualization: {result['vis_path']}\n")
                f.write("\n")
    return result_file



def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x_std * y_std))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    return pearson_corr(rankdata_average_ties(x[mask]), rankdata_average_ties(y[mask]))


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    mask = np.isfinite(scores) & np.isfinite(labels.astype(np.float64))
    scores = scores[mask]
    labels = labels[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average_ties(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def linear_r2(features: np.ndarray, targets: np.ndarray, feature_indices: Optional[List[int]] = None) -> Tuple[float, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if feature_indices is not None:
        x = x[:, feature_indices]
    if len(y) < 2 or x.shape[1] == 0:
        return float("nan"), np.zeros(x.shape[1], dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    x_norm = (x - mean) / std
    design = np.concatenate([np.ones((len(x_norm), 1)), x_norm], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(r2), coef[1:]


def summarize_feature_rows(rows: List[Dict], args: argparse.Namespace) -> Dict:
    if not rows:
        return {
            "num_candidates": 0,
            "num_groups": 0,
            "target_mean": 0.0,
            "target_std": 0.0,
            "target_success_rate": 0.0,
            "features": {},
            "linear": {},
            "selection": {},
        }

    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target"] for row in rows], dtype=np.float64)
    labels = targets >= float(args.ssr_threshold)
    group_keys = sorted({row["group_key"] for row in rows})

    feature_stats = {}
    for idx, name in enumerate(FEATURE_NAMES):
        values = features[:, idx]
        p = pearson_corr(values, targets)
        s = spearman_corr(values, targets)
        feature_stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "pearson": p,
            "spearman": s,
            "r2_univariate": float(p * p) if np.isfinite(p) else float("nan"),
            "auc_success": auc_from_scores(values, labels),
        }

    full_r2, coefs = linear_r2(features, targets)
    leave_one_out = {}
    for idx, name in enumerate(FEATURE_NAMES):
        keep = [j for j in range(len(FEATURE_NAMES)) if j != idx]
        r2_without, _ = linear_r2(features, targets, keep)
        leave_one_out[name] = {
            "r2_without": r2_without,
            "delta_r2": float(full_r2 - r2_without) if np.isfinite(full_r2) and np.isfinite(r2_without) else float("nan"),
        }

    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_key"], []).append(row)
    selection = {}
    for idx, name in enumerate(FEATURE_NAMES):
        selected_targets = []
        oracle_targets = []
        exact_oracle = 0
        for group_rows in by_group.values():
            values = np.asarray([r["features"][idx] for r in group_rows], dtype=np.float64)
            target_values = np.asarray([r["target"] for r in group_rows], dtype=np.float64)
            selected_idx = int(np.argmax(values))
            oracle_idx = int(np.argmax(target_values))
            selected_targets.append(float(target_values[selected_idx]))
            oracle_targets.append(float(target_values[oracle_idx]))
            exact_oracle += int(selected_idx == oracle_idx)
        selected_targets_np = np.asarray(selected_targets, dtype=np.float64)
        oracle_targets_np = np.asarray(oracle_targets, dtype=np.float64)
        selection[name] = {
            "mean_selected_iou": float(selected_targets_np.mean()),
            "ssr": float(np.mean(selected_targets_np >= float(args.ssr_threshold))),
            "oracle_mean_iou": float(oracle_targets_np.mean()),
            "oracle_gap": float(oracle_targets_np.mean() - selected_targets_np.mean()),
            "oracle_pick_rate": float(exact_oracle / max(1, len(by_group))),
        }

    full_linear = {
        "r2": full_r2,
        "standardized_coefficients": {
            name: float(coefs[idx]) for idx, name in enumerate(FEATURE_NAMES)
        },
        "leave_one_out": leave_one_out,
    }
    return {
        "num_candidates": len(rows),
        "num_groups": len(group_keys),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "target_success_rate": float(labels.mean()),
        "features": feature_stats,
        "linear": full_linear,
        "selection": selection,
    }


def collect_feature_rows_for_split(
    split_name: str,
    samples: List[Dict],
    demo_ref: Dict,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
) -> Tuple[List[Dict], List[Tuple[str, str]], Dict[str, int]]:
    rows = []
    skipped = []
    cache_stats = {"hits": 0, "misses": 0, "writes": 0}
    for ordinal, sample in enumerate(tqdm(samples, desc=f"Analyze {split_name}"), start=1):
        try:
            group = load_or_extract_candidate_group(
                sample,
                demo_ref,
                args,
                pred_cache,
                keep_masks=False,
                split_name=split_name,
                cache_stats=cache_stats,
            )
            group_key = (
                f"{split_name}|{sample.get('sample_index', -1)}|"
                f"{sample.get('instruction_index', 0)}|{sample.get('image_path', '')}"
            )
            for candidate_index, candidate in enumerate(group["candidates"]):
                feature_values = [float(v) for v in candidate["features"]]
                rows.append(
                    {
                        "split": split_name,
                        "group_key": group_key,
                        "sample_index": sample.get("sample_index", -1),
                        "instruction_index": sample.get("instruction_index", 0),
                        "image_path": sample.get("image_path", ""),
                        "gt_object": sample.get("gt_object", ""),
                        "candidate_index": candidate_index,
                        "target": float(candidate["target"]),
                        "features": feature_values,
                        "llmseg_topk_rank": candidate.get("llmseg_topk_rank", -1),
                        "llmseg_topk_mask_path": candidate.get("llmseg_topk_mask_path", ""),
                        "vlpart_instance_index": candidate.get("vlpart_instance_index", -1),
                        "pred_class": candidate.get("pred_class", -1),
                        "pred_class_name": candidate.get("pred_class_name", ""),
                    }
                )
        except Exception as exc:
            skipped.append((sample.get("image_path", "unknown"), repr(exc)))
            if len(skipped) <= 5 or args.debug:
                print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}", flush=True)
    print(
        f"  [{split_name} feature cache] hits={cache_stats['hits']} misses={cache_stats['misses']} "
        f"writes={cache_stats['writes']} dir={args.candidate_feature_cache_dir}",
        flush=True,
    )
    return rows, skipped, cache_stats


def write_feature_rows_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_index",
        "instruction_index",
        "image_path",
        "gt_object",
        "candidate_index",
        "target_gt_iou",
        *FEATURE_NAMES,
        "llmseg_topk_rank",
        "llmseg_topk_mask_path",
        "vlpart_instance_index",
        "pred_class",
        "pred_class_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "split": row["split"],
                "sample_index": row["sample_index"],
                "instruction_index": row["instruction_index"],
                "image_path": row["image_path"],
                "gt_object": row["gt_object"],
                "candidate_index": row["candidate_index"],
                "target_gt_iou": row["target"],
                "llmseg_topk_rank": row["llmseg_topk_rank"],
                "llmseg_topk_mask_path": row["llmseg_topk_mask_path"],
                "vlpart_instance_index": row["vlpart_instance_index"],
                "pred_class": row["pred_class"],
                "pred_class_name": row["pred_class_name"],
            }
            for idx, name in enumerate(FEATURE_NAMES):
                out[name] = row["features"][idx]
            writer.writerow(out)


def format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.6f}"


def write_feature_analysis_results(
    args: argparse.Namespace,
    rows_by_split: Dict[str, List[Dict]],
    skipped_by_split: Dict[str, List[Tuple[str, str]]],
    cache_stats_by_split: Dict[str, Dict[str, int]],
) -> str:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = Path(args.output_dir) / f"vlpart_topk_feature_correlation_{timestamp}.txt"
    all_rows = rows_by_split.get("easy", []) + rows_by_split.get("hard", [])
    summaries = {
        "all": summarize_feature_rows(all_rows, args),
        "easy": summarize_feature_rows(rows_by_split.get("easy", []), args),
        "hard": summarize_feature_rows(rows_by_split.get("hard", []), args),
    }
    with result_file.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart Top-K Feature Correlation Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write("Target: per-candidate GT IoU / quality used by the reranker.\n")
        f.write(f"Feature names: {FEATURE_NAMES}\n")
        f.write(f"LLMSeg top-K masks: {args.llmseg_topk_masks_dir}\n")
        f.write(f"Top-K K: {args.topk_mask_k}\n")
        f.write(f"SSR threshold for AUC/selection: {args.ssr_threshold}\n")
        f.write(f"Feature cache dir: {args.candidate_feature_cache_dir}\n\n")

        for split_name in ["all", "easy", "hard"]:
            summary = summaries[split_name]
            f.write("=" * 80 + "\n")
            f.write(f"  {split_name.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"candidates={summary['num_candidates']} groups={summary['num_groups']} "
                f"target_mean={summary['target_mean']:.6f} target_std={summary['target_std']:.6f} "
                f"success_rate={summary['target_success_rate']:.6f}\n\n"
            )
            f.write("Candidate-level correlation with target GT IoU:\n")
            f.write("| Feature | Pearson | Spearman | R2(corr^2) | AUC@SSR | Mean | Std | Min | Max |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for name in FEATURE_NAMES:
                stats = summary["features"].get(name, {})
                f.write(
                    f"| {name} | {format_float(stats.get('pearson', float('nan')))} | "
                    f"{format_float(stats.get('spearman', float('nan')))} | "
                    f"{format_float(stats.get('r2_univariate', float('nan')))} | "
                    f"{format_float(stats.get('auc_success', float('nan')))} | "
                    f"{format_float(stats.get('mean', float('nan')))} | "
                    f"{format_float(stats.get('std', float('nan')))} | "
                    f"{format_float(stats.get('min', float('nan')))} | "
                    f"{format_float(stats.get('max', float('nan')))} |\n"
                )
            f.write("\nSingle-feature selection: choose max feature within each instruction group.\n")
            f.write("| Feature | Mean selected IoU | SSR | Oracle mean IoU | Oracle gap | Oracle pick rate |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for name in FEATURE_NAMES:
                stats = summary["selection"].get(name, {})
                f.write(
                    f"| {name} | {format_float(stats.get('mean_selected_iou', float('nan')))} | "
                    f"{format_float(stats.get('ssr', float('nan')))} | "
                    f"{format_float(stats.get('oracle_mean_iou', float('nan')))} | "
                    f"{format_float(stats.get('oracle_gap', float('nan')))} | "
                    f"{format_float(stats.get('oracle_pick_rate', float('nan')))} |\n"
                )
            linear = summary["linear"]
            f.write("\nLinear target fit with z-scored features:\n")
            f.write(f"full_r2={format_float(linear.get('r2', float('nan')))}\n")
            f.write("| Feature | Standardized coef | R2 without feature | Delta R2 |\n")
            f.write("|---|---:|---:|---:|\n")
            for name in FEATURE_NAMES:
                coef = linear.get("standardized_coefficients", {}).get(name, float("nan"))
                loo = linear.get("leave_one_out", {}).get(name, {})
                f.write(
                    f"| {name} | {format_float(coef)} | "
                    f"{format_float(loo.get('r2_without', float('nan')))} | "
                    f"{format_float(loo.get('delta_r2', float('nan')))} |\n"
                )
            f.write("\n")

        f.write("Skipped samples:\n")
        for split_name in ["easy", "hard"]:
            f.write(f"{split_name}: {len(skipped_by_split.get(split_name, []))}\n")
            for path, err in skipped_by_split.get(split_name, [])[:5]:
                f.write(f"  {path}: {err}\n")
        f.write("\nFeature cache stats:\n")
        for split_name, stats in cache_stats_by_split.items():
            f.write(
                f"{split_name}: hits={stats.get('hits', 0)} misses={stats.get('misses', 0)} "
                f"writes={stats.get('writes', 0)}\n"
            )

    if args.save_feature_rows:
        csv_path = Path(args.output_dir) / f"vlpart_topk_feature_rows_{timestamp}.csv"
        write_feature_rows_csv(csv_path, all_rows)
        print(f"Feature rows CSV: {csv_path}")
    return str(result_file)


def analyze_features(args: argparse.Namespace) -> None:
    demo_ref = {"demo": None}
    samples_by_split = load_instruction_samples(args)
    pred_cache = PredictionCache(args.prediction_cache_size)
    rows_by_split = {"easy": [], "hard": []}
    skipped_by_split = {"easy": [], "hard": []}
    cache_stats_by_split = {}

    for split_name in ["easy", "hard"]:
        if not samples_by_split[split_name]:
            continue
        rows, skipped, cache_stats = collect_feature_rows_for_split(
            split_name,
            samples_by_split[split_name],
            demo_ref,
            args,
            pred_cache,
        )
        rows_by_split[split_name] = rows
        skipped_by_split[split_name] = skipped
        cache_stats_by_split[split_name] = cache_stats

    result_file = write_feature_analysis_results(
        args,
        rows_by_split,
        skipped_by_split,
        cache_stats_by_split,
    )
    all_summary = summarize_feature_rows(rows_by_split["easy"] + rows_by_split["hard"], args)
    print(f"Feature correlation file: {result_file}")
    print(
        f"Candidates={all_summary['num_candidates']} groups={all_summary['num_groups']} "
        f"target_mean={all_summary['target_mean']:.4f}"
    )
    for name in FEATURE_NAMES:
        stats = all_summary["features"].get(name, {})
        select = all_summary["selection"].get(name, {})
        print(
            f"{name}: pearson={format_float(stats.get('pearson', float('nan')))} "
            f"spearman={format_float(stats.get('spearman', float('nan')))} "
            f"auc={format_float(stats.get('auc_success', float('nan')))} "
            f"select_iou={format_float(select.get('mean_selected_iou', float('nan')))}"
        )



def formula_alpha_sweep_values(args: argparse.Namespace) -> List[float]:
    start = float(args.formula_alpha_start)
    end = float(args.formula_alpha_end)
    step = float(args.formula_alpha_step)
    if step <= 0:
        raise ValueError("--formula_alpha_step must be > 0")
    if end < start:
        raise ValueError("--formula_alpha_end must be >= --formula_alpha_start")
    values = []
    current = start
    while current <= end + step * 0.5:
        alpha = min(max(current, 0.0), 1.0)
        alpha = round(alpha, 10)
        if not values or abs(values[-1] - alpha) > 1e-8:
            values.append(alpha)
        current += step
    return values


def formula_scores_np(features: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    pred_similarity = arr[:, FEATURE_NAMES.index("pred_similarity")]
    pred_iou = arr[:, FEATURE_NAMES.index("pred_iou")]
    vlpart_score = arr[:, FEATURE_NAMES.index("vlpart_score")]
    return (float(alpha) * pred_similarity + (1.0 - float(alpha)) * pred_iou) * vlpart_score


def load_formula_sweep_groups_for_split(
    split_name: str,
    samples: List[Dict],
    demo_ref: Dict,
    args: argparse.Namespace,
    pred_cache: PredictionCache,
) -> Tuple[List[Tuple[int, Dict, Dict]], List[Tuple[str, str]], Dict[str, int]]:
    records = []
    skipped = []
    cache_stats = {"hits": 0, "misses": 0, "writes": 0}
    for ordinal, sample in enumerate(tqdm(samples, desc=f"Load formula sweep {split_name}"), start=1):
        try:
            group = load_or_extract_candidate_group(
                sample,
                demo_ref,
                args,
                pred_cache,
                keep_masks=False,
                split_name=split_name,
                cache_stats=cache_stats,
            )
            records.append((ordinal, sample, group))
        except Exception as exc:
            skipped.append((sample.get("image_path", "unknown"), repr(exc)))
            if len(skipped) <= 5 or args.debug:
                print(f"  [{split_name} Error] {sample.get('image_path', 'unknown')}: {exc}")
    print(
        f"  [{split_name} feature cache] hits={cache_stats['hits']} misses={cache_stats['misses']} "
        f"writes={cache_stats['writes']} dir={args.candidate_feature_cache_dir}",
        flush=True,
    )
    return records, skipped, cache_stats


def evaluate_formula_alpha_records(
    split_name: str,
    records: List[Tuple[int, Dict, Dict]],
    alpha: float,
    args: argparse.Namespace,
) -> List[Dict]:
    results = []
    for _ordinal, sample, group in records:
        scores = formula_scores_np(group["features"], alpha)
        best_idx = int(np.argmax(scores))
        candidate = group["candidates"][best_idx]
        iou = float(candidate["target"])
        instruction_index = int(sample.get("instruction_index", 0))
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
                "success": iou >= args.ssr_threshold,
                "formula_score": float(scores[best_idx]),
                "alpha": float(alpha),
                "target_quality": iou,
                "llmseg_topk_rank": candidate["llmseg_topk_rank"],
                "llmseg_topk_mask_path": candidate["llmseg_topk_mask_path"],
                "vlpart_instance_index": candidate["vlpart_instance_index"],
                "features": candidate["features"],
                "pred_class": candidate["pred_class"],
                "pred_class_name": candidate["pred_class_name"],
                "pred_score": candidate["vlpart_score"],
                "num_candidates": len(group["candidates"]),
                "pred_mask_path": "",
                "vis_path": "",
            }
        )
    return results


def write_formula_alpha_sweep_results(
    args: argparse.Namespace,
    rows: List[Dict],
    skipped_by_split: Dict[str, List[Tuple[str, str]]],
    cache_stats_by_split: Dict[str, Dict[str, int]],
) -> Tuple[str, str]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = str(Path(args.output_dir) / f"vlpart_topk_formula_alpha_sweep_{timestamp}.txt")
    csv_file = str(Path(args.output_dir) / f"vlpart_topk_formula_alpha_sweep_{timestamp}.csv")

    best_iou = max(rows, key=lambda item: item["avg_ic_iou"]) if rows else None
    best_ssr = max(rows, key=lambda item: item["avg_ssr"]) if rows else None

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  VLPart Top-K Formula Alpha Sweep Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Formula: (alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score\n")
        f.write("Training: none; alpha is fixed for each sweep row.\n")
        f.write(f"LLMSeg top-K masks: {args.llmseg_topk_masks_dir}\n")
        f.write(f"Top-K K: {args.topk_mask_k}\n")
        f.write(f"SSR threshold: {args.ssr_threshold}\n")
        f.write(f"Feature cache dir: {args.candidate_feature_cache_dir}\n\n")
        if best_iou is not None:
            f.write(
                f"Best Avg IC-IoU: alpha={best_iou['alpha']:.1f}, "
                f"Avg IC-IoU={best_iou['avg_ic_iou']:.4f}, Avg SSR={best_iou['avg_ssr']:.4f}\n"
            )
        if best_ssr is not None:
            f.write(
                f"Best Avg SSR: alpha={best_ssr['alpha']:.1f}, "
                f"Avg IC-IoU={best_ssr['avg_ic_iou']:.4f}, Avg SSR={best_ssr['avg_ssr']:.4f}\n"
            )
        f.write("\n")
        f.write(
            "| Alpha | IC-IoU Easy | IC-IoU Hard | IC-IoU Avg | "
            f"SSR@{args.ssr_threshold:.1f} Easy | SSR@{args.ssr_threshold:.1f} Hard | "
            f"SSR@{args.ssr_threshold:.1f} Avg | Count |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['alpha']:.1f} | {row['easy_ic_iou']:.4f} | "
                f"{row['hard_ic_iou']:.4f} | {row['avg_ic_iou']:.4f} | "
                f"{row['easy_ssr']:.4f} | {row['hard_ssr']:.4f} | "
                f"{row['avg_ssr']:.4f} | {row['count']} |\n"
            )
        f.write("\nSkipped samples:\n")
        for split_name in ["easy", "hard"]:
            f.write(f"{split_name}: {len(skipped_by_split.get(split_name, []))}\n")
            for path, err in skipped_by_split.get(split_name, [])[:5]:
                f.write(f"  {path}: {err}\n")
        f.write("\nFeature cache stats:\n")
        for split_name, stats in cache_stats_by_split.items():
            f.write(
                f"{split_name}: hits={stats.get('hits', 0)} misses={stats.get('misses', 0)} "
                f"writes={stats.get('writes', 0)}\n"
            )

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "alpha",
            "easy_ic_iou",
            "hard_ic_iou",
            "avg_ic_iou",
            "easy_ssr",
            "hard_ssr",
            "avg_ssr",
            "count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return result_file, csv_file


def test_formula_alpha_sweep(args: argparse.Namespace) -> None:
    if args.save_vis or args.save_pred_masks:
        raise ValueError("Formula alpha sweep uses numeric feature cache and does not support visualization outputs")
    if not args.skip_icr:
        print("  [formula alpha sweep] forcing --skip_icr because numeric feature cache does not store masks")
        args.skip_icr = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    samples_by_split = load_instruction_samples(args)
    pred_cache = PredictionCache(args.prediction_cache_size)
    demo_ref = {"demo": None}
    records_by_split = {"easy": [], "hard": []}
    skipped_by_split = {"easy": [], "hard": []}
    cache_stats_by_split = {}

    for split_name in ["easy", "hard"]:
        if not samples_by_split[split_name]:
            continue
        records, skipped, cache_stats = load_formula_sweep_groups_for_split(
            split_name,
            samples_by_split[split_name],
            demo_ref,
            args,
            pred_cache,
        )
        records_by_split[split_name] = records
        skipped_by_split[split_name] = skipped
        cache_stats_by_split[split_name] = cache_stats

    rows = []
    for alpha in formula_alpha_sweep_values(args):
        results = {
            "easy": evaluate_formula_alpha_records("easy", records_by_split["easy"], alpha, args),
            "hard": evaluate_formula_alpha_records("hard", records_by_split["hard"], alpha, args),
        }
        summary = compute_split_summary(results["easy"], results["hard"], args.ssr_threshold)
        row = {
            "alpha": float(alpha),
            "easy_ic_iou": summary["easy"]["ic_iou"],
            "hard_ic_iou": summary["hard"]["ic_iou"],
            "avg_ic_iou": summary["avg_ic_iou"],
            "easy_ssr": summary["easy"]["ssr"],
            "hard_ssr": summary["hard"]["ssr"],
            "avg_ssr": summary["avg_ssr"],
            "count": summary["total_count"],
        }
        rows.append(row)
        print(
            f"alpha={alpha:.1f} "
            f"IC-IoU easy={row['easy_ic_iou']:.4f} hard={row['hard_ic_iou']:.4f} avg={row['avg_ic_iou']:.4f} "
            f"SSR easy={row['easy_ssr']:.4f} hard={row['hard_ssr']:.4f} avg={row['avg_ssr']:.4f}"
        )

    result_file, csv_file = write_formula_alpha_sweep_results(
        args,
        rows,
        skipped_by_split,
        cache_stats_by_split,
    )
    print(f"Formula alpha sweep result file: {result_file}")
    print(f"Formula alpha sweep CSV: {csv_file}")

def test(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
    model, mean, std, feature_indices, ranker_feature_names = load_reranker_checkpoint(args, device)
    print(f"  [ranker features] {ranker_feature_names}")
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
            feature_indices,
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
    elif args.mode == "train_formula_scorer":
        train_formula_scorer(args)
    elif args.mode == "extract_train_cache":
        extract_feature_cache(args)
    elif args.mode == "analyze_features":
        analyze_features(args)
    elif args.mode == "test_formula_alpha_sweep":
        test_formula_alpha_sweep(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
