import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


VLPART_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLPART_ROOT))
sys.path.insert(0, str(VLPART_ROOT / "demo"))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from predictor import get_clip_embeddings, reset_cls_test  # noqa: E402
from test_vigor_vlpart import choose_single_mask  # noqa: E402
from vlpart.config import add_vlpart_config  # noqa: E402
from vlpart.config_object_prior import add_object_prior_config  # noqa: E402

import vlpart.modeling.meta_arch.vlm_rcnn_object_prior  # noqa: F401,E402
import vlpart.modeling.roi_heads.object_prior_roi_heads  # noqa: F401,E402


def parse_args():
    parser = argparse.ArgumentParser("Quick checks for VIGOR GT object-prior experiment")
    parser.add_argument(
        "--config-file",
        default="configs/vigor/swinbase_vigor_stage1_gt_object_prior.yaml",
    )
    parser.add_argument(
        "--weights",
        default="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_easy_stage1_bs16_lr4e-5/model_final.pth",
    )
    parser.add_argument(
        "--data_dir",
        default="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new",
    )
    parser.add_argument("--easy_json_file", default="open_vocab_grasp_easy_object_mix.json")
    parser.add_argument("--hard_json_file", default="open_vocab_grasp_hard_object_mix.json")
    parser.add_argument("--output_dir", default="vlpart_vigor_outputs/gt_object_prior_quick_check")
    parser.add_argument(
        "--custom_vocabulary",
        default="cylindrical side surface,hexagonal side face,flat side surface,whole object",
    )
    parser.add_argument("--samples_per_split", type=int, default=20)
    parser.add_argument("--prior_score_samples", type=int, default=8)
    parser.add_argument("--topk_proposals", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser.parse_args()


def resolve_repo_path(path: str) -> str:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(VLPART_ROOT / expanded)


def split_path_list(path_text: str) -> List[str]:
    return [item.strip() for item in str(path_text).split(",") if item.strip()]


def resolve_data_path(data_root: str, split: str, path: str) -> str:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(Path(data_root) / split / path)


def load_split_samples(data_root: str, split: str, json_name: str) -> List[Dict]:
    json_file = Path(data_root) / split / json_name
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

    samples = []
    for idx, sample in enumerate(raw_samples):
        scene = str(sample.get("scene", "")).strip()
        gt_mask_paths = split_path_list(sample.get("gt_mask_path", ""))
        object_prior_paths = split_path_list(sample.get("gt_object_mask_path", ""))
        if not scene or not gt_mask_paths or not object_prior_paths:
            continue
        scene_path = resolve_data_path(data_root, split, f"{scene}.png")
        gt_mask_path = resolve_data_path(data_root, split, gt_mask_paths[0])
        object_prior_path = resolve_data_path(data_root, split, object_prior_paths[0])
        if not os.path.exists(scene_path) or not os.path.exists(gt_mask_path) or not os.path.exists(object_prior_path):
            continue
        samples.append(
            {
                "index": idx,
                "scene": scene,
                "image_path": scene_path,
                "gt_mask_path": gt_mask_path,
                "object_prior_mask_path": object_prior_path,
                "gt_object": sample.get("gt_object", sample.get("object", "")),
                "gt_object_first": split_path_list(sample.get("gt_object", ""))[0]
                if split_path_list(sample.get("gt_object", ""))
                else sample.get("gt_object", sample.get("object", "")),
                "instructions": sample.get("instructions", []),
            }
        )
    return samples


def read_image_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    h, w = image_shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(image_rgb: np.ndarray, fg: np.ndarray, color: Tuple[int, int, int], alpha=0.45):
    if fg.shape != image_rgb.shape[:2]:
        h, w = image_rgb.shape[:2]
        fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = image_rgb.copy()
    overlay[fg] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image_rgb, 1.0 - alpha, 0)


def save_rgb(path: Path, image_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def setup_cfg(args):
    cfg = get_cfg()
    add_vlpart_config(cfg)
    add_object_prior_config(cfg)
    cfg.merge_from_file(resolve_repo_path(args.config_file))
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = resolve_repo_path(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.OBJECT_PRIOR.ENABLED = True
    cfg.MODEL.OBJECT_PRIOR.MODE = "gt"
    cfg.freeze()
    return cfg


class QuickCheckPredictor:
    def __init__(self, cfg, vocabulary: List[str]):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.metadata = MetadataCatalog.get("__vigor_gt_prior_quick_check")
        self.metadata.thing_classes = vocabulary
        reset_cls_test(self.model, get_clip_embeddings(vocabulary))

    def _prepare_inputs(self, image_rgb: np.ndarray, object_prior_mask: np.ndarray):
        original = image_rgb
        if self.input_format == "BGR":
            original = original[:, :, ::-1]
        height, width = original.shape[:2]
        transform = self.aug.get_transform(original)
        image = transform.apply_image(original)
        prior = transform.apply_segmentation(object_prior_mask)
        object_prior = (prior == 0).astype("float32")[None, :, :]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        prior_tensor = torch.as_tensor(np.ascontiguousarray(object_prior))
        return {
            "image": image_tensor,
            "object_prior": prior_tensor,
            "height": height,
            "width": width,
        }

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, object_prior_mask: np.ndarray):
        inputs = self._prepare_inputs(image_rgb, object_prior_mask)
        return self.model([inputs])[0]

    @torch.no_grad()
    def collect_prior_scores(self, image_rgb: np.ndarray, object_prior_mask: np.ndarray):
        inputs = self._prepare_inputs(image_rgb, object_prior_mask)
        batched_inputs = [inputs]
        model = self.model
        images = model.preprocess_image(batched_inputs)
        object_priors = model.preprocess_object_prior(batched_inputs)
        features = model.backbone(images.tensor)
        features = model.apply_object_prior_to_fpn(features, object_priors)
        proposals, _ = model.proposal_generator(images, features)
        proposals = model.apply_object_prior_to_proposals(proposals, object_priors)
        if not proposals or not proposals[0].has("object_prior_scores"):
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        scores = proposals[0].object_prior_scores.detach().float().cpu().numpy()
        top_scores = scores[: min(len(scores), 20)]
        return scores, top_scores


def save_overlay_set(
    predictor: QuickCheckPredictor,
    sample: Dict,
    split_name: str,
    ordinal: int,
    output_root: Path,
    run_model: bool,
):
    image_rgb = read_image_rgb(sample["image_path"])
    prior_mask = read_mask(sample["object_prior_mask_path"], image_rgb.shape[:2])
    affordance_mask = read_mask(sample["gt_mask_path"], image_rgb.shape[:2])

    stem = f"{split_name}_{ordinal:03d}_scene{sample['scene']}_idx{sample['index']}"
    split_dir = output_root / "overlays" / split_name
    save_rgb(split_dir / f"{stem}_full_rgb.png", image_rgb)
    save_rgb(
        split_dir / f"{stem}_gt_object_prior_overlay.png",
        overlay_mask(image_rgb, prior_mask == 0, (255, 0, 0)),
    )
    save_rgb(
        split_dir / f"{stem}_gt_affordance_overlay.png",
        overlay_mask(image_rgb, affordance_mask == 0, (0, 255, 0)),
    )

    if run_model and predictor is not None:
        predictions = predictor.predict(image_rgb, prior_mask)
        pred_fg, pred_info = choose_single_mask(
            predictions=predictions,
            metadata=predictor.metadata,
            image_shape=image_rgb.shape[:2],
            mask_selection="top1",
        )
        pred_overlay = overlay_mask(image_rgb, pred_fg, (0, 0, 255))
        title = f"{pred_info.get('pred_class_name', '')} {pred_info.get('pred_score', 0.0):.3f}"
        cv2.putText(
            pred_overlay,
            title[:120],
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        save_rgb(split_dir / f"{stem}_predicted_mask_overlay.png", pred_overlay)


def print_prior_score_report(
    predictor: QuickCheckPredictor,
    samples: List[Tuple[str, Dict]],
    topk: int,
):
    print("\n" + "=" * 80)
    print("Prior-score proposal distribution")
    print("=" * 80)
    for split_name, sample in samples:
        image_rgb = read_image_rgb(sample["image_path"])
        prior_mask = read_mask(sample["object_prior_mask_path"], image_rgb.shape[:2])
        scores, top_scores = predictor.collect_prior_scores(image_rgb, prior_mask)
        if scores.size == 0:
            print(f"[{split_name}] scene={sample['scene']} idx={sample['index']}: no prior scores")
            continue
        print(
            f"[{split_name}] scene={sample['scene']} idx={sample['index']} "
            f"object={sample['gt_object_first']} proposals={scores.size} "
            f"min={scores.min():.4f} mean={scores.mean():.4f} max={scores.max():.4f}"
        )
        nonzero_ratio = float((scores > 1e-6).mean())
        top_rank_text = ", ".join(f"{x:.4f}" for x in top_scores[:topk])
        top_prior = np.sort(scores)[::-1][:topk]
        top_prior_text = ", ".join(f"{x:.4f}" for x in top_prior)
        print(f"  prior_score > 0 ratio: {nonzero_ratio:.4f}")
        print(f"  first {min(topk, len(top_scores))} proposals after objectness/prior rerank: {top_rank_text}")
        print(f"  highest {min(topk, len(top_prior))} prior_score among all proposals: {top_prior_text}")


def main():
    args = parse_args()
    setup_logger(name="fvcore")
    setup_logger()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_easy = load_split_samples(args.data_dir, "train", args.easy_json_file)
    train_hard = load_split_samples(args.data_dir, "train", args.hard_json_file)
    easy_selected = random.sample(train_easy, min(args.samples_per_split, len(train_easy)))
    hard_selected = random.sample(train_hard, min(args.samples_per_split, len(train_hard)))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    vocabulary = [x.strip() for x in args.custom_vocabulary.split(",") if x.strip()]

    predictor = None
    if not args.skip_model:
        cfg = setup_cfg(args)
        predictor = QuickCheckPredictor(cfg, vocabulary)

    for i, sample in enumerate(easy_selected, start=1):
        save_overlay_set(predictor, sample, "easy", i, output_root, run_model=not args.skip_model)
    for i, sample in enumerate(hard_selected, start=1):
        save_overlay_set(predictor, sample, "hard", i, output_root, run_model=not args.skip_model)

    print(f"Saved overlays to: {output_root / 'overlays'}")

    if not args.skip_model:
        score_samples = []
        combined = [("easy", x) for x in easy_selected] + [("hard", x) for x in hard_selected]
        random.shuffle(combined)
        score_samples = combined[: args.prior_score_samples]
        print_prior_score_report(predictor, score_samples, args.topk_proposals)
    else:
        print("Skipped model predictions and prior-score report (--skip_model).")


if __name__ == "__main__":
    main()
