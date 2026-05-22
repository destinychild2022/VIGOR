#!/usr/bin/env python
import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np


DEFAULT_JSON = "/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/train/open_vocab_grasp_easy_object_mix.json"
DEFAULT_DATA_ROOT = "/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/train"
DEFAULT_OUTPUT_DIR = "output/VLPart/vigor_swinbase_easy_stage1_gt_plus_nearby_noisy_ratio20_top1_dist150/noisy_visualizations"


def split_paths(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve(root, path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(root) / path


def load_samples(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"] if isinstance(data, dict) and "samples" in data else data


def scene_name(sample):
    scene = str(sample.get("scene", "")).strip()
    if scene:
        return scene
    for key in ("gt_mask_path", "gt_object_mask_path", "gt_object_path"):
        paths = split_paths(sample.get(key, ""))
        if paths:
            match = re.match(r"^(\d+)_", Path(paths[0]).name)
            if match:
                return match.group(1)
    return ""


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def read_fg(mask_path, shape=None):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if shape is not None and mask.shape != tuple(shape):
        h, w = shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask == 0


def mask_info(mask_path, shape=None):
    fg = read_fg(mask_path, shape)
    ys, xs = np.where(fg)
    if len(xs) == 0:
        raise ValueError(f"Empty foreground mask: {mask_path}")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "fg": fg,
        "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
        "center": ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
        "area": int(fg.sum()),
    }


def squared_distance(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def ensure_scene_mask_info(scene, by_scene):
    for item in by_scene.get(scene, []):
        if "center" in item and "bbox" in item:
            continue
        info = mask_info(item["object_mask_path"])
        item["bbox"] = info["bbox"]
        item["center"] = info["center"]


def choose_partner(target, by_scene, rng, nearby_topk, max_center_distance):
    ensure_scene_mask_info(target["scene"], by_scene)
    if "center" not in target:
        return None, None, "missing_target_mask"
    pool = []
    for item in by_scene[target["scene"]]:
        if item["sample_index"] == target["sample_index"]:
            continue
        distance = float(np.sqrt(squared_distance(target["center"], item["center"])))
        pool.append((item, distance))
    if not pool:
        return None, None, "without_partner"
    pool.sort(key=lambda item_and_distance: item_and_distance[1])
    if float(max_center_distance) > 0:
        pool = [item_and_distance for item_and_distance in pool if item_and_distance[1] <= float(max_center_distance)]
        if not pool:
            return None, None, "too_far"
    partner, distance = rng.choice(pool[: max(1, min(int(nearby_topk), len(pool)))])
    return partner, distance, "ok"


def masked_region(scene, fg):
    out = np.zeros_like(scene)
    out[fg] = scene[fg]
    return out


def overlay(scene, masks_and_colors):
    out = scene.copy()
    layer = scene.copy()
    for fg, color in masks_and_colors:
        layer[fg] = color
    return cv2.addWeighted(layer, 0.45, out, 0.55, 0.0)


def draw_box(image, bbox, color, label):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def panel(image, label, width=320):
    h, w = image.shape[:2]
    scale = width / max(w, 1)
    resized = cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    header = np.zeros((32, resized.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, resized])


def hstack_padded(images):
    max_h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    return np.hstack(padded)


def sanitize(value):
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value))
    return value.strip("_") or "item"


def build_selected_pairs(args):
    samples = load_samples(args.json_file)
    candidates = []
    by_scene = {}
    for idx, sample in enumerate(samples):
        scene = scene_name(sample)
        object_mask_paths = split_paths(sample.get("gt_object_mask_path", ""))
        aff_mask_paths = split_paths(sample.get("gt_mask_path", ""))
        object_image_paths = split_paths(sample.get("gt_object_path", ""))
        if not scene or len(object_mask_paths) != 1 or len(aff_mask_paths) != 1 or len(object_image_paths) != 1:
            continue
        item = {
            "sample_index": idx,
            "sample": sample,
            "scene": scene,
            "scene_path": resolve(args.data_root, f"{scene}.png"),
            "object_mask_path": resolve(args.data_root, object_mask_paths[0]),
            "aff_mask_path": resolve(args.data_root, aff_mask_paths[0]),
            "object_image_path": resolve(args.data_root, object_image_paths[0]),
            "gt_object": sample.get("gt_object", sample.get("object", "object")),
        }
        candidates.append(item)
        by_scene.setdefault(scene, []).append(item)

    eligible = [item for item in candidates if len(by_scene.get(item["scene"], [])) > 1]
    rng = random.Random(int(args.random_seed))
    rng.shuffle(eligible)
    target_noisy = int(round(len(samples) * max(float(args.noisy_ratio), 0.0)))
    pairs = []
    skipped_too_far = 0
    skipped_other = 0
    for target in eligible:
        partner, distance, reason = choose_partner(
            target,
            by_scene,
            rng,
            args.nearby_topk,
            args.max_center_distance,
        )
        if partner is None:
            if reason == "too_far":
                skipped_too_far += 1
            else:
                skipped_other += 1
            continue
        pairs.append((target, partner, distance))
        if len(pairs) >= args.num_samples:
            break
    return samples, candidates, eligible, target_noisy, pairs, skipped_too_far, skipped_other


def render_pair(target, partner, distance):
    scene = read_image(target["scene_path"])
    shape = scene.shape[:2]
    target_info = mask_info(target["object_mask_path"], shape)
    partner_info = mask_info(partner["object_mask_path"], shape)
    aff_fg = read_fg(target["aff_mask_path"], shape)
    union_fg = np.logical_or(target_info["fg"], partner_info["fg"])

    scene_panel = overlay(scene, [(target_info["fg"], (0, 220, 0)), (partner_info["fg"], (0, 165, 255))])
    draw_box(scene_panel, target_info["bbox"], (0, 255, 0), "target")
    draw_box(scene_panel, partner_info["bbox"], (0, 165, 255), "nearby")

    target_region = masked_region(scene, target_info["fg"])
    partner_region = masked_region(scene, partner_info["fg"])
    combined = masked_region(scene, union_fg)
    supervision = overlay(scene, [(aff_fg, (0, 0, 255))])
    draw_box(supervision, target_info["bbox"], (0, 255, 0), "same GT")

    labels = [
        "scene {}: target+nearby".format(target["scene"]),
        "target {}".format(target["gt_object"]),
        "nearby {}".format(partner["gt_object"]),
        "combined input dist {:.1f}".format(distance),
        "unchanged supervision",
    ]
    return hstack_padded([
        panel(scene_panel, labels[0]),
        panel(target_region, labels[1]),
        panel(partner_region, labels[2]),
        panel(combined, labels[3]),
        panel(supervision, labels[4]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", default=DEFAULT_JSON)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--noisy_ratio", type=float, default=0.2)
    parser.add_argument("--nearby_topk", type=int, default=1)
    parser.add_argument("--max_center_distance", type=float, default=150.0)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, candidates, eligible, target_noisy, pairs, skipped_too_far, skipped_other = build_selected_pairs(args)

    summary_lines = [
        f"json_file: {args.json_file}",
        f"clean_samples: {len(samples)}",
        f"requested_noisy: {target_noisy}",
        f"eligible_targets: {len(eligible)}",
        f"visualized: {len(pairs)}",
        f"nearby_topk: {args.nearby_topk}",
        f"max_center_distance: {args.max_center_distance}",
        f"skipped_too_far_before_visualized: {skipped_too_far}",
        f"skipped_other_before_visualized: {skipped_other}",
        f"random_seed: {args.random_seed}",
        "",
    ]

    for idx, (target, partner, distance) in enumerate(pairs):
        image = render_pair(target, partner, distance)
        name = "{:03d}_scene{}_target{}_nearby{}.png".format(
            idx,
            sanitize(target["scene"]),
            sanitize(target["gt_object"]),
            sanitize(partner["gt_object"]),
        )
        out_path = args.output_dir / name
        cv2.imwrite(str(out_path), image)
        summary_lines.append(
            "{:03d}: target_sample={} target={} nearby_sample={} nearby={} scene={} {} file={}".format(
                idx,
                target["sample_index"],
                target["gt_object"],
                partner["sample_index"],
                partner["gt_object"],
                target["scene"],
                "dist={:.1f}".format(distance),
                name,
            )
        )

    (args.output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Wrote {} visualizations to {}".format(len(pairs), args.output_dir))


if __name__ == "__main__":
    main()
