import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


VLPART_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLPART_ROOT))

from vlpart.modeling.text_encoder.text_encoder import build_text_encoder


def parse_args():
    parser = argparse.ArgumentParser("Build CLIP text embeddings for VIGOR vocabulary")
    parser.add_argument(
        "--mapping",
        default="configs/vigor/vigor_easy_object_to_vocabulary.json",
        help="Path to object-to-vocabulary mapping JSON.",
    )
    parser.add_argument(
        "--output",
        default="datasets/metadata/vigor_easy_clip_RN50_a+cname.npy",
        help="Output .npy path. Shape is C x D.",
    )
    parser.add_argument(
        "--prompt",
        default="a ",
        help="Prompt prefix used by VLPart built-in classifiers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mapping_path = Path(args.mapping)
    if not mapping_path.is_absolute():
        mapping_path = VLPART_ROOT / mapping_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = VLPART_ROOT / output_path

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    vocabulary = mapping["vocabulary"]
    texts = [args.prompt + name.lower().replace(":", " ") for name in vocabulary]

    text_encoder = build_text_encoder(pretrain=True, visual_type="RN50")
    text_encoder.eval()
    with torch.no_grad():
        embeddings = text_encoder(texts).detach().cpu().numpy().astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"saved {output_path}")
    print(f"shape {embeddings.shape}")
    print("vocabulary:")
    for idx, name in enumerate(vocabulary):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
