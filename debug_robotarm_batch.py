import os
import random

import transformers
from torch.utils.data import DataLoader

from model.llava import conversation as conversation_lib
from utils.dataset import collate_fn_new
from utils.robot_arm_dataset import RobotArmDataset
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG


def main():
    raw_pic_base_dir = "/opt/data/private/LLMSeg/dataset/raw_pic"
    gt_mask_base_dir = "/opt/data/private/LLMSeg/dataset/GT_mask"
    sam_candidate_base_dir = "/opt/data/private/LLMSeg/dataset/sam_candidate"
    vision_tower = "/opt/data/private/model/clip-vit-large-patch14"

    json_paths = []
    for view in ["robot_arm_01", "robot_arm_02", "robot_arm_03"]:
        p = os.path.join(gt_mask_base_dir, view, "annotations.json")
        if os.path.exists(p):
            json_paths.append(p)
    print("json_paths:", json_paths)

    tok = transformers.AutoTokenizer.from_pretrained(
        "/opt/data/private/model/LISA_Plus_7b", use_fast=False
    )
    tok.pad_token = tok.unk_token
    # Make sure conversation template matches training (collate_fn_new expects llava_v1 format)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    sam_mask_helpers = {}
    for view in ["robot_arm_01", "robot_arm_02", "robot_arm_03"]:
        d = os.path.join(sam_candidate_base_dir, view)
        if os.path.exists(d):
            sam_mask_helpers[view] = SAM_Mask_Reader_PNG(d)

    ds = RobotArmDataset(
        json_paths=json_paths,
        tokenizer=tok,
        vision_tower=vision_tower,
        precision="bf16",
        image_size=896,
        raw_pic_base_dir=raw_pic_base_dir,
        gt_mask_base_dir=gt_mask_base_dir,
        sam_candidate_base_dir=sam_candidate_base_dir,
        sam_mask_helpers=sam_mask_helpers,
        max_samples_per_view=100,
        is_train=True,
    )

    # pick a random element and collate as a single-batch like training
    idx = random.randrange(len(ds))
    batch = [ds[idx]]
    input_dict = collate_fn_new(
        batch,
        tokenizer=tok,
        conv_type="llava_v1",
        use_mm_start_end=True,
        local_rank=0,
    )

    print("\n=== Random dataset item ===")
    print("idx:", idx)
    print("image_paths[0]:", input_dict["image_paths"][0])
    print("segmentation_paths[0]:", input_dict.get("segmentation_paths", [""])[0])
    cand0 = input_dict.get("candidate_mask_paths_list", [[]])[0]
    print("candidate_mask_paths_list[0] count:", len(cand0))
    for p in cand0[:5]:
        print("  cand:", p, "exists:", os.path.exists(p))

    print("\n=== input_dict tensor shapes ===")
    for k in ["images", "images_clip", "input_ids", "labels", "attention_masks", "offset"]:
        v = input_dict.get(k)
        if v is None:
            continue
        try:
            print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        except Exception:
            print(f"{k}: {type(v)}")

    print("\n=== list payload sizes ===")
    for k in ["conversation_list", "masks_list", "sam_segs_list", "sam_ious_list", "sam_iops_list", "origin_segs_list"]:
        v = input_dict.get(k)
        if v is None:
            continue
        try:
            print(f"{k}: len={len(v)} type0={type(v[0]) if len(v) else None}")
        except Exception:
            print(f"{k}: {type(v)}")


if __name__ == "__main__":
    main()


