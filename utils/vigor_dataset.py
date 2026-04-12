import glob
import json
import os
import random
import re
from typing import List, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import pycocotools.mask as mask_util

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)
from .sam_mask_reader_png import SAM_Mask_Reader_PNG
from .utils import compute_all_iou, compute_all_iop

DEFAULT_AUTODL_TMP_DIR = os.environ.get("AUTODL_TMP_DIR", os.path.join("..", "root", "autodl-tmp"))
DEFAULT_VIGOR_DATA_DIR = os.environ.get("VIGOR_DATA_DIR", os.path.join(DEFAULT_AUTODL_TMP_DIR, "VIGOR-100K_new"))


class VIGORDataset(torch.utils.data.Dataset):
    """
    VIGOR-100K数据集
    支持从VIGOR-100K格式的JSON文件加载数据
    使用instructions字段中的自然语言指令
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        json_path: str,  # VIGOR-100K格式的JSON文件路径（如open_vocab_grasp_easy.json）
        tokenizer,
        vision_tower,
        precision: str = "bf16",
        image_size: int = 896,
        data_base_dir: str = DEFAULT_VIGOR_DATA_DIR,  # VIGOR-100K根目录
        split: str = "train",  # train/test/unseen
        sam_mask_helper: SAM_Mask_Reader_PNG = None,  # SAM候选mask helper（可选）
        max_samples: int = None,  # 最多使用多少个样本（None表示使用全部）
        is_train: bool = True,
        samples: List[Dict] = None,  # 允许外部直接传入已构建的samples
        debug_meta: bool = False,
    ):
        self.json_path = json_path
        self.data_base_dir = data_base_dir
        self.split = split
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        import os
        local_files_only = os.path.exists(vision_tower) and os.path.isdir(vision_tower)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            vision_tower,
            local_files_only=local_files_only
        )
        
        self.max_samples = max_samples
        self.is_train = is_train
        self.debug_meta = bool(debug_meta)
        self.sam_mask_helper = sam_mask_helper
        
        # 加载所有样本（支持外部传入切分后的samples）
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self.load_all_samples()
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        
        print(f"Loaded {len(self.samples)} samples from {json_path} (split={split})")
    
    def load_all_samples(self):
        """
        从VIGOR-100K格式的JSON文件加载样本
        """
        samples = []
        
        # 读取JSON文件
        with open(self.json_path, "r") as f:
            data = json.load(f)
        
        # VIGOR-100K格式：{"num_samples": N, "samples": [...]}
        if isinstance(data, dict) and "samples" in data:
            annotations = data["samples"]
        elif isinstance(data, list):
            annotations = data
        else:
            raise ValueError(f"Unexpected JSON format in {self.json_path}")
        
        # 限制样本数量
        if self.max_samples is not None:
            annotations = annotations[:self.max_samples]
        
        # 处理每个样本
        for ann in annotations:
            scene = ann.get('scene', '')
            object_name = ann.get('object', 'object')
            gt_mask_path_rel = ann.get('gt_mask_path', '')  # 相对路径，如 "masks/1_part_00_outer_seat.usd.png"
            instructions = ann.get('instructions', [])  # 自然语言指令列表
            
            if not gt_mask_path_rel:
                continue
            
            # 从mask文件名提取图像编号
            # 格式：masks/1_part_00_outer_seat.usd.png -> 图像是 1.png
            mask_filename = os.path.basename(gt_mask_path_rel)
            match = re.match(r'(\d+)_', mask_filename)
            if not match:
                print(f"Warning: Cannot extract image number from mask filename: {mask_filename}")
                continue
            
            img_num = match.group(1)
            img_name = f"{img_num}.png"
            
            # 构建完整路径
            # 图像路径：{data_base_dir}/{split}/{img_name}
            image_path = os.path.join(self.data_base_dir, self.split, img_name)
            
            # GT mask路径：{data_base_dir}/{split}/{gt_mask_path_rel}
            gt_mask_path = os.path.join(self.data_base_dir, self.split, gt_mask_path_rel)
            
            # 从instructions中随机选择一条指令（如果没有则使用默认格式）
            if instructions and len(instructions) > 0:
                instruction = random.choice(instructions)
            else:
                instruction = f"segment the {object_name}"
            
            # 加上<image>前缀
            question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
            
            samples.append({
                'image_path': image_path,
                'gt_mask_path': gt_mask_path,
                'question': question,
                'instruction': instruction,  # 保存原始指令（不含<image>）
                'object': object_name,
                'scene': scene,
                'img_name': img_name,
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        gt_mask_path = sample['gt_mask_path']
        img_name = sample['img_name']
        
        # 读取原图
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        
        # 预处理图像用于CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        # 读取GT mask
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            # 如果GT mask不存在，创建一个空mask
            print(f"Warning: GT mask not found: {gt_mask_path}, using empty mask")
            # 约定：0=掩码(前景)，1=背景；空mask应为全背景
            gt_mask = np.ones(ori_size, dtype=np.uint8)
        else:
            # ✅ 重要：黑色（值为0）是掩码区域，白色（值为255）是背景
            # 统一约定：0=掩码(前景)，1=背景（与验证/可视化一致）
            gt_mask = (gt_mask > 0).astype(np.float32)  # 黑0->0(前景)，白>0->1(背景)
        
        sampled_masks = [gt_mask]
        # 使用已经包含<image>前缀的question
        sampled_sents = [sample['question']]
        is_sentence = True  # VIGOR的指令都是句子
        
        # 获取SAM候选segments（如果有）
        if self.sam_mask_helper is not None:
            segs_dict = self.sam_mask_helper.extract_sam_segs(img_name)
        else:
            # 如果没有SAM helper，创建空的候选masks
            h, w = ori_size
            segs_origin = np.zeros((h, w, 1), dtype=np.uint8)
            segs_square = np.zeros((max(h, w), max(h, w), 1), dtype=np.uint8)
            segs_dict = {
                "segs_square": segs_square,
                "segs_origin": segs_origin,
                "bbox": [[0, 0, 0, 0]]
            }
        
        segs_square = segs_dict["segs_square"]
        segs_origin = segs_dict["segs_origin"]

        # ✅ 关键：候选 mask 的空间变换必须与原图一致（ResizeLongestSide -> pad 到 896）
        if segs_origin is not None and isinstance(segs_origin, np.ndarray) and segs_origin.ndim == 3:
            K = segs_origin.shape[2]
            segs_resized_list = []
            for k in range(K):
                mask_k = segs_origin[:, :, k]  # (H, W), 值为 0/1
                mask_k_uint8 = (mask_k * 255).astype(np.uint8)  # 0->0(前景), 1->255(背景)
                mask_k_resized = self.transform.apply_image(mask_k_uint8)  # (H', W')
                mask_k_resized = (mask_k_resized.astype(np.float32) / 255.0)  # 0/1（可能有插值产生的中间值）
                segs_resized_list.append(mask_k_resized)

            if len(segs_resized_list) > 0:
                resized_h, resized_w = segs_resized_list[0].shape
                segs_resized = np.stack(segs_resized_list, axis=2)  # (H', W', K)
            else:
                segs_resized = segs_origin

            h2, w2, _ = segs_resized.shape
            padh = self.image_size - h2
            padw = self.image_size - w2
            if padh < 0 or padw < 0:
                raise ValueError(f"segs_resized larger than image_size={self.image_size}: {(h2, w2)}")
            segs_square = np.pad(
                segs_resized,
                ((0, padh), (0, padw), (0, 0)),
                mode="constant",
                constant_values=1,  # 背景
            )
        else:
            # fallback：保持原逻辑
            segs_square = segs_dict["segs_square"]

        # 转换为tensor并插值到 256x256（作为 mask_pooling 的 weight_maps）
        segs_square_hwk_after_pad = segs_square.shape if isinstance(segs_square, np.ndarray) else None  # (H, W, K)
        segs_square = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous()  # (K, H, W)
        segs = F.interpolate(
            segs_square.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (K, 256, 256)
        
        # 计算IoU和IoP
        # ✅ 注意：本数据集内部约定为 0=掩码(前景)，1=背景
        # compute_iou/compute_iop 期望前景为 1，所以这里临时转换成前景=1的二值图再计算
        gt_fg = (gt_mask == 0).astype(np.uint8)
        segs_fg = (segs_origin == 0).astype(np.uint8)
        sampled_ious = [
            compute_all_iou(segs_fg, gt_fg) for _ in sampled_masks
        ]
        sampled_iops = [
            compute_all_iop(segs_fg, gt_fg) for _ in sampled_masks
        ]
        
        # 设置精度类型
        precision_type = torch.float32
        if self.precision == "fp16":
            precision_type = torch.float16
        elif self.precision == "bf16":
            precision_type = torch.bfloat16
        segs = segs.to(precision_type)
        
        # 预处理图像用于SAM
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        
        # 生成对话
        # 注意：sampled_sents已经包含了<image>前缀，所以直接使用
        questions = []
        answers = []
        for text in sampled_sents:
            # 如果text已经包含<image>，直接使用；否则使用模板
            if DEFAULT_IMAGE_TOKEN in text:
                # 已经包含<image>，直接使用
                questions.append(text)
            else:
                # 使用模板（虽然VIGOR应该不会走到这里）
                if is_sentence:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=text))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))
        
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for i in range(len(questions)):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
        
        # 预处理图像tensor
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
        # 堆叠masks和ious
        masks = np.stack(sampled_masks, axis=0)
        ious = np.stack(sampled_ious, axis=0)
        iops = np.stack(sampled_iops, axis=0)
        masks = torch.from_numpy(masks)
        ious = torch.from_numpy(ious)
        iops = torch.from_numpy(iops)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        
        # 候选 mask 路径（与 segs_origin 的 K 维顺序一致）
        candidate_mask_paths = segs_dict.get("mask_paths", [])

        debug_meta = None
        if self.debug_meta:
            try:
                debug_meta = {
                    "idx": int(idx),
                    "image_path": image_path,
                    "raw_image_shape": tuple(cv2.imread(image_path).shape) if os.path.exists(image_path) else None,
                    "ori_size_hw": tuple(ori_size),
                    "image_after_resize_longest_hw": tuple(resize),
                    "image_after_preprocess_chw": tuple(image.shape),
                    "gt_mask_path": gt_mask_path,
                    "gt_mask_shape_hw": tuple(gt_mask.shape) if hasattr(gt_mask, "shape") else None,
                    "segs_origin_shape_hwk": tuple(segs_origin.shape) if isinstance(segs_origin, np.ndarray) else None,
                    "segs_interp_256_shape_khw": tuple(segs.shape),
                    "num_candidate_masks": int(len(candidate_mask_paths)) if isinstance(candidate_mask_paths, list) else None,
                }
            except Exception as e:
                debug_meta = {"idx": int(idx), "error": f"failed to build debug_meta: {repr(e)}"}
        
        return {
            'image_path': image_path,
            'images': image,
            'images_clip': image_clip,
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': questions,
            'sampled_classes': sampled_sents,
            'segs': segs,
            'ious': ious,
            'iops': iops,
            'segs_origin': segs_origin,
            'bbox': segs_dict.get('bbox', []),
            'inference': False,
            'segmentation_paths': gt_mask_path,
            'conversation_list': conversations,
            'origin_segs_list': segs_origin,
            'sam_ious_list': ious.tolist() if isinstance(ious, torch.Tensor) else ious,
            'candidate_mask_paths_list': candidate_mask_paths,
            'debug_meta': debug_meta,
        }
