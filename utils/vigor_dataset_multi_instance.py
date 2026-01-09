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


class VIGORDatasetMultiInstance(torch.utils.data.Dataset):
    """
    VIGOR-100K数据集（多实例版本）
    支持从VIGOR-100K格式的JSON文件加载数据
    每个样本返回3个训练实例，每个实例对应一条指令
    支持easy和hard混合训练
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        json_path: str,  # VIGOR-100K格式的JSON文件路径
        tokenizer,
        vision_tower,
        precision: str = "bf16",
        image_size: int = 896,
        data_base_dir: str = "/opt/data/private/LLMSeg/dataset/VIGOR-100K",
        split: str = "train",
        sam_mask_helper: SAM_Mask_Reader_PNG = None,
        max_samples: int = None,
        is_train: bool = True,
        samples: List[Dict] = None,
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
        
        # 加载所有样本
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
        
        with open(self.json_path, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "samples" in data:
            annotations = data["samples"]
        elif isinstance(data, list):
            annotations = data
        else:
            raise ValueError(f"Unexpected JSON format in {self.json_path}")
        
        if self.max_samples is not None:
            annotations = annotations[:self.max_samples]
        
        for ann in annotations:
            scene = ann.get('scene', '')
            object_name = ann.get('object', 'object')
            gt_mask_path_rel = ann.get('gt_mask_path', '')
            instructions = ann.get('instructions', [])
            
            if not gt_mask_path_rel:
                continue
            
            mask_filename = os.path.basename(gt_mask_path_rel)
            match = re.match(r'(\d+)_', mask_filename)
            if not match:
                print(f"Warning: Cannot extract image number from mask filename: {mask_filename}")
                continue
            
            img_num = match.group(1)
            img_name = f"{img_num}.png"
            
            image_path = os.path.join(self.data_base_dir, self.split, img_name)
            gt_mask_path = os.path.join(self.data_base_dir, self.split, gt_mask_path_rel)
            
            samples.append({
                'image_path': image_path,
                'gt_mask_path': gt_mask_path,
                'instructions': instructions,
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
            print(f"Warning: GT mask not found: {gt_mask_path}, using empty mask")
            gt_mask = np.ones(ori_size, dtype=np.uint8)
        else:
            gt_mask = (gt_mask > 0).astype(np.float32)
        
        # 获取SAM候选segments（如果有）
        if self.sam_mask_helper is not None:
            segs_dict = self.sam_mask_helper.extract_sam_segs(img_name)
        else:
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

        if segs_origin is not None and isinstance(segs_origin, np.ndarray) and segs_origin.ndim == 3:
            K = segs_origin.shape[2]
            segs_resized_list = []
            for k in range(K):
                mask_k = segs_origin[:, :, k]
                mask_k_uint8 = (mask_k * 255).astype(np.uint8)
                mask_k_resized = self.transform.apply_image(mask_k_uint8)
                mask_k_resized = (mask_k_resized.astype(np.float32) / 255.0)
                segs_resized_list.append(mask_k_resized)

            if len(segs_resized_list) > 0:
                resized_h, resized_w = segs_resized_list[0].shape
                segs_resized = np.stack(segs_resized_list, axis=2)
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
                constant_values=1,
            )
        else:
            segs_square = segs_dict["segs_square"]

        segs_square_hwk_after_pad = segs_square.shape if isinstance(segs_square, np.ndarray) else None
        segs_square = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous()
        segs = F.interpolate(
            segs_square.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        # 计算IoU和IoP
        gt_fg = (gt_mask == 0).astype(np.uint8)
        segs_fg = (segs_origin == 0).astype(np.uint8)
        sampled_ious = [
            compute_all_iou(segs_fg, gt_fg) for _ in [gt_mask]
        ]
        sampled_iops = [
            compute_all_iop(segs_fg, gt_fg) for _ in [gt_mask]
        ]
        
        precision_type = torch.float32
        if self.precision == "fp16":
            precision_type = torch.float16
        elif self.precision == "bf16":
            precision_type = torch.bfloat16
        segs = segs.to(precision_type)
        
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        
        # 获取所有3条指令（关键修改：不再随机选择1条）
        instructions = sample.get('instructions', [])
        if not instructions:
            instructions = [f"segment the {sample.get('object', 'object')}"]
        
        # 为每条指令创建一个训练实例
        instances = []
        for i, instruction in enumerate(instructions):
            question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
            
            # 生成对话
            conversation_list = [self._build_conversation(instruction)]
            
            instance = {
                'image_path': image_path,
                'images': image,
                'images_clip': image_clip,
                'conversations': conversation_list,
                'masks': segs,
                'label': torch.ones(segs.shape[1], segs.shape[2]) * self.ignore_label,
                'resize': resize,
                'questions': [question],
                'sampled_classes': [instruction],
                'segs': segs,
                'ious': sampled_ious,
                'iops': sampled_iops,
                'segs_origin': segs_origin,
                'bbox': segs_dict.get('bbox', []),
                'inference': False,
                'segmentation_paths': gt_mask_path,
                'conversation_list': conversation_list,
                'origin_segs_list': segs_origin,
                'sam_ious_list': sampled_ious.tolist() if isinstance(sampled_ious, torch.Tensor) else sampled_ious,
                'candidate_mask_paths_list': segs_dict.get("mask_paths", []),
                'debug_meta': None,
            }
            instances.append(instance)
        
        return instances
    
    def _build_conversation(self, instruction):
        """构建单条指令的对话"""
        conv = conversation_lib.default_conversation.copy()
        
        # 生成随机回答
        answer = random.choice(self.answer_list)
        
        # 设置对话格式
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], answer)
        
        return conv.get_prompt()
