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
        max_instructions: int = 3, # 每个样本使用的最大指令数 (默认3)
        is_train: bool = True,
        samples: List[Dict] = None,  # 改为 raw_samples，但保持向后兼容
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
        self.max_instructions = max_instructions
        self.is_train = is_train
        self.debug_meta = bool(debug_meta)
        self.sam_mask_helper = sam_mask_helper
        self.depth_dir = os.path.join(self.data_base_dir, self.split, "depth")
        self.camera_intrinsic, self.camera_size = self._load_camera_intrinsic()
        
        # 加载并标准化所有样本
        if samples is not None:
            self.samples = self.normalize_samples(samples)
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
            # ✅ 优先使用 gt_object，如果没有则使用 object
            object_name = ann.get('gt_object', ann.get('object', 'object'))
            gt_mask_path_rel = ann.get('gt_mask_path', '')
            instructions = ann.get('instructions', [])
            
            # 🔍 调试: 打印JSON中的字段
            # 只打印Hard样本(object != gt_object)
            if ann.get('object') != ann.get('gt_object'):
                hard_count = sum(1 for s in samples if s.get('object') != s.get('gt_object'))
                if hard_count < 5:  # 只打印前5个Hard样本
                    print(f"\n[Hard Sample #{hard_count + 1}]:", flush=True)
                    print(f"  - object (被遮挡): {ann.get('object', 'N/A')}", flush=True)
                    print(f"  - gt_object (要抓取): {ann.get('gt_object', 'N/A')}", flush=True)
                    print(f"  - 使用: {object_name}", flush=True)
            
            
            if not gt_mask_path_rel:
                continue
            
            mask_filename = os.path.basename(gt_mask_path_rel)
            # ✅ 修复: 使用 \d+ 匹配任意位数的数字 (1位、2位、3位都可以)
            match = re.match(r'^(\d+)_', mask_filename)
            if not match:
                print(f"Warning: Cannot extract image number from mask filename: {mask_filename}")
                continue
            
            img_num = match.group(1)
            img_name = f"{img_num}.png"
            
            image_path = os.path.join(self.data_base_dir, self.split, img_name)
            # ✅ 修复: 处理逗号分隔的多个相对路径，确保每个都拼上绝对路径前缀
            rel_paths = [p.strip() for p in gt_mask_path_rel.split(',')]
            abs_paths = [os.path.join(self.data_base_dir, self.split, p) for p in rel_paths]
            gt_mask_path = ",".join(abs_paths)
            
            samples.append({
                'image_path': image_path,
                'gt_mask_path': gt_mask_path,
                'instructions': instructions,
                'object': object_name,      # 保持兼容性
                'gt_object': object_name,   # ✅ 添加gt_object键,值来自JSON的gt_object字段
                'scene': scene,
                'img_name': img_name,
            })
        
        return samples
    
    def normalize_samples(self, raw_samples):
        """
        将原始样本（可能来自JSON文件的原始格式）标准化为Dataset期望的格式
        确保所有样本都包含必要的键：image_path, gt_mask_path, instructions等
        """
        samples = []
        if self.max_samples is not None:
            raw_samples = raw_samples[:self.max_samples]
        
        for ann in raw_samples:
            # 处理不同的样本格式
            image_path_rel = ann.get('image', ann.get('image_path', ''))
            gt_mask_path_rel = ann.get('gt_mask_path', '')
            # ✅ 优先使用 gt_object
            object_name = ann.get('gt_object', ann.get('object', 'object'))
            instructions = ann.get('instructions', [])
            
            if not gt_mask_path_rel:
                continue
            
            # 构建完整的绝对路径
            mask_filename = os.path.basename(gt_mask_path_rel)
            match = re.match(r'(\d+)_', mask_filename)
            if not match:
                print(f"Warning: Cannot extract image number from mask filename: {mask_filename}")
                continue
            
            img_num = match.group(1)
            img_name = f"{img_num}.png"
            
            image_path = os.path.join(self.data_base_dir, self.split, img_name)
            # ✅ 修复: 处理逗号分隔的多个相对路径，确保每个都拼上绝对路径前缀
            rel_paths = [p.strip() for p in gt_mask_path_rel.split(',')]
            abs_paths = [os.path.join(self.data_base_dir, self.split, p) for p in rel_paths]
            gt_mask_path = ",".join(abs_paths)
            
            samples.append({
                'image_path': image_path,  # 这是关键：完整的绝对路径
                'gt_mask_path': gt_mask_path,
                'instructions': instructions,
                'object': object_name,      # 保持兼容性
                'gt_object': object_name,   # ✅ 添加gt_object键
                'scene': ann.get('scene', ''),
                'img_name': img_name,
            })
        
        return samples
    
    def __len__(self):
        # 修改：返回总任务数（每个样本有 N 条指令参与）
        return len(self.samples) * self.max_instructions
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _load_camera_intrinsic(self):
        camera_paths = [
            os.path.join(self.data_base_dir, self.split, "depth", "camera.json"),
            os.path.join(self.data_base_dir, "train", "depth", "camera.json"),
        ]
        for camera_path in camera_paths:
            if not os.path.exists(camera_path):
                continue
            with open(camera_path, "r") as f:
                camera = json.load(f)
            intr = camera.get("intrinsics", {})
            if all(k in intr for k in ("fx", "fy", "cx", "cy")):
                K = np.array(
                    [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            else:
                K = np.array(intr.get("intrinsic", np.eye(3)), dtype=np.float32)
            size = (int(camera.get("height", 0)), int(camera.get("width", 0)))
            return K, size
        return None, None

    def _load_depth_and_intrinsic(self, img_name: str, resize):
        if self.camera_intrinsic is None:
            return None, None
        depth_path = os.path.join(self.depth_dir, os.path.splitext(img_name)[0] + ".npy")
        if not os.path.exists(depth_path):
            return None, None

        depth = np.load(depth_path)
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape for {depth_path}: {depth.shape}")
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.maximum(depth, 0.0)

        resized_h, resized_w = resize
        depth_resized = cv2.resize(depth, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padh = self.image_size - resized_h
        padw = self.image_size - resized_w
        if padh < 0 or padw < 0:
            raise ValueError(f"depth_resized larger than image_size={self.image_size}: {(resized_h, resized_w)}")
        depth_square = np.pad(depth_resized, ((0, padh), (0, padw)), mode="constant", constant_values=0.0)

        cam_h, cam_w = self.camera_size if self.camera_size is not None else depth.shape
        if cam_h <= 0 or cam_w <= 0:
            cam_h, cam_w = depth.shape
        K = self.camera_intrinsic.copy()
        K[0, :] *= resized_w / float(cam_w)
        K[1, :] *= resized_h / float(cam_h)
        return torch.from_numpy(depth_square).unsqueeze(0).float(), torch.from_numpy(K).float()
    
    def __getitem__(self, idx):
        # 修改：根据idx计算样本索引和指令索引
        sample_idx = idx // self.max_instructions  
        instruction_idx = idx % self.max_instructions
        
        sample = self.samples[sample_idx]
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
        
        # 读取GT mask(可能有多个，使用逗号分隔)
        gt_mask_paths = [p.strip() for p in gt_mask_path.split(',')]
        gt_masks = []
        for p in gt_mask_paths:
            mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                gt_masks.append((mask > 0).astype(np.float32))
            else:
                print(f"Warning: GT mask not found: {p}")
        
        if not gt_masks:
            print(f"Warning: No valid GT masks found for {image_path}, using empty mask")
            gt_masks = [np.ones(ori_size, dtype=np.uint8)]
        
        # # 🔍 调试: 打印GT mask和物体信息 (每100个样本打印一次)
        # if idx % 1 == 0:
        #     print(f"[Dataset Debug] Sample {idx}:", flush=True)
        #     print(f"  - Object name: {sample.get('object', 'N/A')}", flush=True)
        #     print(f"  - GT mask path: {gt_mask_path}", flush=True)
        #     print(f"  - Original image size: {ori_size}", flush=True)
        #     print(f"  - GT mask original shape: {gt_mask.shape}", flush=True)
        #     print(f"  - GT mask value range: [{gt_mask.min():.2f}, {gt_mask.max():.2f}]", flush=True)
        
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
        segs_fg = (segs_origin == 0).astype(np.uint8)
        all_ious = []
        all_iops = []
        for mask in gt_masks:
            gt_fg = (mask == 0).astype(np.uint8)
            all_ious.append(compute_all_iou(segs_fg, gt_fg))
            all_iops.append(compute_all_iop(segs_fg, gt_fg))
            
        sampled_ious = [np.max(np.stack(all_ious), axis=0)]
        sampled_iops = [np.max(np.stack(all_iops), axis=0)]
        gt_mask_cat = np.stack(gt_masks) # (N, H, W)
        
        precision_type = torch.float32
        if self.precision == "fp16":
            precision_type = torch.float16
        elif self.precision == "bf16":
            precision_type = torch.bfloat16
        segs = segs.to(precision_type)

        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        depth, intrinsic = self._load_depth_and_intrinsic(img_name, resize)

        # ❌ 删除GT mask预处理 - 可视化需要原始尺寸的mask
        # 直接返回原始GT mask (H_orig, W_orig),让可视化代码自己resize
        # 这与 vigor_dataset.py 的行为一致

        # 按照robot_arm_dataset.py的方式添加预处理（归一化 + padding到正方形）
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # 获取指令列表，并根据instruction_idx选择对应指令
        instructions = sample.get('instructions', [])
        if not instructions:
            instructions = [f"segment {sample.get('object', 'object')}"]

        # 确保instruction_idx不超出范围
        if instruction_idx >= len(instructions):
            instruction_idx = 0

        instruction = instructions[instruction_idx]
        question = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"

        # 生成对话
        conversation_list = [self._build_conversation(question)]

        # # 🔍 调试: 打印返回的object_name
        # if idx < 3:
        #     print(f"\n[__getitem__ Debug] idx={idx}:", flush=True)
        #     print(f"  - sample dict keys: {list(sample.keys())}", flush=True)
        #     print(f"  - sample.get('object'): {sample.get('object', 'N/A')}", flush=True)
        #     print(f"  - sample.get('gt_object'): {sample.get('gt_object', 'N/A')}", flush=True)
        #     print(f"  - 返回的 object_name: {sample.get('gt_object', sample.get('object', 'object'))}", flush=True)
        
        # 返回单个实例（不再是列表）
        return {
            'image_path': image_path,
            'images': image,  # 现在是tensor
            'images_clip': image_clip,
            'depth': depth,
            'intrinsic': intrinsic,
            'conversations': conversation_list,
            'masks': torch.from_numpy(gt_mask_cat),  # ✅ 返回所有原始尺寸的GT mask 
            'label': torch.ones(gt_mask_cat.shape[1], gt_mask_cat.shape[2]) * self.ignore_label,
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
            'object_name': sample['gt_object'],  # ✅ 直接使用 gt_object (已在第138行添加)
        }
    
    def _build_conversation(self, instruction):
        """构建单条指令的对话"""
        conv = conversation_lib.default_conversation.copy()
        
        # 生成随机回答
        answer = random.choice(self.answer_list)
        
        # 设置对话格式
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], answer)
        
        return conv.get_prompt()
