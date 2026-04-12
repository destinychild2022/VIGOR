import glob
import json
import os
import random
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


class RobotArmDataset(torch.utils.data.Dataset):
    """
    机器人手臂数据集
    支持多个视角（robot_arm_01, robot_arm_02, robot_arm_03）
    每个视角只使用前100张标注好的图像进行训练
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        json_paths: List[str],  # 多个annotations.json文件路径
        tokenizer,
        vision_tower,
        precision: str = "bf16",
        image_size: int = 896,
        raw_pic_base_dir: str = os.path.join(".", "dataset", "raw_pic"),
        gt_mask_base_dir: str = os.path.join(".", "dataset", "GT_mask"),
        sam_candidate_base_dir: str = os.path.join(".", "dataset", "sam_candidate"),
        sam_mask_helpers: Dict[str, SAM_Mask_Reader_PNG] = None,
        max_samples_per_view: int = 100,  # 每个视角最多使用100张图像
        is_train: bool = True,
        samples: List[Dict] = None,  # 允许外部直接传入已构建/已切分的 samples
        debug_meta: bool = False,  # 是否返回 debug_meta（用于每个epoch打印一次维度追踪）
    ):
        self.json_paths = json_paths
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
        
        self.raw_pic_base_dir = raw_pic_base_dir
        self.gt_mask_base_dir = gt_mask_base_dir
        self.sam_candidate_base_dir = sam_candidate_base_dir
        self.max_samples_per_view = max_samples_per_view
        self.is_train = is_train
        self.debug_meta = bool(debug_meta)
        
        # SAM mask helpers（每个视角一个）
        self.sam_mask_helpers = sam_mask_helpers or {}
        
        # 加载所有样本（支持外部传入切分后的 samples，避免重复读 json/重复划分）
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self.load_all_samples()
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        
        print(f"Loaded {len(self.samples)} samples from {len(json_paths)} annotation files")
    
    def load_all_samples(self):
        """
        从多个annotations.json文件加载样本
        每个视角只使用前max_samples_per_view张图像
        """
        samples = []
        
        for json_path in self.json_paths:
            # 从路径提取视角名称（例如 robot_arm_01）
            view_name = os.path.basename(os.path.dirname(json_path))
            
            # 读取annotations.json
            with open(json_path, "r") as f:
                annotations = json.load(f)
            
            # 限制每个视角的样本数量（只使用前max_samples_per_view张）
            if self.is_train:
                annotations = annotations[:self.max_samples_per_view]
            
            # 处理每个标注
            for ann in annotations:
                img_name = ann['img_name']
                object_name = ann.get('object', 'object')
                gt_path = ann.get('gt_path', '')
                
                # 构建完整路径
                # 原图路径：raw_pic/robot_arm_XX/{img_name}
                image_path = os.path.join(self.raw_pic_base_dir, view_name, img_name)
                
                # GT mask路径：GT_mask/robot_arm_XX/masks/{gt_path中的文件名}
                # gt_path格式可能是 "robot_arm_01\masks\1_seat_mask.png" 或 "robot_arm_01/masks/1_seat_mask.png"
                gt_mask_filename = os.path.basename(gt_path.replace('\\', '/'))
                gt_mask_path = os.path.join(self.gt_mask_base_dir, view_name, "masks", gt_mask_filename)
                
                # 生成问题（基于object名称）
                question = f"segment the {object_name}"
                
                samples.append({
                    'image_path': image_path,
                    'gt_mask_path': gt_mask_path,
                    'question': question,
                    'object': object_name,
                    'view_name': view_name,
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
        view_name = sample['view_name']
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
        sampled_sents = [sample['question']]
        is_sentence = True
        
        # 获取SAM候选segments
        if view_name in self.sam_mask_helpers:
            segs_dict = self.sam_mask_helpers[view_name].extract_sam_segs(img_name)
        else:
            # 如果没有对应的helper，创建空的
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
        # 否则 sam_segs_list（用于 mask_pooling）会与 image_embeddings（来自 896 预处理图）空间不对齐，
        # 导致不同候选 mask 特征非常相似、pred_similarity 区分度低、AlignLoss 难下降。
        # 当前语义：0=前景(掩码)，1=背景；pad 区域应为背景(1)。
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
            # antialias=False (默认值，与原图插值保持一致)
        ).squeeze(0)  # (K, 256, 256)
        
        # 计算IoU和IoP
        # ✅ 注意：本数据集内部约定为 0=掩码(前景)，1=背景
        # compute_iou/compute_iop 期望前景为 1，所以这里临时转换成前景=1的二值图再计算
        # gt_fg: (H,W) 前景为1
        gt_fg = (gt_mask == 0).astype(np.uint8)
        # segs_fg: (H,W,K) 前景为1
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
        questions = []
        answers = []
        for text in sampled_sents:
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
                # 计算每一步变换后的mask面积占比（前景像素数/总像素数）
                segs_origin_area_ratio = None
                segs_resized_area_ratio = None
                segs_square_area_ratio = None
                segs_interp_256_area_ratio = None
                
                if isinstance(segs_origin, np.ndarray) and segs_origin.ndim == 3:
                    # segs_origin: (H_raw, W_raw, K), 0=前景
                    H_orig, W_orig, K_orig = segs_origin.shape
                    total_pixels_orig = H_orig * W_orig
                    fg_pixels_orig = np.sum(segs_origin == 0, axis=(0, 1))  # (K,)
                    segs_origin_area_ratio = (fg_pixels_orig / total_pixels_orig).tolist() if total_pixels_orig > 0 else None
                
                if 'segs_resized' in locals() and isinstance(segs_resized, np.ndarray) and segs_resized.ndim == 3:
                    # segs_resized: (H_resized, W_resized, K), 0=前景
                    H_res, W_res, K_res = segs_resized.shape
                    total_pixels_res = H_res * W_res
                    fg_pixels_res = np.sum(segs_resized < 0.5, axis=(0, 1))  # 考虑插值可能产生中间值
                    segs_resized_area_ratio = (fg_pixels_res / total_pixels_res).tolist() if total_pixels_res > 0 else None
                
                if isinstance(segs_square, np.ndarray) and segs_square.ndim == 3:
                    # segs_square: (H_square, W_square, K), 0=前景
                    H_sq, W_sq, K_sq = segs_square.shape
                    total_pixels_sq = H_sq * W_sq
                    fg_pixels_sq = np.sum(segs_square < 0.5, axis=(0, 1))  # 考虑插值可能产生中间值
                    segs_square_area_ratio = (fg_pixels_sq / total_pixels_sq).tolist() if total_pixels_sq > 0 else None
                elif isinstance(segs_square, torch.Tensor) and segs_square.dim() == 3:
                    # 注意：后续代码会把 segs_square 变成 torch.Tensor(K, H, W)
                    # 这里兼容 torch 的统计，避免打印 N/A
                    K_sq, H_sq, W_sq = segs_square.shape
                    total_pixels_sq = int(H_sq * W_sq)
                    segs_sq_cpu = segs_square.detach().float().cpu()
                    fg_pixels_sq = (segs_sq_cpu < 0.5).sum(dim=(1, 2))  # (K,)
                    segs_square_area_ratio = (
                        (fg_pixels_sq / float(total_pixels_sq)).tolist() if total_pixels_sq > 0 else None
                    )
                
                if isinstance(segs, torch.Tensor) and segs.dim() == 3:
                    # segs: (K, 256, 256), 0=前景
                    # 注意：segs 可能是 bf16，直接 .numpy() 会报错；这里用 torch 计算避免 dtype 问题
                    K_interp, H_interp, W_interp = segs.shape
                    total_pixels_interp = int(H_interp * W_interp)
                    segs_cpu = segs.detach().float().cpu()
                    fg_pixels_interp = (segs_cpu < 0.5).sum(dim=(1, 2))  # (K,)
                    segs_interp_256_area_ratio = (
                        (fg_pixels_interp / float(total_pixels_interp)).tolist() if total_pixels_interp > 0 else None
                    )
                
                debug_meta = {
                    "idx": int(idx),
                    "image_path": image_path,
                    "raw_image_shape": tuple(cv2.imread(image_path).shape) if os.path.exists(image_path) else None,
                    "ori_size_hw": tuple(ori_size),
                    "image_after_resize_longest_hw": tuple(resize),
                    "image_after_preprocess_chw": tuple(image.shape),
                    "gt_mask_path": gt_mask_path,
                    "gt_mask_shape_hw": tuple(gt_mask.shape) if hasattr(gt_mask, "shape") else None,
                    "gt_mask_minmax": (float(np.min(gt_mask)), float(np.max(gt_mask))) if isinstance(gt_mask, np.ndarray) else None,
                    "gt_fg_pixels": int(np.sum(gt_mask == 0)) if isinstance(gt_mask, np.ndarray) else None,
                    "gt_mask_area_ratio": float(np.sum(gt_mask == 0) / (gt_mask.shape[0] * gt_mask.shape[1])) if isinstance(gt_mask, np.ndarray) else None,
                    "segs_origin_shape_hwk": tuple(segs_origin.shape) if isinstance(segs_origin, np.ndarray) else None,
                    "segs_origin_minmax": (int(segs_origin.min()), int(segs_origin.max())) if isinstance(segs_origin, np.ndarray) else None,
                    "segs_origin_area_ratio": segs_origin_area_ratio,  # list of K ratios
                    # 候选mask：ResizeLongestSide 后（与原图一致）
                    "segs_resized_shape_hwk": tuple(segs_resized.shape) if ("segs_resized" in locals() and isinstance(segs_resized, np.ndarray)) else None,
                    # 候选mask：pad 前后（H,W,K）与 (K,H,W)
                    "segs_square_after_pad_shape_hwk": tuple(segs_square_hwk_after_pad) if segs_square_hwk_after_pad is not None else None,
                    "segs_square_shape_hwk": tuple(segs_dict.get("segs_square", np.zeros((0, 0, 0))).shape),
                    "segs_square_after_align_shape_khw": tuple(segs_square.shape),
                    "segs_square_tensor_shape_khw": tuple(segs_square.shape),
                    "segs_resized_area_ratio": segs_resized_area_ratio,  # list of K ratios
                    "segs_square_area_ratio": segs_square_area_ratio,  # list of K ratios
                    "segs_interp_256_shape_khw": tuple(segs.shape),
                    "segs_interp_256_area_ratio": segs_interp_256_area_ratio,  # list of K ratios
                    "num_candidate_masks": int(len(candidate_mask_paths)) if isinstance(candidate_mask_paths, list) else None,
                    "candidate_mask_paths_head": candidate_mask_paths[:3] if isinstance(candidate_mask_paths, list) else None,
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
            'segmentation_paths': gt_mask_path,  # 单个路径字符串，不是列表
            'conversation_list': conversations,  # 训练代码期望列表，通过索引访问
            'origin_segs_list': segs_origin,
            'sam_ious_list': ious.tolist() if isinstance(ious, torch.Tensor) else ious,
            'candidate_mask_paths_list': candidate_mask_paths,
            'debug_meta': debug_meta,
        }

