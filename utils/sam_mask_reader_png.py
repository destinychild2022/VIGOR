import os
import glob
from typing import List, Dict
import numpy as np
import cv2
import time


class SAM_Mask_Reader_PNG:
    """
    从 PNG 文件目录读取 SAM 候选 mask
    目录结构：base_dir/{image_id}/masks/mask_*.png
    """
    
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        print(f"Initializing SAM_Mask_Reader_PNG from: {base_dir}")
        start_time = time.time()
        
        # 构建图像ID到mask路径的索引
        self.mask_index = self.build_mask_index()
        
        end_time = time.time()
        print(f"Built mask index in {end_time - start_time:.2f} seconds")
        print(f"Total images with masks: {len(self.mask_index)}")
    
    def build_mask_index(self) -> Dict[str, List[str]]:
        """
        构建图像ID到mask文件路径列表的索引
        目录结构：base_dir/{image_id}/masks/mask_*.png
        """
        mask_index = {}
        
        # 遍历所有子目录（每个子目录对应一个图像ID）
        if not os.path.exists(self.base_dir):
            print(f"Warning: Directory {self.base_dir} does not exist")
            return mask_index
        
        for image_id_dir in os.listdir(self.base_dir):
            image_id_path = os.path.join(self.base_dir, image_id_dir)
            if not os.path.isdir(image_id_path):
                continue
            
            masks_dir = os.path.join(image_id_path, "masks")
            if not os.path.exists(masks_dir):
                continue
            
            # 获取所有mask文件，按文件名排序（通常包含iou信息）
            mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.png")))
            if mask_files:
                # 使用图像ID作为key（去掉目录路径，只保留ID）
                mask_index[image_id_dir] = mask_files
        
        return mask_index
    
    def get_mask_paths(self, image_id: str) -> List[str]:
        """
        根据图像ID获取对应的mask文件路径列表
        image_id: 图像ID（例如 "1", "100" 等）
        """
        if image_id not in self.mask_index:
            # 尝试字符串匹配
            for key in self.mask_index.keys():
                if str(image_id) == str(key):
                    return self.mask_index[key]
            raise ValueError(f"Image ID {image_id} not found in mask index")
        return self.mask_index[image_id]
    
    def preprocess_mask(self, masks: np.ndarray):
        """
        预处理mask：padding到正方形
        masks: (H, W, K)
        """
        masks = masks.astype(np.float64)
        h, w, _ = masks.shape
        padh = max(h, w) - h
        padw = max(h, w) - w
        # 约定：0=掩码(前景)，1=背景；padding 区域应为背景(1)，避免把 padding 当成前景
        masks = np.pad(masks, ((0, padh), (0, padw), (0, 0)), mode="constant", constant_values=1)
        assert masks.shape[0] == masks.shape[1]
        return masks
    
    def extract_sam_segs(self, image_name: str):
        """
        提取SAM候选segments
        image_name: 图像文件名（例如 "1.png"），需要提取出图像ID
        返回格式与 SAM_Mask_Reader.extract_sam_segs 一致
        """
        # 从文件名提取图像ID（去掉扩展名）
        image_id = os.path.splitext(image_name)[0]
        
        # 获取mask文件路径列表
        mask_paths = self.get_mask_paths(image_id)
        
        # 读取所有mask
        seg_list = []
        bbox_list = []
        
        used_mask_paths = []
        for mask_path in mask_paths:
            # 读取mask（灰度图，0-255）
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # ✅ 修正：确保与GT mask格式一致（0=掩码(前景)，1=背景）
            # 假设：低值（黑色）为掩码区域，高值（白色）为背景
            # 但是某些mask文件可能值范围不同，所以使用<128作为背景的阈值
            mask_binary = (mask < 128).astype(np.uint8)  # 低值(黑色)->1(背景)，高值(白色)->0(前景)
            seg_list.append(mask_binary)
            used_mask_paths.append(mask_path)
            
            # 计算bbox（可选，如果需要的话）
            # bbox 应基于前景(掩码)区域：mask_binary==0
            fg = (mask_binary == 0)
            rows = np.any(fg, axis=1)
            cols = np.any(fg, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                h, w = mask_binary.shape
                bbox = [x_min / w, y_min / h, (x_max - x_min) / w, (y_max - y_min) / h]
            else:
                bbox = [0, 0, 0, 0]
            bbox_list.append(bbox)
        
        if len(seg_list) == 0:
            # 如果没有找到mask，返回空的
            h, w = 896, 896  # 默认尺寸
            # 约定：0=前景，1=背景；空mask应为全背景
            segs_origin = np.ones((h, w, 1), dtype=np.uint8)
            segs_square = self.preprocess_mask(segs_origin)
            return {
                "segs_square": segs_square,
                "segs_origin": segs_origin,
                "bbox": [[0, 0, 0, 0]],
                "mask_paths": [],
            }
        
        # 堆叠所有mask: (H, W, K)
        segs_origin = np.stack(seg_list, axis=2)
        
        # 预处理：padding到正方形
        segs_square = self.preprocess_mask(segs_origin)
        
        return {
            "segs_square": segs_square,
            "segs_origin": segs_origin,
            "bbox": bbox_list,
            # 与 segs_origin 的 K 维一一对应的候选 mask 文件路径
            "mask_paths": used_mask_paths,
        }
