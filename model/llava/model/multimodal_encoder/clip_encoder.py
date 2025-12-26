import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            import os
            # 检查是否是本地路径（绝对路径或相对路径）
            is_local_path = (
                os.path.exists(self.vision_tower_name) and 
                os.path.isdir(self.vision_tower_name) and
                os.path.exists(os.path.join(self.vision_tower_name, "config.json"))
            )
            # 如果不是本地路径，尝试强制使用本地文件（避免网络请求）
            local_files_only = is_local_path
            if not is_local_path:
                # 如果路径不存在，但仍然尝试使用 local_files_only=True 避免网络请求
                # 这会在文件不存在时抛出更清晰的错误
                local_files_only = True
            
            self.cfg_only = CLIPVisionConfig.from_pretrained(
                self.vision_tower_name,
                local_files_only=local_files_only
            )

    def load_model(self):
        import os
        # 如果是本地路径，强制使用本地文件，避免网络请求
        local_files_only = os.path.exists(self.vision_tower_name) and os.path.isdir(self.vision_tower_name)
        
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name,
            local_files_only=local_files_only
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, 
            low_cpu_mem_usage=True,
            local_files_only=local_files_only
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        torch.cuda.empty_cache()
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
