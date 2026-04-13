from typing import List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

from .loss import sigmoid_align_loss, softmax_align_loss, iou_regression_loss
from .transformer import Attention, MLPBlock, LISA_TwoWayAttentionBlock

class ProposalGeometryPrior(nn.Module):
    def __init__(
        self,
        num_heads: int,
        initial_value: float = 2.0,
        heads_range: float = 4.0,
        weight_init: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)
        self.weight = nn.Parameter(torch.full((2, 1, 1, 1), weight_init), requires_grad=True)
        self.eps = eps
        self.last_active = False
        self.last_num_proposals = 0
        self.last_num_valid = 0
        self.last_bias_abs_mean = 0.0
        self.last_weight_grad = None
        self.weight.register_hook(self._capture_weight_grad)

    def _capture_weight_grad(self, grad):
        self.last_weight_grad = grad.detach()
        return grad

    def _mark_inactive(self):
        self.last_active = False
        self.last_num_proposals = 0
        self.last_num_valid = 0
        self.last_bias_abs_mean = 0.0

    def proposal_valid_mask(self, masks_fg: torch.Tensor):
        if masks_fg is None or masks_fg.dim() != 3:
            return None
        return masks_fg.float().flatten(1).sum(dim=1) > self.eps

    def _masked_depth_median(self, masks: torch.Tensor, depth: torch.Tensor, area: torch.Tensor):
        depth_flat = depth.flatten()
        mask_flat = masks.flatten(1) > 0.5
        soft_mean = (masks * depth.unsqueeze(0)).flatten(1).sum(dim=1) / area
        medians = []
        for idx in range(mask_flat.shape[0]):
            selected = depth_flat[mask_flat[idx]]
            if selected.numel() == 0:
                selected = depth_flat[masks.flatten(1)[idx] > 0]
            if selected.numel() == 0:
                medians.append(soft_mean[idx])
            else:
                medians.append(selected.median())
        return torch.stack(medians, dim=0)

    def forward(self, masks_fg: torch.Tensor, depth_map: torch.Tensor):
        if masks_fg is None or depth_map is None:
            self._mark_inactive()
            return None
        if masks_fg.dim() != 3:
            self._mark_inactive()
            return None

        masks = masks_fg.float().clamp(0.0, 1.0)
        k, h, w = masks.shape
        if k == 0:
            self._mark_inactive()
            return None

        depth = depth_map.float()
        if depth.dim() == 3:
            depth = depth[0]
        elif depth.dim() == 4:
            depth = depth[0, 0]
        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        area_raw = masks.flatten(1).sum(dim=1)
        valid = area_raw > self.eps
        area = area_raw.clamp_min(self.eps)

        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=masks.device, dtype=masks.dtype),
            torch.linspace(0.0, 1.0, w, device=masks.device, dtype=masks.dtype),
            indexing="ij",
        )
        cx = (masks * xx).flatten(1).sum(dim=1) / area
        cy = (masks * yy).flatten(1).sum(dim=1) / area
        centers = torch.stack([cx, cy], dim=-1)
        pos_dist = (centers[:, None, :] - centers[None, :, :]).abs().sum(dim=-1)

        proposal_depth = self._masked_depth_median(masks, depth, area)
        depth_dist = (proposal_depth[:, None] - proposal_depth[None, :]).abs()

        decay = self.decay.to(device=masks.device, dtype=masks.dtype)[None, :, None, None]
        geo_bias = decay * (
            self.weight[0].to(dtype=masks.dtype) * pos_dist[None, None]
            + self.weight[1].to(dtype=masks.dtype) * depth_dist[None, None]
        )

        if not bool(valid.all()):
            invalid_key = (~valid)[None, None, None, :]
            geo_bias = geo_bias.masked_fill(invalid_key, -1e4)

        with torch.no_grad():
            valid_pair = valid[None, None, :, None] & valid[None, None, None, :]
            valid_bias = geo_bias.detach().masked_select(valid_pair)
            self.last_active = True
            self.last_num_proposals = int(k)
            self.last_num_valid = int(valid.sum().item())
            self.last_bias_abs_mean = float(valid_bias.abs().mean().item()) if valid_bias.numel() > 0 else 0.0
        return geo_bias

class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            # Transformers 的 from_pretrained 可能会把 train_mask_decoder/out_dim 当作 config kwargs
            # 先消费掉，导致它们不再出现在这里的 kwargs 里，所以必须做 fallback。
            self.config.train_mask_decoder = kwargs.get(
                "train_mask_decoder", getattr(self.config, "train_mask_decoder", False)
            )
            self.config.out_dim = kwargs.get("out_dim", getattr(self.config, "out_dim", 256))
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        print("Initializing LISA modules...")        
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # DINO-V2
        # 优先从数据盘/本地路径加载权重文件，torch.hub 代码缓存也放到 TORCH_HOME。
        dinov2_local_path = os.environ.get(
            "DINOV2_LOCAL_PATH",
            os.path.join(
                os.environ.get("MODEL_BASE_DIR", os.path.join(os.environ.get("AUTODL_TMP_DIR", "/root/autodl-tmp"), "model")),
                "dinov2_vitl14",
            ),
        )
        torch_home = os.environ.get("TORCH_HOME", os.path.join(os.environ.get("AUTODL_TMP_DIR", "/root/autodl-tmp"), "torch_cache"))
        os.environ.setdefault("TORCH_HOME", torch_home)
        torch.hub.set_dir(os.path.join(torch_home, "hub"))
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        dinov2_weight_path = None
        
        # 检查本地路径是否有权重文件
        if os.path.exists(dinov2_local_path):
            import glob
            pth_files = glob.glob(os.path.join(dinov2_local_path, "*.pth"))
            if pth_files:
                dinov2_weight_path = pth_files[0]
                print(f"Found local DINOv2 weight: {dinov2_weight_path}")
        
        # ✅ 防止多进程同时下载 DINOv2 代码仓库导致冲突
        # 获取当前进程的 rank，只有 rank 0 才下载，其他进程等待
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        
        # 加载模型结构（从 torch.hub，这会下载代码但可能失败）
        # 如果本地有权重文件，使用 pretrained=False 避免下载权重
        use_pretrained = dinov2_weight_path is None
        
        dinov2_vitl14 = None
        cache_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        print(f"DINOv2 local path: {dinov2_local_path}")
        print(f"DINOv2 torch.hub cache: {cache_dir}")

        def load_dinov2_from_hub(pretrained):
            local_hubconf = os.path.join(cache_dir, "hubconf.py")
            if os.path.exists(local_hubconf):
                return torch.hub.load(cache_dir, 'dinov2_vitl14', pretrained=pretrained, source='local')
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=pretrained)
        
        # 如果是多 GPU 训练，使用文件锁机制
        import torch.distributed as dist
        if dist.is_initialized():
            if local_rank == 0:
                # Rank 0 负责下载/加载
                try:
                    # 先清理可能损坏的缓存
                    import shutil
                    broken_cache = os.path.join(torch.hub.get_dir(), "facebookresearch-dinov2-b194f00")
                    if os.path.exists(broken_cache):
                        shutil.rmtree(broken_cache, ignore_errors=True)
                    
                    dinov2_vitl14 = load_dinov2_from_hub(use_pretrained)
                    print(f"[Rank 0] Loaded DINOv2 model structure from torch.hub")
                except Exception as e:
                    print(f"[Rank 0] Warning: Failed to load DINOv2 from torch.hub ({e})")
                    if os.path.exists(cache_dir):
                        try:
                            dinov2_vitl14 = torch.hub.load(cache_dir, 'dinov2_vitl14', pretrained=use_pretrained, source='local')
                            print("[Rank 0] Loaded DINOv2 model structure from local cache")
                        except Exception as e2:
                            print(f"[Rank 0] Error: Cannot load DINOv2 from cache: {e2}")
                            raise e2
                    else:
                        raise e
            
            # 等待 Rank 0 完成下载
            dist.barrier()
            
            if local_rank != 0:
                # 其他 Rank 从缓存加载
                try:
                    dinov2_vitl14 = torch.hub.load(cache_dir, 'dinov2_vitl14', pretrained=use_pretrained, source='local')
                    print(f"[Rank {local_rank}] Loaded DINOv2 model structure from local cache")
                except Exception as e:
                    print(f"[Rank {local_rank}] Error: Cannot load DINOv2 from cache: {e}")
                    raise e
        else:
            # 单 GPU 训练，直接加载
            try:
                dinov2_vitl14 = load_dinov2_from_hub(use_pretrained)
                if use_pretrained:
                    print("Loaded DINOv2 from torch.hub (with pretrained weights)")
                else:
                    print("Loaded DINOv2 model structure from torch.hub")
            except Exception as e:
                print(f"Warning: Failed to load DINOv2 from torch.hub ({e})")
                if os.path.exists(cache_dir):
                    try:
                        dinov2_vitl14 = torch.hub.load(cache_dir, 'dinov2_vitl14', pretrained=use_pretrained, source='local')
                        print("Loaded DINOv2 model structure from local cache")
                    except Exception as e2:
                        print(f"Error: Cannot load DINOv2 model structure from cache.")
                        raise e2
                else:
                    raise e
        
        # 如果本地有权重文件，加载本地权重
        if dinov2_weight_path and os.path.exists(dinov2_weight_path):
            try:
                print(f"Loading DINOv2 weights from: {dinov2_weight_path}")
                state_dict = torch.load(dinov2_weight_path, map_location='cpu')
                dinov2_vitl14.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded DINOv2 weights from local path")
            except Exception as e:
                print(f"Warning: Failed to load weights from local path ({e}), using pretrained weights")
                dinov2_vitl14 = load_dinov2_from_hub(True)
        
        self.visual_model_dinov2 = dinov2_vitl14
        for param in self.visual_model_dinov2.parameters():
            param.requires_grad = False

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        # # add fc for mask embedding
        # mask_fc = [
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(out_dim, out_dim),
        #     nn.Dropout(0.0),
        # ]
        # # redudant code, but for clarity
        # self.mask_hidden_fcs = nn.ModuleList([nn.Sequential(*mask_fc)])
        # self.mask_hidden_fcs.train()
        # for param in self.mask_hidden_fcs.parameters():
        #     param.requires_grad = True

        # # learnable temperature and bias
        # self.temperature = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        # self.bias = nn.Parameter(torch.tensor(-10.0), requires_grad=True)

        # # use attention to replace the fc layer
        # self.lisa_mask_self_attn= Attention(256, 8)
        # self.lisa_norm1 = nn.LayerNorm(256)
        # self.lisa_cross_attn_mask_to_text = Attention(256, 8)
        # self.lisa_norm2 = nn.LayerNorm(256)
        # self.lisa_mlp = MLPBlock(256, 2048, torch.nn.ReLU)

        # 1x1 conv to reduce the dimension of the image feature to 256
        self.lisa_dino_conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lisa_geo_prior = ProposalGeometryPrior(num_heads=8)

        self.lisa_attention_layers = nn.ModuleList()
        depth = 2
        for i in range(depth):
            self.lisa_attention_layers.append(
                LISA_TwoWayAttentionBlock(
                    embedding_dim=256,
                    num_heads=8,
                    mlp_dim=2048,
                    attention_downsample_rate=1,
                )
            )
        self.lisa_final_attn = Attention(
            embedding_dim=256, num_heads=8, downsample_rate=1
        )
        self.lisa_norm_final_attn = nn.LayerNorm(256)

        self.lisa_iou_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.lisa_embedding_head = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )



class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # 这些 loss 权重在 forward 中总会用到；即使 config 里已有 train_mask_decoder 也必须有默认值
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.align_loss_weight = kwargs.pop("align_loss_weight", 1.0)
        self.regression_loss_weight = kwargs.pop("regression_loss_weight", 1.0)

        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            # ⚠️ 注意：Transformers 的 from_pretrained 可能会把 vision_tower/mm_vision_tower 当作 config kwargs
            # 先消耗掉，从而不会再出现在这里的 kwargs 里。
            # 因此这里必须优先保留 config.mm_vision_tower（已经从 config.json 或 config kwargs 里设置好），
            # 只有当 config 里没有时才从 kwargs 读取，最后才回退到 HF id。
            config.mm_vision_tower = getattr(config, "mm_vision_tower", None) or kwargs.get(
                "mm_vision_tower", kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            )
            #self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            #self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            # 即使配置文件中已有 train_mask_decoder，也要强制使用命令行传入的 vision_tower
            # 这样可以确保使用本地路径而不是配置文件中的 Hugging Face 路径
            if "vision_tower" in kwargs:
                config.mm_vision_tower = kwargs["vision_tower"]

        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def get_dinov2_visual_embs(self, pixel_values: torch.FloatTensor):

        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings_dict = self.model.visual_model_dinov2.forward_features(pixel_values[i].unsqueeze(0))
                image_embeddings = image_embeddings_dict['x_norm_patchtokens']
                # 1*4096*1024 -> 1*1024*256*256
                image_embeddings = image_embeddings.permute(0, 2, 1).reshape(1, 1024, 64, 64)
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def mask_pooling(self, image_embeddings: torch.FloatTensor, weight_maps: torch.FloatTensor):
        # image_embeddings: [256, 64, 64]
        # weight_maps: [K, 64, 64]
        # output: [K, 256]

        # [256, 64, 64] -> [256, 4096]
        image_embeddings = image_embeddings.flatten(1, 2)
        # [K, 64, 64] -> [K, 4096]
        weight_maps = weight_maps.flatten(1, 2)
        # [K, 4096] -> [K, 256]
        output = weight_maps @ image_embeddings.T
        # normalize
        output = output / (weight_maps.sum(-1, keepdim=True) + 1e-8)

        assert output.shape[0] == weight_maps.shape[0]
        assert output.shape[1] == image_embeddings.shape[0]

        return output

    @staticmethod
    def _cosine_sim_stats_kd(feat_kd: torch.Tensor):
        """计算 (K,D) 特征之间的余弦相似度统计（仅统计非对角元素）"""
        if feat_kd is None or (not isinstance(feat_kd, torch.Tensor)):
            return None
        if feat_kd.dim() != 2:
            return None
        K = feat_kd.shape[0]
        if K < 2:
            return None
        x = feat_kd.detach().float().cpu()
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.clamp(x @ x.T, -1.0, 1.0)
        mask = ~torch.eye(K, dtype=torch.bool)
        vals = sim[mask]
        return {
            "shape": [int(K), int(feat_kd.shape[1])],
            "mean": float(vals.mean().item()),
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "std": float(vals.std().item()),
        }

    @staticmethod
    def _mask_iou_stats_khw(masks_khw: torch.Tensor):
        """计算 (K,H,W) mask之间的IoU相似度统计（仅统计非对角元素）"""
        if masks_khw is None or (not isinstance(masks_khw, torch.Tensor)):
            return None
        if masks_khw.dim() != 3:
            return None
        K = masks_khw.shape[0]
        if K < 2:
            return None
        # 将mask二值化（>0.5为前景）
        masks_binary = (masks_khw > 0.5).float()
        # 计算每对mask之间的IoU
        ious = []
        for i in range(K):
            for j in range(i + 1, K):
                mask_i = masks_binary[i].flatten()  # (H*W,)
                mask_j = masks_binary[j].flatten()  # (H*W,)
                intersection = (mask_i * mask_j).sum()
                union = (mask_i + mask_j).clamp(0, 1).sum()
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0.0
                ious.append(float(iou))
        if len(ious) == 0:
            return None
        ious_tensor = torch.tensor(ious)
        return {
            "shape": [int(K), int(masks_khw.shape[1]), int(masks_khw.shape[2])],
            "mean": float(ious_tensor.mean().item()),
            "min": float(ious_tensor.min().item()),
            "max": float(ious_tensor.max().item()),
            "std": float(ious_tensor.std().item()),
        }

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        sam_segs_list: List[torch.FloatTensor],
        sam_ious_list: List[torch.FloatTensor],
        sam_iops_list: List[torch.FloatTensor],
        depths: List[torch.FloatTensor] = None,
        inference: bool = False,
        **kwargs,
    ):
        # 训练阶段可选返回可视化所需字段（避免为了可视化额外跑一次 inference forward）
        return_vis = bool(kwargs.pop("return_vis", False))
        debug_train_shapes = bool(kwargs.pop("debug_train_shapes", False))
        debug_epoch = kwargs.pop("debug_epoch", None)
        # image_embeddings = self.get_visual_embs(images)

        # === 原图 -> 视觉特征（供 mask_pooling 使用） ===
        dino_embeds = self.get_dinov2_visual_embs(images)
        image_embeddings = self.model.lisa_dino_conv(dino_embeds)

        #import pdb; pdb.set_trace()
        
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # import pdb; pdb.set_trace()

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            # 推理阶段也需要支持 batch>1（训练可视化会用 batch_size>1）。
            # 使用 offset 将每张图对应的 images_clip 复制到该图的对话轮次范围内，
            # 使 images_clip 的 batch 维与 input_ids / attention_masks 对齐。
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # forward of LLaVA   
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        # `output_hidden_states` is expected to be a tuple/list (per-layer),
        # but some code paths may return a single tensor. Handle both safely.
        if isinstance(output_hidden_states, (tuple, list)):
            last_layer_hs = output_hidden_states[-1]
        else:
            last_layer_hs = output_hidden_states

        # Some rare paths may accidentally drop batch dim when batch_size==1.
        # Align it back to [B, L, H] to match `seg_token_mask` ([B, L]).
        if last_layer_hs is not None and last_layer_hs.dim() == 2 and input_ids.dim() == 2 and input_ids.shape[0] == 1:
            last_layer_hs = last_layer_hs.unsqueeze(0)

        hidden_states.append(self.model.text_hidden_fcs[0](last_layer_hs))

        # import pdb; pdb.set_trace()

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        # Debug print (once) to help track shape issues in validation/inference.
        if not hasattr(self, "_llmseg_shape_debug_printed"):
            try:
                print(
                    "[ShapeDebug] input_ids:",
                    tuple(input_ids.shape),
                    "attention_masks:",
                    tuple(attention_masks.shape),
                    "seg_token_mask:",
                    tuple(seg_token_mask.shape),
                    "last_hidden_state:",
                    tuple(last_hidden_state.shape),
                    "output_hidden_states_type:",
                    type(output_hidden_states),
                    flush=True,
                )
            except Exception:
                pass
            self._llmseg_shape_debug_printed = True

        # Make boolean indexing robust for both [B, L, D] and [L, D] hidden states.
        seg_mask = seg_token_mask
        if last_hidden_state.dim() == 2 and seg_mask.dim() == 2 and seg_mask.shape[0] == 1:
            seg_mask = seg_mask[0]
        # print(f"DEBUG: input_ids.shape={input_ids.shape}")
        # print(f"DEBUG: seg_token_mask.shape={seg_token_mask.shape}")
        # print(f"DEBUG: last_hidden_state.shape={last_hidden_state.shape}")
        # print(f"DEBUG: seg_mask.shape={seg_mask.shape}")

        pred_embeddings = last_hidden_state[seg_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # import pdb; pdb.set_trace()

        # align sam proposals and pred_embeddings
        # sam_segs_list: list of (K, 64, 64), the length is batch_size, K is the number of SAM proposals
        # sam_ious_list: list of (K), the length is batch_size
        # pred_embeddings: list of (C, D) C is the number of conversations, D is the embedding dimension=256

        # upsample the image_embedding to 256x256
        # do not support fp16, because interpolate does not support fp16
        # see the disucssion here: https://github.com/pytorch/pytorch/issues/88536
        # first convert to float32
        # 记录 interpolate 前后的维度（用于“每一步都要”的维度追踪）
        image_embeddings_pre_interp_shape = tuple(image_embeddings.shape)
        image_embeddings_pre_interp_dtype = image_embeddings.dtype

        origin_dtype = image_embeddings.dtype
        image_embeddings = image_embeddings.to(dtype=torch.float32)
        image_embeddings = F.interpolate(image_embeddings, size=(256, 256), mode="bilinear", align_corners=False)
        # convert back to original dtype
        image_embeddings = image_embeddings.to(dtype=origin_dtype)

        image_embeddings_post_interp_shape = tuple(image_embeddings.shape)

        # ✅ 训练阶段：每个 epoch 只打印一次“从进入模型到 loss”的关键维度链路（验证/推理不打印）
        if debug_train_shapes and (not inference):
            try:
                if not hasattr(self, "_debug_epoch_printed"):
                    self._debug_epoch_printed = set()
                epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                if epoch_key not in self._debug_epoch_printed:
                    print("\n" + "=" * 90, flush=True)
                    print(f"[debug] 模型-训练维度链路(epoch={epoch_key}) | batch_size={batch_size}", flush=True)
                    # 每个 epoch 固定随机选 3 个样本做更详细的打印（不足3则全选）
                    try:
                        import random as _py_random
                        k = 3 if batch_size >= 3 else batch_size
                        rng = _py_random.Random(int(epoch_key) + 12345)
                        chosen = sorted(rng.sample(list(range(batch_size)), k=k)) if batch_size > 0 else []
                    except Exception:
                        chosen = [0] if batch_size > 0 else []
                    self._debug_epoch_sample_indices = getattr(self, "_debug_epoch_sample_indices", {})
                    self._debug_epoch_sample_indices[int(epoch_key)] = chosen
                    print(f"[debug] 本epoch随机选中的batch内样本索引: {chosen}", flush=True)
                    print(f"[debug] 第1步-模型输入(原图tensor) | images.shape={tuple(images.shape)} | dtype={images.dtype}", flush=True)
                    print(f"[debug] 第2步-模型输入(CLIP图像) | images_clip.shape={tuple(images_clip.shape)} | dtype={images_clip.dtype}", flush=True)
                    print(f"[debug] 第3步-模型输入(文本) | input_ids.shape={tuple(input_ids.shape)} | attention_masks.shape={tuple(attention_masks.shape)}", flush=True)
                    print(f"[debug] 第4步-offset展开 | offset={offset.detach().cpu().tolist() if isinstance(offset, torch.Tensor) else offset}", flush=True)
                    # 原图->dinov2->conv 的输出（插值前）
                    try:
                        print(f"[debug] 第5步-视觉特征(dinov2输出) | shape={tuple(dino_embeds.shape)} | dtype={dino_embeds.dtype}", flush=True)
                    except Exception:
                        pass
                    print(f"[debug] 第6步-视觉特征(conv后,插值前) | shape={image_embeddings_pre_interp_shape} | dtype={image_embeddings_pre_interp_dtype}", flush=True)
                    print(f"[debug] 第7步-视觉特征(插值到256x256后) | shape={image_embeddings_post_interp_shape} | dtype={image_embeddings.dtype}", flush=True)
                    # sam_segs_list: list of (K, 256, 256)
                    if isinstance(sam_segs_list, list) and len(sam_segs_list) > 0 and hasattr(sam_segs_list[0], "shape"):
                        print(f"[debug] 第8步-候选掩码进入模型 | sam_segs_list[0].shape={tuple(sam_segs_list[0].shape)} (K,256,256)", flush=True)
                    if isinstance(masks_list, list) and len(masks_list) > 0 and hasattr(masks_list[0], "shape"):
                        print(f"[debug] 第9步-GT掩码进入模型 | gt_masks_list[0].shape={tuple(masks_list[0].shape)}", flush=True)
                    if isinstance(sam_ious_list, list) and len(sam_ious_list) > 0 and hasattr(sam_ious_list[0], "shape"):
                        print(f"[debug] 第10步-候选掩码与GT IoU | sam_ious_list[0].shape={tuple(sam_ious_list[0].shape)}", flush=True)
                    if isinstance(sam_iops_list, list) and len(sam_iops_list) > 0 and hasattr(sam_iops_list[0], "shape"):
                        print(f"[debug] 第11步-候选掩码与GT IoP | sam_iops_list[0].shape={tuple(sam_iops_list[0].shape)}", flush=True)
                    self._debug_epoch_printed.add(epoch_key)
                    print("=" * 90 + "\n", flush=True)
            except Exception as e:
                print("[TrainShapePipeline] debug print failed:", repr(e), flush=True)

        # mask pooling
        sam_segs_feature_list = []
        sam_pred_ious_list = []
        for batch_idx in range(len(sam_segs_list)):
            segs = sam_segs_list[batch_idx]
            # ✅ 重要：mask_pooling 的 weight_maps 期望“前景权重越大贡献越大”。
            # 但机器人手臂数据目前的 mask 语义是：前景(掩码)=0，背景=1（训练可视化里也按 ==0 作为掩码区域）。
            # 若直接把 segs 当权重，会变成在“背景”上 pooling，导致不同候选 mask 的特征非常相似，
            # 进而出现 pred_similarity 区分度低、AlignLoss 难下降的问题。
            # 因此这里将权重转换为“前景权重”：segs_fg = 1 - segs（反转mask：掩码区域=1，背景区域=0）。
            # 注意：使用 1 - segs 而不是 (segs < 0.5)，可以保留插值后的平滑过渡信息。
            if isinstance(segs, torch.Tensor):
                segs_fg = 1 - segs
            else:
                segs_fg = 1 - segs
            
            # ✅ 打印初始候选mask之间的相似度（在mask_pooling之前）
            if debug_train_shapes and (not inference):
                try:
                    epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                    chosen = getattr(self, "_debug_epoch_sample_indices", {}).get(int(epoch_key), [0])
                    if batch_idx in chosen:
                        if not hasattr(self, "_debug_epoch_mask_iou_printed"):
                            self._debug_epoch_mask_iou_printed = {}
                        printed_mask_iou = self._debug_epoch_mask_iou_printed.get(int(epoch_key), set())
                        if batch_idx not in printed_mask_iou:
                            # 计算初始候选mask之间的IoU相似度
                            stats_mask_iou = self._mask_iou_stats_khw(segs_fg)
                            if stats_mask_iou is not None:
                                print(f"[debug] 第11步-初始候选mask相似度(IoU) | sample(batch_idx={batch_idx}) | "
                                      f"shape={stats_mask_iou['shape']} | "
                                      f"mean={stats_mask_iou['mean']:.6f} min={stats_mask_iou['min']:.6f} "
                                      f"max={stats_mask_iou['max']:.6f} std={stats_mask_iou['std']:.6f}", flush=True)
                                if stats_mask_iou["mean"] > 0.8:
                                    print(f"[debug] 注意-初始候选mask过于相似 | mean IoU={stats_mask_iou['mean']:.3f}>0.8（可能候选mask本身重复度高）", flush=True)
                                elif stats_mask_iou["mean"] < 0.3:
                                    print(f"[debug] 信息-初始候选mask区分度较好 | mean IoU={stats_mask_iou['mean']:.3f}<0.3", flush=True)
                            printed_mask_iou.add(batch_idx)
                            self._debug_epoch_mask_iou_printed[int(epoch_key)] = printed_mask_iou
                except Exception as e:
                    pass
            
            segs_feature = self.mask_pooling(image_embeddings[batch_idx], segs_fg)

            # 进一步打印 pooling 细节（只在该 epoch 第一次；只看 batch_idx=0，避免刷屏）
            if debug_train_shapes and (not inference):
                try:
                    epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                    chosen = getattr(self, "_debug_epoch_sample_indices", {}).get(int(epoch_key), [0])
                    # 如果启用了“随机3张”，就不要再重复打印 sample0 的那一套
                    if hasattr(self, "_debug_epoch_printed") and epoch_key in self._debug_epoch_printed and batch_idx == 0 and len(chosen) == 1 and chosen[0] == 0:
                        # 只在首次打印中补充一次更细节的统计
                        if not hasattr(self, "_debug_epoch_pool_stats_printed"):
                            self._debug_epoch_pool_stats_printed = set()
                        if epoch_key not in self._debug_epoch_pool_stats_printed:
                            ws = segs_fg
                            ws_sum = ws.sum(dim=(1, 2)) if isinstance(ws, torch.Tensor) else None
                            if ws_sum is not None:
                                # 计算mask面积占比（前景像素数/总像素数）
                                total_pixels = ws.shape[1] * ws.shape[2]  # H * W
                                area_ratios = ws_sum.float() / total_pixels  # (K,)
                                print(f"[debug] 第12步-mask_pooling输入前景像素数统计(sample0) | "
                                      f"min={float(ws_sum.min().item()):.1f} max={float(ws_sum.max().item()):.1f} "
                                      f"mean={float(ws_sum.float().mean().item()):.1f}", flush=True)
                                print(f"[debug] 第13步-mask_pooling输入前景占比(sample0, 256x256) | "
                                      f"min={float(area_ratios.min().item()):.4f} max={float(area_ratios.max().item()):.4f} "
                                      f"mean={float(area_ratios.mean().item()):.4f}", flush=True)
                            print(f"[debug] 第14步-mask_pooling输出特征 | segs_feature.shape={tuple(segs_feature.shape)} (K, D)", flush=True)
                            self._debug_epoch_pool_stats_printed.add(epoch_key)
                except Exception:
                    pass

            # ✅ 额外：对本epoch随机选中的 3 个样本分别打印 mask 前景占比与 pooling 输出维度（每个epoch只打印一次）
            if debug_train_shapes and (not inference):
                try:
                    epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                    chosen = getattr(self, "_debug_epoch_sample_indices", {}).get(int(epoch_key), [0])
                    if batch_idx in chosen:
                        if not hasattr(self, "_debug_epoch_pool_stats_printed_multi"):
                            self._debug_epoch_pool_stats_printed_multi = {}
                        printed_set = self._debug_epoch_pool_stats_printed_multi.get(int(epoch_key), set())
                        if batch_idx not in printed_set:
                            # 额外打印：进入 mask_pooling 前，原图特征与候选mask的输入维度
                            print(f"[debug] 第12步-mask_pooling输入(原图特征) | sample(batch_idx={batch_idx}) | image_embeddings.shape={tuple(image_embeddings[batch_idx].shape)}", flush=True)
                            print(f"[debug] 第13步-mask_pooling输入(候选mask) | sample(batch_idx={batch_idx}) | segs.shape={tuple(segs.shape)} | segs_fg.shape={tuple(segs_fg.shape)}", flush=True)

                            ws = segs_fg
                            if isinstance(ws, torch.Tensor):
                                ws_sum = ws.sum(dim=(1, 2))  # (K,)
                                total_pixels = ws.shape[1] * ws.shape[2]
                                area_ratios = ws_sum.float() / total_pixels
                                print(f"[debug] 第14步-mask_pooling输入前景像素数统计 | sample(batch_idx={batch_idx}) | min={float(ws_sum.min().item()):.1f} max={float(ws_sum.max().item()):.1f} mean={float(ws_sum.float().mean().item()):.1f}", flush=True)
                                print(f"[debug] 第15步-mask_pooling输入前景占比(256x256) | sample(batch_idx={batch_idx}) | min={float(area_ratios.min().item()):.4f} max={float(area_ratios.max().item()):.4f} mean={float(area_ratios.mean().item()):.4f}", flush=True)
                            print(f"[debug] 第16步-mask_pooling输出特征 | sample(batch_idx={batch_idx}) | segs_feature.shape={tuple(segs_feature.shape)} (K, D)", flush=True)
                            printed_set.add(batch_idx)
                            self._debug_epoch_pool_stats_printed_multi[int(epoch_key)] = printed_set
                except Exception:
                    pass

            # ✅ 打印mask_pooling之后的特征相似度（并保存用于总结）
            if debug_train_shapes and (not inference):
                try:
                    epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                    chosen = getattr(self, "_debug_epoch_sample_indices", {}).get(int(epoch_key), [0])
                    if batch_idx in chosen:
                        if not hasattr(self, "_debug_epoch_masksim_printed_multi"):
                            self._debug_epoch_masksim_printed_multi = {}
                        if not hasattr(self, "_debug_epoch_sim_stats_cache"):
                            self._debug_epoch_sim_stats_cache = {}
                        printed = self._debug_epoch_masksim_printed_multi.get(int(epoch_key), set())
                        if batch_idx not in printed:
                            stats_pool = self._cosine_sim_stats_kd(segs_feature)
                            # 保存用于总结
                            cache_key = f"{epoch_key}_{batch_idx}"
                            if cache_key not in self._debug_epoch_sim_stats_cache:
                                self._debug_epoch_sim_stats_cache[cache_key] = {}
                            self._debug_epoch_sim_stats_cache[cache_key]["pool"] = stats_pool
                            if stats_pool is not None:
                                print(f"[debug] 第17步-mask_pooling后特征相似度 | sample(batch_idx={batch_idx}) | "
                                      f"shape={stats_pool['shape']} | "
                                      f"mean={stats_pool['mean']:.6f} min={stats_pool['min']:.6f} "
                                      f"max={stats_pool['max']:.6f} std={stats_pool['std']:.6f}", flush=True)
                                if stats_pool["mean"] > 0.95:
                                    print(f"[debug] ⚠️ 问题-mask_pooling后特征过于相似 | mean={stats_pool['mean']:.3f}>0.95（可能mask_pooling有问题）", flush=True)
                                elif stats_pool["mean"] < 0.5:
                                    print(f"[debug] ✅ 信息-mask_pooling后特征区分度较好 | mean={stats_pool['mean']:.3f}<0.5", flush=True)
                except Exception:
                    pass
            if not hasattr(self, "_segs_fg_semantic_check_printed"):
                try:
                    area_ratio = segs_fg.float().flatten(1).mean(dim=1)
                    print(
                        "[MaskSemanticCheck] assuming SAM candidates use 0=foreground, 1=background; "
                        f"after inversion segs_fg foreground-area ratio: "
                        f"min={float(area_ratio.min().item()):.6f} "
                        f"mean={float(area_ratio.mean().item()):.6f} "
                        f"max={float(area_ratio.max().item()):.6f}",
                        flush=True,
                    )
                    if float(area_ratio.mean().item()) > 0.8:
                        print("[MaskSemanticCheck] WARNING: segs_fg mean area is very large; re-check mask polarity.", flush=True)
                    if float(area_ratio.max().item()) <= 1e-6:
                        print("[MaskSemanticCheck] WARNING: all candidate foreground masks are empty after inversion.", flush=True)
                except Exception:
                    pass
                self._segs_fg_semantic_check_printed = True

            depth_map = None
            if depths is not None and batch_idx < len(depths):
                depth_map = depths[batch_idx]
            proposal_valid = self.model.lisa_geo_prior.proposal_valid_mask(segs_fg)
            geo_attn_bias = self.model.lisa_geo_prior(segs_fg, depth_map)

            # use attention to update the mask feature, keep text embedding unchanged
            text_feature = pred_embeddings[batch_idx] # (C, D)
            # # add one dimension to text_feature （C, 1, D）
            text_feature = text_feature.unsqueeze(1)

            number_conversations = text_feature.shape[0]

            segs_feature = segs_feature.unsqueeze(0) # (1, K, D)
            proposal_valid_expanded = None
            if number_conversations > 0:
                # expand segs_feature to (C, K, D)
                segs_feature = segs_feature.expand(number_conversations, -1, -1)
                if proposal_valid is not None:
                    proposal_valid_expanded = proposal_valid.unsqueeze(0).expand(number_conversations, -1)
                    segs_feature = segs_feature * proposal_valid_expanded.unsqueeze(-1).to(segs_feature.dtype)
                if geo_attn_bias is not None:
                    geo_attn_bias = geo_attn_bias.expand(number_conversations, -1, -1, -1)

            # import pdb; pdb.set_trace()    

            for layer in self.model.lisa_attention_layers:
                segs_feature, text_feature = layer(
                    queries=segs_feature,
                    keys=text_feature,
                    self_attn_bias=geo_attn_bias,
                    query_valid_mask=proposal_valid_expanded,
                )

            attn_out = self.model.lisa_final_attn(q=segs_feature, k=text_feature, v=text_feature)
            segs_feature = segs_feature + attn_out
            segs_feature = self.model.lisa_norm_final_attn(segs_feature)
            if proposal_valid_expanded is not None:
                segs_feature = segs_feature * proposal_valid_expanded.unsqueeze(-1).to(segs_feature.dtype)

            if debug_train_shapes and (not inference):
                try:
                    epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                    chosen = getattr(self, "_debug_epoch_sample_indices", {}).get(int(epoch_key), [0])
                    if batch_idx in chosen:
                        if not hasattr(self, "_debug_epoch_masksim_printed_multi"):
                            self._debug_epoch_masksim_printed_multi = {}
                        if not hasattr(self, "_debug_epoch_sim_stats_cache"):
                            self._debug_epoch_sim_stats_cache = {}
                        printed = self._debug_epoch_masksim_printed_multi.get(int(epoch_key), set())
                        if batch_idx not in printed:
                            print(f"[debug] 第18步-attention后mask特征 | sample(batch_idx={batch_idx}) | segs_feature.shape={tuple(segs_feature.shape)} (C, K, D)", flush=True)
                            stats_attn = self._cosine_sim_stats_kd(segs_feature[0])
                            # 保存用于总结
                            cache_key = f"{epoch_key}_{batch_idx}"
                            if cache_key not in self._debug_epoch_sim_stats_cache:
                                self._debug_epoch_sim_stats_cache[cache_key] = {}
                            self._debug_epoch_sim_stats_cache[cache_key]["attn"] = stats_attn
                            if stats_attn is not None:
                                print(f"[debug] 第19步-attention后mask特征相似度 | sample(batch_idx={batch_idx}) | "
                                      f"shape={stats_attn['shape']} | "
                                      f"mean={stats_attn['mean']:.6f} min={stats_attn['min']:.6f} "
                                      f"max={stats_attn['max']:.6f} std={stats_attn['std']:.6f}", flush=True)
                                if stats_attn["mean"] > 0.95:
                                    print(f"[debug] ⚠️ 问题-attention后mask特征过于相似 | mean={stats_attn['mean']:.3f}>0.95（可能attention导致特征collapse）", flush=True)
                                elif stats_attn["mean"] < 0.5:
                                    print(f"[debug] ✅ 信息-attention后mask特征区分度较好 | mean={stats_attn['mean']:.3f}<0.5", flush=True)
                except Exception:
                    pass

            # use MLP to reduce the seg_features to 1 dimension
            sam_iou = self.model.lisa_iou_head(segs_feature) # (C, K, 1)
            if proposal_valid_expanded is not None:
                sam_iou = sam_iou.masked_fill(~proposal_valid_expanded.unsqueeze(-1), 0.0)
            sam_pred_ious_list.append(sam_iou) # (C, K, 1)

            segs_feature = self.model.lisa_embedding_head(segs_feature) # (C, K, D)
            if proposal_valid_expanded is not None:
                segs_feature = segs_feature * proposal_valid_expanded.unsqueeze(-1).to(segs_feature.dtype)
            sam_segs_feature_list.append(segs_feature)

            if debug_train_shapes and (not inference):
                try:
                    epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                    chosen = getattr(self, "_debug_epoch_sample_indices", {}).get(int(epoch_key), [0])
                    if batch_idx in chosen:
                        if not hasattr(self, "_debug_epoch_masksim_printed_multi"):
                            self._debug_epoch_masksim_printed_multi = {}
                        printed = self._debug_epoch_masksim_printed_multi.get(int(epoch_key), set())
                        if batch_idx not in printed:
                            print(f"[debug] 第20步-IoU回归头输出 | sample(batch_idx={batch_idx}) | sam_iou.shape={tuple(sam_iou.shape)} (C, K, 1)", flush=True)
                            print(f"[debug] 第21步-embedding_head输出特征 | sample(batch_idx={batch_idx}) | segs_feature.shape={tuple(segs_feature.shape)} (C, K, D)", flush=True)
                            stats_final = self._cosine_sim_stats_kd(segs_feature[0])
                            if stats_final is not None:
                                print(f"[debug] 第22步-最终mask特征相似度 | sample(batch_idx={batch_idx}) | "
                                      f"shape={stats_final['shape']} | "
                                      f"mean={stats_final['mean']:.6f} min={stats_final['min']:.6f} "
                                      f"max={stats_final['max']:.6f} std={stats_final['std']:.6f}", flush=True)
                                if stats_final["mean"] > 0.95:
                                    print(f"[debug] ⚠️ 问题-最终mask特征过于相似 | mean={stats_final['mean']:.3f}>0.95（可能attention或embedding_head导致特征collapse）", flush=True)
                                elif stats_final["mean"] > 0.85:
                                    print(f"[debug] ⚠️ 注意-最终mask特征相似度偏高 | mean={stats_final['mean']:.3f}>0.85", flush=True)
                                else:
                                    print(f"[debug] ✅ 信息-最终mask特征区分度较好 | mean={stats_final['mean']:.3f}<=0.85", flush=True)
                            
                            # 打印相似度变化总结（从缓存中获取之前步骤的统计信息）
                            print(f"\n[debug] ========== 相似度变化总结 (sample batch_idx={batch_idx}) ==========", flush=True)
                            try:
                                cache_key = f"{epoch_key}_{batch_idx}"
                                cache = getattr(self, "_debug_epoch_sim_stats_cache", {}).get(cache_key, {})
                                
                                # 1. 初始mask IoU（重新计算，因为segs_fg还在）
                                stats_mask_iou = self._mask_iou_stats_khw(segs_fg)
                                if stats_mask_iou:
                                    print(f"[debug] 步骤1-初始候选mask IoU相似度: mean={stats_mask_iou['mean']:.6f}", flush=True)
                                
                                # 2. mask_pooling后特征相似度（从缓存获取）
                                stats_pool = cache.get("pool")
                                if stats_pool:
                                    print(f"[debug] 步骤2-mask_pooling后特征相似度: mean={stats_pool['mean']:.6f}", flush=True)
                                
                                # 3. attention后特征相似度（从缓存获取）
                                stats_attn = cache.get("attn")
                                if stats_attn:
                                    print(f"[debug] 步骤3-attention后特征相似度: mean={stats_attn['mean']:.6f}", flush=True)
                                
                                # 4. embedding_head后特征相似度（最终）
                                if stats_final:
                                    print(f"[debug] 步骤4-embedding_head后特征相似度(最终): mean={stats_final['mean']:.6f}", flush=True)
                                
                                # 打印变化趋势
                                print(f"[debug] 相似度变化趋势:", flush=True)
                                if stats_mask_iou:
                                    print(f"[debug]   初始mask IoU: {stats_mask_iou['mean']:.6f}", flush=True)
                                if stats_pool:
                                    if stats_mask_iou:
                                        print(f"[debug]   mask_pooling后: {stats_pool['mean']:.6f} (相对初始mask变化: {stats_pool['mean'] - stats_mask_iou['mean']:+.6f})", flush=True)
                                    else:
                                        print(f"[debug]   mask_pooling后: {stats_pool['mean']:.6f}", flush=True)
                                if stats_attn:
                                    if stats_pool:
                                        print(f"[debug]   attention后: {stats_attn['mean']:.6f} (相对pooling变化: {stats_attn['mean'] - stats_pool['mean']:+.6f})", flush=True)
                                    else:
                                        print(f"[debug]   attention后: {stats_attn['mean']:.6f}", flush=True)
                                if stats_final:
                                    if stats_attn:
                                        print(f"[debug]   embedding_head后: {stats_final['mean']:.6f} (相对attention变化: {stats_final['mean'] - stats_attn['mean']:+.6f})", flush=True)
                                    elif stats_pool:
                                        print(f"[debug]   embedding_head后: {stats_final['mean']:.6f} (相对pooling变化: {stats_final['mean'] - stats_pool['mean']:+.6f})", flush=True)
                                    else:
                                        print(f"[debug]   embedding_head后: {stats_final['mean']:.6f}", flush=True)
                                
                                # 诊断信息
                                if stats_final and stats_pool:
                                    if stats_final['mean'] > stats_pool['mean'] + 0.1:
                                        print(f"[debug] ⚠️ 警告: embedding_head后相似度显著增加 ({stats_final['mean']:.3f} > {stats_pool['mean']:.3f} + 0.1)，可能特征collapse", flush=True)
                                    elif stats_final['mean'] < stats_pool['mean'] - 0.1:
                                        print(f"[debug] ✅ 信息: embedding_head后相似度降低，特征区分度提升", flush=True)
                                if stats_attn and stats_pool:
                                    if stats_attn['mean'] > stats_pool['mean'] + 0.1:
                                        print(f"[debug] ⚠️ 警告: attention后相似度显著增加 ({stats_attn['mean']:.3f} > {stats_pool['mean']:.3f} + 0.1)，可能attention导致特征collapse", flush=True)
                            except Exception as e:
                                print(f"[debug] 相似度总结计算失败: {repr(e)}", flush=True)
                            print(f"[debug] ============================================================\n", flush=True)
                            
                            # 标记该 sample 已输出（每个epoch仅对 chosen 的样本各输出一次）
                            printed.add(batch_idx)
                            self._debug_epoch_masksim_printed_multi[int(epoch_key)] = printed
                except Exception as e:
                    pass


        if inference:
            # during inference , C = 1
            pred_similarity = []
            for batch_idx in range(len(pred_embeddings)):
                pred_embedding = pred_embeddings[batch_idx] # (1, D)
                pred_embedding_normlized = pred_embedding / pred_embedding.norm(dim=-1, keepdim=True)
                sam_features = sam_segs_feature_list[batch_idx][0, :, :] # (K, D)
                sam_features_normlized = sam_features / sam_features.norm(dim=-1, keepdim=True)
                similarity = pred_embedding_normlized @ sam_features_normlized.T # (1, K)
                pred_similarity.append(similarity)

            pred_ious = []
            for batch_idx in range(len(sam_pred_ious_list)):
                sam_pred_ious = sam_pred_ious_list[batch_idx][0, :, :] # (K, 1)
                pred_ious.append(sam_pred_ious.T) # (1, K)

            return {
                "pred_similarity": pred_similarity,
                "gt_masks": masks_list,
                "pred_iou": pred_ious,
            }

        ce_loss = output.loss  # loss of LLaVA

        # ✅ 打印loss计算时的关键维度（每个epoch只打印一次）
        if debug_train_shapes and (not inference):
            try:
                epoch_key = int(debug_epoch) if debug_epoch is not None else -1
                if not hasattr(self, "_debug_epoch_loss_printed"):
                    self._debug_epoch_loss_printed = set()
                if epoch_key not in self._debug_epoch_loss_printed:
                    if len(pred_embeddings) > 0 and len(sam_segs_feature_list) > 0:
                        print(f"[debug] 第18步-loss计算输入维度(batch_idx=0):", flush=True)
                        print(f"[debug]  - pred_embeddings[0].shape={tuple(pred_embeddings[0].shape)} (C, D)", flush=True)
                        print(f"[debug]  - sam_segs_feature_list[0].shape={tuple(sam_segs_feature_list[0].shape)} (C, K, D)", flush=True)
                        if len(sam_ious_list) > 0:
                            print(f"[debug]  - sam_ious_list[0].shape={tuple(sam_ious_list[0].shape)} (R, K)", flush=True)
                        if len(sam_iops_list) > 0:
                            print(f"[debug]  - sam_iops_list[0].shape={tuple(sam_iops_list[0].shape)} (R, K)", flush=True)
                        if len(sam_pred_ious_list) > 0:
                            print(f"[debug]  - sam_pred_ious_list[0].shape={tuple(sam_pred_ious_list[0].shape)} (C, K, 1)", flush=True)
                        print(f"[debug]  - ce_loss.shape={tuple(ce_loss.shape) if hasattr(ce_loss, 'shape') else 'scalar'}", flush=True)
                    self._debug_epoch_loss_printed.add(epoch_key)
            except Exception as e:
                print(f"[TrainShapePipeline] loss维度打印失败: {repr(e)}", flush=True)

        # compute align loss
        align_loss = 0.0 
        regression_loss = 0.0

        valid_batch = 0
        for batch_idx in range(len(sam_segs_feature_list)):
            segs_feature = sam_segs_feature_list[batch_idx]  # (C, K, D) D=256
            gt_iou = sam_ious_list[batch_idx]            # (R,K)
            gt_iop = sam_iops_list[batch_idx]            # (R,K)
            pred_iou = sam_pred_ious_list[batch_idx]
            
            # Convert to tensor if numpy array
            if isinstance(gt_iou, np.ndarray):
                gt_iou = torch.from_numpy(gt_iou)
            if isinstance(gt_iop, np.ndarray):
                gt_iop = torch.from_numpy(gt_iop) 
            # # multiple conversations
            # assert gt_iou.shape[0] == pred_embeddings[batch_idx].shape[0], "number of rounds mismatch, gt_iou.shape: {}, pred_embeddings.shape: {}".format(gt_iou.shape, pred_embeddings[batch_idx].shape)
            # assert gt_iou.shape[0] != 0, "number of rounds = 0; gt_iou.shape: {}".format(gt_iou.shape)
            number_rounds = pred_embeddings[batch_idx].shape[0]

            align_loss_round = 0.0
            regression_losss_round = 0.0
            if number_rounds == 0:
                # throw an error 
                raise ValueError("number of rounds = 0; gt_iou.shape: {}".format(gt_iou.shape))

            for round_idx in range(number_rounds):
                gt_iou_round = gt_iou[round_idx] # (K)
                gt_iop_round = gt_iop[round_idx] # (K)
                
                # Ensure they are tensors (indexing may return numpy in some cases)
                if isinstance(gt_iou_round, np.ndarray):
                    gt_iou_round = torch.from_numpy(gt_iou_round)
                if isinstance(gt_iop_round, np.ndarray):
                    gt_iop_round = torch.from_numpy(gt_iop_round)
                
                gt_iou_round = gt_iou_round.unsqueeze(1) # (K, 1)
                gt_iou_round = gt_iou_round.to(dtype=pred_iou.dtype, device=pred_iou.device)
                gt_iop_round = gt_iop_round.unsqueeze(1).to(dtype=pred_iou.dtype, device=pred_iou.device) # (K, 1)

                target_embedding = pred_embeddings[batch_idx][round_idx].unsqueeze(0)  # (1, D)
                # align_loss_round += sigmoid_align_loss(segs_feature, target_embedding, gt_iou_round, 
                #                                        self.model.temperature, self.model.bias)
                align_loss_round += softmax_align_loss(segs_feature[round_idx], target_embedding, gt_iou_round)
                regression_losss_round += iou_regression_loss(pred_iou[round_idx], gt_iop_round)
            
            if number_rounds > 0:
                valid_batch += 1

            align_loss +=  align_loss_round / (number_rounds + 1e-8)
            regression_loss += regression_losss_round / (number_rounds + 1e-8)

        if valid_batch > 0:
            align_loss = align_loss / valid_batch
            regression_loss = regression_loss / valid_batch

        # regression_loss = regression_loss.to(ce_loss.dtype)
        # loss = self.ce_loss_weight * ce_loss +  self.align_loss_weight * align_loss + self.regression_loss_weight * regression_loss
        
        ce_loss = ce_loss * self.ce_loss_weight
        align_loss = align_loss * self.align_loss_weight
        regression_loss = regression_loss * self.regression_loss_weight
        loss = ce_loss + align_loss + regression_loss

        out = {
            "loss": loss,
            "ce_loss": ce_loss,
            "align_loss": align_loss,
            "regression_loss": regression_loss,
        }

        # ✅ 参考 finetune_llmseg_copy.py：训练过程中做可视化时，直接复用训练 forward 的输出
        # 返回与 inference 分支一致的字段格式（list，元素 shape 为 (1,K)），方便训练脚本直接 argmax。
        if return_vis:
            pred_similarity = []
            for batch_idx in range(len(pred_embeddings)):
                pred_embedding = pred_embeddings[batch_idx]
                # 多轮对话时取第0轮；一般训练数据是一轮
                if isinstance(pred_embedding, torch.Tensor) and pred_embedding.dim() == 2 and pred_embedding.shape[0] > 0:
                    pred_embedding = pred_embedding[0:1]  # (1, D)
                pred_embedding_normlized = pred_embedding / (pred_embedding.norm(dim=-1, keepdim=True) + 1e-8)

                # sam_segs_feature_list: (C, K, D)，取第0轮
                sam_features = sam_segs_feature_list[batch_idx][0, :, :]  # (K, D)
                sam_features_normlized = sam_features / (sam_features.norm(dim=-1, keepdim=True) + 1e-8)
                similarity = pred_embedding_normlized @ sam_features_normlized.T  # (1, K)
                pred_similarity.append(similarity)

            pred_ious = []
            for batch_idx in range(len(sam_pred_ious_list)):
                # sam_pred_ious_list: (C, K, 1)，取第0轮并转为 (1, K)
                sam_pred_ious = sam_pred_ious_list[batch_idx][0, :, :]  # (K, 1)
                pred_ious.append(sam_pred_ious.T)  # (1, K)

            out.update(
                {
                    "pred_similarity": pred_similarity,
                    "gt_masks": masks_list,
                    "pred_iou": pred_ious,
                }
            )

        return out


    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
