import argparse
import json
import os 
import shutil
import sys
import time
from functools import partial
import copy

import deepspeed 
import numpy as np
import torch 
import tqdm 
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import cv2
import random

# 导入 SwanLab（现在应该可以正常工作了）
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except (ImportError, AttributeError, NameError) as e:
    swanlab = None
    SWANLAB_AVAILABLE = False
    print(f"Warning: swanlab not available ({e}), training will continue without SwanLab logging")

# ✅ 全局变量：收集维度追踪信息（用于记录到SwanLab）
_dimension_tracking_info = []

def drop_file_cache(file_path):
    """使用 posix_fadvise 告诉内核释放文件的页面缓存。
    
    这对于大文件写入后释放缓存非常有用，可以避免 cgroup 内存超限。
    """
    try:
        with open(file_path, 'rb') as f:
            # POSIX_FADV_DONTNEED = 4: 告诉内核我们不再需要这个文件的缓存
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except (AttributeError, OSError) as e:
        # Windows 或某些系统不支持 posix_fadvise
        return False

def drop_directory_cache(dir_path):
    """释放目录中所有文件的缓存"""
    dropped_count = 0
    total_size = 0
    if os.path.exists(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    fsize = os.path.getsize(fpath)
                    if drop_file_cache(fpath):
                        dropped_count += 1
                        total_size += fsize
                except:
                    pass
    return dropped_count, total_size / (1024**3)  # 返回文件数和 GB 数

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn, ValDataSet_ReasonSeg, collate_fn_new, ValDataSet_LLMSeg
from utils.llm_seg_dataset import LLMSegDataset
from utils.robot_arm_dataset import RobotArmDataset
from utils.vigor_dataset import VIGORDataset
from utils.vigor_dataset_multi_instance import VIGORDatasetMultiInstance
from utils.sam_mask_reader import SAM_Mask_Reader
from utils.sam_mask_reader_png import SAM_Mask_Reader_PNG
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=896, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="refer_seg||reason_seg", type=str
    )
    # VIGOR-100K数据集参数
    parser.add_argument("--vigor_data_base_dir", default="/opt/data/private/LLMSeg/dataset/VIGOR-100K", type=str, help="VIGOR-100K数据集根目录")
    parser.add_argument("--vigor_json_file", default="open_vocab_grasp_easy.json", type=str, help="VIGOR JSON文件名（如open_vocab_grasp_easy.json）")
    parser.add_argument("--vigor_split", default="train", type=str, help="VIGOR数据集划分（train/test/unseen）")
    parser.add_argument("--vigor_val_split", default="test", type=str, help="VIGOR验证集划分（train/test/unseen）")
    parser.add_argument("--vigor_max_samples", default=None, type=int, help="VIGOR数据集最多使用多少个样本（None表示使用全部）")
    parser.add_argument("--vigor_train_sam_masks_dir", default=None, type=str, help="VIGOR训练集SAM候选masks目录")
    parser.add_argument("--vigor_val_sam_masks_dir", default=None, type=str, help="VIGOR验证集SAM候选masks目录")
    
    parser.add_argument("--sample_rates", default="10, 1", type=str)

    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="/cluster/scratch/leikel/junchi/lisa_dataset", type=str)
    parser.add_argument("--sam_masks_dir", default="/home/leikel/junchi/processed_data", type=str)
    # 机器人手臂数据集路径参数
    parser.add_argument("--dataset_base_dir", default="/opt/data/private/LLMSeg/dataset/raw_pic", type=str, help="原图数据集路径")
    parser.add_argument("--gt_mask_base_dir", default="/opt/data/private/LLMSeg/dataset/GT_mask", type=str, help="GT mask数据集路径")
    parser.add_argument("--sam_masks_base_dir", default="/opt/data/private/LLMSeg/dataset/sam_candidate", type=str, help="SAM候选mask数据集路径")
    # 机器人手臂数据集划分参数：每视角取前N张（默认100，共≈300），再按9:1随机划分 train/val
    parser.add_argument("--robotarm_max_per_view", default=100, type=int, help="每个视角最多使用前N张（用于构建300样本池）")
    parser.add_argument("--robotarm_val_ratio", default=0.1, type=float, help="从样本池中划分到验证集的比例（默认0.1，即9:1）")
    parser.add_argument("--robotarm_split_seed", default=0, type=int, help="划分随机种子（保证可复现）")
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="debug", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--align_loss_weight", default=1.0, type=float)
    parser.add_argument("--regression_loss_weight", default=1.0, type=float)
    parser.add_argument("--align_temperature", default=0.1, type=float, help="温度参数，控制softmax分布的尖锐程度。较小的值（如0.05）会使分布更尖锐，较大的值（如0.1-0.2）会使分布更平滑，有助于梯度传播。建议从0.1开始尝试。")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/LLM-Seg-deepspeed", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--iou_selection_only", action="store_true", default=False)
    parser.add_argument("--train_vis_dir", default="train_vis", type=str, help="训练可视化保存目录（相对于log_dir）")
    parser.add_argument("--val_vis_dir", default="val_vis", type=str, help="验证可视化保存目录（相对于log_dir）")
    parser.add_argument("--eval_vis_dir", default="eval_vis_iop", type=str, help="评估可视化保存目录（相对于log_dir）")
    parser.add_argument(
        "--debug_epoch_shapes",
        action="store_true",
        default=False,
        help="每个epoch在第一个batch打印一次原图/GT/候选mask的关键维度与语义统计（不刷屏）",
    )
    # Checkpoint保存格式参数
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        default=False,
        help="保存safetensors格式的checkpoint（内存效率更高）"
    )
    parser.add_argument(
        "--save_deepspeed_only",
        action="store_true",
        default=False,
        help="只保存DeepSpeed格式，不保存safetensors"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="每隔多少个 epoch 保存一次 checkpoint（默认每个 epoch 都保存）"
    )
    return parser.parse_args(args)


def _build_robotarm_samples(
    gt_mask_base_dir: str,
    raw_pic_base_dir: str,
    max_per_view: int,
    view_names=None,
):
    """
    构建 RobotArmDataset 需要的 samples 列表：
    - 只取每个视角 annotations.json 的前 max_per_view 条
    - 不做随机划分（划分在外层完成）
    """
    if view_names is None:
        view_names = ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']

    all_samples = []
    for view_name in view_names:
        annotations_path = os.path.join(gt_mask_base_dir, view_name, 'annotations.json')
        if not os.path.exists(annotations_path):
            continue
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations[:max_per_view]

        for ann in annotations:
            img_name = ann['img_name']
            object_name = ann.get('object', 'object')
            gt_path = ann.get('gt_path', '')

            image_path = os.path.join(raw_pic_base_dir, view_name, img_name)
            gt_mask_filename = os.path.basename(gt_path.replace('\\', '/'))
            gt_mask_path = os.path.join(gt_mask_base_dir, view_name, "masks", gt_mask_filename)
            question = f"segment the {object_name}"

            all_samples.append({
                'image_path': image_path,
                'gt_mask_path': gt_mask_path,
                'question': question,
                'object': object_name,
                'view_name': view_name,
                'img_name': img_name,
            })

    return all_samples


def _split_samples_train_val(samples, val_ratio: float, seed: int):
    """
    以固定 seed 随机打乱后按比例切分。
    返回 (train_samples, val_samples)
    """
    if samples is None:
        samples = []
    n = len(samples)
    if n == 0:
        return [], []

    ratio = float(val_ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"robotarm_val_ratio 必须在 [0,1) 范围内，当前: {val_ratio}")

    idxs = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(idxs)

    val_n = int(n * ratio)
    # 只要开启验证，就至少给1个样本（避免 val_loader 为空导致后续逻辑出错）
    if val_n == 0 and ratio > 0:
        val_n = 1
    if val_n >= n:
        val_n = n - 1

    val_idxs = set(idxs[:val_n])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idxs]
    val_samples = [s for i, s in enumerate(samples) if i in val_idxs]
    return train_samples, val_samples


def init_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    _ = tokenizer.add_tokens("[SEG]")
    # 注意：在 LLaMA/SentencePiece tokenizer 中，"[SEG]" 可能会被分成多个 token（例如 "▁" + "[SEG]"）。
    # 我们需要拿到真正的 "[SEG]" token id，否则会把大量 "▁" 当成 seg token，导致 round 数量错乱。
    seg_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    args.seg_token_idx = seg_ids[-1]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    return tokenizer

def init_LISA_model(args, tokenizer):
    # ✅ 抑制 HuggingFace transformers 的权重警告信息
    import warnings
    import transformers
    from transformers import logging as transformers_logging
    
    # 设置 transformers 日志级别为 ERROR，只显示错误信息
    transformers_logging.set_verbosity_error()
    
    # 临时抑制 Python warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        model_args = {
            "train_mask_decoder": args.train_mask_decoder,
            "out_dim": args.out_dim,
            "ce_loss_weight": args.ce_loss_weight,
            "align_loss_weight": args.align_loss_weight,
            "regression_loss_weight": args.regression_loss_weight,
            "align_temperature": args.align_temperature,  # 温度参数，控制softmax分布的尖锐程度
            # "dice_loss_weight": args.dice_loss_weight,
            # "bce_loss_weight": args.bce_loss_weight,
            "seg_token_idx": args.seg_token_idx,
            "vision_pretrained": args.vision_pretrained,
            "vision_tower": args.vision_tower,
            "use_mm_start_end": args.use_mm_start_end,
            # 关键：在模型初始化前就设置 mm_vision_tower，避免使用配置中的旧值
            "mm_vision_tower": args.vision_tower,
        }
        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.half

        # 确保模型加载到 CPU，不使用 device_map（与 DeepSpeed 可能冲突）
        model = LISAForCausalLM.from_pretrained(
            # 方案A：关闭 low_cpu_mem_usage，避免残留 meta tensor 导致 DeepSpeed 初始化阶段 .to(device) 崩溃
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
        )
        
        # 双重保险：确保 config.mm_vision_tower 使用命令行传入的本地路径
        # （虽然 LISA.py 中已经处理，但这里再次确认）
        if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower != args.vision_tower:
            old_vision_tower = model.config.mm_vision_tower
            model.config.mm_vision_tower = args.vision_tower
            # 如果 vision_tower 已经初始化且路径不同，需要删除旧的让 initialize_vision_modules 重新创建
            if hasattr(model.get_model(), 'vision_tower') and old_vision_tower != args.vision_tower:
                delattr(model.get_model(), 'vision_tower')
    
    # 恢复 transformers 日志级别（可选，如果需要看到其他警告可以设置为 WARNING）
    # transformers_logging.set_verbosity_warning()

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 创建 model_args 对象用于 initialize_vision_modules
    class ModelArgs:
        def __init__(self, args_dict):
            for k, v in args_dict.items():
                setattr(self, k, v)
            # 设置默认值
            self.mm_vision_select_layer = getattr(args, 'mm_vision_select_layer', -2)
            self.mm_vision_select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
            self.pretrain_mm_mlp_adapter = getattr(args, 'pretrain_mm_mlp_adapter', None)
    
    model_args_obj = ModelArgs(model_args)
    model.get_model().initialize_vision_modules(model_args_obj)
    vision_tower = model.get_model().get_vision_tower()  # CLIP
    # 修复：使用正确的 device 格式（cuda:0, cuda:1 等）
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    vision_tower.to(dtype=torch_dtype, device=device)
    model.get_model().initialize_lisa_modules(model.get_model().config)  # SAM and others

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    # 修复mm_projector权重加载问题
    def fix_mm_projector_weights(model):
        """修复mm_projector权重加载问题"""
        import torch.nn as nn
        
        if hasattr(model.model, 'mm_projector') and isinstance(model.model.mm_projector, nn.Linear):
            # 检查是否有meta tensor
            has_meta = any(param.is_meta for param in model.model.mm_projector.parameters())
            
            if has_meta:
                print("🔧 检测到mm_projector参数为meta tensor，应用修复...")
                
                # 获取配置信息
                mm_hidden_size = model.config.mm_hidden_size
                hidden_size = model.config.hidden_size
                intermediate_size = hidden_size
                
                # 重新构建为Sequential结构
                new_mm_projector = nn.Sequential(
                    nn.Linear(mm_hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size)
                )
                
                # 加载预训练权重
                checkpoint_path = "/opt/data/private/model/LISA_Plus_7b/pytorch_model-00002-of-00002.bin"
                
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    
                    # 加载权重
                    state_dict = {
                        '0.weight': checkpoint['model.mm_projector.0.weight'],
                        '0.bias': checkpoint['model.mm_projector.0.bias'],
                        '2.weight': checkpoint['model.mm_projector.2.weight'], 
                        '2.bias': checkpoint['model.mm_projector.2.bias']
                    }
                    
                    new_mm_projector.load_state_dict(state_dict, strict=True)
                    
                    # 替换原有的mm_projector
                    model.model.mm_projector = new_mm_projector
                    
                    print("✅ mm_projector权重修复完成")
                    
                    # 验证修复结果
                    for name, param in model.model.mm_projector.named_parameters():
                        print(f"  - {name}: is_meta={param.is_meta}, device={param.device}")
                        
                except Exception as e:
                    print(f"❌ mm_projector权重修复失败: {e}")
                    raise e

    # 应用修复
    fix_mm_projector_weights(model)

    # init LoRA
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "lisa_attention_layers",
                                "lisa_final_attn",
                                "lisa_norm_final_attn",
                                "lisa_iou_head",
                                "lisa_embedding_head",
                                "lisa_dino_conv",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()  # 注释掉：打印可训练参数信息

    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs",
                          "lisa_attention_layers", "lisa_final_attn", "lisa_norm_final_attn",
                          "lisa_iou_head", "lisa_embedding_head", "lisa_dino_conv"]
            ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)  # 注释掉：打印每个参数的名称和形状
            p.requires_grad = True
    # model.print_trainable_parameters()  # 注释掉：打印可训练参数信息
    return model

def init_training_dataset(args, tokenizer):

    world_size = torch.cuda.device_count()

    # 根据数据集类型选择不同的数据集
    if args.dataset == "vigor":
        # 使用 VIGOR 数据集
        vigor_json_path = os.path.join(args.vigor_data_base_dir, args.vigor_split, args.vigor_json_file)
        
        # 创建 SAM mask helper（训练集）- 严格模式
        sam_mask_helper = None
        
        if args.vigor_train_sam_masks_dir is not None:
            if os.path.exists(args.vigor_train_sam_masks_dir):
                sam_mask_helper = SAM_Mask_Reader_PNG(args.vigor_train_sam_masks_dir)
                print(f"使用SAM候选masks: {args.vigor_train_sam_masks_dir}")
            else:
                raise FileNotFoundError(f"SAM masks目录不存在: {args.vigor_train_sam_masks_dir}")
        else:
            print("警告: 未指定SAM masks目录 (--vigor_train_sam_masks_dir)，将不使用候选masks")
        
        # 创建 VIGOR 训练数据集（使用多实例版本）
        # 总是使用混合数据集（包含easy和hard）并使用多实例模式
        print(f"模式：混合训练 (Easy + Hard) - 多实例模式 (3 instructions/image)")
        
        # 加载 Easy 和 Hard 数据
        easy_json_path = os.path.join(args.vigor_data_base_dir, args.vigor_split, "open_vocab_grasp_easy.json")
        hard_json_path = os.path.join(args.vigor_data_base_dir, args.vigor_split, "open_vocab_grasp_hard.json")
        
        print(f"Loading Easy samples from: {easy_json_path}")
        print(f"Loading Hard samples from: {hard_json_path}")

        # 预先加载并合并 samples
        combined_samples = []
        
        # Helper function to load samples from a specific JSON
        def load_vigor_samples(json_file):
            if not os.path.exists(json_file):
                print(f"Warning: File not found {json_file}")
                return []
            with open(json_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "samples" in data:
                return data["samples"]
            elif isinstance(data, list):
                return data
            else:
                return []

        easy_samples = load_vigor_samples(easy_json_path)
        hard_samples = load_vigor_samples(hard_json_path)
        print(f"Original Samples Loaded: Easy={len(easy_samples)}, Hard={len(hard_samples)}")
        print(f"Total Combined Raw Samples: {len(easy_samples) + len(hard_samples)}")
        
        # 合并 raw samples (注意：这里只是原始标注的列表，具体的展开在 Dataset 类中处理)
        all_raw_samples = easy_samples + hard_samples
        
        if args.vigor_max_samples is not None:
             all_raw_samples = all_raw_samples[:args.vigor_max_samples]

        # 使用多实例VIGORDataset（内部会自动将1个raw sample展开为3个training instances）
        train_dataset = VIGORDatasetMultiInstance(
            json_path=None, # 我们直接传入 samples，不需要 json_path
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            precision=args.precision,
            image_size=args.image_size,
            data_base_dir=args.vigor_data_base_dir,
            split=args.vigor_split,
            sam_mask_helper=sam_mask_helper,
            max_samples=args.vigor_max_samples,
            is_train=True,
            samples=all_raw_samples, # 传入合并后的 list
            debug_meta=getattr(args, "debug_epoch_shapes", False),
        )
        
        print(f"VIGOR训练集加载完成: {len(train_dataset)} 个样本 (Total Instances)")
        
    else:
        # 使用 RobotArm 数据集（原有逻辑）
        # 使用命令行参数传入的数据集路径
        raw_pic_base_dir = args.dataset_base_dir
        gt_mask_base_dir = args.gt_mask_base_dir
        sam_candidate_base_dir = args.sam_masks_base_dir
        
        # 准备多个 JSON 文件路径列表（三个视角）
        json_paths = []
        sam_mask_helpers = {}
        
        for view_name in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
            annotations_path = os.path.join(gt_mask_base_dir, view_name, 'annotations.json')
            if os.path.exists(annotations_path):
                json_paths.append(annotations_path)
                
                # 为每个视角创建SAM mask helper
                sam_mask_dir = os.path.join(sam_candidate_base_dir, view_name)
                if os.path.exists(sam_mask_dir):
                    sam_mask_helpers[view_name] = SAM_Mask_Reader_PNG(sam_mask_dir)
                else:
                    print(f"Warning: SAM candidate directory not found: {sam_mask_dir}")
        
        # 先构建"每视角前N张"的样本池，再按9:1随机划分（可复现）
        pooled_samples = _build_robotarm_samples(
            gt_mask_base_dir=gt_mask_base_dir,
            raw_pic_base_dir=raw_pic_base_dir,
            max_per_view=args.robotarm_max_per_view,
        )
        train_samples, _ = _split_samples_train_val(
            pooled_samples,
            val_ratio=args.robotarm_val_ratio,
            seed=args.robotarm_split_seed,
        )

        # 使用 RobotArmDataset（传入切分后的 samples，确保训练/验证不重叠）
        train_dataset = RobotArmDataset(
            json_paths=json_paths,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            precision=args.precision,
            image_size=args.image_size,
            raw_pic_base_dir=raw_pic_base_dir,
            gt_mask_base_dir=gt_mask_base_dir,
            sam_candidate_base_dir=sam_candidate_base_dir,
            sam_mask_helpers=sam_mask_helpers,
            max_samples_per_view=args.robotarm_max_per_view,
            is_train=True,
            samples=train_samples,
            debug_meta=getattr(args, "debug_epoch_shapes", False),
        )
            
        print(f"RobotArm训练集加载完成: {len(train_dataset)} 个样本")

    return train_dataset


def init_validation_dataset(args, tokenizer):
    if args.no_eval:
        return None

    # 根据数据集类型选择不同的数据集
    if args.dataset == "vigor":
        # 使用 VIGOR 数据集
        vigor_json_path = os.path.join(args.vigor_data_base_dir, args.vigor_val_split, args.vigor_json_file)
        
        # 创建 SAM mask helper（验证集）- 严格模式
        sam_mask_helper = None
        
        if args.vigor_val_sam_masks_dir is not None:
            if os.path.exists(args.vigor_val_sam_masks_dir):
                sam_mask_helper = SAM_Mask_Reader_PNG(args.vigor_val_sam_masks_dir)
                print(f"使用SAM候选masks: {args.vigor_val_sam_masks_dir}")
            else:
                raise FileNotFoundError(f"SAM masks目录不存在: {args.vigor_val_sam_masks_dir}")
        else:
            print("警告: 未指定SAM masks目录 (--vigor_val_sam_masks_dir)，将不使用候选masks")
        
        # 创建 VIGOR 验证数据集
        # ✅ 统一使用 VIGORDatasetMultiInstance（每个样本有3条指令）
        val_dataset = VIGORDatasetMultiInstance(
            json_path=vigor_json_path,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            precision=args.precision,
            image_size=args.image_size,
            data_base_dir=args.vigor_data_base_dir,
            split=args.vigor_val_split,
            sam_mask_helper=sam_mask_helper,
            max_samples=2,  # 限制验证集大小，加快验证速度（100个原始样本 = 300个实例）
            is_train=False,
            debug_meta=getattr(args, "debug_epoch_shapes", False),
        )
        
        print(f"VIGOR验证集加载完成: {len(val_dataset)} 个样本")
        
    else:
        # 使用 RobotArm 数据集（原有逻辑）
        # 使用命令行参数传入的数据集路径（与训练集相同）
        raw_pic_base_dir = args.dataset_base_dir
        gt_mask_base_dir = args.gt_mask_base_dir
        sam_candidate_base_dir = args.sam_masks_base_dir
        
        # 准备多个 JSON 文件路径列表（三个视角）
        json_paths = []
        sam_mask_helpers = {}
        
        for view_name in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
            annotations_path = os.path.join(gt_mask_base_dir, view_name, 'annotations.json')
            if os.path.exists(annotations_path):
                json_paths.append(annotations_path)
                
                # 为每个视角创建SAM mask helper
                sam_mask_dir = os.path.join(sam_candidate_base_dir, view_name)
                if os.path.exists(sam_mask_dir):
                    sam_mask_helpers[view_name] = SAM_Mask_Reader_PNG(sam_mask_dir)
                else:
                    print(f"Warning: SAM candidate directory not found: {sam_mask_dir}")
        
        # 与训练集使用同一套"样本池 + 随机切分"，保证严格不重叠
        pooled_samples = _build_robotarm_samples(
            gt_mask_base_dir=gt_mask_base_dir,
            raw_pic_base_dir=raw_pic_base_dir,
            max_per_view=args.robotarm_max_per_view,
        )
        _, val_samples = _split_samples_train_val(
            pooled_samples,
            val_ratio=args.robotarm_val_ratio,
            seed=args.robotarm_split_seed,
        )

        # 使用 RobotArmDataset（传入切分后的 samples）
        val_dataset = RobotArmDataset(
            json_paths=json_paths,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            precision=args.precision,
            image_size=args.image_size,
            raw_pic_base_dir=raw_pic_base_dir,
            gt_mask_base_dir=gt_mask_base_dir,
            sam_candidate_base_dir=sam_candidate_base_dir,
            sam_mask_helpers=sam_mask_helpers,
            max_samples_per_view=args.robotarm_max_per_view,
            is_train=False,
            samples=val_samples,
            debug_meta=getattr(args, "debug_epoch_shapes", False),
        )
        
        print(f"RobotArm验证集加载完成: {len(val_dataset)} 个样本")

    return val_dataset


def init_deepseed_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "ignore_unused_parameters": True,
        },
    }

    return ds_config



def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        # 初始化 SwanLab
        if SWANLAB_AVAILABLE and swanlab is not None:
            try:
                # 设置环境变量（参考 finetune_sam_lora_point.py）
                os.environ['SWANLAB_API_KEY'] = "BBd5HKuM6sIhTwyWmgZ6Z"
                # 直接初始化（不需要先 login）
                swanlab.init(
                    project="LLMSeg",
                    experiment_name=args.exp_name,
                    config={
                        "exp_name": args.exp_name,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "steps_per_epoch": args.steps_per_epoch,
                        "precision": args.precision,
                        "image_size": args.image_size,
                    }
                )
            except Exception as e:
                print(f"Warning: swanlab init failed: {e}, continuing without SwanLab")
                # 不使用 swanlab = None，避免局部变量问题
        elif not SWANLAB_AVAILABLE:
            print("Warning: swanlab not installed, skipping SwanLab logging")
    else:
        writer = None

    # set random seed

    random.seed(0+args.local_rank)
    np.random.seed(0+args.local_rank)

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    tokenizer = init_tokenizer(args)
    model = init_LISA_model(args, tokenizer)

    train_dataset = init_training_dataset(args, tokenizer)
    val_dataset = init_validation_dataset(args, tokenizer)
    # val_dataset = None
    if train_dataset is not None:
        print(f"Training with {len(train_dataset)} examples.")
    if val_dataset is not None:
        print(f"Validation with {len(val_dataset)} examples.")
    # init deepspeed distributed training
    ds_config = init_deepseed_config(args)
    
    # 方案B：在 DeepSpeed 初始化前扫描 meta tensor 参数，帮助定位 “Cannot copy out of meta tensor; no data!”
    try:
        meta_param_names = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
        if len(meta_param_names) > 0:
            print(f"[MetaTensorCheck] Found {len(meta_param_names)} meta parameters before deepspeed.initialize(). "
                  f"Showing first 20:")
            for n in meta_param_names[:20]:
                print(f"  - {n}")
        else:
            print("[MetaTensorCheck] No meta parameters found before deepspeed.initialize().")
    except Exception as e:
        print(f"[MetaTensorCheck] Failed to scan meta parameters: {e}")

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,batch_size=args.batch_size, sampler=sampler, 
        num_workers=args.workers, collate_fn=partial(
            collate_fn_new,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ))


    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_new,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    # resume checkpoint
    # 支持两种格式:
    # 1. DeepSpeed checkpoint (目录)
    # 2. SafeTensors checkpoint (.safetensors文件)
    if args.auto_resume and len(args.resume) == 0:
        # 自动搜索checkpoint
        # 优先级: safetensors > deepspeed
        
        # 搜索safetensors文件 (在子目录中)
        import glob
        safetensors_dir = os.path.join(args.log_dir, "safetensors_checkpoints")
        safetensors_files = glob.glob(os.path.join(safetensors_dir, "model_epoch*.safetensors"))
        if safetensors_files:
            # 找到最新的safetensors文件 (按文件名排序)
            latest_safetensors = sorted(safetensors_files)[-1]
            args.resume = latest_safetensors
            print(f"[Auto Resume] 找到SafeTensors checkpoint: {latest_safetensors}")
        else:
            # 搜索DeepSpeed checkpoint
            resume = os.path.join(args.log_dir, "ckpt_model")
            if os.path.exists(resume):
                args.resume = resume
                print(f"[Auto Resume] 找到DeepSpeed checkpoint: {resume}")
    
    if args.resume:
        # 📊 调试信息: 打印checkpoint文件大小和内存使用
        if args.local_rank == 0:
            print(f"\n{'='*80}")
            print(f"[Checkpoint Debug] 准备加载checkpoint: {args.resume}")
            print(f"{'='*80}")
            
            # 打印checkpoint文件大小
            import glob
            ckpt_files = glob.glob(os.path.join(args.resume, "**/", "*.pt"), recursive=True)
            total_size_bytes = 0
            for f in ckpt_files:
                size = os.path.getsize(f)
                total_size_bytes += size
                size_gb = size / (1024**3)
                print(f"  {os.path.basename(f)}: {size_gb:.2f} GB")
            
            total_size_gb = total_size_bytes / (1024**3)
            print(f"  总大小: {total_size_gb:.2f} GB")
            
            # 打印当前cgroup内存使用
            try:
                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                    mem_usage = int(f.read().strip())
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    mem_limit = int(f.read().strip())
                
                mem_usage_gb = mem_usage / (1024**3)
                mem_limit_gb = mem_limit / (1024**3)
                mem_percent = (mem_usage / mem_limit) * 100
                
                print(f"\n  当前cgroup内存: {mem_usage_gb:.2f} GB / {mem_limit_gb:.2f} GB ({mem_percent:.1f}%)")
                print(f"  剩余内存: {mem_limit_gb - mem_usage_gb:.2f} GB")
            except:
                pass
            
            print(f"\n  开始加载...")
            print(f"{'='*80}\n")
        
        # 加载checkpoint
        # 支持两种格式: DeepSpeed (.pt) 和 SafeTensors (.safetensors)
        
        if args.resume.endswith('.safetensors'):
            # 💾 加载SafeTensors格式
            if args.local_rank == 0:
                print(f"\n{'='*80}")
                print(f"💾 加载SafeTensors checkpoint: {args.resume}")
                print(f"{'='*80}\n")
            
            try:
                from safetensors.torch import load_file
                
                # 加载safetensors文件
                if args.local_rank == 0:
                    print("步骤1: 加载safetensors文件...")
                
                state_dict = load_file(args.resume)
                
                if args.local_rank == 0:
                    print(f"  加载了 {len(state_dict)} 个参数")
                    print("\n步骤2: 分离模型和Optimizer状态...")
                
                # 分离模型参数和optimizer states
                model_state_dict = {}
                optimizer_states = {}
                
                for key, value in state_dict.items():
                    if key.startswith('optimizer.state.'):
                        # Optimizer state
                        optimizer_states[key] = value
                    else:
                        # 模型参数
                        model_state_dict[key] = value
                
                if args.local_rank == 0:
                    print(f"  模型参数: {len(model_state_dict)}")
                    print(f"  Optimizer states: {len(optimizer_states)}")
                
                # 步骤3: 加载模型参数
                if args.local_rank == 0:
                    print("\n步骤3: 加载模型参数...")
                
                missing_keys, unexpected_keys = model_engine.module.load_state_dict(
                    model_state_dict, 
                    strict=False
                )
                
                if args.local_rank == 0:
                    if missing_keys:
                        print(f"  ⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
                    if unexpected_keys:
                        print(f"  ⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
                
                # 步骤4: 从 DeepSpeed checkpoint 加载 Optimizer states (混合策略)
                # 先加载 DeepSpeed（获取 Optimizer），再用 SafeTensors 覆盖模型参数
                if args.local_rank == 0:
                    print("\n步骤4: 尝试从 DeepSpeed checkpoint 加载 Optimizer states...")
                
                optimizer_loaded = False
                try:
                    # 查找对应的 DeepSpeed checkpoint 目录
                    # 根据 SafeTensors 路径判断是 best 还是 newest
                    safetensors_dir = os.path.dirname(args.resume)
                    safetensors_type = os.path.basename(safetensors_dir)  # "best" 或 "newest"
                    safetensors_base = os.path.dirname(safetensors_dir)  # safetensors_checkpoints
                    log_dir = os.path.dirname(safetensors_base)  # log_dir
                    deepspeed_ckpt_dir = os.path.join(log_dir, "ckpt_model")
                    
                    # 根据 SafeTensors 类型选择对应的 DeepSpeed 目录
                    target_ckpt_dir = None
                    if safetensors_type in ["best", "newest"]:
                        # 新格式：safetensors_checkpoints/best/ 或 safetensors_checkpoints/newest/
                        matching_ds_dir = os.path.join(deepspeed_ckpt_dir, safetensors_type)
                        if os.path.exists(matching_ds_dir):
                            target_ckpt_dir = matching_ds_dir
                            if args.local_rank == 0:
                                print(f"  找到对应的 DeepSpeed checkpoint ({safetensors_type}): {matching_ds_dir}")
                        else:
                            # 如果对应目录不存在，尝试其他目录
                            other_type = "newest" if safetensors_type == "best" else "best"
                            other_dir = os.path.join(deepspeed_ckpt_dir, other_type)
                            if os.path.exists(other_dir):
                                target_ckpt_dir = other_dir
                                if args.local_rank == 0:
                                    print(f"  ⚠️  未找到 {safetensors_type}，使用 {other_type}: {other_dir}")
                    else:
                        # 兼容旧格式：直接在 safetensors_checkpoints/ 下
                        log_dir = os.path.dirname(safetensors_dir)  # log_dir
                        deepspeed_ckpt_dir = os.path.join(log_dir, "ckpt_model")
                        
                        newest_dir = os.path.join(deepspeed_ckpt_dir, "newest")
                        best_dir = os.path.join(deepspeed_ckpt_dir, "best")
                        
                        if os.path.exists(newest_dir):
                            target_ckpt_dir = newest_dir
                            if args.local_rank == 0:
                                print(f"  找到 DeepSpeed checkpoint (newest): {newest_dir}")
                        elif os.path.exists(best_dir):
                            target_ckpt_dir = best_dir
                            if args.local_rank == 0:
                                print(f"  找到 DeepSpeed checkpoint (best): {best_dir}")
                        elif os.path.exists(deepspeed_ckpt_dir):
                            # 兼容更旧的格式：直接查找 global_step 目录
                            import glob
                            step_dirs = glob.glob(os.path.join(deepspeed_ckpt_dir, "global_step*"))
                            if step_dirs:
                                target_ckpt_dir = deepspeed_ckpt_dir
                                if args.local_rank == 0:
                                    print(f"  找到 DeepSpeed checkpoint (旧格式): {target_ckpt_dir}")
                    
                    if target_ckpt_dir:
                        # DeepSpeed 会同时加载模型和 Optimizer
                        _, client_state = model_engine.load_checkpoint(
                            target_ckpt_dir,
                            load_module_only=False,
                            load_optimizer_states=True,
                            load_lr_scheduler_states=True,
                            load_module_strict=False
                        )
                        optimizer_loaded = True
                        
                        if args.local_rank == 0:
                            print(f"  ✅ Optimizer states 已从 DeepSpeed checkpoint 恢复")
                    else:
                        if args.local_rank == 0:
                            print(f"  ⚠️  未找到 DeepSpeed checkpoint 目录")
                
                except Exception as e:
                    if args.local_rank == 0:
                        print(f"  ⚠️  DeepSpeed 加载失败: {e}")
                        print(f"  Optimizer 将重新初始化")
                
                # 步骤5: 用 SafeTensors 覆盖模型参数（确保模型是最新的）
                if args.local_rank == 0:
                    print("\n步骤5: 用 SafeTensors 覆盖模型参数...")
                
                missing_keys, unexpected_keys = model_engine.module.load_state_dict(
                    model_state_dict, 
                    strict=False
                )
                
                if args.local_rank == 0:
                    print(f"  ✅ 模型参数已从 SafeTensors 恢复")
                
                # 加载元数据 (epoch, scheduler 等)
                metadata_path = args.resume.replace('.safetensors', '_metadata.pt')
                if os.path.exists(metadata_path):
                    metadata = torch.load(metadata_path, map_location='cpu')
                    
                    # 恢复 epoch 信息
                    if 'epoch' in metadata:
                        args.start_epoch = metadata['epoch'] + 1
                        if args.local_rank == 0:
                            print(f"  ✅ 将从 epoch {args.start_epoch} 继续训练")
                
                if args.local_rank == 0:
                    print(f"\n✅ 混合加载完成!")
                    print(f"  ✅ 模型参数: SafeTensors")
                    if optimizer_loaded:
                        print(f"  ✅ Optimizer: DeepSpeed checkpoint")
                        # 验证 optimizer 状态是否真的加载了
                        try:
                            opt_state = model_engine.optimizer.state_dict()
                            opt_step = opt_state['base_optimizer_state']['state'][0]['step']
                            print(f"  📊 Optimizer step: {opt_step} (>0 说明状态已恢复)")
                        except Exception as e:
                            print(f"  ⚠️  无法读取 optimizer step: {e}")
                    else:
                        print(f"  ⚠️  Optimizer: 重新初始化")
                    print(f"{'='*80}\n")
                
                # 清理内存
                del state_dict, model_state_dict, optimizer_states
                torch.cuda.empty_cache()
                
            except Exception as e:
                if args.local_rank == 0:
                    print(f"❌ SafeTensors加载失败: {e}")
                    import traceback
                    traceback.print_exc()
                raise
        
        else:
            # 💾 加载DeepSpeed格式
            if args.local_rank == 0:
                print(f"\n{'='*80}")
                print(f"💾 加载DeepSpeed checkpoint: {args.resume}")
                print(f"{'='*80}\n")
            
            load_path, client_state = model_engine.load_checkpoint(
                args.resume, 
                load_optimizer_states=True,  # DeepSpeed可以加载optimizer
                load_lr_scheduler_states=True
            )
        
        # 加载后打印内存
        if args.local_rank == 0:
            print(f"\n{'='*80}")
            print(f"[Checkpoint Debug] 加载完成")
            print(f"{'='*80}")
            
            try:
                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                    mem_usage = int(f.read().strip())
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    mem_limit = int(f.read().strip())
                
                mem_usage_gb = mem_usage / (1024**3)
                mem_limit_gb = mem_limit / (1024**3)
                mem_percent = (mem_usage / mem_limit) * 100
                
                print(f"  加载后cgroup内存: {mem_usage_gb:.2f} GB / {mem_limit_gb:.2f} GB ({mem_percent:.1f}%)")
                print(f"  剩余内存: {mem_limit_gb - mem_usage_gb:.2f} GB")
            except:
                pass
            
            print(f"{'='*80}\n")
        # with open(os.path.join(args.resume, "latest"), "r") as f:
        #     ckpt_dir = f.readlines()[0].strip()
        # args.start_epoch = (
        #     int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        # )
        # print(
        #     "resume training from {}, start from epoch {}".format(
        #         args.resume, args.start_epoch
        #     )
        # )

    train_iter = iter(train_loader)

    # 获取 swanlab logger（仅在 local_rank == 0 时不为 None）
    swanlab_logger = swanlab if (args.local_rank == 0 and SWANLAB_AVAILABLE and swanlab is not None) else None

    if args.eval_only:
        # giou, ciou = validate_threshold_from_topIoU(val_loader, model_engine, 0, writer, args, threshold=0.5, swanlab_logger=swanlab_logger)
        # giou, ciou = validate_iou_iop(val_loader, model_engine, 0, writer, args, threshold=0.5, swanlab_logger=swanlab_logger)
        # giou, ciou = validate(val_loader, model_engine, 0, writer, args, swanlab_logger)
        giou, ciou = validate_threshold(val_loader, model_engine, 0, writer, args, threshold=0.5, swanlab_logger=swanlab_logger)
        # for i in range(10):
        #     threshold = 0.1 * (i + 1)
        #     giou, ciou = validate_threshold(val_loader, model_engine, 0, writer, args, threshold=threshold)
        #     print("results from threshold {}: giou={}, ciou={}".format(threshold, giou, ciou))
        exit()

    best_score, cur_ciou = 0.0, 0.0

    # 获取 swanlab logger（仅在 local_rank == 0 时不为 None）
    swanlab_logger = swanlab if (args.local_rank == 0 and SWANLAB_AVAILABLE and swanlab is not None) else None

    try:
        for epoch in range(args.start_epoch, args.epochs):
            train_iter = train(
                train_loader,
                model_engine,
                epoch,
                scheduler,
                writer,
                train_iter,
                args,
                swanlab_logger,
            )

            if args.no_eval == False:
                giou, ciou = validate(val_loader, model_engine, epoch, writer, args, swanlab_logger)
                if not args.iou_selection_only:
                    giou, ciou = validate_threshold(val_loader, model_engine, epoch, writer, args, threshold=0.5, swanlab_logger=swanlab_logger)
                print("results from threshold: giou={}, ciou={}".format(giou, ciou))

                is_best = giou > best_score
                best_score = max(giou, best_score)
                cur_ciou = ciou if is_best else cur_ciou
            else:
                # ✅ 修复：--no_eval 时设置默认值
                is_best = False
                giou = 0.0
                ciou = 0.0

            
            # save checkpoint
            # ✅ 修改：根据 save_every 参数控制保存频率
            # 保存条件：每 save_every 个 epoch 保存一次，或者是最佳模型时保存
            should_save = ((epoch + 1) % args.save_every == 0) or is_best
            
            if should_save: 
                ckpt_base_dir = os.path.join(args.log_dir, "ckpt_model")
                newest_dir = os.path.join(ckpt_base_dir, "newest")
                best_dir = os.path.join(ckpt_base_dir, "best")
                
                if args.local_rank == 0:
                    # 每一轮开始保存前，打印当前状态
                    print(f"\n[信息] Epoch {epoch} 结束，准备保存权重...")
                    if is_best:
                        print(f"🌟 检测到性能提升! 当前 gIoU: {giou:.4f} (历史最佳: {best_score:.4f})")
                    
                    # ✅ 保存前先检查并释放内存
                    print("\n" + "="*60)
                    print("[内存] 保存前内存清理")
                    print("="*60)
                    import gc
                    
                    # 1. 清理前状态
                    try:
                        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                            mem_before = int(f.read().strip()) / (1024**3)
                        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                            mem_limit = int(f.read().strip()) / (1024**3)
                        print(f"[内存] 清理前: {mem_before:.2f} GB / {mem_limit:.2f} GB ({mem_before/mem_limit*100:.1f}%)")
                    except:
                        mem_before = 0
                        mem_limit = 62
                    
                    # 2. 执行清理
                    print("[内存] 执行 gc.collect() + torch.cuda.empty_cache()...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # ✅ 尝试释放之前加载模型权重时产生的文件缓存
                    print("[缓存] 释放模型权重文件缓存...")
                    model_paths = [
                        getattr(args, 'version', ''),  # LLaMA 模型路径
                        getattr(args, 'vision_pretrained', ''),  # SAM 权重路径
                        '/opt/data/private/model/dinov2_vitl14',  # DINOv2 路径
                    ]
                    for model_path in model_paths:
                        if model_path and os.path.exists(model_path):
                            if os.path.isdir(model_path):
                                dropped_n, dropped_gb = drop_directory_cache(model_path)
                                if dropped_n > 0:
                                    print(f"[缓存] 释放 {os.path.basename(model_path)}: {dropped_n} 文件, {dropped_gb:.2f} GB")
                            elif os.path.isfile(model_path):
                                if drop_file_cache(model_path):
                                    fsize = os.path.getsize(model_path) / (1024**3)
                                    print(f"[缓存] 释放 {os.path.basename(model_path)}: {fsize:.2f} GB")
                    
                    # 3. 清理后状态
                    try:
                        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                            mem_after = int(f.read().strip()) / (1024**3)
                        mem_freed = mem_before - mem_after
                        mem_remaining = mem_limit - mem_after
                        print(f"[内存] 清理后: {mem_after:.2f} GB / {mem_limit:.2f} GB ({mem_after/mem_limit*100:.1f}%)")
                        print(f"[内存] 释放了: {mem_freed:.2f} GB")
                        print(f"[内存] 剩余可用: {mem_remaining:.2f} GB")
                        
                        # 警告：如果剩余内存太少
                        if mem_remaining < 20:
                            print(f"[内存] ⚠️  警告: 剩余内存不足 20 GB，保存可能失败！")
                    except:
                        import psutil
                        mem = psutil.virtual_memory()
                        print(f"[内存] 清理后: {mem.used/(1024**3):.2f} GB / {mem.total/(1024**3):.2f} GB ({mem.percent}%)")
                    
                    print("="*60 + "\n")
                    
                    torch.save(
                        {"epoch": epoch, "best_score": best_score, "giou": giou},
                        os.path.join(
                            args.log_dir,
                            "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                                best_score, cur_ciou
                            ),
                        ),
                    )
                    
                    # 1. 更新 newest 目录（每次都更新）
                    import shutil
                    if os.path.exists(newest_dir):
                        try:
                            shutil.rmtree(newest_dir, ignore_errors=True)
                        except Exception as e:
                            print(f"  [警告] 删除旧 newest 目录失败: {e}")
                
                # ✅ 关键：等待所有进程同步
                torch.distributed.barrier()
                
                # 再次释放内存（所有进程）
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # 保存到 newest 目录
                model_engine.save_checkpoint(newest_dir)
                
                if args.local_rank == 0:
                    print(f"📁 newest checkpoint saved in {newest_dir}")
                    
                    # 2. 如果是最佳性能，更新 best 目录
                    if is_best:
                        if os.path.exists(best_dir):
                            try:
                                shutil.rmtree(best_dir, ignore_errors=True)
                            except Exception as e:
                                print(f"  [警告] 删除旧 best 目录失败: {e}")
                        # 复制 newest 到 best
                        shutil.copytree(newest_dir, best_dir)
                        print(f"📁 best checkpoint saved in {best_dir}")
                    
                    # ✅ 关键：DeepSpeed 保存完成后，强制释放内存和文件缓存
                    print("\n[内存] DeepSpeed 保存完成，释放内存...")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # ✅ 使用 posix_fadvise 释放 DeepSpeed checkpoint 的文件缓存
                    print("[缓存] 释放 DeepSpeed checkpoint 文件缓存...")
                    dropped_n, dropped_gb_n = drop_directory_cache(newest_dir)
                    print(f"[缓存] newest: 释放了 {dropped_n} 个文件, 约 {dropped_gb_n:.2f} GB")
                    if is_best:
                        dropped_b, dropped_gb_b = drop_directory_cache(best_dir)
                        print(f"[缓存] best: 释放了 {dropped_b} 个文件, 约 {dropped_gb_b:.2f} GB")
                    
                    # 打印当前内存状态
                    try:
                        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                            mem_used = int(f.read().strip()) / (1024**3)
                        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                            mem_limit = int(f.read().strip()) / (1024**3)
                        mem_remaining = mem_limit - mem_used
                        print(f"[内存] DeepSpeed 保存后: {mem_used:.2f} GB / {mem_limit:.2f} GB ({mem_used/mem_limit*100:.1f}%)")
                        print(f"[内存] 剩余可用: {mem_remaining:.2f} GB")
                    except:
                        import psutil
                        mem = psutil.virtual_memory()
                        mem_remaining = mem.available / (1024**3)
                        print(f"[内存] DeepSpeed 保存后: {mem.used/(1024**3):.2f} GB / {mem.total/(1024**3):.2f} GB ({mem.percent}%)")
                        print(f"[内存] 剩余可用: {mem_remaining:.2f} GB")
                    
                    # 💾 额外保存safetensors格式 (如果启用)
                    if args.save_safetensors and not args.save_deepspeed_only:
                        try:
                            print("\n" + "="*80)
                            print("💾 保存SafeTensors格式...")
                            print("="*80)
                            
                            from safetensors.torch import save_file
                            import pickle
                            
                            # 步骤1: 提取模型状态 (只保存模型参数，optimizer 由 DeepSpeed 保存)
                            print("步骤1: 提取模型状态...")
                            
                            # 打印当前内存状态
                            try:
                                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                                    mem_used = int(f.read().strip()) / (1024**3)
                                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                                    mem_limit = int(f.read().strip()) / (1024**3)
                                print(f"[内存] 提取模型前: {mem_used:.2f} GB / {mem_limit:.2f} GB ({mem_used/mem_limit*100:.1f}%)")
                            except:
                                pass
                            
                            # 获取完整的state_dict
                            model_state_dict = {}
                            for name, param in model_engine.module.named_parameters():
                                # 保存到CPU避免GPU内存问题
                                model_state_dict[name] = param.detach().cpu()
                            
                            # 也保存buffers (如running_mean, running_var等)
                            for name, buffer in model_engine.module.named_buffers():
                                model_state_dict[name] = buffer.detach().cpu()
                            
                            print(f"  提取了 {len(model_state_dict)} 个模型参数/buffer")
                            
                            # 打印提取后内存状态
                            try:
                                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                                    mem_used = int(f.read().strip()) / (1024**3)
                                print(f"[内存] 提取模型后: {mem_used:.2f} GB / {mem_limit:.2f} GB ({mem_used/mem_limit*100:.1f}%)")
                            except:
                                pass
                            
                            # 步骤2: 保存模型参数到 newest/ 和 best/ 目录 (与 DeepSpeed 对应)
                            safetensors_base_dir = os.path.join(args.log_dir, "safetensors_checkpoints")
                            newest_st_dir = os.path.join(safetensors_base_dir, "newest")
                            best_st_dir = os.path.join(safetensors_base_dir, "best")
                            
                            # 2.1 保存到 newest/ 目录 (每次都更新)
                            os.makedirs(newest_st_dir, exist_ok=True)
                            newest_st_path = os.path.join(newest_st_dir, f"model_epoch{epoch}_giou{giou:.3f}.safetensors")
                            newest_meta_path = newest_st_path.replace('.safetensors', '_metadata.pt')
                            
                            # 清理 newest 目录中的旧文件
                            import glob as glob_module
                            for old_file in glob_module.glob(os.path.join(newest_st_dir, "model_epoch*.safetensors")):
                                if old_file != newest_st_path:
                                    os.remove(old_file)
                                    meta_file = old_file.replace('.safetensors', '_metadata.pt')
                                    if os.path.exists(meta_file):
                                        os.remove(meta_file)
                            
                            print(f"步骤2a: 保存 newest 到 {newest_st_path}...")
                            save_file(model_state_dict, newest_st_path)
                            
                            # 保存 newest 元数据
                            metadata = {
                                'epoch': epoch,
                                'best_score': best_score,
                                'cur_ciou': cur_ciou,
                                'lr_scheduler_state': scheduler.state_dict() if scheduler else None,
                                'type': 'newest',  # 标记类型
                            }
                            torch.save(metadata, newest_meta_path)
                            
                            # 2.2 如果是最佳性能，也保存到 best/ 目录
                            if is_best:
                                os.makedirs(best_st_dir, exist_ok=True)
                                best_st_path = os.path.join(best_st_dir, f"model_epoch{epoch}_giou{giou:.3f}.safetensors")
                                best_meta_path = best_st_path.replace('.safetensors', '_metadata.pt')
                                
                                # 清理 best 目录中的旧文件
                                for old_file in glob_module.glob(os.path.join(best_st_dir, "model_epoch*.safetensors")):
                                    if old_file != best_st_path:
                                        os.remove(old_file)
                                        meta_file = old_file.replace('.safetensors', '_metadata.pt')
                                        if os.path.exists(meta_file):
                                            os.remove(meta_file)
                                
                                print(f"步骤2b: 保存 best 到 {best_st_path}...")
                                save_file(model_state_dict, best_st_path)
                                
                                # 保存 best 元数据
                                best_metadata = {
                                    'epoch': epoch,
                                    'best_score': best_score,
                                    'cur_ciou': cur_ciou,
                                    'lr_scheduler_state': scheduler.state_dict() if scheduler else None,
                                    'type': 'best',  # 标记类型
                                }
                                torch.save(best_metadata, best_meta_path)
                            
                            # 获取文件大小
                            file_size_gb = os.path.getsize(newest_st_path) / (1024**3)
                            meta_size_mb = os.path.getsize(newest_meta_path) / (1024**2)
                            
                            print(f"✅ SafeTensors保存成功!")
                            print(f"  📁 newest: {newest_st_path} ({file_size_gb:.2f} GB)")
                            if is_best:
                                print(f"  📁 best: {best_st_path} ({file_size_gb:.2f} GB)")
                            print(f"  参数数量: {len(model_state_dict)}")
                            print(f"  💡 Optimizer 状态由 DeepSpeed checkpoint 保存")
                            print("="*80 + "\n")
                            
                            # 清理内存
                            del model_state_dict
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # ✅ 使用 posix_fadvise 释放 SafeTensors 文件缓存
                            print("[缓存] 释放 SafeTensors 文件缓存...")
                            if drop_file_cache(newest_st_path):
                                print(f"[缓存] newest: 释放了 {file_size_gb:.2f} GB")
                            if is_best and 'best_st_path' in locals():
                                if drop_file_cache(best_st_path):
                                    print(f"[缓存] best: 释放了 {file_size_gb:.2f} GB")
                            
                            # 打印清理后内存状态
                            try:
                                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                                    mem_used = int(f.read().strip()) / (1024**3)
                                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                                    mem_limit = int(f.read().strip()) / (1024**3)
                                mem_remaining = mem_limit - mem_used
                                print(f"[内存] SafeTensors 保存后 (已清理): {mem_used:.2f} GB / {mem_limit:.2f} GB ({mem_used/mem_limit*100:.1f}%)")
                                print(f"[内存] 剩余可用: {mem_remaining:.2f} GB")
                            except:
                                pass
                            
                        except Exception as e:
                            print(f"⚠️  SafeTensors保存失败: {e}")
                            print(f"  DeepSpeed checkpoint仍然可用")
                            import traceback
                            traceback.print_exc()
    except KeyboardInterrupt:
        if args.local_rank == 0:
            print("\n[信息] 收到中断信号（Ctrl+C），正在优雅退出...")
            print("[信息] 训练已停止，可以安全退出")
        # 不打印堆栈信息，直接退出
        import sys
        sys.exit(0)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    swanlab_logger=None,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    align_losses = AverageMeter("AlignLoss", ":.4f")
    regression_losses = AverageMeter("RegressionLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            align_losses,
            regression_losses
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    # switch to train mode
    model.train()
    end = time.time()
    
    # ✅ 计算全局步数起始值（用于SwanLab记录）
    global_step_start = epoch * args.steps_per_epoch
    
    # ✅ 训练可视化：在训练数据流中“随机抽样”保存（只在 rank0）
    # 规则：已知本 epoch 会处理的样本数 N = steps_per_epoch * grad_accumulation_steps * batch_size，
    # 使用流式无放回抽样（每个样本以 remaining_needed/remaining_items 的概率入选），保证每个 epoch 恰好保存 max_vis_samples 张。
    max_vis_samples = 4
    vis_rng = random.Random(12345 + epoch)  # 可复现的随机（不同 epoch 不同）
    total_stream_samples = None
    seen_stream_samples = 0
    
    vis_count = 0
    sample_idx = 0  # 当前样本索引（在当前GPU中的局部索引）
    
    # 训练主循环
    for global_step in range(args.steps_per_epoch):
        # 梯度累积循环
        for i in range(args.grad_accumulation_steps):
            # 获取训练数据，如果迭代器耗尽则重新创建
            try:
                input_dict = next(train_iter)
            except Exception:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            
            # 使用正确的 device
            device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
            input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)

            # 训练可视化抽样：先决定本 batch 哪些样本需要可视化（流式无放回抽样，保证最终数量= max_vis_samples）
            batch_size = input_dict["images"].size(0)
            if total_stream_samples is None:
                total_stream_samples = args.steps_per_epoch * args.grad_accumulation_steps * batch_size

            selected_in_batch = []
            if getattr(args, "visualize", False) and args.local_rank == 0 and vis_count < max_vis_samples:
                for batch_sample_idx in range(batch_size):
                    remaining_items = max(total_stream_samples - seen_stream_samples, 1)
                    remaining_needed = max_vis_samples - vis_count
                    if remaining_items <= remaining_needed:
                        take = True
                    else:
                        take_prob = remaining_needed / float(remaining_items)
                        take = vis_rng.random() < take_prob
                    if take:
                        selected_in_batch.append(batch_sample_idx)
                        vis_count += 1  # 预占名额，保证全局计数正确
                    seen_stream_samples += 1
            else:
                # 即使本 batch 没选中，也要推进流式计数（上面只在 rank0+visualize 才推进）
                if getattr(args, "visualize", False) and args.local_rank == 0 and total_stream_samples is not None:
                    seen_stream_samples += batch_size

            # ✅ 参考 finetune_llmseg_copy.py：如果本 batch 需要可视化，让训练 forward 直接返回 pred_similarity/gt_masks/pred_iou
            # 这样就不需要额外再跑一次 inference=True 的 forward（更快更稳）
            if selected_in_batch:
                input_dict["return_vis"] = True

            # 抽样信息
            if (
                args.local_rank == 0
                and epoch == 0
                and global_step == 0
                and i == 0
                and not hasattr(train, "_train_vis_debug_printed")
            ):
                print(
                    f"[TrainVisDebug] selected_in_batch={selected_in_batch} vis_count={vis_count} "
                    f"seen_stream_samples={seen_stream_samples} total_stream_samples={total_stream_samples}",
                    flush=True,
                )
                train._train_vis_debug_printed = True

            # ✅ 每个 epoch 打印一次“原图/GT/候选mask”处理维度追踪（来自 Dataset -> collate 透传）
            if (
                getattr(args, "debug_epoch_shapes", False)
                and args.local_rank == 0
                and global_step == 0
                and i == 0
                and not hasattr(train, "_train_epoch_debug_printed")
            ):
                train._train_epoch_debug_printed = set()
            if (
                getattr(args, "debug_epoch_shapes", False)
                and epoch not in getattr(train, "_train_epoch_debug_printed", set())
            ):
                # 同步让模型 forward 打印“从进入模型到 loss”的每一步关键维度（训练专用，不影响验证）
                input_dict["debug_train_shapes"] = True
                input_dict["debug_epoch"] = int(epoch)
                
                # 🔍 打印指令和回答 (优先打印Hard样本)
                if 'conversation_list' in input_dict and 'object_name_list' in input_dict:
                    print("\n" + "=" * 90, flush=True)
                    print(f"[训练指令示例] Epoch {epoch}, Step {global_step}", flush=True)
                    print("=" * 90, flush=True)
                    
                    # 识别Hard样本 (包含遮挡关键词)
                    hard_keywords = ['under', 'beneath', 'blocking', 'move', 'first', 'on top']
                    hard_indices = []
                    easy_indices = []
                    
                    for idx, conv in enumerate(input_dict['conversation_list']):
                        if any(kw in conv.lower() for kw in hard_keywords):
                            hard_indices.append(idx)
                        else:
                            easy_indices.append(idx)
                    
                    # 优先选择Hard样本,不足则补充Easy样本
                    selected = (hard_indices[:3] + easy_indices)[:3]
                    
                    for i, idx in enumerate(selected):
                        obj_name = input_dict['object_name_list'][idx] if idx < len(input_dict['object_name_list']) else 'N/A'
                        conv = input_dict['conversation_list'][idx]
                        sample_type = "Hard" if idx in hard_indices else "Easy"
                        
                        print(f"\n样本 {i + 1} [{sample_type}]:", flush=True)
                        print(f"  物体: {obj_name}", flush=True)
                        print(f"  完整对话:", flush=True)
                        print(f"  {conv}", flush=True)
                        print("-" * 90, flush=True)
                
                meta_list = input_dict.get("debug_meta_list", None)
                print("\n" + "=" * 90, flush=True)
                print(f"[EpochShapeDebug] epoch={epoch} (showing batch0 random 3 samples debug_meta)", flush=True)
                if not isinstance(meta_list, list):
                    print(
                        f"[EpochShapeDebug] WARNING: debug_meta_list not found or not a list. "
                        f"Got type={type(meta_list)}. input_dict keys={list(input_dict.keys())}",
                        flush=True,
                    )
                elif len(meta_list) == 0:
                    print("[EpochShapeDebug] WARNING: debug_meta_list is empty.", flush=True)
                else:
                    # 随机抽样：在 batch0 中随机选 3 个样本（不足 3 个则全选），每个 epoch 固定（可复现）
                    try:
                        import random as _py_random
                        bs0 = len(meta_list)
                        k = 3 if bs0 >= 3 else bs0
                        rng = _py_random.Random(int(epoch) + 12345)
                        chosen = sorted(rng.sample(list(range(bs0)), k=k)) if bs0 > 0 else []
                    except Exception:
                        chosen = [0] if len(meta_list) > 0 else []

                    # 统一打印格式：[debug] + 中文步骤 + 维度/面积占比
                    def _fmt_ratio_stats(x):
                        if x is None or (not isinstance(x, list)) or len(x) == 0:
                            return "N/A"
                        mn = min(x)
                        mx = max(x)
                        mean = sum(x) / len(x)
                        return f"min={mn:.4f} max={mx:.4f} mean={mean:.4f} (K={len(x)})"

                    print(f"[debug] 本epoch随机选中的batch内样本索引: {chosen}", flush=True)

                    for bidx in chosen:
                        meta0 = meta_list[bidx]
                        if meta0 is None:
                            print(
                                f"[debug] WARNING: debug_meta_list[{bidx}] is None. "
                                "Make sure you passed --debug_epoch_shapes and RobotArmDataset(debug_meta=True) is used.",
                                flush=True,
                            )
                            continue

                        idx0 = meta0.get("idx", "N/A")
                        if isinstance(meta0, dict) and meta0.get("error", None):
                            print(f"[debug] 数据集debug_meta构建失败 | batch_idx={bidx} idx={idx0} | error={meta0.get('error')}", flush=True)

                        print(f"[debug] 数据集-样本索引 | batch_idx={bidx} | idx={idx0}", flush=True)
                        print(f"[debug] 数据集-原图路径 | batch_idx={bidx} | image_path={meta0.get('image_path', 'N/A')}", flush=True)

                        print(
                            f"[debug] 第1步-读取原图 | batch_idx={bidx} | raw_image_shape={meta0.get('raw_image_shape', 'N/A')} | ori_size_hw={meta0.get('ori_size_hw', 'N/A')}",
                            flush=True,
                        )
                        print(
                            f"[debug] 第2步-ResizeLongestSide(原图) | batch_idx={bidx} | image_after_resize_longest_hw={meta0.get('image_after_resize_longest_hw', 'N/A')}",
                            flush=True,
                        )
                        print(
                            f"[debug] 第3步-预处理+Pad到正方形(原图) | batch_idx={bidx} | image_after_preprocess_chw={meta0.get('image_after_preprocess_chw', 'N/A')}",
                            flush=True,
                        )

                        gt_area_ratio = meta0.get("gt_mask_area_ratio", None)
                        gt_area_ratio_s = f"{gt_area_ratio:.4f}" if isinstance(gt_area_ratio, (float, int)) else "N/A"
                        print(
                            f"[debug] 第4步-GT掩码(原始尺寸) | batch_idx={bidx} | gt_mask_shape_hw={meta0.get('gt_mask_shape_hw', 'N/A')} | 前景占比={gt_area_ratio_s}",
                            flush=True,
                        )

                        print(
                            f"[debug] 第5步-候选掩码(原始尺寸) | batch_idx={bidx} | segs_origin_shape_hwk={meta0.get('segs_origin_shape_hwk', 'N/A')} | 前景占比={_fmt_ratio_stats(meta0.get('segs_origin_area_ratio', None))}",
                            flush=True,
                        )
                        print(
                            f"[debug] 第6步-ResizeLongestSide(候选掩码) | batch_idx={bidx} | segs_resized_shape_hwk={meta0.get('segs_resized_shape_hwk', 'N/A')} | 前景占比={_fmt_ratio_stats(meta0.get('segs_resized_area_ratio', None))}",
                            flush=True,
                        )
                        print(
                            f"[debug] 第7步-Pad到正方形(候选掩码) | batch_idx={bidx} | segs_square_after_pad_shape_hwk={meta0.get('segs_square_after_pad_shape_hwk', 'N/A')} | segs_square_tensor_shape_khw={meta0.get('segs_square_tensor_shape_khw', meta0.get('segs_square_after_align_shape_khw', 'N/A'))} | 前景占比={_fmt_ratio_stats(meta0.get('segs_square_area_ratio', None))}",
                            flush=True,
                        )
                        print(
                            f"[debug] 第8步-插值到256x256(候选掩码) | batch_idx={bidx} | segs_interp_256_shape_khw={meta0.get('segs_interp_256_shape_khw', 'N/A')} | 前景占比={_fmt_ratio_stats(meta0.get('segs_interp_256_area_ratio', None))}",
                            flush=True,
                        )
                        print(
                            f"[debug] 第9步-候选掩码数量 | batch_idx={bidx} | num_candidate_masks={meta0.get('num_candidate_masks', 'N/A')}",
                            flush=True,
                        )
                print("=" * 90 + "\n", flush=True)
                # train._train_epoch_debug_printed.add(epoch)  # ❗ 移除: 函数不能添加属性

            # print("forward for rank: ", args.local_rank)
            output_dict = model(**input_dict)
            # print("forward done for rank: ", args.local_rank)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            align_loss = output_dict["align_loss"]
            regression_loss = output_dict["regression_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            align_losses.update(align_loss.item(), input_dict["images"].size(0))
            regression_losses.update(regression_loss.item(), input_dict["images"].size(0))
            
            # print("backward for rank: ", args.local_rank)
            model.backward(loss)
            model.step()
            # print("backward done for rank: ", args.local_rank)
            
            # 训练可视化（随机抽样）：参考 finetune_llmseg_copy.py，直接复用训练 forward 的 output_dict
            if selected_in_batch:
                try:
                    for batch_sample_idx in selected_in_batch:
                        # 获取预测相似度并选择最佳候选mask
                        pred_similarity = output_dict["pred_similarity"][batch_sample_idx]
                        max_idx = torch.argmax(pred_similarity).item()

                        # 读取图像和mask
                        image_path = input_dict.get('image_paths', [None])[batch_sample_idx]
                        # 兼容：path 可能不是纯 str（例如 numpy.str_ / PathLike）
                        if image_path is not None and not isinstance(image_path, str):
                            try:
                                image_path = os.fspath(image_path)
                            except Exception:
                                image_path = str(image_path)

                        if not image_path or (isinstance(image_path, str) and not os.path.exists(image_path)):
                            if args.local_rank == 0:
                                print(f"  [训练可视化警告] 图像路径不存在: {image_path!r} (type={type(image_path)})", flush=True)
                            continue

                        image = cv2.imread(image_path)
                        if image is None:
                            if args.local_rank == 0:
                                print(f"  [训练可视化警告] cv2.imread 失败: {image_path!r}", flush=True)
                            continue

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        sam_segs = input_dict["origin_segs_list"][batch_sample_idx]
                        gt_mask = output_dict["gt_masks"][batch_sample_idx]

                        if sam_segs is not None and gt_mask is not None:
                            # 获取预测的mask（选择相似度最高的候选mask）
                            if hasattr(sam_segs, "shape") and len(sam_segs.shape) == 3:
                                K = sam_segs.shape[2]
                                if max_idx < 0 or max_idx >= K:
                                    if args.local_rank == 0:
                                        print(f"  [训练可视化警告] max_idx 越界: max_idx={max_idx}, K={K}, image_path={image_path!r}", flush=True)
                                    continue
                            pred_seg = sam_segs[:, :, max_idx]  # (H, W)
                            pred_seg = torch.from_numpy(pred_seg).unsqueeze(0)  # (1, H, W)
                            # 使用正确的 device
                            device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
                            pred_seg = pred_seg.to(device=device)
                            
                            # 调整大小以匹配GT mask
                            if pred_seg.shape != gt_mask.shape:
                                pred_seg = torch.nn.functional.interpolate(
                                    pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
                                ).squeeze(0)
                            
                            # 转换为numpy数组并调整大小以匹配图像
                            pred_mask_np = pred_seg.detach().cpu().numpy()[0]
                            gt_mask_np = gt_mask.detach().cpu().numpy()[0]
                            h, w = image.shape[:2]
                            if pred_mask_np.shape != (h, w):
                                pred_mask_np = cv2.resize(pred_mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            if gt_mask_np.shape != (h, w):
                                gt_mask_np = cv2.resize(gt_mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            
                            # 归一化mask到0-1范围
                            if pred_mask_np.max() > 1:
                                pred_mask_np = pred_mask_np.astype(np.float32) / 255.0
                            else:
                                pred_mask_np = pred_mask_np.astype(np.float32)
                            if gt_mask_np.max() > 1:
                                gt_mask_np = gt_mask_np.astype(np.float32) / 255.0
                            else:
                                gt_mask_np = gt_mask_np.astype(np.float32)
                            
                            # 黑色部分（值为0）是掩码区域
                            pred_mask_region = (pred_mask_np == 0)
                            gt_mask_region = (gt_mask_np == 0)
                            
                            # 创建叠加图像：左侧GT，右侧预测
                            gt_overlay = image.copy()
                            gt_overlay[gt_mask_region] = (
                                image[gt_mask_region] * 0.5 + 
                                np.array([0, 255, 0]) * 0.5
                            ).astype(np.uint8)
                            
                            pred_overlay = image.copy()
                            pred_overlay[pred_mask_region] = (
                                image[pred_mask_region] * 0.5 + 
                                np.array([255, 0, 0]) * 0.5
                            ).astype(np.uint8)

                            # 后处理/保存（下面这段原本缩进多了一层，直接导致 IndentationError）
                            if True:
                                # 拼接图像
                                combined_image = np.hstack([gt_overlay, pred_overlay])
                                    
                                # ✅ 从dataset获取GT物体名称
                                object_name = input_dict.get('object_name_list', ['object'])[batch_sample_idx]
                                
                                # 计算IoU
                                intersection = np.logical_and(pred_mask_region, gt_mask_region)
                                union = np.logical_or(pred_mask_region, gt_mask_region)
                                current_iou = np.sum(intersection) / (np.sum(union) + 1e-8)
                                
                                # 获取mask路径（用于显示）
                                gt_mask_path = input_dict['segmentation_paths'][batch_sample_idx]
                                if len(gt_mask_path) > 60:
                                    gt_mask_path = "..." + gt_mask_path[-57:]
                                candidate_paths = input_dict['candidate_mask_paths_list'][batch_sample_idx]
                                pred_mask_path = candidate_paths[max_idx] if len(candidate_paths) > max_idx else ""
                                if pred_mask_path and len(pred_mask_path) > 60:
                                    pred_mask_path = "..." + pred_mask_path[-57:]
                                
                                # 添加文字标题和路径信息
                                title_text = f"Object: {object_name} | IoU: {current_iou:.3f} | Epoch: {epoch} | Step: {global_step}"
                                left_text = f"GT (Green): {gt_mask_path}" if gt_mask_path else "GT (Green)"
                                right_text = f"Pred (Red): {pred_mask_path}" if pred_mask_path else "Pred (Red)"
                                
                                # 在图像上添加文字
                                text_height = 120 if (gt_mask_path or pred_mask_path) else 80
                                combined_image_with_text = np.ones((combined_image.shape[0] + text_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
                                combined_image_with_text[text_height:, :] = combined_image
                                
                                # 添加标题和标签
                                cv2.putText(combined_image_with_text, title_text, 
                                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                cv2.putText(combined_image_with_text, left_text, 
                                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                cv2.putText(combined_image_with_text, right_text, 
                                           (w + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                
                                # 如果路径太长，换行显示
                                if gt_mask_path and len(gt_mask_path) > 60:
                                    cv2.putText(combined_image_with_text, gt_mask_path[:60], 
                                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                if pred_mask_path and len(pred_mask_path) > 60:
                                    cv2.putText(combined_image_with_text, pred_mask_path[:60], 
                                               (w + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                
                                # 保存图像
                                save_dir = os.path.join(args.log_dir, args.train_vis_dir)
                                os.makedirs(save_dir, exist_ok=True)
                                image_name = os.path.basename(image_path)
                                save_path = os.path.join(save_dir, f"epoch{epoch}_step{global_step}_sample{vis_count}_{image_name}")
                                combined_image_bgr = cv2.cvtColor(combined_image_with_text, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(save_path, combined_image_bgr)
                                print(f"  [训练可视化] 已保存图片: {save_path}")

                                # 打印 gt_ious 和 pred_similarity 的分布概率值（IoU相关信息）
                                print(f"\n{'='*80}")
                                print(f"  [分布概率值] Epoch: {epoch}, Step: {global_step}, Sample: {vis_count}")
                                print(f"  图像: {image_name}")
                                print(f"  选择的mask索引: {max_idx}")
                                print(f"{'='*80}")
                                    
                                # 获取gt_ious并处理为(K, 1)格式
                                gt_ious_raw = input_dict['sam_ious_list'][batch_sample_idx]
                                if isinstance(gt_ious_raw, list):
                                    gt_ious_round = torch.tensor(gt_ious_raw[0], dtype=torch.float32)
                                elif isinstance(gt_ious_raw, torch.Tensor):
                                    gt_ious_round = gt_ious_raw.cpu()
                                else:
                                    gt_ious_round = torch.tensor(gt_ious_raw, dtype=torch.float32)
                                
                                # 处理维度：如果是2D则取第一行，如果是1D则保持
                                if len(gt_ious_round.shape) == 2:
                                    gt_ious_round = gt_ious_round[0]
                                if len(gt_ious_round.shape) == 1:
                                    gt_ious_round = gt_ious_round.unsqueeze(1)
                                
                                # 计算gt_ious的softmax分布（用于对齐损失）
                                # ✅ 使用模型中的温度参数（与训练时一致）
                                temperature = args.align_temperature if hasattr(args, 'align_temperature') else 0.1
                                gt_iou_temp = gt_ious_round / temperature
                                gt_dis = torch.nn.functional.softmax(gt_iou_temp, dim=0)
                                
                                # 处理pred_similarity为(K, 1)格式
                                pred_similarity_cpu = pred_similarity.detach().cpu()
                                if len(pred_similarity_cpu.shape) == 2:
                                    sim_scores = pred_similarity_cpu.squeeze(0)
                                else:
                                    sim_scores = pred_similarity_cpu
                                if len(sim_scores.shape) == 1:
                                    sim_scores = sim_scores.unsqueeze(1)
                                
                                # 计算 sim_dis
                                sim_scores_temp = sim_scores / temperature
                                sim_dis = torch.nn.functional.softmax(sim_scores_temp, dim=0)
                                
                                # # 打印IoU相关信息
                                # K = gt_ious_round.shape[0]
                                # print(f"\n  候选mask数量 (K): {K}")
                                # print(f"\n  1. 原始 gt_ious (IoU值):")
                                # for i in range(K):
                                #     print(f"     mask[{i:2d}]: {gt_ious_round[i, 0].item():.6f}")
                                # print(f"\n  2. gt_ious 经过 softmax 后的分布 (gt_dis):")
                                # for i in range(K):
                                #     print(f"     mask[{i:2d}]: {gt_dis[i, 0].item():.10f}")
                                # print(f"\n  3. 原始 pred_similarity (相似度分数):")
                                # for i in range(K):
                                #     print(f"     mask[{i:2d}]: {sim_scores[i, 0].item():.6f}")
                                # print(f"\n  4. pred_similarity 经过 softmax 后的分布 (sim_dis):")
                                # for i in range(K):
                                #     print(f"     mask[{i:2d}]: {sim_dis[i, 0].item():.10f}")
                                
                                # 诊断：pred_similarity 值差异小的原因分析
                                sim_scores_min = sim_scores.min().item()
                                sim_scores_max = sim_scores.max().item()
                                sim_scores_mean = sim_scores.mean().item()
                                sim_scores_std = sim_scores.std().item()
                                print(f"\n  [诊断] pred_similarity 统计信息:")
                                print(f"     最小值: {sim_scores_min:.6f}")
                                print(f"     最大值: {sim_scores_max:.6f}")
                                print(f"     平均值: {sim_scores_mean:.6f}")
                                print(f"     标准差: {sim_scores_std:.6f}")
                                print(f"     范围: {sim_scores_max - sim_scores_min:.6f} (仅占余弦相似度全范围[-1,1]的 {(sim_scores_max - sim_scores_min) / 2.0 * 100:.2f}%)")
                                
                                # ✅ 诊断：检查 pred_embeddings 和 segs_feature 的区分度
                                print(f"\n  [诊断] 特征区分度分析:")
                                
                                # 1. 检查 pred_embeddings（文本特征）的区分度
                                try:
                                    pred_embeddings_raw = output_dict.get("pred_embeddings", None)
                                    if pred_embeddings_raw is not None:
                                        pred_emb = pred_embeddings_raw[batch_sample_idx]  # (C, D) 或 (1, D)
                                        if isinstance(pred_emb, torch.Tensor):
                                            pred_emb_cpu = pred_emb.detach().cpu()
                                            if len(pred_emb_cpu.shape) == 2 and pred_emb_cpu.shape[0] > 1:
                                                # 多个对话轮次，计算不同轮次之间的相似度
                                                pred_emb_norm = pred_emb_cpu / pred_emb_cpu.norm(dim=-1, keepdim=True)
                                                inter_round_sim = pred_emb_norm @ pred_emb_norm.T  # (C, C)
                                                print(f"     pred_embeddings (文本特征):")
                                                print(f"       形状: {list(pred_emb_cpu.shape)}")
                                                print(f"       不同轮次之间的相似度矩阵:")
                                                for i in range(pred_emb_cpu.shape[0]):
                                                    sim_str = " ".join([f"{inter_round_sim[i, j].item():.4f}" for j in range(pred_emb_cpu.shape[0])])
                                                    print(f"         轮次[{i}]: [{sim_str}]")
                                                # 计算非对角线元素的平均值（不同轮次之间的相似度）
                                                mask = ~torch.eye(pred_emb_cpu.shape[0], dtype=torch.bool)
                                                inter_round_mean = inter_round_sim[mask].mean().item()
                                                print(f"       不同轮次之间的平均相似度: {inter_round_mean:.6f}")
                                                print(f"       {'⚠️  如果接近1，说明文本特征区分度不够' if inter_round_mean > 0.9 else '✅ 文本特征有较好的区分度'}")
                                            else:
                                                # 单个对话轮次
                                                pred_emb_norm = pred_emb_cpu / pred_emb_cpu.norm(dim=-1, keepdim=True)
                                                print(f"     pred_embeddings (文本特征):")
                                                print(f"       形状: {list(pred_emb_cpu.shape)}")
                                                print(f"       单轮对话，无法计算轮次间相似度")
                                                # 计算特征向量的范数
                                                emb_norm = pred_emb_cpu.norm().item()
                                                print(f"       特征向量范数: {emb_norm:.6f}")
                                except Exception as e:
                                    print(f"     [错误] 无法分析 pred_embeddings: {e}")
                                
                                # 2. 检查 segs_feature（mask特征）的区分度 - 分步骤诊断
                                print(f"\n  [诊断] mask特征相似度 - 分步骤检查:")
                                
                                def compute_similarity_stats(feat, step_name):
                                    """计算特征相似度统计"""
                                    if feat is None or not isinstance(feat, torch.Tensor):
                                        return None
                                    feat_cpu = feat.detach().cpu()
                                    # 处理维度
                                    if len(feat_cpu.shape) == 3:
                                        feat_round = feat_cpu[-1]  # (K, D) - 使用最后一轮
                                    elif len(feat_cpu.shape) == 2:
                                        feat_round = feat_cpu  # (K, D)
                                    else:
                                        return None
                                    
                                    K_check = feat_round.shape[0]
                                    if K_check < 2:
                                        return None
                                    
                                    # 归一化并计算相似度
                                    feat_norm = feat_round / (feat_round.norm(dim=-1, keepdim=True) + 1e-8)
                                    inter_mask_sim = feat_norm @ feat_norm.T  # (K, K)
                                    inter_mask_sim = torch.clamp(inter_mask_sim, -1.0, 1.0)
                                    
                                    # 计算统计信息
                                    mask_matrix = ~torch.eye(K_check, dtype=torch.bool)
                                    inter_mask_mean = inter_mask_sim[mask_matrix].mean().item()
                                    inter_mask_min = inter_mask_sim[mask_matrix].min().item()
                                    inter_mask_max = inter_mask_sim[mask_matrix].max().item()
                                    inter_mask_std = inter_mask_sim[mask_matrix].std().item()
                                    
                                    return {
                                        "mean": inter_mask_mean,
                                        "min": inter_mask_min,
                                        "max": inter_mask_max,
                                        "std": inter_mask_std,
                                        "shape": list(feat_round.shape)
                                    }
                                
                                # 检查第4步：mask_pooling之后
                                try:
                                    feat_after_pooling = output_dict.get("sam_segs_feature_after_pooling", None)
                                    if feat_after_pooling is not None and batch_sample_idx < len(feat_after_pooling):
                                        stats_pooling = compute_similarity_stats(feat_after_pooling[batch_sample_idx], "mask_pooling")
                                        if stats_pooling:
                                            print(f"     第4步（mask_pooling之后）:")
                                            print(f"       形状: {stats_pooling['shape']}")
                                            print(f"       不同mask平均相似度: {stats_pooling['mean']:.6f}")
                                            print(f"       不同mask最小相似度: {stats_pooling['min']:.6f}")
                                            print(f"       不同mask最大相似度: {stats_pooling['max']:.6f}")
                                            print(f"       不同mask相似度标准差: {stats_pooling['std']:.6f}")
                                            if stats_pooling['mean'] > 0.95:
                                                print(f"       ⚠️  问题: 初始mask特征就已经过于相似！")
                                            elif stats_pooling['mean'] < 0.5:
                                                print(f"       ✅ 初始mask特征有较好的区分度")
                                except Exception as e:
                                    print(f"     [错误] 无法分析第4步特征: {e}")
                                
                                # 检查第6步：attention之后
                                try:
                                    feat_after_attention = output_dict.get("sam_segs_feature_after_attention", None)
                                    if feat_after_attention is not None and batch_sample_idx < len(feat_after_attention):
                                        stats_attention = compute_similarity_stats(feat_after_attention[batch_sample_idx], "attention")
                                        if stats_attention:
                                            print(f"     第6步（attention之后）:")
                                            print(f"       形状: {stats_attention['shape']}")
                                            print(f"       不同mask平均相似度: {stats_attention['mean']:.6f}")
                                            print(f"       不同mask最小相似度: {stats_attention['min']:.6f}")
                                            print(f"       不同mask最大相似度: {stats_attention['max']:.6f}")
                                            print(f"       不同mask相似度标准差: {stats_attention['std']:.6f}")
                                            if stats_attention['mean'] > 0.95:
                                                print(f"       ⚠️  严重问题: attention层让mask特征变得过于相似！")
                                            elif stats_attention['mean'] < 0.5:
                                                print(f"       ✅ attention层保持了较好的区分度")
                                except Exception as e:
                                    print(f"     [错误] 无法分析第6步特征: {e}")
                                
                                # 检查第7步：embedding_head之后（最终特征）
                                try:
                                    sam_features_raw = output_dict.get("sam_segs_feature_list", None)
                                    if sam_features_raw is not None:
                                        sam_feat = sam_features_raw[batch_sample_idx]  # (C, K, D) 或 (K, D)
                                        stats_final = compute_similarity_stats(sam_feat, "embedding_head")
                                        if stats_final:
                                            print(f"     第7步（embedding_head之后，最终特征）:")
                                            print(f"       形状: {stats_final['shape']}")
                                            print(f"       不同mask平均相似度: {stats_final['mean']:.6f}")
                                            print(f"       不同mask最小相似度: {stats_final['min']:.6f}")
                                            print(f"       不同mask最大相似度: {stats_final['max']:.6f}")
                                            print(f"       不同mask相似度标准差: {stats_final['std']:.6f}")
                                            
                                            if stats_final['mean'] > 0.95:
                                                print(f"       ⚠️  严重问题: 最终mask特征过于相似（平均相似度>{stats_final['mean']:.2f}）")
                                                # 判断问题出现在哪一步
                                                if stats_attention and stats_attention['mean'] > 0.95:
                                                    print(f"       问题出现在: 第6步（attention层）")
                                                elif stats_pooling and stats_pooling['mean'] > 0.95:
                                                    print(f"       问题出现在: 第4步（mask_pooling）")
                                                else:
                                                    print(f"       问题出现在: 第7步（embedding_head）")
                                            elif stats_final['mean'] > 0.85:
                                                print(f"       ⚠️  问题: mask特征相似度较高（平均相似度>{stats_final['mean']:.2f}）")
                                            else:
                                                print(f"       ✅ 最终mask特征有较好的区分度")
                                except Exception as e:
                                    print(f"     [错误] 无法分析第7步特征: {e}")
                                
                                # 3. 总结分析
                                print(f"\n  [诊断] pred_similarity 计算过程分析:")
                                print(f"     pred_similarity 计算: pred_embedding_norm @ sam_features_norm.T")
                                print(f"     - pred_embedding: 文本embedding（来自语言模型hidden states）")
                                print(f"     - sam_features: mask特征（mask_pooling -> attention -> embedding_head）")
                                print(f"     - 两者都经过L2归一化后计算余弦相似度")
                                print(f"     可能原因:")
                                print(f"       1. attention层让所有mask特征变得相似（过度关注文本特征）")
                                print(f"       2. lisa_embedding_head 未充分学习区分性特征")
                                print(f"       3. 文本embedding本身区分度不够")
                                print(f"       4. 训练不充分，特征还未收敛")
                                
                                # 打印最大值索引和KL散度
                                gt_max_idx = torch.argmax(gt_ious_round, dim=0).item()
                                sim_max_idx = torch.argmax(sim_scores, dim=0).item()
                                print(f"\n  5. 最大值索引:")
                                print(f"     gt_ious 最大值索引: {gt_max_idx} (IoU={gt_ious_round[gt_max_idx, 0].item():.6f})")
                                print(f"     pred_similarity 最大值索引: {sim_max_idx} (相似度={sim_scores[sim_max_idx, 0].item():.6f})")
                                print(f"     选择的mask索引: {max_idx}")
                                print(f"     ⚠️  问题: gt_ious最大值在mask[{gt_max_idx}]，但pred_similarity最大值在mask[{sim_max_idx}]，不一致！")
                                kl_div = torch.nn.functional.kl_div(sim_dis.log(), gt_dis, reduction='sum')
                                print(f"\n  6. KL散度 (sim_dis || gt_dis): {kl_div.item():.6f} (值越大说明分布差异越大)")
                                print(f"{'='*80}\n")
                except Exception as e:
                    # 之前这里经常打印为空，改为 repr + traceback（只打印一次完整 traceback，避免刷屏）
                    import traceback
                    if not hasattr(train, "_train_vis_tb_printed"):
                        train._train_vis_tb_printed = True
                        print(f"  [训练可视化] 保存图片时出错: {repr(e)}", flush=True)
                        print(traceback.format_exc(), flush=True)
                    else:
                        print(f"  [训练可视化] 保存图片时出错: {repr(e)}", flush=True)

            # 更新 sample_idx（保留，可能用于其他调试/统计）
            sample_idx += batch_size

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                align_losses.all_reduce()
                regression_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/align_loss", align_losses.avg, global_step
                )
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )
                # 记录到 SwanLab（使用全局步数）
                if swanlab_logger is not None:
                    global_step_total = global_step_start + global_step
                    log_dict = {
                        "train/loss": losses.avg,
                        "train/ce_loss": ce_losses.avg,
                        "train/align_loss": align_losses.avg,
                        "train/regression_loss": regression_losses.avg,
                        "metrics/total_secs_per_batch": batch_time.avg,
                        "metrics/data_secs_per_batch": data_time.avg,
                    }
                    # ✅ 在第一个step时，将维度追踪信息记录到SwanLab
                    if epoch == 0 and global_step == 0:
                        # 检查是否有维度追踪信息
                        if len(_dimension_tracking_info) > 0:
                            dimension_info_text = "\n".join(_dimension_tracking_info)
                            try:
                                # 方式1: 尝试记录到config
                                swanlab.config.update({"dimension_tracking": dimension_info_text})
                                print(f"\n[信息] 维度追踪信息已记录到SwanLab config (共{len(_dimension_tracking_info)}行)")
                            except Exception as e1:
                                try:
                                    # 方式2: 如果config失败，尝试记录为文本指标（只记录一次）
                                    # 将文本信息分段记录（SwanLab可能不支持长文本）
                                    info_preview = dimension_info_text[:500] + "..." if len(dimension_info_text) > 500 else dimension_info_text
                                    log_dict["debug/dimension_tracking_preview"] = info_preview
                                    print(f"\n[信息] 维度追踪信息已记录到SwanLab log (预览，共{len(_dimension_tracking_info)}行)")
                                    # 同时打印完整信息到终端
                                    print(f"\n[完整维度追踪信息]\n{dimension_info_text}\n")
                                except Exception as e2:
                                    print(f"\n[警告] 无法将维度追踪信息记录到SwanLab: config错误={e1}, log错误={e2}")
                                    print(f"\n[完整维度追踪信息]\n{dimension_info_text}\n")
                        else:
                            print(f"\n[警告] 维度追踪信息列表为空，可能Dataset/Model的打印还未执行")
                    swanlab.log(log_dict, step=global_step_total)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            align_losses.reset()
            regression_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)
                # 记录学习率到 SwanLab（使用全局步数）
                if swanlab_logger is not None:
                    global_step_total = global_step_start + global_step
                    swanlab.log({"train/lr": curr_lr[0]}, step=global_step_total)

    return train_iter
        

def validate(val_loader, model_engine, epoch, writer, args, swanlab_logger=None):
    print("start validating ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    # ✅ 随机选择3张图片进行可视化
    import random
    max_vis_samples = 4
    # 先获取验证集的总长度（用于随机选择）
    try:
        total_samples = len(val_loader.dataset)
    except:
        total_samples = 1000  # 如果无法获取，使用一个较大的默认值
    
    # ✅ 修复：在分布式训练中，需要考虑每个GPU只处理部分样本
    # 获取当前GPU处理的样本范围
    if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'num_replicas'):
        num_replicas = val_loader.sampler.num_replicas
        rank = val_loader.sampler.rank
        samples_per_gpu = total_samples // num_replicas
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank < num_replicas - 1 else total_samples
        # 在当前GPU的样本范围内随机选择
        local_total = end_idx - start_idx
        if local_total > max_vis_samples:
            local_selected = random.sample(range(local_total), max_vis_samples)
            selected_indices = set(start_idx + idx for idx in local_selected)
        else:
            selected_indices = set(range(start_idx, end_idx))
    else:
        # 非分布式训练，直接随机选择
        if total_samples > max_vis_samples:
            selected_indices = set(random.sample(range(total_samples), max_vis_samples))
        else:
            selected_indices = set(range(total_samples))
    
    vis_count = 0
    sample_idx = 0  # 当前样本索引（在当前GPU中的局部索引）

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        # 使用正确的 device
        device = next(model_engine.parameters()).device
        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)
        # ✅ 验证阶段需要走推理分支，模型才会返回 pred_similarity / gt_masks 等字段
        input_dict["inference"] = True


        with torch.no_grad():
                output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        pred_seg = sam_segs[:, :, max_idx] # (H, W)
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        # send pred_seg and gt_mask to GPU (使用正确的 device)
        device = next(model_engine.parameters()).device
        pred_seg = pred_seg.to(device=device)
        gt_mask = gt_mask.to(device=device)

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

        # ✅ 可视化：随机选择3张图片（GT掩码和预测掩码拼接）
        # 注意：在分布式训练中，sample_idx是局部索引，需要转换为全局索引
        if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'num_replicas'):
            num_replicas = val_loader.sampler.num_replicas
            rank = val_loader.sampler.rank
            samples_per_gpu = len(val_loader.dataset) // num_replicas
            global_idx = rank * samples_per_gpu + sample_idx
        else:
            global_idx = sample_idx
        should_visualize = (args.local_rank == 0 and global_idx in selected_indices and vis_count < max_vis_samples)
        if should_visualize:
            try:
                # 获取图像路径
                image_path = input_dict['image_paths'][0]
                # ✅ 添加调试信息：检查为什么只有2张图
                if vis_count == 0:
                    print(f"  [可视化调试] 验证集总样本数: {total_samples}, 选中的索引: {selected_indices}")
                if not os.path.exists(image_path):
                    print(f"  [可视化警告] 图像路径不存在: {image_path}, sample_idx={sample_idx}, vis_count={vis_count}")
                if os.path.exists(image_path):
                    # 读取原始图像
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 转换为numpy数组
                    # ✅ 修正：mask的黑色部分（值为0）才是掩码区域，白色部分（值为1）保持原图
                    pred_mask_np = pred_seg.detach().cpu().numpy()[0]
                    gt_mask_np = gt_mask.detach().cpu().numpy()[0]
                    if len(gt_mask_np.shape) == 2:
                        pass  # 已经是2D
                    
                    # 调整掩码大小以匹配图像
                    h, w = image.shape[:2]
                    if pred_mask_np.shape != (h, w):
                        pred_mask_np = cv2.resize(pred_mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    if gt_mask_np.shape != (h, w):
                        gt_mask_np = cv2.resize(gt_mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # ✅ 黑色部分（值为0）是掩码区域，白色部分（值为1或255）是非掩码区域
                    # 将mask归一化到0-1范围
                    if pred_mask_np.max() > 1:
                        pred_mask_np = pred_mask_np.astype(np.float32) / 255.0
                    else:
                        pred_mask_np = pred_mask_np.astype(np.float32)
                    if gt_mask_np.max() > 1:
                        gt_mask_np = gt_mask_np.astype(np.float32) / 255.0
                    else:
                        gt_mask_np = gt_mask_np.astype(np.float32)
                    
                    # ✅ 修正：黑色部分（值为0）是掩码区域，白色部分（值>0）是背景
                    # 掩码区域：值为0（黑色）的地方，白色部分（值>0）保持原图
                    # 注意：mask已经转换为二值，值为0是掩码区域，值为1是背景
                    pred_mask_region = (pred_mask_np == 0)  # 值为0（黑色）是掩码区域
                    gt_mask_region = (gt_mask_np == 0)     # 值为0（黑色）是掩码区域
                    
                    # 创建叠加图像：左侧GT，右侧预测
                    # GT掩码叠加（绿色）- 只在黑色区域叠加，白色部分保持原图
                    gt_overlay = image.copy()
                    gt_overlay[gt_mask_region] = (
                        image[gt_mask_region] * 0.5 + 
                        np.array([0, 255, 0]) * 0.5
                    ).astype(np.uint8)
                    
                    # 预测掩码叠加（红色）- 只在黑色区域叠加，白色部分保持原图
                    pred_overlay = image.copy()
                    pred_overlay[pred_mask_region] = (
                        image[pred_mask_region] * 0.5 + 
                        np.array([255, 0, 0]) * 0.5
                    ).astype(np.uint8)
                    
                    # 拼接图像：左侧GT，右侧预测
                    combined_image = np.hstack([gt_overlay, pred_overlay])
                    
                    # ✅ 从dataset获取GT物体名称
                    object_name = input_dict.get('object_name_list', ['object'])[0]
                    
                    # 计算IoU用于显示
                    # ✅ 修正：黑色部分（值为0）是掩码区域，所以使用类别0
                    current_iou = acc_iou[0]  # 类别0是掩码区域（前景）
                    
                    # 获取mask路径信息
                    gt_mask_path = ""
                    pred_mask_path = ""
                    
                    # GT mask路径
                    if 'segmentation_paths' in input_dict and len(input_dict['segmentation_paths']) > 0:
                        gt_mask_path = input_dict['segmentation_paths'][0]
                        if gt_mask_path:
                            # 只显示文件名和部分路径（避免太长）
                            if len(gt_mask_path) > 60:
                                gt_mask_path = "..." + gt_mask_path[-57:]
                    
                    # PRED mask路径（候选掩码）
                    if 'candidate_mask_paths_list' in input_dict and len(input_dict['candidate_mask_paths_list']) > 0:
                        candidate_paths = input_dict['candidate_mask_paths_list'][0]
                        if candidate_paths and len(candidate_paths) > max_idx:
                            pred_mask_path = candidate_paths[max_idx]
                            if pred_mask_path:
                                # 只显示文件名和部分路径
                                if len(pred_mask_path) > 60:
                                    pred_mask_path = "..." + pred_mask_path[-57:]
                    
                    # 添加文字标题
                    title_text = f"Object: {object_name} | IoU: {current_iou:.3f} | Epoch: {epoch}"
                    left_text = f"GT (Green): {gt_mask_path}" if gt_mask_path else "GT (Green)"
                    right_text = f"Pred (Red): {pred_mask_path}" if pred_mask_path else "Pred (Red)"
                    
                    # 在图像上添加文字（增加高度以容纳路径信息）
                    text_height = 120 if (gt_mask_path or pred_mask_path) else 80
                    combined_image_with_text = np.ones((combined_image.shape[0] + text_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
                    combined_image_with_text[text_height:, :] = combined_image
                    
                    # 添加标题（顶部居中）
                    cv2.putText(combined_image_with_text, title_text, 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # 添加左右标签（包含路径）- 使用黑色字体，字号0.7，粗细2（必须是整数）
                    cv2.putText(combined_image_with_text, left_text, 
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # 黑色，字号0.7，粗细2
                    cv2.putText(combined_image_with_text, right_text, 
                               (w + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # 黑色，字号0.7，粗细2
                    
                    # 如果路径太长，换行显示
                    if gt_mask_path and len(gt_mask_path) > 60:
                        cv2.putText(combined_image_with_text, gt_mask_path[:60], 
                                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色，字号0.5
                    if pred_mask_path and len(pred_mask_path) > 60:
                        cv2.putText(combined_image_with_text, pred_mask_path[:60], 
                                   (w + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色，字号0.5
                    
                    # 保存图像
                    save_dir = os.path.join(args.log_dir, args.val_vis_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    image_name = os.path.basename(image_path)
                    save_path = os.path.join(save_dir, f"epoch{epoch}_sample{vis_count}_{image_name}")
                    
                    # 转换为BGR格式保存
                    combined_image_bgr = cv2.cvtColor(combined_image_with_text, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, combined_image_bgr)
                    
                    print(f"  [可视化] 已保存验证图片: {save_path}")
                    vis_count += 1
            except Exception as e:
                print(f"  [可视化] 保存图片时出错: {e}")
        
        sample_idx += 1  # 更新样本索引

            # # compute iou
            # intersection, union, acc_iou = 0.0, 0.0, 0.0
            # for mask_i, output_i in zip(masks_list, output_list):
            #     intersection_i, union_i, _ = intersectionAndUnionGPU(
            #         output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            #     )
            #     intersection += intersection_i
            #     union += union_i
            #     acc_iou += intersection_i / (union_i + 1e-5)
            #     acc_iou[union_i == 0] += 1.0  # no-object target
            # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            # intersection_meter.update(intersection), union_meter.update(
            #     union
            # ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # ✅ 修正：黑色部分（值为0）是掩码区域，所以使用类别0而不是类别1
    ciou = iou_class[0]  # 类别0是掩码区域（前景）
    giou = acc_iou_meter.avg[0]  # 类别0是掩码区域（前景）

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        # 记录到 SwanLab
        if swanlab_logger is not None:
            swanlab.log({
                "val/giou": giou,
                "val/ciou": ciou,
            }, step=epoch)
        
    return giou, ciou


def validate_threshold(val_loader, model_engine, epoch, writer, args, threshold=0.5, swanlab_logger=None):
    print("start validating ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    # ✅ 随机选择3张图片进行可视化
    import random
    max_vis_samples = 3
    # 先获取验证集的总长度（用于随机选择）
    try:
        total_samples = len(val_loader.dataset)
    except:
        total_samples = 1000  # 如果无法获取，使用一个较大的默认值
    # 随机选择3个索引（不重复）
    if total_samples > max_vis_samples:
        selected_indices = set(random.sample(range(total_samples), max_vis_samples))
    else:
        selected_indices = set(range(total_samples))
    
    vis_count = 0
    sample_idx = 0  # 当前样本索引

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        # 使用正确的 device
        device = next(model_engine.parameters()).device
        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)

        with torch.no_grad():
            # ✅ 关键：验证/推理路径需要 inference=True，模型才会返回 pred_similarity / pred_iou / gt_masks
            input_dict["inference"] = True
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_iou"][0]
        # print(pred_similarity)
        # get the seg with highest similarity
        max_ids = []
        for i in range(pred_similarity.shape[1]):
            if pred_similarity[0][i] > threshold:
                max_ids.append(i)


        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        # pred_seg = sam_segs[:, :, max_idx] # (H, W)
        # pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        # ✅ 修正：黑色部分（值为0）是掩码区域，白色部分（值为1）是背景
        # 合并多个mask：如果某个位置在任何一个mask中是掩码区域（0），合并后也应该是掩码区域（0）
        # 使用逻辑或：如果所有mask在该位置都是背景（1），合并后才是背景（1）
        pred_seg = np.ones_like(sam_segs[:, :, 0])  # 初始化为全背景（1）
        for i in max_ids:
            # 如果某个位置在任何一个mask中是掩码区域（0），合并后也应该是掩码区域（0）
            pred_seg = np.minimum(pred_seg, sam_segs[:, :, i])  # 取最小值：0（掩码）优先
        pred_seg = pred_seg.astype(np.uint8)

        # send pred_seg and gt_mask to GPU
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)

        # # resize pred_seg and gt_mask to 1024x1024
        # pred_seg = torch.nn.functional.interpolate(
        #     pred_seg.unsqueeze(0), size=(1024, 1024), mode="nearest"
        # ).squeeze(0)
        # gt_mask = torch.nn.functional.interpolate(
        #     gt_mask.unsqueeze(0), size=(1024, 1024), mode="nearest"
        # ).squeeze(0)

        # 使用正确的 device
        device = next(model_engine.parameters()).device
        pred_seg = pred_seg.to(device=device)
        gt_mask = gt_mask.to(device=device)

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

        # ✅ 可视化：已取消 val_vis_threshold 的可视化保存
        # should_visualize = (args.local_rank == 0 and sample_idx in selected_indices and vis_count < max_vis_samples)
        # if should_visualize:
        #     ... (可视化代码已注释掉)
        
        sample_idx += 1  # 更新样本索引

        if args.eval_only and args.visualize:
            # save the evaluation result
            image_path = input_dict['image_paths'][0]
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))
                continue
            
            image_name = os.path.basename(image_path)
            # image and mask has the same shape
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pred_mask = pred_seg.detach().cpu().numpy()
            pred_mask = pred_mask[0]      
            pred_mask = pred_mask > 0      

            gt_mask = gt_mask.detach().cpu().numpy()
            gt_mask = gt_mask[0]
            gt_mask[gt_mask == 255] = 0  # ignored label
            gt_mask = gt_mask > 0

            overlap_image = image.copy()
            overlap_image[pred_mask] = (
                image * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            overlap_image = cv2.cvtColor(overlap_image, cv2.COLOR_RGB2BGR)
            
            overlap_image_gt = image.copy()
            overlap_image_gt[gt_mask] = (
                image * 0.5
                + gt_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[gt_mask]
            overlap_image_gt = cv2.cvtColor(overlap_image_gt, cv2.COLOR_RGB2BGR)

            # process pred_mask for save
            pred_mask = pred_mask.astype(np.uint8) * 255
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

            # ✅ 修正：黑色部分（值为0）是掩码区域，所以使用类别0
            iou = acc_iou[0]  # 类别0是掩码区域（前景）

            # conversations is a string
            conversations = input_dict["conversation_list"][0]
            conversations = conversations.replace("<im_patch>", "")


            # save to dir
            save_dir = os.path.join(args.log_dir, args.eval_vis_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            all_iops = pred_similarity
            # save conversations as text
            with open(os.path.join(save_dir, image_name+".txt"), 'w') as file:
                file.write(conversations)
                file.write("\n\n")
                file.write("iou = "+str(iou))
                file.write("\n\n")
                file.write("all_iops = "+str(all_iops))

            # convert image back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # save images
            cv2.imwrite(os.path.join(save_dir, image_name+".png"), image)
            cv2.imwrite(os.path.join(save_dir, image_name+"_mask.png"), pred_mask)
            cv2.imwrite(os.path.join(save_dir, image_name+"_pred.png"), overlap_image)
            cv2.imwrite(os.path.join(save_dir, image_name+"_gt.png"), overlap_image_gt)

            # # compute iou
            # intersection, union, acc_iou = 0.0, 0.0, 0.0
            # for mask_i, output_i in zip(masks_list, output_list):
            #     intersection_i, union_i, _ = intersectionAndUnionGPU(
            #         output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            #     )
            #     intersection += intersection_i
            #     union += union_i
            #     acc_iou += intersection_i / (union_i + 1e-5)
            #     acc_iou[union_i == 0] += 1.0  # no-object target
            # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            # intersection_meter.update(intersection), union_meter.update(
            #     union
            # ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # ✅ 修正：黑色部分（值为0）是掩码区域，所以使用类别0而不是类别1
    ciou = iou_class[0]  # 类别0是掩码区域（前景）
    giou = acc_iou_meter.avg[0]  # 类别0是掩码区域（前景）

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        # 记录到 SwanLab
        if swanlab_logger is not None:
            swanlab.log({
                "val/giou_threshold": giou,
                "val/ciou_threshold": ciou,
            }, step=epoch)
        
    return giou, ciou

def validate_iou_iop(val_loader, model_engine, epoch, writer, args, threshold=0.5, swanlab_logger=None):
    print("start validating using iou+iop ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        # 使用正确的 device
        device = next(model_engine.parameters()).device
        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)

        with torch.no_grad():
            # ✅ 关键：验证/推理路径需要 inference=True，模型才会返回 pred_similarity / pred_iou / gt_masks
            input_dict["inference"] = True
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        pred_iop = output_dict["pred_iou"][0]

        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        # pred_seg = sam_segs[:, :, max_idx] # (H, W)
        # pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)

        max_ids = [max_idx]
        for i in range(pred_iop.shape[1]):
            if pred_iop[0][i] > threshold and i != max_idx:
                max_ids.append(i)

        # ✅ 修正：黑色部分（值为0）是掩码区域，白色部分（值为1）是背景
        # 合并多个mask：如果某个位置在任何一个mask中是掩码区域（0），合并后也应该是掩码区域（0）
        # 使用逻辑或：如果所有mask在该位置都是背景（1），合并后才是背景（1）
        pred_seg = np.ones_like(sam_segs[:, :, 0])  # 初始化为全背景（1）
        for i in max_ids:
            # 如果某个位置在任何一个mask中是掩码区域（0），合并后也应该是掩码区域（0）
            pred_seg = np.minimum(pred_seg, sam_segs[:, :, i])  # 取最小值：0（掩码）优先
        pred_seg = pred_seg.astype(np.uint8)

        # send pred_seg and gt_mask to GPU (使用正确的 device)
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        device = next(model_engine.parameters()).device
        pred_seg = pred_seg.to(device=device)
        gt_mask = gt_mask.to(device=device)

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # ✅ 修正：黑色部分（值为0）是掩码区域，所以使用类别0而不是类别1
    ciou = iou_class[0]  # 类别0是掩码区域（前景）
    giou = acc_iou_meter.avg[0]  # 类别0是掩码区域（前景）

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        # 记录到 SwanLab
        if swanlab_logger is not None:
            swanlab.log({
                "val/giou_iou_iop": giou,
                "val/ciou_iou_iop": ciou,
            }, step=epoch)
        
    return giou, ciou

def validate_threshold_from_topIoU(val_loader, model_engine, epoch, writer, args, threshold=0.5, swanlab_logger=None):
    print("start validating using threshold from top IoU ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        # 使用正确的 device
        device = next(model_engine.parameters()).device
        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype, device=device)

        with torch.no_grad():
            # ✅ 关键：验证/推理路径需要 inference=True，模型才会返回 pred_similarity / pred_iou / gt_masks
            input_dict["inference"] = True
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        pred_iop = output_dict["pred_iou"][0]

        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        # pred_seg = sam_segs[:, :, max_idx] # (H, W)
        # pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)

        # max_ids = [max_idx]
        # for i in range(pred_iop.shape[1]):
        #     if pred_iop[0][i] > threshold and i != max_idx:
        #         max_ids.append(i)

        K = 5
        if K > pred_similarity.shape[-1]:
            K = pred_similarity.shape[-1]
        
        # import pdb; pdb.set_trace()


        topK_ids = torch.topk(pred_similarity[0], K, dim=0).indices
        max_ids = []
        for i in topK_ids:
            if pred_iop[0][i] > threshold:
                max_ids.append(i)

        # ✅ 修正：黑色部分（值为0）是掩码区域，白色部分（值为1）是背景
        # 合并多个mask：如果某个位置在任何一个mask中是掩码区域（0），合并后也应该是掩码区域（0）
        # 使用逻辑或：如果所有mask在该位置都是背景（1），合并后才是背景（1）
        pred_seg = np.ones_like(sam_segs[:, :, 0])  # 初始化为全背景（1）
        for i in max_ids:
            # 如果某个位置在任何一个mask中是掩码区域（0），合并后也应该是掩码区域（0）
            pred_seg = np.minimum(pred_seg, sam_segs[:, :, i])  # 取最小值：0（掩码）优先
        pred_seg = pred_seg.astype(np.uint8)

        # send pred_seg and gt_mask to GPU (使用正确的 device)
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        device = next(model_engine.parameters()).device
        pred_seg = pred_seg.to(device=device)
        gt_mask = gt_mask.to(device=device)

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # ✅ 修正：黑色部分（值为0）是掩码区域，所以使用类别0而不是类别1
    ciou = iou_class[0]  # 类别0是掩码区域（前景）
    giou = acc_iou_meter.avg[0]  # 类别0是掩码区域（前景）

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        # 记录到 SwanLab
        if swanlab_logger is not None:
            swanlab.log({
                "val/giou_topIoU": giou,
                "val/ciou_topIoU": ciou,
            }, step=epoch)
        
    return giou, ciou

if __name__ == "__main__":
    main(sys.argv[1:])
