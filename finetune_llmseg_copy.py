import argparse
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

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn, ValDataSet_ReasonSeg, collate_fn_new, ValDataSet_LLMSeg
from utils.llm_seg_dataset import LLMSegDataset
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
    parser.add_argument("--auto_resume", action="store_true", default=True)
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
    return parser.parse_args(args)


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
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
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
        }
        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.half

        model = LISAForCausalLM.from_pretrained(
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
        )
    
    # 恢复 transformers 日志级别（可选，如果需要看到其他警告可以设置为 WARNING）
    # transformers_logging.set_verbosity_warning()

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()  # CLIP
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)  # SAM and others

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

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

    # 使用用户自定义的数据集路径
    dataset_base_dir = "/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/dataset"
    sam_masks_base_dir = "/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/test_output_point"
    
    # ✅ 直接读取原始 annotations.json 文件（三个数据集）
    import json
    
    # 准备多个 JSON 文件路径列表
    json_paths = []
    mask_dirs = {}
    dataset_configs = []
    
    # 图像目录（三个子数据集）
    robot_arm_01_image_dir = os.path.join(dataset_base_dir, "picture", "robot_arm_01")
    robot_arm_02_image_dir = os.path.join(dataset_base_dir, "picture", "robot_arm_02")
    robot_arm_03_image_dir = os.path.join(dataset_base_dir, "picture", "robot_arm_03")
    
    for subdir in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
        annotations_path = os.path.join(dataset_base_dir, 'mask', subdir, 'annotations.json')
        if os.path.exists(annotations_path):
            json_paths.append(annotations_path)
            
            # 保存 mask 目录
            mask_dirs[subdir] = os.path.join(dataset_base_dir, 'mask', subdir, 'masks')
            
            # 准备数据集配置（参考 finetune_sam_lora_point.py）
            if subdir == 'robot_arm_01':
                image_dir = robot_arm_01_image_dir
            elif subdir == 'robot_arm_02':
                image_dir = robot_arm_02_image_dir
            else:  # robot_arm_03
                image_dir = robot_arm_03_image_dir
            
            dataset_configs.append({
                'name': subdir,
                'images_dir': image_dir,
                'masks_dir': mask_dirs[subdir],
            })
    
    # 使用 PNG Reader 读取候选掩码（三个子数据集）
    robot_arm_01_sam_mask_helper = SAM_Mask_Reader_PNG(
        os.path.join(sam_masks_base_dir, "robot_arm_01")
    )
    robot_arm_02_sam_mask_helper = SAM_Mask_Reader_PNG(
        os.path.join(sam_masks_base_dir, "robot_arm_02")
    )
    robot_arm_03_sam_mask_helper = SAM_Mask_Reader_PNG(
        os.path.join(sam_masks_base_dir, "robot_arm_03")
    )
    
    # ✅ 直接传入多个 JSON 文件路径，数据集类内部会处理数据划分
    train_dataset = LLMSegDataset(
        json_paths=json_paths,  # 传入多个 JSON 文件路径列表
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        precision=args.precision,
        image_size=args.image_size,
        egoobjects_sam_mask_helper=robot_arm_02_sam_mask_helper,
        coco_sam_mask_helper=robot_arm_01_sam_mask_helper,
        robot_arm_03_sam_mask_helper=robot_arm_03_sam_mask_helper,
        coco_image_dir=robot_arm_01_image_dir,
        ego_objects_image_dir=robot_arm_02_image_dir,
        robot_arm_03_image_dir=robot_arm_03_image_dir,
        mask_dirs=mask_dirs,  # 传入 mask 目录字典
        is_train=True,  # 训练集
        dataset_configs=dataset_configs,  # 传入数据集配置
    )
        
    print(f"训练集加载完成: {len(train_dataset)} 个样本")

    return train_dataset


def init_validation_dataset(args, tokenizer):
    if args.no_eval:
        return None

    # 使用用户自定义的数据集路径
    dataset_base_dir = "/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/dataset"
    sam_masks_base_dir = "/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/test_output_point"
    
    # ✅ 直接读取原始 annotations.json 文件（三个数据集）
    import json
    
    # 准备多个 JSON 文件路径列表（与训练集相同）
    json_paths = []
    mask_dirs = {}
    dataset_configs = []
    
    # 图像目录（三个子数据集）
    robot_arm_01_image_dir = os.path.join(dataset_base_dir, "picture", "robot_arm_01")
    robot_arm_02_image_dir = os.path.join(dataset_base_dir, "picture", "robot_arm_02")
    robot_arm_03_image_dir = os.path.join(dataset_base_dir, "picture", "robot_arm_03")
    
    for subdir in ['robot_arm_01', 'robot_arm_02', 'robot_arm_03']:
        annotations_path = os.path.join(dataset_base_dir, 'mask', subdir, 'annotations.json')
        if os.path.exists(annotations_path):
            json_paths.append(annotations_path)
            
            # 保存 mask 目录
            mask_dirs[subdir] = os.path.join(dataset_base_dir, 'mask', subdir, 'masks')
            
            # 准备数据集配置（参考 finetune_sam_lora_point.py）
            if subdir == 'robot_arm_01':
                image_dir = robot_arm_01_image_dir
            elif subdir == 'robot_arm_02':
                image_dir = robot_arm_02_image_dir
            else:  # robot_arm_03
                image_dir = robot_arm_03_image_dir
            
            dataset_configs.append({
                'name': subdir,
                'images_dir': image_dir,
                'masks_dir': mask_dirs[subdir],
            })

    robot_arm_01_sam_mask_helper = SAM_Mask_Reader_PNG(
        os.path.join(sam_masks_base_dir, "robot_arm_01")
    )
    robot_arm_02_sam_mask_helper = SAM_Mask_Reader_PNG(
        os.path.join(sam_masks_base_dir, "robot_arm_02")
    )
    robot_arm_03_sam_mask_helper = SAM_Mask_Reader_PNG(
        os.path.join(sam_masks_base_dir, "robot_arm_03")
    )
    
    # ✅ 直接传入多个 JSON 文件路径，数据集类内部会根据 is_train=False 自动划分验证集
    from utils.llm_seg_dataset import LLMSegDataset
    val_dataset = LLMSegDataset(
        json_paths=json_paths,  # 传入多个 JSON 文件路径列表
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        precision=args.precision,
        image_size=args.image_size,
        egoobjects_sam_mask_helper=robot_arm_02_sam_mask_helper,
        coco_sam_mask_helper=robot_arm_01_sam_mask_helper,
        robot_arm_03_sam_mask_helper=robot_arm_03_sam_mask_helper,
        coco_image_dir=robot_arm_01_image_dir,
        ego_objects_image_dir=robot_arm_02_image_dir,
        robot_arm_03_image_dir=robot_arm_03_image_dir,
        mask_dirs=mask_dirs,
        is_train=False,  # 验证集
        dataset_configs=dataset_configs,  # 传入数据集配置
    )
    
    print(f"验证集加载完成: {len(val_dataset)} 个样本")

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

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume, load_optimizer_states=False, load_lr_scheduler_states=False)
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

            
            # save checkpoint
            if args.no_eval or is_best:
                # pass
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                        {"epoch": epoch},
                        os.path.join(
                            args.log_dir,
                            "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                                best_score, cur_ciou
                            ),
                        ),
                    )
                    # ✅ 修复：使用更安全的方式删除checkpoint目录
                    # 问题：DeepSpeed会在checkpoint目录下创建global_step*子目录，直接删除可能失败
                    # 解决：先等待所有进程同步，然后重命名旧checkpoint为备份，最后删除
                    if os.path.exists(save_dir):
                        try:
                            # 方案1：重命名旧checkpoint为备份（更安全，不会丢失数据）
                            backup_dir = save_dir + f"_backup_epoch{epoch}"
                            if os.path.exists(backup_dir):
                                shutil.rmtree(backup_dir, ignore_errors=True)
                            os.rename(save_dir, backup_dir)
                            # 异步删除备份目录（不阻塞训练）
                            import threading
                            def delete_backup():
                                try:
                                    shutil.rmtree(backup_dir, ignore_errors=True)
                                except:
                                    pass
                            threading.Thread(target=delete_backup, daemon=True).start()
                        except Exception as e:
                            # 如果重命名失败，尝试直接删除（使用ignore_errors确保不会崩溃）
                            print(f"  [警告] 重命名checkpoint目录失败: {e}，尝试直接删除")
                            try:
                                shutil.rmtree(save_dir, ignore_errors=True)
                            except Exception as e2:
                                print(f"  [警告] 删除checkpoint目录也失败: {e2}，DeepSpeed将覆盖保存")
                # ✅ 关键：等待所有进程同步，确保删除操作完成后再保存
                torch.distributed.barrier()
                # DeepSpeed会自动创建新的checkpoint目录和global_step*子目录
                model_engine.save_checkpoint(save_dir)
                if args.local_rank == 0:
                    print("checkpoint saved in {}".format(save_dir))
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
    
    # ✅ 随机选择4张图片进行可视化（训练时）
    max_vis_samples = 4
    try:
        total_samples = len(train_loader.dataset)
    except:
        total_samples = 1000  # 如果无法获取，使用一个较大的默认值
    
    # ✅ 在分布式训练中，需要考虑每个GPU只处理部分样本
    if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'num_replicas'):
        num_replicas = train_loader.sampler.num_replicas
        rank = train_loader.sampler.rank
        samples_per_gpu = total_samples // num_replicas
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank < num_replicas - 1 else total_samples
        local_total = end_idx - start_idx
        if local_total > max_vis_samples:
            local_selected = random.sample(range(local_total), max_vis_samples)
            selected_indices = set(start_idx + idx for idx in local_selected)
        else:
            selected_indices = set(range(start_idx, end_idx))
    else:
        if total_samples > max_vis_samples:
            selected_indices = set(random.sample(range(total_samples), max_vis_samples))
        else:
            selected_indices = set(range(total_samples))
    
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
            
            input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

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
            
            # 训练可视化：随机选择4张图片进行可视化（GT掩码和预测掩码拼接）
            batch_size = input_dict["images"].size(0)
            for batch_sample_idx in range(batch_size):
                # 计算全局索引（分布式训练时需要）
                if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'num_replicas'):
                    num_replicas = train_loader.sampler.num_replicas
                    rank = train_loader.sampler.rank
                    samples_per_gpu = len(train_loader.dataset) // num_replicas
                    global_idx = rank * samples_per_gpu + sample_idx + batch_sample_idx
                else:
                    global_idx = sample_idx + batch_sample_idx
                
                # 判断是否需要可视化（只在主GPU上，且是选中的样本）
                should_visualize = (args.local_rank == 0 and global_idx in selected_indices and vis_count < max_vis_samples)
                if should_visualize:
                    try:
                        with torch.no_grad():
                            # 获取预测相似度并选择最佳候选mask
                            pred_similarity = output_dict["pred_similarity"][batch_sample_idx]
                            max_idx = torch.argmax(pred_similarity).item()
                            
                            # 读取图像和mask
                            image_path = input_dict['image_paths'][batch_sample_idx]
                            image = cv2.imread(image_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            sam_segs = input_dict["origin_segs_list"][batch_sample_idx]
                            gt_mask = output_dict["gt_masks"][batch_sample_idx]
                            
                            if sam_segs is not None and gt_mask is not None:
                                # 获取预测的mask（选择相似度最高的候选mask）
                                pred_seg = sam_segs[:, :, max_idx]  # (H, W)
                                pred_seg = torch.from_numpy(pred_seg).unsqueeze(0)  # (1, H, W)
                                pred_seg = pred_seg.cuda()
                                
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
                                
                                # 拼接图像
                                combined_image = np.hstack([gt_overlay, pred_overlay])
                                    
                                # 从对话中提取物体名称
                                object_name = "object"
                                conversation = input_dict['conversation_list'][batch_sample_idx]
                                if "segment the" in conversation.lower():
                                    parts = conversation.lower().split("segment the")
                                    if len(parts) > 1:
                                        object_name = parts[1].split()[0].strip() or "object"
                                
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
                                
                                # 打印IoU相关信息
                                K = gt_ious_round.shape[0]
                                print(f"\n  候选mask数量 (K): {K}")
                                print(f"\n  1. 原始 gt_ious (IoU值):")
                                for i in range(K):
                                    print(f"     mask[{i:2d}]: {gt_ious_round[i, 0].item():.6f}")
                                print(f"\n  2. gt_ious 经过 softmax 后的分布 (gt_dis):")
                                for i in range(K):
                                    print(f"     mask[{i:2d}]: {gt_dis[i, 0].item():.10f}")
                                print(f"\n  3. 原始 pred_similarity (相似度分数):")
                                for i in range(K):
                                    print(f"     mask[{i:2d}]: {sim_scores[i, 0].item():.6f}")
                                print(f"\n  4. pred_similarity 经过 softmax 后的分布 (sim_dis):")
                                for i in range(K):
                                    print(f"     mask[{i:2d}]: {sim_dis[i, 0].item():.10f}")
                                
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
                                
                                vis_count += 1
                    except Exception as e:
                        print(f"  [训练可视化] 保存图片时出错: {e}")
            
            # 更新样本索引（考虑batch_size）
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

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
                output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        pred_seg = sam_segs[:, :, max_idx] # (H, W)
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        # send pred_seg and gt_mask to GPU
        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

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
                    
                    # 提取物体名称（从conversation中）
                    object_name = "object"
                    if 'conversation_list' in input_dict and len(input_dict['conversation_list']) > 0:
                        conversation = input_dict['conversation_list'][0]
                        # 尝试从 "segment the {object_name}" 中提取
                        if "segment the" in conversation.lower():
                            parts = conversation.lower().split("segment the")
                            if len(parts) > 1:
                                obj_part = parts[1].split()[0] if parts[1].strip() else "object"
                                object_name = obj_part.strip()
                    
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

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
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

        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

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

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
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

        # send pred_seg and gt_mask to GPU
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

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

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
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

        # send pred_seg and gt_mask to GPU
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

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
