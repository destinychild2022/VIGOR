#!/usr/bin/env python3

import torch

checkpoint_path = '/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point/checkpoint_epoch_5.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('✅ Checkpoint文件可以正常加载')
    print(f'包含的键: {list(checkpoint.keys())}')
    
    if 'epoch' in checkpoint:
        print(f'Epoch: {checkpoint["epoch"]}')
    
    if 'model_state_dict' in checkpoint:
        print(f'模型参数数量: {len(checkpoint["model_state_dict"])}')
    
    if 'lora_state_dict' in checkpoint:
        print(f'LoRA参数数量: {len(checkpoint["lora_state_dict"])}')
        print(f'LoRA键示例: {list(checkpoint["lora_state_dict"].keys())[:5]}')
    
    if 'val_loss' in checkpoint:
        print(f'验证损失: {checkpoint["val_loss"]}')
    
    if 'train_loss' in checkpoint:
        print(f'训练损失: {checkpoint["train_loss"]}')
        
except Exception as e:
    print(f'❌ Checkpoint加载失败: {e}')
    import traceback
    traceback.print_exc()
