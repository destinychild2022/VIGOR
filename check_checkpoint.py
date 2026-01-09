#!/usr/bin/env python3

import torch

checkpoint_path = '/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point2/checkpoint_epoch_1.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('✅ Checkpoint文件可以正常加载')
    print(f'包含的键: {list(checkpoint.keys())}')
    
    if 'epoch' in checkpoint:
        print(f'Epoch: {checkpoint["epoch"]}')
    
    if 'model_state_dict' in checkpoint:
        print(f'模型参数数量: {len(checkpoint["model_state_dict"])}')
        print(f'模型权重键示例: {list(checkpoint["model_state_dict"].keys())[:10]}')
        
        # 检查是否有module.前缀（分布式训练的标志）
        keys_with_module = [k for k in checkpoint["model_state_dict"].keys() if k.startswith('module.')]
        keys_without_module = [k for k in checkpoint["model_state_dict"].keys() if not k.startswith('module.')]
        print(f'带module.前缀的键: {len(keys_with_module)}')
        print(f'不带module.前缀的键: {len(keys_without_module)}')
        
        # 检查关键组件
        image_encoder_keys = [k for k in checkpoint["model_state_dict"].keys() if 'image_encoder' in k]
        prompt_encoder_keys = [k for k in checkpoint["model_state_dict"].keys() if 'prompt_encoder' in k]
        mask_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if 'mask_decoder' in k]
        
        print(f'Image encoder权重: {len(image_encoder_keys)}')
        print(f'Prompt encoder权重: {len(prompt_encoder_keys)}')
        print(f'Mask decoder权重: {len(mask_decoder_keys)}')
        
        # 检查LoRA权重
        lora_keys = [k for k in checkpoint["model_state_dict"].keys() if 'lora' in k.lower()]
        print(f'LoRA权重在model_state_dict中: {len(lora_keys)}')
        if lora_keys:
            print(f'LoRA权重示例: {lora_keys[:5]}')
    
    if 'lora_state_dict' in checkpoint:
        print(f'独立的LoRA参数数量: {len(checkpoint["lora_state_dict"])}')
        print(f'LoRA键示例: {list(checkpoint["lora_state_dict"].keys())[:5]}')
    else:
        print('❌ 没有独立的lora_state_dict')
    
    if 'val_loss' in checkpoint:
        print(f'验证损失: {checkpoint["val_loss"]}')
    
    if 'train_loss' in checkpoint:
        print(f'训练损失: {checkpoint["train_loss"]}')
        
except Exception as e:
    print(f'❌ Checkpoint加载失败: {e}')
    import traceback
    traceback.print_exc()
