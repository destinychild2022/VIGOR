#!/usr/bin/env python3
"""
验证SAM checkpoint文件是否可用（CPU模式，不占用显存）
检查 /opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point2 目录下的权重
"""

import sys
import torch
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入SAM相关模块
from model.segment_anything import sam_model_registry

def validate_checkpoint(checkpoint_path, checkpoint_name, sam_checkpoint):
    """验证单个checkpoint文件"""
    print(f"\n{'='*60}")
    print(f"验证checkpoint: {checkpoint_name}")
    print(f"路径: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        # 1. 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"❌ 文件不存在: {checkpoint_path}")
            return False
        
        # 2. 加载checkpoint（使用CPU）
        print("1. 加载checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint加载成功")
        print(f"   包含键: {list(checkpoint.keys())}")
        
        # 检查必要的信息
        required_keys = ['model_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            print(f"❌ 缺少必要键: {missing_keys}")
            return False
        
        print(f"   Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"   验证损失: {checkpoint['val_loss']:.4f}")
        if 'train_loss' in checkpoint:
            print(f"   训练损失: {checkpoint['train_loss']:.4f}")
        
        # 3. 加载原始SAM模型（在CPU上）
        print("\n2. 加载原始SAM模型...")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.eval()
        sam.to('cpu')
        print(f"✅ SAM模型加载成功")
        
        # 4. 检查checkpoint中的权重结构
        print("\n3. 检查权重结构...")
        model_state_dict = checkpoint['model_state_dict']
        print(f"   模型权重数量: {len(model_state_dict)}")
        
        # 检查是否有module.前缀（分布式训练的痕迹）
        keys_with_module = [k for k in model_state_dict.keys() if k.startswith('module.')]
        if keys_with_module:
            print(f"   ⚠️  发现 {len(keys_with_module)} 个带module.前缀的键（可能来自分布式训练）")
        else:
            print(f"   ✅ 没有module.前缀，权重格式正确")
        
        # 检查LoRA权重
        lora_keys = [k for k in model_state_dict.keys() if 'lora' in k.lower()]
        if lora_keys:
            print(f"   ✅ 发现 {len(lora_keys)} 个LoRA权重（使用了LoRA训练）")
        else:
            print(f"   ℹ️  没有发现LoRA权重（可能使用全量微调）")
        
        # 5. 尝试加载权重到模型
        print("\n4. 尝试加载权重到模型...")
        
        # 检查权重键的匹配情况
        model_keys = set(sam.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        
        missing_in_checkpoint = model_keys - checkpoint_keys
        extra_in_checkpoint = checkpoint_keys - model_keys
        
        print(f"   模型权重总数: {len(model_keys)}")
        print(f"   Checkpoint权重总数: {len(checkpoint_keys)}")
        print(f"   缺失的权重: {len(missing_in_checkpoint)}")
        print(f"   多余的权重: {len(extra_in_checkpoint)}")
        
        if missing_in_checkpoint:
            print(f"   缺失的权重示例: {list(missing_in_checkpoint)[:3]}")
        
        if extra_in_checkpoint:
            print(f"   多余的权重示例: {list(extra_in_checkpoint)[:3]}")
        
        # 尝试加载权重
        try:
            load_result = sam.load_state_dict(model_state_dict, strict=False)
            print(f"✅ 权重加载成功")
            print(f"   缺失的键: {len(load_result.missing_keys)}")
            print(f"   意外的键: {len(load_result.unexpected_keys)}")
            
            # 检查关键组件是否加载成功
            critical_components = ['image_encoder', 'prompt_encoder', 'mask_decoder']
            for component in critical_components:
                component_keys = [k for k in model_state_dict.keys() if component in k]
                if component_keys:
                    loaded_component_keys = [k for k in load_result.unexpected_keys if component not in k and k in model_state_dict]
                    print(f"   {component}: {len(component_keys)} 个权重，加载了 {len(component_keys) - len([k for k in load_result.unexpected_keys if component in k])} 个")
            
        except Exception as load_error:
            print(f"❌ 权重加载失败: {load_error}")
            return False
        
        # 6. 测试模型前向传播（在CPU上）
        print("\n5. 测试模型前向传播...")
        try:
            with torch.no_grad():
                # 创建测试输入
                batched_input = [{
                    'image': torch.randn(3, 1024, 1024),
                    'original_size': (720, 1280),
                    'point_coords': torch.tensor([[[640, 360]]], dtype=torch.float32),
                    'point_labels': torch.tensor([[1]], dtype=torch.int),
                }]
                
                outputs = sam(batched_input, multimask_output=False)
            
            print(f"✅ 模型前向传播成功")
            print(f"   输出keys: {list(outputs[0].keys())}")
            print(f"   masks形状: {outputs[0]['masks'].shape}")
            print(f"   scores形状: {outputs[0]['iou_predictions'].shape}")
            
        except Exception as forward_error:
            print(f"❌ 模型前向传播失败: {forward_error}")
            return False
        
        # 7. 检查LoRA权重（如果存在）
        if 'lora_state_dict' in checkpoint:
            print("\n6. 检查LoRA权重...")
            lora_state_dict = checkpoint['lora_state_dict']
            print(f"   ✅ LoRA权重存在，包含 {len(lora_state_dict)} 个参数")
            
            # 检查LoRA权重的形状
            for key, tensor in list(lora_state_dict.items())[:5]:  # 只显示前5个
                print(f"   {key}: {tensor.shape}")
        else:
            print("\n6. LoRA权重: 未保存（可能未使用LoRA或保存配置不同）")
        
        print(f"\n✅ Checkpoint '{checkpoint_name}' 验证通过，可以正常使用")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 设置路径
    checkpoint_dir = '/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point2'
    sam_checkpoint = '/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth'
    
    print("=== SAM Checkpoint验证工具 ===")
    print(f"检查目录: {checkpoint_dir}")
    print(f"SAM原始权重: {sam_checkpoint}")
    print("验证模式: CPU（不占用显存）")
    
    # 检查原始SAM权重是否存在
    if not os.path.exists(sam_checkpoint):
        print(f"❌ 原始SAM权重不存在: {sam_checkpoint}")
        return
    
    # 列出要验证的checkpoint文件
    checkpoint_files = [
        ('best_model.pth', '最佳模型'),
        ('checkpoint_epoch_56.pth', '最新epoch(56)'),
        ('checkpoint_epoch_55.pth', 'epoch(55)'),
    ]
    
    # 验证结果统计
    results = {}
    
    # 逐个验证checkpoint
    for filename, description in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        success = validate_checkpoint(checkpoint_path, description, sam_checkpoint)
        results[filename] = success
    
    # 输出总结
    print(f"\n{'='*60}")
    print("验证结果总结:")
    print(f"{'='*60}")
    
    total_count = len(results)
    success_count = sum(results.values())
    
    for filename, success in results.items():
        status = "✅ 可用" if success else "❌ 不可用"
        print(f"  {filename}: {status}")
    
    print(f"\n总计: {success_count}/{total_count} 个checkpoint可用")
    
    if success_count == total_count:
        print("🎉 所有checkpoint都可以正常使用！")
    else:
        print("⚠️  部分checkpoint存在问题，建议检查训练过程")

if __name__ == "__main__":
    main()
