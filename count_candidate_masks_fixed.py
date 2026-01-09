import os
import json
from glob import glob
from collections import defaultdict
import numpy as np

def count_candidate_masks(base_path, output_file):
    """
    统计候选mask数量
    base_path: 基础路径（如 /opt/data/private/LLMSeg/dataset/VIGOR-100K/）
    output_file: 输出结果文件
    """
    
    # 两个待统计的目录
    finetuned_path = os.path.join(base_path, "test_masks_finetuned_fixed_0.8_0.8")
    sam_path = os.path.join(base_path, "sam_masks_0.8_0.8")
    
    print(f"开始统计候选mask数量...")
    print(f"Finetuned路径: {finetuned_path}")
    print(f"SAM路径: {sam_path}")
    
    # 检查目录是否存在
    if not os.path.exists(finetuned_path):
        print(f"错误: Finetuned目录不存在: {finetuned_path}")
        return
    
    if not os.path.exists(sam_path):
        print(f"错误: SAM目录不存在: {sam_path}")
        return
    
    # 统计函数
    def count_masks_in_directory(dir_path, dir_name):
        """统计目录中每个图片的候选mask数量"""
        print(f"\n正在统计 {dir_name}...")
        
        # 查找所有子目录（对应不同的图片）
        subdirs = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        
        subdirs.sort()
        print(f"找到 {len(subdirs)} 个子目录")
        
        total_masks = 0
        image_counts = []
        detailed_counts = []
        
        for subdir in subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            
            # 首先尝试直接在子目录下的masks目录
            masks_dir = os.path.join(subdir_path, "masks")
            mask_files = []
            
            if os.path.exists(masks_dir):
                # 统计masks目录中的图片文件数量
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    mask_files.extend(glob(os.path.join(masks_dir, ext)))
            else:
                # 如果masks目录不存在，尝试7/masks路径
                masks_dir_7 = os.path.join(subdir_path, "7", "masks")
                if os.path.exists(masks_dir_7):
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                        mask_files.extend(glob(os.path.join(masks_dir_7, ext)))
                else:
                    # 最后尝试直接在子目录中的mask文件
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                        mask_files.extend(glob(os.path.join(subdir_path, ext)))
            
            count = len(mask_files)
            total_masks += count
            image_counts.append(count)
            
            detailed_counts.append({
                'image_id': subdir,
                'mask_count': count,
                'mask_files': mask_files
            })
            
            # 显示统计信息
            if len(subdirs) <= 10 or count > 100 or count == 2:  # 显示前10个、有异常的或只有2个的
                if os.path.exists(os.path.join(subdir_path, "masks")):
                    print(f"  图片 {subdir}: {count} 个候选mask (在masks目录中)")
                elif os.path.exists(os.path.join(subdir_path, "7", "masks")):
                    print(f"  图片 {subdir}: {count} 个候选mask (在7/masks目录中)")
                elif count > 0:
                    print(f"  图片 {subdir}: {count} 个候选mask (在子目录根目录中)")
                else:
                    print(f"  图片 {subdir}: {count} 个候选mask (未找到mask文件)")
            else:
                print(f"  图片 {subdir}: {count} 个候选mask")
        
        # 计算统计信息
        if image_counts:
            avg_masks = np.mean(image_counts)
            min_masks = np.min(image_counts)
            max_masks = np.max(image_counts)
            median_masks = np.median(image_counts)
            std_masks = np.std(image_counts)
        else:
            avg_masks = min_masks = max_masks = median_masks = std_masks = 0
        
        # 转换为Python原生类型以避免JSON序列化问题
        stats = {
            'directory': dir_name,
            'total_images': len(subdirs),
            'total_masks': int(total_masks),
            'average_masks_per_image': float(avg_masks),
            'min_masks_per_image': int(min_masks),
            'max_masks_per_image': int(max_masks),
            'median_masks_per_image': float(median_masks),
            'std_masks_per_image': float(std_masks),
            'image_details': detailed_counts
        }
        
        return stats
    
    # 统计两个目录
    finetuned_stats = count_masks_in_directory(finetuned_path, "Finetuned (test_masks_finetuned_fixed_0.8_0.8)")
    sam_stats = count_masks_in_directory(sam_path, "SAM (sam_masks_0.8_0.8)")
    
    # 生成报告
    report = []
    report.append("=" * 80)
    report.append("候选mask数量统计报告")
    report.append("=" * 80)
    report.append("")
    
    # Finetuned统计
    report.append("1. Finetuned候选mask统计 (test_masks_finetuned_fixed_0.8_0.8):")
    report.append(f"   总图片数: {finetuned_stats['total_images']}")
    report.append(f"   总候选mask数: {finetuned_stats['total_masks']}")
    report.append(f"   平均每张图片候选mask数: {finetuned_stats['average_masks_per_image']:.2f}")
    report.append(f"   最小候选mask数: {finetuned_stats['min_masks_per_image']}")
    report.append(f"   最大候选mask数: {finetuned_stats['max_masks_per_image']}")
    report.append(f"   中位数候选mask数: {finetuned_stats['median_masks_per_image']:.2f}")
    report.append(f"   标准差: {finetuned_stats['std_masks_per_image']:.2f}")
    report.append("")
    
    # SAM统计
    report.append("2. SAM候选mask统计 (sam_masks_0.8_0.8):")
    report.append(f"   总图片数: {sam_stats['total_images']}")
    report.append(f"   总候选mask数: {sam_stats['total_masks']}")
    report.append(f"   平均每张图片候选mask数: {sam_stats['average_masks_per_image']:.2f}")
    report.append(f"   最小候选mask数: {sam_stats['min_masks_per_image']}")
    report.append(f"   最大候选mask数: {sam_stats['max_masks_per_image']}")
    report.append(f"   中位数候选mask数: {sam_stats['median_masks_per_image']:.2f}")
    report.append(f"   标准差: {sam_stats['std_masks_per_image']:.2f}")
    report.append("")
    
    # 总计对比
    total_images_all = finetuned_stats['total_images'] + sam_stats['total_images']
    total_masks_all = finetuned_stats['total_masks'] + sam_stats['total_masks']
    
    report.append("3. 总计对比:")
    report.append(f"   两个目录总图片数: {total_images_all}")
    report.append(f"   两个目录总候选mask数: {total_masks_all}")
    if total_images_all > 0:
        report.append(f"   所有图片平均候选mask数: {total_masks_all/total_images_all:.2f}")
    else:
        report.append("   所有图片平均候选mask数: 0")
    report.append("")
    
    # 保存详细统计到JSON文件
    detailed_data = {
        'finetuned_stats': finetuned_stats,
        'sam_stats': sam_stats,
        'summary': {
            'total_images': total_images_all,
            'total_masks': total_masks_all,
            'average_masks_per_image': float(total_masks_all/total_images_all) if total_images_all > 0 else 0
        }
    }
    
    # 保存JSON文件
    json_output_file = output_file.replace('.txt', '_detailed.json')
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n详细统计已保存到: {json_output_file}")
    
    # 写入txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n报告已保存到: {output_file}")
    print("统计完成!")
    
    return detailed_data

def main():
    # 设置路径
    base_path = "/opt/data/private/LLMSeg/dataset/VIGOR-100K/"
    output_file = "/opt/data/private/LLMSeg/candidate_masks_count.txt"
    
    # 执行统计
    count_candidate_masks(base_path, output_file)

if __name__ == "__main__":
    main()
