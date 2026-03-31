
import os
import json
import glob

# 1. 统计 PNG 图片数量
vis_path = "/opt/data/private/LLMSeg/vis_output_hard/hard"  #"/opt/data/private/LLMSeg/vis_output2/easy"
png_count = len(glob.glob(os.path.join(vis_path, "*.png")))

# 2. 统计 JSON 中的 sample 数量
json_path = "/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/open_vocab_grasp_hard.json"
sample_count = 0

if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 统计 samples 列表的长度
            if isinstance(data, dict) and "samples" in data:
                sample_count = len(data["samples"])
            elif isinstance(data, list):
                sample_count = len(data)
    except Exception as e:
        print(f"读取 JSON 出错: {e}")
else:
    print(f"JSON 文件不存在: {json_path}")

print("-" * 50)
print(f"🚀 图片统计结果:")
print(f"  目录: {vis_path}")
print(f"  PNG 数量: {png_count}")
print("-" * 50)
print(f"📊 数据集统计结果:")
print(f"  文件: {json_path}")
print(f"  Sample 数量: {sample_count}")
print("-" * 50)
