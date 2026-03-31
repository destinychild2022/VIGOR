import json
import os

# 1. 配置路径 (请根据服务器实际路径修改)
JSON_PATH = "/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/open_vocab_grasp_hard.json"
VIS_DIR = "/opt/data/private/LLMSeg/vis_output2/hard"
REPORT_PATH = "/tmp/missing_samples_report.txt"

def find_missing():
    # A. 加载 JSON 数据
    print(f"-> 正在加载数据集: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误: JSON 文件不存在")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        samples = data.get("samples", [])
    
    # B. 解析已有图片，存入集合提高查询速度
    # 键定义为：(img_name, object_clean)
    existing_samples = set()
    if not os.path.exists(VIS_DIR):
        print(f"❌ 错误: 图片目录不存在 {VIS_DIR}")
        return

    print(f"-> 正在扫描输出目录: {VIS_DIR}")
    for filename in os.listdir(VIS_DIR):
        if not filename.endswith(".png") or "_instr" not in filename:
            continue
        
        # 精确解析: 36.png_keys_and_keyways_splines_instr0_iou0.964.png
        try:
            # 1. 先去掉结尾的 _instr... 后缀
            # 得到: 36.png_keys_and_keyways_splines
            prefix = filename.split('_instr')[0]
            
            # 2. 找到第一个下划线作为分界点
            # 图片名在前，物体名在后
            idx = prefix.find('_')
            if idx != -1:
                img_name = prefix[:idx]     # "36.png"
                obj_name = prefix[idx+1:]   # "keys_and_keyways_splines"
                existing_samples.add((img_name, obj_name))
        except Exception:
            continue

    print(f"-> 结果文件夹中包含 {len(existing_samples)} 个独立样本 (已处理带下划线的复杂命名)")

    # C. 进行比对
    missing_list = []
    match_examples = []
    
    for idx, s in enumerate(samples):
        # 1. 提取 JSON 里的图片名
        gt_path = s.get("gt_mask_path", "")
        # 从 "masks/7_part_02..." 提取 "7"，拼成 "7.png"
        img_num = os.path.basename(gt_path).split('_')[0]
        img_name_json = f"{img_num}.png"
        
        # 2. 提取并清理对象名 (与 test_llmseg_vigor.py 的逻辑保持一致)
        obj_raw = s.get("gt_object", s.get("object", "Unknown"))
        obj_clean = obj_raw.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        key = (img_name_json, obj_clean)
        
        # 3. 检查是否存在
        if key in existing_samples:
            if len(match_examples) < 5:
                match_examples.append(f"JSON Key: {key}")
        else:
            missing_list.append({
                "index": idx,
                "img": img_name_json,
                "obj": obj_raw
            })

    # D. 输出报告
    print("\n" + "="*60)
    print("[逻辑验证] 匹配成功的前 5 个样例 (如果这里为空说明逻辑依然有误):")
    for ex in match_examples:
        print(f"  ✅ {ex}")

    print("\n" + "="*60)
    print(f"📊 统计结果:")
    print(f"  数据集总样本: {len(samples)}")
    print(f"  成功匹配样本: {len(samples) - len(missing_list)}")
    print(f"  缺失样本数量: {len(missing_list)}")
    print("="*60)
    
    # 打印前 10 个缺失的样本
    if missing_list:
        print("\n前 10 个缺失样本清单:")
        for m in missing_list[:10]:
            print(f"  Index: {m['index']:<5} | Image: {m['img']:<10} | Object: {m['obj']}")
    
    # 保存详细列表
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"VIGOR 缺失样本详细报告\n缺失总数: {len(missing_list)}\n")
        f.write("-" * 60 + "\n")
        for m in missing_list:
            f.write(f"Index: {m['index']:<5} | Image: {m['img']:<10} | Object: {m['obj']}\n")
    
    print(f"\n✅ 详细名单已保存至: {REPORT_PATH}")

if __name__ == "__main__":
    find_missing()
