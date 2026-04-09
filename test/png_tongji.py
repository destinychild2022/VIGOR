import pathlib

def count_png_images(directory_path, recursive=True):
    # 将路径转换为 Path 对象
    path = pathlib.Path(directory_path)
    
    if not path.exists():
        print(f"错误: 路径 '{directory_path}' 不存在。")
        return 0
    
    if not path.is_dir():
        print(f"错误: '{directory_path}' 不是一个文件夹。")
        return 0

    # 查找所有 .png 文件 (忽略大小写)
    pattern = "**/*.png" if recursive else "*.png"
    # 使用 rglob 处理递归，或者 glob 处理当前目录
    png_files = list(path.rglob("*.png") if recursive else path.glob("*.png"))
    
    count = len(png_files)
    print(f"在目录 '{directory_path}' 中找到 {count} 个 .png 图片。")
    return count

if __name__ == "__main__":
    # --- 你可以在这里修改目标文件夹路径 ---
    target_dir = r"/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/train/masks_new"  # "./" 表示当前文件夹，也可以输入绝对路径
    
    count_png_images(target_dir, recursive=True)
