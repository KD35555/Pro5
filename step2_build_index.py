import os
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# ==========================================
# 核心：定义一个处理“一批”图片的函数
# ==========================================
def process_batch(image_paths):
    # 每个进程独立加载一次模型
    # 注意：这里我们不需要 try-except 包裹模型加载，
    # 如果模型文件坏了，直接报错让我们知道反而更好。
    if not os.path.exists("vit-dinov2-base.npz"):
        return [], []
        
    weights = np.load("vit-dinov2-base.npz")
    vit = Dinov2Numpy(weights)

    batch_features = []
    batch_paths = []

    for path in image_paths:
        try:
            # 1. 快速检查文件大小，跳过损坏的小文件 (<1KB)
            if os.path.getsize(path) < 1024: 
                continue

            # 2. 预处理
            input_tensor = resize_short_side(path)
            
            # 3. 模型推理
            feature = vit(input_tensor)
            
            # 4. 收集结果
            batch_features.append(feature)
            batch_paths.append(path)
        except:
            # 遇到任何坏图直接跳过，不报错
            continue
            
    return batch_features, batch_paths

# ==========================================
# 主程序
# ==========================================
def build_index_fast():
    # --- 1. 智能路径选择逻辑 (核心修改) ---
    # 优先寻找完整图库 'gallery'，如果没有，则退化为 'demo_data' 模式
    target_folder = ""
    
    if os.path.exists('gallery'):
        target_folder = 'gallery'
        print("=" * 50)
        print("✅ 检测到完整图库 'gallery'。")
        print("🚀 正在启动完整构建模式 (Full Mode)...")
        print("⏳ 提示: 处理 10k+ 图片约需 20-25 分钟 (CPU)，请耐心等待。")
        print("=" * 50)
    elif os.path.exists('demo_data'):
        target_folder = 'demo_data'
        print("=" * 50)
        print("⚠️ 未找到 'gallery'，但检测到 'demo_data'。")
        print("🚀 正在启动快速演示模式 (Demo Mode)...")
        print("⚡ 提示: 仅处理少量图片，预计耗时 < 5秒。")
        print("=" * 50)
    else:
        print("❌ 错误: 未找到 'gallery' 或 'demo_data' 文件夹。")
        print("请先运行 step1_download.py 下载数据，或确保 demo_data 存在。")
        return

    # --- 2. 扫描图片 ---
    print(f"正在扫描 {target_folder} 中的图片文件...")
    # 兼容 jpg 和 png
    all_paths = sorted(glob(os.path.join(target_folder, "*.jpg")) + glob(os.path.join(target_folder, "*.png")))
    total_imgs = len(all_paths)
    
    if total_imgs == 0:
        print(f"❌ {target_folder} 中没有找到图片！")
        return

    print(f"找到 {total_imgs} 张图片。")

    # --- 3. 准备多进程 ---
    # 将图片分成很多小批次 (每批 100 张)
    batch_size = 100
    chunks = [all_paths[i:i + batch_size] for i in range(0, total_imgs, batch_size)]

    all_features = []
    valid_paths = []

    # 强制设置为 4 个进程，防止电脑卡死
    num_processes = 4 
    print(f"🚀 已启动 {num_processes} 个进程并发计算...")

    # 使用 if __name__ 保护是 Windows 下多进程的硬性要求
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度条
        for features, paths in tqdm(pool.imap(process_batch, chunks), total=len(chunks), unit="batch"):
            if len(features) > 0:
                all_features.extend(features)
                valid_paths.extend(paths)

    # --- 4. 整合保存 ---
    print("\n正在整合数据并保存索引...")
    if len(all_features) > 0:
        final_features = np.concatenate(all_features, axis=0)
        final_paths = np.array(valid_paths)
        
        # 保存为 .npy 文件
        np.save("index_features.npy", final_features)
        np.save("index_paths.npy", final_paths)
        
        print("-" * 30)
        print(f"✅ 索引构建成功！(Index Built Successfully)")
        print(f"📂 来源文件夹: {target_folder}")
        print(f"📊 成功处理: {len(final_paths)} / {total_imgs} 张图片")
        print(f"💾 特征矩阵: {final_features.shape}")
        print("-" * 30)
    else:
        print("❌ 失败：没有生成任何特征，可能是图片全部损坏。")

if __name__ == "__main__":
    # Windows系统下必须把执行代码放在 if __name__ == "__main__": 之下
    # 否则多进程会报错
    try:
        build_index_fast()
    except KeyboardInterrupt:
        print("\n⛔ 用户强制停止任务。")