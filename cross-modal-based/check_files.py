import os
import pandas as pd

# --- 1. 定义路径 ---
# 请确保这里的路径是正确的
image_dir = "/mnt/dataX/ysc/mycode/AAAI-2026-paper-1/cross-modal-based/dataset/Cross_modal_based_v0_5/new_500/"
excel_file = "/mnt/dataX/ysc/mycode/AAAI-2026-paper-1/cross-modal-based/dataset/Cross_modal_based_v0_5/new_500_label.xlsx"

# --- 2. 从图片文件夹中获取所有ID ---
print(f"正在扫描图片文件夹: {image_dir}")
image_ids = set()
try:
    # 获取目录下所有文件名
    all_files = os.listdir(image_dir)
    # 筛选出图片文件并提取ID
    for filename in all_files:
        # 只处理常见的图片格式，避免处理如.DS_Store等隐藏文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # os.path.splitext可以分离文件名和扩展名，例如 ('pointing_87', '.jpg')
            base_name = os.path.splitext(filename)[0]
            image_ids.add(base_name)
    print(f"在文件夹中找到 {len(image_ids)} 个图片ID。")
except FileNotFoundError:
    print(f"错误: 找不到图片文件夹路径 '{image_dir}'。请检查路径是否正确。")
    exit()  # 如果文件夹不存在，直接退出程序

# --- 3. 从Excel文件中获取所有ID ---
print(f"正在读取Excel文件: {excel_file}")
excel_ids = set()
try:
    # 使用pandas读取Excel文件。header=None表示第一行不是标题
    # usecols=[0] 表示只读取第一列，提高效率
    df = pd.read_excel(excel_file, header=None, usecols=[0])

    # 遍历第一列 (df[0]) 的所有行
    for item in df[0]:
        # pd.notna 检查值是否为非空 (Not a Number)
        if pd.notna(item):
            # 将ID转换为字符串并添加到集合中
            excel_ids.add(str(item))
    print(f"在Excel文件中找到 {len(excel_ids)} 个ID。")
except FileNotFoundError:
    print(f"错误: 找不到Excel文件路径 '{excel_file}'。请检查路径是否正确。")
    exit()  # 如果文件不存在，直接退出程序
except Exception as e:
    print(f"读取Excel文件时发生错误: {e}")
    exit()

# --- 4. 比较两个ID集合并打印结果 ---
print("\n" + "=" * 20 + " 检查结果 " + "=" * 20)

# 4.1 找出两者共有的ID (交集)
common_ids = sorted(list(image_ids.intersection(excel_ids)))

# 4.2 找出只在图片文件夹中存在的ID (差集)
only_in_images = sorted(list(image_ids.difference(excel_ids)))

# 4.3 找出只在Excel中存在的ID (差集)
only_in_excel = sorted(list(excel_ids.difference(image_ids)))

# 打印汇总信息
print(f"\n[汇总信息]")
print(f" - 图片文件夹ID总数: {len(image_ids)}")
print(f" - Excel文件ID总数: {len(excel_ids)}")
print(f" - 两者共有的ID数量: {len(common_ids)}")
print(f" - 仅存在于图片文件夹的ID数量: {len(only_in_images)}")
print(f" - 仅存在于Excel文件的ID数量: {len(only_in_excel)}")

# 打印详细列表
print(f"\n--- 1. 在图片文件夹和Excel中【都存在】的ID ({len(common_ids)}个): ---")
if common_ids:
    for item in common_ids:
        print(item)
else:
    print("无")

print(f"\n--- 2. 【仅在图片文件夹中存在】，Excel中缺失的ID ({len(only_in_images)}个): ---")
if only_in_images:
    for item in only_in_images:
        print(item)
else:
    print("无")

print(f"\n--- 3. 【仅在Excel中存在】，图片文件夹中缺失的ID ({len(only_in_excel)}个): ---")
if only_in_excel:
    for item in only_in_excel:
        print(item)
else:
    print("无")

print("\n" + "=" * 50)
print("检查完成。")