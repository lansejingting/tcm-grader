"""
把 data/Dataset_1_Cleaned（拉丁名文件夹, 用户自备原始数据集）复制到
grading/data/NB-TCM-CHM（中文名文件夹）。

映射规则基于 rules.json 中的 name_latin → name_cn。
文件名保持不变，只换文件夹。
"""

import json
import shutil
from pathlib import Path

# ============ 路径 ============
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "Dataset_1_Cleaned"      # 原始数据集（自备）
DST = REPO_ROOT / "grading" / "data" / "NB-TCM-CHM"
RULES = REPO_ROOT / "grading" / "data" / "rules.json"

# ============ 从 rules.json 构建映射 ============
rules = json.load(open(RULES, "r", encoding="utf-8"))

# 拉丁名（空格）→ 源文件夹名（下划线）→ 中文名
latin_to_folder = {}
for h in rules["herbs"]:
    latin_norm = h["name_latin"].replace(" ", "_")
    latin_to_folder[latin_norm] = h["name_cn"]

print("=== 拉丁名 → 中文名映射 ===")
for latin, cn in latin_to_folder.items():
    print(f"  {latin:40s} → {cn}")

# ============ 遍历源目录复制 ============
print(f"\n=== 复制结果 ===")
total_copied = 0
total_skipped = 0
not_in_target = []

for src_folder in sorted(SRC.iterdir()):
    if not src_folder.is_dir():
        continue
    folder_name = src_folder.name

    # 匹配目标中文名
    cn_name = latin_to_folder.get(folder_name)
    if not cn_name:
        # 尝试模糊匹配（源数据里有个 Trichosanthis_Pericarpoium 对应 rules 里的 Trichosanthis_Fructus=瓜蒌）
        matched = None
        for latin_norm, cn in latin_to_folder.items():
            if folder_name.startswith(latin_norm.split("_")[0]):
                matched = cn
                break
        if not matched:
            not_in_target.append(folder_name)
            print(f"  ⚠ 跳过（不在目标20种中）: {folder_name}")
            total_skipped += 1
            continue
        cn_name = matched

    dst_folder = DST / cn_name
    # 清空占位的 .gitkeep
    if dst_folder.exists():
        for f in dst_folder.iterdir():
            if f.name == ".gitkeep":
                f.unlink()
    dst_folder.mkdir(parents=True, exist_ok=True)

    # 复制图片
    imgs = [p for p in src_folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    for img in imgs:
        # 避免重名冲突（同一中文名可能来自多个源文件夹）
        dst_path = dst_folder / img.name
        if dst_path.exists():
            # 加前缀避免覆盖
            dst_path = dst_folder / f"{folder_name}_{img.name}"
        shutil.copy2(img, dst_path)

    print(f"  ✅ {folder_name:40s} → {cn_name}  ({len(imgs)} 张)")
    total_copied += len(imgs)

print(f"\n=== 汇总 ===")
print(f"复制成功：{total_copied} 张")
print(f"跳过：{total_skipped} 个文件夹（不在目标20种中）")
if not_in_target:
    print(f"未匹配文件夹：{not_in_target}")

# 最终统计
print(f"\n=== 目标目录最终统计 ===")
for d in sorted(DST.iterdir()):
    if d.is_dir() and not d.name.startswith("."):
        cnt = len([f for f in d.iterdir() if f.is_file()])
        print(f"  {d.name}: {cnt} 张")
