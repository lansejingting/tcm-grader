# -*- coding: utf-8 -*-
"""
中药材 20 类分类器训练脚本
- 数据: grading/data/NB-TCM-CHM (20 类, ~3300 张 jpg)
- 模型: torchvision 预训练 ResNet18 (迁移学习, 替换最后一层 fc 为 20 类)
- 数据划分: 按类别分层抽样 85% 训练 / 15% 验证 (固定随机种子, 划分结果落盘可复现)
- 输出: outputs/ 下保存最优模型、类别映射、指标 (accuracy / per-class P,R,F1 / 混淆矩阵)
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ----------------------------- 配置 -----------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "grading" / "data" / "NB-TCM-CHM"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

IMG_SIZE = 224
VAL_RATIO = 0.15
SEED = 42
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------- 数据 -----------------------------
def scan_dataset(data_dir: Path):
    """扫描数据集, 返回 (类别列表, [(路径, 类别索引), ...])"""
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    name2idx = {n: i for i, n in enumerate(class_names)}
    samples = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), name2idx[d.name]))
    return class_names, samples


def stratified_split(samples, val_ratio: float, seed: int):
    """按类别分层抽样, 返回 (train_samples, val_samples)"""
    rng = random.Random(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val = [], []
    for label in sorted(by_class):
        items = sorted(by_class[label])  # 先排序保证可复现
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class TCMImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------- 指标 -----------------------------
def compute_metrics(y_true, y_pred, n_classes):
    """手算混淆矩阵与各类 Precision/Recall/F1"""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    f1s, supports = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[i] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }
        f1s.append(f1)
        supports.append(cm[i, :].sum())

    support_arr = np.array(supports, dtype=np.float64)
    f1_arr = np.array(f1s)
    metrics = {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_f1": round(float(f1_arr.mean()), 4),
        "weighted_f1": round(float((f1_arr * support_arr).sum() / support_arr.sum()), 4),
    }
    return metrics


def render_confusion_matrix(cm, class_names, out_path: Path):
    """matplotlib 混淆矩阵热力图 (支持中文)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("验证集混淆矩阵")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- 训练 -----------------------------
def build_model(n_classes: int, device):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_top2 = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        top2 = logits.topk(2, dim=1).indices.cpu()
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_top2.extend(top2.tolist())

    correct1 = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    correct2 = sum(int(t in p) for p, t in zip(all_top2, all_labels))
    n = len(all_labels)
    return all_labels, all_preds, correct1 / n, correct2 / n


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    class_names, samples = scan_dataset(DATA_DIR)
    print(f"数据集: {DATA_DIR}")
    print(f"类别数: {len(class_names)}, 图片总数: {len(samples)}")
    dist = Counter(lbl for _, lbl in samples)
    for i, name in enumerate(class_names):
        print(f"  [{i:2d}] {name}: {dist[i]}")

    train_samples, val_samples = stratified_split(samples, VAL_RATIO, SEED)
    print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

    # 划分结果落盘, 保证可复现 / 可分析
    split_info = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "classes": class_names,
        "train": [p for p, _ in train_samples],
        "val": [p for p, _ in val_samples],
    }
    (OUT_DIR / "split.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"划分结果已保存: {OUT_DIR / 'split.json'}")

    # ImageNet 归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = TCMImageDataset(train_samples, train_tf)
    val_ds = TCMImageDataset(val_samples, val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
                            persistent_workers=NUM_WORKERS > 0)

    model = build_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_path = OUT_DIR / "best_model.pth"
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("\n开始训练...")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_total += labels.size(0)
        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        _, _, val_acc1, val_acc2 = evaluate(model, val_loader, device)

        marker = ""
        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save({
                "model_state": model.state_dict(),
                "classes": class_names,
                "img_size": IMG_SIZE,
                "val_acc": val_acc1,
            }, best_path)
            marker = "  <- 保存最优"
        log(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_acc={val_acc1:.4f} val_top2={val_acc2:.4f}{marker}")

    log(f"\n训练完成, 总耗时 {time.time() - t0:.1f}s, 最优 val_acc={best_acc:.4f}")

    # ----------------------------- 最终评估 -----------------------------
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    y_true, y_pred, acc1, acc2 = evaluate(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred, len(class_names))
    metrics["overall_accuracy"] = round(acc1, 4)
    metrics["top2_accuracy"] = round(acc2, 4)
    metrics["num_val"] = len(y_true)

    log(f"\n最终评估 (最优模型):")
    log(f"  Top-1 准确率: {acc1:.4f} ({round(acc1 * len(y_true))}/{len(y_true)})")
    log(f"  Top-2 准确率: {acc2:.4f}")
    log(f"  Macro F1:  {metrics['macro_f1']}")
    log(f"  Weighted F1: {metrics['weighted_f1']}")
    log("\n各类别指标:")
    log(f"  {'类别':<8} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>8}")
    for i, name in enumerate(class_names):
        m = metrics["per_class"][i]
        log(f"  {name:<8} {m['precision']:>9.4f} {m['recall']:>9.4f} {m['f1']:>9.4f} {m['support']:>8d}")

    # F1 较低的类别提示 (易混淆)
    weak = sorted(range(len(class_names)), key=lambda i: metrics["per_class"][i]["f1"])[:5]
    log("\nF1 最低的 5 个类别 (易混淆, 重点关注):")
    for i in weak:
        m = metrics["per_class"][i]
        # 找出该类最容易被误判成谁
        cm = np.array(metrics["confusion_matrix"])
        row = cm[i].copy()
        row[i] = 0
        confuse_with = class_names[row.argmax()] if row.sum() > 0 else "-"
        log(f"  {class_names[i]}: f1={m['f1']:.4f}, 最常误判为: {confuse_with}")

    (OUT_DIR / "metrics.json").write_text(
        json.dumps({**metrics, "classes": class_names}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    render_confusion_matrix(np.array(metrics["confusion_matrix"]), class_names,
                            OUT_DIR / "confusion_matrix.png")
    (OUT_DIR / "training_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    (OUT_DIR / "classes.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n输出文件: {best_path}")
    print(f"  metrics.json / confusion_matrix.png / classes.json / split.json")


if __name__ == "__main__":
    main()
