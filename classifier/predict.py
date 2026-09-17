# -*- coding: utf-8 -*-
"""
加载训练好的分类器进行预测
用法:
  python predict.py <图片路径或文件夹路径>
输出: Top-1 类别 + 置信度, 以及 Top-3 候选
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

OUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUT_DIR / "best_model.pth"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt["img_size"]


def predict(model, classes, img_size, img_path: Path, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[0], dim=0).cpu()
    top = probs.topk(3)
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    files = sorted([p for p in target.iterdir() if p.suffix.lower() in IMG_EXTS]) \
        if target.is_dir() else [target]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, img_size = load_model(device)
    print(f"模型已加载: {MODEL_PATH} (设备: {device})")

    for f in files:
        top3 = predict(model, classes, img_size, f, device)
        name, conf = top3[0]
        others = ", ".join(f"{n} {c:.1%}" for n, c in top3[1:])
        print(f"{f.name}: 识别为 [{name}] (置信度 {conf:.1%}) | Top-3: {others}")


if __name__ == "__main__":
    main()
