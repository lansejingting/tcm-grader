# -*- coding: utf-8 -*-
"""
GradeHub 定级调度中心
======================
衔接两个子系统，组成"小模型识品种 → 大模型定级"流水线：

  1. 本地视觉分类器（本目录, ResNet18 迁移学习, 验证集 Top-1 95.1%）
     → 毫秒级品种识别, 输出 Top-1 + 置信度 + Top-3 候选
  2. grading/ 大模型定级流水线（规则检索 + qwen-vl-max 木桶效应定级）
     → 通过 SourceFileLoader 复用 02/03 模块, 不复制、不修改原代码

统一入口：
    process_image(image_path, forced_herb=None) -> dict
"""

import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TCM_GRADING_DIR = BASE_DIR.parent / "grading"

sys.path.insert(0, str(TCM_GRADING_DIR))

from importlib.machinery import SourceFileLoader

# ---- 复用 tcm_grading 的配置 / 检索 / 定级模块 ----
tcm_config = SourceFileLoader("tcm_config", str(TCM_GRADING_DIR / "config.py")).load_module()
step2 = SourceFileLoader("tcm_step2", str(TCM_GRADING_DIR / "02_retrieve_rules.py")).load_module()
step3 = SourceFileLoader("tcm_step3", str(TCM_GRADING_DIR / "03_grade_herbs.py")).load_module()

RETRIEVER = step2.RuleRetriever()

# ---- 对外暴露的常用配置（demo2 直接从这里取）----
HERB_LIST = tcm_config.HERB_LIST
DATASET_DIR = tcm_config.DATASET_DIR
IMAGE_EXTENSIONS = tcm_config.IMAGE_EXTENSIONS
UPLOAD_DIR = BASE_DIR / "results" / "demo_uploads"
REVIEW_LOG = BASE_DIR / "results" / "review_log.jsonl"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---- 本地分类器 ----
CLASSIFIER_CKPT = BASE_DIR / "outputs" / "best_model.pth"
# 本地分类置信度低于该值 → 注入复核标记, 提示人工核对品种
CLASSIFIER_REVIEW_THRESHOLD = 0.60

_MODEL = None
_CLASSES = None
_IMG_SIZE = 224
_DEVICE = None
_lock = threading.Lock()


def _load_classifier():
    """懒加载本地 ResNet18 分类器（进程内单例, CPU/GPU 自动选择）。"""
    global _MODEL, _CLASSES, _IMG_SIZE, _DEVICE
    with _lock:
        if _MODEL is not None:
            return
        import torch
        from torchvision import models

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(CLASSIFIER_CKPT, map_location=_DEVICE, weights_only=False)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["classes"]))
        model.load_state_dict(ckpt["model_state"])
        model.to(_DEVICE).eval()
        _MODEL, _CLASSES = model, ckpt["classes"]
        _IMG_SIZE = ckpt.get("img_size", 224)


def has_api_key() -> bool:
    """大模型 API Key 是否已配置（grading/.env 会自动加载）。"""
    return bool(tcm_config.OPENAI_API_KEY)


def predict_species(image_path: str) -> dict:
    """本地 CNN 品种识别：Top-1 + 置信度 + Top-3 候选（毫秒级，不调用大模型）。"""
    _load_classifier()
    import torch
    from PIL import Image
    from torchvision import transforms

    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(int(_IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        probs = torch.softmax(
            _MODEL(tf(img).unsqueeze(0).to(_DEVICE))[0], dim=0).cpu()
    top = probs.topk(3)
    top3 = [{"herb": _CLASSES[int(i)], "prob": round(float(p), 4)} for p, i in zip(top.values, top.indices)]
    return {
        "identified_herb": top3[0]["herb"],
        "herb_confidence": top3[0]["prob"],
        "herb_top3": top3,
        "method": "local_cnn",
    }


def process_image(image_path: str, forced_herb: str = None) -> dict:
    """完整流水线：品种识别（本地模型 / 人工指定）→ 规则检索 → 大模型定级。"""
    image_data_url = step3.load_and_prepare_image(image_path)

    if forced_herb:
        ident = {"identified_herb": forced_herb, "herb_confidence": 1.0,
                 "herb_reasoning": "人工指定品种，跳过识别", "herb_top3": [], "method": "manual"}
    else:
        ident = predict_species(image_path)
        if ident["herb_confidence"] < CLASSIFIER_REVIEW_THRESHOLD:
            ident["herb_reasoning"] = (
                f"本地分类器置信度偏低（{ident['herb_confidence']:.2f} < {CLASSIFIER_REVIEW_THRESHOLD}），"
                f"建议对照 Top-3 候选人工确认品种")

    grade_result = step3.grade_herb(RETRIEVER, ident["identified_herb"], image_data_url)
    if grade_result["status"] != "ok":
        return {"ok": False, "error": grade_result.get("error", "定级失败"),
                "identification": ident}

    grading = grade_result["result"]
    # 分类器低置信 → 注入复核标记（大模型不知道品种来源可靠性）
    if not forced_herb and ident["herb_confidence"] < CLASSIFIER_REVIEW_THRESHOLD:
        grading["review_flag"] = True
        grading["review_reason"] = (
            grading.get("review_reason", "") + " 本地品种识别置信度偏低，请人工核对品种").strip()

    return {
        "ok": True,
        "identification": ident,
        "grading": grading,
        "retrieved_rules": grade_result.get("retrieved_rules", []),
        "image_path": image_path,
    }


if __name__ == "__main__":
    # 自检：分类器 + 检索器 + API Key 是否就绪
    print(f"[1/3] 分类器权重: {CLASSIFIER_CKPT} ...")
    _load_classifier()
    print(f"      已加载, 类别数 {len(_CLASSES)}, 设备: "
          f"{'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
    print(f"[2/3] 规则检索器: {len(RETRIEVER.chunks) if hasattr(RETRIEVER, 'chunks') else 'ok'}")
    print(f"[3/3] 大模型 API Key: {'已配置' if has_api_key() else '未配置 (请检查 grading/.env)'}")
