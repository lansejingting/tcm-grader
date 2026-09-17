"""
TCM-Grader 全局配置
通过环境变量管理敏感信息，不硬编码 API Key
"""

import os
from pathlib import Path

# ============ 路径配置 ============
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "NB-TCM-CHM"
RULES_FILE = DATA_DIR / "rules.json"
RESULTS_DIR = BASE_DIR / "results"

# 向量知识库输出路径
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.json"

# ============ 本地 .env 加载（优先级低于真实环境变量） ============
def _load_dotenv():
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

# ============ API 配置（默认千问/DashScope，可通过环境变量覆盖） ============
# DashScope 兼容模式：https://help.aliyun.com/zh/model-studio/developer-reference/compatible-mode
OPENAI_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
# 多模态视觉模型：qwen-vl-max（更强）或 qwen-vl-plus（更经济）
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen-vl-max")

# Embedding 模型（用于 RAG 检索）
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", OPENAI_BASE_URL)
# 千问 Embedding：text-embedding-v3（dim=1024）或 text-embedding-v2（dim=1536）
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

# ============ 20 种目标中药材（与 rules.json 和数据集一致） ============
HERB_LIST = [
    "枸杞子", "五味子", "山楂", "砂仁", "连翘",
    "补骨脂", "草豆蔻", "栀子", "川楝子", "地肤子",
    "豆蔻", "覆盆子", "瓜蒌", "金樱子", "苦杏仁",
    "木瓜", "山茱萸", "桃仁", "乌梅", "小茴香",
]

# ============ 定级维度 & 等级 ============
# 定级维度（对应用户新 rules.json 中 criteria 的 key）
GRADING_DIMENSIONS = [
    "size",         # 尺寸/大小
    "color",        # 色泽
    "plumpness",    # 饱满度
    "defect_rate",  # 缺陷率/杂质霉变虫蛀
]

# 等级梯队（五档制，从高到低）
GRADE_TIERS = ["特优级", "特级", "甲级", "乙级", "等外品"]
# 等级 code 映射（从高到低排序索引）
GRADE_CODE_ORDER = ["S", "A", "B", "C", "D"]

# ============ 推理配置 ============
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "2.0"))  # 秒

# 置信度阈值：低于此值自动标记为人工复核
REVIEW_CONFIDENCE_THRESHOLD = float(os.environ.get("REVIEW_CONFIDENCE_THRESHOLD", "0.70"))

# ============ RAG 配置 ============
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "3"))
# 同品种精确过滤后，再用语义检索补充边界规则
HERB_FIRST_FILTER = True

# ============ 图片处理 ============
# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 传给模型的最大图片尺寸（长边像素），防止 base64 过大
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
