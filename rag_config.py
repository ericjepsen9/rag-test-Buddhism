# rag_config.py
# 稳定版配置（对齐 rag_answer.py / build_faiss.py）
# 目标：新增产品尽量只改 knowledge/<product_id> 下的 main/faq/alias + 这里的 PRODUCTS

from __future__ import annotations
from pathlib import Path

# ========= 基础路径 =========
BASE_DIR = Path(__file__).resolve().parent

# knowledge：放“源文件”（每个产品一套 main/faq/alias）
# 结构示例：
#   knowledge/feiluoao/main.txt
#   knowledge/feiluoao/faq.txt
#   knowledge/feiluoao/alias.txt
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# stores：放“索引产物”（每个产品一套 docs.jsonl + index.faiss）
# 结构示例：
#   stores/feiluoao/docs.jsonl
#   stores/feiluoao/index.faiss
STORE_DIR = BASE_DIR / "stores"

# 默认产品（用户不写品牌时的兜底）
DEFAULT_PRODUCT = "feiluoao"

# ========= 向量模型 =========
MODEL_NAME = "BAAI/bge-m3"
USE_FP16 = True

# ========= Chunk 参数（建库用）=========
CHUNK_SIZE = 420
CHUNK_OVERLAP = 100

# ========= 回答输出 =========
OUT_PATH = str(BASE_DIR / "answer.txt")
SCORE_THRESHOLD_DEFAULT = 0.35

# ========= 问题类型配置 =========
QUESTION_TYPE_CONFIG = {
    "contact": {"k": 6, "threshold": 0.20},
    "anti_fake": {"k": 8, "threshold": 0.20},
    "contraindication": {"k": 8, "threshold": 0.25},
    "aftercare": {"k": 8, "threshold": 0.25},
    "operation": {"k": 10, "threshold": 0.25},
    "summarize": {"k": 10, "threshold": 0.30},
    "define": {"k": 6, "threshold": 0.30},
    "complex": {"k": 10, "threshold": 0.30},
    "fact_short": {"k": 6, "threshold": 0.35},
}

# ========= brief/full 模式 =========
ANSWER_MODE_CONFIG = {
    "brief": {
        "max_items_default": 8,
        "anti_fake_max_lines": 30,   # brief 也要答清楚，防伪不建议太短
        "operation_max_lines": 20,
        "contraindication_max_lines": 12,
        "aftercare_max_lines": 14,
    },
    "full": {
        "max_items_default": 14,
        "anti_fake_max_lines": 80,
        "operation_max_lines": 40,
        "contraindication_max_lines": 24,
        "aftercare_max_lines": 30,
    },
}

# ========= 产品配置 =========
# 只要你给每个产品准备 knowledge/<product_id>/{main,faq,alias}.txt
# 并执行 python build_faiss.py --product <product_id> 即可建索引
PRODUCTS = {
    "feiluoao": {
        "display_name": "赛罗菲提升（CELLOFILL / 菲罗奥）",
        "aliases": [
            "菲罗奥", "非罗奥", "菲洛奥",
            "赛罗菲", "赛罗菲提升",
            "CELLOFILL", "FILLOUP",
        ],
        "strong_keywords": [
            "HiddenTag", "防伪", "正品认证",
            "PCL", "提升", "微针", "水光", "中胚层",
        ],
    },
    "sailuofei_vface": {
        "display_name": "赛洛菲V脸溶脂",
        "aliases": [
            "赛洛菲V脸溶脂", "赛洛菲v脸溶脂", "赛洛菲溶脂", "V脸溶脂",
        ],
        "strong_keywords": ["溶脂", "V脸", "面部轮廓", "脂肪", "瘦脸"],
    },
}

# 易混淆词（命中这些词但无法判定产品时提示用户说明）
AMBIGUOUS_TOKENS = ["赛罗菲", "赛洛菲"]

UNCLEAR_PRODUCT_PROMPT = (
    "你提到的产品名称可能有歧义（例如“赛罗菲/赛洛菲”可能指不同产品）。\n"
    "请补充完整产品名后再问，例如：\n"
    "- 赛罗菲提升（CELLOFILL / 菲罗奥）\n"
    "- 赛洛菲V脸溶脂\n"
)
