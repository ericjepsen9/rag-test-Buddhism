# rag_config.py
# 稳定版配置（佛教知识问答 RAG 系统）
# 目标：新增知识领域只需在 knowledge/<topic_id> 下添加 main/faq/alias 文件并更新 PRODUCTS

from __future__ import annotations
from pathlib import Path

# ========= 基础路径 =========
BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STORE_DIR = BASE_DIR / "stores"

# 默认知识领域
DEFAULT_PRODUCT = "buddhism"

# ========= 向量模型 =========
MODEL_NAME = "BAAI/bge-m3"
USE_FP16 = True

# ========= Chunk 参数（建库用）=========
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# ========= 回答输出 =========
OUT_PATH = str(BASE_DIR / "answer.txt")
SCORE_THRESHOLD_DEFAULT = 0.35

# ========= 问题类型配置 =========
QUESTION_TYPE_CONFIG = {
    "doctrine": {"k": 8, "threshold": 0.25},
    "history": {"k": 8, "threshold": 0.25},
    "practice": {"k": 8, "threshold": 0.25},
    "scripture": {"k": 8, "threshold": 0.25},
    "sect": {"k": 8, "threshold": 0.25},
    "concept": {"k": 8, "threshold": 0.25},
    "ritual": {"k": 6, "threshold": 0.25},
    "basic": {"k": 6, "threshold": 0.30},
    "summarize": {"k": 10, "threshold": 0.30},
    "define": {"k": 6, "threshold": 0.30},
    "complex": {"k": 10, "threshold": 0.30},
    "fact_short": {"k": 6, "threshold": 0.35},
}

# ========= brief/full 模式 =========
ANSWER_MODE_CONFIG = {
    "brief": {
        "max_items_default": 8,
        "doctrine_max_lines": 20,
        "practice_max_lines": 20,
        "scripture_max_lines": 14,
        "history_max_lines": 14,
        "sect_max_lines": 14,
        "concept_max_lines": 14,
        "ritual_max_lines": 12,
    },
    "full": {
        "max_items_default": 14,
        "doctrine_max_lines": 40,
        "practice_max_lines": 40,
        "scripture_max_lines": 30,
        "history_max_lines": 30,
        "sect_max_lines": 30,
        "concept_max_lines": 30,
        "ritual_max_lines": 24,
    },
}

# ========= 知识领域配置 =========
PRODUCTS = {
    "buddhism": {
        "display_name": "佛教知识（Buddhism）",
        "aliases": [
            "佛教", "佛法", "佛学", "Buddhism",
            "释迦牟尼", "佛陀",
        ],
        "strong_keywords": [
            "四圣谛", "八正道", "涅槃", "轮回", "因果",
            "菩萨", "禅修", "念佛", "般若", "空性",
            "戒定慧", "三宝", "五戒", "六度",
        ],
    },
}

# 易混淆词
AMBIGUOUS_TOKENS = []

UNCLEAR_PRODUCT_PROMPT = (
    "请明确您想了解的佛教领域或主题，例如：\n"
    "- 佛教基础教义（四圣谛、八正道等）\n"
    "- 修行方法（禅修、念佛等）\n"
    "- 佛教经典（心经、金刚经等）\n"
)
