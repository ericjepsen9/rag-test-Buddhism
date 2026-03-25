import os as _os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ===== 基础路径 =====
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STORE_ROOT = BASE_DIR / "stores"
OUT_PATH = BASE_DIR / "answer.txt"

# ===== OpenAI / 兼容 API 开关 =====
# 支持 Cherry Studio 等 OpenAI 兼容 API 服务：
#   export RAG_USE_OPENAI=1
#   export OPENAI_API_KEY=cs-sk-...
#   export OPENAI_API_BASE=http://127.0.0.1:23333/v1
#   export RAG_OPENAI_MODEL=your-model-name
USE_OPENAI = _os.environ.get("RAG_USE_OPENAI", "").strip().lower() in ("1", "true", "yes")
OPENAI_MODEL = _os.environ.get("RAG_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE = _os.environ.get("OPENAI_API_BASE", "").strip() or None

# ===== 输出与调试 =====
DEBUG = False
DEFAULT_MODE = "brief"   # brief / full
DEFAULT_TOP_K = 8
DEFAULT_USER_LEVEL = "beginner"  # beginner / experienced

# ===== 回答模板 =====
REFERENCE_NOTE = "以上内容基于佛教经典与传统教义整理，仅供学习参考。"
RISK_NOTE = "深入修行建议亲近善知识，依止有经验的法师指导。"

# ===== 知识领域配置 =====
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

# 默认知识领域
DEFAULT_PRODUCT = "buddhism"

# 易混淆词（命中时提示用户明确）
AMBIGUOUS_TOKENS = []

# ===== 共享知识实体（非产品级，跨产品通用） =====
SHARED_ENTITY_DIRS = {
    "scripture":    "scriptures",
    "master":       "masters",
    "glossary":     "glossary",
}


def _safe_int(key: str, default: str) -> int:
    """安全读取环境变量并转为 int，非法值回退默认值并打印警告。"""
    raw = _os.environ.get(key, default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        print(f"[WARN] Env var {key}='{raw}' cannot convert to int, using default {default}")
        return int(default)


# ===== 消歧引导配置 =====
CLARIFICATION_ENABLED = _os.environ.get("RAG_CLARIFICATION", "1").strip().lower() in ("1", "true", "yes")
CLARIFICATION_MIN_QUERY_LEN = _safe_int("RAG_CLARIFY_MIN_LEN", "6")
CLARIFICATION_MAX_QUERY_LEN = _safe_int("RAG_CLARIFY_MAX_LEN", "15")

UNCLEAR_PRODUCT_PROMPT = (
    "请明确您想了解的佛教领域或主题，例如：\n"
    "- 佛教基础教义（四圣谛、八正道等）\n"
    "- 修行方法（禅修、念佛等）\n"
    "- 佛教经典（心经、金刚经等）\n"
)

# ===== 无知识兜底回复 =====
PRICE_REPLY = "佛教修行不涉及价格问题，建议向当地寺院或佛学院咨询相关信息。"
COMPARISON_REPLY = "不同修行法门各有殊胜之处，建议亲近善知识，根据自身根机选择合适的修行方法。"
LOCATION_REPLY = "建议通过当地佛教协会或正规寺院查询相关信息。"

# ===== 实体关联 =====
RELATIONS_FILE = KNOWLEDGE_DIR / "relations.json"

# ===== 模型参数 =====
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_USE_FP16 = True
EMBED_BATCH_SIZE_BUILD = 8
EMBED_BATCH_SIZE_QUERY = 1
EMBED_MAX_LENGTH_BUILD = 8192
EMBED_MAX_LENGTH_QUERY = 1024
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# ===== LLM 参数 =====
LLM_REWRITE_ENABLED = _os.environ.get("RAG_LLM_REWRITE", "1").strip().lower() in ("1", "true", "yes")

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS_BRIEF = 1500
LLM_MAX_TOKENS_FULL = 2500

# 路由专属温度：教义/概念类需要确定性低温，修行/历史可稍高
ROUTE_LLM_TEMPERATURE = {
    "doctrine":   0.1,
    "concept":    0.1,
    "scripture":  0.2,
    "basic":      0.2,
    "practice":   0.3,
    "sect":       0.3,
    "history":    0.4,
    "ritual":     0.3,
    "life":       0.4,
}

# ===== 搜索调优 =====


def _safe_float(key: str, default: str) -> float:
    """安全读取环境变量并转为 float，非法值回退默认值并打印警告。"""
    raw = _os.environ.get(key, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        print(f"[WARN] Env var {key}='{raw}' cannot convert to float, using default {default}")
        return float(default)


BM25_K1 = _safe_float("RAG_BM25_K1", "1.5")
BM25_B = _safe_float("RAG_BM25_B", "0.75")
SIGMOID_SCALE = _safe_float("RAG_SIGMOID_SCALE", "5.0")
ROUTE_BOOST = _safe_float("RAG_ROUTE_BOOST", "0.12")
CACHE_MAX_PRODUCTS = 32

# ===== 回答构建 =====
MAX_SUB_QUESTIONS = 4    # 单次问答最多拆分的子问题数
MAX_EVIDENCE_CHUNKS = 6  # build_evidence / answer_formatter 保留的最大证据片段数

# ===== 检索 =====
VECTOR_TOP_K = _safe_int("RAG_VECTOR_TOP_K", "12")
KEYWORD_TOP_K = _safe_int("RAG_KEYWORD_TOP_K", "12")
HYBRID_VECTOR_WEIGHT = _safe_float("RAG_HYBRID_VW", "0.65")
HYBRID_KEYWORD_WEIGHT = _safe_float("RAG_HYBRID_KW", "0.35")

# ===== Rerank 配置 =====
# cross-encoder 精排（BGE-reranker-v2-m3），在 hybrid merge + filter 之后使用
USE_RERANK = True
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 6
RERANK_SCORE_THRESHOLD = 0.1
# 同时保持与参考项目兼容的开关（BGE-M3 compute_score 模式）
RERANK_ENABLED = _os.environ.get("RAG_RERANK_ENABLED", "0").strip().lower() in ("1", "true", "yes")
RERANK_TOP_N = _safe_int("RAG_RERANK_TOP_N", "10")

# ===== 动态阈值 =====
DYNAMIC_THRESHOLD_ENABLED = _os.environ.get("RAG_DYN_THRESHOLD", "1").strip().lower() in ("1", "true", "yes")
DYNAMIC_THRESHOLD_RATIO = _safe_float("RAG_DYN_RATIO", "0.40")
DYNAMIC_THRESHOLD_FLOOR_RATIO = _safe_float("RAG_DYN_FLOOR_RATIO", "0.70")

# ===== 中文分词 =====
# 佛教项目使用自建词典分词，jieba 作为可选补充
JIEBA_ENABLED = _os.environ.get("RAG_JIEBA_ENABLED", "0").strip().lower() in ("1", "true", "yes")

# ===== FAISS 索引类型 =====
FAISS_INDEX_TYPE = _os.environ.get("RAG_FAISS_INDEX", "flat").strip().lower()
FAISS_HNSW_M = _safe_int("RAG_HNSW_M", "32")
FAISS_HNSW_EF_CONSTRUCTION = _safe_int("RAG_HNSW_EFC", "200")
FAISS_HNSW_EF_SEARCH = _safe_int("RAG_HNSW_EFS", "128")

# ===== 按问题类型调整检索参数 =====
QUESTION_TYPE_CONFIG = {
    "doctrine":  {"k": 8,  "threshold": 0.25},
    "practice":  {"k": 8,  "threshold": 0.25},
    "scripture": {"k": 8,  "threshold": 0.25, "vw": 0.55, "kw": 0.45},
    "sect":      {"k": 8,  "threshold": 0.25},
    "concept":   {"k": 8,  "threshold": 0.25},
    "history":   {"k": 8,  "threshold": 0.25},
    "ritual":    {"k": 6,  "threshold": 0.25},
    "basic":     {"k": 6,  "threshold": 0.30},
    "life":      {"k": 10, "threshold": 0.20, "vw": 0.60, "kw": 0.40},
}

# 默认分数阈值（路由未配置时使用）
SCORE_THRESHOLD = 0.25

# ===== FAQ 快速路径阈值 =====
FAQ_FAST_PATH_THRESHOLDS = {
    "doctrine":  {"score": 0.55, "ratio": 0.60},
    "practice":  {"score": 0.50, "ratio": 0.55},
    "scripture": {"score": 0.55, "ratio": 0.60},
    "concept":   {"score": 0.55, "ratio": 0.60},
    "basic":     {"score": 0.50, "ratio": 0.55},
    "life":      {"score": 0.45, "ratio": 0.50},
}
FAQ_FAST_PATH_DEFAULT = {"score": 0.55, "ratio": 0.60}

# ===== brief/full 模式配置 =====
ANSWER_MODE_CONFIG = {
    "brief": {
        "max_items_default": 8,
        "doctrine": 20,
        "practice": 20,
        "scripture": 14,
        "sect": 14,
        "concept": 14,
        "history": 14,
        "ritual": 12,
        "basic": 12,
        "life": 16,
    },
    "full": {
        "max_items_default": 14,
        "doctrine": 40,
        "practice": 40,
        "scripture": 30,
        "sect": 30,
        "concept": 30,
        "history": 30,
        "ritual": 24,
        "basic": 20,
        "life": 30,
    },
}

# ===== 问题路由 =====
QUESTION_ROUTES = {
    "doctrine": ["四圣谛", "八正道", "十二因缘", "三法印", "四法印", "缘起", "中道",
                  "苦谛", "集谛", "灭谛", "道谛", "无常", "无我", "三毒", "贪嗔痴",
                  "四谛", "正见", "正思维", "正语", "正业", "正命", "正精进", "正定"],
    "practice": ["禅修", "禅定", "打坐", "念佛", "持咒", "诵经", "抄经", "放生",
                 "布施", "修行", "早晚课", "正念", "内观", "止观", "冥想", "坐禅",
                 "数息", "大悲咒", "楞严咒", "六字大明咒", "怎么修", "如何修",
                 "持名念佛", "观想", "法门", "净土法门", "在家居士",
                 "断除", "对治", "调伏", "降伏", "断烦恼",
                 "贪心", "嗔心", "痴心", "贪欲", "嗔恨", "嗔恚", "愚痴",
                 "忏悔", "发愿", "回向", "发心", "发菩提心",
                 "持戒", "忍辱", "精进", "学处", "菩萨戒", "别解脱戒",
                 "闻思修", "加行", "前行", "道次第", "修法"],
    "scripture": ["心经", "金刚经", "法华经", "华严经", "楞严经", "地藏经", "药师经",
                  "阿弥陀经", "无量寿经", "坛经", "维摩诘经", "般若经", "三藏", "经典",
                  "佛经", "大般若经", "观无量寿佛经", "六祖坛经", "经藏", "律藏", "论藏",
                  "入行论", "入菩萨行论", "中论", "入中论", "大智度论", "瑜伽师地论",
                  "现观庄严论", "俱舍论", "寂天"],
    "sect": ["禅宗", "净土宗", "天台宗", "华严宗", "唯识宗", "法相宗", "三论宗",
             "律宗", "密宗", "宗派", "南传", "北传", "上座部", "大乘", "小乘",
             "藏传", "格鲁派", "噶举派", "宁玛派", "萨迦派",
             "金刚乘", "真言宗", "曹洞宗", "临济宗", "日莲宗", "教派"],
    "concept": ["空性", "涅槃", "轮回", "因果", "业报", "六道", "菩萨", "阿罗汉",
                "佛性", "法身", "菩提", "烦恼", "五蕴", "三宝", "皈依", "般若",
                "业力", "善业", "恶业", "六波罗蜜", "六度", "菩提心", "法界",
                "缘起性空", "中观", "唯识", "如来藏", "佛法僧",
                "天道", "人道", "畜生道", "饿鬼道", "地狱道", "阿修罗道",
                "解脱", "执著", "无明", "三学", "戒定慧",
                "因果报应", "生死", "五戒",
                "自他交换", "自他相换", "安忍", "忍辱", "不放逸", "正知正念",
                "暇满人身", "人身难得", "二谛", "世俗谛", "胜义谛"],
    "history": ["佛教历史", "传入中国", "白马寺", "玄奘", "鸠摩罗什", "达摩", "慧能",
                "太虚", "印顺", "人间佛教", "佛教传播", "善导", "智者大师",
                "道宣", "法藏", "龙树", "宗喀巴", "星云", "证严",
                "八大宗派", "魏晋南北朝", "隋唐"],
    "ritual": ["礼仪", "合掌", "顶礼", "上香", "供花", "供灯", "法会", "节日",
               "浴佛", "盂兰盆", "佛诞", "皈依仪式", "受戒", "腊八", "成道日",
               "涅槃日", "观音圣诞", "药师佛圣诞", "阿弥陀佛圣诞"],
    "basic": ["是什么", "什么是", "介绍", "基础", "入门", "了解", "概述", "简介",
              "怎么理解", "含义", "意思", "定义",
              "释迦牟尼", "悉达多", "佛陀生平", "菩提树", "鹿野苑", "三藏经典"],
    "life": ["生活", "日常", "工作", "职场", "压力", "焦虑", "情绪", "脾气", "愤怒", "生气",
             "家庭", "婚姻", "夫妻", "孩子", "父母", "亲人", "人际关系", "同事", "朋友",
             "吵架", "矛盾", "误解", "冤枉", "委屈", "内疚", "自责", "抑郁", "迷茫",
             "失眠", "焦虑", "躺平", "没有动力", "拖延", "懈怠",
             "手机", "上瘾", "注意力", "分心", "专注",
             "怎么办", "怎么面对", "怎么处理", "怎么调节", "怎么化解",
             "死亡", "去世", "离世", "丧亲", "临终",
             "财富", "金钱", "理财", "素食", "饮食",
             "换位思考", "将心比心", "同理心",
             "佛法与生活", "佛教与生活", "生活中修行", "生活应用",
             "开示", "演讲", "讲座", "法师开示", "法师演讲"],
}

# ===== 章节标题映射（主文档优先） =====
SECTION_RULES = {
    "doctrine": {
        "titles": ["二、四圣谛", "四圣谛", "三、八正道", "八正道"],
        "stops": ["四、", "五、", "六、"],
    },
    "practice": {
        "titles": ["六、修行方法", "修行方法"],
        "stops": ["七、", "八、", "九、"],
    },
    "scripture": {
        "titles": ["五、重要经典", "重要经典"],
        "stops": ["六、", "七、", "八、"],
    },
    "sect": {
        "titles": ["四、主要宗派", "主要宗派"],
        "stops": ["五、", "六、", "七、"],
    },
    "concept": {
        "titles": ["七、核心概念", "核心概念"],
        "stops": ["八、", "九、", "十、"],
    },
    "history": {
        "titles": ["八、佛教在中国的传播", "佛教在中国的传播", "佛教历史"],
        "stops": ["九、", "十、"],
    },
    "ritual": {
        "titles": ["九、佛教节日与礼仪", "佛教节日与礼仪", "基本礼仪"],
        "stops": ["十、"],
    },
    "basic": {
        "titles": ["一、佛教基础知识", "佛教基础知识"],
        "stops": ["二、", "三、", "四、"],
    },
    "life": {
        "titles": ["七、在日常生活中的应用", "日常生活中的应用", "佛法与生活"],
        "stops": ["八、", "九、", "十、"],
    },
}

# ===== 别名与多实体 =====
PRODUCT_ALIASES = {
    "buddhism": ["佛教", "佛法", "佛学", "Buddhism", "释迦牟尼", "佛陀"],
}

PROJECT_ALIASES = {
    "禅修": ["禅修", "禅定", "打坐", "坐禅", "冥想", "止观", "内观"],
    "念佛": ["念佛", "净土", "阿弥陀佛", "极乐世界", "往生"],
    "般若": ["般若", "智慧", "空性", "中观", "缘起性空"],
    "戒律": ["戒律", "五戒", "菩萨戒", "持戒", "受戒"],
    "经典": ["经典", "经文", "佛经", "三藏", "心经", "金刚经"],
    "菩萨": ["菩萨", "观音", "文殊", "普贤", "地藏", "菩提心"],
    "轮回": ["轮回", "六道", "因果", "业报", "业力", "投生"],
}

BUDDHIST_TERMS = [
    "四圣谛", "八正道", "十二因缘", "涅槃", "轮回", "因果", "菩萨", "空性",
    "般若", "五蕴", "三宝", "六度", "三法印", "缘起", "中道", "菩提",
    "三毒", "贪嗔痴", "戒定慧", "三学", "五戒", "六波罗蜜", "菩提心",
    "禅修", "念佛", "净土", "禅宗", "心经", "金刚经", "法华经",
    "入行论", "入菩萨行论", "寂天", "自他交换", "安忍", "不放逸",
    "正知正念", "静虑", "回向", "暇满",
]
CONCEPT_TERMS = [
    "无常", "无我", "苦", "空", "缘起", "业力", "解脱", "菩提",
    "烦恼", "执著", "无明", "法身", "佛性", "如来藏", "法界",
]

# ===== 生活化表述→佛学术语映射（用于查询改写和同义词扩展） =====
LIFE_SYNONYM_MAP = {
    "生气": "嗔恨", "发火": "嗔恨", "发脾气": "嗔恨", "愤怒": "嗔恨",
    "忍耐": "安忍", "忍受": "安忍", "耐心": "安忍", "忍气吞声": "安忍",
    "控制情绪": "安忍", "情绪管理": "安忍",
    "努力": "精进", "勤奋": "精进", "上进": "精进", "动力": "精进",
    "懈怠": "不放逸", "拖延": "不放逸", "躺平": "不放逸",
    "专注": "正知正念", "注意力": "正知正念", "觉察": "正知正念",
    "活在当下": "正知正念", "分心": "正知正念",
    "换位思考": "自他交换", "将心比心": "自他交换", "同理心": "自他交换",
    "利他": "菩提心", "为他人着想": "菩提心", "助人为乐": "菩提心",
    "后悔": "忏悔", "自责": "忏悔", "内疚": "忏悔",
    "焦虑": "空性", "迷茫": "空性", "想不开": "空性",
    "看不开": "空性", "执着": "执著", "放不下": "执著",
}

# ===== FAQ 精确匹配映射 =====
FAQ_KEYWORD_MAP = {
    "四圣谛": "四圣谛",
    "八正道": "八正道",
    "三宝": "三宝",
    "因果报应": "因果报应",
    "因果": "因果报应",
    "轮回": "轮回",
    "宗派": "主要宗派",
    "空性": "空性",
    "涅槃": "涅槃",
    "菩萨": "菩萨",
    "禅修": "禅修",
    "念佛": "念佛法门",
    "心经": "心经",
    "金刚经": "金刚经",
    "五戒": "五戒",
    "十二因缘": "十二因缘",
    "礼仪": "基本礼仪",
    "生死": "生死",
    "在家居士": "在家居士",
    "节日": "重要节日",
    "入行论": "入行论",
    "入菩萨行论": "入行论",
    "寂天": "入行论",
    "自他交换": "自他交换",
    "自他相换": "自他交换",
    "安忍": "安忍品",
    "智慧品": "智慧品",
}

# ===== 媒体 =====
# 每个产品的媒体文件位于 knowledge/{product_id}/media.json
# 由 media_router.py 按 product_id 加载

# ===== 测试 =====
REGRESSION_CASES_FILE = BASE_DIR / "regression_cases.json"

# ===== 服务器配置 =====
SERVER_HOST = _os.environ.get("RAG_SERVER_HOST", "0.0.0.0")
SERVER_PORT = _safe_int("RAG_SERVER_PORT", "8000")

# ===== 运行时热更新支持 =====
import json as _json
import threading as _threading

_CONFIG_FILE = BASE_DIR / "data" / "runtime_overrides.json"
_SERVER_CONFIG_FILE = BASE_DIR / "data" / "server_config.json"
_config_lock = _threading.Lock()

# 模型提供商预设
MODEL_PRESETS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "api_base": "",
    },
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
    },
    "minimax": {
        "label": "MiniMax",
        "models": ["MiniMax-Text-01", "abab6.5s-chat", "abab5.5-chat"],
        "default_model": "MiniMax-Text-01",
        "api_base": "https://api.minimax.chat/v1",
    },
    "custom": {
        "label": "Custom / 自定义",
        "models": [],
        "default_model": "",
        "api_base": "",
    },
}

# 可热更新参数定义
TUNABLE_PARAMS = {
    "bm25_k1":          ("BM25_K1",          float, 0.5, 5.0,   "BM25 词频饱和参数"),
    "bm25_b":           ("BM25_B",           float, 0.0, 1.0,   "BM25 文档长度归一化"),
    "sigmoid_scale":    ("SIGMOID_SCALE",    float, 1.0, 20.0,  "BM25 分数 sigmoid 缩放"),
    "route_boost":      ("ROUTE_BOOST",      float, 0.0, 0.5,   "路由匹配加分"),
    "vector_top_k":     ("VECTOR_TOP_K",     int,   1,   50,    "向量检索返回数"),
    "keyword_top_k":    ("KEYWORD_TOP_K",    int,   1,   50,    "关键词检索返回数"),
    "hybrid_vw":        ("HYBRID_VECTOR_WEIGHT",  float, 0.0, 1.0, "混合检索向量权重"),
    "hybrid_kw":        ("HYBRID_KEYWORD_WEIGHT", float, 0.0, 1.0, "混合检索关键词权重"),
    "rerank_enabled":   ("RERANK_ENABLED",   bool, None, None,  "启用 Reranker 重排序"),
    "rerank_top_n":     ("RERANK_TOP_N",     int,   5,   50,    "Reranker 候选数"),
    "use_rerank":       ("USE_RERANK",       bool, None, None,  "启用 CrossEncoder 精排"),
    "rerank_top_k":     ("RERANK_TOP_K",     int,   1,   20,    "CrossEncoder 精排 Top-K"),
    "dyn_threshold_enabled": ("DYNAMIC_THRESHOLD_ENABLED", bool, None, None, "启用动态阈值"),
    "dyn_ratio":        ("DYNAMIC_THRESHOLD_RATIO",       float, 0.1, 0.9, "动态阈值比率"),
    "dyn_floor_ratio":  ("DYNAMIC_THRESHOLD_FLOOR_RATIO", float, 0.3, 1.0, "动态阈值下限比率"),
    "llm_temperature":  ("LLM_TEMPERATURE",       float, 0.0, 1.0,  "LLM 默认温度"),
    "llm_max_brief":    ("LLM_MAX_TOKENS_BRIEF",  int,   100, 4000, "LLM brief 最大 token"),
    "llm_max_full":     ("LLM_MAX_TOKENS_FULL",   int,   200, 8000, "LLM full 最大 token"),
    "llm_rewrite":      ("LLM_REWRITE_ENABLED",   bool,  None, None, "启用 LLM 查询改写"),
    "chunk_size":       ("CHUNK_SIZE",       int,   100, 2000, "文本分块大小（字符）"),
    "chunk_overlap":    ("CHUNK_OVERLAP",    int,   0,   500,  "分块重叠长度"),
    "faiss_index_type": ("FAISS_INDEX_TYPE", str,   None, None, "FAISS 索引类型 (flat/hnsw)"),
    "use_openai":       ("USE_OPENAI",       bool, None, None, "启用 LLM（OpenAI 兼容）"),
    "openai_model":     ("OPENAI_MODEL",     str,  None, None, "LLM 模型名称"),
    "openai_api_base":  ("OPENAI_API_BASE",  str,  None, None, "LLM API 地址"),
    "clarification_enabled": ("CLARIFICATION_ENABLED", bool, None, None, "启用模糊查询消歧引导"),
    "default_user_level": ("DEFAULT_USER_LEVEL", str, None, None, "默认用户级别 (beginner/experienced)"),
}


def get_tunable_config() -> dict:
    """获取所有可调参数的当前值"""
    import rag_runtime_config as _mod
    result = {}
    for key, (var_name, vtype, vmin, vmax, desc) in TUNABLE_PARAMS.items():
        val = getattr(_mod, var_name, None)
        result[key] = {
            "value": val,
            "type": vtype.__name__,
            "min": vmin,
            "max": vmax,
            "description": desc,
            "var_name": var_name,
        }
    return result


def update_tunable_config(updates: dict) -> dict:
    """热更新可调参数，返回实际更新的字段"""
    import rag_runtime_config as _mod
    changed = {}
    for key, new_val in updates.items():
        if key not in TUNABLE_PARAMS:
            continue
        var_name, vtype, vmin, vmax, desc = TUNABLE_PARAMS[key]
        try:
            if vtype == bool:
                if isinstance(new_val, str):
                    new_val = new_val.strip().lower() in ("1", "true", "yes", "on")
                else:
                    new_val = bool(new_val)
            elif vtype == int:
                new_val = int(new_val)
            elif vtype == float:
                new_val = float(new_val)
            else:
                new_val = str(new_val).strip()
        except (ValueError, TypeError):
            continue
        if vtype in (int, float) and vmin is not None and vmax is not None:
            new_val = max(vmin, min(vmax, new_val))
        if key == "faiss_index_type" and new_val not in ("flat", "hnsw"):
            continue
        old_val = getattr(_mod, var_name, None)
        if old_val != new_val:
            setattr(_mod, var_name, new_val)
            changed[key] = {"old": old_val, "new": new_val}
    if changed:
        # 持久化 clamped 后的值，而非原始输入
        clamped = {k: v["new"] for k, v in changed.items()}
        _persist_overrides(clamped)
    return changed


def _persist_overrides(updates: dict) -> None:
    """将运行时覆盖保存到文件，下次启动时自动加载"""
    with _config_lock:
        data_dir = _CONFIG_FILE.parent
        data_dir.mkdir(parents=True, exist_ok=True)
        existing = {}
        if _CONFIG_FILE.exists():
            try:
                existing = _json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        for key, val in updates.items():
            if key in TUNABLE_PARAMS:
                existing[key] = val
        tmp = _CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_CONFIG_FILE)


def load_persisted_overrides() -> dict:
    """启动时加载持久化的覆盖值"""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        data = _json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if data:
            changed = update_tunable_config(data)
            if changed:
                print(f"[INFO] Loaded {len(changed)} runtime config overrides: {list(changed.keys())}")
            return changed
    except Exception as e:
        print(f"[WARN] Failed to load runtime config overrides: {e}")
    return {}


def get_model_config() -> dict:
    """获取当前模型配置（读取模块级活值，而非 import 时快照）"""
    import rag_runtime_config as _mod
    return {
        "use_openai": _mod.USE_OPENAI,
        "model": _mod.OPENAI_MODEL,
        "api_base": _mod.OPENAI_API_BASE or "",
        "api_key_set": bool(_os.environ.get("OPENAI_API_KEY", "").strip()),
        "llm_rewrite": _mod.LLM_REWRITE_ENABLED,
        "presets": MODEL_PRESETS,
    }


def switch_model_provider(provider: str, model: str = "", api_base: str = "",
                          api_key: str = "") -> dict:
    """切换模型提供商"""
    import rag_runtime_config as _mod
    preset = MODEL_PRESETS.get(provider)
    if not preset and provider != "custom":
        return {"error": f"未知提供商: {provider}"}
    if preset and not model:
        model = preset["default_model"]
    if preset and not api_base:
        api_base = preset["api_base"]
    _mod.USE_OPENAI = True
    _mod.OPENAI_MODEL = model
    _mod.OPENAI_API_BASE = api_base or None
    if api_key:
        _os.environ["OPENAI_API_KEY"] = api_key
    try:
        from rag_answer import _get_openai_client
        import rag_answer
        rag_answer._openai_client = None
        rag_answer._openai_client_checked = False
    except Exception:
        pass
    try:
        from llm_client import sync_from_legacy
        sync_from_legacy()
    except Exception:
        pass
    _persist_overrides({
        "use_openai": True,
        "openai_model": model,
        "openai_api_base": api_base or "",
    })
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "api_base": api_base or "",
    }


# ===== 服务器 / 域名配置管理 =====

def get_server_config() -> dict:
    data = _load_server_config_file()
    return {
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "domain": data.get("domain", ""),
        "ssl_enabled": data.get("ssl_enabled", False),
        "cors_origins": _os.environ.get("CORS_ORIGINS", "*"),
        "chat_path": "/chat",
        "admin_path": "/admin",
        "api_path": "/ask",
    }


def update_server_config(updates: dict) -> dict:
    import rag_runtime_config as _mod
    data = _load_server_config_file()
    changed = {}
    allowed_keys = {"domain", "ssl_enabled", "ssl_cert_path", "ssl_key_path",
                    "cors_origins", "auto_start"}
    for key, val in updates.items():
        if key not in allowed_keys:
            continue
        if key == "ssl_enabled":
            val = bool(val)
        elif key == "cors_origins":
            val = str(val).strip()
            _os.environ["CORS_ORIGINS"] = val
        else:
            val = str(val).strip()
        old = data.get(key, "")
        if old != val:
            data[key] = val
            changed[key] = {"old": old, "new": val}
    if "host" in updates:
        new_host = str(updates["host"]).strip()
        if new_host != _mod.SERVER_HOST:
            old_host = _mod.SERVER_HOST
            _mod.SERVER_HOST = new_host
            data["host"] = new_host
            changed["host"] = {"old": old_host, "new": new_host}
    if "port" in updates:
        try:
            new_port = int(updates["port"])
            if 1 <= new_port <= 65535 and new_port != _mod.SERVER_PORT:
                old_port = _mod.SERVER_PORT
                _mod.SERVER_PORT = new_port
                data["port"] = new_port
                changed["port"] = {"old": old_port, "new": new_port}
        except (ValueError, TypeError):
            pass
    if changed:
        _save_server_config_file(data)
    return changed


def _load_server_config_file() -> dict:
    if not _SERVER_CONFIG_FILE.exists():
        return {}
    try:
        return _json.loads(_SERVER_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_server_config_file(data: dict) -> None:
    with _config_lock:
        d = _SERVER_CONFIG_FILE.parent
        d.mkdir(parents=True, exist_ok=True)
        tmp = _SERVER_CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_SERVER_CONFIG_FILE)


# ===== BGE-M3 嵌入模型控制 =====

def get_embedding_status() -> dict:
    try:
        import rag_answer
        model = getattr(rag_answer, "_model", None)
        return {
            "loaded": model is not None,
            "model_name": EMBED_MODEL_NAME,
            "use_fp16": EMBED_USE_FP16,
        }
    except Exception:
        return {"loaded": False, "model_name": EMBED_MODEL_NAME}


def start_embedding_model() -> dict:
    try:
        from rag_answer import get_model, embed_query
        get_model()
        embed_query("预热")
        return {"ok": True, "message": "BGE-M3 模型已加载"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stop_embedding_model() -> dict:
    try:
        import rag_answer
        import gc
        rag_answer._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return {"ok": True, "message": "BGE-M3 模型已卸载"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===== LLM 服务控制 =====

def get_llm_status() -> dict:
    import rag_runtime_config as _mod
    try:
        import rag_answer
        client = getattr(rag_answer, "_openai_client", None)
        checked = getattr(rag_answer, "_openai_client_checked", False)
        return {
            "enabled": _mod.USE_OPENAI,
            "client_ready": client is not None,
            "client_checked": checked,
            "model": _mod.OPENAI_MODEL,
            "api_base": _mod.OPENAI_API_BASE or "",
            "api_key_set": bool(_os.environ.get("OPENAI_API_KEY", "").strip()),
            "rewrite_enabled": _mod.LLM_REWRITE_ENABLED,
            "temperature": _mod.LLM_TEMPERATURE,
        }
    except Exception:
        return {
            "enabled": _mod.USE_OPENAI,
            "client_ready": False,
            "model": _mod.OPENAI_MODEL,
        }


def start_llm_service(api_key: str = "") -> dict:
    import rag_runtime_config as _mod
    if api_key:
        _os.environ["OPENAI_API_KEY"] = api_key
    _mod.USE_OPENAI = True
    try:
        # 重置所有 LLM client 缓存，确保使用最新配置
        try:
            from llm_client import reset_all_clients, sync_from_legacy
            reset_all_clients()
            sync_from_legacy()
        except Exception:
            pass
        import rag_answer
        with rag_answer._openai_client_lock:
            rag_answer._openai_client = None
            rag_answer._openai_client_checked = False
        client = rag_answer._get_openai_client()
        if client is None:
            return {"ok": False, "error": "LLM client 创建失败，请检查 API Key 和 API Base"}
        _persist_overrides({"use_openai": True})
        return {"ok": True, "message": f"LLM 服务已启动 (model={OPENAI_MODEL})"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stop_llm_service() -> dict:
    import rag_runtime_config as _mod
    _mod.USE_OPENAI = False
    try:
        import rag_answer
        with rag_answer._openai_client_lock:
            rag_answer._openai_client = None
            rag_answer._openai_client_checked = False
    except Exception:
        pass
    try:
        from llm_client import reset_all_clients, sync_from_legacy
        reset_all_clients()
        sync_from_legacy()
    except Exception:
        pass
    _persist_overrides({"use_openai": False})
    return {"ok": True, "message": "LLM 服务已停止"}


# 启动时自动加载持久化覆盖
load_persisted_overrides()

# 启动时加载服务器配置
_server_data = _load_server_config_file()
if _server_data.get("host"):
    SERVER_HOST = _server_data["host"]
if _server_data.get("port"):
    try:
        SERVER_PORT = int(_server_data["port"])
    except (ValueError, TypeError):
        pass


# ===== Nginx 配置生成 =====

def generate_nginx_config(domain: str, port: int = 0, ssl: bool = False,
                          cert_path: str = "", key_path: str = "") -> str:
    """生成 nginx 反向代理配置"""
    port = port or SERVER_PORT
    if ssl and cert_path and key_path:
        return f"""server {{
    listen 80;
    server_name {domain};
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {domain};

    ssl_certificate     {cert_path};
    ssl_certificate_key {key_path};
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    client_max_body_size 100m;

    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }}

    location /v1/chat/completions {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }}
}}"""
    else:
        return f"""server {{
    listen 80;
    server_name {domain};

    client_max_body_size 100m;

    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }}

    location /v1/chat/completions {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }}
}}"""
