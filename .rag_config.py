# rag_config.py
# RAG 配置文件（可长期复用）
# 新增产品时，优先改这里，不要频繁改 rag_answer.py

OUT_PATH = "answer.txt"
SCORE_THRESHOLD_DEFAULT = 0.35

# -----------------------------------------------------------------------------
# 问题类型参数（你后面可以微调）
# -----------------------------------------------------------------------------
QUESTION_TYPE_CONFIG = {
    "contact": {
        "k": 6,
        "threshold": 0.20,
    },
    "anti_fake": {
        "k": 8,
        "threshold": 0.20,
    },
    "contraindication": {
        "k": 8,
        "threshold": 0.25,
    },
    "aftercare": {
        "k": 8,
        "threshold": 0.25,
    },
    "operation": {
        "k": 10,
        "threshold": 0.25,
    },
    "summarize": {
        "k": 10,
        "threshold": 0.30,
    },
    "define": {
        "k": 6,
        "threshold": 0.30,
    },
    "complex": {
        "k": 10,
        "threshold": 0.30,
    },
    "fact_short": {
        "k": 6,
        "threshold": 0.35,
    },
}

# -----------------------------------------------------------------------------
# 回答模式配置：brief / full
# brief = 简洁但答清楚
# full  = 完整答全但不啰嗦
# -----------------------------------------------------------------------------
ANSWER_MODE_CONFIG = {
    "brief": {
        "max_items_default": 6,
        "anti_fake_max_steps": 12,
        "operation_max_items": 12,
        "contraindication_max_items": 10,
        "aftercare_max_items": 10,
    },
    "full": {
        "max_items_default": 12,
        "anti_fake_max_steps": 30,
        "operation_max_items": 20,
        "contraindication_max_items": 15,
        "aftercare_max_items": 18,
    },
}

# -----------------------------------------------------------------------------
# 产品配置（核心）
# 每个产品一个 product_id，后续 main/faq/alias 建库时写入 meta["product_id"]
# -----------------------------------------------------------------------------
PRODUCTS = {
    # 你当前这个产品（菲罗奥 / 赛罗菲提升 / CELLOFILL）
    "cellofill_lifting": {
        "display_name": "赛罗菲提升（CELLOFILL / 菲罗奥）",
        "aliases": [
            "菲罗奥", "非罗奥", "菲洛奥",
            "赛罗菲", "赛罗菲提升",
            "CELLOFILL", "FILLOUP"
        ],
        "strong_keywords": [
            "HiddenTag", "防伪", "正品认证",
            "PCL", "提升", "微针", "水光", "中胚层"
        ],
        "file_patterns": [
            "feiluoao_main",
            "feiluoao_faq",
            "feiluoao_alias",
            "cellofill"
        ],
    },

    # 未来你要加的相似名字产品（示例）
    # 注意：现在先配置上，哪怕你还没建库，也能提前做名字去混淆
    "cellofill_v_face_fat_dissolve": {
        "display_name": "赛洛菲V脸溶脂",
        "aliases": [
            "赛洛菲V脸溶脂", "赛洛菲v脸溶脂", "赛洛菲溶脂", "V脸溶脂"
        ],
        "strong_keywords": [
            "溶脂", "V脸", "面部轮廓", "脂肪"
        ],
        "file_patterns": [
            "vface",
            "face_slim",
            "rongzhi"
        ],
    },
}

# -----------------------------------------------------------------------------
# 通用别名（用于 query 扩展）
# 注意：这里可以更宽松；真正防串答靠 PRODUCTS + product_id
# -----------------------------------------------------------------------------
ALIASES = {
    "赛罗菲提升（CELLOFILL / 菲罗奥）": [
        "菲罗奥", "非罗奥", "菲洛奥", "赛罗菲", "赛罗菲提升", "CELLOFILL", "FILLOUP"
    ],
    "赛洛菲V脸溶脂": [
        "赛洛菲V脸溶脂", "赛洛菲v脸溶脂", "赛洛菲溶脂", "V脸溶脂"
    ],
}

# 触发品牌扩展的词（宽松）
BRAND_TRIGGERS = [
    "菲罗奥", "非罗奥", "菲洛奥", "赛罗菲", "赛罗菲提升",
    "CELLOFILL", "FILLOUP",
    "赛洛菲", "V脸溶脂", "溶脂"
]

# 易混淆词（命中这些词时，如果没识别清楚产品，就提示用户说清楚）
AMBIGUOUS_TOKENS = [
    "赛罗菲", "赛洛菲"
]

# -----------------------------------------------------------------------------
# 去混淆规则（命中特定关键词时，给某产品额外加分）
# -----------------------------------------------------------------------------
PRODUCT_DISAMBIGUATION_RULES = [
    {
        # 命中这些词，倾向“赛洛菲V脸溶脂”
        "if_any": ["溶脂", "v脸", "瘦脸", "脂肪"],
        "boost_product_id": "cellofill_v_face_fat_dissolve",
        "boost": 3,
    },
    {
        # 命中这些词，倾向“赛罗菲提升 / 菲罗奥”
        "if_any": ["HiddenTag", "防伪", "PCL", "微针", "水光", "注射", "赛罗菲提升", "菲罗奥", "非罗奥"],
        "boost_product_id": "cellofill_lifting",
        "boost": 3,
    },
]

# 当产品名称不清晰时的提示语
UNCLEAR_PRODUCT_PROMPT = (
    "你提到的产品名称可能有歧义（例如“赛罗菲”可能指不同产品）。\n"
    "请补充完整产品名后再问，例如：\n"
    "- 赛罗菲提升（CELLOFILL / 菲罗奥）\n"
    "- 赛洛菲V脸溶脂"
)