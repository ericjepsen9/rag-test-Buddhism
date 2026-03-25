"""消歧引导引擎：当用户查询模糊且缺乏上下文时，生成候选选项帮助用户精确提问。

设计原则：
- 不是每次都弹选项，仅在查询短且歧义大时触发
- 同时返回一个"最可能的答案"方向 + 候选选项，不阻断流程
- 选项数量控制在 2-4 个，外加一个"直接搜索"兜底
- 如果上下文已明确（历史对话中有经典/宗派），跳过消歧
"""

import re
from typing import Dict, List, Any, Optional

from rag_runtime_config import (
    PRODUCT_ALIASES, PROJECT_ALIASES, QUESTION_ROUTES,
    CLARIFICATION_ENABLED, CLARIFICATION_MIN_QUERY_LEN, CLARIFICATION_MAX_QUERY_LEN,
)

# ============================================================
# 消歧规则表：模糊关键词 → 候选场景
# key: 触发词（用户输入中包含即匹配）
# value: 候选选项列表，每个选项包含 label（展示）、query（替代查询）、route（路由提示）
# ============================================================

_CLARIFICATION_RULES: Dict[str, List[Dict[str, str]]] = {
    # === 概念类模糊查询 ===
    "空": [
        {"label": "空性（般若思想中的核心概念）", "query": "空性是什么意思 般若 缘起性空", "route": "concept"},
        {"label": "空观（修行中的观法）", "query": "空观怎么修 禅修方法", "route": "practice"},
        {"label": "空宗（中观学派）", "query": "空宗是什么 中观学派 龙树", "route": "school"},
    ],
    "无我": [
        {"label": "无我的教义含义", "query": "无我是什么意思 佛教基本教义", "route": "concept"},
        {"label": "如何体悟无我", "query": "如何体悟无我 修行方法", "route": "practice"},
        {"label": "无我与轮回的关系", "query": "无我与轮回有什么关系", "route": "concept"},
    ],
    "因果": [
        {"label": "因果报应的教义", "query": "因果报应是什么 佛教基本教义", "route": "concept"},
        {"label": "因果与业力的关系", "query": "因果和业力有什么关系", "route": "concept"},
        {"label": "如何理解因果不虚", "query": "因果不虚怎么理解 修行指导", "route": "practice"},
    ],
    "轮回": [
        {"label": "轮回的基本概念", "query": "轮回是什么 六道轮回 佛教基本概念", "route": "concept"},
        {"label": "如何解脱轮回", "query": "如何解脱轮回 修行方法 解脱道", "route": "practice"},
        {"label": "轮回在不同宗派的解释", "query": "不同宗派对轮回的解释 比较", "route": "school"},
    ],
    "涅槃": [
        {"label": "涅槃的含义", "query": "涅槃是什么意思 佛教终极目标", "route": "concept"},
        {"label": "如何证得涅槃", "query": "如何证得涅槃 修行次第", "route": "practice"},
        {"label": "大乘与小乘涅槃观的区别", "query": "大乘小乘涅槃观有什么区别", "route": "school"},
    ],

    # === 修行类模糊查询 ===
    "禅": [
        {"label": "禅宗（宗派介绍）", "query": "禅宗是什么 历史 祖师 思想", "route": "school"},
        {"label": "禅修方法（打坐、参禅）", "query": "禅修怎么修 打坐方法 参禅", "route": "practice"},
        {"label": "禅定（修行境界）", "query": "禅定是什么 四禅八定 修行境界", "route": "concept"},
    ],
    "念佛": [
        {"label": "念佛的方法", "query": "念佛怎么念 持名念佛 方法要领", "route": "practice"},
        {"label": "念佛的功德利益", "query": "念佛有什么功德 利益", "route": "effect"},
        {"label": "净土宗念佛法门", "query": "净土宗念佛法门 理论基础", "route": "school"},
    ],
    "持咒": [
        {"label": "持咒的方法与注意事项", "query": "持咒怎么持 方法 注意事项", "route": "practice"},
        {"label": "常见咒语介绍", "query": "常见佛教咒语有哪些 大悲咒 六字真言", "route": "scripture"},
        {"label": "持咒的功德", "query": "持咒有什么功德 利益", "route": "effect"},
    ],
    "打坐": [
        {"label": "打坐的基本方法", "query": "打坐怎么坐 姿势 方法 入门", "route": "practice"},
        {"label": "打坐中的常见问题", "query": "打坐常见问题 腿疼 昏沉 散乱", "route": "practice"},
        {"label": "打坐与禅定的关系", "query": "打坐和禅定有什么关系", "route": "concept"},
    ],
    "修行": [
        {"label": "修行的基本方法", "query": "佛教修行有哪些基本方法", "route": "practice"},
        {"label": "在家居士如何修行", "query": "在家居士怎么修行 日常功课", "route": "practice"},
        {"label": "修行的次第与阶段", "query": "修行有哪些次第 阶段", "route": "concept"},
    ],

    # === 经典类模糊查询 ===
    "心经": [
        {"label": "《心经》全文与释义", "query": "心经全文 释义 般若波罗蜜多心经", "route": "scripture"},
        {"label": "《心经》的核心思想", "query": "心经核心思想是什么 色即是空", "route": "concept"},
        {"label": "如何读诵《心经》", "query": "心经怎么读诵 方法 注意事项", "route": "practice"},
    ],
    "金刚经": [
        {"label": "《金刚经》基本介绍", "query": "金刚经是什么 基本介绍 内容", "route": "scripture"},
        {"label": "《金刚经》核心要义", "query": "金刚经核心要义 应无所住而生其心", "route": "concept"},
        {"label": "如何读诵《金刚经》", "query": "金刚经怎么读诵 方法", "route": "practice"},
    ],
    "经": [
        {"label": "佛经有哪些分类", "query": "佛经有哪些分类 经律论 三藏", "route": "scripture"},
        {"label": "入门推荐读哪些经", "query": "初学佛推荐读什么经 入门经典", "route": "scripture"},
        {"label": "如何正确读经", "query": "如何正确读经 诵经方法 注意事项", "route": "practice"},
    ],

    # === 宗派类模糊查询 ===
    "净土": [
        {"label": "净土宗基本介绍", "query": "净土宗是什么 基本教义 历史", "route": "school"},
        {"label": "净土宗的修行方法", "query": "净土宗怎么修行 念佛法门", "route": "practice"},
        {"label": "净土经典有哪些", "query": "净土宗经典有哪些 阿弥陀经 无量寿经", "route": "scripture"},
    ],
    "密宗": [
        {"label": "密宗基本介绍", "query": "密宗是什么 藏传佛教 基本介绍", "route": "school"},
        {"label": "密宗的修行特点", "query": "密宗修行有什么特点 灌顶 传承", "route": "practice"},
        {"label": "密宗与显宗的区别", "query": "密宗和显宗有什么区别", "route": "school"},
    ],
    "大乘": [
        {"label": "大乘佛教基本介绍", "query": "大乘佛教是什么 基本教义 菩萨道", "route": "school"},
        {"label": "大乘与小乘的区别", "query": "大乘和小乘有什么区别", "route": "school"},
        {"label": "大乘经典有哪些", "query": "大乘佛教主要经典有哪些", "route": "scripture"},
    ],
    "小乘": [
        {"label": "上座部佛教基本介绍", "query": "上座部佛教是什么 南传佛教 基本介绍", "route": "school"},
        {"label": "小乘与大乘的区别", "query": "小乘和大乘有什么区别 异同", "route": "school"},
        {"label": "上座部的修行方法", "query": "上座部佛教修行方法 四念处 内观", "route": "practice"},
    ],

    # === 人物类模糊查询 ===
    "佛": [
        {"label": "释迦牟尼佛的生平", "query": "释迦牟尼佛生平 成道 弘法", "route": "basic"},
        {"label": "佛的含义与境界", "query": "佛是什么意思 佛的境界 觉悟", "route": "concept"},
        {"label": "其他佛（阿弥陀佛、药师佛等）", "query": "佛教有哪些佛 阿弥陀佛 药师佛", "route": "basic"},
    ],
    "菩萨": [
        {"label": "菩萨的含义与分类", "query": "菩萨是什么意思 有哪些菩萨", "route": "concept"},
        {"label": "菩萨道的修行", "query": "菩萨道怎么修 六度万行 发菩提心", "route": "practice"},
        {"label": "观世音菩萨", "query": "观世音菩萨介绍 大悲咒 普门品", "route": "basic"},
    ],

    # === 生活/应用类模糊查询 ===
    "吃素": [
        {"label": "佛教吃素的意义", "query": "佛教为什么要吃素 吃素的意义", "route": "precept"},
        {"label": "在家居士是否必须吃素", "query": "在家居士必须吃素吗 五戒", "route": "precept"},
        {"label": "吃素的注意事项", "query": "佛教吃素有什么注意事项 营养", "route": "practice"},
    ],
    "放生": [
        {"label": "放生的意义与功德", "query": "放生有什么意义 功德", "route": "practice"},
        {"label": "如何正确放生", "query": "如何正确放生 注意事项 方法", "route": "practice"},
        {"label": "放生的争议与思考", "query": "放生有什么争议 生态问题", "route": "concept"},
    ],
    "烧香": [
        {"label": "烧香拜佛的正确方法", "query": "烧香拜佛怎么做 正确方法 礼仪", "route": "practice"},
        {"label": "烧香的意义", "query": "烧香有什么意义 佛教礼仪", "route": "concept"},
    ],
    "拜佛": [
        {"label": "拜佛的正确方法", "query": "拜佛怎么拜 正确方法 礼仪", "route": "practice"},
        {"label": "拜佛的意义", "query": "拜佛有什么意义 恭敬心", "route": "concept"},
    ],

    # === 效果/功德类模糊查询 ===
    "功德": [
        {"label": "什么是功德", "query": "功德是什么意思 佛教概念", "route": "concept"},
        {"label": "如何积累功德", "query": "如何积累功德 修行方法", "route": "practice"},
        {"label": "功德回向的方法", "query": "功德回向怎么做 方法", "route": "practice"},
    ],
    "回向": [
        {"label": "回向的含义与意义", "query": "回向是什么意思 有什么意义", "route": "concept"},
        {"label": "回向的方法与偈文", "query": "回向怎么做 回向偈 方法", "route": "practice"},
    ],
    "开悟": [
        {"label": "开悟是什么", "query": "开悟是什么意思 佛教觉悟", "route": "concept"},
        {"label": "如何才能开悟", "query": "如何才能开悟 修行方法 条件", "route": "practice"},
        {"label": "开悟的境界与表现", "query": "开悟有什么表现 境界", "route": "concept"},
    ],

    # === 戒律类模糊查询 ===
    "戒": [
        {"label": "佛教戒律基本介绍", "query": "佛教戒律有哪些 五戒 十善", "route": "precept"},
        {"label": "在家居士的戒律", "query": "在家居士持什么戒 五戒 八关斋戒", "route": "precept"},
        {"label": "戒律的意义与作用", "query": "持戒有什么意义 戒律的作用", "route": "concept"},
    ],
    "五戒": [
        {"label": "五戒的具体内容", "query": "五戒是哪五戒 具体内容", "route": "precept"},
        {"label": "如何受持五戒", "query": "如何受持五戒 方法 注意事项", "route": "practice"},
        {"label": "犯戒了怎么办", "query": "犯五戒了怎么办 忏悔方法", "route": "precept"},
    ],

    # === 单字高歧义术语 ===
    "苦": [
        {"label": "苦谛（四圣谛之一）", "query": "苦谛是什么 四圣谛 八苦", "route": "concept"},
        {"label": "如何离苦得乐", "query": "如何离苦得乐 修行方法 灭苦之道", "route": "practice"},
    ],
    "定": [
        {"label": "禅定（修行方法）", "query": "禅定是什么 四禅八定 修行方法", "route": "practice"},
        {"label": "戒定慧三学", "query": "戒定慧三学是什么 修行次第", "route": "concept"},
    ],
    "慧": [
        {"label": "般若智慧", "query": "般若智慧是什么 佛教智慧观", "route": "concept"},
        {"label": "如何修慧", "query": "如何修慧 闻思修 修行方法", "route": "practice"},
    ],

    # === 是什么类通用查询 ===
    "是什么": [
        {"label": "佛教基本概念介绍", "query": "佛教基本概念 基础知识 入门", "route": "basic"},
        {"label": "佛教核心教义", "query": "佛教核心教义 四圣谛 八正道", "route": "concept"},
        {"label": "佛教适合什么人学习", "query": "什么人适合学佛 入门指南", "route": "basic"},
    ],

    # === 怎么做类查询 ===
    "怎么": [
        {"label": "修行方法指南", "query": "佛教修行方法有哪些 入门指南", "route": "practice"},
        {"label": "日常功课安排", "query": "在家居士日常功课怎么安排", "route": "practice"},
    ],

    # === 区别/对比类 ===
    "区别": [
        {"label": "不同宗派的区别", "query": "佛教各宗派有什么区别 特点", "route": "school"},
        {"label": "佛教与其他宗教的区别", "query": "佛教和其他宗教有什么区别", "route": "concept"},
    ],
}

# 预编译：触发词长度排序（长词优先匹配，避免短词过早命中）
_SORTED_TRIGGER_KEYS = sorted(_CLARIFICATION_RULES.keys(), key=len, reverse=True)


_merged_rules_cache: Optional[Dict[str, List[Dict[str, str]]]] = None
_merged_rules_ts: float = 0.0
_MERGED_RULES_TTL = 60.0  # 60 秒缓存


def _get_merged_rules() -> Dict[str, List[Dict[str, str]]]:
    """合并静态规则和动态自定义规则（动态优先覆盖静态），带 TTL 缓存"""
    global _merged_rules_cache, _merged_rules_ts
    import time
    now = time.monotonic()
    if _merged_rules_cache is not None and (now - _merged_rules_ts) < _MERGED_RULES_TTL:
        return _merged_rules_cache
    merged = dict(_CLARIFICATION_RULES)
    try:
        from keyword_store import get_clarification_rules
        custom = get_clarification_rules()
        for trigger, rule_data in custom.items():
            merged[trigger] = rule_data.get("options", [])
    except Exception:
        pass
    _merged_rules_cache = merged
    _merged_rules_ts = now
    return merged

# 不触发消歧的上下文关键词：当用户输入中已包含这些词时，说明意图已明确
_CLEAR_INTENT_PATTERNS = re.compile(
    r"(修行方法|修行次第|经典出处|哪部经|哪个宗派|禅宗|净土宗|密宗|天台宗|华严宗"
    r"|怎么修|怎么念|怎么持|怎么拜|怎么打坐|怎么观|怎么理解"
    r"|功德|利益|意义|区别|关系|出处|全文|释义|核心思想"
    r"|入门|初学|在家居士|出家|受戒|第\d+|日常|每天"
    r"|是什么|有哪些|包括什么|包括哪些|哪些方法|什么方法"
    r"|如何断|如何修|如何对治|如何理解|如何获得|如何证得"
    r"|学处|戒条|戒律|菩萨戒)"
)

# 从配置读取阈值
_MIN_QUERY_LEN_FOR_CLARIFY = CLARIFICATION_MIN_QUERY_LEN
_MAX_QUERY_LEN_FOR_CLARIFY = CLARIFICATION_MAX_QUERY_LEN


def should_clarify(
    question: str,
    products: list,
    projects: list,
    detected_routes: list,
    history_product: str = "",
    history_route: str = "",
    is_chitchat: bool = False,
    is_offtopic: bool = False,
) -> bool:
    """判断是否需要触发消歧引导。

    触发条件（全部满足）：
    1. 非闲聊、非离题
    2. 查询长度在 [2, _MAX_QUERY_LEN_FOR_CLARIFY] 字符之间
    3. 查询中不包含明确意图词（_CLEAR_INTENT_PATTERNS）
    4. 以下至少一个为真：
       a. 无经典/宗派上下文（当前 + 历史都没有）
       b. 检测到的路由模糊（多于1个候选或仅 basic）
       c. 查询过短（< _MIN_QUERY_LEN_FOR_CLARIFY 字符）且路由不明确
    """
    if not CLARIFICATION_ENABLED:
        return False
    if is_chitchat or is_offtopic:
        return False

    q = question.strip()
    q_len = len(q)

    # 太长（>15字）不触发
    if q_len > _MAX_QUERY_LEN_FOR_CLARIFY:
        return False

    # 单字查询：只有消歧规则中有匹配时才触发（避免无意义的单字消歧）
    if q_len == 1:
        merged = _get_merged_rules()
        return q.lower() in merged

    # 已有明确意图表达 → 不需要消歧
    if _CLEAR_INTENT_PATTERNS.search(q):
        return False

    # 有经典/宗派上下文 + 有明确路由 → 不需要消歧
    has_product = bool(products) or bool(history_product)
    has_clear_route = (
        bool(detected_routes)
        and len(detected_routes) == 1
        and detected_routes[0] != "basic"
    )
    if has_product and has_clear_route:
        return False

    # 查询较短且缺少上下文 → 需要消歧
    if q_len < _MIN_QUERY_LEN_FOR_CLARIFY:
        return True

    # 路由模糊（多候选或仅 basic 或无）→ 需要消歧
    if not detected_routes or (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return True

    # 多路由候选 → 需要消歧
    if len(detected_routes) > 1:
        return True

    return False


def generate_clarification(
    question: str,
    products: list = None,
    projects: list = None,
    history_product: str = "",
) -> Optional[Dict[str, Any]]:
    """生成消歧选项。

    返回:
        None: 未命中任何消歧规则
        Dict: {
            "message": 提示语,
            "options": [
                {"label": "空性（般若核心概念）", "query": "...", "route": "concept"},
                ...
            ],
            "fallback_option": {"label": "直接搜索", "query": 原始查询}
        }
    """
    q = question.strip()
    q_lower = q.lower()

    # 合并静态 + 动态消歧规则，长词优先匹配
    merged = _get_merged_rules()
    sorted_keys = sorted(merged.keys(), key=len, reverse=True)

    matched_options = None
    matched_key = None
    for key in sorted_keys:
        if key in q_lower:
            matched_options = merged[key]
            matched_key = key
            break

    if not matched_options:
        return None

    # 如果有经典/宗派上下文，在选项的 query 中补充名称
    product_prefix = ""
    if products:
        pid = products[0]
        aliases = PRODUCT_ALIASES.get(pid, [])
        if aliases:
            product_prefix = aliases[0]
    elif history_product:
        product_prefix = history_product

    # 构建选项（注入上下文）
    options = []
    for opt in matched_options:
        enriched_query = opt["query"]
        if product_prefix and product_prefix not in enriched_query:
            enriched_query = f"{product_prefix} {enriched_query}"
        options.append({
            "label": opt["label"],
            "query": enriched_query,
            "route": opt.get("route", ""),
        })

    # 兜底选项
    fallback_query = q
    if product_prefix and product_prefix not in q:
        fallback_query = f"{product_prefix} {q}"

    return {
        "message": f"请问您想了解关于「{matched_key}」的哪方面问题？",
        "options": options,
        "fallback_option": {
            "label": f"直接搜索「{q}」相关内容",
            "query": fallback_query,
        },
    }


def format_clarification_text(clarification: Dict[str, Any]) -> str:
    """将消歧选项格式化为文本（用于非结构化响应场景）。

    输出格式：
    请问您想了解关于「空」的哪方面问题？
    1. 空性（般若思想中的核心概念）
    2. 空观（修行中的观法）
    3. 空宗（中观学派）
    4. 直接搜索「空」相关内容

    您可以直接回复数字选择，或继续描述您的问题。
    """
    lines = [clarification["message"]]
    for i, opt in enumerate(clarification["options"], 1):
        lines.append(f"{i}. {opt['label']}")
    fb = clarification.get("fallback_option")
    if fb:
        lines.append(f"{len(clarification['options']) + 1}. {fb['label']}")
    lines.append("")
    lines.append("您可以直接回复数字选择，或继续描述您的问题。")
    return "\n".join(lines)


def resolve_numeric_choice(
    user_input: str,
    previous_clarification: Dict[str, Any],
) -> Optional[str]:
    """解析用户的数字选择，返回对应的查询字符串。

    如果用户输入不是数字或超出范围，返回 None。
    """
    s = user_input.strip()
    if not s.isdigit() or len(s) > 5:  # 防止超长数字字符串
        return None
    idx = int(s)
    options = previous_clarification.get("options", [])
    fb = previous_clarification.get("fallback_option")

    if 1 <= idx <= len(options):
        return options[idx - 1]["query"]
    if fb and idx == len(options) + 1:
        return fb["query"]
    return None
