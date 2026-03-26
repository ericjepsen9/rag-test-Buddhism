"""问题类型分类器：从用户问题中提取结构化意图。

用于驱动搜索策略选择、FAQ 跳过逻辑、来源过滤等决策。
纯规则实现，零成本零延迟。
"""

import re
from typing import Dict, Any

# ============================================================
# 问题类型正则
# ============================================================

_PATTERNS = {
    # --- 知识查询类 ---
    "definition":      re.compile(r"(是什么意思|是什么$|什么是|什么叫|含义|定义)"),
    "method":          re.compile(r"(如何|怎么修|怎样修|怎么断|怎么对治|怎么发|怎么念|怎么持|怎么忏)"),
    "reason":          re.compile(r"(为什么|为何|原因|有什么功德|有什么过患|有什么利益)"),
    "comparison":      re.compile(r"(.{1,12}?)(和|与|跟)(.{1,12}?)(的|有什么|有何)?(区别|不同|异同|差别|对比)"),
    "debate":          re.compile(r"(还是|是否|到底|究竟|矛盾|相违|怎么理解.{0,4}矛盾|既然.{2,10}为什么|那谁|那什么)"),
    "list":            re.compile(r"(有哪些|哪几种|包括什么|包括哪些|列举|哪些方法)"),
    "overview":        re.compile(r"(讲了什么|讲什么|主要内容|内容是什么|概述)"),
    "who":             re.compile(r"(是谁|谁写的|谁造的|作者|哪位)"),
    "which":           re.compile(r"(属于什么|属于哪|归于|哪个宗派|哪一派|什么宗)"),
    "verse":           re.compile(r"(这句话|这个颂词|出自哪|是什么意思.{0,4}$)"),
    "howmany":         re.compile(r"(有几|有多少|几品|几种|几个)"),
    "reference":       re.compile(r"(出自哪|哪部经|哪本书|谁说的|原文|出处|引用自)"),
    "relation":        re.compile(r"(.{1,8})(和|与|跟)(.{1,8})(的|有什么)?(关系|联系|关联)"),
    "sequence":        re.compile(r"(先后|顺序|次第|先修|后修|第一步|步骤|先学什么)"),
    "story":           re.compile(r"(公案|故事|典故|传说|讲一个|有没有.{0,4}故事)"),
    "chanting":        re.compile(r"(全文|怎么念|怎么诵|念诵方法|回向文|发愿文|仪轨文)"),
    # --- 修行实操类 ---
    "advice":          re.compile(r"(适不适合|应该选|该不该|可不可以|能不能学|好不好|我该|我应该|先学|适合.{0,4}吗)"),
    "practice_detail": re.compile(r"(腿疼|昏沉|散乱|走神|杂念|念多少|多长时间|坐多久|什么姿势|注意什么)"),
    "difficulty":      re.compile(r"(没有进步|没进步|退步|退转|修不下去|坚持不了|总是犯|做不到|很难)"),
    "experience":      re.compile(r"(看到光|感应|梦到|梦见|流泪|发抖|身体发热|正常吗|怎么回事)"),
    "precept":         re.compile(r"(破戒|犯戒|果报|报应|能不能吃|能不能杀|戒律|开缘|持戒)"),
    # --- 生活应用类 ---
    "application":     re.compile(r"(在生活中|在工作中|日常中|怎么用|怎么应用|如何运用|落实到)"),
    "emotion":         re.compile(r"(我很痛苦|我很难过|我很烦|心里难受|想不开|很绝望|很迷茫|很焦虑|受不了|崩溃)"),
    "taboo":           re.compile(r"(禁忌|不能做|忌讳|注意事项|有什么讲究|不可以做)"),
    "misconception":   re.compile(r"(是不是迷信|是消极|是逃避|是厌世|是不是什么都不要|是忍气吞声吗)"),
    # --- 现代/交互类 ---
    "contemporary":    re.compile(r"(佛教如何看待.{0,4}(AI|科技|环保|堕胎|安乐死|同性|克隆)|现代.{0,4}佛教)"),
    "verification":    re.compile(r"(对不对|正确吗|对吗|理解得对|这样理解|说得对|有没有错)"),
    "followup":        re.compile(r"(详细说说|展开讲|再说说|具体说|能再解释|多说一些|还有吗|继续)"),
    "recommendation":  re.compile(r"(推荐.{0,4}书|推荐.{0,4}经|看什么书|读什么|入门书|学什么好)"),
    "etiquette":       re.compile(r"(怎么称呼|怎么拜|怎么上香|穿什么|寺院.{0,4}规矩|礼仪)"),
    "calendar":        re.compile(r"(哪天|几月几号|什么时候|日期|佛诞|圣诞日|节日.*日期)"),
    "meta":            re.compile(r"(你能|你会|你知道|你是谁|可以问什么|什么问题|你的功能)"),
}

# 问题类型 → 搜索策略
STRATEGY = {
    # 知识查询类
    "definition":      {"skip_faq": False, "prefer_lecture": False},
    "who":             {"skip_faq": False, "prefer_lecture": False},
    "which":           {"skip_faq": False, "prefer_lecture": False},
    "howmany":         {"skip_faq": False, "prefer_lecture": False},
    "overview":        {"skip_faq": False, "prefer_lecture": False},
    "reference":       {"skip_faq": False, "prefer_lecture": True},
    "method":          {"skip_faq": True,  "prefer_lecture": True},
    "reason":          {"skip_faq": True,  "prefer_lecture": True},
    "list":            {"skip_faq": True,  "prefer_lecture": True, "boost_top_k": 20},
    "comparison":      {"skip_faq": True,  "prefer_lecture": False, "split_query": True},
    "debate":          {"skip_faq": True,  "prefer_lecture": True},
    "relation":        {"skip_faq": True,  "prefer_lecture": True},
    "sequence":        {"skip_faq": True,  "prefer_lecture": True},
    "story":           {"skip_faq": True,  "prefer_lecture": True},
    "chanting":        {"skip_faq": False, "prefer_lecture": False},
    "verse":           {"skip_faq": True,  "prefer_lecture": True},
    # 修行实操类
    "advice":          {"skip_faq": True,  "prefer_lecture": True},
    "practice_detail": {"skip_faq": True,  "prefer_lecture": True},
    "difficulty":      {"skip_faq": True,  "prefer_lecture": True},
    "experience":      {"skip_faq": True,  "prefer_lecture": True},
    "precept":         {"skip_faq": False, "prefer_lecture": True},
    # 生活应用类
    "application":     {"skip_faq": False, "prefer_lecture": False},
    "emotion":         {"skip_faq": True,  "prefer_lecture": False},
    "taboo":           {"skip_faq": False, "prefer_lecture": False},
    "misconception":   {"skip_faq": True,  "prefer_lecture": True},
    # 现代/交互类
    "contemporary":    {"skip_faq": True,  "prefer_lecture": True},
    "verification":    {"skip_faq": True,  "prefer_lecture": True},
    "followup":        {"skip_faq": True,  "prefer_lecture": True},
    "recommendation":  {"skip_faq": False, "prefer_lecture": False},
    "etiquette":       {"skip_faq": False, "prefer_lecture": False},
    "calendar":        {"skip_faq": False, "prefer_lecture": False},
    "meta":            {"skip_faq": False, "prefer_lecture": False},
}

DEFAULT_STRATEGY = {"skip_faq": False, "prefer_lecture": False}


def classify(question: str) -> Dict[str, Any]:
    """分类用户问题，返回问题类型和搜索策略。

    Returns:
        {
            "question_type": "method" | "definition" | "comparison" | ...,
            "strategy": {"skip_faq": bool, "prefer_lecture": bool, ...},
            "comparison_concepts": ("A", "B") | None,
        }
    """
    q = (question or "").strip()

    # 比较类优先检测（有特殊处理）
    comp_match = _PATTERNS["comparison"].search(q)
    if comp_match:
        concept_a = comp_match.group(1).strip()
        concept_b = comp_match.group(3).strip()
        concept_b = re.sub(r"(有什么|有何|的)$", "", concept_b).strip()
        return {
            "question_type": "comparison",
            "strategy": STRATEGY["comparison"],
            "comparison_concepts": (concept_a, concept_b),
        }

    # 按优先级检测其他类型
    for qtype, pattern in _PATTERNS.items():
        if qtype == "comparison":
            continue
        if pattern.search(q):
            return {
                "question_type": qtype,
                "strategy": STRATEGY.get(qtype, DEFAULT_STRATEGY),
                "comparison_concepts": None,
            }

    return {
        "question_type": "unknown",
        "strategy": DEFAULT_STRATEGY,
        "comparison_concepts": None,
    }
