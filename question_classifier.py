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
    "definition":  re.compile(r"(是什么意思|是什么$|什么是|什么叫|含义|定义)"),
    "method":      re.compile(r"(如何|怎么修|怎样修|怎么断|怎么对治|怎么发|怎么念|怎么持|怎么忏)"),
    "reason":      re.compile(r"(为什么|为何|原因|有什么功德|有什么过患|有什么利益)"),
    "comparison":  re.compile(r"(.{1,12}?)(和|与|跟)(.{1,12}?)(的|有什么|有何)?(区别|不同|异同|差别|对比)"),
    "debate":      re.compile(r"(还是|是否|到底|究竟|矛盾|相违|怎么理解.{0,4}矛盾|既然.{2,10}为什么|那谁|那什么)"),
    "advice":      re.compile(r"(适不适合|应该选|该不该|可不可以|能不能学|好不好|我该|我应该|先学|适合.{0,4}吗)"),
    "list":        re.compile(r"(有哪些|哪几种|包括什么|包括哪些|列举|哪些方法)"),
    "overview":    re.compile(r"(讲了什么|讲什么|主要内容|内容是什么|概述)"),
    "who":         re.compile(r"(是谁|谁写的|谁造的|作者|哪位)"),
    "which":       re.compile(r"(属于什么|属于哪|归于|哪个宗派|哪一派|什么宗)"),
    "verse":       re.compile(r"(这句话|这个颂词|出自哪|是什么意思.{0,4}$)"),
    "howmany":     re.compile(r"(有几|有多少|几品|几种|几个)"),
}

# 问题类型 → 搜索策略
STRATEGY = {
    "definition":  {"skip_faq": False, "prefer_lecture": False},
    "who":         {"skip_faq": False, "prefer_lecture": False},
    "which":       {"skip_faq": False, "prefer_lecture": False},
    "howmany":     {"skip_faq": False, "prefer_lecture": False},
    "overview":    {"skip_faq": False, "prefer_lecture": False},
    "method":      {"skip_faq": True,  "prefer_lecture": True},
    "reason":      {"skip_faq": True,  "prefer_lecture": True},
    "list":        {"skip_faq": True,  "prefer_lecture": True, "boost_top_k": 20},
    "comparison":  {"skip_faq": True,  "prefer_lecture": False, "split_query": True},
    "debate":      {"skip_faq": True,  "prefer_lecture": True},
    "advice":      {"skip_faq": True,  "prefer_lecture": True},
    "verse":       {"skip_faq": True,  "prefer_lecture": True},
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
