#!/usr/bin/env python3
"""RAG 回答质量自动化测试框架

用法:
  python test_quality.py                    # 运行所有测试
  python test_quality.py --api              # 通过 API 端到端测试
  python test_quality.py --routing-only     # 仅测试路由检测
  python test_quality.py --report report.json  # 输出 JSON 报告

每次修改 query_rewrite / rag_answer / search_utils 后，务必运行此测试。
"""

import sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# 测试用例定义
# 每个用例包含:
#   q: 用户问题
#   expect_route: 期望的主路由 (可选)
#   must_not_offtopic: True 表示不应被判为离题
#   must_not_clarify: True 表示不应触发消歧
#   expect_sources: 期望的来源关键词列表 (如 ["lecture/"] 表示应从讲记检索)
#   must_contain: 回答中必须包含的关键词
#   must_not_contain: 回答中不应包含的关键词
# ============================================================

TEST_CASES = [
    # ================================================================
    # A. 入行论基础信息 (应从 FAQ 或知识库中准确回答)
    # ================================================================
    {"q": "入行论是谁写的", "must_not_offtopic": True, "must_contain": ["寂天"], "must_not_contain": ["领导", "职场", "同事"], "category": "入行论-基础"},
    {"q": "入行论有几品", "must_not_offtopic": True, "must_contain": ["十"], "category": "入行论-基础"},
    {"q": "入行论属于哪个宗派", "must_not_offtopic": True, "must_contain": ["中观"], "category": "入行论-基础"},
    {"q": "入行论第六品讲了什么", "must_not_offtopic": True, "must_contain": ["安忍"], "category": "入行论-基础"},
    {"q": "入行论第八品讲了什么", "must_not_offtopic": True, "must_contain": ["静虑|禅定|自他"], "category": "入行论-基础"},
    {"q": "入行论第九品讲了什么", "must_not_offtopic": True, "must_contain": ["智慧|空性|般若"], "category": "入行论-基础"},
    {"q": "入行论引用了哪些论典", "must_not_offtopic": True, "must_contain": ["中论"], "category": "入行论-基础"},
    {"q": "学习入行论有什么次第", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-基础"},

    # ================================================================
    # B. 入行论十品核心内容 — 每品至少一个问题
    # ================================================================
    # 第一品 菩提心利益
    {"q": "菩提心有什么功德", "must_not_offtopic": True, "must_contain": ["菩提心"], "category": "入行论-十品"},
    {"q": "暇满人身为什么难得", "must_not_offtopic": True, "must_contain": ["暇满"], "category": "入行论-十品"},
    # 第二品 忏悔罪业
    {"q": "忏悔罪业的方法有哪些", "must_not_offtopic": True, "must_contain": ["忏悔"], "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},
    {"q": "七支供养是什么", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},
    # 第三品 受持菩提心
    {"q": "菩提心的学处是什么", "must_not_offtopic": True, "must_not_clarify": True, "must_contain": ["学处", "菩提心"], "category": "入行论-十品"},
    {"q": "愿菩提心和行菩提心的区别", "must_not_offtopic": True, "must_contain": ["愿", "行"], "category": "入行论-十品"},
    # 第四品 不放逸
    {"q": "不放逸是什么意思", "must_not_offtopic": True, "must_contain": ["放逸"], "category": "入行论-十品"},
    {"q": "为什么发了菩提心后要不放逸", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},
    # 第五品 正知正念
    {"q": "什么是正知正念", "must_not_offtopic": True, "must_contain": ["正知", "正念"], "category": "入行论-十品"},
    {"q": "如何护持自己的心", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},
    # 第六品 安忍
    {"q": "嗔恨心的过患是什么", "must_not_offtopic": True, "must_contain": ["嗔"], "category": "入行论-十品"},
    {"q": "如何修安忍", "must_not_offtopic": True, "must_contain": ["安忍"], "must_not_contain": ["领导", "职场"], "category": "入行论-十品"},
    {"q": "一嗔能摧毁千劫所积聚是什么意思", "must_not_offtopic": True, "must_contain": ["嗔", "善根"], "category": "入行论-十品"},
    {"q": "为什么说怨敌是修安忍的助缘", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},
    # 第七品 精进
    {"q": "什么是精进", "must_not_offtopic": True, "must_contain": ["精进"], "category": "入行论-十品"},
    {"q": "精进的障碍有哪些", "must_not_offtopic": True, "must_contain": ["懈怠|懒惰|懒"], "category": "入行论-十品"},
    # 第八品 静虑
    {"q": "自他交换是什么", "must_not_offtopic": True, "must_contain": ["自他"], "category": "入行论-十品"},
    {"q": "怎么修自他交换", "must_not_offtopic": True, "must_contain": ["自他"], "category": "入行论-十品"},
    {"q": "入行论中提到了哪些断除贪心的方法", "must_not_offtopic": True, "must_contain": ["贪"], "must_not_contain": ["领导", "职场", "同事"], "category": "入行论-十品"},
    # 第九品 智慧
    {"q": "入行论中的二谛是什么", "must_not_offtopic": True, "must_contain": ["世俗", "胜义"], "category": "入行论-十品"},
    {"q": "中观应成派的观点是什么", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},
    {"q": "为什么说修行需要空性智慧", "must_not_offtopic": True, "must_contain": ["空性", "智慧"], "category": "入行论-十品"},
    # 第十品 回向
    {"q": "回向是什么意思", "must_not_offtopic": True, "must_contain": ["回向"], "category": "入行论-十品"},
    {"q": "为什么要把功德回向给众生", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "入行论-十品"},

    # ================================================================
    # C. 佛教基础概念 (FAQ 已覆盖)
    # ================================================================
    {"q": "什么是佛教", "must_not_offtopic": True, "must_contain": ["释迦牟尼"], "category": "佛教基础"},
    {"q": "四圣谛是什么", "must_not_offtopic": True, "must_contain": ["苦", "集", "灭", "道"], "category": "佛教基础"},
    {"q": "什么是八正道", "must_not_offtopic": True, "must_contain": ["正见"], "category": "佛教基础"},
    {"q": "什么是三宝", "must_not_offtopic": True, "must_contain": ["佛", "法", "僧"], "category": "佛教基础"},
    {"q": "什么是因果报应", "must_not_offtopic": True, "must_contain": ["因果"], "category": "佛教基础"},
    {"q": "什么是轮回", "must_not_offtopic": True, "must_contain": ["六道"], "category": "佛教基础"},
    {"q": "什么是空性", "must_not_offtopic": True, "must_contain": ["空"], "category": "佛教基础"},
    {"q": "什么是涅槃", "must_not_offtopic": True, "must_contain": ["涅槃"], "category": "佛教基础"},
    {"q": "什么是菩萨", "must_not_offtopic": True, "must_contain": ["菩萨"], "category": "佛教基础"},
    {"q": "五戒是什么", "must_not_offtopic": True, "must_contain": ["不杀生"], "category": "佛教基础"},
    {"q": "什么是十二因缘", "must_not_offtopic": True, "must_contain": ["无明"], "category": "佛教基础"},
    {"q": "心经讲什么", "must_not_offtopic": True, "must_contain": ["空"], "category": "佛教基础"},
    {"q": "金刚经讲什么", "must_not_offtopic": True, "must_contain": ["金刚经"], "category": "佛教基础"},

    # ================================================================
    # D. 修行方法类问题 (需要从讲记中检索详细内容)
    # ================================================================
    {"q": "如何断除贪心", "expect_route": "practice", "must_not_offtopic": True, "must_contain": ["贪"], "must_not_contain": ["不在我的服务范围"], "category": "修行方法"},
    {"q": "如何对治嗔恨心", "must_not_offtopic": True, "must_contain": ["嗔"], "category": "修行方法"},
    {"q": "如何发菩提心", "must_not_offtopic": True, "must_contain": ["菩提心"], "category": "修行方法"},
    {"q": "如何修忍辱", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "修行方法"},
    {"q": "如何修禅定", "must_not_offtopic": True, "must_contain": ["禅定", "定"], "category": "修行方法"},
    {"q": "布施有哪几种", "must_not_offtopic": True, "must_contain": ["布施"], "category": "修行方法"},
    {"q": "六波罗蜜是什么", "must_not_offtopic": True, "must_contain": ["布施", "持戒"], "category": "修行方法"},
    {"q": "在家居士如何修行", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "修行方法"},

    # ================================================================
    # E. 颂词理解类 (检索颂词+讲解)
    # ================================================================
    {"q": "善逝法身佛子伴是什么意思", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "颂词理解"},
    {"q": "暇满人身极难得这句话出自哪里", "must_not_offtopic": True, "must_contain": ["入行论", "暇满"], "category": "颂词理解"},
    {"q": "身心若远离散乱即不生是什么意思", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "颂词理解"},
    {"q": "若久修空性必断实有习是什么意思", "must_not_offtopic": True, "must_contain": ["空性"], "category": "颂词理解"},

    # ================================================================
    # F. 短查询 (不应误判离题)
    # ================================================================
    {"q": "暇满人身", "must_not_offtopic": True, "must_contain": ["暇满"], "category": "短查询"},
    {"q": "暇满", "must_not_offtopic": True, "must_contain": ["暇"], "category": "短查询"},
    {"q": "精进", "must_not_offtopic": True, "category": "短查询"},
    {"q": "忏悔", "must_not_offtopic": True, "category": "短查询"},
    {"q": "贪心", "must_not_offtopic": True, "category": "短查询"},
    {"q": "嗔恨", "must_not_offtopic": True, "category": "短查询"},
    {"q": "菩提心", "must_not_offtopic": True, "category": "短查询"},
    {"q": "空性", "must_not_offtopic": True, "category": "短查询"},
    {"q": "回向", "must_not_offtopic": True, "category": "短查询"},
    {"q": "安忍", "must_not_offtopic": True, "category": "短查询"},
    {"q": "无常", "must_not_offtopic": True, "category": "短查询"},
    {"q": "因果", "must_not_offtopic": True, "category": "短查询"},

    # ================================================================
    # G. 跨品综合问题
    # ================================================================
    {"q": "入行论的核心思想是什么", "must_not_offtopic": True, "must_contain": ["菩提心"], "category": "综合"},
    {"q": "入行论中六度是怎么讲的", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "综合"},
    {"q": "入行论对修行人最重要的教言是什么", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "综合"},
    {"q": "寂天菩萨如何论述空性与大悲的关系", "must_not_offtopic": True, "must_contain": ["空性|般若|智慧"], "category": "综合"},

    # ================================================================
    # H. 易混淆/边界情况
    # ================================================================
    {"q": "入行论和入中论有什么区别", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "边界"},
    {"q": "寂天菩萨和龙树菩萨的关系", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "边界"},
    {"q": "佛教和其他宗教有什么不同", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "边界"},
    {"q": "大乘和小乘有什么区别", "must_not_offtopic": True, "must_contain": ["大乘", "小乘"], "category": "边界"},

    # ================================================================
    # J. 佛法与生活 (应返回生活应用层内容)
    # ================================================================
    {"q": "工作压力大怎么用佛法调节", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "佛法与生活"},
    {"q": "如何用佛法面对焦虑", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "佛法与生活"},
    {"q": "被同事冤枉了怎么办", "must_not_offtopic": True, "must_contain": ["安忍|忍辱|忍耐"], "category": "佛法与生活"},
    {"q": "孩子不听话怎么从佛法角度看", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "佛法与生活"},
    {"q": "如何在生活中修行", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "佛法与生活"},
    {"q": "佛教怎么看待财富", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "佛法与生活"},
    {"q": "亲人去世了怎么面对", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "佛法与生活"},
    {"q": "总是拖延懈怠怎么办", "must_not_offtopic": True, "must_contain": ["精进|不放逸|懈怠"], "category": "佛法与生活"},

    # ================================================================
    # K. 心理学与佛法 (应能回答，不误判离题)
    # ================================================================
    {"q": "佛教如何看待抑郁症", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "心理与佛法"},
    {"q": "正念冥想和佛教禅修有什么关系", "must_not_offtopic": True, "must_contain": ["正念|禅修"], "category": "心理与佛法"},
    {"q": "佛法如何帮助控制情绪", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "心理与佛法"},
    {"q": "佛教的慈悲心和心理学的共情有什么关系", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "心理与佛法"},

    # ================================================================
    # L1. 多观点/辩论类问题 (应呈现多个角度)
    # ================================================================
    {"q": "菩提心是心王还是心所", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "多观点辩论"},
    {"q": "佛教说无我，那谁在轮回", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "多观点辩论"},
    {"q": "既然一切皆空，为什么还要修行", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "多观点辩论"},
    {"q": "中观和唯识对空性的理解有什么不同", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "多观点辩论"},

    # ================================================================
    # L2. 个人建议/选择类问题 (应给出原则而非断言)
    # ================================================================
    {"q": "我适不适合学习密宗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "个人建议"},
    {"q": "如何选择上师", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "个人建议"},
    {"q": "初学者应该先学哪部经", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "个人建议"},
    {"q": "我应该修净土还是禅宗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "个人建议"},
    {"q": "学佛后可以吃肉吗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "个人建议"},
    {"q": "在家人可以受菩萨戒吗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "个人建议"},

    # ================================================================
    # L3. 多轮对话模拟 (上下文理解)
    # ================================================================
    {"q": "入行论第六品讲的是什么", "must_not_offtopic": True, "must_contain": ["安忍"], "category": "多轮对话"},
    # 以下问题模拟追问（无历史上下文时应独立回答）
    {"q": "那第七品呢", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "多轮对话"},
    {"q": "寂天菩萨是哪个时代的", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "多轮对话"},

    # ================================================================
    # N. 修行实操/困难/体验/戒律
    # ================================================================
    {"q": "打坐时腿疼怎么办", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "修行实操"},
    {"q": "修了很久感觉没有进步怎么办", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "修行实操"},
    {"q": "打坐时看到光是怎么回事", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "修行实操"},
    {"q": "破戒了怎么办", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "修行实操"},

    # ================================================================
    # O. 生活应用/情绪/禁忌/误解澄清
    # ================================================================
    {"q": "安忍在生活中怎么用", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "应用与误解"},
    {"q": "我很痛苦想不开怎么办", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "应用与误解"},
    {"q": "学佛有什么禁忌", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "应用与误解"},
    {"q": "佛教是迷信吗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "应用与误解"},

    # ================================================================
    # P. 现代议题/推荐/出处/次第/念诵
    # ================================================================
    {"q": "佛教如何看待AI", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "扩展问题"},
    {"q": "推荐几本佛教入门书", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "扩展问题"},
    {"q": "修行的先后顺序是什么", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "扩展问题"},
    {"q": "五蕴和十二因缘有什么关系", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "扩展问题"},
    {"q": "回向文怎么念", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "扩展问题"},

    # ================================================================
    # M. 容易混淆的问题
    # ================================================================
    {"q": "空和无有什么区别", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "易混淆"},
    {"q": "佛教说的忍辱是忍气吞声吗", "must_not_offtopic": True, "must_contain": ["安忍|忍辱"], "category": "易混淆"},
    {"q": "放下执著是不是什么都不要了", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "易混淆"},
    {"q": "佛教是消极避世的吗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "易混淆"},

    # ================================================================
    # Q. 安全/纠正/梵文/复合问题
    # ================================================================
    {"q": "我不想活了怎么办", "must_not_offtopic": True, "must_contain": ["生命|珍贵|热线|心理"], "category": "安全"},
    {"q": "什么是sunyata", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "梵文"},
    {"q": "prajna和wisdom一样吗", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "梵文"},
    {"q": "karma是什么意思", "must_not_offtopic": True, "must_not_contain": ["不在我的服务范围"], "category": "梵文"},

    # ================================================================
    # I. 应被拒绝的离题问题
    # ================================================================
    {"q": "今天天气怎么样", "expect_offtopic": True, "category": "离题"},
    {"q": "Python怎么写", "expect_offtopic": True, "category": "离题"},
    {"q": "股票怎么买", "expect_offtopic": True, "category": "离题"},
    {"q": "推荐一部电影", "expect_offtopic": True, "category": "离题"},
]


def test_routing():
    """测试路由检测层：offtopic/clarify/route 判断是否正确"""
    from query_rewrite import rewrite_query

    results = []
    passed = 0
    failed = 0

    for tc in TEST_CASES:
        q = tc["q"]
        rw = rewrite_query(q)

        is_offtopic = rw.get("is_offtopic", False)
        needs_clarify = rw.get("needs_clarification", False)
        routes = rw.get("detected_routes", [])
        primary_route = routes[0] if routes else "none"

        errors = []

        # Check offtopic
        if tc.get("must_not_offtopic") and is_offtopic:
            errors.append(f"误判为离题")
        if tc.get("expect_offtopic") and not is_offtopic:
            errors.append(f"应该被判为离题但未被拦截")

        # Check clarification
        if tc.get("must_not_clarify") and needs_clarify:
            errors.append(f"不应触发消歧但触发了")

        # Check route
        if tc.get("expect_route") and primary_route != tc["expect_route"]:
            errors.append(f"路由={primary_route}, 期望={tc['expect_route']}")

        status = "PASS" if not errors else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        results.append({
            "q": q,
            "status": status,
            "errors": errors,
            "route": primary_route,
            "offtopic": is_offtopic,
            "clarify": needs_clarify,
            "category": tc.get("category", ""),
        })

    return results, passed, failed


def test_api(base_url="http://127.0.0.1:8080"):
    """端到端 API 测试：检查回答内容质量"""
    import urllib.request

    results = []
    passed = 0
    failed = 0

    for tc in TEST_CASES:
        if tc.get("expect_offtopic"):
            continue  # 跳过离题测试用例

        q = tc["q"]
        errors = []

        try:
            data = json.dumps({"question": q, "mode": "full"}).encode()
            req = urllib.request.Request(
                f"{base_url}/ask",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            start = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            elapsed = int((time.time() - start) * 1000)

            answer = body.get("answer", "")

            # Check must_contain (支持 "A|B" 同义词，任一命中即通过)
            for kw in tc.get("must_contain", []):
                alternatives = kw.split("|")
                if not any(alt in answer for alt in alternatives):
                    errors.append(f"回答缺少关键词 '{kw}'")

            # Check must_not_contain
            for kw in tc.get("must_not_contain", []):
                if kw in answer:
                    errors.append(f"回答包含不该有的 '{kw}'")

            # Check basic quality
            if "不在我的服务范围" in answer:
                errors.append("被误判为离题")
            if len(answer) < 30:
                errors.append(f"回答太短 ({len(answer)} 字)")

            status = "PASS" if not errors else "FAIL"
            results.append({
                "q": q, "status": status, "errors": errors,
                "ms": elapsed, "answer_len": len(answer),
                "answer_preview": answer[:100],
                "category": tc.get("category", ""),
            })

        except Exception as e:
            status = "ERROR"
            errors.append(str(e))
            results.append({
                "q": q, "status": "ERROR", "errors": errors,
                "ms": 0, "answer_len": 0,
                "category": tc.get("category", ""),
            })

        if status == "PASS":
            passed += 1
        else:
            failed += 1
        time.sleep(1)  # 避免触发 429 速率限制

    return results, passed, failed


def print_results(results, passed, failed, title=""):
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f" {title} — {passed}/{total} 通过 ({100 * passed // max(total, 1)}%)")
    print(f"{'=' * 60}\n")

    # Group by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.get("category", "other")].append(r)

    for cat, items in by_cat.items():
        cat_pass = sum(1 for r in items if r["status"] == "PASS")
        print(f"  [{cat}] {cat_pass}/{len(items)}")
        for r in items:
            icon = "  " if r["status"] == "PASS" else "XX"
            line = f"    [{icon}] {r['q']}"
            if r.get("ms"):
                line += f"  ({r['ms']}ms)"
            print(line)
            if r["errors"]:
                for e in r["errors"]:
                    print(f"         -> {e}")
        print()

    if failed:
        print(f"  失败用例:")
        for r in results:
            if r["status"] != "PASS":
                print(f"    - {r['q']}: {'; '.join(r['errors'])}")


def main():
    parser = argparse.ArgumentParser(description="RAG 质量测试")
    parser.add_argument("--api", action="store_true", help="通过 API 端到端测试")
    parser.add_argument("--routing-only", action="store_true", help="仅测试路由")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="API 地址")
    parser.add_argument("--report", help="输出 JSON 报告")
    args = parser.parse_args()

    all_results = {}

    # Always run routing tests
    print("运行路由检测测试...")
    r1, p1, f1 = test_routing()
    print_results(r1, p1, f1, "路由检测测试")
    all_results["routing"] = {"results": r1, "passed": p1, "failed": f1}

    # API tests if requested
    if args.api and not args.routing_only:
        print("\n运行 API 端到端测试...")
        r2, p2, f2 = test_api(args.url)
        print_results(r2, p2, f2, "API 端到端测试")
        all_results["api"] = {"results": r2, "passed": p2, "failed": f2}

    # Save report
    if args.report:
        Path(args.report).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n报告已保存到 {args.report}")

    # Exit code
    total_failed = sum(v["failed"] for v in all_results.values())
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
