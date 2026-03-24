"""全面准确度测试：路由检测、搜索准确性、同义词、消歧、边界场景、格式化

测试维度：
1. 路由检测准确度 — 覆盖所有路由
2. 搜索工具链准确度 — BM25、向量搜索、混合排序
3. 同义词扩展准确度 — 口语映射、佛学术语
4. 查询改写准确度 — 多轮对话上下文补全
5. 多问题拆分准确度 — 复合问题、选择问题、列举问题
6. 产品/实体检测准确度 — 别名、异译
"""
import pytest
import re
from search_utils import (
    _extract_terms, _extract_terms_bigram, keyword_score_bm25,
    normalize_text, normalize_lines, uniq, section_block,
    split_multi_question, keyword_search, merge_hybrid, detect_terms,
    expand_synonyms, rerank_hits, compute_dynamic_threshold,
    _sigmoid_norm,
)
from query_rewrite import rewrite_query, _resolve_context, _extract_history_context
from rag_answer import (
    detect_route, detect_product, build_evidence, _truncate_to_sentence,
    _build_context, _detect_special_intent,
)
from rag_runtime_config import QUESTION_ROUTES, PRODUCT_ALIASES, PROJECT_ALIASES


# ============================================================
# 1. 路由检测准确度 — 全路由覆盖
# ============================================================

class TestRouteDetectionAccuracy:
    """全面测试路由检测的准确性，覆盖所有已知路由"""

    @pytest.mark.parametrize("q,expected", [
        ("佛教的基本介绍", "basic"),
        ("释迦牟尼佛是谁", "basic"),
        ("佛教入门指南", "basic"),
    ])
    def test_basic_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("四圣谛是什么", "doctrine"),
        ("八正道的内容", "doctrine"),
        ("十二因缘怎么理解", "doctrine"),
    ])
    def test_doctrine_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("空性是什么意思", "concept"),
        ("什么是因果报应", "concept"),
        ("五蕴的含义", "concept"),
        ("六道轮回是什么", "concept"),
    ])
    def test_concept_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("禅修怎么修", "practice"),
        ("念佛的方法", "practice"),
        ("打坐怎么入门", "practice"),
        ("如何持咒", "practice"),
    ])
    def test_practice_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("心经讲了什么", "scripture"),
        ("金刚经的核心思想", "scripture"),
        ("法华经的主要内容", "scripture"),
    ])
    def test_scripture_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("禅宗有什么特点", "sect"),
        ("净土宗和禅宗有什么区别", "sect"),
        ("密宗的修行特色", "sect"),
    ])
    def test_sect_route(self, q, expected):
        assert detect_route(q) == expected


class TestRouteEdgeCases:
    """路由边界场景"""

    def test_empty_question(self):
        assert detect_route("") == "basic"

    def test_none_question(self):
        assert detect_route(None) == "basic"

    def test_single_keyword_routes(self):
        """单个关键词应能正确路由"""
        assert detect_route("空性") == "concept"


# ============================================================
# 2. 搜索准确度 — BM25 + 混合排序
# ============================================================

class TestBM25Accuracy:
    """BM25 评分准确度"""

    def setup_method(self):
        self.docs = [
            "四圣谛是佛教的核心教义，包括苦谛、集谛、灭谛、道谛",
            "禅修需要注意正确的坐姿和呼吸方法",
            "五戒包括不杀生、不偷盗、不邪淫、不妄语、不饮酒",
            "般若波罗蜜多心经是大乘佛教最重要的经典之一",
            "净土宗以念佛往生为主要修行方法",
        ]
        self.avg_dl = sum(len(d) for d in self.docs) / len(self.docs)
        self.n_docs = len(self.docs)

    def test_relevant_doc_scores_highest(self):
        doc_freqs = {"四圣谛": 1, "教义": 1}
        scores = [keyword_score_bm25("四圣谛 教义", d, self.avg_dl, self.n_docs, doc_freqs)
                  for d in self.docs]
        assert scores[0] == max(scores)

    def test_precept_query_matches_precept_doc(self):
        doc_freqs = {"五戒": 1, "不杀生": 1}
        scores = [keyword_score_bm25("五戒 不杀生", d, self.avg_dl, self.n_docs, doc_freqs)
                  for d in self.docs]
        assert scores[2] == max(scores)

    def test_zero_score_for_no_match(self):
        s = keyword_score_bm25("完全不相关的词语", "四圣谛教义", 10.0, 3, {})
        assert s == 0.0


class TestKeywordSearchAccuracy:
    """keyword_search 整体准确度"""

    def test_returns_relevant_results(self):
        docs = [
            {"text": "般若波罗蜜多心经是大乘佛教重要经典"},
            {"text": "禅修需要注意坐姿和呼吸"},
            {"text": "五戒的具体内容"},
        ]
        results = keyword_search("心经 般若", docs, top_k=3)
        assert len(results) > 0
        assert "般若" in results[0]["text"] or "心经" in results[0]["text"]

    def test_empty_docs(self):
        assert keyword_search("任何查询", [], top_k=5) == []

    def test_empty_query(self):
        docs = [{"text": "some content"}]
        assert keyword_search("", docs, top_k=5) == []


class TestMergeHybridAccuracy:
    """混合搜索合并准确度"""

    def test_both_sources_contribute(self):
        v_hits = [{"text": "doc1", "score": 0.9, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "doc2", "keyword_score": 0.8, "meta": {"source_file": "b", "chunk_id": "2"}}]
        result = merge_hybrid(v_hits, k_hits, 0.6, 0.4, 10)
        assert len(result) == 2

    def test_same_doc_merges_scores(self):
        v_hits = [{"text": "same", "score": 0.8, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "same", "keyword_score": 0.7, "meta": {"source_file": "a", "chunk_id": "1"}}]
        result = merge_hybrid(v_hits, k_hits, 0.6, 0.4, 10)
        assert len(result) == 1
        expected = 0.8 * 0.6 + 0.7 * 0.4
        assert abs(result[0]["hybrid_score"] - expected) < 0.01

    def test_top_k_limit(self):
        v_hits = [{"text": f"doc{i}", "score": 0.1 * i, "meta": {"source_file": f"{i}", "chunk_id": "1"}}
                  for i in range(10)]
        result = merge_hybrid(v_hits, [], 1.0, 0.0, 3)
        assert len(result) == 3


# ============================================================
# 3. 同义词扩展准确度
# ============================================================

class TestSynonymExpansionAccuracy:
    """同义词扩展完整性和准确度"""

    def test_no_expansion_for_plain_text(self):
        """无同义词时应原样返回"""
        result = expand_synonyms("今天天气不错")
        assert result == "今天天气不错"

    def test_expansion_not_unbounded(self):
        """扩展结果不应超过2000字符"""
        long_q = "什么是般若" * 100
        result = expand_synonyms(long_q)
        assert len(result) <= 2000


# ============================================================
# 4. 查询改写准确度
# ============================================================

class TestQueryRewriteAccuracy:
    """多轮对话查询改写"""

    def test_empty_history(self):
        """无历史时应返回原始问题"""
        result = rewrite_query("四圣谛是什么", history=[])
        assert "四圣谛" in result["original"]


# ============================================================
# 5. 多问题拆分准确度
# ============================================================

class TestMultiQuestionSplitAccuracy:
    """复合问题拆分准确度"""

    def test_question_mark_split(self):
        """问号分隔的多个问题应拆分"""
        result = split_multi_question("空性是什么？八正道有哪些？")
        assert len(result) >= 2

    def test_selection_not_split(self):
        """'还是' 选择问题不应拆分"""
        result = split_multi_question("禅宗好还是净土宗好")
        assert len(result) == 1

    def test_single_question_no_split(self):
        """单个问题不应拆分"""
        result = split_multi_question("什么是四圣谛")
        assert len(result) == 1

    def test_short_comma_not_split(self):
        """短逗号分隔不应拆分"""
        result = split_multi_question("学佛之后，可以吃素")
        assert len(result) == 1


# ============================================================
# 自动导入辅助函数测试
# ============================================================

class TestAutoImportHelpers:
    """验证自动导入的类型推断和 ID 生成逻辑"""

    def test_auto_detect_scripture(self):
        from api_server import _auto_detect_type
        assert _auto_detect_type("心经讲解", "般若波罗蜜多心经全文") == "scripture"

    def test_auto_detect_practice(self):
        from api_server import _auto_detect_type
        assert _auto_detect_type("禅修入门", "打坐的基本方法") == "practice"

    def test_auto_detect_sect(self):
        from api_server import _auto_detect_type
        assert _auto_detect_type("禅宗简介", "禅宗的历史与传承") == "sect"

    def test_auto_detect_master(self):
        from api_server import _auto_detect_type
        assert _auto_detect_type("虚云老和尚", "一代高僧的生平") == "master"

    def test_auto_detect_default_doctrine(self):
        from api_server import _auto_detect_type
        assert _auto_detect_type("佛教基础", "四圣谛与八正道") == "doctrine"

    def test_title_to_id_chinese(self):
        from api_server import _title_to_id
        result = _title_to_id("心经讲解")
        assert result == "心经讲解"

    def test_title_to_id_special_chars(self):
        from api_server import _title_to_id
        result = _title_to_id("【佛学】心经 - 完整版")
        assert "_" not in result or result.replace("_", "")  # 不应全是下划线
        assert len(result) <= 50

    def test_title_to_id_empty(self):
        from api_server import _title_to_id
        result = _title_to_id("")
        assert result.startswith("auto_")
