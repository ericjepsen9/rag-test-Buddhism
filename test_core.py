"""核心算法单元测试：BM25、路由检测、上下文补全、工具函数"""
import pytest
from search_utils import (
    _extract_terms, _extract_terms_bigram, bm25_score, normalize_text,
    normalize_lines, uniq, section_block, split_multi_question,
    keyword_search, merge_hybrid, detect_terms,
    rerank_hits, compute_dynamic_threshold,
)
from query_rewrite import (
    rewrite_query, _resolve_context, _extract_history_context,
)
from rag_answer import (
    detect_route, detect_product, build_evidence, _truncate_to_sentence,
    _build_context,
)


# ============================================================
# search_utils 单元测试
# ============================================================

class TestCountTerm:
    """测试字符串计数（str.count 基本功能验证）"""
    def test_basic(self):
        assert "ababab".count("ab") == 3

    def test_no_match(self):
        assert "hello world".count("xyz") == 0

    def test_empty_term(self):
        assert "anything".count("") == len("anything") + 1

    def test_empty_text(self):
        assert "".count("a") == 0

    def test_chinese(self):
        assert "般若波罗蜜多心经，般若波罗蜜多".count("般若波罗蜜多") == 2

    def test_single_char(self):
        assert "banana".count("a") == 3


class TestExtractTerms:
    def test_basic(self):
        terms = _extract_terms("四圣谛 教义")
        assert "四圣谛" in terms
        assert "教义" in terms

    def test_bigram_split(self):
        terms = _extract_terms_bigram("禅修方法")
        assert "禅修" in terms
        assert "修方" in terms
        assert "方法" in terms

    def test_dedup(self):
        terms = _extract_terms("教义 教义")
        assert terms.count("教义") == 1

    def test_punctuation_split(self):
        terms = _extract_terms("空性，般若？涅槃")
        assert "空性" in terms
        assert "般若" in terms
        assert "涅槃" in terms

    def test_jieba_buddhist_terms(self):
        """jieba 模式下应正确切分佛学术语"""
        terms = _extract_terms("般若波罗蜜多心经怎么念")
        assert "般若" in terms or "般若波罗蜜多" in terms


class TestBM25Score:
    def setup_method(self):
        self.docs = [
            "四圣谛是佛教核心教义",
            "禅修注意事项包括正确坐姿",
            "五戒包括不杀生和不偷盗",
        ]
        self.avg_dl = sum(len(d) for d in self.docs) / len(self.docs)
        self.n_docs = len(self.docs)

    def test_relevant_higher(self):
        doc_freqs = {"四圣谛": 1, "教义": 1}
        s1 = bm25_score("四圣谛", self.docs[0], self.avg_dl, self.n_docs, doc_freqs)
        s2 = bm25_score("四圣谛", self.docs[1], self.avg_dl, self.n_docs, doc_freqs)
        assert s1 > s2

    def test_zero_for_empty(self):
        assert bm25_score("", "some text", 10.0, 3, {}) == 0.0
        assert bm25_score("query", "", 10.0, 3, {}) == 0.0

    def test_positive_score(self):
        doc_freqs = {"禅修": 1}
        s = bm25_score("禅修", self.docs[1], self.avg_dl, self.n_docs, doc_freqs)
        assert s > 0


class TestNormalize:
    def test_bom_removal(self):
        assert normalize_text("\ufeffhello") == "hello"

    def test_crlf(self):
        assert normalize_text("a\r\nb") == "a\nb"

    def test_normalize_lines_empty(self):
        lines = normalize_lines("  \n---\n===\n  hello  ")
        assert lines == ["hello"]


class TestUniq:
    def test_basic(self):
        assert uniq(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_whitespace(self):
        assert uniq(["  a  ", "a", " a"]) == ["a"]

    def test_empty(self):
        assert uniq(["", None, "a"]) == ["a"]


class TestSectionBlock:
    def test_extract(self):
        text = "一、基本概念\n内容A\n二、修行方法\n内容B"
        block = section_block(text, ["一、基本概念"], ["二、修行方法"])
        assert "内容A" in block
        assert "内容B" not in block

    def test_no_match(self):
        assert section_block("some text", ["不存在的标题"], []) == ""

    def test_empty(self):
        assert section_block("", ["标题"], []) == ""


class TestSplitMultiQuestion:
    def test_question_mark(self):
        parts = split_multi_question("空性是什么？八正道有哪些？")
        assert len(parts) >= 2

    def test_and_pattern(self):
        parts = split_multi_question("禅宗和净土宗分别是什么")
        assert len(parts) == 2

    def test_no_split_short(self):
        parts = split_multi_question("学佛之后，可以吃素")
        assert len(parts) == 1


class TestDetectTerms:
    def test_alias_match(self):
        aliases = {"heart_sutra": ["心经", "般若心经", "Heart Sutra"]}
        assert detect_terms("心经怎么念", aliases) == ["heart_sutra"]

    def test_no_match(self):
        aliases = {"heart_sutra": ["心经"]}
        assert detect_terms("天气怎么样", aliases) == []


class TestKeywordSearch:
    def test_returns_sorted(self):
        docs = [
            {"text": "四圣谛是佛教核心教义，包括苦集灭道"},
            {"text": "禅修注意正确坐姿和呼吸"},
            {"text": "五戒包括不杀生"},
        ]
        results = keyword_search("四圣谛 教义", docs, top_k=3)
        assert len(results) > 0
        assert "四圣谛" in results[0]["text"]

    def test_empty_docs(self):
        assert keyword_search("query", [], top_k=5) == []

    def test_score_range(self):
        docs = [{"text": "四圣谛 苦集灭道 教义"}]
        results = keyword_search("四圣谛", docs, top_k=1)
        if results:
            assert 0.5 < results[0]["keyword_score"] < 1.0


class TestMergeHybrid:
    def test_merge(self):
        v_hits = [{"text": "A", "score": 0.8, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "A", "keyword_score": 0.6, "meta": {"source_file": "a", "chunk_id": "1"}}]
        merged = merge_hybrid(v_hits, k_hits, 0.7, 0.3, 5)
        assert len(merged) == 1
        assert merged[0]["hybrid_score"] == pytest.approx(0.8 * 0.7 + 0.6 * 0.3)

    def test_disjoint(self):
        v_hits = [{"text": "A", "score": 0.8, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "B", "keyword_score": 0.6, "meta": {"source_file": "b", "chunk_id": "2"}}]
        merged = merge_hybrid(v_hits, k_hits, 0.7, 0.3, 5)
        assert len(merged) == 2


# ============================================================
# query_rewrite 单元测试
# ============================================================

class TestResolveContext:
    def test_offtopic_no_resolve(self):
        ctx = {"product": "四圣谛", "projects": []}
        result = _resolve_context("今天天气怎么样", ctx)
        assert "四圣谛" not in result

    def test_empty_history(self):
        result = _resolve_context("空性是什么", {})
        assert result == "空性是什么"


class TestRewriteQuery:
    def test_chitchat(self):
        result = rewrite_query("你好")
        assert result["is_chitchat"] is True

    def test_normal_query(self):
        result = rewrite_query("四圣谛是什么")
        assert result["is_chitchat"] is False


# ============================================================
# rag_answer 单元测试
# ============================================================

class TestDetectRoute:
    def test_basic_fallback(self):
        assert detect_route("佛教是什么") == "basic"


class TestTruncateToSentence:
    def test_short_text(self):
        assert _truncate_to_sentence("短文本", 200) == "短文本"

    def test_sentence_boundary(self):
        text = "第一句话。第二句话。第三句话很长很长很长很长很长很长很长很长很长"
        result = _truncate_to_sentence(text, 15)
        assert result.endswith("。")

    def test_no_boundary(self):
        text = "没有句子结束符的超长文本" * 20
        result = _truncate_to_sentence(text, 50)
        assert len(result) <= 50


class TestBuildContext:
    def test_chunk_boundary(self):
        hits = [
            {"text": "A" * 100, "meta": {"source_file": "a.txt", "chunk_id": "1"}},
            {"text": "B" * 100, "meta": {"source_file": "b.txt", "chunk_id": "2"}},
        ]
        context = _build_context(hits, max_chars=150)
        assert "A" in context

    def test_empty_hits(self):
        assert _build_context([]) == ""


class TestBuildEvidence:
    def test_truncation(self):
        hits = [{"text": "x" * 600, "meta": {"source_file": "f.txt"}}]
        ev = build_evidence(hits)
        assert len(ev[0]["text"]) <= 450

    def test_sentence_aware(self):
        hits = [{"text": "第一句。第二句。" + "x" * 500, "meta": {}}]
        ev = build_evidence(hits)
        text = ev[0]["text"]
        assert text.endswith("。") or len(text) <= 450


class TestNormalizeTextUnicode:
    def test_nfc_normalization(self):
        import unicodedata
        nfd = unicodedata.normalize("NFD", "e\u0301")
        result = normalize_text(nfd)
        assert result == unicodedata.normalize("NFC", "e\u0301")

    def test_bom_removal(self):
        assert "\ufeff" not in normalize_text("\ufeffhello")

    def test_crlf(self):
        assert "\r" not in normalize_text("a\r\nb")


class TestChitchatVariations:
    def test_greeting_with_punctuation(self):
        from query_rewrite import _CHITCHAT_PATTERNS
        assert _CHITCHAT_PATTERNS.match("你好！")
        assert _CHITCHAT_PATTERNS.match("hello~")
        assert _CHITCHAT_PATTERNS.match("嗨啊")
        assert _CHITCHAT_PATTERNS.match("hi!")

    def test_thanks_with_suffix(self):
        from query_rewrite import _CHITCHAT_PATTERNS
        assert _CHITCHAT_PATTERNS.match("谢谢啊")
        assert _CHITCHAT_PATTERNS.match("好的！")

    def test_non_chitchat_not_matched(self):
        from query_rewrite import _CHITCHAT_PATTERNS
        assert not _CHITCHAT_PATTERNS.match("你好，请问四圣谛是什么")
