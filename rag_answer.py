import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from rag_runtime_config import (
    KNOWLEDGE_DIR, STORE_ROOT, OUT_PATH, DEFAULT_MODE, DEFAULT_TOP_K,
    USE_OPENAI, OPENAI_MODEL, DEBUG, QUESTION_ROUTES, SECTION_RULES,
    PRODUCT_ALIASES, PROJECT_ALIASES, VECTOR_TOP_K, KEYWORD_TOP_K,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, FAQ_KEYWORD_MAP,
    SCORE_THRESHOLD, QUESTION_TYPE_CONFIG, ANSWER_MODE_CONFIG,
    DEFAULT_PRODUCT, USE_RERANK, RERANK_MODEL, RERANK_TOP_K,
    RERANK_SCORE_THRESHOLD,
)
from search_utils import (
    normalize_text, normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms, match_faq
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer
from rag_logger import log_qa, log_error

_model = None
_faiss = None
_store_cache = {}  # 缓存已加载的 index + docs
_store_mtime = {}  # 缓存文件修改时间，用于自动失效
_reranker = None   # cross-encoder reranker 模型缓存


def get_reranker():
    """懒加载 cross-encoder reranker 模型"""
    global _reranker
    if _reranker is None:
        if not USE_RERANK:
            return None
        try:
            from sentence_transformers import CrossEncoder
            if DEBUG:
                print(f"[INFO] 加载 reranker 模型：{RERANK_MODEL}")
            _reranker = CrossEncoder(RERANK_MODEL, max_length=1024)
        except Exception as e:
            log_error("reranker_load", repr(e))
            if DEBUG:
                print(f"[WARN] Reranker 加载失败，回退到无 rerank 模式: {e}")
            return None
    return _reranker


def rerank_hits(query: str, hits: List[Dict], top_k: int = None,
                score_threshold: float = None) -> List[Dict]:
    """
    使用 cross-encoder 对检索结果做精排。

    Cross-encoder 与 bi-encoder 的区别：
    - Bi-encoder（向量搜索）：分别编码 query 和 doc，速度快但精度有限
    - Cross-encoder（rerank）：同时编码 query+doc，精度更高但速度慢

    因此流程是：bi-encoder 粗召回 → cross-encoder 精排 → 取 top-K

    Args:
        query: 用户问题
        hits: hybrid search + filter + dedup 后的候选结果
        top_k: 精排后保留的最大数量
        score_threshold: 精排分数低于此值的丢弃
    Returns:
        重排序后的 hits 列表
    """
    if not hits:
        return hits
    if top_k is None:
        top_k = RERANK_TOP_K
    if score_threshold is None:
        score_threshold = RERANK_SCORE_THRESHOLD

    reranker = get_reranker()
    if reranker is None:
        # reranker 不可用时，保持原始排序，截断到 top_k
        return hits[:top_k]

    try:
        # 构建 query-doc 对
        pairs = []
        for h in hits:
            text = h.get("text", "").strip()
            if not text:
                text = "(empty)"
            pairs.append((query, text))

        # cross-encoder 打分
        scores = reranker.predict(pairs, show_progress_bar=False)

        # 将 rerank 分数写入 hit
        for h, score in zip(hits, scores):
            h["rerank_score"] = float(score)

        # 按 rerank 分数降序排列
        hits_sorted = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # 过滤低分 + 截断
        result = [h for h in hits_sorted if h.get("rerank_score", 0.0) >= score_threshold]
        return result[:top_k]
    except Exception as e:
        log_error("rerank_hits", repr(e))
        if DEBUG:
            print(f"[WARN] Rerank 执行失败，回退到原始排序: {e}")
        return hits[:top_k]


def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_mod
        _faiss = _faiss_mod
    return _faiss


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model


def save_answer(text: str):
    """保存答案到 answer.txt（与原始 RAG 架构一致）"""
    OUT_PATH.write_text((text or "").strip() + "\n", encoding="utf-8-sig")


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([text], normalize_embeddings=False)
    vec = np.asarray(vec, dtype="float32")
    get_faiss().normalize_L2(vec)
    return vec


def load_store(product: str):
    """加载向量索引和文档，带基于 mtime 的缓存失效"""
    store_dir = STORE_ROOT / product
    index_path = store_dir / "index.faiss"
    docs_path = store_dir / "docs.jsonl"
    if not index_path.exists() or not docs_path.exists():
        _store_cache[product] = (None, [])
        _store_mtime.pop(product, None)
        return _store_cache[product]
    current_mtime = (index_path.stat().st_mtime, docs_path.stat().st_mtime)
    if product in _store_cache and _store_mtime.get(product) == current_mtime:
        return _store_cache[product]
    if product in _store_cache and DEBUG:
        print(f"[DEBUG] 重新加载 {product} 索引（文件已更新）")
    index = get_faiss().read_index(str(index_path))
    docs = []
    with docs_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                doc = json.loads(line)
                meta = doc.get("meta", {})
                if not meta.get("chunk_id"):
                    meta["chunk_id"] = f"_doc{i}"
                    doc["meta"] = meta
                docs.append(doc)
    result = (index, docs)
    _store_cache[product] = result
    _store_mtime[product] = current_mtime
    return result


def vector_search(product: str, query: str, top_k: int) -> List[Dict]:
    index, docs = load_store(product)
    if index is None or not docs:
        return []
    qv = embed_query(query)
    if qv.shape[1] != index.d:
        if DEBUG:
            print(f"[WARN] 向量维度不匹配：查询={qv.shape[1]}, 索引={index.d}")
        return []
    scores, ids = index.search(qv, min(top_k, len(docs)))
    hits = []
    for i, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(docs):
            continue
        d = dict(docs[idx])
        d["score"] = float(scores[0][i])
        hits.append(d)
    return hits


def read_knowledge_file(product: str, fname: str) -> str:
    p = KNOWLEDGE_DIR / product / fname
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def detect_product(question: str) -> str:
    found = detect_terms(question, PRODUCT_ALIASES)
    if found:
        return found[0]
    if (KNOWLEDGE_DIR / DEFAULT_PRODUCT).exists():
        return DEFAULT_PRODUCT
    dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()] if KNOWLEDGE_DIR.exists() else []
    return dirs[0] if dirs else DEFAULT_PRODUCT


def detect_route(question: str) -> str:
    """多类评分路由：按匹配关键词的总字符长度评分，长词匹配权重更高"""
    q = (question or "").lower()
    route_scores = {}
    for route_name, keywords in QUESTION_ROUTES.items():
        score = sum(len(kw) for kw in keywords if kw.lower() in q)
        if score > 0:
            route_scores[route_name] = score
    if not route_scores:
        return "basic"
    return max(route_scores.items(), key=lambda x: x[1])[0]


def _get_route_config(route: str) -> Dict:
    """获取路由级别的检索配置（k 和 threshold），使用 QUESTION_TYPE_CONFIG"""
    return QUESTION_TYPE_CONFIG.get(route, {"k": DEFAULT_TOP_K, "threshold": SCORE_THRESHOLD})


def _get_mode_limit(route: str, mode: str) -> int:
    """获取路由在指定模式下的最大输出条目数，使用 ANSWER_MODE_CONFIG"""
    mode_cfg = ANSWER_MODE_CONFIG.get(mode, ANSWER_MODE_CONFIG["brief"])
    return mode_cfg.get(route, mode_cfg.get("max_items_default", 8))


def _suggest_related_topics(question: str, route: str) -> List[str]:
    """根据问题和路由，推荐知识库中已有的相近主题"""
    q = question.strip()
    suggestions = []
    route_kws = QUESTION_ROUTES.get(route, [])
    for kw in route_kws:
        if any(ch in q for ch in kw if ch not in "的是了在"):
            suggestions.append(kw)
    if len(suggestions) < 3:
        for r, kws in QUESTION_ROUTES.items():
            if r == route:
                continue
            for kw in kws:
                if any(ch in q for ch in kw if ch not in "的是了在"):
                    suggestions.append(kw)
                    if len(suggestions) >= 5:
                        break
            if len(suggestions) >= 5:
                break
    if len(suggestions) < 3:
        for kw in FAQ_KEYWORD_MAP:
            if any(ch in q for ch in kw if ch not in "的是了在"):
                suggestions.append(kw)
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= 5:
            break
    return unique


def _build_not_found_response(question: str, route: str, rewrite: dict) -> str:
    """构建未命中时的友好回复，附带相关主题建议"""
    suggestions = _suggest_related_topics(question, route)
    fallback = ["当前知识库未找到与该问题直接相关的内容。"]
    if suggestions:
        fallback.append("您可以尝试以下相关主题：" + "、".join(suggestions))
    else:
        fallback.append("建议尝试更具体的佛教术语进行提问，如：四圣谛、八正道、菩提心、入行论等。")
    fallback.append("如需深入了解，建议查阅相关佛教经典或咨询法师。")
    return format_structured_answer(route, fallback, [], add_risk_note=False)


def build_evidence(hits: List[Dict]) -> List[Dict]:
    ev = []
    for h in hits[:6]:
        meta = h.get("meta", {})
        entry = {"meta": meta}
        if meta.get("kepan_breadcrumb"):
            entry["kepan_breadcrumb"] = meta["kepan_breadcrumb"]
        ev.append(entry)
    return ev


def filter_by_score(hits: List[Dict], threshold: float = None) -> List[Dict]:
    """过滤低于分数阈值的检索结果。
    使用分通道阈值：任一通道超过阈值即保留，避免 keyword-only hits 被加权后误杀。
    """
    if threshold is None:
        threshold = SCORE_THRESHOLD
    result = []
    for h in hits:
        hybrid = h.get("hybrid_score", 0.0)
        vec_score = float(h.get("score", 0.0))
        kw_score = float(h.get("keyword_score", 0.0))
        if hybrid >= threshold or vec_score >= threshold or kw_score >= threshold:
            result.append(h)
    return result


def _text_overlap_ratio(a: str, b: str) -> float:
    """计算两段文本的字符重叠率（基于较短文本）"""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    n = 4
    if len(shorter) < n:
        return 1.0 if shorter in longer else 0.0
    ngrams_short = set(shorter[i:i+n] for i in range(len(shorter) - n + 1))
    ngrams_long = set(longer[i:i+n] for i in range(len(longer) - n + 1))
    if not ngrams_short:
        return 0.0
    return len(ngrams_short & ngrams_long) / len(ngrams_short)


def _deduplicate_hits(hits: List[Dict], base_overlap_threshold: float = 0.7,
                      max_per_source: int = 3) -> List[Dict]:
    """语义去重 + 来源多样性控制"""
    if not hits:
        return []
    selected = []
    source_counts: Dict[str, int] = {}
    for h in hits:
        text = h.get("text", "").strip()
        if not text:
            continue
        source = h.get("meta", {}).get("source_file", "_unknown")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        is_dup = False
        for sel in selected:
            sel_text = sel.get("text", "")
            min_len = min(len(text), len(sel_text))
            threshold = base_overlap_threshold + 0.15 * max(0, 1 - min_len / 200)
            if _text_overlap_ratio(text, sel_text) > threshold:
                is_dup = True
                break
        if is_dup:
            continue
        selected.append(h)
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected


def extract_chunks_as_context(hits: List[Dict], max_chunks: int = 6) -> str:
    """从检索结果中提取 chunk 文本 + 来源元数据作为 LLM 上下文"""
    if not hits:
        return ""
    chunks = []
    seen = set()
    for idx, h in enumerate(hits[:max_chunks], 1):
        text = h.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        meta = h.get("meta", {})
        source = meta.get("source_file", "")
        kepan = meta.get("kepan_breadcrumb", "")
        section = meta.get("section_title", "")
        ctype = meta.get("content_type", "")
        label_parts = []
        if source:
            label_parts.append(source)
        if kepan:
            label_parts.append(f"科判：{kepan}")
        elif section:
            label_parts.append(f"章节：{section}")
        if ctype:
            label_parts.append(f"类型：{ctype}")
        label = "｜".join(label_parts) if label_parts else f"段落{idx}"
        chunks.append(f"[来源{idx}：{label}]\n{text}")
    return "\n\n---\n\n".join(chunks)


def extract_from_hits(hits: List[Dict], route: str, mode: str) -> List[str]:
    """从向量检索结果中提取答案段落"""
    if not hits:
        return []
    paragraphs = []
    seen = set()
    for h in hits:
        text = h.get("text", "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text)[:200]
        if key in seen:
            continue
        seen.add(key)
        cleaned_lines = []
        for ln in normalize_lines(text):
            clean = ln.lstrip("-").strip()
            if not clean or is_faq_line(clean):
                continue
            cleaned_lines.append(clean)
        if cleaned_lines:
            paragraphs.append("\n".join(cleaned_lines))
    limit = _get_mode_limit(route, mode)
    return paragraphs[:limit]


def parse_bullets_from_section(main_text: str, faq_text: str, route: str, mode: str) -> List[str]:
    rule = SECTION_RULES.get(route)
    if not rule:
        return []
    block = section_block(main_text, rule["titles"], rule["stops"])
    if not block:
        block = section_block(faq_text, rule["titles"], [])
    if not block:
        return []

    lines = [ln for ln in normalize_lines(block) if not is_faq_line(ln)]
    items = []
    for ln in lines:
        clean = ln.lstrip("-").strip()
        if not clean:
            continue
        eff_len = len(re.sub(r"[\s\u3000，。、！？；：""''（）【】《》\-—·]", "", clean))
        if eff_len < 8:
            continue
        if re.match(r"^\d+[）\)]", clean):
            items.append(clean)
            continue
        if len(clean) <= 120:
            items.append(clean)

    items = uniq(items)
    limit = _get_mode_limit(route, mode)
    return items[:limit]


def answer_one(question: str, mode: str) -> str:
    product = detect_product(question)
    route = detect_route(question)
    route_cfg = _get_route_config(route)
    rewrite = rewrite_query(question)

    # 1. 尝试 FAQ 精确匹配（带别名扩展，包含子目录 FAQ）
    faq_text = read_knowledge_file(product, "faq.txt")
    pdir = KNOWLEDGE_DIR / product
    if pdir.exists():
        for fp in sorted(pdir.rglob("faq*.txt")):
            if fp.name == "faq.txt":
                continue
            sub_faq = fp.read_text(encoding="utf-8", errors="replace")
            if sub_faq.strip():
                faq_text = faq_text + "\n" + sub_faq
    alias_text = read_knowledge_file(product, "alias.txt")
    faq_answer = match_faq(question, faq_text, FAQ_KEYWORD_MAP, alias_text)
    if faq_answer:
        faq_evidence = [{"meta": {
            "source_file": "faq.txt",
            "source_type": "faq",
            "chunk_id": "faq_match",
        }}]
        return format_structured_answer(route, [faq_answer], faq_evidence, add_risk_note=False)

    # 2. 向量 + 关键词混合检索（使用路由配置的 top_k）
    route_k = route_cfg.get("k", DEFAULT_TOP_K)
    vector_hits = vector_search(product, question, max(route_k, VECTOR_TOP_K))
    _, docs = load_store(product)
    keyword_hits = keyword_search(rewrite["expanded"], docs, max(route_k, KEYWORD_TOP_K)) if docs else []
    hits = merge_hybrid(vector_hits, keyword_hits, HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, route_k) if (vector_hits or keyword_hits) else []

    # 过滤低分结果（使用路由配置的 threshold）
    route_threshold = route_cfg.get("threshold", SCORE_THRESHOLD)
    hits = filter_by_score(hits, route_threshold)
    hits = _deduplicate_hits(hits)

    # 3. Rerank：cross-encoder 精排（粗召回后用更精确的模型重排序）
    if USE_RERANK and hits:
        hits = rerank_hits(question, hits, top_k=RERANK_TOP_K)

    # 置信度检测
    low_confidence = False
    if hits:
        best_vec = max((float(h.get("score", 0.0)) for h in hits), default=0.0)
        best_kw = max((float(h.get("keyword_score", 0.0)) for h in hits), default=0.0)
        if best_vec < 0.45 and best_kw < 0.3:
            low_confidence = True

    if not hits:
        return _build_not_found_response(question, route, rewrite)

    # 4. 优先使用 LLM 基于检索上下文生成答案（真正的 RAG）
    context = extract_chunks_as_context(hits, max_chunks=6)
    llm_answer = openai_rag_generate(question, context, route, low_confidence=low_confidence)
    if llm_answer and len(llm_answer.strip()) >= 15:
        q_stripped = question.strip()
        a_stripped = llm_answer.strip()
        if len(q_stripped) >= 10 and len(a_stripped) < len(q_stripped) * 3:
            echo_ratio = _text_overlap_ratio(q_stripped, a_stripped[:len(q_stripped) * 3])
            if echo_ratio > 0.85:
                llm_answer = ""
    if llm_answer and len(llm_answer.strip()) >= 15:
        evidence = build_evidence(hits)
        return format_structured_answer(route, [llm_answer.strip()], evidence, add_risk_note=(route == "practice"))

    # 5. 无 LLM 时：向量检索结果直接作为答案段落
    body_lines = extract_from_hits(hits, route, mode)
    body_lines = [
        p for p in body_lines
        if len(re.sub(r"[\s\u3000，。、！？；：""''（）【】《》]", "", p)) >= 15
    ]

    if not body_lines:
        main_text = read_knowledge_file(product, "main.txt")
        body_lines = parse_bullets_from_section(main_text, faq_text, route, mode)

    if not body_lines:
        return _build_not_found_response(question, route, rewrite)

    text = format_structured_answer(route, body_lines, build_evidence(hits), add_risk_note=(route == "practice"))
    return text


def openai_rag_generate(question: str, context: str, route: str, low_confidence: bool = False) -> str:
    """真正的 RAG：将检索到的知识库内容作为上下文，让 LLM 生成准确答案"""
    if not USE_OPENAI:
        return ""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return ""
    if not context.strip():
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        system_prompt = (
            "你是一个佛教知识问答助手。请基于【参考资料】回答用户的问题。\n\n"
            "## 回答规则\n"
            "1. **以参考资料为主**：优先使用参考资料中的内容。如果你具备的佛学常识可以补充说明，"
            "可简要添加，但必须标注「（补充说明）」以区分\n"
            "2. **部分可答则答**：如果资料只能回答问题的一部分，先回答能回答的部分，"
            "然后注明「关于XX部分，现有资料未涉及」\n"
            "3. **资料不相关时坦诚说明**：如果参考资料与用户问题明显不相关或无法回答该问题，"
            "请直接说明「现有资料库未收录该主题的相关内容」，不要强行从不相关资料中拼凑答案\n"
            "4. 回答要条理清晰，使用分点或分段组织\n"
            "5. 如参考资料中有经典原文，引用时用「」括起\n\n"
            "## 来源标注\n"
            "- 参考资料标记为 [来源1：...]、[来源2：...] 等\n"
            "- 在回答中引用具体内容后，用 [来源N] 标注，N 为对应编号\n"
            "- 如果多个来源说法不同，分别列出并标注各自来源\n\n"
            "## 格式\n"
            "- 回答末尾加上：「以上内容基于佛教经典与传统教义整理，仅供学习参考。」\n"
        )
        if low_confidence:
            system_prompt += (
                "\n## 重要提醒\n"
                "本次检索的参考资料与用户问题的相关度较低。请你仔细判断资料是否真正回答了问题。"
                "如果资料内容与问题无关，请回答：「该问题超出了当前知识库的覆盖范围，"
                "建议查阅相关佛教经典或咨询法师获取更准确的解答。」\n"
            )

        user_prompt = (
            f"【参考资料】\n{context}\n\n"
            f"【用户问题】\n{question}\n\n"
            "请基于以上参考资料回答问题。"
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=1500,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        if answer and getattr(choice, "finish_reason", None) == "length":
            answer += "\n\n（注：回答因长度限制被截断，如需完整内容请缩小问题范围。）"
        return answer if answer else ""
    except Exception as e:
        log_error("openai_rag_generate", repr(e))
        if DEBUG:
            print(f"[DEBUG] OpenAI RAG generation failed: {e}")
        return ""


def answer_question(question: str, mode: str) -> str:
    """主入口：回答问题，返回字符串"""
    if not (question or "").strip():
        return "请输入您想了解的佛教问题。"
    question = question.strip()[:500]
    rewrite = rewrite_query(question)
    outputs = []
    seen = set()
    for subq in rewrite["sub_questions"][:4]:
        subq_key = subq.strip()
        if subq_key in seen:
            continue
        seen.add(subq_key)
        try:
            ans = answer_one(subq, mode)
        except Exception as e:
            log_error("answer_one", repr(e), meta={"question": subq})
            ans = ""
        if ans and ans.strip():
            outputs.append(ans)
    return "\n\n".join(outputs)


def main():
    if len(sys.argv) < 2:
        print('Usage: python rag_answer.py "你的佛教问题" [brief|full]')
        return
    question = sys.argv[1].strip()
    mode = DEFAULT_MODE
    if len(sys.argv) >= 4 and sys.argv[3].strip() in ("brief", "full"):
        mode = sys.argv[3].strip()
    elif len(sys.argv) >= 3 and sys.argv[2].strip() in ("brief", "full"):
        mode = sys.argv[2].strip()

    ans = answer_question(question, mode)
    save_answer(ans)
    print(ans)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_msg = "ERROR: " + repr(e)
        save_answer(err_msg)
        print(err_msg)
