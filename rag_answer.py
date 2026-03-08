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
    KNOWLEDGE_DIR, STORE_ROOT, DEFAULT_MODE, DEFAULT_TOP_K,
    USE_OPENAI, OPENAI_MODEL, DEBUG, QUESTION_ROUTES, SECTION_RULES,
    PRODUCT_ALIASES, PROJECT_ALIASES, VECTOR_TOP_K, KEYWORD_TOP_K,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, FAQ_KEYWORD_MAP,
    SCORE_THRESHOLD
)
from search_utils import (
    normalize_text, normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms, match_faq
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer

_model = None
_faiss = None
_store_cache = {}  # 缓存已加载的 index + docs
_store_mtime = {}  # 缓存文件修改时间，用于自动失效


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
    # 检查文件是否更新（重建索引后自动重新加载）
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
                # 确保每个 doc 都有 chunk_id（兼容旧索引）
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
    # 维度校验：embedding 维度必须与索引一致
    if qv.shape[1] != index.d:
        if DEBUG:
            print(f"[WARN] 向量维度不匹配：查询={qv.shape[1]}, 索引={index.d}")
        return []
    scores, ids = index.search(qv, min(top_k, len(docs)))
    hits = []
    invalid_count = 0
    for i, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(docs):
            invalid_count += 1
            continue
        d = dict(docs[idx])
        d["score"] = float(scores[0][i])
        hits.append(d)
    if invalid_count > 0 and DEBUG:
        print(f"[WARN] 向量搜索跳过 {invalid_count} 个无效索引（索引可能与文档不同步）")
    return hits


def read_knowledge_file(product: str, fname: str) -> str:
    p = KNOWLEDGE_DIR / product / fname
    if not p.exists():
        return ""
    # 编码容错：优先 UTF-8，回退 GBK，最后 latin-1（不会失败）
    for enc in ("utf-8", "gb2312", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return ""


def detect_product(question: str) -> str:
    found = detect_terms(question, PRODUCT_ALIASES)
    if found:
        return found[0]
    if (KNOWLEDGE_DIR / "buddhism").exists():
        return "buddhism"
    dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()] if KNOWLEDGE_DIR.exists() else []
    return dirs[0] if dirs else "buddhism"


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


def build_evidence(hits: List[Dict]) -> List[Dict]:
    ev = []
    for h in hits[:6]:
        meta = h.get("meta", {})
        entry = {"meta": meta}
        # 如果有科判面包屑，加入 evidence 以便格式化时引用
        if meta.get("kepan_breadcrumb"):
            entry["kepan_breadcrumb"] = meta["kepan_breadcrumb"]
        ev.append(entry)
    return ev


def filter_by_score(hits: List[Dict], threshold: float = None) -> List[Dict]:
    """过滤低于分数阈值的检索结果"""
    if threshold is None:
        threshold = SCORE_THRESHOLD
    return [h for h in hits if h.get("hybrid_score", h.get("score", 0.0)) >= threshold]


def _text_overlap_ratio(a: str, b: str) -> float:
    """计算两段文本的字符重叠率（基于较短文本）"""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # 使用滑窗 n-gram 近似计算重叠
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
    """语义去重 + 来源多样性控制
    - 移除与已选 chunk 重叠度超过阈值的 chunk
    - 阈值自适应：短 chunk（<200字）提高到 0.85，避免因共同短语误杀
    - 限制同一来源文件最多 max_per_source 个 chunk
    """
    if not hits:
        return []
    selected = []
    source_counts: Dict[str, int] = {}
    for h in hits:
        text = h.get("text", "").strip()
        if not text:
            continue
        # 来源多样性控制
        source = h.get("meta", {}).get("source_file", "_unknown")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        # 语义去重：短文本用更高阈值避免误杀
        is_dup = False
        for sel in selected:
            sel_text = sel.get("text", "")
            min_len = min(len(text), len(sel_text))
            # 短 chunk 阈值提高（200字以下用 0.85），长 chunk 用基础阈值
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
        # 构建来源标签
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
    """从向量检索结果中提取答案段落（保留完整段落，不再按行碎片化）"""
    if not hits:
        return []
    paragraphs = []
    seen = set()
    for h in hits:
        text = h.get("text", "").strip()
        if not text:
            continue
        # 保留完整段落而非拆成单行
        key = re.sub(r"\s+", " ", text)[:200]
        if key in seen:
            continue
        seen.add(key)
        # 清理 FAQ 标记行但保留其他内容
        cleaned_lines = []
        for ln in normalize_lines(text):
            clean = ln.lstrip("-").strip()
            if not clean or is_faq_line(clean):
                continue
            cleaned_lines.append(clean)
        if cleaned_lines:
            paragraphs.append("\n".join(cleaned_lines))
    limit = 6 if mode == "brief" else 12
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
        # 最小有效字符检查：去除标点后至少 8 字
        eff_len = len(re.sub(r"[\s\u3000，。、！？；：""''（）【】《》\-—·]", "", clean))
        if eff_len < 8:
            continue
        if re.match(r"^\d+[）\)]", clean):
            items.append(clean)
            continue
        if len(clean) <= 120:
            items.append(clean)

    items = uniq(items)

    limits = {
        "doctrine": (20, 40),
        "practice": (20, 40),
        "scripture": (14, 30),
        "sect": (14, 30),
        "concept": (14, 30),
        "history": (14, 30),
        "ritual": (12, 24),
    }
    brief_limit, full_limit = limits.get(route, (12, 20))
    limit = brief_limit if mode == "brief" else full_limit
    return items[:limit]


def answer_one(question: str, mode: str) -> str:
    product = detect_product(question)
    route = detect_route(question)
    rewrite = rewrite_query(question)

    # 1. 尝试 FAQ 精确匹配（带别名扩展，包含子目录 FAQ）
    faq_text = read_knowledge_file(product, "faq.txt")
    # 合并子目录中的 FAQ 文件（如 入行论/faq_入行论.txt）
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

    # 2. 向量 + 关键词混合检索
    # 向量搜索用原始问题（保持语义纯净），关键词搜索用扩展查询（覆盖别名/同义词）
    vector_hits = vector_search(product, question, VECTOR_TOP_K)
    _, docs = load_store(product)
    keyword_hits = keyword_search(rewrite["expanded"], docs, KEYWORD_TOP_K) if docs else []
    hits = merge_hybrid(vector_hits, keyword_hits, HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, DEFAULT_TOP_K) if (vector_hits or keyword_hits) else []

    # 过滤低分结果，减少噪音和幻觉风险
    hits = filter_by_score(hits)
    # 语义去重 + 来源多样性控制
    hits = _deduplicate_hits(hits)

    if not hits:
        fallback = [
            "当前知识库未覆盖该问题的直接内容。",
            "建议方向：请查阅相关佛教经典或咨询法师。",
            "您也可以尝试更具体的关键词进行提问。",
        ]
        return format_structured_answer(route, fallback, [], add_risk_note=False)

    # 3. 优先使用 LLM 基于检索上下文生成答案（真正的 RAG）
    context = extract_chunks_as_context(hits, max_chunks=6)
    llm_answer = openai_rag_generate(question, context, route)
    # 验证 LLM 答案：至少 15 字 + 不是单纯复述问题
    if llm_answer and len(llm_answer.strip()) >= 15:
        # 回声检测：答案长度与问题接近且高度重叠时才判为复述
        # 短问题（<10字）跳过检测，因为合理答案必然包含问题关键词
        q_stripped = question.strip()
        a_stripped = llm_answer.strip()
        if len(q_stripped) >= 10 and len(a_stripped) < len(q_stripped) * 3:
            echo_ratio = _text_overlap_ratio(q_stripped, a_stripped[:len(q_stripped) * 3])
            if echo_ratio > 0.85:
                llm_answer = ""  # 走 fallback
    if llm_answer and len(llm_answer.strip()) >= 15:
        # LLM 答案统一走 format_structured_answer，保持格式一致
        evidence = build_evidence(hits)
        return format_structured_answer(route, [llm_answer.strip()], evidence, add_risk_note=(route == "practice"))

    # 4. 无 LLM 时：向量检索结果直接作为答案段落（保留完整段落）
    # 二次质量检查：过滤过短的段落（有效内容 <15 字的跳过）
    body_lines = extract_from_hits(hits, route, mode)
    body_lines = [
        p for p in body_lines
        if len(re.sub(r"[\s\u3000，。、！？；：""''（）【】《》]", "", p)) >= 15
    ]

    if not body_lines:
        # 兜底：章节提取
        main_text = read_knowledge_file(product, "main.txt")
        body_lines = parse_bullets_from_section(main_text, faq_text, route, mode)

    if not body_lines:
        fallback = [
            "当前知识库未覆盖该问题的直接内容。",
            "建议方向：请查阅相关佛教经典或咨询法师。",
            "您也可以尝试更具体的关键词进行提问。",
        ]
        return format_structured_answer(route, fallback, build_evidence(hits), add_risk_note=False)

    text = format_structured_answer(route, body_lines, build_evidence(hits), add_risk_note=(route == "practice"))
    return text


def openai_rag_generate(question: str, context: str, route: str) -> str:
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
            "3. 回答要条理清晰，使用分点或分段组织\n"
            "4. 如参考资料中有经典原文，引用时用「」括起\n\n"
            "## 来源标注\n"
            "- 参考资料标记为 [来源1：...]、[来源2：...] 等\n"
            "- 在回答中引用具体内容后，用 [来源N] 标注，N 为对应编号\n"
            "- 如果多个来源说法不同，分别列出并标注各自来源\n\n"
            "## 格式\n"
            "- 回答末尾加上：「以上内容基于佛教经典与传统教义整理，仅供学习参考。」\n"
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
        answer = (resp.choices[0].message.content or "").strip()
        return answer if answer else ""
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] OpenAI RAG generation failed: {e}")
        return ""


def answer_question(question: str, mode: str) -> str:
    """主入口：回答问题，返回字符串"""
    rewrite = rewrite_query(question)
    outputs = []
    seen = set()
    for subq in rewrite["sub_questions"][:4]:
        # 用问题文本去重，避免重复调用 LLM 回答同义子问题
        subq_key = subq.strip()
        if subq_key in seen:
            continue
        seen.add(subq_key)
        ans = answer_one(subq, mode)
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
    print(ans)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR: " + repr(e))
