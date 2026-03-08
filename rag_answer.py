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
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, FAQ_KEYWORD_MAP
)
from search_utils import (
    normalize_text, normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms, match_faq
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer

_model = None
_faiss = None
_BGEM3 = None
_store_cache = {}  # 缓存已加载的 index + docs


def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_mod
        _faiss = _faiss_mod
    return _faiss


def get_bg_cls():
    global _BGEM3
    if _BGEM3 is None:
        from FlagEmbedding import BGEM3FlagModel as _cls
        _BGEM3 = _cls
    return _BGEM3


def get_model():
    global _model
    if _model is None:
        _model = get_bg_cls()("BAAI/bge-m3", use_fp16=False)
    return _model


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    out = model.encode([text], batch_size=1, max_length=1024)
    if isinstance(out, dict):
        if "dense_vecs" in out:
            vec = out["dense_vecs"]
        elif "dense" in out:
            vec = out["dense"]
        elif "embeddings" in out:
            vec = out["embeddings"]
        else:
            raise ValueError("未找到查询向量字段")
    else:
        vec = out
    vec = np.asarray(vec, dtype="float32")
    get_faiss().normalize_L2(vec)
    return vec


def load_store(product: str):
    """加载向量索引和文档，带缓存"""
    if product in _store_cache:
        return _store_cache[product]
    store_dir = STORE_ROOT / product
    index_path = store_dir / "index.faiss"
    docs_path = store_dir / "docs.jsonl"
    if not index_path.exists() or not docs_path.exists():
        result = (None, [])
        _store_cache[product] = result
        return result
    index = get_faiss().read_index(str(index_path))
    docs = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    result = (index, docs)
    _store_cache[product] = result
    return result


def vector_search(product: str, query: str, top_k: int) -> List[Dict]:
    index, docs = load_store(product)
    if index is None or not docs:
        return []
    qv = embed_query(query)
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
    return p.read_text(encoding="utf-8")


def detect_product(question: str) -> str:
    found = detect_terms(question, PRODUCT_ALIASES)
    if found:
        return found[0]
    if (KNOWLEDGE_DIR / "buddhism").exists():
        return "buddhism"
    dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()] if KNOWLEDGE_DIR.exists() else []
    return dirs[0] if dirs else "buddhism"


def detect_route(question: str) -> str:
    q = (question or "").lower()
    order = ["doctrine", "practice", "scripture", "sect", "concept", "history", "ritual", "basic"]
    for route in order:
        for kw in QUESTION_ROUTES.get(route, []):
            if kw.lower() in q:
                return route
    return "basic"


def build_evidence(hits: List[Dict]) -> List[Dict]:
    ev = []
    for h in hits[:6]:
        ev.append({"meta": h.get("meta", {})})
    return ev


def extract_chunks_as_context(hits: List[Dict], max_chunks: int = 6) -> str:
    """从检索结果中提取完整 chunk 文本作为 LLM 上下文"""
    if not hits:
        return ""
    chunks = []
    seen = set()
    for h in hits[:max_chunks]:
        text = h.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append(text)
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

    # 1. 尝试 FAQ 精确匹配
    faq_text = read_knowledge_file(product, "faq.txt")
    faq_answer = match_faq(question, faq_text, FAQ_KEYWORD_MAP)
    if faq_answer:
        return format_structured_answer(route, [faq_answer], [], add_risk_note=False)

    # 2. 向量 + 关键词混合检索
    vector_hits = vector_search(product, rewrite["expanded"], VECTOR_TOP_K)
    _, docs = load_store(product)
    keyword_hits = keyword_search(rewrite["expanded"], docs, KEYWORD_TOP_K) if docs else []
    hits = merge_hybrid(vector_hits, keyword_hits, HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, DEFAULT_TOP_K) if (vector_hits or keyword_hits) else []

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
    if llm_answer:
        return llm_answer

    # 4. 无 LLM 时：向量检索结果直接作为答案段落（保留完整段落）
    body_lines = extract_from_hits(hits, route, mode)

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
            "你是一个佛教知识问答助手。请严格基于以下【参考资料】回答用户的问题。\n"
            "规则：\n"
            "1. 只使用参考资料中的内容回答，不要编造或添加资料中没有的信息\n"
            "2. 如果参考资料不足以完整回答问题，请如实说明哪些部分无法回答\n"
            "3. 回答要条理清晰、准确专业\n"
            "4. 适当引用经典原文（如果参考资料中有的话）\n"
            "5. 回答末尾加上：「以上内容基于佛教经典与传统教义整理，仅供学习参考。」\n"
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
            temperature=0.3,
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
        ans = answer_one(subq, mode)
        key = ans.strip()
        if key and key not in seen:
            seen.add(key)
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
