import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

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
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT
)
from search_utils import (
    normalize_text, normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer

_model = None
_faiss = None
_BGEM3 = None


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


def save_answer(text: str):
    OUT_PATH.write_text((text or "").strip() + "\n", encoding="utf-8-sig")


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
    store_dir = STORE_ROOT / product
    index_path = store_dir / "index.faiss"
    docs_path = store_dir / "docs.jsonl"
    if not index_path.exists() or not docs_path.exists():
        return None, []
    index = get_faiss().read_index(str(index_path))
    docs = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return index, docs


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
        ev.append({
            "meta": h.get("meta", {})
        })
    return ev


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


def parse_answer(route: str, product: str, mode: str) -> List[str]:
    main_text = read_knowledge_file(product, "main.txt")
    faq_text = read_knowledge_file(product, "faq.txt")
    return parse_bullets_from_section(main_text, faq_text, route, mode)


def openai_rewrite_answer(text: str, route: str) -> str:
    if not USE_OPENAI:
        return text
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = (
            "请在不改变事实的前提下，将以下基于佛教知识库的回答整理得更专业、更流畅。"
            "不要新增事实。保留结构化格式。\n\n" + text
        )
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
        return (resp.output_text or "").strip() or text
    except Exception:
        return text


def answer_one(question: str, mode: str) -> str:
    product = detect_product(question)
    route = detect_route(question)
    rewrite = rewrite_query(question)

    vector_hits = vector_search(product, rewrite["expanded"], VECTOR_TOP_K)
    _, docs = load_store(product)
    keyword_hits = keyword_search(rewrite["expanded"], docs, KEYWORD_TOP_K) if docs else []
    hits = merge_hybrid(vector_hits, keyword_hits, HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, DEFAULT_TOP_K) if (vector_hits or keyword_hits) else []

    body_lines = parse_answer(route, product, mode)

    if not body_lines:
        fallback = [
            "当前知识库未覆盖该问题的直接内容。",
            "建议方向：请查阅相关佛教经典或咨询法师。",
            "您也可以尝试更具体的关键词进行提问。",
        ]
        return format_structured_answer(route, fallback, build_evidence(hits), add_risk_note=False)

    text = format_structured_answer(route, body_lines, build_evidence(hits), add_risk_note=(route == "practice"))
    return openai_rewrite_answer(text, route)


def answer_question(question: str, mode: str) -> str:
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
    save_answer(ans)
    print("\n===== Answer saved =====")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save_answer("ERROR: " + repr(e))
        print("\n===== Answer saved =====")
        print(f"Saved to: {OUT_PATH}")
