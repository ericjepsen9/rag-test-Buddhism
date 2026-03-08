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
    if (KNOWLEDGE_DIR / "feiluoao").exists():
        return "feiluoao"
    dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()] if KNOWLEDGE_DIR.exists() else []
    return dirs[0] if dirs else "feiluoao"


def detect_route(question: str) -> str:
    q = (question or "").lower()
    order = ["risk", "combo", "aftercare", "operation", "anti_fake", "contraindication", "basic"]
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


def parse_anti_fake(main_text: str, faq_text: str, mode: str) -> List[str]:
    rule = SECTION_RULES["anti_fake"]
    block = section_block(main_text, rule["titles"], rule["stops"])
    if not block:
        block = section_block(faq_text, ["防伪", "HiddenTag"], [])
    if not block:
        return []

    lines = [ln for ln in normalize_lines(block) if not is_faq_line(ln)]
    subject, official, notes = [], [], []
    steps = {i: [] for i in range(1, 6)}
    current = None
    in_notes = False

    for ln in lines:
        if "防伪验证主体" in ln:
            current = None
            in_notes = False
            continue
        if "官方验证方式" in ln:
            current = None
            in_notes = False
            continue
        if "【防伪步骤】" in ln:
            current = None
            in_notes = False
            continue
        if "【防伪注意事项】" in ln:
            current = None
            in_notes = True
            continue

        m = re.match(r"STEP\s*(\d+)", ln, re.I)
        if m:
            current = int(m.group(1))
            in_notes = False
            continue

        clean = ln.lstrip("-").strip()
        if not clean:
            continue

        if in_notes:
            notes.append(clean)
            continue

        if current in steps:
            steps[current].append(clean)
            continue

        if "G-international" in clean and ("公司" in clean or len(clean) <= 40):
            subject.append(clean)
            continue

        # 官方验证方式只收核心句，避免把 step1 吃掉
        if ("HiddenTag APP 扫描" in clean) or ("扫码方式无效" in clean) or ("其他扫码方式无效" in clean):
            official.append(clean)
            continue

    # FAQ 兜底补缺
    faq_lines = normalize_lines(faq_text)
    if not subject:
        for ln in faq_lines:
            if "G-international" in ln and ("官方认证" in ln or "公司" in ln):
                subject.append("韩国(株)G-international 公司")
                break

    if not official:
        for x in ["使用 HiddenTag APP 扫描验证", "其他扫码方式无效（资料描述）"]:
            official.append(x)

    # 步骤硬兜底：避免 step 消失
    defaults = {
        1: ["在手机应用商店下载 HiddenTag APP"],
        2: ["打开 APP，点击“正品认证”"],
        3: ["肉眼确认产品标签是否为正品标签"],
        4: ["扫描产品上的 HiddenTag 标签", "建议在不反光环境下扫描，提高识别成功率"],
        5: ["验证成功后，APP 显示韩国(株)G-international 官方认证图片"],
    }
    for i in range(1, 6):
        if not steps[i]:
            steps[i] = defaults[i][:]

    if not notes:
        notes = ["仅 HiddenTag APP 可用于验证", "标签保持平整、避免反光", "以官方认证结果为准"]

    out = ["防伪验证主体："]
    for x in uniq(subject or ["韩国(株)G-international 公司"]):
        out.append(x)
    out.append("官方验证方式：")
    for x in uniq(official):
        out.append(x)
    out.append("【防伪步骤】")
    for i in range(1, 6):
        out.append(f"STEP {i}：")
        lim = 1 if mode == "brief" and i in (1, 2, 3, 5) else 2
        for x in uniq(steps[i])[:lim]:
            out.append(x)
    out.append("【防伪注意事项】")
    for x in uniq(notes)[:(2 if mode == "brief" else 6)]:
        out.append(x)
    return out


def parse_bullets_from_section(main_text: str, faq_text: str, route: str, mode: str) -> List[str]:
    rule = SECTION_RULES[route]
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
        # 保留小节标题和条目
        if re.match(r"^\d+[）\)]", clean):
            items.append(clean)
            continue
        if any(k in clean for k in ["术后", "洗脸", "辛辣", "禁酒", "面膜", "保湿", "熬夜", "按摩", "洁面仪", "多喝水", "水果", "蔬菜", "针头", "深度", "注射量", "点间距", "微针", "水光", "过敏", "妊娠", "哺乳", "免疫"]):
            items.append(clean)
            continue
        if route == "basic" and len(clean) <= 60:
            items.append(clean)

    items = uniq(items)

    if route == "contraindication":
        items = [x for x in items if not ("术后一周内不要" in x or "洁面仪" in x or "怎么验真伪" in x or "正品验证" in x)]
        if "具体是否适用需由专业医生评估。" not in items:
            items.append("具体是否适用需由专业医生评估。")

    if route == "operation":
        filtered = []
        for x in items:
            if any(k in x for k in ["针头", "深度", "注射", "0.8", "1.0", "0.3ml", "2cm", "MTS", "水光", "涂抹"]):
                filtered.append(x)
        items = uniq(filtered)

    if route == "aftercare":
        # brief 也尽量完整
        limit = 20 if mode == "brief" else 28
    elif route == "operation":
        limit = 16 if mode == "brief" else 24
    elif route == "contraindication":
        limit = 12 if mode == "brief" else 18
    else:
        limit = 12 if mode == "brief" else 20

    return items[:limit]


def parse_answer(route: str, product: str, mode: str) -> List[str]:
    main_text = read_knowledge_file(product, "main.txt")
    faq_text = read_knowledge_file(product, "faq.txt")
    if route == "anti_fake":
        return parse_anti_fake(main_text, faq_text, mode)
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
            "请在不改变事实的前提下，将以下基于知识库的回答整理得更专业、更自然。"
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
            "当前知识库未覆盖该问题的直接结论。",
            "可确认方向：请核对产品主文档、FAQ 或补充对应知识库章节。",
            "该问题可能涉及医生判断范围，建议由专业医师评估。",
        ]
        return format_structured_answer(route, fallback, build_evidence(hits), add_risk_note=(route == "risk"))

    text = format_structured_answer(route, body_lines, build_evidence(hits), add_risk_note=(route == "risk"))
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
        print('Usage: python rag_answer.py "你的问题" [k] [brief|full]')
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
