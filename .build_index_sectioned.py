# build_index_sectioned.py
import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

from section_parser import split_main_by_sections, split_section_to_subchunks

STORE_DIR = "faiss_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
DOCS_PATH = os.path.join(STORE_DIR, "docs.jsonl")
DIM = 1024  # bge-m3 dense dim

DATA_DIR = "data"  # 你的原始 txt 放这里（按实际改）

def ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)

def save_docs(docs: List[Dict[str, Any]]):
    ensure_store()
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def save_index(vecs: np.ndarray):
    ensure_store()
    index = faiss.IndexFlatIP(DIM)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def iter_source_files():
    """
    约定命名（你按自己文件名改）：
    - main_xxx.txt   主文档（章节化切）
    - faq_xxx.txt    FAQ
    - alias_xxx.txt  别名/纠错
    """
    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith(".txt"):
            continue
        yield os.path.join(DATA_DIR, fn)

def infer_source_type(filename: str) -> str:
    s = filename.lower()
    if "main" in s:
        return "main"
    if "faq" in s:
        return "faq"
    if "alias" in s:
        return "alias"
    return "other"

def build_docs() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    for path in iter_source_files():
        fn = os.path.basename(path)
        source_type = infer_source_type(fn)
        raw = read_text(path).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            continue

        if source_type == "main":
            sections = split_main_by_sections(raw)
            for sec in sections:
                # 你可以先不拆子块，直接整章入库（最稳）
                # subchunks = [sec["text"]]

                # 如果章节很长，再按【小标题】/STEP温和拆分
                subchunks = split_section_to_subchunks(sec["text"], max_chars=1200)

                for i, chunk in enumerate(subchunks, 1):
                    docs.append({
                        "text": chunk,
                        "meta": {
                            "source_file": fn,
                            "source_type": "main",
                            "section_title": sec["section_title"],
                            "section_key": sec["section_key"],
                            "sub_index": i,
                        }
                    })

        elif source_type == "faq":
            # FAQ 尽量整段入库，保持【Q】【A】结构
            docs.append({
                "text": raw,
                "meta": {
                    "source_file": fn,
                    "source_type": "faq",
                    "section_title": "FAQ",
                    "section_key": "faq",
                    "sub_index": 1,
                }
            })

        else:
            # alias / other：简单按段落切
            parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
            for i, p in enumerate(parts, 1):
                docs.append({
                    "text": p,
                    "meta": {
                        "source_file": fn,
                        "source_type": source_type,
                        "section_title": "",
                        "section_key": source_type,
                        "sub_index": i,
                    }
                })

    return docs

def embed_texts(model, texts: List[str]) -> np.ndarray:
    # BGEM3FlagModel 返回结构依版本可能不同，这里兼容常见形式
    out = model.encode(
        texts,
        batch_size=8,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    if isinstance(out, dict):
        vecs = out.get("dense_vecs") or out.get("dense_embeddings")
    else:
        vecs = out

    arr = np.array(vecs, dtype="float32")
    return arr

def main():
    docs = build_docs()
    if not docs:
        print("No docs found.")
        return

    print(f"Docs to index: {len(docs)}")

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    texts = [d["text"] for d in docs]
    vecs = embed_texts(model, texts)

    save_docs(docs)
    save_index(vecs)

    print("Index built.")
    print(f"- docs: {DOCS_PATH}")
    print(f"- index: {INDEX_PATH}")

if __name__ == "__main__":
    main()