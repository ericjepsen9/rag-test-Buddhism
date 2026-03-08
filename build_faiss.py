import os
import sys
import json
import argparse
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

from rag_runtime_config import KNOWLEDGE_DIR, STORE_ROOT
from search_utils import normalize_text

MODEL_NAME = "BAAI/bge-m3"
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[INFO] 加载模型：{MODEL_NAME}")
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)
    return _model


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80):
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(texts):
    model = get_model()
    out = model.encode(texts, batch_size=8, max_length=1024)
    if isinstance(out, dict):
        if "dense_vecs" in out:
            vecs = out["dense_vecs"]
        elif "dense" in out:
            vecs = out["dense"]
        elif "embeddings" in out:
            vecs = out["embeddings"]
        else:
            raise ValueError("encode 输出中未找到向量字段")
    else:
        vecs = out
    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"向量维度异常: {vecs.shape}")
    faiss.normalize_L2(vecs)
    return vecs


def collect_product_records(product: str):
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise FileNotFoundError(f"未找到产品目录：{pdir}")
    files = [("main.txt", "main"), ("faq.txt", "faq"), ("alias.txt", "alias")]
    records = []
    for fname, stype in files:
        f = pdir / fname
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        chunks = [text] if stype == "alias" else chunk_text(text)
        print(f"[OK] {product}/{fname}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            records.append({
                "text": chunk,
                "meta": {
                    "product_id": product,
                    "source_file": f.name,
                    "source_type": stype,
                    "chunk_id": i,
                }
            })
    if not records:
        raise ValueError(f"{product} 没有可用文本")
    return records


def build_for_product(product: str):
    records = collect_product_records(product)
    texts = [r["text"] for r in records]
    print(f"[INFO] Total chunks: {len(texts)}")
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    vecs = embed_texts(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    out_dir = STORE_ROOT / product
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    with docs_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    faiss.write_index(index, str(index_path))

    print(f"[DONE] Built store")
    print(f"       product: {product}")
    print(f"       chunks : {len(records)}")
    print(f"       dim    : {dim}")


def list_products():
    if not KNOWLEDGE_DIR.exists():
        print(f"[ERROR] knowledge 目录不存在：{KNOWLEDGE_DIR}")
        return
    for p in sorted([x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()]):
        print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=str, help="产品目录名")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_products()
        return
    if not args.product:
        ap.error("请使用 --product <name> 或 --list")
    build_for_product(args.product.strip())


if __name__ == "__main__":
    main()
