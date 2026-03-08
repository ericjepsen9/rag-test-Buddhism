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
from sentence_transformers import SentenceTransformer

from rag_runtime_config import KNOWLEDGE_DIR, STORE_ROOT, CHUNK_SIZE, CHUNK_OVERLAP
from search_utils import normalize_text, has_kepan_structure, split_by_kepan

MODEL_NAME = "BAAI/bge-m3"
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[INFO] 加载模型：{MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
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


def chunk_by_kepan(text: str, chunk_size: int = 600, overlap: int = 80):
    """
    按科判结构切分文本。每个科判节点生成带层级面包屑的 chunk。
    如果某个科判下的正文太长，再用滑动窗口二次切分。
    返回列表: [{"text": ..., "kepan_breadcrumb": ..., "kepan_marker": ..., "kepan_title": ...}, ...]
    """
    text = normalize_text(text)
    if not text:
        return []

    sections = split_by_kepan(text)
    if not sections:
        # 没有科判结构，回退到普通分块
        return [{"text": c} for c in chunk_text(text, chunk_size, overlap)]

    results = []
    for sec in sections:
        body = sec["body"].strip()
        breadcrumb = sec["breadcrumb"]
        marker = sec["marker"]
        title = sec["title"]

        if not body:
            # 科判标题本身也作为 chunk（有些科判只有标题和子科判）
            prefix = f"【{breadcrumb}】\n{title}"
            results.append({
                "text": prefix,
                "kepan_breadcrumb": breadcrumb,
                "kepan_marker": marker,
                "kepan_title": title,
            })
            continue

        # 在正文前加上层级面包屑作为上下文
        prefix = f"【{breadcrumb}】\n"
        effective_size = chunk_size - len(prefix)
        if effective_size < 100:
            effective_size = 100

        if len(body) <= effective_size:
            # 正文较短，直接作为一个 chunk
            results.append({
                "text": prefix + body,
                "kepan_breadcrumb": breadcrumb,
                "kepan_marker": marker,
                "kepan_title": title,
            })
        else:
            # 正文较长，滑动窗口二次切分
            sub_chunks = chunk_text(body, effective_size, overlap)
            for sc in sub_chunks:
                results.append({
                    "text": prefix + sc,
                    "kepan_breadcrumb": breadcrumb,
                    "kepan_marker": marker,
                    "kepan_title": title,
                })

    return results


def embed_texts(texts):
    model = get_model()
    vecs = model.encode(texts, batch_size=8, show_progress_bar=True, normalize_embeddings=False)
    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"向量维度异常: {vecs.shape}")
    faiss.normalize_L2(vecs)
    return vecs


def collect_product_records(product: str):
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise FileNotFoundError(f"未找到产品目录：{pdir}")
    # 收集目录下所有 .txt 文件
    known_files = [("main.txt", "main"), ("faq.txt", "faq"), ("alias.txt", "alias")]
    known_names = {name for name, _ in known_files}
    # 自动发现额外的 .txt 文件（如论典原文）
    extra_files = []
    for fp in sorted(pdir.glob("*.txt")):
        if fp.name not in known_names:
            extra_files.append((fp.name, "main"))

    all_files = known_files + extra_files
    records = []
    for fname, stype in all_files:
        f = pdir / fname
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")

        if stype == "alias":
            chunks_data = [{"text": text}]
        elif stype == "main" and has_kepan_structure(text):
            # 检测到科判结构，使用科判分块
            print(f"[INFO] {product}/{fname}: 检测到科判结构，使用层级分块")
            chunks_data = chunk_by_kepan(text, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            chunks_data = [{"text": c} for c in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)]

        print(f"[OK] {product}/{fname}: {len(chunks_data)} chunks")
        for i, cd in enumerate(chunks_data, 1):
            meta = {
                "product_id": product,
                "source_file": f.name,
                "source_type": stype,
                "chunk_id": i,
            }
            # 保存科判元数据（如果有）
            if "kepan_breadcrumb" in cd:
                meta["kepan_breadcrumb"] = cd["kepan_breadcrumb"]
                meta["kepan_marker"] = cd.get("kepan_marker", "")
                meta["kepan_title"] = cd.get("kepan_title", "")
            records.append({
                "text": cd["text"],
                "meta": meta,
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
