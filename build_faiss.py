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
from search_utils import (
    normalize_text, has_kepan_structure, split_by_kepan,
    has_pin_structure, split_by_pin, split_semantic_paragraphs,
)

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


def _sub_chunk_semantic(body: str, max_size: int, overlap: int) -> List[str]:
    """
    将一段正文按语义边界切分为子块。
    优先在段落/颂词/引用边界处分割，避免把颂词和注释拆开。
    如果单个语义段落超长，才回退到滑动窗口。
    """
    paragraphs = split_semantic_paragraphs(body)
    if not paragraphs:
        return chunk_text(body, max_size, overlap)

    chunks = []
    current_parts = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_size:
            # 这个段落本身超长，先把积攒的内容产出，再对超长段落滑动切分
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            for sc in chunk_text(para, max_size, overlap):
                chunks.append(sc)
            continue

        # 加上这段后是否超限
        new_len = current_len + para_len + (2 if current_parts else 0)
        if new_len > max_size and current_parts:
            # 产出当前积攒的内容
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(para)
        current_len += para_len + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_by_kepan(text: str, chunk_size: int = 600, overlap: int = 80):
    """
    按科判结构切分文本。每个科判节点生成带层级面包屑的 chunk。
    正文按语义边界（颂词/引用/段落）切分，保证颂词和注释不被拆开。
    返回列表: [{"text": ..., "kepan_breadcrumb": ..., "kepan_marker": ..., "kepan_title": ...}, ...]
    """
    text = normalize_text(text)
    if not text:
        return []

    sections = split_by_kepan(text)
    if not sections:
        return [{"text": c} for c in chunk_text(text, chunk_size, overlap)]

    results = []
    for sec in sections:
        body = sec["body"].strip()
        breadcrumb = sec["breadcrumb"]
        marker = sec["marker"]
        title = sec["title"]

        if not body:
            prefix = f"【{breadcrumb}】\n{title}"
            results.append({
                "text": prefix,
                "kepan_breadcrumb": breadcrumb,
                "kepan_marker": marker,
                "kepan_title": title,
            })
            continue

        prefix = f"【{breadcrumb}】\n"
        effective_size = chunk_size - len(prefix)
        if effective_size < 100:
            effective_size = 100

        if len(body) <= effective_size:
            results.append({
                "text": prefix + body,
                "kepan_breadcrumb": breadcrumb,
                "kepan_marker": marker,
                "kepan_title": title,
            })
        else:
            # 按语义边界切分，保留颂词+注释完整性
            sub_chunks = _sub_chunk_semantic(body, effective_size, overlap)
            for sc in sub_chunks:
                results.append({
                    "text": prefix + sc,
                    "kepan_breadcrumb": breadcrumb,
                    "kepan_marker": marker,
                    "kepan_title": title,
                })

    return results


def chunk_by_pin(text: str, chunk_size: int = 600, overlap: int = 80):
    """
    按品结构切分文本（无科判的论典）。
    每个品内部再按语义段落切分。
    """
    text = normalize_text(text)
    if not text:
        return []

    pin_sections = split_by_pin(text)
    if not pin_sections:
        return [{"text": c} for c in chunk_text(text, chunk_size, overlap)]

    results = []
    for sec in pin_sections:
        pin_title = sec["title"]
        body = sec["body"].strip()

        if not body:
            results.append({"text": pin_title, "pin_title": pin_title})
            continue

        # 品内可能包含科判
        if has_kepan_structure(body):
            kepan_chunks = chunk_by_kepan(body, chunk_size, overlap)
            for kc in kepan_chunks:
                # 在面包屑前加品名
                if "kepan_breadcrumb" in kc:
                    kc["kepan_breadcrumb"] = f"{pin_title} > {kc['kepan_breadcrumb']}"
                    kc["text"] = f"【{kc['kepan_breadcrumb']}】\n" + kc["text"].split("】\n", 1)[-1]
                else:
                    kc["pin_title"] = pin_title
                results.append(kc)
        else:
            # 品内无科判，按语义段落切分
            prefix = f"【{pin_title}】\n"
            effective_size = chunk_size - len(prefix)
            if effective_size < 100:
                effective_size = 100

            if len(body) <= effective_size:
                results.append({"text": prefix + body, "pin_title": pin_title})
            else:
                sub_chunks = _sub_chunk_semantic(body, effective_size, overlap)
                for sc in sub_chunks:
                    results.append({"text": prefix + sc, "pin_title": pin_title})

    return results


def chunk_smart(text: str, chunk_size: int = 600, overlap: int = 80):
    """
    智能分块：自动检测文本结构并选择最佳分块策略。
    优先级：科判 > 品 > 语义段落 > 滑动窗口
    """
    text = normalize_text(text)
    if not text:
        return []

    # 1. 有科判结构
    if has_kepan_structure(text):
        return chunk_by_kepan(text, chunk_size, overlap)

    # 2. 有品结构
    if has_pin_structure(text):
        return chunk_by_pin(text, chunk_size, overlap)

    # 3. 无明显结构，用语义段落分块
    paragraphs = split_semantic_paragraphs(text)
    if len(paragraphs) > 1:
        sub_chunks = _sub_chunk_semantic(text, chunk_size, overlap)
        return [{"text": c} for c in sub_chunks]

    # 4. 回退到滑动窗口
    return [{"text": c} for c in chunk_text(text, chunk_size, overlap)]


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
        elif stype in ("main", "faq"):
            # 智能检测结构：科判 > 品 > 语义段落 > 滑动窗口
            if has_kepan_structure(text):
                print(f"[INFO] {product}/{fname}: 检测到科判结构，使用层级分块")
                chunks_data = chunk_by_kepan(text, CHUNK_SIZE, CHUNK_OVERLAP)
            elif has_pin_structure(text):
                print(f"[INFO] {product}/{fname}: 检测到品结构，使用品章节分块")
                chunks_data = chunk_by_pin(text, CHUNK_SIZE, CHUNK_OVERLAP)
            else:
                chunks_data = chunk_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)
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
            # 保存结构元数据（科判/品）
            if "kepan_breadcrumb" in cd:
                meta["kepan_breadcrumb"] = cd["kepan_breadcrumb"]
                meta["kepan_marker"] = cd.get("kepan_marker", "")
                meta["kepan_title"] = cd.get("kepan_title", "")
            if "pin_title" in cd:
                meta["pin_title"] = cd["pin_title"]
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
