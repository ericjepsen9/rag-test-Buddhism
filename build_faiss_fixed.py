# build_faiss.py
# 用法：
#   python build_faiss.py --list
#   python build_faiss.py --product feiluoao
#
# 作用：
# - 从 knowledge/<product_id>/{main,faq,alias}.txt 读取文本
# - 自动切块（按段落/标题优先 + overlap）
# - 生成 stores/<product_id>/docs.jsonl 和 stores/<product_id>/index.faiss
#
# 说明：
# - 兼容 Windows / 中文文本
# - 解决 numpy array 在 `or` 链里触发 ValueError 的问题（用显式 None 判断）

import os
import re
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("请先安装 faiss-cpu：pip install faiss-cpu") from e

try:
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
except Exception as e:
    raise RuntimeError("请先安装 FlagEmbedding：pip install FlagEmbedding") from e


# ====== 路径约定（不依赖其他文件）======
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STORES_DIR = BASE_DIR / "stores"

MODEL_NAME = os.environ.get("BGE_MODEL_NAME", "BAAI/bge-m3")


# ====== chunk 参数（可按需要调）======
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))


def read_text_auto(p: Path) -> str:
    if not p.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return p.read_text(errors="ignore")


def normalize_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def is_title_like(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    patterns = [
        r"^[一二三四五六七八九十]+、",
        r"^\d+[）\.\)]",
        r"^第[一二三四五六七八九十0-9]+",
        r"^STEP\s*\d+",
        r"^【.+】$",
        r"^#+\s*",
        r"^=+$",
        r"^-{3,}$",
    ]
    return any(re.match(p, s, flags=re.IGNORECASE) for p in patterns)


def split_into_paragraphs(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in (text or "").split("\n")]
    paras: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            s = "\n".join(buf).strip()
            if s:
                paras.append(s)
            buf = []

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            continue
        if is_title_like(s):
            flush()
            paras.append(s)
            continue
        buf.append(s)

    flush()
    return paras


def merge_paragraphs_to_chunks(paragraphs: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    cur = ""

    def add_chunk(x: str):
        x = x.strip()
        if x:
            chunks.append(x)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 超长段落切分
        if len(para) > int(chunk_size * 1.2):
            if cur:
                add_chunk(cur)
                cur = ""
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(para):
                add_chunk(para[start:start + chunk_size])
                start += step
            continue

        candidate = (cur + "\n" + para).strip() if cur else para
        if len(candidate) <= chunk_size:
            cur = candidate
        else:
            if cur:
                add_chunk(cur)
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                cur = (tail + "\n" + para).strip()
                if len(cur) > int(chunk_size * 1.3):
                    cur = para
            else:
                cur = para

    if cur:
        add_chunk(cur)

    # 去重
    dedup: List[str] = []
    seen = set()
    for c in chunks:
        k = re.sub(r"\s+", " ", c.strip())
        if k and k not in seen:
            dedup.append(c.strip())
            seen.add(k)
    return dedup


def chunk_text(text: str) -> List[str]:
    t = normalize_text(text)
    if not t:
        return []
    paras = split_into_paragraphs(t)
    return merge_paragraphs_to_chunks(paras, CHUNK_SIZE, CHUNK_OVERLAP)


def embed_texts(model: BGEM3FlagModel, texts: List[str]) -> np.ndarray:
    # BGEM3FlagModel.encode 返回 dict，dense_vecs 是 numpy.ndarray
    out = model.encode(texts, batch_size=8, max_length=8192)
    vecs = None
    if isinstance(out, dict):
        if out.get("dense_vecs") is not None:
            vecs = out.get("dense_vecs")
        elif out.get("dense") is not None:
            vecs = out.get("dense")
        elif out.get("embeddings") is not None:
            vecs = out.get("embeddings")
    elif isinstance(out, (list, tuple, np.ndarray)):
        vecs = out

    if vecs is None:
        raise ValueError("模型 encode 输出不包含 dense 向量（dense_vecs/dense/embeddings）。")

    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"向量形状异常：{vecs.shape}")
    return vecs


def build_index(vecs: np.ndarray) -> "faiss.Index":
    # cosine 相似度：先 L2 normalize，再用 inner product
    faiss.normalize_L2(vecs)
    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def write_docs_jsonl(records: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_index(index: "faiss.Index", out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))


def build_for_product(product_id: str):
    product_id = (product_id or "").strip()
    if not product_id:
        raise ValueError("product_id 不能为空")

    pdir = KNOWLEDGE_DIR / product_id
    if not pdir.exists():
        print(f"[ERROR] knowledge 目录不存在：{pdir}")
        print("请先创建目录结构，例如 knowledge/feiluoao/main.txt")
        return

    parts = [("main", pdir / "main.txt"), ("faq", pdir / "faq.txt"), ("alias", pdir / "alias.txt")]
    records: List[Dict[str, Any]] = []

    for part, fp in parts:
        if not fp.exists():
            print(f"[WARN] 缺少文件: {fp}")
            continue
        raw = read_text_auto(fp)
        chunks = chunk_text(raw)
        print(f"[OK] {product_id}/{fp.name}: {len(chunks)} chunks")

        for i, ch in enumerate(chunks):
            records.append({
                "id": f"{product_id}/{fp.name}::chunk_{i}",
                "text": ch,
                "meta": {
                    "product_id": product_id,
                    "source_type": part,
                    "source_file": str(fp.relative_to(BASE_DIR)).replace("\\", "/"),
                }
            })

    if not records:
        print(f"[SKIP] {product_id}: 没有可用文本")
        return

    print(f"[INFO] 加载模型：{MODEL_NAME}（首次会下载）")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    print(f"[INFO] Embedding {len(records)} chunks ...")
    vecs = embed_texts(model, [r["text"] for r in records])

    print("[INFO] Building FAISS index ...")
    index = build_index(vecs)

    out_dir = STORES_DIR / product_id
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    write_docs_jsonl(records, docs_path)
    save_index(index, index_path)

    print("[DONE] Built")
    print(f"  product: {product_id}")
    print(f"  docs   : {docs_path}")
    print(f"  index  : {index_path}")
    print(f"  dim    : {index.d}")


def list_products():
    if not KNOWLEDGE_DIR.exists():
        print(f"[ERROR] knowledge 目录不存在：{KNOWLEDGE_DIR}")
        return
    prods = sorted([p.name for p in KNOWLEDGE_DIR.iterdir() if p.is_dir()])
    if not prods:
        print("[INFO] knowledge 下没有产品目录")
        return
    print("Products:")
    for p in prods:
        print(" -", p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, default="", help="产品ID，例如 feiluoao")
    parser.add_argument("--list", action="store_true", help="列出 knowledge 下的产品目录")
    args = parser.parse_args()

    if args.list:
        list_products()
        return

    if args.product:
        build_for_product(args.product)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
