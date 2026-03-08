import os
import re
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
    detect_content_type, split_by_topic, split_by_ritual_section,
    split_by_steps, split_by_headings,
    split_by_qa, split_by_gongan, split_by_commentary,
    split_by_verse_collection, split_by_letter,
    split_mixed_body, merge_sub_segments,
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
    将一段正文按内容感知边界切分为子块。

    优先使用混合内容分段（识别颂词/讲解/公案/问答边界），
    保证：颂词+讲解不拆开、问+答不拆开、公案完整。
    回退到语义段落分段，最后回退到滑动窗口。
    """
    # 1. 尝试混合内容感知分段
    segments = split_mixed_body(body)
    if len(segments) > 1:
        return merge_sub_segments(segments, max_size)

    # 2. 回退到语义段落分段
    paragraphs = split_semantic_paragraphs(body)
    if not paragraphs:
        return chunk_text(body, max_size, overlap)

    chunks = []
    current_parts = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            for sc in chunk_text(para, max_size, overlap):
                chunks.append(sc)
            continue

        new_len = current_len + para_len + (2 if current_parts else 0)
        if new_len > max_size and current_parts:
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


def _chunk_sectioned(sections: List[Dict], chunk_size: int, overlap: int,
                     content_type: str) -> List[Dict]:
    """
    通用：将 split_by_xxx 产出的 sections 列表分块。
    每个 section 加上标题前缀，正文按语义边界切分。
    """
    results = []
    for sec in sections:
        sec_title = sec["title"]
        body = sec["body"].strip()
        if not body:
            results.append({"text": sec_title, "section_title": sec_title,
                            "content_type": content_type})
            continue

        prefix = f"【{sec_title}】\n"
        effective_size = chunk_size - len(prefix)
        if effective_size < 100:
            effective_size = 100

        if len(body) <= effective_size:
            results.append({"text": prefix + body, "section_title": sec_title,
                            "content_type": content_type})
        else:
            sub_chunks = _sub_chunk_semantic(body, effective_size, overlap)
            for sc in sub_chunks:
                results.append({"text": prefix + sc, "section_title": sec_title,
                                "content_type": content_type})
    return results


def chunk_smart(text: str, chunk_size: int = 600, overlap: int = 80):
    """
    智能分块：自动检测文本结构并选择最佳分块策略。
    优先级：科判 > 品 > 仪轨 > 问答 > 公案 > 注疏 > 偈颂集 > 书信 >
            方法步骤 > 演讲话题 > 文章小节 > 语义段落 > 滑动窗口
    """
    text = normalize_text(text)
    if not text:
        return []

    content_type = detect_content_type(text)

    # 分类型处理的映射表
    _splitter_map = {
        "ritual":           (split_by_ritual_section, "ritual"),
        "qa":               (split_by_qa,             "qa"),
        "gongan":           (split_by_gongan,         "gongan"),
        "commentary":       (split_by_commentary,     "commentary"),
        "verse_collection": (split_by_verse_collection, "verse_collection"),
        "letter":           (split_by_letter,         "letter"),
        "method":           (split_by_steps,          "method"),
        "talk":             (split_by_topic,          "talk"),
        "article":          (split_by_headings,       "article"),
    }

    # 1. 科判结构（正式论典）
    if content_type == "kepan":
        return chunk_by_kepan(text, chunk_size, overlap)

    # 2. 品结构
    if content_type == "pin":
        return chunk_by_pin(text, chunk_size, overlap)

    # 3-11. 其他结构化类型
    if content_type in _splitter_map:
        splitter, ctype = _splitter_map[content_type]
        sections = splitter(text)
        if sections:
            return _chunk_sectioned(sections, chunk_size, overlap, ctype)

    # 7. 无明显结构，用语义段落分块
    paragraphs = split_semantic_paragraphs(text)
    if len(paragraphs) > 1:
        sub_chunks = _sub_chunk_semantic(text, chunk_size, overlap)
        return [{"text": c} for c in sub_chunks]

    # 8. 回退到滑动窗口
    return [{"text": c} for c in chunk_text(text, chunk_size, overlap)]


def embed_texts(texts):
    model = get_model()
    vecs = model.encode(texts, batch_size=8, show_progress_bar=True, normalize_embeddings=False)
    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"向量维度异常: {vecs.shape}")
    faiss.normalize_L2(vecs)
    return vecs


def _infer_source_type(fname: str) -> str:
    """根据文件名推断 source_type"""
    name_lower = fname.lower().replace(".txt", "")
    if name_lower in ("faq", "faq_", "常见问题"):
        return "faq"
    if name_lower in ("alias", "aliases", "别名"):
        return "alias"
    return "main"


def _collect_txt_files(pdir):
    """
    收集产品目录及子目录下所有 .txt 文件。
    支持子目录分类，如：
      knowledge/buddhism/
        main.txt              → 通用知识
        faq.txt               → FAQ
        alias.txt             → 别名
        入菩萨行论.txt          → 独立论典
        讲记/                  → 子目录
          菩提心讲记.txt
          禅修开示.txt
        仪轨/
          金刚萨埵修法.txt
    返回: [(文件路径, source_type, 显示名), ...]
    """
    results = []

    # 1. 根目录下的文件
    for fp in sorted(pdir.glob("*.txt")):
        stype = _infer_source_type(fp.name)
        results.append((fp, stype, fp.name))

    # 2. 子目录下的文件（一级子目录）
    for subdir in sorted(pdir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for fp in sorted(subdir.glob("*.txt")):
            stype = _infer_source_type(fp.name)
            # 显示名带上子目录前缀，如 "讲记/菩提心讲记.txt"
            display_name = f"{subdir.name}/{fp.name}"
            results.append((fp, stype, display_name))

    return results


def collect_product_records(product: str):
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise FileNotFoundError(f"未找到产品目录：{pdir}")

    all_files = _collect_txt_files(pdir)
    if not all_files:
        raise FileNotFoundError(f"目录为空：{pdir}")
    records = []
    for fpath, stype, display_name in all_files:
        text = fpath.read_text(encoding="utf-8")

        if stype == "alias":
            chunks_data = [{"text": text}]
        else:
            # 统一使用智能分块
            ctype = detect_content_type(text)
            type_names = {
                "kepan": "科判", "pin": "品", "ritual": "仪轨",
                "qa": "问答体", "gongan": "公案/语录", "commentary": "注疏体",
                "verse_collection": "偈颂集", "letter": "书信体",
                "method": "方法步骤", "talk": "演讲/开示",
                "article": "文章", "plain": "通用",
            }
            print(f"[INFO] {product}/{display_name}: 检测到「{type_names.get(ctype, ctype)}」结构")
            chunks_data = chunk_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # 过滤低质量 chunk：太短（<30字）或纯标记/元数据
        before_filter = len(chunks_data)
        chunks_data = [
            cd for cd in chunks_data
            if len(re.sub(r"[\s【】\[\]()（）《》「」\-—·：:、，。？！]", "", cd.get("text", ""))) >= 20
        ]
        if len(chunks_data) < before_filter:
            print(f"[FILTER] {product}/{display_name}: 过滤 {before_filter - len(chunks_data)} 个低质量 chunk")
        print(f"[OK] {product}/{display_name}: {len(chunks_data)} chunks")
        for i, cd in enumerate(chunks_data, 1):
            meta = {
                "product_id": product,
                "source_file": display_name,
                "source_type": stype,
                "chunk_id": f"{display_name}#{i}",
            }
            # 保存结构元数据（科判/品/内容类型）
            if "kepan_breadcrumb" in cd:
                meta["kepan_breadcrumb"] = cd["kepan_breadcrumb"]
                meta["kepan_marker"] = cd.get("kepan_marker", "")
                meta["kepan_title"] = cd.get("kepan_title", "")
            if "pin_title" in cd:
                meta["pin_title"] = cd["pin_title"]
            if "content_type" in cd:
                meta["content_type"] = cd["content_type"]
            if "section_title" in cd:
                meta["section_title"] = cd["section_title"]
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
