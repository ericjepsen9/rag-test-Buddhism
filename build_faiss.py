import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rag_runtime_config import (
    KNOWLEDGE_DIR, STORE_ROOT,
    EMBED_MODEL_NAME, EMBED_USE_FP16, EMBED_BATCH_SIZE_BUILD, EMBED_MAX_LENGTH_BUILD,
    CHUNK_SIZE as _DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP as _DEFAULT_CHUNK_OVERLAP,
    SHARED_ENTITY_DIRS as _SHARED_ENTITY_DIRS,
    FAISS_INDEX_TYPE, FAISS_HNSW_M, FAISS_HNSW_EF_CONSTRUCTION, FAISS_HNSW_EF_SEARCH,
)

# 预计算共享目录名，避免 _is_product_dir 每次调用重建 set
_SHARED_DIR_NAMES = frozenset(_SHARED_ENTITY_DIRS.values())

from search_utils import (
    normalize_text, has_kepan_structure, split_by_kepan,
    has_pin_structure, split_by_pin, split_semantic_paragraphs,
    detect_content_type, split_by_topic, split_by_ritual_section,
    split_by_steps, split_by_headings,
    split_by_qa, split_by_gongan, split_by_commentary,
    split_by_verse_collection, split_by_letter,
    split_mixed_body, merge_sub_segments,
)

MODEL_NAME = EMBED_MODEL_NAME
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", str(_DEFAULT_CHUNK_SIZE)))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", str(_DEFAULT_CHUNK_OVERLAP)))
MIN_CHUNK_CHARS = 30

_model = None
_np = None
_faiss = None


# ====== 懒加载 numpy / faiss ======

def _get_np():
    global _np
    if _np is None:
        import numpy as _np_mod
        _np = _np_mod
    return _np


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_mod
        _faiss = _faiss_mod
    return _faiss


# ====== 模型加载 ======

def get_model():
    global _model
    if _model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"[INFO] 加载模型（BGEM3FlagModel）：{MODEL_NAME}")
            _model = BGEM3FlagModel(MODEL_NAME, use_fp16=EMBED_USE_FP16)
        except ImportError:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] FlagEmbedding 不可用，回退 SentenceTransformer：{MODEL_NAME}")
            _model = SentenceTransformer(MODEL_NAME)
    return _model


# ====== 多编码读取 ======

def read_text_auto(p: Path) -> str:
    """多编码读取：依次尝试 utf-8-sig、gbk，最后 errors='replace'"""
    if not p.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
        try:
            return p.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return p.read_text(errors="replace")


# ====== FAQ 问答对拆分 ======

_RE_FAQ_QA = re.compile(r"【Q】(.*?)(?=【Q】|\Z)", re.DOTALL)
_RE_FAQ_A = re.compile(r"【A】(.*)", re.DOTALL)


def split_faq_pairs(text: str) -> List[dict]:
    """将 FAQ 文本拆分为问答对，返回 [{"q": ..., "a": ..., "full": ...}]。
    每个 FAQ 条目生成两条记录：
    - 问题文本（用于向量检索时更好匹配用户问题）
    - 完整问答对（用于答案生成）
    """
    pairs = []
    for m in _RE_FAQ_QA.finditer(text):
        block = m.group(1).strip()
        lines = block.split("\n")
        q_text = lines[0].strip() if lines else ""
        a_match = _RE_FAQ_A.search(block)
        a_text = a_match.group(1).strip() if a_match else ""
        full_text = f"【Q】{q_text}\n【A】{a_text}"
        if q_text and a_text:
            pairs.append({"q": q_text, "a": a_text, "full": full_text})
    return pairs


# ====== 基础滑动窗口切块 ======

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


# ====== 内容感知子块切分 ======

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


# ====== 科判结构切分 ======

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


# ====== 品结构切分 ======

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


# ====== 通用分节切块 ======

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


# ====== 智能分块 ======

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


# ====== 向量编码 ======

def embed_texts(texts):
    """编码文本列表为向量。支持 BGEM3FlagModel (dict 输出) 和 SentenceTransformer。
    大批量时分批编码并报告进度。"""
    if not texts:
        raise ValueError("embed_texts: 输入文本列表为空")
    # 空白文本用占位符替换，避免模型编码异常或维度不匹配
    texts = [t if t and t.strip() else " " for t in texts]
    model = get_model()
    np = _get_np()
    faiss = _get_faiss()

    # 分批编码，报告进度
    n = len(texts)
    bs = EMBED_BATCH_SIZE_BUILD
    all_vecs = []
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = texts[start:end]
        if n > bs:
            print(f"[PROGRESS] Embedding batch {start // bs + 1}/{(n + bs - 1) // bs} ({end}/{n} chunks)")
        out = model.encode(batch, batch_size=bs, max_length=EMBED_MAX_LENGTH_BUILD)

        # 处理 dict 输出（BGEM3FlagModel 返回 dict）
        batch_vecs = None
        if isinstance(out, dict):
            if out.get("dense_vecs") is not None:
                batch_vecs = out["dense_vecs"]
            elif out.get("dense") is not None:
                batch_vecs = out["dense"]
            elif out.get("embeddings") is not None:
                batch_vecs = out["embeddings"]
        else:
            batch_vecs = out

        if batch_vecs is None:
            raise ValueError("encode 输出中未找到向量字段")

        batch_vecs = np.asarray(batch_vecs, dtype="float32")
        if batch_vecs.ndim != 2:
            raise ValueError(f"向量维度异常: {batch_vecs.shape}")
        if batch_vecs.shape[0] != len(batch):
            raise ValueError(f"向量行数({batch_vecs.shape[0]})与文本数({len(batch)})不一致")
        all_vecs.append(batch_vecs)

    vecs = np.concatenate(all_vecs, axis=0) if len(all_vecs) > 1 else all_vecs[0]
    faiss.normalize_L2(vecs)
    return vecs


# ====== 文件与类型推断 ======

def _infer_source_type(fname: str) -> str:
    """根据文件名推断 source_type"""
    name_lower = fname.lower().replace(".txt", "")
    if name_lower in ("faq", "faq_", "常见问题") or name_lower.startswith("faq"):
        return "faq"
    if name_lower in ("alias", "aliases", "别名") or name_lower.startswith("alias"):
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


# ====== 跨来源去重 ======

def _dedup_records(records: List[dict]) -> List[dict]:
    """跨来源去重：相同文本内容（忽略空白差异）只保留第一个来源的记录"""
    import hashlib as _hl
    seen_hashes = set()
    deduped = []
    for r in records:
        normalized = " ".join(r["text"].split())
        digest = _hl.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        deduped.append(r)
    return deduped


# ====== 收集产品记录 ======

def collect_product_records(product: str):
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise FileNotFoundError(f"未找到产品目录：{pdir}")

    all_files = _collect_txt_files(pdir)
    if not all_files:
        raise FileNotFoundError(f"目录为空：{pdir}")
    records = []
    for fpath, stype, display_name in all_files:
        text = read_text_auto(fpath)

        if stype == "alias":
            # 别名文件不需要索引（仅用于 FAQ 匹配），跳过
            print(f"[SKIP] {product}/{display_name}: 别名文件不索引")
            continue
        elif stype == "faq":
            # FAQ 独立嵌入：按问答对拆分，问题和完整问答分别入库
            faq_pairs = split_faq_pairs(text)
            if faq_pairs:
                faq_q_count = 0
                for i, pair in enumerate(faq_pairs, 1):
                    # 完整问答对（用于答案生成和 FAQ 快速路径）
                    records.append({
                        "text": pair["full"],
                        "meta": {
                            "product_id": product,
                            "source_file": display_name,
                            "source_type": "faq",
                            "chunk_id": f"{display_name}#faq{i}",
                        }
                    })
                    # 问题文本独立嵌入（向量检索时更好匹配用户问题）
                    if len(pair["q"]) >= MIN_CHUNK_CHARS:
                        faq_q_count += 1
                        records.append({
                            "text": pair["q"],
                            "meta": {
                                "product_id": product,
                                "source_file": display_name,
                                "source_type": "faq_question",
                                "chunk_id": f"{display_name}#faqq{i}",
                                "faq_answer": pair["a"],
                            }
                        })
                print(f"[OK] {product}/{display_name}: {len(faq_pairs)} QA pairs + {faq_q_count} question embeddings")
            else:
                # 回退：无法解析 FAQ 格式时按普通文本切块
                ctype = detect_content_type(text)
                print(f"[INFO] {product}/{display_name}: FAQ 格式未识别，按「{ctype}」结构回退分块")
                chunks_data = chunk_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)
                chunks_data = _filter_low_quality(chunks_data, product, display_name)
                print(f"[OK] {product}/{display_name}: {len(chunks_data)} chunks (faq fallback)")
                for i, cd in enumerate(chunks_data, 1):
                    meta = {
                        "product_id": product,
                        "source_file": display_name,
                        "source_type": stype,
                        "chunk_id": f"{display_name}#{i}",
                    }
                    _attach_buddhist_meta(meta, cd)
                    records.append({"text": cd["text"], "meta": meta})
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

            # 过滤低质量 chunk
            chunks_data = _filter_low_quality(chunks_data, product, display_name)

            print(f"[OK] {product}/{display_name}: {len(chunks_data)} chunks")
            for i, cd in enumerate(chunks_data, 1):
                meta = {
                    "product_id": product,
                    "source_file": display_name,
                    "source_type": stype,
                    "chunk_id": f"{display_name}#{i}",
                }
                # 保存结构元数据（科判/品/内容类型）
                _attach_buddhist_meta(meta, cd)
                records.append({
                    "text": cd["text"],
                    "meta": meta,
                })

    if not records:
        raise ValueError(f"{product} 没有可用文本")

    # 跨来源去重
    before = len(records)
    records = _dedup_records(records)
    if len(records) < before:
        print(f"[INFO] {product}: 跨来源去重 {before} → {len(records)} records")

    return records


def _filter_low_quality(chunks_data: List[Dict], product: str, display_name: str) -> List[Dict]:
    """过滤低质量 chunk：太短（<20 有效字符）或纯标记/元数据"""
    before_filter = len(chunks_data)
    chunks_data = [
        cd for cd in chunks_data
        if len(re.sub(r"[\s【】\[\]()（）《》「」\-—·：:、，。？！]", "", cd.get("text", ""))) >= 20
    ]
    if len(chunks_data) < before_filter:
        print(f"[FILTER] {product}/{display_name}: 过滤 {before_filter - len(chunks_data)} 个低质量 chunk")
    return chunks_data


def _attach_buddhist_meta(meta: dict, cd: dict):
    """将 Buddhist 结构元数据（科判/品/内容类型/章节标题）附加到 meta"""
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


# ====== FAISS 索引创建 ======

def _create_faiss_index(dim: int, n_vectors: int):
    """根据配置创建 FAISS 索引：支持 flat 和 hnsw 两种类型。
    小规模数据（< 100 条）始终使用 flat 避免 HNSW 开销。"""
    faiss = _get_faiss()
    if FAISS_INDEX_TYPE == "hnsw" and n_vectors >= 100:
        index = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
        print(f"[INFO] 使用 HNSW 索引 (M={FAISS_HNSW_M}, efC={FAISS_HNSW_EF_CONSTRUCTION}, efS={FAISS_HNSW_EF_SEARCH})")
        return index
    return faiss.IndexFlatIP(dim)


# ====== 构建索引 ======

def build_for_product(product: str):
    records = collect_product_records(product)
    texts = [r["text"] for r in records]
    print(f"[INFO] Total chunks: {len(texts)}")
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    vecs = embed_texts(texts)
    dim = vecs.shape[1]
    index = _create_faiss_index(dim, len(texts))
    index.add(vecs)

    out_dir = STORE_ROOT / product
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    # 原子写入：先写临时文件，再 rename，防止进程中断导致文件损坏
    tmp_docs = out_dir / "docs.jsonl.tmp"
    tmp_index = out_dir / "index.faiss.tmp"
    with tmp_docs.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _get_faiss().write_index(index, str(tmp_index))
    # 备份旧 docs 以便回滚（如果 index replace 失败）
    docs_backup = out_dir / "docs.jsonl.bak"
    if docs_path.exists():
        try:
            os.replace(str(docs_path), str(docs_backup))
        except OSError:
            docs_backup = None
    else:
        docs_backup = None
    os.replace(str(tmp_docs), str(docs_path))
    try:
        os.replace(str(tmp_index), str(index_path))
    except Exception:
        # index 写入失败，回滚 docs 以保持一致性
        if docs_backup and docs_backup.exists():
            os.replace(str(docs_backup), str(docs_path))
        raise
    # 清理备份
    if docs_backup and docs_backup.exists():
        docs_backup.unlink(missing_ok=True)

    print(f"[DONE] Built store")
    print(f"       product: {product}")
    print(f"       chunks : {len(records)}")
    print(f"       dim    : {dim}")
    print(f"       index  : {FAISS_INDEX_TYPE}")


def collect_shared_records():
    """收集所有共享知识实体的文本记录。"""
    from rag_runtime_config import SHARED_ENTITY_DIRS
    records = []
    for entity_type, subdir in SHARED_ENTITY_DIRS.items():
        edir = KNOWLEDGE_DIR / subdir
        if not edir.exists():
            continue
        # 两种结构：1) subdir/main.txt (单文件实体) 2) subdir/{name}/main.txt (多实例)
        main_file = edir / "main.txt"
        if main_file.exists():
            # 单文件实体
            text = read_text_auto(main_file)
            chunks_data = chunk_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"[OK] {subdir}/main.txt: {len(chunks_data)} chunks")
            for i, cd in enumerate(chunks_data, 1):
                meta = {
                    "product_id": "_shared",
                    "entity_type": entity_type,
                    "source_file": f"{subdir}/main.txt",
                    "source_type": entity_type,
                    "chunk_id": i,
                }
                _attach_buddhist_meta(meta, cd)
                records.append({"text": cd["text"], "meta": meta})
        # 多实例子目录
        for inst in sorted(edir.iterdir()):
            if not inst.is_dir():
                continue
            for fname, stype in [("main.txt", "main"), ("faq.txt", "faq")]:
                f = inst / fname
                if not f.exists():
                    continue
                text = read_text_auto(f)
                chunks_data = chunk_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)
                label = f"{subdir}/{inst.name}/{fname}"
                print(f"[OK] {label}: {len(chunks_data)} chunks")
                for i, cd in enumerate(chunks_data, 1):
                    meta = {
                        "product_id": "_shared",
                        "entity_type": entity_type,
                        "entity_id": inst.name,
                        "source_file": label,
                        "source_type": stype,
                        "chunk_id": i,
                    }
                    _attach_buddhist_meta(meta, cd)
                    records.append({"text": cd["text"], "meta": meta})
    # 跨来源去重
    before = len(records)
    records = _dedup_records(records)
    if len(records) < before:
        print(f"[INFO] shared: 跨来源去重 {before} → {len(records)} records")
    return records


def build_shared():
    """构建共享知识索引（存储在 stores/_shared/）"""
    records = collect_shared_records()
    if not records:
        print("[WARN] 无共享知识可索引")
        return
    texts = [r["text"] for r in records]
    print(f"[INFO] Shared total chunks: {len(texts)}")
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    vecs = embed_texts(texts)
    dim = vecs.shape[1]
    index = _create_faiss_index(dim, len(texts))
    index.add(vecs)

    out_dir = STORE_ROOT / "_shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    # 原子写入：先写临时文件，再 rename
    tmp_docs = out_dir / "docs.jsonl.tmp"
    tmp_index = out_dir / "index.faiss.tmp"
    with tmp_docs.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _get_faiss().write_index(index, str(tmp_index))
    # 备份旧 docs 以便回滚
    docs_backup = out_dir / "docs.jsonl.bak"
    if docs_path.exists():
        try:
            os.replace(str(docs_path), str(docs_backup))
        except OSError:
            docs_backup = None
    else:
        docs_backup = None
    os.replace(str(tmp_docs), str(docs_path))
    try:
        os.replace(str(tmp_index), str(index_path))
    except Exception:
        if docs_backup and docs_backup.exists():
            os.replace(str(docs_backup), str(docs_path))
        raise
    if docs_backup and docs_backup.exists():
        docs_backup.unlink(missing_ok=True)

    print(f"[DONE] Built shared store ({len(records)} chunks, dim={dim})")


def _is_product_dir(p: Path) -> bool:
    """顶层目录且有 main.txt，且不是共享知识目录"""
    return p.is_dir() and (p / "main.txt").exists() and p.name not in _SHARED_DIR_NAMES


def list_products():
    if not KNOWLEDGE_DIR.exists():
        print(f"[ERROR] knowledge 目录不存在：{KNOWLEDGE_DIR}")
        return
    # 列出产品目录（顶层有 main.txt 且不是共享知识目录）
    for p in sorted(KNOWLEDGE_DIR.iterdir()):
        if _is_product_dir(p):
            print(f"[product] {p.name}")
    # 列出共享知识目录
    for entity_type, subdir in _SHARED_ENTITY_DIRS.items():
        edir = KNOWLEDGE_DIR / subdir
        if not edir.exists():
            continue
        if (edir / "main.txt").exists():
            print(f"[{entity_type}] {subdir}/")
        for inst in sorted(edir.iterdir()):
            if inst.is_dir() and (inst / "main.txt").exists():
                print(f"[{entity_type}] {subdir}/{inst.name}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=str, help="产品目录名")
    ap.add_argument("--shared", action="store_true", help="构建共享知识索引")
    ap.add_argument("--all", action="store_true", help="构建所有产品+共享知识")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_products()
        return
    if args.all:
        # 构建所有产品（排除共享知识目录）
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if _is_product_dir(p):
                print(f"\n{'='*40}\n构建产品: {p.name}\n{'='*40}")
                build_for_product(p.name)
        # 构建共享知识
        print(f"\n{'='*40}\n构建共享知识\n{'='*40}")
        build_shared()
        return
    if args.shared:
        build_shared()
        return
    if not args.product:
        ap.error("请使用 --product <name> / --shared / --all / --list")
    build_for_product(args.product.strip())


if __name__ == "__main__":
    main()
