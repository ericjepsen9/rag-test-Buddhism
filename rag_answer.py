"""
Buddhist RAG answer pipeline.

Merged architecture from the reference medical-aesthetics implementation,
adapted for Buddhist domain knowledge.  Key features carried over:
- Thread safety with threading.Lock (store cache, model loading)
- ThreadPoolExecutor for parallel vector + keyword search
- Embedding cache (dict-based LRU)
- FAQ fast path (bigram overlap matching from search hits)
- _build_context helper function
- Multi-provider LLM support via llm_client module
- llm_generate_answer with conversation history support
- _fallback_from_hits for low-confidence scenarios
- Knowledge gap logging
- Chitchat and special intent detection (Buddhist equivalents)
- Enhanced detect_route with disambiguation and weighted scoring

Buddhist-specific features preserved:
- CrossEncoder rerank integration
- Buddhist dedup logic
- Buddhist context building with kepan breadcrumbs
- Buddhist topic suggestions
- All Buddhist route names
- Buddhist QUESTION_ROUTES, SECTION_RULES from rag_runtime_config
- answer_formatter.format_structured_answer usage
- relation_engine.enrich_answer usage
- media_router.find_media usage
"""

import os
import sys
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from rag_runtime_config import (
    KNOWLEDGE_DIR, STORE_ROOT, OUT_PATH, DEFAULT_MODE, DEFAULT_TOP_K,
    USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE, DEBUG, QUESTION_ROUTES, SECTION_RULES,
    PRODUCT_ALIASES, PROJECT_ALIASES, VECTOR_TOP_K, KEYWORD_TOP_K,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, FAQ_KEYWORD_MAP,
    SCORE_THRESHOLD, QUESTION_TYPE_CONFIG, ANSWER_MODE_CONFIG,
    DEFAULT_PRODUCT, USE_RERANK, RERANK_MODEL, RERANK_TOP_K,
    RERANK_SCORE_THRESHOLD,
    MAX_SUB_QUESTIONS, MAX_EVIDENCE_CHUNKS,
    EMBED_MODEL_NAME, EMBED_USE_FP16, EMBED_BATCH_SIZE_QUERY, EMBED_MAX_LENGTH_QUERY,
    EMBED_BATCH_SIZE_BUILD, EMBED_MAX_LENGTH_BUILD,
    LLM_TEMPERATURE, LLM_MAX_TOKENS_BRIEF, LLM_MAX_TOKENS_FULL, ROUTE_LLM_TEMPERATURE,
    RELATIONS_FILE,
    PRICE_REPLY, COMPARISON_REPLY, LOCATION_REPLY,
    FAQ_FAST_PATH_THRESHOLDS, FAQ_FAST_PATH_DEFAULT,
    RERANK_ENABLED, RERANK_TOP_N,
    DYNAMIC_THRESHOLD_ENABLED, DYNAMIC_THRESHOLD_RATIO, DYNAMIC_THRESHOLD_FLOOR_RATIO,
)
from search_utils import (
    normalize_text, normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms, match_faq,
    rerank_hits as bgem3_rerank_hits, compute_dynamic_threshold, expand_synonyms,
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer
from relation_engine import enrich_answer as relation_enrich
from media_router import find_media as media_find
from rag_logger import log_qa, log_error

# ---------------------------------------------------------------------------
# Pre-computed lowercase keyword map for detect_route (avoid per-call .lower())
# ---------------------------------------------------------------------------
_QUESTION_ROUTES_LOWER: Dict[str, List[str]] = {
    route: [kw.lower() for kw in keywords]
    for route, keywords in QUESTION_ROUTES.items()
}

# ---------------------------------------------------------------------------
# Module-level singletons and caches
# ---------------------------------------------------------------------------
_model = None
_faiss = None
_store_cache: Dict[str, tuple] = {}   # {product: (index, docs, mtime)}
_store_lock = threading.Lock()
_store_product_locks: Dict[str, threading.Lock] = {}
_store_product_locks_guard = threading.Lock()
_STORE_PRODUCT_LOCKS_MAX = 100  # prevent unbounded growth
_reranker = None  # cross-encoder reranker model cache

# Module-level thread pool for parallel search
_search_pool = ThreadPoolExecutor(max_workers=6)

# Thread-local storage for last route/product
_thread_local = threading.local()

# ---------------------------------------------------------------------------
# Buddhist domain signal rules for detect_route disambiguation
# ---------------------------------------------------------------------------
_ROUTE_ORDER = [
    "life", "scripture", "doctrine", "practice", "sect", "concept",
    "history", "ritual", "basic",
]
_ROUTE_ORDER_IDX = {r: i for i, r in enumerate(_ROUTE_ORDER)}

# Buddhist disambiguation signals
_DOCTRINE_SIGNALS = ("四圣谛", "八正道", "十二因缘", "缘起", "空性", "中观",
                     "唯识", "如来藏", "般若", "涅槃", "三法印", "四法印")
_PRACTICE_SIGNALS = ("禅修", "打坐", "冥想", "念佛", "持咒", "观想", "止观",
                     "正念", "精进", "戒律", "布施", "持戒", "忍辱")
_SCRIPTURE_SIGNALS = ("经", "论", "律", "心经", "金刚经", "法华经", "楞严经",
                      "华严经", "阿含经", "维摩经", "入行论", "菩提道次第")
_SECT_SIGNALS = ("宗派", "禅宗", "净土宗", "天台宗", "华严宗", "密宗",
                 "藏传", "南传", "上座部", "大乘", "小乘", "传承")
_CONCEPT_SIGNALS = ("概念", "含义", "是什么", "什么意思", "定义", "解释",
                    "菩提心", "佛性", "法身", "般若", "三宝", "五蕴")
_HISTORY_SIGNALS = ("历史", "朝代", "传入", "发展", "祖师", "高僧",
                    "达摩", "玄奘", "鸠摩罗什", "佛教史")
_RITUAL_SIGNALS = ("仪轨", "法会", "供养", "礼拜", "早课", "晚课",
                   "回向", "发愿", "忏悔", "法事", "放生")
_LIFE_SIGNALS = ("生活", "日常", "工作", "职场", "压力", "焦虑", "情绪", "脾气",
                 "愤怒", "生气", "发火", "家庭", "婚姻", "夫妻", "孩子", "父母",
                 "人际关系", "同事", "吵架", "矛盾", "误解", "冤枉", "委屈",
                 "内疚", "自责", "抑郁", "迷茫", "躺平", "没有动力", "拖延",
                 "手机", "上瘾", "注意力", "分心", "怎么办", "怎么面对",
                 "死亡", "去世", "离世", "临终", "换位思考", "同理心")

# Chitchat regex patterns — 使用非锚定模式支持句中匹配
_RE_CHAT_GREETING = re.compile(r"(你好|嗨|hi|hello|hey|您好|在吗|在不在)", re.IGNORECASE)
_RE_CHAT_THANKS = re.compile(r"(谢谢|感谢|多谢|辛苦了|谢啦|thx|thanks)", re.IGNORECASE)
_RE_CHAT_BYE = re.compile(r"(再见|拜拜|bye|回头见|下次再聊)", re.IGNORECASE)

# _fallback_from_hits pre-compiled patterns
_RE_FALLBACK_SPLIT = re.compile(r"[\s,，;；、？?！!。【】]+")
_RE_CJK_SINGLE = re.compile(r"[\u4e00-\u9fff]")
_RE_CN_SECTION_TITLE = re.compile(r"^[一二三四五六七八九十]+、")
_SEPARATOR_CHARS = frozenset("=-_ —")

# Special intent keywords (Buddhist equivalents)
_PRICE_KWS = ("多少钱", "价格", "费用", "收费")
_COMPARE_KWS = ("区别", "对比", "哪个好", "差别", "比较")
_LOCATION_KWS = ("哪里可以", "哪家寺院", "附近", "哪里有", "去哪",
                  "哪个城市", "推荐寺院", "哪里学佛")


# ===================================================================
# CrossEncoder rerank (Buddhist feature)
# ===================================================================

def get_reranker():
    """Lazy-load cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        if not USE_RERANK:
            return None
        try:
            from sentence_transformers import CrossEncoder
            if DEBUG:
                print(f"[INFO] Loading reranker model: {RERANK_MODEL}")
            _reranker = CrossEncoder(RERANK_MODEL, max_length=1024)
        except Exception as e:
            log_error("reranker_load", repr(e))
            if DEBUG:
                print(f"[WARN] Reranker load failed, falling back: {e}")
            return None
    return _reranker


def rerank_hits(query: str, hits: List[Dict], top_k: int = None,
                score_threshold: float = None) -> List[Dict]:
    """
    Cross-encoder rerank of retrieval results.
    Bi-encoder (vector search): encodes query and doc separately, fast but less precise.
    Cross-encoder (rerank): encodes query+doc together, more precise but slower.
    Pipeline: bi-encoder coarse recall -> cross-encoder rerank -> top-K
    """
    if not hits:
        return hits
    if top_k is None:
        top_k = RERANK_TOP_K
    if score_threshold is None:
        score_threshold = RERANK_SCORE_THRESHOLD

    reranker = get_reranker()
    if reranker is None:
        return hits[:top_k]

    try:
        pairs = []
        for h in hits:
            text = h.get("text", "").strip()
            if not text:
                text = "(empty)"
            pairs.append((query, text))

        scores = reranker.predict(pairs, show_progress_bar=False)

        for h, score in zip(hits, scores):
            h["rerank_score"] = float(score)

        hits_sorted = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        result = [h for h in hits_sorted if h.get("rerank_score", 0.0) >= score_threshold]
        return result[:top_k]
    except Exception as e:
        log_error("rerank_hits", repr(e))
        if DEBUG:
            print(f"[WARN] Rerank failed, falling back to original order: {e}")
        return hits[:top_k]


# ===================================================================
# Model / FAISS loading (thread-safe)
# ===================================================================

_faiss_lock = threading.Lock()


def get_faiss():
    global _faiss
    if _faiss is None:
        with _faiss_lock:
            if _faiss is None:
                import faiss as _faiss_mod
                _faiss = _faiss_mod
    return _faiss


_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        if DEBUG:
                            print(f"[INFO] CUDA available, using GPU: {gpu_name}")
                    else:
                        if DEBUG:
                            print("[INFO] CUDA not available, model will run on CPU")
                except ImportError:
                    if DEBUG:
                        print("[INFO] PyTorch not installed, cannot detect GPU")
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


# ===================================================================
# Output helpers
# ===================================================================

def _get_out_path() -> Path:
    """Support RAG_ANSWER_FILE env var for output path (avoid concurrent overwrite)."""
    env_path = os.environ.get("RAG_ANSWER_FILE", "").strip()
    if env_path:
        return Path(env_path)
    return OUT_PATH


def save_answer(text: str):
    _get_out_path().write_text((text or "").strip() + "\n", encoding="utf-8-sig")


# ===================================================================
# Embedding cache
# ===================================================================

_embed_cache: OrderedDict = OrderedDict()
_EMBED_CACHE_MAX = 1024
_embed_cache_lock = threading.Lock()


def embed_query(text: str) -> np.ndarray:
    with _embed_cache_lock:
        cached = _embed_cache.get(text)
        if cached is not None:
            _embed_cache.move_to_end(text)  # LRU: mark as recently used
            return cached.copy()

    model = get_model()
    vec = model.encode([text], normalize_embeddings=False)
    vec = np.asarray(vec, dtype="float32")
    get_faiss().normalize_L2(vec)

    # Write to cache with LRU eviction (thread-safe)
    with _embed_cache_lock:
        if text in _embed_cache:
            _embed_cache[text] = vec.copy()
            _embed_cache.move_to_end(text)
        else:
            while len(_embed_cache) >= _EMBED_CACHE_MAX:
                try:
                    _embed_cache.popitem(last=False)
                except KeyError:
                    break
            _embed_cache[text] = vec.copy()
    return vec


# ===================================================================
# Store loading (thread-safe with per-product locks)
# ===================================================================

def _evict_cache(cache: dict, max_size: int) -> None:
    """Generic cache eviction: remove oldest entries when over limit."""
    while len(cache) >= max_size:
        try:
            oldest = next(iter(cache))
            cache.pop(oldest, None)
        except (StopIteration, RuntimeError):
            break


def invalidate_store_cache(product: str) -> None:
    """Thread-safe store cache invalidation (index + BM25)."""
    with _store_lock:
        _store_cache.pop(product, None)
    # Also clear BM25 corpus cache to avoid stale DF/avgDL data
    try:
        from search_utils import invalidate_bm25_cache
        invalidate_bm25_cache(product)
    except Exception:
        pass


def _auto_rebuild_index(product: str, docs: List[Dict], store_dir: Path):
    """Auto-rebuild FAISS index when docs.jsonl count != index.faiss vector count.
    Uses the already-loaded embed model. Returns new index or None on failure."""
    try:
        texts = [(d.get("text") or "").strip() for d in docs]
        texts = [t if t else " " for t in texts]  # placeholder for empty text
        model = get_model()
        vec = model.encode(texts, normalize_embeddings=False)
        if isinstance(vec, dict):
            vecs = vec.get("dense_vecs") or vec.get("dense") or vec.get("embeddings")
        else:
            vecs = vec
        if vecs is None:
            log_error("index_rebuild", "Failed to get vectors",
                      meta={"product": product})
            return None
        vecs = np.asarray(vecs, dtype="float32")
        faiss = get_faiss()
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        # Write back to disk
        index_path = store_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        if DEBUG:
            print(f"[INFO] Auto-rebuilt index for {product}: {len(docs)} vectors, dim={dim}")
        return index
    except Exception as e:
        log_error("index_rebuild", f"Auto-rebuild failed: {e}",
                  meta={"product": product})
        return None


def load_store(product: str):
    """Load vector index and docs with thread-safe mtime-based caching."""
    store_dir = STORE_ROOT / product
    index_path = store_dir / "index.faiss"
    docs_path = store_dir / "docs.jsonl"
    if not docs_path.exists():
        return None, []

    mtime = docs_path.stat().st_mtime
    with _store_lock:
        cached = _store_cache.get(product)
        if cached and cached[2] == mtime:
            return cached[0], cached[1]

    # Per-product lock to prevent concurrent loading of same product
    with _store_product_locks_guard:
        if product not in _store_product_locks:
            # Prevent unbounded growth: evict locks for products not in cache
            if len(_store_product_locks) >= _STORE_PRODUCT_LOCKS_MAX:
                cached_products = set(_store_cache.keys())
                for k in list(_store_product_locks):
                    if k not in cached_products:
                        del _store_product_locks[k]
            _store_product_locks[product] = threading.Lock()
        product_lock = _store_product_locks[product]

    with product_lock:
        # Double-check after acquiring lock
        with _store_lock:
            cached = _store_cache.get(product)
            if cached and cached[2] == mtime:
                return cached[0], cached[1]

        # Load documents
        skipped = 0
        docs = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    meta = doc.get("meta", {})
                    if not meta.get("chunk_id"):
                        meta["chunk_id"] = f"_doc{line_num}"
                        doc["meta"] = meta
                    docs.append(doc)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
        if skipped > 0:
            log_error("load_store", f"Skipped {skipped} corrupted lines",
                      meta={"product": product, "docs_path": str(docs_path)})

        # Load FAISS index
        index = None
        if index_path.exists():
            try:
                index = get_faiss().read_index(str(index_path))
                # Auto-rebuild if doc count != index vector count
                if index.ntotal != len(docs):
                    if DEBUG:
                        print(f"[INFO] Index/doc count mismatch for {product}: "
                              f"index={index.ntotal}, docs={len(docs)}, rebuilding...")
                    index = _auto_rebuild_index(product, docs, store_dir)
            except Exception as e:
                log_error("load_store", f"Index load failed: {e}",
                          meta={"product": product, "index_path": str(index_path)})
                index = None

        # Write to cache
        with _store_lock:
            _store_cache[product] = (index, docs, mtime)
    return index, docs


# ===================================================================
# Search functions
# ===================================================================

def vector_search(product: str, query: str, top_k: int) -> List[Dict]:
    index, docs = load_store(product)
    if index is None or not docs:
        return []
    qv = embed_query(query)
    if qv.shape[1] != index.d:
        if DEBUG:
            print(f"[WARN] Dimension mismatch: query={qv.shape[1]}, index={index.d}")
        return []
    scores, ids = index.search(qv, min(top_k, len(docs)))
    hits = []
    for i, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(docs):
            continue
        d = dict(docs[idx])
        d["score"] = float(scores[0][i])
        hits.append(d)
    return hits


_knowledge_file_cache: Dict[str, tuple] = {}
_knowledge_file_lock = threading.Lock()
_KNOWLEDGE_CACHE_MAX = 128


def read_knowledge_file(product: str, fname: str) -> str:
    p = KNOWLEDGE_DIR / product / fname
    if not p.exists():
        return ""
    key = str(p)
    mtime = p.stat().st_mtime
    with _knowledge_file_lock:
        cached = _knowledge_file_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    content = p.read_text(encoding="utf-8", errors="replace")
    with _knowledge_file_lock:
        _evict_cache(_knowledge_file_cache, _KNOWLEDGE_CACHE_MAX)
        _knowledge_file_cache[key] = (mtime, content)
    return content


# ===================================================================
# Product / route detection
# ===================================================================

def detect_product(question: str) -> str:
    found = detect_terms(question, PRODUCT_ALIASES)
    if found:
        return found[0]
    if (KNOWLEDGE_DIR / DEFAULT_PRODUCT).exists():
        return DEFAULT_PRODUCT
    dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()] if KNOWLEDGE_DIR.exists() else []
    return dirs[0] if dirs else DEFAULT_PRODUCT


def _detect_special_intent(q: str) -> str:
    """Detect special intents without knowledge coverage: price, location.
    Note: comparison intent removed — Buddhist comparisons (大乘vs小乘,
    愿菩提心vs行菩提心) should go through normal search, not return template."""
    if any(k in q for k in _PRICE_KWS):
        return "price"
    # comparison 已移除：佛教对比类问题应走正常搜索流程
    if any(k in q for k in _LOCATION_KWS):
        return "location"
    return ""


def detect_route(question: str) -> str:
    """Enhanced multi-class route scoring with Buddhist disambiguation signals."""
    q = (question or "").lower()

    # Collect matched keywords per route (using pre-computed lowercase)
    matched = {}
    for route in _ROUTE_ORDER:
        hits = [kw for kw in _QUESTION_ROUTES_LOWER.get(route, []) if kw in q]
        if hits:
            matched[route] = hits

    if not matched:
        return "basic"

    # Weighted scoring: longer keyword matches score higher
    scores = {}
    for route, hits in matched.items():
        score = sum(max(1.0, len(kw) / 2) for kw in hits)
        scores[route] = score

    # Buddhist disambiguation boost rules
    # Collect all matched keywords from OTHER routes to avoid substring over-matching
    _signal_map = {
        "doctrine": _DOCTRINE_SIGNALS, "practice": _PRACTICE_SIGNALS,
        "scripture": _SCRIPTURE_SIGNALS, "sect": _SECT_SIGNALS,
        "concept": _CONCEPT_SIGNALS, "history": _HISTORY_SIGNALS,
        "ritual": _RITUAL_SIGNALS, "life": _LIFE_SIGNALS,
    }
    # 跟踪每个信号词被哪些路由使用，避免同一信号同时加分多个路由
    signal_used_by: Dict[str, str] = {}
    for route, signals in _signal_map.items():
        if route not in scores:
            continue
        other_kws = [kw for r, hits in matched.items() if r != route for kw in hits]
        signal_hits = [s for s in signals if s in q]
        # 排除其它路由更长关键词的子串
        independent = [s for s in signal_hits
                       if not any(s in okw and len(okw) > len(s) for okw in other_kws)]
        # 排除已被更高优先级路由使用的信号词（避免 "般若" 同时加分 doctrine 和 concept）
        independent = [s for s in independent if s not in signal_used_by]
        if independent:
            scores[route] += 5.0
            for s in independent:
                signal_used_by[s] = route

    # doctrine vs concept disambiguation
    if "doctrine" in scores and "concept" in scores:
        # If asking about a specific concept definition, prefer concept route
        if any(s in q for s in ("是什么", "什么意思", "含义", "定义")):
            scores["concept"] += 4.0
        # If asking about doctrinal system/logic, prefer doctrine route
        if any(s in q for s in ("体系", "关系", "修证", "次第")):
            scores["doctrine"] += 4.0

    # practice vs ritual disambiguation
    if "practice" in scores and "ritual" in scores:
        if any(s in q for s in ("怎么修", "如何修", "修行方法")):
            scores["practice"] += 4.0
        if any(s in q for s in ("仪轨", "法会", "仪式", "流程")):
            scores["ritual"] += 4.0

    # scripture vs doctrine: specific text reference -> scripture
    if "scripture" in scores and "doctrine" in scores:
        if any(s in q for s in ("经", "论", "律", "第几品", "哪一品", "原文")):
            scores["scripture"] += 4.0

    # life vs practice disambiguation: daily life context -> life route
    if "life" in scores and "practice" in scores:
        if any(s in q for s in ("怎么办", "怎么面对", "怎么处理", "怎么调节", "怎么化解",
                                 "生活", "日常", "工作", "家庭", "同事", "压力", "焦虑")):
            scores["life"] += 4.0
        if any(s in q for s in ("怎么修", "如何修", "修行方法", "法门", "次第")):
            scores["practice"] += 4.0

    # life vs concept disambiguation: asking about emotional/practical advice -> life
    if "life" in scores and "concept" in scores:
        if any(s in q for s in ("怎么办", "怎么面对", "怎么处理", "怎么调节")):
            scores["life"] += 4.0

    # Multi-entity mention -> sect comparison
    mentioned_projects = detect_terms(q, PROJECT_ALIASES)
    if "sect" in scores and len(mentioned_projects) >= 2:
        scores["sect"] += 6.0

    if not scores:
        return "basic"
    best = max(scores.keys(), key=lambda r: (scores[r], -_ROUTE_ORDER_IDX.get(r, 99)))
    return best


def detect_secondary_routes(question: str, primary_route: str) -> list:
    """检测次要路由：当问题同时涉及多个维度时返回次要路由列表。
    例如"入行论怎么处理家庭矛盾" → primary=scripture, secondary=[life]"""
    q = (question or "").lower()
    secondary = []

    # 检查所有路由的信号词
    _signal_map = {
        "life": _LIFE_SIGNALS,
        "scripture": _SCRIPTURE_SIGNALS,
        "doctrine": _DOCTRINE_SIGNALS,
        "practice": _PRACTICE_SIGNALS,
        "concept": _CONCEPT_SIGNALS,
    }

    for route, signals in _signal_map.items():
        if route == primary_route:
            continue
        if any(s in q for s in signals):
            secondary.append(route)

    # 限制最多2个次要路由
    return secondary[:2]


# ===================================================================
# Route config helpers
# ===================================================================

def _get_route_config(route: str) -> Dict:
    return QUESTION_TYPE_CONFIG.get(route, {"k": DEFAULT_TOP_K, "threshold": SCORE_THRESHOLD})


def _get_mode_limit(route: str, mode: str) -> int:
    mode_cfg = ANSWER_MODE_CONFIG.get(mode, ANSWER_MODE_CONFIG.get("brief", {}))
    return mode_cfg.get(route, mode_cfg.get("max_items_default", 8))


# ===================================================================
# Buddhist topic suggestions
# ===================================================================

def _suggest_related_topics(question: str, route: str) -> List[str]:
    """Suggest related topics from the knowledge base."""
    q = question.strip()
    suggestions = []
    route_kws = QUESTION_ROUTES.get(route, [])
    for kw in route_kws:
        if any(ch in q for ch in kw if ch not in "的是了在"):
            suggestions.append(kw)
    if len(suggestions) < 3:
        for r, kws in QUESTION_ROUTES.items():
            if r == route:
                continue
            for kw in kws:
                if any(ch in q for ch in kw if ch not in "的是了在"):
                    suggestions.append(kw)
                    if len(suggestions) >= 5:
                        break
            if len(suggestions) >= 5:
                break
    if len(suggestions) < 3:
        for kw in FAQ_KEYWORD_MAP:
            if any(ch in q for ch in kw if ch not in "的是了在"):
                suggestions.append(kw)
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= 5:
            break
    return unique


# ===================================================================
# Evidence building (with kepan breadcrumbs)
# ===================================================================

def _truncate_to_sentence(text: str, max_chars: int = 450) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ["。", "；", "！", "？", "）", ". ", "! ", "? ", "\n"]:
        pos = truncated.rfind(sep)
        if pos > max_chars // 2:
            return truncated[:pos + len(sep)].strip()
    return truncated.strip()


def build_evidence(hits: List[Dict]) -> List[Dict]:
    """Build evidence list with kepan breadcrumbs, deduplicated."""
    seen = set()
    ev = []
    for i, h in enumerate(hits):
        meta = h.get("meta", {})
        sf = meta.get("source_file", "")
        cid = meta.get("chunk_id", "")
        key = (sf, cid) if sf or cid else ("_idx_", str(i))
        if key in seen:
            continue
        seen.add(key)
        entry = {"meta": meta}
        if meta.get("kepan_breadcrumb"):
            entry["kepan_breadcrumb"] = meta["kepan_breadcrumb"]
        entry["text"] = _truncate_to_sentence((h.get("text") or "").strip())
        ev.append(entry)
        if len(ev) >= MAX_EVIDENCE_CHUNKS:
            break
    return ev


# ===================================================================
# Score filtering and Buddhist dedup
# ===================================================================

def filter_by_score(hits: List[Dict], threshold: float = None) -> List[Dict]:
    """Filter hits below threshold. Multi-channel: keep if any channel exceeds."""
    if threshold is None:
        threshold = SCORE_THRESHOLD
    result = []
    for h in hits:
        hybrid = h.get("hybrid_score", 0.0)
        vec_score = float(h.get("score", 0.0))
        kw_score = float(h.get("keyword_score", 0.0))
        if hybrid >= threshold or vec_score >= threshold or kw_score >= threshold:
            result.append(h)
    return result


def _text_overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    n = 4
    if len(shorter) < n:
        return 1.0 if shorter in longer else 0.0
    ngrams_short = set(shorter[i:i+n] for i in range(len(shorter) - n + 1))
    ngrams_long = set(longer[i:i+n] for i in range(len(longer) - n + 1))
    if not ngrams_short:
        return 0.0
    return len(ngrams_short & ngrams_long) / len(ngrams_short)


def _deduplicate_hits(hits: List[Dict], base_overlap_threshold: float = 0.7,
                      max_per_source: int = 3) -> List[Dict]:
    """Buddhist semantic dedup + source diversity control."""
    if not hits:
        return []
    selected = []
    source_counts: Dict[str, int] = {}
    for h in hits:
        text = h.get("text", "").strip()
        if not text:
            continue
        source = h.get("meta", {}).get("source_file", "_unknown")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        is_dup = False
        for sel in selected:
            sel_text = sel.get("text", "")
            min_len = min(len(text), len(sel_text))
            threshold = base_overlap_threshold + 0.15 * max(0, 1 - min_len / 200)
            if _text_overlap_ratio(text, sel_text) > threshold:
                is_dup = True
                break
        if is_dup:
            continue
        selected.append(h)
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected


# ===================================================================
# Context building (reference architecture)
# ===================================================================

def _build_context(hits: List[Dict], max_chars: int = 5000, min_score: float = 0.15) -> str:
    """Build LLM context string from hits, truncating at chunk boundaries.
    First 3 snippets include full metadata header; subsequent ones only get index.
    Drops hits below min_score to avoid feeding irrelevant content to LLM."""
    parts = []
    total = 0
    for i, h in enumerate(hits, 1):
        # 过滤低相关度 chunks，避免噪音干扰 LLM 生成
        score = h.get("hybrid_score") or h.get("score", 0.0)
        if i > 1 and score < min_score:
            continue
        text = (h.get("text") or "").strip()
        if not text:
            continue
        if i <= 3:
            meta = h.get("meta") or {}
            source = meta.get("source_file", "unknown")
            chunk_id = meta.get("chunk_id", "?")
            kepan = meta.get("kepan_breadcrumb", "")
            score = h.get("hybrid_score") or h.get("score", 0.0)
            header_parts = [f"[片段{i} | {source}#{chunk_id} | 相关度:{score:.2f}"]
            if kepan:
                header_parts.append(f" | 科判:{kepan}")
            header = "".join(header_parts) + "]"
        else:
            header = f"[片段{i}]"
        part = f"{header}\n{text}"
        sep_len = 2 if parts else 0
        if parts and total + sep_len + len(part) > max_chars:
            break
        parts.append(part)
        total += sep_len + len(part)
    return "\n\n".join(parts)


def extract_chunks_as_context(hits: List[Dict], max_chunks: int = 6) -> str:
    """Extract chunk text + metadata as LLM context (Buddhist kepan breadcrumbs)."""
    if not hits:
        return ""
    chunks = []
    seen = set()
    for idx, h in enumerate(hits[:max_chunks], 1):
        text = h.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        meta = h.get("meta", {})
        source = meta.get("source_file", "")
        kepan = meta.get("kepan_breadcrumb", "")
        section = meta.get("section_title", "")
        ctype = meta.get("content_type", "")
        label_parts = []
        if source:
            label_parts.append(source)
        if kepan:
            label_parts.append(f"科判：{kepan}")
        elif section:
            label_parts.append(f"章节：{section}")
        if ctype:
            label_parts.append(f"类型：{ctype}")
        label = "｜".join(label_parts) if label_parts else f"段落{idx}"
        chunks.append(f"[来源{idx}：{label}]\n{text}")
    return "\n\n---\n\n".join(chunks)


# ===================================================================
# FAQ fast path (bigram overlap from reference architecture)
# ===================================================================

def _normalize_for_bigram(text: str) -> str:
    """Normalize text for bigram matching: lowercase, strip whitespace."""
    return text.lower().replace(" ", "")


def _extract_faq_from_hits(hits: List[Dict], question: str,
                            _q_bigrams: set = None) -> List[str]:
    """Extract highly relevant FAQ answers from search hits via bigram overlap."""
    if _q_bigrams is not None:
        q_bigrams = _q_bigrams
    else:
        q_norm = _normalize_for_bigram(question)
        if len(q_norm) < 2:
            return []
        q_bigrams = set(q_norm[i:i+2] for i in range(len(q_norm) - 1))
    min_overlap = 2 if len(question) < 15 else 3

    faq_candidates = []
    for h in hits:
        meta = h.get("meta", {})
        if meta.get("source_type") != "faq":
            continue
        text = (h.get("text") or "").strip()
        if "【Q】" not in text or "【A】" not in text:
            continue
        q_part = text.split("【A】")[0].replace("【Q】", "")
        faq_norm = _normalize_for_bigram(q_part)
        faq_bigrams = set(faq_norm[i:i+2] for i in range(len(faq_norm) - 1))
        if not faq_bigrams:
            continue
        overlap = len(q_bigrams & faq_bigrams)
        ratio = overlap / max(len(q_bigrams), 1)
        if overlap >= min_overlap and ratio >= 0.3:
            _, _, a_part = text.partition("【A】")
            a_part = a_part.strip()
            if a_part:
                faq_candidates.append((ratio, a_part))
    faq_candidates.sort(key=lambda x: x[0], reverse=True)
    return [a_part for _, a_part in faq_candidates[:2]]


def _try_faq_fast_path(hits: List[Dict], question: str, route: str,
                       rewrite: dict, log_meta: dict,
                       _q_bigrams: set = None) -> str:
    """FAQ exact match fast path: when top-1 hit is a high-confidence FAQ entry."""
    if not hits:
        return ""

    top_hit = hits[0]
    meta = top_hit.get("meta", {})
    if meta.get("source_type") != "faq":
        return ""
    # 非 life 路由时，排除 faq_life 来源的命中
    source_file = meta.get("source_file", "")
    if route != "life" and "life" in source_file.lower():
        return ""

    thresholds = FAQ_FAST_PATH_THRESHOLDS.get(route, FAQ_FAST_PATH_DEFAULT)
    score = top_hit.get("hybrid_score") or top_hit.get("score", 0.0)
    if score < thresholds["score"]:
        return ""

    text = (top_hit.get("text") or "").strip()
    if "【Q】" not in text or "【A】" not in text:
        return ""

    q_part = text.split("【A】")[0].replace("【Q】", "")
    _, _, a_part = text.partition("【A】")
    a_part = a_part.strip()
    if not a_part:
        return ""

    if _q_bigrams is not None:
        q_bigrams = _q_bigrams
    else:
        q_norm = _normalize_for_bigram(question)
        q_bigrams = set(q_norm[i:i+2] for i in range(len(q_norm) - 1))
    faq_norm = _normalize_for_bigram(q_part)
    faq_bigrams = set(faq_norm[i:i+2] for i in range(len(faq_norm) - 1))
    if len(q_bigrams) < 3 or len(faq_bigrams) < 3:
        return ""
    overlap = len(q_bigrams & faq_bigrams)
    ratio = overlap / max(len(q_bigrams), 1)
    if overlap < 3 or ratio < thresholds["ratio"]:
        return ""

    body_lines = [a_part]
    evidence = build_evidence(hits[:1])
    add_risk = route in ("practice", "ritual")
    answer = format_structured_answer(route, body_lines, evidence, add_risk_note=add_risk)

    log_qa(question, answer, rewritten_query=rewrite.get("expanded", ""),
           matched_sources=evidence, hit=True,
           meta={**log_meta, "method": "faq_fast_path", "faq_score": score,
                 "faq_overlap_ratio": round(ratio, 3)})
    return answer


# ===================================================================
# Bullet extraction from sections
# ===================================================================

def extract_from_hits(hits: List[Dict], route: str, mode: str) -> List[str]:
    if not hits:
        return []
    paragraphs = []
    seen = set()
    for h in hits:
        text = h.get("text", "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text)[:200]
        if key in seen:
            continue
        seen.add(key)
        cleaned_lines = []
        for ln in normalize_lines(text):
            clean = ln.lstrip("-").strip()
            if not clean or is_faq_line(clean):
                continue
            cleaned_lines.append(clean)
        if cleaned_lines:
            paragraphs.append("\n".join(cleaned_lines))
    limit = _get_mode_limit(route, mode)
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
        eff_len = len(re.sub(r"[\s\u3000，。、！？；：""''（）【】《》\x2d—·]", "", clean))
        if eff_len < 8:
            continue
        if re.match(r"^\d+[）\)]", clean):
            items.append(clean)
            continue
        if len(clean) <= 120:
            items.append(clean)

    items = uniq(items)
    limit = _get_mode_limit(route, mode)
    return items[:limit]


# ===================================================================
# _fallback_from_hits (reference architecture feature)
# ===================================================================

def _fallback_from_hits(hits: List[Dict], max_lines: int = 8,
                        query: str = "") -> List[str]:
    """When rule extraction fails, extract text from search hits as fallback.
    Prioritize lines containing query keywords, sorted by match count."""
    query_terms = [t for t in _RE_FALLBACK_SPLIT.split(query.lower())
                   if t and (len(t) >= 2 or _RE_CJK_SINGLE.fullmatch(t))]
    scored_lines = []
    seen_lines = set()
    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue
        for ln in text.split("\n"):
            ln = ln.strip()
            if not ln or len(ln) <= 6:
                continue
            if _RE_CN_SECTION_TITLE.match(ln) or not (set(ln) - _SEPARATOR_CHARS):
                continue
            ln_key = " ".join(ln.split())
            if ln_key in seen_lines:
                continue
            seen_lines.add(ln_key)
            ln_lower = ln.lower()
            match_count = sum(1 for t in query_terms if t in ln_lower) if query_terms else 0
            scored_lines.append((match_count, ln))
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    return [ln for _, ln in scored_lines[:max_lines]]


# ===================================================================
# Knowledge gap logging
# ===================================================================

_GAP_LOG = Path(__file__).resolve().parent / "logs" / "knowledge_gap.jsonl"


def _log_knowledge_gap(question: str, route: str, rewrite: dict,
                       hits: list, log_meta: dict) -> None:
    """Log queries not covered by knowledge base for gap analysis."""
    try:
        from rag_logger import _append_jsonl, _ensure_dir
        top_score = max((h.get("hybrid_score") or h.get("score", 0.0) for h in hits), default=0.0)
        top_text = (hits[0].get("text", "")[:100] if hits else "")
        payload = {
            "question": question,
            "expanded_query": rewrite.get("expanded", ""),
            "route": route,
            "hit_count": len(hits),
            "top_score": round(top_score, 3),
            "top_snippet": top_text,
            "product": log_meta.get("product", ""),
        }
        _ensure_dir()
        _append_jsonl(_GAP_LOG, payload)
    except Exception:
        pass


# ===================================================================
# Chitchat detection
# ===================================================================

_CHITCHAT_REPLIES = {
    "greeting": "阿弥陀佛！我是佛教知识问答助手，请问有什么佛法上的问题可以帮您？",
    "thanks":   "不客气！随喜您对佛法的探究，如有其他问题随时请教。",
    "bye":      "再见！愿您吉祥如意，六时吉祥。",
    "ack":      "好的，如有其他佛法问题请继续提问。",
}


def _chitchat_reply(raw: str) -> str:
    s = raw.strip().rstrip("！!。.~啊呀哇？?")
    if _RE_CHAT_GREETING.search(s):
        return _CHITCHAT_REPLIES["greeting"]
    if _RE_CHAT_THANKS.search(s):
        return _CHITCHAT_REPLIES["thanks"]
    if _RE_CHAT_BYE.search(s):
        return _CHITCHAT_REPLIES["bye"]
    return _CHITCHAT_REPLIES["ack"]


# ===================================================================
# LLM client (multi-provider via llm_client)
# ===================================================================

_openai_client = None
_openai_client_checked = False
_openai_client_lock = threading.Lock()


def _get_chat_model() -> str:
    """Get chat LLM model name (prefer llm_client, fallback to global OPENAI_MODEL)."""
    try:
        from llm_client import get_model as _get_multi_model, is_enabled as _is_enabled
        if _is_enabled("chat"):
            m = _get_multi_model("chat")
            if m:
                return m
    except ImportError:
        pass
    return OPENAI_MODEL


def _get_openai_client():
    """Get chat LLM client (prefer llm_client multi-provider, fallback to legacy singleton)."""
    global _openai_client, _openai_client_checked
    try:
        from llm_client import get_client as _get_multi_client, is_enabled as _is_enabled
        if _is_enabled("chat"):
            client = _get_multi_client("chat")
            if client is not None:
                return client
    except ImportError:
        pass
    # Legacy fallback with proper double-checked locking
    if _openai_client_checked:
        return _openai_client
    with _openai_client_lock:
        if _openai_client_checked:
            return _openai_client
        if not USE_OPENAI:
            _openai_client_checked = True
            return None
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            _openai_client_checked = True
            return None
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": key}
            if OPENAI_API_BASE:
                client_kwargs["base_url"] = OPENAI_API_BASE
            _openai_client = OpenAI(**client_kwargs)
        except Exception:
            _openai_client = None
        _openai_client_checked = True
        return _openai_client


# ===================================================================
# LLM generate answer (with conversation history support)
# ===================================================================

def llm_generate_answer(question: str, context: str, route: str, mode: str,
                        history_summary: str = "",
                        history_pairs: list = None,
                        low_confidence: bool = False,
                        user_level: str = "",
                        question_type: str = "") -> str:
    """RAG: use retrieved context with LLM to generate answer.
    Supports conversation history for multi-turn dialogue.
    user_level: 'beginner' or 'experienced' — adjusts tone and depth."""
    client = _get_openai_client()
    if client is None:
        return ""
    if not context.strip():
        return ""

    # 确定用户级别
    if not user_level:
        from rag_runtime_config import DEFAULT_USER_LEVEL
        user_level = DEFAULT_USER_LEVEL

    length_hint = "控制在300-500字，重点突出、层次清晰" if mode == "brief" else "详细全面，可适当展开，800字以内"

    # 根据问题类型给出具体的回答策略
    _QTYPE_INSTRUCTIONS = {
        "method": (
            "\n## 回答策略：方法类问题\n"
            "用户想知道具体方法/步骤，请：\n"
            "- 先用一句话概括核心方法\n"
            "- 然后分点列出具体方法（3-5个要点）\n"
            "- 每个方法用自己的话简要解释，必要时引用关键颂词\n"
            "- 不要大段复制讲记原文，而是提炼要义\n"
        ),
        "definition": (
            "\n## 回答策略：定义类问题\n"
            "用户想了解概念的含义，请：\n"
            "- 先给出简明定义（1-2句话）\n"
            "- 再展开解释含义和意义\n"
            "- 可引用经典原文作为依据\n"
        ),
        "reason": (
            "\n## 回答策略：原因类问题\n"
            "用户想了解背后的原因/道理，请：\n"
            "- 直接回答「为什么」\n"
            "- 用逻辑清晰的论证说明原因\n"
            "- 引用经论依据支撑论点\n"
        ),
        "comparison": (
            "\n## 回答策略：比较类问题\n"
            "用户想对比两个概念，请：\n"
            "- 分别解释两者的定义\n"
            "- 列出主要相同点和不同点\n"
            "- 总结两者的关系\n"
        ),
        "list": (
            "\n## 回答策略：列举类问题\n"
            "用户想知道有哪些，请：\n"
            "- 用编号列表列出所有项目\n"
            "- 每项简要说明（1-2句话）\n"
        ),
        "overview": (
            "\n## 回答策略：概述类问题\n"
            "用户想了解整体内容，请：\n"
            "- 概括主要内容和结构\n"
            "- 突出核心要点\n"
        ),
        "verse": (
            "\n## 回答策略：颂词解释类问题\n"
            "用户想理解某句颂词/经文，请：\n"
            "- 先引用原文\n"
            "- 逐句解释含义\n"
            "- 说明在修行中的意义\n"
        ),
    }
    qtype_instruction = _QTYPE_INSTRUCTIONS.get(question_type, "")

    route_hints = {
        "basic": "介绍佛教基本知识。",
        "doctrine": "详细解释教义体系、逻辑关系和修证次第。",
        "practice": "说明具体修行方法、次第和注意事项，提醒依止善知识。",
        "scripture": "引用经典原文并解释其含义、背景和意义。",
        "sect": "介绍宗派的历史渊源、核心教义和修行特色。",
        "concept": "详细解释佛教概念的含义、出处和在修行中的意义。",
        "history": "说明佛教历史事件、人物和发展脉络。",
        "ritual": "说明仪轨的具体步骤、意义和注意事项。",
        "life": "结合日常生活场景给出具体可操作的建议，引用相关颂词或教言并用通俗语言解读。",
    }

    history_block = ""
    if history_pairs:
        pairs_text = "\n".join(
            f"   用户：{p.get('user', '')}\n   助手：{p.get('assistant', '')}"
            for p in history_pairs
        )
        history_block = (
            "7. 以下是之前的对话记录，请结合上下文理解用户当前问题的真实意图，\n"
            "   不要重复回答用户已经问过的内容，聚焦当前问题：\n"
            f"{pairs_text}\n"
        )
    elif history_summary:
        history_block = (
            "7. 用户之前的对话脉络如下，请结合对话上下文理解用户当前问题的真实意图，\n"
            "   不要重复回答用户已经问过的内容，聚焦当前问题：\n"
            f"   对话脉络：「{history_summary}」\n"
        )

    # 用户级别提示
    if user_level == "experienced":
        level_hint = (
            "\n## 用户级别：有经验的佛教修行者\n"
            "- 可以直接使用佛学专业术语（如空性、缘起、中观、唯识等），无需逐一解释基础概念\n"
            "- 可以引用梵文/巴利文术语和原典原文\n"
            "- 讨论可以深入教理层面，涉及不同宗派观点的辨析\n"
            "- 修行建议可以更具体、更深入，包含具体的观修方法和次第\n"
            "- 语气可以更为直接精炼\n"
        )
    else:
        level_hint = (
            "\n## 用户级别：初学者\n"
            "- 使用通俗易懂的语言，避免过多专业术语\n"
            "- 遇到佛学术语时，请用括号简要解释（如：般若（智慧）、嗔恨（愤怒）等）\n"
            "- 多用生活中的比喻和例子帮助理解\n"
            "- 修行建议要简单可操作，适合日常生活中实践\n"
            "- 语气亲切温和，像一位有耐心的老师\n"
        )

    system_prompt = (
        "你是一个佛教知识问答助手。请基于【参考资料】回答用户的问题。\n\n"
        "## 回答规则\n"
        "1. **以参考资料为主**：优先使用参考资料中的内容。如果你具备的佛学常识可以补充说明，"
        "可简要添加，但必须标注「（补充说明）」以区分\n"
        "2. **部分可答则答**：如果资料只能回答问题的一部分，先回答能回答的部分，"
        "然后注明「关于XX部分，现有资料未涉及」\n"
        "3. **资料不足时的处理**：如果参考资料与用户问题不够相关：\n"
        "   - 若该问题属于佛教通识（如基本教义、常见概念对比），可基于你的佛学知识回答，末尾注明「（补充说明）」\n"
        "   - 若该问题涉及特定讲记/法师观点而资料不足，说明「该内容知识库暂未完全覆盖」\n"
        "   - 不要强行从不相关资料中拼凑答案\n"
        "4. 回答要条理清晰，使用分点或分段组织\n"
        "5. 如参考资料中有经典原文，引用时用「」括起\n\n"
        f"{level_hint}\n"
        "## 来源标注\n"
        "- 参考资料标记为 [来源1：...]、[来源2：...] 等\n"
        "- 在回答中引用具体内容后，用 [来源N] 标注，N 为对应编号\n"
        "- 如果多个来源说法不同，分别列出并标注各自来源\n\n"
        "## 格式\n"
        f"- 回答长度：{length_hint}\n"
        f"- 重点方向：{route_hints.get(route, '根据问题自然组织回答内容。')}\n"
        "- 不要大段复制参考资料原文，而是用自己的话提炼归纳，关键处引用原文\n"
        "- 科判编号（甲一、乙二等）仅在用户明确问科判时保留，否则转化为通俗表述\n"
        "- 回答末尾加上：「以上内容基于佛教经典与传统教义整理，仅供学习参考。」\n"
        f"{qtype_instruction}"
        f"{history_block}"
    )
    if low_confidence:
        system_prompt += (
            "\n## 重要提醒\n"
            "本次检索的参考资料与用户问题的相关度较低。请你仔细判断资料是否真正回答了问题。"
            "如果资料内容与问题无关，请回答：「该问题超出了当前知识库的覆盖范围，"
            "建议查阅相关佛教经典或咨询法师获取更准确的解答。」\n"
        )

    user_prompt = (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        "请基于以上参考资料回答问题。"
    )

    try:
        resp = client.chat.completions.create(
            model=_get_chat_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=ROUTE_LLM_TEMPERATURE.get(route, LLM_TEMPERATURE),
            max_tokens=LLM_MAX_TOKENS_BRIEF if mode == "brief" else LLM_MAX_TOKENS_FULL,
        )
        if not resp.choices:
            return ""
        choice = resp.choices[0]
        msg = choice.message
        answer = (msg.content or "").strip() if msg else ""
        if answer and getattr(choice, "finish_reason", None) == "length":
            answer += "\n\n（注：回答因长度限制被截断，如需完整内容请缩小问题范围。）"
        return answer if answer else ""
    except Exception as e:
        log_error("llm_generate_answer", repr(e),
                  meta={"route": route, "question": question[:100]})
        if DEBUG:
            print(f"[DEBUG] LLM generation failed: {e}")
        return ""


# Legacy compat: openai_rag_generate -> llm_generate_answer wrapper
def openai_rag_generate(question: str, context: str, route: str,
                        low_confidence: bool = False) -> str:
    return llm_generate_answer(question, context, route, "brief",
                               low_confidence=low_confidence)


# ===================================================================
# LLM fallback / static fallback
# ===================================================================

def _build_knowledge_topics() -> str:
    """Dynamically build knowledge topics from PRODUCT_ALIASES instead of hardcoding."""
    prod_names = [aliases[0] for aliases in PRODUCT_ALIASES.values() if aliases]
    prod_hint = "、".join(prod_names) if prod_names else "佛学主题"
    return (
        f"{prod_hint}等相关的佛教教义（四圣谛、八正道、十二因缘、缘起性空等）、"
        "佛法与日常生活应用（情绪管理、人际关系、工作压力、家庭矛盾等）、"
        "法师开示与演讲、"
        "修行方法（禅修、念佛、持咒、止观等）、"
        "佛教经典（心经、金刚经、法华经、入行论等）、"
        "宗派传承（禅宗、净土宗、天台宗、藏传佛教等）、"
        "佛教概念（菩提心、佛性、般若、涅槃等）、"
        "佛教历史与仪轨"
    )

_KNOWLEDGE_TOPICS = _build_knowledge_topics()


def _llm_fallback_answer(question: str, route: str, hits: list) -> str:
    """When search fails, use LLM for intelligent guidance (not fabrication)."""
    client = _get_openai_client()
    if client is None:
        return ""

    partial_context = ""
    if hits:
        snippets = [h.get("text", "")[:200] for h in hits[:3] if h.get("text")]
        if snippets:
            partial_context = (
                "\n以下是检索到的部分相关片段（相关度较低，仅供参考）：\n"
                + "\n---\n".join(snippets)
            )

    system_prompt = (
        "你是一位佛教知识问答助手。用户问了一个知识库中尚未完全覆盖的问题。\n"
        "你的任务是：\n"
        "1. 如果该问题属于佛教基础知识（如大小乘区别、基本教义、常见概念），"
        "请直接基于你的佛学知识回答，但末尾注明「（以上为佛教通识，非来自本知识库讲记内容）」\n"
        "2. 如果提供了部分相关片段，优先基于片段内容回答（注明来源）\n"
        "3. 如果该问题涉及特定法师的讲解或论典的具体章节内容，而你没有相关资料，"
        "则坦诚说明知识库暂未覆盖，并推荐1-2个知识库能回答的相关话题\n"
        "4. 语气自然亲切\n"
        f"\n当前知识库覆盖的主题包括：\n{_KNOWLEDGE_TOPICS}\n"
    )

    user_prompt = f"用户问题：{question}{partial_context}"

    try:
        resp = client.chat.completions.create(
            model=_get_chat_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        if not resp.choices:
            return ""
        text = (resp.choices[0].message.content or "").strip()
        return text
    except Exception as e:
        log_error("llm_fallback_answer", f"LLM fallback failed: {e}",
                  meta={"route": route, "question": question[:100]})
        return ""


def _static_fallback(hits: list) -> list:
    """Static fallback when no LLM is available."""
    if hits:
        return [
            "知识库中可能存在相关信息，但置信度不足以生成准确结论。",
            "建议换一种表述重新提问，或查阅相关佛教经典。",
        ]
    return [
        "该问题目前知识库尚未覆盖。",
        "您可以尝试问我以下方面的问题：佛教教义、修行方法、"
        "佛教经典、宗派传承、佛教概念、仪轨等。",
        "如需深入了解，建议查阅相关佛教经典或咨询法师。",
    ]


def _build_not_found_response(question: str, route: str, rewrite: dict) -> str:
    """Build friendly response when nothing found, with topic suggestions."""
    suggestions = _suggest_related_topics(question, route)
    fallback = ["当前知识库未找到与该问题直接相关的内容。"]
    if suggestions:
        fallback.append("您可以尝试以下相关主题：" + "、".join(suggestions))
    else:
        fallback.append("建议尝试更具体的佛教术语进行提问，如：四圣谛、八正道、菩提心、入行论等。")
    fallback.append("如需深入了解，建议查阅相关佛教经典或咨询法师。")
    return format_structured_answer(route, fallback, [], add_risk_note=False)


# ===================================================================
# Thread-local route/product accessor
# ===================================================================

def get_last_route_product():
    """Return the most recent (route, product_id) from the current thread."""
    return (getattr(_thread_local, "route", ""),
            getattr(_thread_local, "product", ""))


# ===================================================================
# answer_one: main single-question pipeline
# ===================================================================

_NO_MATCH_REPLY = "抱歉，暂时无法回答该问题。请尝试询问佛教教义、修行方法、佛教经典等相关问题。"
_OFFTOPIC_REPLY = "抱歉，该问题不在我的服务范围内。我是佛教知识问答助手，可以为您解答佛教教义、修行方法、经典解读、宗派传承等相关问题。"
_SPECIAL_INTENT_REPLIES = {
    "price": PRICE_REPLY,
    "comparison": COMPARISON_REPLY,
    "location": LOCATION_REPLY,
}


def answer_one(question: str, mode: str, rewrite: dict = None,
               route_override: str = "", user_level: str = "") -> str:
    product = detect_product(question)
    route = route_override or detect_route(question)
    _thread_local.route = route
    _thread_local.product = product
    if rewrite is None:
        rewrite = rewrite_query(question)

    _log_meta = {"product": product, "route": route, "mode": mode}

    # Route config
    route_cfg = _get_route_config(route)
    route_top_k = route_cfg.get("k", DEFAULT_TOP_K)
    route_threshold = route_cfg.get("threshold", SCORE_THRESHOLD)

    # 1. 问题分类 → 驱动后续搜索策略
    from question_classifier import classify as _classify_question
    _qclass = _classify_question(question)
    _qtype = _qclass["question_type"]
    _strategy = _qclass["strategy"]
    skip_faq = _strategy.get("skip_faq", False)

    faq_answer = ""
    if not skip_faq:
        faq_text = read_knowledge_file(product, "faq.txt")
        pdir = KNOWLEDGE_DIR / product
        if pdir.exists():
            for fp in sorted(pdir.rglob("faq*.txt")):
                if fp.name == "faq.txt":
                    continue
                # 非 life 路由时排除 faq_life 文件，避免生活应用内容抢占教义问答
                if route != "life" and "life" in fp.name.lower():
                    continue
                sub_faq = fp.read_text(encoding="utf-8", errors="replace")
                if sub_faq.strip():
                    faq_text = faq_text + "\n" + sub_faq
        alias_text = read_knowledge_file(product, "alias.txt")
        faq_answer = match_faq(question, faq_text, FAQ_KEYWORD_MAP, alias_text)
    if faq_answer:
        faq_evidence = [{"meta": {
            "source_file": "faq.txt",
            "source_type": "faq",
            "chunk_id": "faq_match",
        }}]
        answer = format_structured_answer(route, [faq_answer], faq_evidence, add_risk_note=False)
        log_qa(question, answer, rewritten_query=rewrite.get("expanded", ""),
               matched_sources=faq_evidence, hit=True,
               meta={**_log_meta, "method": "faq_exact"})
        return answer

    # 2. Parallel vector + keyword hybrid search (ThreadPoolExecutor)
    # 比较类问题：拆分为两个子查询分别检索，合并结果
    if _qtype == "comparison" and _qclass.get("comparison_concepts"):
        concept_a, concept_b = _qclass["comparison_concepts"]
        # 分别检索两个概念
        _hits_a = vector_search(product, concept_a + " 定义 含义", route_top_k)
        _hits_b = vector_search(product, concept_b + " 定义 含义", route_top_k)
        # 合并去重
        _seen = set()
        _combined = []
        for h in _hits_a + _hits_b:
            key = h.get("text", "")[:100]
            if key not in _seen:
                _seen.add(key)
                _combined.append(h)
        if _combined:
            # 比较类问题用更低的阈值，因为两个概念分开搜索分数可能较低
            hits = sorted(_combined, key=lambda x: x.get("score", 0), reverse=True)[:route_top_k]
            hits = filter_by_score(hits, max(0.10, route_threshold - 0.10))
            hits = _deduplicate_hits(hits)
            if USE_OPENAI and hits:
                context = _build_context(hits)
                if context:
                    comp_hint = f"\n[提示：用户在对比「{concept_a}」和「{concept_b}」，请分别解释两者，然后总结异同。]\n"
                    llm_answer = llm_generate_answer(
                        question, comp_hint + context, route, mode,
                        history_summary=rewrite.get("history_summary", ""),
                        history_pairs=rewrite.get("history_pairs", []),
                        user_level=user_level,
                        question_type="comparison",
                    )
                    if llm_answer and len(llm_answer.strip()) >= 15:
                        evidence = build_evidence(hits[:3])
                        answer = format_structured_answer(route, [llm_answer], evidence)
                        log_qa(question, answer, rewritten_query=rewrite.get("expanded", ""),
                               matched_sources=evidence, hit=True,
                               meta={**_log_meta, "method": "comparison_split"})
                        return answer

    search_q = rewrite.get("search_query", rewrite.get("original", question))

    def _do_vector(store_name):
        return vector_search(store_name, search_q, max(route_top_k, VECTOR_TOP_K))

    def _do_keyword(store_name):
        _, d = load_store(store_name)
        return keyword_search(rewrite["expanded"], d, max(route_top_k, KEYWORD_TOP_K)) if d else []

    futures = {}
    futures["v_prod"] = _search_pool.submit(_do_vector, product)
    futures["k_prod"] = _search_pool.submit(_do_keyword, product)

    # 同时搜索共享索引（scripture, master, glossary）
    _shared_store = STORE_ROOT / "_shared"
    if _shared_store.exists() and (_shared_store / "index.faiss").exists():
        futures["v_shared"] = _search_pool.submit(_do_vector, "_shared")
        futures["k_shared"] = _search_pool.submit(_do_keyword, "_shared")

    vector_hits, keyword_hits = [], []
    for key, fut in futures.items():
        try:
            result = fut.result(timeout=30)
        except Exception as e:
            log_error("answer_one", f"Search timeout/error: {key}: {e}",
                      meta={"product": product, "route": route})
            result = []
        if key.startswith("v_"):
            vector_hits.extend(result)
        else:
            keyword_hits.extend(result)

    # Route-aware merge weights with secondary route support
    vw = route_cfg.get("vw", HYBRID_VECTOR_WEIGHT)
    kw = route_cfg.get("kw", HYBRID_KEYWORD_WEIGHT)
    secondary_routes = detect_secondary_routes(question, route)
    hits = merge_hybrid(vector_hits, keyword_hits, vw, kw, route_top_k,
                         route=route, secondary_routes=secondary_routes) if (vector_hits or keyword_hits) else []

    # Score filtering
    hits = filter_by_score(hits, route_threshold)
    hits = _deduplicate_hits(hits)

    # 来源过滤：根据问题类型和 content_layer 调整权重
    if route != "life":
        for h in hits:
            meta = h.get("meta") or {}
            layer = meta.get("content_layer", "")
            src = meta.get("source_file", "")
            # content_layer 标记优先，文件名回退
            is_life = (layer == "life") or (not layer and "life" in src.lower())
            if is_life and _strategy.get("prefer_lecture"):
                h["hybrid_score"] = h.get("hybrid_score", 0) * 0.3
            elif layer == "doctrine" and _strategy.get("prefer_lecture"):
                h["hybrid_score"] = h.get("hybrid_score", 0) * 1.15
        hits = sorted(hits, key=lambda x: x.get("hybrid_score", 0), reverse=True)

    # 3. CrossEncoder rerank (Buddhist feature)
    if USE_RERANK and hits:
        hits = rerank_hits(question, hits, top_k=RERANK_TOP_K)

    # Precompute question bigrams for FAQ fast path
    _q_norm = _normalize_for_bigram(question)
    _q_bigrams = set(_q_norm[i:i+2] for i in range(len(_q_norm) - 1)) if len(_q_norm) >= 2 else set()

    # Strategy 0: FAQ fast path from search hits
    if hits:
        faq_fast = _try_faq_fast_path(hits, question, route, rewrite, _log_meta,
                                       _q_bigrams=_q_bigrams)
        if faq_fast:
            return faq_fast

    # Confidence detection
    low_confidence = False
    if hits:
        best_score = max((h.get("hybrid_score") or h.get("score", 0.0) for h in hits), default=0.0)
        if best_score < 0.35:
            low_confidence = True

    if not hits:
        # LLM smart fallback
        if USE_OPENAI:
            llm_fb = _llm_fallback_answer(question, route, hits)
            if llm_fb:
                _log_knowledge_gap(question, route, rewrite, hits, _log_meta)
                log_qa(question, llm_fb, rewritten_query=rewrite.get("expanded", ""),
                       matched_sources=[], hit=False,
                       meta={**_log_meta, "method": "llm_fallback"})
                return llm_fb
        return _build_not_found_response(question, route, rewrite)

    # Strategy 1: LLM RAG (primary) with context from hits
    if USE_OPENAI:
        context = _build_context(hits)
        if context:
            history_summary = rewrite.get("history_summary", "")
            history_pairs = rewrite.get("history_pairs", [])
            llm_answer = llm_generate_answer(
                question, context, route, mode,
                history_summary=history_summary,
                history_pairs=history_pairs,
                low_confidence=low_confidence,
                user_level=user_level,
                question_type=_qtype,
            )
            # Validate: not too short, not echo of question
            if llm_answer and len(llm_answer.strip()) >= 15:
                q_stripped = question.strip()
                a_stripped = llm_answer.strip()
                echo_ok = True
                if len(q_stripped) >= 10 and len(a_stripped) < len(q_stripped) * 3:
                    echo_ratio = _text_overlap_ratio(q_stripped, a_stripped[:len(q_stripped) * 3])
                    if echo_ratio > 0.85:
                        echo_ok = False
                if echo_ok:
                    # Enrich with relation_engine and media_router (Buddhist features)
                    relation_lines = relation_enrich(route, product, question)
                    if relation_lines:
                        llm_answer += "\n\n【关联信息】\n" + "\n".join(relation_lines[:6])
                    media_items = media_find(question, product_id=product, route=route)
                    if media_items:
                        media_refs = ["【相关资料】"]
                        for mi in media_items[:3]:
                            title = mi.get("title", "")
                            url = mi.get("url", "")
                            if title:
                                media_refs.append(f"- {title}" + (f"：{url}" if url else ""))
                        if len(media_refs) > 1:
                            llm_answer += "\n\n" + "\n".join(media_refs)

                    evidence = build_evidence(hits)
                    log_qa(question, llm_answer, rewritten_query=rewrite.get("expanded", ""),
                           matched_sources=evidence, hit=True,
                           meta={**_log_meta, "method": "llm_rag"})
                    return format_structured_answer(route, [llm_answer.strip()], evidence,
                                                    add_risk_note=(route == "practice"))

    # Strategy 2: Rule extraction fallback
    body_lines = extract_from_hits(hits, route, mode)
    body_lines = [
        p for p in body_lines
        if len(re.sub(r"[\s\u3000，。、！？；：""''（）【】《》]", "", p)) >= 15
    ]

    # FAQ supplement from hits
    if hits and len(body_lines) < 6:
        faq_supplement = _extract_faq_from_hits(hits, question, _q_bigrams=_q_bigrams)
        if faq_supplement:
            if body_lines:
                body_lines = faq_supplement + [""] + body_lines
            else:
                body_lines = faq_supplement

    # Relation engine enrichment (Buddhist feature)
    relation_lines = relation_enrich(route, product, question)
    if relation_lines:
        body_lines = body_lines + ["", "【关联信息】"] + relation_lines[:6]

    if not body_lines:
        main_text = read_knowledge_file(product, "main.txt")
        body_lines = parse_bullets_from_section(main_text, faq_text, route, mode)

    if not body_lines:
        # Fallback from hits
        fallback_lines = _fallback_from_hits(hits, query=question)
        if fallback_lines:
            body_lines = fallback_lines
        else:
            # LLM smart fallback
            if USE_OPENAI:
                llm_fb = _llm_fallback_answer(question, route, hits)
                if llm_fb:
                    _log_knowledge_gap(question, route, rewrite, hits, _log_meta)
                    log_qa(question, llm_fb, rewritten_query=rewrite.get("expanded", ""),
                           matched_sources=build_evidence(hits), hit=False,
                           meta={**_log_meta, "method": "llm_fallback"})
                    return llm_fb
            # Static fallback
            _log_knowledge_gap(question, route, rewrite, hits, _log_meta)
            evidence = build_evidence(hits)
            text = format_structured_answer(route, _static_fallback(hits), evidence,
                                            add_risk_note=(route == "practice"))
            log_qa(question, text, rewritten_query=rewrite.get("expanded", ""),
                   matched_sources=evidence, hit=False,
                   meta={**_log_meta, "method": "low_confidence" if hits else "no_hit"})
            return text

    # Media enrichment (Buddhist feature)
    media_items = media_find(question, product_id=product, route=route)
    if media_items:
        media_refs = ["【相关资料】"]
        for mi in media_items[:3]:
            title = mi.get("title", "")
            url = mi.get("url", "")
            if title:
                media_refs.append(f"- {title}" + (f"：{url}" if url else ""))
        if len(media_refs) > 1:
            body_lines = body_lines + [""] + media_refs

    evidence = build_evidence(hits)
    text = format_structured_answer(route, body_lines, evidence,
                                    add_risk_note=(route == "practice"))
    log_qa(question, text, rewritten_query=rewrite.get("expanded", ""),
           matched_sources=evidence, hit=True,
           meta={**_log_meta, "method": "rule_extract"})
    return text


# ===================================================================
# answer_question: main entry point
# ===================================================================

def _detect_route_with_history(question: str, rewrite: dict) -> str:
    """Route detection with history inheritance for follow-up questions."""
    route = detect_route(question)
    if route != "basic":
        return route

    raw_input = rewrite.get("raw_input", "")
    should_inherit = (
        rewrite.get("context_resolved")
        or (raw_input and len(raw_input.strip()) <= 10)
    )

    if should_inherit:
        routed_q = rewrite.get("last_routed_q") or rewrite.get("last_user_q", "")
        if routed_q:
            history_route = detect_route(routed_q)
            if history_route != "basic":
                return history_route

    return route


def answer_question(question: str, mode: str, history: list = None,
                    rewrite: dict = None, user_level: str = "") -> str:
    """Main entry point: answer a question, return string.
    user_level: 'beginner' or 'experienced' — adjusts answer tone and depth."""
    q = (question or "").strip()
    if not q:
        return "请输入您想了解的佛教问题。"
    q = q[:500]

    if rewrite is None:
        rewrite = rewrite_query(q, history=history) if history else rewrite_query(q)

    # Chitchat fast path
    if rewrite.get("is_chitchat"):
        reply = _chitchat_reply(rewrite.get("raw_input", q))
        log_qa(q, reply, rewritten_query="", matched_sources=[], hit=False,
               meta={"method": "chitchat"})
        return reply

    # Off-topic fast path
    if rewrite.get("is_offtopic"):
        log_qa(q, _OFFTOPIC_REPLY, rewritten_query="", matched_sources=[], hit=False,
               meta={"method": "offtopic"})
        return _OFFTOPIC_REPLY

    # Special intent fast path (Buddhist equivalents)
    special = _detect_special_intent(q)
    if special:
        reply = _SPECIAL_INTENT_REPLIES.get(special, "")
        if reply:
            log_qa(q, reply, rewritten_query="", matched_sources=[], hit=False,
                   meta={"method": "special_intent", "intent": special})
            return reply

    sub_questions = rewrite.get("sub_questions", [q])[:MAX_SUB_QUESTIONS]

    # Precompute rewrites and routes for sub-questions
    sub_tasks = []
    seen_questions = set()
    for subq in sub_questions:
        # 按内容去重而非路由去重，避免丢弃同路由的不同子问题
        subq_norm = subq.strip().lower()
        if subq_norm in seen_questions:
            continue
        seen_questions.add(subq_norm)
        sub_rewrite = rewrite if subq == rewrite.get("original", q) else rewrite_query(subq)
        route = _detect_route_with_history(subq, sub_rewrite)
        sub_tasks.append((subq, sub_rewrite, route))

    # Parallel execution for multiple sub-questions
    outputs = []
    if len(sub_tasks) > 1:
        futures = []
        for subq, sub_rewrite, route in sub_tasks:
            fut = _search_pool.submit(answer_one, subq, mode, sub_rewrite, route, user_level)
            futures.append(fut)
        for fut in futures:
            try:
                ans = fut.result(timeout=60)
                if ans and ans.strip():
                    outputs.append(ans)
            except Exception as e:
                log_error("answer_question", f"Sub-question parallel error: {e}",
                          meta={"question": q[:100]})
    else:
        for subq, sub_rewrite, route in sub_tasks:
            try:
                ans = answer_one(subq, mode, rewrite=sub_rewrite, route_override=route, user_level=user_level)
            except Exception as e:
                log_error("answer_one", repr(e), meta={"question": subq})
                ans = ""
            if ans and ans.strip():
                outputs.append(ans)

    return "\n\n".join(outputs) if outputs else _NO_MATCH_REPLY


# ===================================================================
# CLI entry point
# ===================================================================

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
    print(f"Saved to: {_get_out_path()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_msg = "ERROR: " + repr(e)
        save_answer(err_msg)
        print(err_msg)
