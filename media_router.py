import json
from pathlib import Path
from typing import List, Dict, Optional
from rag_runtime_config import KNOWLEDGE_DIR

# media.json 缓存：product_id -> (mtime, items)，避免每次请求读取磁盘
_media_cache: Dict[str, tuple] = {}


def _load_product_media(product_id: str) -> List[Dict]:
    """加载某个产品的 media.json（带 mtime 缓存）"""
    media_file = KNOWLEDGE_DIR / product_id / "media.json"
    if not media_file.exists():
        return []
    try:
        mtime = media_file.stat().st_mtime
        cached = _media_cache.get(product_id)
        if cached and cached[0] == mtime:
            return cached[1]
        items = json.loads(media_file.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            from rag_logger import log_error
            log_error("media_router", f"media.json 应为数组，实际为 {type(items).__name__}",
                      meta={"product_id": product_id})
            return []
        result = [it for it in items if isinstance(it, dict) and "title" in it]
        # 预小写关键词，避免 find_media 每次调用时重复 .lower()
        for it in result:
            kws = it.get("keywords")
            if isinstance(kws, list):
                it["_kw_lower"] = [k.lower() for k in kws]
        _media_cache[product_id] = (mtime, result)
        return result
    except Exception as e:
        try:
            from rag_logger import log_error
            log_error("media_router", f"media.json 加载失败: {e}",
                      meta={"product_id": product_id, "path": str(media_file)})
        except Exception:
            pass
        return []


def invalidate_media_cache(product_id: str = "") -> None:
    """清除媒体缓存（产品知识更新后调用）"""
    if product_id:
        _media_cache.pop(product_id, None)
    else:
        _media_cache.clear()


def find_media(question: str,
               product_id: Optional[str] = None,
               route: Optional[str] = None) -> List[Dict]:
    """根据产品+路由精准匹配媒体，关键词兜底。"""
    if not product_id:
        # 兼容旧调用方式：无 product_id 时尝试 buddhism
        product_id = "buddhism"

    items = _load_product_media(product_id)
    if not items:
        return []

    # 1) 路由精准匹配
    if route:
        route_hits = [it for it in items
                      if isinstance(it.get("routes"), list) and route in it["routes"]]
        if route_hits:
            return route_hits[:6]

    # 2) 关键词兜底
    q = (question or "").lower()
    if not q:
        return []
    kw_hits = []
    for it in items:
        keys = it.get("_kw_lower") or (
            [k.lower() for k in it["keywords"]] if isinstance(it.get("keywords"), list) else []
        )
        if any(k in q for k in keys):
            kw_hits.append(it)
    return kw_hits[:6]
