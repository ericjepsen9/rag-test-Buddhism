"""持久化同义词存储：自动沉淀 LLM 改写成功的词汇映射。

文件格式：JSON，存储在 data/learned_synonyms.json
结构：
{
  "业障": {"mapped_to": "业力", "count": 5, "first_seen": "...", "last_seen": "...", "source": "llm_rewrite"},
  ...
}

线程安全：读写均加锁。
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LEARNED_SYNONYMS_FILE = DATA_DIR / "learned_synonyms.json"

_lock = threading.Lock()


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> Dict[str, Any]:
    """加载已学习的同义词映射"""
    if not LEARNED_SYNONYMS_FILE.exists():
        return {}
    try:
        with LEARNED_SYNONYMS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        from rag_logger import log_error
        log_error("synonym_store", f"同义词文件加载失败: {e}",
                  meta={"path": str(LEARNED_SYNONYMS_FILE)})
        return {}


def _save(data: Dict[str, Any]) -> None:
    """原子写入同义词文件"""
    _ensure_dir()
    tmp = LEARNED_SYNONYMS_FILE.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(LEARNED_SYNONYMS_FILE)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        from rag_logger import log_error
        log_error("synonym_store", f"同义词文件写入失败: {e}",
                  meta={"path": str(LEARNED_SYNONYMS_FILE)})
        raise


def save_learned(original_term: str, mapped_to: str) -> None:
    """保存一条 LLM 改写成功的映射。

    如果该映射已存在，更新计数和最后使用时间。
    """
    original_term = original_term.strip()
    mapped_to = mapped_to.strip()
    if not original_term or not mapped_to or original_term == mapped_to:
        return

    now = datetime.now().isoformat(timespec="seconds")
    with _lock:
        data = _load()
        if original_term in data:
            entry = data[original_term]
            entry["count"] = entry.get("count", 1) + 1
            entry["last_seen"] = now
            # 仅在未审核时更新映射目标，已审核的人工映射不可被 LLM 覆盖
            if entry.get("mapped_to") != mapped_to and not entry.get("approved"):
                entry["mapped_to"] = mapped_to
        else:
            data[original_term] = {
                "mapped_to": mapped_to,
                "count": 1,
                "first_seen": now,
                "last_seen": now,
                "source": "llm_rewrite",
                "approved": False,  # 默认未审核
            }
        _save(data)


def get_all_learned() -> List[Dict[str, Any]]:
    """返回所有已学习的同义词，按 count 降序排列"""
    with _lock:
        data = _load()
    result = []
    for term, entry in data.items():
        result.append({
            "original": term,
            "mapped_to": entry.get("mapped_to", ""),
            "count": entry.get("count", 1),
            "first_seen": entry.get("first_seen", ""),
            "last_seen": entry.get("last_seen", ""),
            "source": entry.get("source", "llm_rewrite"),
            "approved": entry.get("approved", False),
        })
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


def approve_learned(original_term: str) -> bool:
    """标记一条学习到的同义词为已审核"""
    with _lock:
        data = _load()
        if original_term not in data:
            return False
        data[original_term]["approved"] = True
        _save(data)
    return True


def delete_learned(original_term: str) -> bool:
    """删除一条学习到的同义词"""
    with _lock:
        data = _load()
        if original_term not in data:
            return False
        del data[original_term]
        _save(data)
    return True


def add_manual(original_term: str, mapped_to: str) -> Dict[str, Any]:
    """手动添加一条同义词映射（来源标记为 manual）。

    如果原始词已存在，返回错误提示。
    """
    original_term = original_term.strip()
    mapped_to = mapped_to.strip()
    if not original_term or not mapped_to:
        return {"ok": False, "error": "原始词和映射词不能为空"}
    if original_term == mapped_to:
        return {"ok": False, "error": "原始词和映射词不能相同"}

    now = datetime.now().isoformat(timespec="seconds")
    with _lock:
        data = _load()
        if original_term in data:
            return {"ok": False, "error": f"「{original_term}」已存在，请使用编辑功能修改"}
        data[original_term] = {
            "mapped_to": mapped_to,
            "count": 0,
            "first_seen": now,
            "last_seen": now,
            "source": "manual",
            "approved": True,  # 手动添加默认已审核
        }
        _save(data)
    return {"ok": True}


def update_learned(original_term: str, mapped_to: str) -> Dict[str, Any]:
    """编辑已有同义词的映射目标。"""
    original_term = original_term.strip()
    mapped_to = mapped_to.strip()
    if not original_term or not mapped_to:
        return {"ok": False, "error": "原始词和映射词不能为空"}

    with _lock:
        data = _load()
        if original_term not in data:
            return {"ok": False, "error": f"「{original_term}」不存在"}
        data[original_term]["mapped_to"] = mapped_to
        data[original_term]["last_seen"] = datetime.now().isoformat(timespec="seconds")
        _save(data)
    return {"ok": True}


def batch_approve(terms: List[str]) -> Dict[str, Any]:
    """批量审核通过多条同义词。"""
    if not terms:
        return {"ok": False, "error": "terms 列表为空"}
    with _lock:
        data = _load()
        approved = []
        for t in terms:
            t = t.strip()
            if t in data and not data[t].get("approved"):
                data[t]["approved"] = True
                approved.append(t)
        if approved:
            _save(data)
    return {"ok": True, "approved_count": len(approved), "approved": approved}


def batch_delete(terms: List[str]) -> Dict[str, Any]:
    """批量删除多条同义词。"""
    if not terms:
        return {"ok": False, "error": "terms 列表为空"}
    with _lock:
        data = _load()
        deleted = []
        for t in terms:
            t = t.strip()
            if t in data:
                del data[t]
                deleted.append(t)
        if deleted:
            _save(data)
    return {"ok": True, "deleted_count": len(deleted), "deleted": deleted}


def get_static_synonyms() -> List[Dict[str, str]]:
    """返回静态同义词表（来自 search_utils._SYNONYM_MAP）"""
    try:
        from search_utils import _SYNONYM_MAP
        return [{"original": k, "mapped_to": v} for k, v in _SYNONYM_MAP.items()]
    except (ImportError, AttributeError):
        return []


def get_all_synonyms_combined() -> Dict[str, Any]:
    """返回合并后的完整词库：静态 + 已学习"""
    static = get_static_synonyms()
    learned = get_all_learned()
    return {
        "static": static,
        "learned": learned,
        "static_count": len(static),
        "learned_count": len(learned),
        "total_count": len(static) + len(learned),
    }
