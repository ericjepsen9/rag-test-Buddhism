from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
QA_LOG = LOG_DIR / "qa_log.jsonl"
MISS_LOG = LOG_DIR / "miss_log.jsonl"
ERROR_LOG = LOG_DIR / "error_log.jsonl"

# 日志文件大小上限（默认 10MB），超过后轮转
def _safe_int(env_key: str, default: int) -> int:
    try:
        return int(os.environ.get(env_key, str(default)))
    except (ValueError, TypeError):
        return default

LOG_MAX_BYTES = _safe_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
LOG_BACKUP_COUNT = _safe_int("LOG_BACKUP_COUNT", 3)

_write_lock = threading.Lock()


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_if_needed(path: Path) -> None:
    """简易日志轮转：超过 LOG_MAX_BYTES 时重命名为 .1, .2, ..."""
    try:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return
        if size < LOG_MAX_BYTES:
            return
        # 删除最老的备份
        for i in range(LOG_BACKUP_COUNT, 0, -1):
            old = Path(f"{path}.{i}")
            if i == LOG_BACKUP_COUNT:
                old.unlink(missing_ok=True)
            elif old.exists():
                old.rename(Path(f"{path}.{i + 1}"))
        path.rename(Path(f"{path}.1"))
    except OSError:
        pass  # 轮转失败不应阻塞写入


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir()
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _write_lock:
        _rotate_if_needed(path)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass  # 磁盘满等极端情况下不崩溃


def log_qa(
    question: str,
    answer: str,
    *,
    rewritten_query: Optional[str] = None,
    matched_sources: Optional[list[dict[str, Any]]] = None,
    hit: bool = True,
    latency_ms: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "question": question,
        "rewritten_query": rewritten_query or "",
        "answer": answer,
        "hit": bool(hit),
        "latency_ms": latency_ms,
        "matched_sources": matched_sources or [],
        "meta": meta or {},
    }
    _append_jsonl(QA_LOG, payload)
    if not hit:
        _append_jsonl(
            MISS_LOG,
            {
                "question": question,
                "rewritten_query": rewritten_query or "",
                "meta": meta or {},
            },
        )


def log_error(stage: str, error: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    try:
        _append_jsonl(
            ERROR_LOG,
            {
                "stage": stage,
                "error": error,
                "meta": meta or {},
            },
        )
    except Exception:
        # 日志写入本身失败时，输出到 stderr 避免错误完全丢失
        import sys
        try:
            print(f"[LOG_ERROR_FAIL] stage={stage} error={error}", file=sys.stderr)
        except Exception:
            pass


def read_recent(path: Path, limit: int = 20) -> list[Dict[str, Any]]:
    """读取日志文件的最近 N 条记录（从文件末尾反向读取，避免全量扫描）"""
    if not path.exists():
        return []
    cap = max(1, limit)
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        # 小文件直接全量读取
        if size <= 64 * 1024:
            buf: deque[Dict[str, Any]] = deque(maxlen=cap)
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        buf.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return list(reversed(buf))
        # 大文件：从末尾反向读取足够的块
        read_size = min(size, cap * 2048)  # 估算每行 ~2KB
        with path.open("rb") as f:
            f.seek(max(0, size - read_size))
            tail = f.read().decode("utf-8", errors="replace")
        lines = tail.strip().split("\n")
        results = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(results) >= cap:
                break
        return results
    except OSError:
        return []


def get_recent_qa(limit: int = 20) -> list[Dict[str, Any]]:
    return read_recent(QA_LOG, limit=limit)


def get_recent_misses(limit: int = 20) -> list[Dict[str, Any]]:
    return read_recent(MISS_LOG, limit=limit)


def get_recent_errors(limit: int = 20) -> list[Dict[str, Any]]:
    return read_recent(ERROR_LOG, limit=limit)
