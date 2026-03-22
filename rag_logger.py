from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ===== 请求级 Trace ID =====
# 使用 contextvars 实现跨异步/线程的请求追踪
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def set_trace_id(trace_id: str = "") -> str:
    """设置当前请求的 trace_id，返回实际使用的 ID"""
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    """获取当前请求的 trace_id"""
    return _trace_id_var.get("")

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
QA_LOG = LOG_DIR / "qa_log.jsonl"
MISS_LOG = LOG_DIR / "miss_log.jsonl"
ERROR_LOG = LOG_DIR / "error_log.jsonl"
EVENT_LOG = LOG_DIR / "event_log.jsonl"

# 日志文件大小上限（默认 10MB），超过后轮转
def _safe_int(env_key: str, default: int) -> int:
    try:
        return int(os.environ.get(env_key, str(default)))
    except (ValueError, TypeError):
        return default

LOG_MAX_BYTES = _safe_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
LOG_BACKUP_COUNT = _safe_int("LOG_BACKUP_COUNT", 3)

_write_lock = threading.Lock()

# ===== PII 脱敏 =====
_RE_PHONE = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
_RE_IDCARD = re.compile(r'(?<!\d)(\d{6})\d{6,8}(\d{4}[0-9Xx])(?!\d)')
_RE_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_RE_NAME = re.compile(r'(?:我(?:叫|是|姓)|(?:姓名|名字)[是为：:]\s*)([\u4e00-\u9fff]{2,4})')

_SANITIZE_ENABLED = os.environ.get("RAG_LOG_SANITIZE", "1").strip() not in ("0", "false", "")


def _sanitize_pii(text: str) -> str:
    """对文本中的手机号、身份证、邮箱、自报姓名进行脱敏"""
    if not text or not _SANITIZE_ENABLED:
        return text
    text = _RE_PHONE.sub(lambda m: m.group(1)[:3] + '****' + m.group(1)[-4:], text)
    text = _RE_IDCARD.sub(lambda m: m.group(1) + '******' + m.group(2), text)
    text = _RE_EMAIL.sub('***@***.***', text)
    text = _RE_NAME.sub(lambda m: m.group(0).replace(m.group(1), '*' * len(m.group(1))), text)
    return text


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
    trace_id = get_trace_id()
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        **({"trace_id": trace_id} if trace_id else {}),
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
        "question": _sanitize_pii(question),
        "rewritten_query": _sanitize_pii(rewritten_query or ""),
        "answer": _sanitize_pii(answer),
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
                "question": _sanitize_pii(question),
                "rewritten_query": _sanitize_pii(rewritten_query or ""),
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


def log_event(stage: str, message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    """记录 info 级别事件（启动、关闭、配置变更等）"""
    try:
        _append_jsonl(EVENT_LOG, {"stage": stage, "message": message, "meta": meta or {}})
    except Exception:
        print(f"[EVENT] {stage}: {message}")


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
