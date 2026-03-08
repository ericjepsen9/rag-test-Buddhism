from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
QA_LOG = LOG_DIR / "qa_log.jsonl"
MISS_LOG = LOG_DIR / "miss_log.jsonl"
ERROR_LOG = LOG_DIR / "error_log.jsonl"


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir()
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    _append_jsonl(
        ERROR_LOG,
        {
            "stage": stage,
            "error": error,
            "meta": meta or {},
        },
    )


def read_recent(path: Path, limit: int = 20) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-max(1, limit):][::-1]


def get_recent_qa(limit: int = 20) -> list[Dict[str, Any]]:
    return read_recent(QA_LOG, limit=limit)


def get_recent_misses(limit: int = 20) -> list[Dict[str, Any]]:
    return read_recent(MISS_LOG, limit=limit)


def get_recent_errors(limit: int = 20) -> list[Dict[str, Any]]:
    return read_recent(ERROR_LOG, limit=limit)
