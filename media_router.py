import json
from typing import List, Dict
from rag_media_config import MEDIA_FILE


def load_media_items() -> List[Dict]:
    if not MEDIA_FILE.exists():
        return []
    try:
        return json.loads(MEDIA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def find_media(question: str) -> List[Dict]:
    q = (question or "").lower()
    out = []
    for item in load_media_items():
        keys = item.get("keywords", [])
        if any(k.lower() in q for k in keys):
            out.append(item)
    return out[:6]
