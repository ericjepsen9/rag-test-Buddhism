from typing import List, Dict, Optional
from rag_runtime_config import REFERENCE_NOTE, RISK_NOTE, MAX_EVIDENCE_CHUNKS

# 内容类型显示标签（佛教文本类型）
_CTYPE_LABELS = {
    "ritual": "仪轨", "talk": "开示", "method": "方法",
    "article": "文章", "qa": "问答", "gongan": "公案",
    "commentary": "注疏", "verse_collection": "偈颂",
    "letter": "书信", "kepan": "科判", "pin": "品",
    "plain": "通用",
}

_SAFETY_ROUTES = frozenset(("doctrine", "concept"))

_TITLE_MAP = {
    "basic": "佛教基础",
    "doctrine": "教义解说",
    "practice": "修行方法",
    "scripture": "经典介绍",
    "sect": "宗派介绍",
    "concept": "核心概念",
    "history": "佛教历史",
    "ritual": "节日与礼仪",
}


def format_structured_answer(
    route: str,
    body_lines: List[str],
    evidence: Optional[List[Dict]] = None,
    add_risk_note: bool = False,
    custom_title: str = "",
) -> str:
    title = custom_title or _TITLE_MAP.get(route, "回答")
    out = [f"【{title}】", ""]

    # 正文
    valid_lines = [ln for ln in (body_lines or []) if ln and ln.strip()]
    if valid_lines:
        for ln in valid_lines:
            stripped = ln.strip()
            if not stripped:
                out.append("")
            elif stripped.startswith(("-", "•", "·", "【")):
                out.append(stripped)
            else:
                out.append(f"- {stripped}")
    else:
        out.append("- 当前知识库未覆盖该问题的直接内容。")

    # 依据来源：去重 + 过滤无效条目
    if evidence:
        source_entries = []
        seen_sources = set()
        for ev in evidence[:MAX_EVIDENCE_CHUNKS]:
            if ev is None:
                continue
            meta = ev.get("meta", {})
            if not meta:
                continue
            source_file = meta.get("source_file", "")
            chunk = meta.get("chunk_id", "")
            stype = meta.get("source_type", "")
            kepan = ev.get("kepan_breadcrumb") or meta.get("kepan_breadcrumb", "")
            section = meta.get("section_title", "")
            ctype = meta.get("content_type", "")

            if kepan:
                source_key = f"{source_file}|{kepan}"
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                entry = f"- {source_file}｜科判：{kepan}"
                if chunk:
                    entry += f"｜段落：{chunk}"
            elif section:
                source_key = f"{source_file}|{section}"
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                ctype_label = _CTYPE_LABELS.get(ctype, "章节")
                entry = f"- {source_file}｜{ctype_label}：{section}"
            elif source_file:
                source_key = source_file
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                ctype_label = _CTYPE_LABELS.get(stype, _CTYPE_LABELS.get(ctype, ""))
                entry = f"- {source_file}"
                if ctype_label:
                    entry += f"｜类型：{ctype_label}"
            else:
                continue

            source_entries.append(entry)

        if source_entries:
            out.append("")
            out.append("依据：")
            out.extend(source_entries)

    # 提示与风险提醒
    out.append("")
    if add_risk_note or route in _SAFETY_ROUTES:
        out.append(f"修行建议：{RISK_NOTE}")
    out.append(f"提示：{REFERENCE_NOTE}")
    return "\n".join(out).strip()
