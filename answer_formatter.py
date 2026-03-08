from typing import List, Dict
from rag_runtime_config import REFERENCE_NOTE, RISK_NOTE


def format_structured_answer(
    route: str,
    body_lines: List[str],
    evidence: List[Dict] = None,
    add_risk_note: bool = False,
) -> str:
    title_map = {
        "basic": "佛教基础",
        "doctrine": "教义解说",
        "practice": "修行方法",
        "scripture": "经典介绍",
        "sect": "宗派介绍",
        "concept": "核心概念",
        "history": "佛教历史",
        "ritual": "节日与礼仪",
    }
    title = title_map.get(route, "回答")
    out = [f"{title}（资料提取）：", "结论："]
    for ln in body_lines:
        out.append(f"- {ln}")

    if evidence:
        out.append("依据：")
        seen_sources = set()
        for ev in evidence[:6]:
            source_file = ev.get("meta", {}).get("source_file", "unknown")
            chunk = ev.get("meta", {}).get("chunk_id", "?")
            stype = ev.get("meta", {}).get("source_type", "unknown")
            meta = ev.get("meta", {})
            kepan = ev.get("kepan_breadcrumb") or meta.get("kepan_breadcrumb", "")
            section = meta.get("section_title", "")
            ctype = meta.get("content_type", "")
            if kepan:
                source_key = f"{source_file}|{kepan}"
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                out.append(f"- 来源：{source_file}｜科判：{kepan}｜段落：{chunk}")
            elif section:
                ctype_label = {
                    "ritual": "仪轨", "talk": "开示", "method": "方法",
                    "article": "文章", "qa": "问答", "gongan": "公案",
                    "commentary": "注疏", "verse_collection": "偈颂",
                    "letter": "书信",
                }.get(ctype, "章节")
                source_key = f"{source_file}|{section}"
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                out.append(f"- 来源：{source_file}｜{ctype_label}：{section}｜段落：{chunk}")
            else:
                out.append(f"- 来源文件：{source_file}｜段落：{chunk}｜类型：{stype}")

    out.append("提示：")
    out.append(f"- {REFERENCE_NOTE}")
    if add_risk_note:
        out.append("修行建议：")
        out.append(f"- {RISK_NOTE}")
    return "\n".join(out).strip()
