from typing import List, Dict
from rag_runtime_config import REFERENCE_NOTE, RISK_NOTE


def format_structured_answer(
    route: str,
    body_lines: List[str],
    evidence: List[Dict] = None,
    add_risk_note: bool = False,
) -> str:
    title_map = {
        "basic": "基础资料",
        "operation": "操作说明",
        "aftercare": "术后护理",
        "risk": "风险/异常反应",
        "combo": "联合方案",
        "anti_fake": "防伪鉴别",
        "contraindication": "禁忌人群",
    }
    title = title_map.get(route, "回答")
    out = [f"{title}（资料提取）：", "结论："]
    for ln in body_lines:
        out.append(f"- {ln}")

    if evidence:
        out.append("依据：")
        for ev in evidence[:6]:
            source_file = ev.get("meta", {}).get("source_file", "unknown")
            chunk = ev.get("meta", {}).get("chunk_id", "?")
            stype = ev.get("meta", {}).get("source_type", "unknown")
            out.append(f"- 来源文件：{source_file}｜段落：{chunk}｜类型：{stype}")

    out.append("注意事项：")
    out.append(f"- {REFERENCE_NOTE}")
    if add_risk_note:
        out.append("需医生评估项：")
        out.append(f"- {RISK_NOTE}")
    return "\n".join(out).strip()
