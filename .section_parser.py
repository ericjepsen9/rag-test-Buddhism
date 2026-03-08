# section_parser.py
# 将主文档按章节切分（保留换行），避免后续 RAG 截断 STEP / 标题串台

import re
from typing import List, Dict, Any

# 匹配：一、 二、 三、 ... 十、
SECTION_RE = re.compile(r'(?m)^(?P<title>[一二三四五六七八九十]+、[^\n]*)\n?')

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去掉过多连续空行（保留结构）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def slugify_section_title(title: str) -> str:
    # 给章节一个稳定 key，方便后续检索过滤
    t = title.strip()
    if t.startswith("四、"):
        return "operation"
    if t.startswith("五、"):
        return "anti_fake"
    if t.startswith("六、"):
        return "aftercare"
    if t.startswith("七、"):
        return "contraindication"

    # 兜底：根据关键词猜
    if "防伪" in t or "鉴别" in t:
        return "anti_fake"
    if "注射" in t or "操作" in t:
        return "operation"
    if "术后" in t or "护理" in t:
        return "aftercare"
    if "禁忌" in t:
        return "contraindication"
    return "other"

def split_main_by_sections(text: str) -> List[Dict[str, Any]]:
    """
    返回:
    [
      {
        "section_title": "五、防伪鉴别方法（赛罗菲 / CELLOFILL）",
        "section_key": "anti_fake",
        "text": "完整章节文本（保留换行）"
      },
      ...
    ]
    """
    text = normalize_text(text)
    if not text:
        return []

    matches = list(SECTION_RE.finditer(text))
    if not matches:
        # 没有章节标题就整体作为一个块
        return [{
            "section_title": "全文",
            "section_key": "other",
            "text": text
        }]

    sections = []
    for i, m in enumerate(matches):
        title = m.group("title").strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        # 再清理一下，保留原有结构
        block = normalize_text(block)

        sections.append({
            "section_title": title,
            "section_key": slugify_section_title(title),
            "text": block
        })

    return sections


# 可选：把章节再拆成“子块”（按【小标题】或 STEP）——先保守开启，避免截断
SUBHEADER_RE = re.compile(r'(?m)^(【[^】]+】)\s*$')
STEP_RE = re.compile(r'(?m)^(STEP\s*\d+：)\s*$')

def split_section_to_subchunks(section_text: str, max_chars: int = 1200) -> List[str]:
    """
    优先按【小标题】/STEP分块；如果很短就不拆。
    注意：这里不会按固定长度硬切，避免把 STEP 行切断。
    """
    section_text = normalize_text(section_text)
    if len(section_text) <= max_chars:
        return [section_text]

    # 先按行扫，遇到【...】或 STEP 开新块
    lines = section_text.split("\n")
    chunks = []
    cur = []

    def flush():
        nonlocal cur
        if cur:
            txt = "\n".join(cur).strip()
            if txt:
                chunks.append(txt)
            cur = []

    for line in lines:
        line_s = line.strip()
        is_boundary = bool(SUBHEADER_RE.match(line_s) or STEP_RE.match(line_s))
        if is_boundary and cur:
            # 新小节开头，先收上一个块
            flush()
        cur.append(line)

        # 超长再切（尽量按行，不硬切）
        if sum(len(x) + 1 for x in cur) > max_chars:
            flush()

    flush()

    # 去重空块
    out = []
    seen = set()
    for c in chunks:
        k = re.sub(r"\s+", " ", c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out