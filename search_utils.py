import re
from typing import List, Dict, Tuple


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\ufeff", "")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_lines(text: str) -> List[str]:
    out = []
    for ln in normalize_text(text).split("\n"):
        s = re.sub(r"\s+", " ", ln).strip()
        if not s:
            continue
        if set(s) <= {"=", "-", "_", " "}:
            continue
        out.append(s)
    return out


def uniq(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = re.sub(r"\s+", " ", (x or "").strip())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def is_faq_line(line: str) -> bool:
    s = (line or "").strip()
    return s.startswith("【Q】") or s.startswith("【A】")


def section_block(text: str, titles: List[str], stops: List[str]) -> str:
    txt = normalize_text(text)
    if not txt:
        return ""
    start = None
    chosen = None
    for t in titles:
        idx = txt.find(t)
        if idx != -1 and (start is None or idx < start):
            start = idx
            chosen = t
    if start is None:
        return ""
    sub = txt[start:]
    end = None
    for s in stops:
        idx = sub.find(s)
        if idx > 0 and (end is None or idx < end):
            end = idx
    if end:
        sub = sub[:end]
    return sub.strip()


# ===== 中文分词（轻量级，不依赖 jieba）=====
# 使用 n-gram + 佛教术语词典的方式实现中文关键词匹配
_BUDDHIST_VOCAB = {
    "四圣谛", "八正道", "十二因缘", "三法印", "缘起", "中道", "涅槃", "轮回",
    "因果", "菩萨", "空性", "般若", "五蕴", "三宝", "六度", "菩提", "三毒",
    "贪嗔痴", "戒定慧", "五戒", "六波罗蜜", "菩提心", "禅修", "禅定", "念佛",
    "净土", "禅宗", "心经", "金刚经", "法华经", "华严经", "楞严经", "地藏经",
    "药师经", "坛经", "大乘", "小乘", "上座部", "藏传", "密宗", "律宗",
    "天台宗", "华严宗", "唯识宗", "三论宗", "净土宗", "法相宗", "阿罗汉",
    "佛性", "法身", "法界", "如来藏", "无明", "业力", "善业", "恶业",
    "六道", "布施", "持戒", "忍辱", "精进", "正见", "正语", "正业",
    "正命", "正念", "正定", "观音", "文殊", "普贤", "地藏", "释迦牟尼",
    "阿弥陀佛", "极乐世界", "白马寺", "鸠摩罗什", "玄奘", "达摩", "慧能",
    "龙树", "宗喀巴", "人间佛教", "大悲咒", "楞严咒", "六字大明咒",
    "苦谛", "集谛", "灭谛", "道谛", "色即是空", "应无所住",
}


def tokenize_chinese(text: str) -> List[str]:
    """轻量中文分词：先匹配佛教术语词典，再按 2-4 gram 切分"""
    text = (text or "").strip()
    if not text:
        return []
    tokens = []
    # 1. 匹配词典中的术语
    for term in sorted(_BUDDHIST_VOCAB, key=len, reverse=True):
        if term in text:
            tokens.append(term)
    # 2. 提取非中文的完整词（英文、数字等）
    for m in re.finditer(r"[a-zA-Z0-9]+", text):
        tokens.append(m.group().lower())
    # 3. 提取中文 bigram 作为补充
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


def keyword_score(query: str, text: str) -> float:
    q_tokens = tokenize_chinese(query.lower())
    if not q_tokens:
        return 0.0
    t_lower = (text or "").lower()
    hit = 0
    for term in q_tokens:
        if term in t_lower:
            hit += 1
    return hit / max(len(q_tokens), 1)


def keyword_search(query: str, docs: List[Dict], top_k: int = 8) -> List[Dict]:
    scored = []
    for d in docs:
        score = keyword_score(query, d.get("text", ""))
        if score <= 0:
            continue
        x = dict(d)
        x["keyword_score"] = score
        scored.append(x)
    scored.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)
    return scored[:top_k]


def merge_hybrid(vector_hits: List[Dict], keyword_hits: List[Dict], vw: float, kw: float, top_k: int) -> List[Dict]:
    merged = {}
    for h in vector_hits:
        key = h.get("id", h.get("text", ""))
        merged[key] = dict(h)
        merged[key]["hybrid_score"] = float(h.get("score", 0.0)) * vw
    for h in keyword_hits:
        key = h.get("id", h.get("text", ""))
        if key not in merged:
            merged[key] = dict(h)
            merged[key]["score"] = 0.0
            merged[key]["hybrid_score"] = 0.0
        merged[key]["hybrid_score"] += float(h.get("keyword_score", 0.0)) * kw
    out = list(merged.values())
    out.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return out[:top_k]


def split_multi_question(question: str, separators: List[str] = None) -> List[str]:
    separators = separators or ["；", ";", "。", "，另外", "并且", "同时", "还有"]
    parts = [question]
    for sep in separators:
        next_parts = []
        for p in parts:
            next_parts.extend(p.split(sep))
        parts = next_parts
    parts = [p.strip() for p in parts if p.strip()]
    return uniq(parts)


def detect_terms(question: str, term_map: Dict[str, List[str]]) -> List[str]:
    q = question.lower()
    found = []
    for key, aliases in term_map.items():
        for a in aliases:
            if a.lower() in q:
                found.append(key)
                break
    return uniq(found)


def match_faq(question: str, faq_text: str, faq_keyword_map: Dict[str, str]) -> str:
    """尝试精确匹配 FAQ 问答对，命中则直接返回答案"""
    if not faq_text:
        return ""
    q = (question or "").strip()

    # 找到最匹配的 FAQ 关键词
    matched_topic = None
    for keyword, topic in sorted(faq_keyword_map.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in q:
            matched_topic = topic
            break

    if not matched_topic:
        return ""

    # 在 FAQ 文本中查找对应的 Q&A 对
    lines = faq_text.split("\n")
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith("【Q】") and matched_topic in line_s:
            # 收集后续的 【A】 内容
            answer_parts = []
            for j in range(i + 1, len(lines)):
                al = lines[j].strip()
                if al.startswith("【Q】"):
                    break
                if al.startswith("【A】"):
                    answer_parts.append(al[3:].strip())
                elif al and answer_parts:
                    answer_parts[-1] += " " + al
            if answer_parts:
                return "\n".join(answer_parts)
    return ""
