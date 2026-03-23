import hashlib
import math
import os
import re
import sys
import threading
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from typing import Any, List, Dict, Tuple, Optional

from rag_runtime_config import (
    BM25_K1, BM25_B, SIGMOID_SCALE, CACHE_MAX_PRODUCTS, ROUTE_BOOST, JIEBA_ENABLED,
)

# jieba 分词延迟加载
_jieba = None
_jieba_initialized = False
_jieba_lock = threading.Lock()


def _get_jieba():
    """延迟加载 jieba 并添加佛教领域自定义词典"""
    global _jieba, _jieba_initialized
    if _jieba_initialized:
        return _jieba
    with _jieba_lock:
        if _jieba_initialized:
            return _jieba
        if not JIEBA_ENABLED:
            _jieba_initialized = True
            return None
        try:
            import jieba as _jieba_mod
            _CUSTOM_WORDS = list(_BUDDHIST_VOCAB)
            for w in _CUSTOM_WORDS:
                _jieba_mod.add_word(w)
            # 加载导入时提取的自定义词（data/jieba_user_dict.txt）
            try:
                from keyword_extractor import load_jieba_user_dict
                user_dict_count = load_jieba_user_dict(_jieba_mod)
                if user_dict_count > 0:
                    print(f"[INFO] 加载 jieba 用户词典: {user_dict_count} 条")
            except ImportError:
                pass
            _jieba = _jieba_mod
        except ImportError:
            print("[WARN] jieba 未安装，回退到 bigram 分词")
            _jieba = None
        _jieba_initialized = True
        return _jieba

# 预编译常用正则
_RE_WHITESPACE = re.compile(r"\s+")
_RE_TERM_SPLIT = re.compile(r"[\s,，;；、？?！!。]+")
_RE_CJK_WORD = re.compile(r"^[\u4e00-\u9fff]+$")
_SEPARATOR_CHARS = frozenset("=-_ ")

# jieba 分词结果缓存：同一查询在 expand_synonyms / _extract_terms 中避免重复分词
_JIEBA_CUT_CACHE: OrderedDict = OrderedDict()
_JIEBA_CUT_CACHE_MAX = int(os.environ.get("RAG_JIEBA_CACHE_MAX", "2000"))
_jieba_cut_lock = threading.Lock()


def _jieba_cut_cached(text: str, mode: str = "default") -> list:
    """缓存版 jieba 分词，避免相同文本重复分词。mode: 'default' | 'search'"""
    cache_key = f"{mode}|{text}"
    with _jieba_cut_lock:
        if cache_key in _JIEBA_CUT_CACHE:
            _JIEBA_CUT_CACHE.move_to_end(cache_key)
            return _JIEBA_CUT_CACHE[cache_key]
    j = _get_jieba()
    if j is None:
        return []
    if mode == "search":
        result = list(j.cut_for_search(text))
    else:
        result = list(j.cut(text))
    with _jieba_cut_lock:
        if len(_JIEBA_CUT_CACHE) >= _JIEBA_CUT_CACHE_MAX:
            _JIEBA_CUT_CACHE.popitem(last=False)
        _JIEBA_CUT_CACHE[cache_key] = result
    return result


# ===== 科判（层级大纲标记）解析 =====
# 佛教论典常用的层级编号：甲一 > 乙一 > 丙一 > 丁一 > 戊一 > 己一 > 庚一 > 辛一 > 壬一 > 癸一
_KEPAN_TIAN_GAN = "甲乙丙丁戊己庚辛壬癸"
_KEPAN_NUM = "一二三四五六七八九十"

# 匹配科判标记，如 "甲一"、"乙二（论义）"、"丙一（真实宣说）分三"
_KEPAN_RE = re.compile(
    r'^(' + '|'.join(_KEPAN_TIAN_GAN) + ')([' + _KEPAN_NUM + r']+)'
    r'(?:[、，\s（(].*)?$'
)

# 简单编号匹配（一、二、三...）
_SIMPLE_NUM_RE = re.compile(r'^[' + _KEPAN_NUM + r']+、')


def kepan_level(marker_char: str) -> int:
    """返回科判层级，甲=0, 乙=1, 丙=2..."""
    idx = _KEPAN_TIAN_GAN.find(marker_char)
    return idx if idx >= 0 else -1


def parse_kepan_marker(line: str) -> Optional[Dict]:
    """解析一行文本，如果是科判标记则返回信息字典"""
    line = line.strip()
    m = _KEPAN_RE.match(line)
    if not m:
        return None
    gan = m.group(1)       # 如 "甲"
    num = m.group(2)       # 如 "一"
    level = kepan_level(gan)
    return {
        "level": level,
        "marker": f"{gan}{num}",
        "full_line": line,
        "title": line,
    }


def split_by_kepan(text: str) -> List[Dict]:
    """
    按科判标记切分文本。
    返回列表，每项包含:
      - title: 科判标题行
      - level: 层级深度
      - marker: 如 "甲一"
      - breadcrumb: 完整层级路径，如 "甲二（论义）> 乙一（入造论之理）> 丙一（真实宣说）"
      - body: 该科判下的正文内容
    如果文本不含科判，返回空列表。
    """
    lines = text.split("\n")
    sections = []
    # 维护层级栈：stack[level] = title
    stack = {}

    current = None
    body_lines = []

    def flush():
        nonlocal current, body_lines
        if current is not None:
            current["body"] = "\n".join(body_lines).strip()
            sections.append(current)
            body_lines = []

    for line in lines:
        info = parse_kepan_marker(line)
        if info:
            flush()
            level = info["level"]
            # 更新层级栈：清除同级及更深的层级
            for k in list(stack.keys()):
                if k >= level:
                    del stack[k]
            stack[level] = info["title"]
            # 构建面包屑路径
            breadcrumb_parts = []
            for k in sorted(stack.keys()):
                breadcrumb_parts.append(stack[k])
            current = {
                "title": info["title"],
                "level": level,
                "marker": info["marker"],
                "breadcrumb": " > ".join(breadcrumb_parts),
            }
            body_lines = []
        else:
            body_lines.append(line)

    flush()
    return sections


def has_kepan_structure(text: str) -> bool:
    """检查文本是否包含科判结构"""
    for line in text.split("\n")[:200]:
        if parse_kepan_marker(line.strip()):
            return True
    return False


# ===== 品（章节）结构检测 =====
# 匹配 "第一品"、"第二品 菩提心利益" 等
_PIN_RE = re.compile(r'^第?([' + _KEPAN_NUM + r']+)品\s*(.*)')
# 也匹配 "品第一" 格式
_PIN_ALT_RE = re.compile(r'^(.+?)品第([' + _KEPAN_NUM + r']+)')


def parse_pin_marker(line: str) -> Optional[Dict]:
    """解析品标记，如 '第一品 菩提心利益'"""
    line = line.strip()
    m = _PIN_RE.match(line)
    if m:
        return {"num": m.group(1), "subtitle": m.group(2).strip(), "full_line": line}
    m = _PIN_ALT_RE.match(line)
    if m:
        return {"num": m.group(2), "subtitle": m.group(1).strip(), "full_line": line}
    return None


def has_pin_structure(text: str) -> bool:
    """检查文本是否包含品结构"""
    count = 0
    for line in text.split("\n"):
        if parse_pin_marker(line.strip()):
            count += 1
            if count >= 2:
                return True
    return False


def split_by_pin(text: str) -> List[Dict]:
    """
    按品切分文本。
    返回列表，每项包含 title, body。
    品之前的内容作为 "序言" 部分。
    """
    lines = text.split("\n")
    sections = []
    current = None
    body_lines = []
    preamble_lines = []

    def flush():
        nonlocal current, body_lines
        if current is not None:
            current["body"] = "\n".join(body_lines).strip()
            sections.append(current)
            body_lines = []

    for line in lines:
        info = parse_pin_marker(line)
        if info:
            if current is None and preamble_lines:
                # 品之前的序言内容
                sections.append({
                    "title": "序言",
                    "body": "\n".join(preamble_lines).strip(),
                })
            flush()
            subtitle = info["subtitle"]
            title = info["full_line"]
            current = {"title": title}
            body_lines = []
        elif current is None:
            preamble_lines.append(line)
        else:
            body_lines.append(line)

    flush()
    return sections


# ===== 颂词 + 注释分块 =====
# 颂词特征：四句偈（每句以逗号或句号结尾，大致等长），或两句对仗
_VERSE_LINE_RE = re.compile(r'^[^\d【（(]{2,20}[，,]$')  # 偈颂行：短句+逗号结尾
_CITATION_RE = re.compile(r'(?:《.+?》|如|经|论|云|中说|中云|偈云|颂云)[:：]')


def split_semantic_paragraphs(text: str) -> List[str]:
    """
    将一段论典正文按语义边界分成段落。
    分割点：
    1. 双换行（空行）
    2. 颂词起始行（识别偈颂格式）
    3. 经论引用标记（《XX经》中云：）
    保证颂词与其紧随的注释在同一段落内。
    """
    if not text.strip():
        return []

    # 先按空行分段
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        # 没有空行分隔，尝试按颂词/引用边界分
        return _split_by_content_boundary(text)

    return paragraphs


def _split_by_content_boundary(text: str) -> List[str]:
    """当文本没有空行时，按内容边界智能分段"""
    lines = text.split("\n")
    blocks = []
    current_block = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            continue

        # 检查是否是新的内容起点
        is_boundary = False
        if i > 0 and current_block:
            # 颂词/偈颂起始（检测连续的短句+逗号模式）
            if _looks_like_verse_start(stripped, lines, i):
                is_boundary = True
            # 新的引用起始："《XX》中云："、"如云："
            elif re.match(r'^(关于|所谓|《|如云|如经|经中)', stripped):
                is_boundary = True
            # 偈颂/引用内联标记："偈云："、"颂曰："、"论云："
            elif re.match(r'^(偈云|偈曰|颂[云曰]|论[云曰]|经[云曰]|故[云曰]|又[云曰])[:：]', stripped):
                is_boundary = True

        if is_boundary and current_block:
            blocks.append("\n".join(current_block))
            current_block = []

        current_block.append(stripped)

    if current_block:
        blocks.append("\n".join(current_block))

    return [b for b in blocks if b.strip()]


def _looks_like_verse_start(line: str, all_lines: List[str], idx: int) -> bool:
    """
    判断一行是否像偈颂的开头。
    佛教偈颂通常是四句（或两句），每句大致等长，以逗号或句号结尾。
    如：善逝法身佛子伴，及诸应敬我悉礼。
    """
    # 如果这一行包含逗号分隔的两个短语，且下一行也有类似格式
    parts = re.split(r'[，,]', line)
    if len(parts) >= 2:
        lens = [len(p.strip()) for p in parts if p.strip()]
        # 各句长度差别不超过3个字，且每句至少4字
        if lens and min(lens) >= 4 and max(lens) - min(lens) <= 3:
            return True
    return False


# ===== 内容类型检测与分段 =====
# 支持：演讲/开示、仪轨、方法指导、生活佛法、问答体、公案语录、
#       注疏体、偈颂集、书信体等非论典格式

# 话题转换标记（演讲/开示类）
_TOPIC_SHIFT_RE = re.compile(
    r'^(?:'
    r'今天(?:我们)?(?:讲|来谈|来讲|来说|讨论|学习)|'
    r'下面(?:我们)?(?:讲|来谈|来讲|来说|来看|谈)|'
    r'接下来|接着(?:讲|说|谈)|'
    r'第[一二三四五六七八九十\d]+[、，,\s]|'
    r'首先|其次|再[次者]|最后|'
    r'问[:：]|答[:：]|'
    r'(?:有人|弟子|居士|学员|同学)(?:问|提问)|'
    r'(?:上师|法师|师父|堪布|仁波切)(?:答|开示|说|回答)'
    r')'
)

# 仪轨标记
_RITUAL_MARKERS = re.compile(
    r'(?:念[三七二十百千万\d]+遍|'
    r'[（(](?:合掌|顶礼|跪|长跪|起立|站立|绕行|三拜|拈香|问讯)[）)]|'
    r'[（(](?:念诵|唱诵|默念|齐念|和念)[）)]|'
    r'唵|嗡|南无|皈依|发愿文|回向文|回向偈|忏悔文|'
    r'愿以此功德|上报四重恩|'
    r'[（(](?:主法|维那|大众)[）)])'
)

# 方法/步骤标记
_STEP_RE = re.compile(
    r'^(?:'
    r'第[一二三四五六七八九十\d]+步|'
    r'步骤[一二三四五六七八九十\d]+|'
    r'要点[一二三四五六七八九十\d]+|'
    r'方法[一二三四五六七八九十\d]+|'
    r'注意事项|要领|窍诀|关键|'
    r'\d+[）\)\.、]'
    r')'
)

# 标题/小节标记（通用）
_HEADING_RE = re.compile(
    r'^(?:'
    r'【.+?】|'                           # 【标题】
    r'[一二三四五六七八九十]+[、，]\s*\S|'   # 一、标题
    r'\d+[\.、]\s*\S|'                    # 1. 标题 / 1、标题
    r'#{1,4}\s+\S'                        # markdown 标题
    r')'
)

# 问答体标记（问：...答：... 成对出现）
_QA_QUESTION_RE = re.compile(
    r'^(?:'
    r'问[:：]|'
    r'问题[:：]|'
    r'(?:有人|弟子|居士|学员|某某|信众)(?:问|请问|启问)[:：]?|'
    r'Q[:：]|'
    r'【Q】|【问】'
    r')'
)
_QA_ANSWER_RE = re.compile(
    r'^(?:'
    r'答[:：]|'
    r'回答[:：]|'
    r'(?:上师|法师|师父|堪布|仁波切|大师|和尚|长老)(?:答|回答|开示|说)[:：]?|'
    r'A[:：]|'
    r'【A】|【答】'
    r')'
)

# 公案/语录体标记
_GONGAN_RE = re.compile(
    r'^(?:'
    r'(?:师|祖|和尚)(?:云|曰|示众|上堂|问|答)|'
    r'(?:僧|学人|弟子|一人)(?:问|云|曰)|'
    r'举[:：]|颂[:：]|颂曰[:：]?|评唱[:：]|着语[:：]|'
    r'(?:垂示|本则|评唱)[:：]?|'
    r'公案|则\s*$'
    r')'
)

# 注疏体标记（引用原文 + 解释）
_COMMENTARY_QUOTE_RE = re.compile(
    r'^(?:'
    r'[「『"《]|'                         # 引号/书名号开头
    r'(?:经|论|颂|偈)(?:云|曰|中说|言)[:：]?|'
    r'(?:原文|正文|颂词|根本颂|论文)[:：]|'
    r'(?:释|解|注|疏|讲|释义|解释|注释)[:：]'
    r')'
)

# 偈颂集特征：连续短句、等长、以逗号/句号结尾
_VERSE_COUPLET_RE = re.compile(
    r'^[\u4e00-\u9fff]{3,12}[，,。．][\u4e00-\u9fff]{3,12}[，,。．、！]?\s*$'
)

# 书信体标记
_LETTER_RE = re.compile(
    r'^(?:'
    r'.{1,10}(?:居士|法师|仁者|大德|道友|檀越|施主|长老)(?:慈鉴|鉴|尊鉴|道鉴|惠鉴|足下|法席)[：:]?|'
    r'(?:复|答|与|致|上|覆).{1,10}(?:居士|法师|仁者)(?:书|函|启)|'
    r'(?:谨复|敬覆|奉复|恭答)|'
    r'(?:谨此|敬颂|即颂|此致|肃此)\s*(?:法安|道安|吉祥|如意|安好)'
    r')'
)


def detect_content_type(text: str) -> str:
    """
    检测佛教内容的类型。
    返回: "kepan" | "pin" | "ritual" | "qa" | "gongan" | "commentary" |
          "verse_collection" | "letter" | "talk" | "method" | "article" | "plain"

    阈值自适应：根据采样行数动态调整，避免长文档误判。
    短文档（<100行）使用固定最小阈值，长文档要求更高的标记密度。
    """
    if has_kepan_structure(text):
        return "kepan"
    if has_pin_structure(text):
        return "pin"

    lines = text.split("\n")
    # 扩大采样范围到 500 行，更好地覆盖长文档
    sample_lines = [l.strip() for l in lines[:500] if l.strip()]
    n = len(sample_lines)

    ritual_count = 0
    topic_count = 0
    step_count = 0
    heading_count = 0
    qa_q_count = 0
    qa_a_count = 0
    gongan_count = 0
    commentary_count = 0
    verse_count = 0
    letter_count = 0

    for line in sample_lines:
        if _RITUAL_MARKERS.search(line):
            ritual_count += 1
        if _TOPIC_SHIFT_RE.match(line):
            topic_count += 1
        if _STEP_RE.match(line):
            step_count += 1
        if _HEADING_RE.match(line):
            heading_count += 1
        if _QA_QUESTION_RE.match(line):
            qa_q_count += 1
        if _QA_ANSWER_RE.match(line):
            qa_a_count += 1
        if _GONGAN_RE.match(line):
            gongan_count += 1
        if _COMMENTARY_QUOTE_RE.match(line):
            commentary_count += 1
        if _VERSE_COUPLET_RE.match(line):
            verse_count += 1
        if _LETTER_RE.match(line):
            letter_count += 1

    def adaptive_threshold(min_count: int, density: float) -> int:
        """自适应阈值：max(固定最小值, 采样行数 * 密度比例)"""
        return max(min_count, int(n * density))

    # 仪轨类：标记密度 >= 1%，至少3个
    if ritual_count >= adaptive_threshold(3, 0.01):
        return "ritual"
    # 问答体：问答成对出现，各至少2个，密度 >= 0.5%
    qa_thresh = adaptive_threshold(2, 0.005)
    if qa_q_count >= qa_thresh and qa_a_count >= qa_thresh:
        return "qa"
    # 公案/语录体：密度 >= 1%，至少3个
    if gongan_count >= adaptive_threshold(3, 0.01):
        return "gongan"
    # 注疏体：密度 >= 1.5%，至少4个
    if commentary_count >= adaptive_threshold(4, 0.015):
        return "commentary"
    # 偈颂集：至少6行且占采样行数 20%（从30%降低，更合理）
    if verse_count >= 6 and verse_count > n * 0.2:
        return "verse_collection"
    # 书信体：有称谓或落款（保持不变，1个即可）
    if letter_count >= 1:
        return "letter"
    # 方法指导类：密度 >= 1%，至少3个
    if step_count >= adaptive_threshold(3, 0.01):
        return "method"
    # 演讲/开示类：密度 >= 0.5%，至少2个
    if topic_count >= adaptive_threshold(2, 0.005):
        return "talk"
    # 文章类：密度 >= 1%，至少3个
    if heading_count >= adaptive_threshold(3, 0.01):
        return "article"

    return "plain"


def split_by_topic(text: str) -> List[Dict]:
    """
    按话题切分演讲/开示类文本。
    在话题转换处（"下面讲..."、"问："、"第一，"等）分段。
    每段保留完整的话题讨论（问答不拆开）。
    """
    lines = text.split("\n")
    sections = []
    current_title = "开场"
    body_lines = []

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        is_topic = _TOPIC_SHIFT_RE.match(stripped)
        # 问答对中的"答"不作为分割点
        is_answer = re.match(r'^(?:答|(?:上师|法师|师父)(?:答|回答|说))', stripped)

        if is_topic and not is_answer and body_lines:
            flush()
            # 用转换句作为新话题标题
            current_title = stripped[:30]

        body_lines.append(stripped)

    flush()
    return sections


def split_by_ritual_section(text: str) -> List[Dict]:
    """
    切分仪轨类文本。
    按仪轨段落分：发愿、皈依、正行、回向等。
    咒语/念诵指令和正文保持在一起。
    """
    lines = text.split("\n")
    sections = []
    current_title = "仪轨"
    body_lines = []

    # 仪轨大段标记
    ritual_section_re = re.compile(
        r'^(?:【.+?】|'
        r'[一二三四五六七八九十]+[、，]\s*|'
        r'(?:前行|正行|结行|回向|发愿|皈依|忏悔|供养|礼赞|加持|灌顶|传承|观想|持咒|念诵|祈请))'
    )

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        m = ritual_section_re.match(stripped)
        if m and body_lines and len(body_lines) > 1:
            flush()
            current_title = stripped[:30]

        body_lines.append(stripped)

    flush()
    return sections


def split_by_steps(text: str) -> List[Dict]:
    """
    切分方法指导类文本。
    按主要步骤/要点分段。编号子项（1）2）3）等）不单独成段，
    归入其前面的步骤或要点中。
    """
    # 主步骤标记（不含简单的 1）2）3） 编号）
    major_step_re = re.compile(
        r'^(?:'
        r'第[一二三四五六七八九十\d]+步|'
        r'步骤[一二三四五六七八九十\d]+|'
        r'要点[一二三四五六七八九十\d]+|'
        r'方法[一二三四五六七八九十\d]+|'
        r'注意事项|要领|窍诀|关键'
        r')'
    )

    lines = text.split("\n")
    sections = []
    current_title = "概述"
    body_lines = []

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        # 只在主步骤处分段，编号子项不分段
        if major_step_re.match(stripped) and body_lines:
            flush()
            current_title = stripped[:30]

        body_lines.append(stripped)

    flush()
    return sections


def split_by_headings(text: str) -> List[Dict]:
    """
    按标题/小节切分文章类文本。
    识别 【标题】、一、标题、1. 标题、# 标题 等格式。
    """
    lines = text.split("\n")
    sections = []
    current_title = "引言"
    body_lines = []

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        if _HEADING_RE.match(stripped) and body_lines:
            flush()
            # 清理标题：去掉 #、【】 等
            clean_title = re.sub(r'^#+\s*', '', stripped)
            clean_title = re.sub(r'^【(.+?)】.*', r'\1', clean_title)
            current_title = clean_title[:30]

        body_lines.append(stripped)

    flush()
    return sections


def split_by_qa(text: str) -> List[Dict]:
    """
    切分问答体文本。
    每个"问+答"作为一个完整单元，不拆开。
    支持：问：/答：、【Q】/【A】、弟子问/上师答 等格式。
    """
    lines = text.split("\n")
    sections = []
    current_title = "序"
    body_lines = []
    in_qa = False

    def flush():
        nonlocal current_title, body_lines, in_qa
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []
        in_qa = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        is_q = _QA_QUESTION_RE.match(stripped)
        if is_q and body_lines:
            flush()
            # 提取问题摘要作为标题
            q_text = re.sub(r'^(?:问[:：]|【Q】|【问】)\s*', '', stripped)
            current_title = f"问：{q_text[:25]}" if q_text else "问答"
            in_qa = True

        body_lines.append(stripped)

    flush()
    return sections


def split_by_gongan(text: str) -> List[Dict]:
    """
    切分公案/语录体文本。
    每则公案（含举、颂、评唱）作为一个单元。
    禅宗语录按对话轮次分段。
    """
    lines = text.split("\n")
    sections = []
    current_title = "语录"
    body_lines = []

    # 公案起始标记
    case_start_re = re.compile(
        r'^(?:'
        r'第?[一二三四五六七八九十百\d]+则|'
        r'举[:：]|'
        r'垂示[:：]|'
        r'本则[:：]|'
        r'师(?:上堂|示众|云|曰)[:：]?'
        r')'
    )

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        if case_start_re.match(stripped) and body_lines and len(body_lines) > 1:
            flush()
            current_title = stripped[:30]

        body_lines.append(stripped)

    flush()
    return sections


def split_by_commentary(text: str) -> List[Dict]:
    """
    切分注疏体文本。
    将"原文引用 + 解释"作为一个单元。
    引用和紧随的注释保持在一起。
    """
    lines = text.split("\n")
    sections = []
    current_title = "注疏"
    body_lines = []

    # 新的引用段起始
    quote_start_re = re.compile(
        r'^(?:'
        r'[「『"《]|'
        r'(?:经|论|颂|偈)(?:云|曰|中说)[:：]?|'
        r'(?:原文|正文|颂词|根本颂|论文)[:：]'
        r')'
    )

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        if quote_start_re.match(stripped) and body_lines and len(body_lines) > 2:
            flush()
            # 提取引用内容摘要
            quote_preview = re.sub(r'^[「『"《]', '', stripped)[:20]
            current_title = f"注：{quote_preview}" if quote_preview else "注疏"

        body_lines.append(stripped)

    flush()
    return sections


def split_by_verse_collection(text: str) -> List[Dict]:
    """
    切分偈颂集文本。
    每首偈颂（通常4句或8句为一组）作为一个单元。
    如有标题或编号则在标题处分段。
    """
    lines = text.split("\n")
    sections = []
    current_title = "偈颂"
    body_lines = []
    verse_line_count = 0

    # 偈颂标题标记
    verse_title_re = re.compile(
        r'^(?:'
        r'第?[一二三四五六七八九十百\d]+(?:首|偈|颂|章)|'
        r'【.+?】|'
        r'[一二三四五六七八九十]+[、，]'
        r')'
    )

    def flush():
        nonlocal current_title, body_lines, verse_line_count
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []
        verse_line_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 空行后如果已有>=4行偈颂，则分段
            if verse_line_count >= 4 and body_lines:
                flush()
            else:
                body_lines.append(line)
            continue

        if verse_title_re.match(stripped) and body_lines:
            flush()
            current_title = stripped[:30]

        if _VERSE_COUPLET_RE.match(stripped):
            verse_line_count += 1
        else:
            verse_line_count = 0

        body_lines.append(stripped)

    flush()
    return sections


def split_by_letter(text: str) -> List[Dict]:
    """
    切分书信体文本。
    每封信（称谓到落款）作为一个单元。
    支持：复某某居士书、某某居士慈鉴 等格式。
    """
    lines = text.split("\n")
    sections = []
    current_title = "书信"
    body_lines = []

    # 信件起始标记（称谓行或标题行）
    letter_start_re = re.compile(
        r'^(?:'
        r'.{1,10}(?:居士|法师|仁者|大德|道友)(?:慈鉴|鉴|尊鉴|道鉴|惠鉴|足下)[：:]?|'
        r'(?:复|答|与|致|上|覆).{1,10}(?:居士|法师|仁者)(?:书|函|启)'
        r')'
    )

    def flush():
        nonlocal current_title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})
        body_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        m = letter_start_re.match(stripped)
        if m and body_lines:
            flush()
            current_title = stripped[:30]

        body_lines.append(stripped)

    flush()
    return sections


# ===== 混合内容子分段（科判/品内部） =====
# 识别科判体内部的颂词、讲解、公案、问答等子结构边界

# 子内容标记正则
_SUB_CONTENT_RE = re.compile(
    r'^(?:'
    r'颂词[:：]|颂曰[:：]?|偈云[:：]?|'          # 颂词起始
    r'讲解[:：]|释[:：]|解释[:：]|注[:：]|'       # 讲解/注释起始
    r'公案[:：]|故事[:：]|比喻[:：]|'             # 公案/故事起始
    r'问[:：]|答[:：]|'                          # 问答起始
    r'(?:有人|弟子|居士)(?:问|请问)[:：]?|'
    r'(?:上师|法师|师父)(?:答|说)[:：]?|'
    r'[「『"]|'                                  # 引文起始
    r'(?:经|论)(?:云|曰|中说)[:：]?'             # 经论引用
    r')'
)


def split_mixed_body(body: str) -> List[Dict]:
    """
    将科判/品内部的混合内容按子结构边界分段。
    保证：
    - 颂词 + 紧随的讲解 在一起
    - 问 + 答 在一起
    - 公案/故事完整保留
    - 引文 + 解释 在一起

    返回 [{"text": ..., "sub_type": "verse"|"commentary"|"qa"|"story"|"text"}, ...]
    """
    if not body or not body.strip():
        return []

    lines = body.split("\n")
    segments = []
    current_type = "text"
    current_lines = []

    def flush():
        nonlocal current_type, current_lines
        text = "\n".join(current_lines).strip()
        if text:
            segments.append({"text": text, "sub_type": current_type})
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        # 检测子内容类型转换（支持【标记】和 标记： 两种格式）
        new_type = None
        if re.match(r'^(?:【?颂词】?|颂曰|偈云)[:：]?$', stripped) or \
           re.match(r'^(?:颂词|颂曰|偈云)[:：]', stripped):
            new_type = "verse"
        elif re.match(r'^(?:【?讲解】?|【?注释】?)[:：]?$', stripped) or \
             re.match(r'^(?:讲解|释|解释|注)[:：]', stripped):
            new_type = "commentary"
        elif re.match(r'^(?:【?公案】?|【?故事】?|【?比喻】?)[:：]?$', stripped) or \
             re.match(r'^(?:公案|故事|比喻)[:：]', stripped):
            new_type = "story"
        elif re.match(r'^【?引用】?[:：]?$', stripped):
            new_type = "commentary"  # 引用视为注释类，与前文保持一体
        elif _QA_QUESTION_RE.match(stripped):
            new_type = "qa"

        # 类型转换时的合并规则
        if new_type:
            if current_type == "verse" and new_type == "commentary":
                # 颂词后紧跟讲解 → 不分段，合在一起
                current_lines.append(stripped)
                current_type = "verse_commentary"
                continue
            elif current_type == "qa" and new_type == "qa":
                # 连续问答，检查是否是"答"（不分段）
                if _QA_ANSWER_RE.match(stripped):
                    current_lines.append(stripped)
                    continue
                # 新的"问"→ 分段
                flush()
                current_type = new_type
                current_lines.append(stripped)
                continue
            elif new_type != current_type:
                if current_lines:
                    flush()
                current_type = new_type
        elif _QA_ANSWER_RE.match(stripped) and current_type == "qa":
            # 答跟在问后面，不分段
            current_lines.append(stripped)
            continue

        current_lines.append(stripped)

    flush()
    return segments


def merge_sub_segments(segments: List[Dict], max_size: int) -> List[str]:
    """
    将 split_mixed_body 产出的子段落合并为适当大小的块。
    规则：
    1. verse_commentary（颂词+讲解）不拆开
    2. qa（问+答）不拆开
    3. story（公案）不拆开
    4. 超过 max_size 的段落用滑动窗口切分
    5. 短段落可以合并（但不跨类型合并）
    """
    if not segments:
        return []

    chunks = []
    buffer_text = ""

    for seg in segments:
        seg_text = seg["text"]
        seg_type = seg["sub_type"]

        # 如果单个段落超过 max_size，先产出 buffer，再切分超长段
        if len(seg_text) > max_size:
            if buffer_text:
                chunks.append(buffer_text)
                buffer_text = ""
            # 超长段滑动切分，但在标记行处优先分割
            sub_parts = _smart_split_long_segment(seg_text, max_size)
            chunks.extend(sub_parts)
            continue

        # 尝试与 buffer 合并
        if buffer_text:
            combined_len = len(buffer_text) + len(seg_text) + 2
            if combined_len <= max_size:
                buffer_text += "\n\n" + seg_text
                continue
            else:
                chunks.append(buffer_text)
                buffer_text = seg_text
        else:
            buffer_text = seg_text

    if buffer_text:
        chunks.append(buffer_text)

    return chunks


def _smart_split_long_segment(text: str, max_size: int) -> List[str]:
    """对超长段落做智能切分，优先在句号/段落处分割"""
    if len(text) <= max_size:
        return [text]

    chunks = []
    # 按句号分割
    sentences = re.split(r'(?<=[。！？\n])', text)
    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) > max_size and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += sent

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return text


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
    if end is not None:
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
    # 入菩萨行论及论典常见术语
    "入菩萨行论", "寂天", "善说海", "无著菩萨",
    "菩萨戒", "律仪", "暇满", "人身难得", "发心", "回向",
    "安忍", "静虑", "智慧品", "不放逸", "正知正念",
    "礼赞句", "立誓句", "善逝", "法身", "佛子",
    "科判", "品", "偈颂", "颂词", "注释",
}
# 预排序：长词优先，避免每次 tokenize_chinese 调用时重排
_BUDDHIST_VOCAB_SORTED = sorted(_BUDDHIST_VOCAB, key=len, reverse=True)


def tokenize_chinese(text: str) -> List[str]:
    """轻量中文分词：先匹配佛教术语词典，再按 bigram 补充未覆盖部分"""
    text = (text or "").strip()
    if not text:
        return []
    tokens = []
    seen = set()
    # 1. 匹配词典中的术语（长词优先，消除已覆盖的字符）
    remaining = text.lower()
    for term in _BUDDHIST_VOCAB_SORTED:
        if term in remaining:
            if term not in seen:
                tokens.append(term)
                seen.add(term)
            # 标记已覆盖的位置，避免 bigram 重复覆盖
            remaining = remaining.replace(term, "\x00" * len(term))
    # 2. 提取非中文的完整词（英文、数字等）
    for m in re.finditer(r"[a-zA-Z0-9]+", text):
        w = m.group().lower()
        if w not in seen:
            tokens.append(w)
            seen.add(w)
    # 3. 只对未被词典覆盖的部分提取 bigram
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", remaining)
    for i in range(len(chinese_chars) - 1):
        bigram = chinese_chars[i] + chinese_chars[i + 1]
        if bigram not in seen:
            tokens.append(bigram)
            seen.add(bigram)
    return tokens


# ===== 查询词提取（jieba + bigram 混合）=====

def _extract_terms_bigram(query: str) -> List[str]:
    """原始 bigram 分词：按空格/标点分割，再对中文长词做 bigram 切分（jieba 不可用时的回退方案）"""
    raw = [x for x in _RE_TERM_SPLIT.split(query.lower()) if x]
    terms = []
    seen = set()
    for w in raw:
        if w not in seen:
            terms.append(w)
            seen.add(w)
        # 对纯中文且长度>=3的词做 bigram 切分，提高部分匹配能力
        if len(w) >= 3 and _RE_CJK_WORD.match(w):
            for i in range(len(w) - 1):
                bg = w[i:i+2]
                if bg not in seen:
                    terms.append(bg)
                    seen.add(bg)
    return terms


def _extract_terms_jieba(query: str) -> List[str]:
    """jieba 分词 + bigram 补充：先用 jieba 精确分词，再对长中文词补充 bigram 提高部分匹配能力"""
    q_lower = query.lower()
    # jieba 切词（搜索引擎模式：更细粒度，召回更高）——使用缓存版避免重复分词
    raw_words = _jieba_cut_cached(q_lower, "search")
    terms = []
    seen = set()
    for w in raw_words:
        w = w.strip()
        if not w or w in seen:
            continue
        # 过滤纯标点和空白
        if _RE_TERM_SPLIT.fullmatch(w):
            continue
        terms.append(w)
        seen.add(w)
        # 对 jieba 切出的长中文词仍做 bigram 补充，增加部分匹配能力
        if len(w) >= 4 and _RE_CJK_WORD.match(w):
            for i in range(len(w) - 1):
                bg = w[i:i+2]
                if bg not in seen:
                    terms.append(bg)
                    seen.add(bg)
    return terms


def _extract_terms(query: str) -> List[str]:
    """从查询中提取搜索词：优先 jieba 分词，不可用时回退 bigram"""
    if _get_jieba() is not None:
        return _extract_terms_jieba(query)
    return _extract_terms_bigram(query)


# ===== Sigmoid 归一化 =====

def _sigmoid_norm(raw_score: float) -> float:
    """BM25 分数归一化：sigmoid(score/scale)，钳位防 exp 溢出。
    将 raw_score 量化到 2 位小数以提高 LRU 缓存命中率。"""
    z = round(raw_score / SIGMOID_SCALE, 2)
    return _sigmoid_cached(max(-20.0, min(20.0, z)))


@lru_cache(maxsize=512)
def _sigmoid_cached(z: float) -> float:
    """LRU 缓存的 sigmoid：典型 BM25 分数分布在有限区间，命中率高。"""
    return 1.0 / (1.0 + math.exp(-z))


# ===== 佛教术语同义词映射 =====
_SYNONYM_MAP = {
    # 教义别名
    "苦集灭道": "四圣谛", "四谛": "四圣谛",
    "缘起法": "缘起", "十二缘起": "十二因缘",
    "三无漏学": "戒定慧", "三学": "戒定慧",
    "六波罗蜜": "六度", "六波罗蜜多": "六度",
    # 概念别名
    "空": "空性", "真空": "空性",
    "轮转": "轮回", "生死轮回": "轮回",
    "业报": "因果", "因果报应": "因果",
    "菩提道次第": "道次第",
    "自他交换": "自他相换",
    # 修行别名
    "坐禅": "禅修", "打坐": "禅修", "静坐": "禅修",
    "持名": "念佛", "持名念佛": "念佛",
    "诵经": "读经",
    # 宗派别名
    "密教": "密宗", "金刚乘": "密宗",
    "小乘": "上座部", "声闻乘": "上座部",
    "南传佛教": "上座部",
    # 经典别名
    "般若波罗蜜多心经": "心经",
    "金刚般若波罗蜜经": "金刚经",
    "妙法莲华经": "法华经",
    "大方广佛华严经": "华严经",
    "入菩萨行论": "入行论",
    # 人物别名
    "释迦牟尼佛": "佛陀", "世尊": "佛陀",
    "六祖": "慧能", "惠能": "慧能",
}

# 反向映射：同义词扩展（基础静态映射）
_SYNONYM_EXPAND_BASE: Dict[str, set] = {}
for _k, _v in _SYNONYM_MAP.items():
    _SYNONYM_EXPAND_BASE.setdefault(_v, set()).add(_k)
    _SYNONYM_EXPAND_BASE.setdefault(_k, set()).add(_v)

# 运行时同义词表（含学习同义词），使用 frozenset 值防止并发修改
_SYNONYM_EXPAND: Dict[str, frozenset] = {
    k: frozenset(v) for k, v in _SYNONYM_EXPAND_BASE.items()
}


# ===== 学习同义词运行时状态 =====
_LEARNED_SYNONYM_DIRECT: Dict[str, str] = {}
_synonym_reload_lock = threading.Lock()
_learned_loaded: bool = False


def reload_learned_synonyms() -> int:
    """从 synonym_store 加载已审核的学习同义词到运行时扩展表。"""
    global _learned_loaded, _SYNONYM_EXPAND
    try:
        from synonym_store import get_all_learned
        learned = get_all_learned()
    except Exception:
        learned = []
    with _synonym_reload_lock:
        # 在新 dict 上构建，完成后原子替换引用
        new_direct: Dict[str, str] = {}
        new_expand: Dict[str, set] = {k: set(v) for k, v in _SYNONYM_EXPAND_BASE.items()}
        for entry in learned:
            if entry.get("approved"):
                orig = entry.get("original", "")
                mapped = entry.get("mapped_to", "")
                if orig and mapped:
                    new_direct[orig] = mapped
                    new_expand.setdefault(mapped, set()).add(orig)
                    new_expand.setdefault(orig, set()).add(mapped)
        # 冻结所有 set 值，防止并发迭代时被修改
        frozen = {k: frozenset(v) for k, v in new_expand.items()}
        # 原子替换
        _LEARNED_SYNONYM_DIRECT.clear()
        _LEARNED_SYNONYM_DIRECT.update(new_direct)
        _SYNONYM_EXPAND = frozen
        _learned_loaded = True
    return len(new_direct)


def expand_synonyms(query: str) -> str:
    """在查询中追加同义词，提升 BM25 召回率。"""
    extra = set()
    q_lower = query.lower()
    # 取所有 term 的 set 快照，避免迭代时被并发修改
    items = list(_SYNONYM_EXPAND.items())
    for term, synonyms in items:
        if term not in q_lower:
            continue
        # 短词（1-2字）需边界检查：前后字符不应与 term 组成其他同义词词条
        if len(term) <= 2:
            idx = q_lower.find(term)
            if idx > 0 and q_lower[idx - 1:idx + len(term)] in _SYNONYM_EXPAND:
                continue
            end = idx + len(term)
            if end < len(q_lower) and q_lower[idx:end + 1] in _SYNONYM_EXPAND:
                continue
        for syn in synonyms:
            if syn not in q_lower:
                extra.add(syn)
    if extra:
        expanded = query + " " + " ".join(sorted(extra))
        return expanded[:2000]
    return query


# ===== BM25 参数（从配置导入）=====


def _count_term(term: str, text: str) -> int:
    """统计 term 在 text 中的出现次数"""
    count = 0
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(term)
    return count


def keyword_score_bm25(query_or_terms, text: str, avg_dl: float, n_docs: int,
                       doc_freq: Dict[str, int],
                       k1: float = BM25_K1, b: float = BM25_B) -> float:
    """BM25 评分：考虑词频、文档长度和逆文档频率。
    query_or_terms: 预提取的查询词列表，或查询字符串（自动提取）。"""
    if isinstance(query_or_terms, str):
        q_tokens = list(dict.fromkeys(tokenize_chinese(query_or_terms.lower())))
    else:
        q_tokens = query_or_terms
    t_lower = (text or "").lower()
    dl = len(t_lower)
    if not q_tokens or dl == 0 or avg_dl == 0:
        return 0.0

    score = 0.0
    effective_n = max(n_docs, 100)
    for term in q_tokens:
        tf = _count_term(term, t_lower)
        if tf == 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log((effective_n - df + 0.5) / (df + 0.5) + 1.0)
        # 稀有佛教术语加权
        if term in _BUDDHIST_VOCAB and df < n_docs * 0.1:
            idf *= 1.3
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm

    return score


# ===== 语料缓存与倒排索引 =====
_CACHE_MAX_SIZE = CACHE_MAX_PRODUCTS
_bm25_cache: Dict[Any, Tuple[List[str], int, float]] = {}
_df_cache: Dict[Any, Dict[str, int]] = {}
_inverted_index_cache: Dict[Any, Dict[str, List[int]]] = {}
_bm25_cache_lock = threading.Lock()


def _cache_put(cache: dict, key: Any, value: Any, max_size: int = 0) -> None:
    limit = max_size if max_size > 0 else _CACHE_MAX_SIZE
    with _bm25_cache_lock:
        if key in cache:
            cache[key] = value
            return
        while len(cache) >= limit:
            try:
                oldest = next(iter(cache))
                cache.pop(oldest, None)
            except (StopIteration, RuntimeError):
                break
        cache[key] = value


def _cache_get(cache: dict, key: Any):
    with _bm25_cache_lock:
        return cache.get(key)


def invalidate_bm25_cache(product: str = "") -> None:
    """清除 BM25 语料缓存。若指定 product 则尝试按 key 匹配清除，否则清空全部。"""
    if not product:
        _bm25_cache.clear()
        _df_cache.clear()
        _inverted_index_cache.clear()
        return
    # product 维度的 key 由 _corpus_cache_key 生成，无法精确匹配，
    # 保守起见清空全部缓存以确保一致性
    _bm25_cache.clear()
    _df_cache.clear()
    _inverted_index_cache.clear()


def _corpus_cache_key(docs: List[Dict]) -> Tuple:
    n = len(docs)
    if n == 0:
        return (0,)
    # 采样更多文档片段并使用完整 SHA256 摘要，防止缓存碰撞
    parts = []
    sample_indices = {0, n - 1, n // 2, n // 4, n * 3 // 4}
    for i in sorted(sample_indices):
        if 0 <= i < n:
            parts.append((docs[i].get("text") or "")[:128])
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return (n, digest)


def _get_bm25_corpus(docs: List[Dict]) -> Tuple[List[str], int, float, Tuple]:
    key = _corpus_cache_key(docs)
    cached = _cache_get(_bm25_cache, key)
    if cached:
        return cached[0], cached[1], cached[2], key
    texts = [(d.get("text") or "").lower() for d in docs]
    n_docs = len(texts)
    avg_dl = sum(len(t) for t in texts) / max(n_docs, 1)
    _cache_put(_bm25_cache, key, (texts, n_docs, avg_dl))
    return texts, n_docs, avg_dl, key


def _batch_doc_freqs(terms: List[str], texts: List[str], corpus_key: Any) -> Dict[str, int]:
    cached = _cache_get(_df_cache, corpus_key)
    if cached is None:
        cached = {}
        _cache_put(_df_cache, corpus_key, cached)
    uncached = [t for t in terms if t not in cached]
    if uncached:
        counts = {t: 0 for t in uncached}
        for doc_text in texts:
            for t in uncached:
                if t in doc_text:
                    counts[t] += 1
        with _bm25_cache_lock:
            cached.update(counts)
    return {t: cached.get(t, 0) for t in terms}


def _get_inverted_index(texts: List[str], corpus_key: Any) -> Dict[str, List[int]]:
    cached = _cache_get(_inverted_index_cache, corpus_key)
    if cached is not None:
        return cached
    inv: Dict[str, List[int]] = {}
    for i, text in enumerate(texts):
        for j in range(len(text) - 1):
            bg = text[j:j+2]
            if bg not in inv:
                inv[bg] = [i]
            elif inv[bg][-1] != i:
                inv[bg].append(i)
    _cache_put(_inverted_index_cache, corpus_key, inv)
    return inv


def _get_candidate_docs(terms: List[str], inv_index: Dict[str, List[int]],
                         n_docs: int) -> List[int]:
    candidate_set: set = set()
    for term in terms:
        if len(term) >= 2:
            # 使用 term 的所有 bigram 组合查找候选文档，提升多字词召回率
            for j in range(len(term) - 1):
                bg = term[j:j+2]
                doc_ids = inv_index.get(bg)
                if doc_ids is not None:
                    candidate_set.update(doc_ids)
        elif len(term) == 1:
            for bg_key, doc_ids in inv_index.items():
                if term in bg_key:
                    candidate_set.update(doc_ids)
    return sorted(candidate_set)


def keyword_search(query: str, docs: List[Dict], top_k: int = 8,
                    skip_synonym_expand: bool = False) -> List[Dict]:
    """BM25 关键词搜索（带语料缓存、倒排索引加速、sigmoid 归一化）"""
    if not docs:
        return []

    if not skip_synonym_expand:
        query = expand_synonyms(query)
    q_tokens = list(dict.fromkeys(tokenize_chinese(query.lower())))
    if not q_tokens:
        return []

    texts, n_docs, avg_dl, corpus_key = _get_bm25_corpus(docs)
    doc_freqs = _batch_doc_freqs(q_tokens, texts, corpus_key)

    # 倒排索引加速
    inv_index = _get_inverted_index(texts, corpus_key)
    candidates = _get_candidate_docs(q_tokens, inv_index, n_docs)

    scored = []
    for i in candidates:
        s = keyword_score_bm25(q_tokens, texts[i], avg_dl, n_docs, doc_freqs)
        if s <= 0:
            continue
        scored.append({**docs[i], "keyword_score": s})

    scored.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)

    # sigmoid 归一化
    for x in scored:
        x["keyword_score"] = _sigmoid_norm(x["keyword_score"])

    return scored[:top_k]


def _hit_key(h: Dict) -> str:
    """生成检索结果的唯一键"""
    meta = h.get("meta", {})
    src = meta.get("source_file", "")
    cid = meta.get("chunk_id", "")
    if src and cid:
        return f"{src}#{cid}"
    return h.get("text", "")[:200]


def merge_hybrid(vector_hits: List[Dict], keyword_hits: List[Dict], vw: float, kw: float, top_k: int,
                  route: str = "") -> List[Dict]:
    merged = {}
    for h in vector_hits:
        key = _hit_key(h)
        vs = min(1.0, max(0.0, float(h.get("score", 0.0))))
        new_score = vs * vw
        if key in merged:
            if new_score > merged[key].get("hybrid_score", 0.0):
                merged[key] = {**h, "hybrid_score": new_score}
        else:
            merged[key] = {**h, "hybrid_score": new_score}
    for h in keyword_hits:
        key = _hit_key(h)
        if key not in merged:
            merged[key] = {**h, "score": 0.0, "hybrid_score": 0.0}
        kw_contribution = float(h.get("keyword_score", 0.0)) * kw
        current_kw = merged[key].get("_kw_contribution", 0.0)
        if kw_contribution > current_kw:
            merged[key]["hybrid_score"] += kw_contribution - current_kw
            merged[key]["_kw_contribution"] = kw_contribution

    # 路由感知加分
    if route:
        _apply_route_boost(merged, route)

    out = list(merged.values())
    for h in out:
        h.pop("_kw_contribution", None)
    out.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return out[:top_k]


# 佛教路由→章节标题关键词
_ROUTE_SECTION_MARKERS = {
    "doctrine":  ["教义", "四圣谛", "八正道", "十二因缘", "三法印", "缘起"],
    "practice":  ["修行方法", "禅修", "念佛", "持咒", "止观", "正念"],
    "scripture": ["经典", "心经", "金刚经", "法华经", "般若", "三藏"],
    "sect":      ["宗派", "禅宗", "净土宗", "天台宗", "华严宗", "藏传"],
    "concept":   ["概念", "空性", "涅槃", "轮回", "因果", "菩提心", "五蕴"],
    "history":   ["历史", "传入中国", "白马寺", "玄奘", "达摩", "慧能"],
    "ritual":    ["仪轨", "礼仪", "法会", "皈依", "上香", "供灯"],
    "basic":     ["基础", "入门", "概述", "释迦牟尼", "佛陀"],
}
def _apply_route_boost(merged: Dict[str, Dict], route: str) -> None:
    markers = _ROUTE_SECTION_MARKERS.get(route, [])
    if not markers:
        return
    from rag_runtime_config import ROUTE_BOOST as _live_boost
    for h in merged.values():
        text = (h.get("text") or "")[:800]
        if any(m in text for m in markers):
            h["hybrid_score"] += _live_boost


def _ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """快速计算两个短文本的 n-gram 重叠率"""
    if not a or not b or len(a) < n or len(b) < n:
        return 0.0
    sa = set(a[i:i+n] for i in range(len(a) - n + 1))
    sb = set(b[i:i+n] for i in range(len(b) - n + 1))
    return len(sa & sb) / min(len(sa), len(sb)) if sa and sb else 0.0


def split_multi_question(question: str, separators: List[str] = None) -> List[str]:
    separators = separators or ["？", "；", ";", "。", "，另外", "并且", "同时", "还有"]
    parts = [question]
    for sep in separators:
        next_parts = []
        for p in parts:
            next_parts.extend(p.split(sep))
        parts = next_parts
    parts = [p.strip() for p in parts if p.strip()]
    parts = uniq(parts)
    # 语义去重：重叠度 >75% 的子问题只保留较长的
    if len(parts) <= 1:
        return parts
    deduped = [parts[0]]
    for p in parts[1:]:
        is_dup = any(_ngram_overlap(p, d) > 0.75 for d in deduped)
        if not is_dup:
            deduped.append(p)
    return deduped


def detect_terms(question: str, term_map: Dict[str, List[str]]) -> List[str]:
    q = question.lower()
    found = []
    for key, aliases in term_map.items():
        for a in aliases:
            if a.lower() in q:
                found.append(key)
                break
    return uniq(found)


def _load_alias_map(alias_text: str) -> Dict[str, List[str]]:
    """从 alias.txt 构建 {别名: [主词, 其他别名...]} 的映射"""
    alias_map: Dict[str, List[str]] = {}
    if not alias_text:
        return alias_map
    for line in alias_text.strip().split("\n"):
        terms = line.strip().split()
        if len(terms) < 2:
            continue
        for t in terms:
            alias_map[t] = terms
    return alias_map


def _normalize_for_faq(text: str) -> str:
    """归一化文本用于 FAQ 匹配：去空格、统一标点"""
    t = (text or "").strip().lower()
    t = re.sub(r'\s+', '', t)  # 去掉所有空白
    # 统一常见标点
    t = t.replace('？', '').replace('?', '').replace('，', '').replace(',', '')
    t = t.replace('。', '').replace('.', '').replace('！', '').replace('!', '')
    return t


def match_faq(question: str, faq_text: str, faq_keyword_map: Dict[str, str],
              alias_text: str = "") -> str:
    """尝试匹配 FAQ 问答对，支持别名扩展和文本归一化"""
    if not faq_text:
        return ""
    q = (question or "").strip()
    q_norm = _normalize_for_faq(q)

    # 构建别名映射，将问题中的词扩展为所有别名
    alias_map = _load_alias_map(alias_text)
    expanded_keywords = set()
    for keyword in faq_keyword_map.keys():
        expanded_keywords.add(keyword)
        # 加入该关键词的所有别名
        if keyword in alias_map:
            for alias in alias_map[keyword]:
                expanded_keywords.add(alias)

    # 构建 {扩展关键词 -> 原始topic} 的完整映射
    expanded_map: Dict[str, str] = {}
    for keyword, topic in faq_keyword_map.items():
        expanded_map[keyword] = topic
        if keyword in alias_map:
            for alias in alias_map[keyword]:
                if alias not in expanded_map:
                    expanded_map[alias] = topic

    # 按长度降序匹配（长的优先，避免短子串误匹配）
    matched_topic = None
    q_lower = q.lower()
    for keyword, topic in sorted(expanded_map.items(), key=lambda x: len(x[0]), reverse=True):
        # 大小写不敏感匹配，同时检查原始问题和归一化后的问题
        kw_lower = keyword.lower()
        if kw_lower in q_lower or kw_lower in q_norm:
            matched_topic = topic
            break

    if not matched_topic:
        return ""

    # 在 FAQ 文本中查找对应的 Q&A 对
    # 同时用归一化后的 topic 进行匹配，提高容错
    topic_norm = _normalize_for_faq(matched_topic)
    lines = faq_text.split("\n")
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith("【Q】"):
            line_norm = _normalize_for_faq(line_s)
            if matched_topic in line_s or topic_norm in line_norm:
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
    # 配置-数据不一致：关键词匹配到 topic 但 FAQ 中无对应条目
    print(f"[WARN] FAQ 关键词映射到「{matched_topic}」但 faq.txt 中未找到对应【Q】条目", file=sys.stderr)
    return ""


# ===== Reranker：使用 BGE-M3 compute_score 对候选文档重排序 =====

_rerank_cache: Dict[str, List[Tuple[str, float]]] = {}
_RERANK_CACHE_MAX = 512


def _rerank_cache_key(query: str, hits: List[Dict]) -> str:
    hit_keys = "|".join(_hit_key(h) for h in hits)
    return hashlib.md5(f"{query}||{hit_keys}".encode()).hexdigest()


def rerank_hits(query: str, hits: List[Dict], model, top_k: int) -> List[Dict]:
    """使用 BGE-M3 的 compute_score 对混合检索结果重排序。"""
    if not hits or not model:
        return hits[:top_k]

    hits = [dict(h) for h in hits]

    cache_key = _rerank_cache_key(query, hits)
    cached = _cache_get(_rerank_cache, cache_key)
    if cached is not None:
        score_map = dict(cached)
        for h in hits:
            hk = _hit_key(h)
            if hk in score_map:
                h["rerank_score"] = score_map[hk]
                h["hybrid_score_before_rerank"] = h.get("hybrid_score", 0.0)
                h["hybrid_score"] = score_map[hk]
        hits.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return hits[:top_k]

    sentence_pairs = [[query, (h.get("text") or "")[:512]] for h in hits]
    try:
        scores = model.compute_score(sentence_pairs)
    except Exception as e:
        print(f"[WARN] rerank 失败，回退原排序: {e}")
        return hits[:top_k]

    if isinstance(scores, dict):
        final_scores = (
            scores.get("colbert+sparse+dense")
            or scores.get("score")
            or scores.get("dense")
        )
        if final_scores is None:
            col = scores.get("colbert", [0.0] * len(hits))
            spa = scores.get("sparse", [0.0] * len(hits))
            den = scores.get("dense", [0.0] * len(hits))
            final_scores = [0.4 * c + 0.2 * s + 0.4 * d
                           for c, s, d in zip(col, spa, den)]
    elif isinstance(scores, (list, tuple)):
        final_scores = scores
    else:
        return hits[:top_k]

    if final_scores:
        max_s = max(final_scores) if final_scores else 1.0
        min_s = min(final_scores) if final_scores else 0.0
        rng = max_s - min_s if max_s > min_s else 1.0
        cache_entries = []
        for i, h in enumerate(hits):
            raw = final_scores[i] if i < len(final_scores) else 0.0
            normalized = (raw - min_s) / rng
            h["rerank_score"] = normalized
            h["hybrid_score_before_rerank"] = h.get("hybrid_score", 0.0)
            # 混合原始 hybrid_score 与 rerank 分数，避免纯归一化丢失原始校准信息
            orig = h.get("hybrid_score", 0.0)
            h["hybrid_score"] = 0.3 * orig + 0.7 * normalized
            cache_entries.append((_hit_key(h), h["hybrid_score"]))
        _cache_put(_rerank_cache, cache_key, cache_entries, max_size=_RERANK_CACHE_MAX)

    hits.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return hits[:top_k]


# ===== 动态阈值 =====

def compute_dynamic_threshold(hits: List[Dict], route_threshold: float,
                               ratio: float = 0.40, floor_ratio: float = 0.70) -> float:
    """根据 top-1 分数动态计算阈值"""
    if not hits:
        return route_threshold
    top1 = max(h.get("hybrid_score", 0.0) for h in hits)
    dynamic = top1 * ratio
    floor = route_threshold * floor_ratio
    return max(floor, min(dynamic, route_threshold))
