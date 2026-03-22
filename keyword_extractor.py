"""导入时 LLM 辅助关键词提取：从原始文档中提取同义词、自定义分词、路由关键词。

在 import_knowledge.py 导入文档时调用，一次性从原始文档中提取三类关键词：
1. 同义词映射：口语/变体 → 规范术语（写入 learned_synonyms）
2. jieba 自定义词：领域专有名词、多字词（写入 data/jieba_user_dict.txt）
3. 路由关键词：与问题路由相关的触发词（写入 data/extracted_route_keywords.json）

设计原则：
- 导入时一次提取，避免运行时重复调用 LLM
- 提取结果分层存储，支持人工审核
- 增量追加，不覆盖已有词库
- 去重：与已有静态词库对比，只保留新词
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 提取结果的持久化文件
EXTRACTED_KEYWORDS_FILE = DATA_DIR / "extracted_keywords.json"
JIEBA_USER_DICT_FILE = DATA_DIR / "jieba_user_dict.txt"
ROUTE_KEYWORDS_FILE = DATA_DIR / "extracted_route_keywords.json"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LLM 提取 Prompt
# ============================================================

_SYSTEM_EXTRACT_KEYWORDS = """你是佛学知识库的词汇提取专家。你的任务是从一份佛学领域文档中提取三类关键词信息，
用于改善知识库的搜索和检索质量。

你需要输出一个严格的 JSON 对象，包含以下三个字段：

{
  "synonyms": [
    {"original": "口语/俗称/变体", "mapped_to": "文档中使用的规范术语", "context": "简短说明使用场景"}
  ],
  "jieba_words": [
    {"word": "专有名词", "freq": 5, "pos": "n", "reason": "为什么需要作为自定义词"}
  ],
  "route_keywords": [
    {"keyword": "关键词", "routes": ["匹配的路由"], "reason": "为什么关联到该路由"}
  ]
}

## synonyms（同义词映射）提取规则：
重点提取以下类型：
1. **通俗说法 → 规范术语**：如"看破红尘"→"出离心"
2. **经典/概念的别名、异译**：如 般若 → 智慧，涅槃 → 圆寂
3. **口语化描述**：如"成佛"→"证得佛果"
4. **宗派/人物的俗称**：如"六祖"→"慧能大师"
5. **梵文/巴利文 → 中文术语**：如 Dharma → 法，Karma → 业
6. **不同翻译版本的对照**：如"观自在菩萨"→"观世音菩萨"
7. **简称与全称**：如"四谛"→"四圣谛"
不要提取太泛化的映射（如"好"→"善"），关注佛学领域特定的表达。
目标：15-40 条高质量映射。

## jieba_words（分词自定义词）提取规则：
提取文档中出现的、标准分词器可能切错的领域术语：
1. **佛教专有名词**：如"阿弥陀佛"（会被切成"阿/弥/陀/佛"）
2. **梵文音译词**：如"般若波罗蜜多"、"菩提萨埵"
3. **经典名称**：如"金刚经"、"楞严经"、"法华经"
4. **宗派名称**：如"唯识宗"、"天台宗"、"华严宗"
5. **佛学术语**：如"缘起性空"、"四圣谛"、"八正道"
freq 建议：常见词=10，普通术语=5，罕见术语=3
pos 使用 jieba 词性标注：n=名词, v=动词, nr=人名, nz=专有名词, eng=英文
目标：10-30 个需要保护的词。

## route_keywords（路由关键词）提取规则：
从文档内容中提取可用于问题路由判断的关键词。可用路由包括：
- basic: 基本信息介绍
- concept: 教义概念解释
- practice: 修行方法指导
- scripture: 经典文献
- school: 宗派介绍
- precept: 戒律相关
- effect: 功德利益
- history: 佛教历史
- person: 人物介绍
- ritual: 仪轨礼仪
- combo: 综合修行方案
- comparison: 对比分析
每个关键词可以关联 1-2 个最相关的路由。
目标：10-25 个关键词。

## 注意事项：
- 只基于文档内容提取，不要凭空编造
- 优先提取文档中明确出现的术语和表达
- 口语变体可以基于文档内容合理推断（标注 context 说明）
- 只输出 JSON，不要其他文字"""


def extract_keywords_from_document(
    client,
    model: str,
    raw_text: str,
    entity_type: str,
    entity_id: str = "",
    existing_synonyms: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """从原始文档中提取关键词。

    Args:
        client: OpenAI 兼容客户端
        model: 模型名称
        raw_text: 原始文档文本
        entity_type: 实体类型 (scripture/concept/...)
        entity_id: 实体 ID
        existing_synonyms: 已有同义词映射（用于去重）

    Returns:
        {
            "synonyms": [...],
            "jieba_words": [...],
            "route_keywords": [...],
            "entity_type": "...",
            "entity_id": "...",
            "extracted_at": "..."
        }
    """
    # 截断过长文档（保留前 8000 字 + 后 2000 字，避免 token 超限）
    if len(raw_text) > 10000:
        truncated = raw_text[:8000] + "\n\n...(中间省略)...\n\n" + raw_text[-2000:]
    else:
        truncated = raw_text

    # 构建已有同义词的去重提示
    dedup_hint = ""
    if existing_synonyms:
        # 取样部分已有映射，帮助 LLM 避免重复
        sample = list(existing_synonyms.items())[:50]
        dedup_lines = [f"「{k}」→「{v}」" for k, v in sample]
        dedup_hint = (
            "\n\n【已有同义词（不要重复这些）】：\n"
            + "、".join(dedup_lines)
        )

    # 清理 entity_id 中的特殊字符，防止破坏 prompt 结构
    _safe_entity = (entity_id or entity_type).replace("\n", " ").replace("\r", " ").strip()
    user_prompt = (
        f"以下是关于「{_safe_entity}」（类型：{entity_type}）的佛学文档。"
        f"请提取关键词信息。\n\n"
        f"【文档内容】\n{truncated}"
        f"{dedup_hint}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_EXTRACT_KEYWORDS},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        raw_result = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        from rag_logger import log_error
        log_error("keyword_extract", f"LLM 调用失败: {e}",
                  meta={"type": entity_type, "id": entity_id})
        return _empty_result(entity_type, entity_id)

    # 解析 JSON
    result = _parse_json_result(raw_result)
    if result is None:
        from rag_logger import log_error
        log_error("keyword_extract", "LLM 结果解析失败",
                  meta={"type": entity_type, "id": entity_id, "raw": raw_result[:200]})
        return _empty_result(entity_type, entity_id)

    # 标准化和验证
    result = _validate_and_clean(result, existing_synonyms or {})
    result["entity_type"] = entity_type
    result["entity_id"] = entity_id
    result["extracted_at"] = datetime.now().isoformat(timespec="seconds")

    return result


def _empty_result(entity_type: str, entity_id: str) -> Dict[str, Any]:
    return {
        "synonyms": [],
        "jieba_words": [],
        "route_keywords": [],
        "entity_type": entity_type,
        "entity_id": entity_id,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


def _parse_json_result(text: str) -> Optional[Dict]:
    """解析 LLM 返回的 JSON（兼容 markdown 代码块包裹）"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', cleaned)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return None


def _validate_and_clean(
    result: Dict, existing_synonyms: Dict[str, str]
) -> Dict[str, Any]:
    """验证和清洗提取结果"""
    # 清洗 synonyms
    clean_synonyms = []
    existing_keys = set(existing_synonyms.keys())
    seen = set()
    for item in result.get("synonyms", []):
        orig = (item.get("original") or "").strip()
        mapped = (item.get("mapped_to") or "").strip()
        if not orig or not mapped or orig == mapped:
            continue
        if orig in existing_keys or orig in seen:
            continue
        seen.add(orig)
        clean_synonyms.append({
            "original": orig,
            "mapped_to": mapped,
            "context": (item.get("context") or "").strip(),
        })

    # 清洗 jieba_words
    clean_jieba = []
    seen_words = set()
    for item in result.get("jieba_words", []):
        word = (item.get("word") or "").strip()
        if not word or len(word) < 2 or word in seen_words:
            continue
        seen_words.add(word)
        freq = item.get("freq", 5)
        if not isinstance(freq, int) or freq < 1:
            freq = 5
        pos = (item.get("pos") or "n").strip()
        clean_jieba.append({
            "word": word,
            "freq": freq,
            "pos": pos,
            "reason": (item.get("reason") or "").strip(),
        })

    # 清洗 route_keywords
    valid_routes = {
        "basic", "concept", "practice", "scripture", "school",
        "precept", "effect", "history", "person", "ritual",
        "combo", "comparison", "indication_q", "procedure_q",
    }
    clean_routes = []
    for item in result.get("route_keywords", []):
        kw = (item.get("keyword") or "").strip()
        routes = item.get("routes", [])
        if not kw or not routes:
            continue
        # 过滤无效路由
        routes = [r for r in routes if r in valid_routes]
        if not routes:
            continue
        clean_routes.append({
            "keyword": kw,
            "routes": routes,
            "reason": (item.get("reason") or "").strip(),
        })

    return {
        "synonyms": clean_synonyms,
        "jieba_words": clean_jieba,
        "route_keywords": clean_routes,
    }


# ============================================================
# 提取结果的持久化和应用
# ============================================================

def save_extraction_result(result: Dict[str, Any]) -> Dict[str, int]:
    """保存提取结果到各个存储。

    Returns:
        {"synonyms_added": N, "jieba_words_added": N, "route_keywords_added": N}
    """
    _ensure_dir()
    stats = {"synonyms_added": 0, "jieba_words_added": 0, "route_keywords_added": 0}

    # 1. 同义词 → synonym_store（标记 source=import_extract，默认待审核）
    stats["synonyms_added"] = _save_synonyms(result.get("synonyms", []))

    # 2. jieba 自定义词 → data/jieba_user_dict.txt（追加）
    stats["jieba_words_added"] = _save_jieba_words(result.get("jieba_words", []))

    # 3. 路由关键词 → data/extracted_route_keywords.json（合并）
    stats["route_keywords_added"] = _save_route_keywords(
        result.get("route_keywords", []),
        result.get("entity_type", ""),
        result.get("entity_id", ""),
    )

    # 4. 保存完整提取记录（用于审计和回溯）
    _append_extraction_log(result)

    return stats


def _save_synonyms(synonyms: List[Dict]) -> int:
    """将同义词保存到 learned_synonyms 存储"""
    if not synonyms:
        return 0
    try:
        from synonym_store import add_manual, _load as _load_learned
        existing = _load_learned()
    except ImportError:
        return 0

    added = 0
    for item in synonyms:
        orig = item["original"]
        mapped = item["mapped_to"]
        if orig in existing:
            continue
        # 使用 add_manual 写入，但修改 source 标记
        result = add_manual(orig, mapped)
        if result.get("ok"):
            # 修改 source 标记为 import_extract（区分手动添加和导入提取）
            from synonym_store import _load, _save, _lock
            import threading
            with _lock:
                data = _load()
                if orig in data:
                    data[orig]["source"] = "import_extract"
                    data[orig]["approved"] = False  # 导入提取的默认待审核
                    context = item.get("context", "")
                    if context:
                        data[orig]["context"] = context
                    _save(data)
            added += 1

    return added


def _save_jieba_words(words: List[Dict]) -> int:
    """将 jieba 自定义词追加到 data/jieba_user_dict.txt"""
    if not words:
        return 0

    _ensure_dir()

    # 读取已有的自定义词（容错处理损坏文件）
    existing_words = set()
    if JIEBA_USER_DICT_FILE.exists():
        try:
            for line in JIEBA_USER_DICT_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if parts:
                        existing_words.add(parts[0])
        except (OSError, UnicodeDecodeError) as e:
            from rag_logger import log_error
            log_error("keyword_extractor", f"jieba 用户词典读取失败，将重建: {e}")

    # 也读取 search_utils 中的硬编码自定义词
    try:
        existing_words.update(_get_hardcoded_jieba_words())
    except Exception:
        pass

    # 追加新词
    new_lines = []
    for item in words:
        word = item["word"]
        if word in existing_words:
            continue
        freq = item.get("freq", 5)
        pos = item.get("pos", "n")
        # jieba 用户词典格式: 词 频率 词性
        new_lines.append(f"{word} {freq} {pos}")
        existing_words.add(word)

    if new_lines:
        # 原子写入：读取现有内容 + 追加新词 → 写临时文件 → 替换
        existing_content = ""
        if JIEBA_USER_DICT_FILE.exists():
            try:
                existing_content = JIEBA_USER_DICT_FILE.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass
        if existing_content and not existing_content.endswith("\n"):
            existing_content += "\n"
        tmp = JIEBA_USER_DICT_FILE.with_suffix(".tmp")
        tmp.write_text(existing_content + "\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(JIEBA_USER_DICT_FILE)

    return len(new_lines)


def _get_hardcoded_jieba_words() -> set:
    """获取 search_utils.py 中硬编码的 _CUSTOM_WORDS"""
    hardcoded = {
        "阿弥陀佛", "释迦牟尼", "观世音菩萨", "地藏菩萨", "文殊菩萨",
        "普贤菩萨", "大势至菩萨", "弥勒菩萨",
        "般若波罗蜜多", "菩提萨埵", "阿耨多罗三藐三菩提",
        "四圣谛", "八正道", "十二因缘", "六度万行", "三法印",
        "金刚经", "心经", "法华经", "楞严经", "华严经", "阿弥陀经",
        "无量寿经", "地藏经", "楞伽经", "维摩诘经", "圆觉经",
        "大悲咒", "六字真言", "往生咒", "准提咒",
        "禅宗", "净土宗", "密宗", "天台宗", "华严宗", "唯识宗",
        "律宗", "三论宗", "法相宗",
        "缘起性空", "诸行无常", "诸法无我", "涅槃寂静",
        "菩提心", "出离心", "慈悲心", "五蕴皆空",
        "四念处", "五戒十善", "八关斋戒", "菩萨戒",
        "因果报应", "六道轮回", "三界", "五蕴",
        "色即是空", "应无所住", "明心见性",
    }
    return hardcoded


def _save_route_keywords(
    keywords: List[Dict], entity_type: str, entity_id: str
) -> int:
    """将路由关键词合并到 data/extracted_route_keywords.json"""
    if not keywords:
        return 0

    _ensure_dir()

    # 加载已有
    existing = {}
    if ROUTE_KEYWORDS_FILE.exists():
        try:
            existing = json.loads(ROUTE_KEYWORDS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    # 按路由名分组合并
    added = 0
    for item in keywords:
        kw = item["keyword"]
        for route in item["routes"]:
            if route not in existing:
                existing[route] = {}
            if kw not in existing[route]:
                existing[route][kw] = {
                    "source_entities": [],
                    "reason": item.get("reason", ""),
                }
                added += 1
            # 追加来源实体
            source_key = f"{entity_type}:{entity_id}" if entity_id else entity_type
            if source_key not in existing[route][kw]["source_entities"]:
                existing[route][kw]["source_entities"].append(source_key)

    # 原子写入
    tmp = ROUTE_KEYWORDS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    tmp.replace(ROUTE_KEYWORDS_FILE)

    return added


def _append_extraction_log(result: Dict[str, Any]) -> None:
    """追加完整提取记录到日志文件"""
    if EXTRACTED_KEYWORDS_FILE.exists():
        try:
            records = json.loads(EXTRACTED_KEYWORDS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            records = []
    else:
        records = []

    if not isinstance(records, list):
        records = []

    records.append(result)

    # 限制日志大小（保留最近 100 条）
    records = records[-100:]

    tmp = EXTRACTED_KEYWORDS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(EXTRACTED_KEYWORDS_FILE)


# ============================================================
# jieba 自定义词典加载（供 search_utils 调用）
# ============================================================

def load_jieba_user_dict(jieba_mod) -> int:
    """加载 data/jieba_user_dict.txt 中的自定义词到 jieba。

    Args:
        jieba_mod: jieba 模块实例

    Returns:
        加载的词条数
    """
    if not JIEBA_USER_DICT_FILE.exists():
        return 0

    count = 0
    for line in JIEBA_USER_DICT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        word = parts[0]
        try:
            freq = int(parts[1]) if len(parts) > 1 else 5
        except (ValueError, TypeError):
            freq = 5
        jieba_mod.add_word(word, freq=freq)
        count += 1

    return count


# ============================================================
# 路由关键词加载（供 rag_answer.py 调用）
# ============================================================

def load_route_keywords() -> Dict[str, List[str]]:
    """加载提取的路由关键词，返回 {route: [keyword, ...]} 格式。

    供 detect_route 合并使用。
    """
    if not ROUTE_KEYWORDS_FILE.exists():
        return {}

    try:
        data = json.loads(ROUTE_KEYWORDS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    result = {}
    for route, kw_dict in data.items():
        result[route] = list(kw_dict.keys())

    return result


# ============================================================
# 统计和审查工具
# ============================================================

def get_extraction_stats() -> Dict[str, Any]:
    """获取提取关键词的统计信息"""
    stats = {
        "jieba_user_dict_count": 0,
        "route_keywords_count": 0,
        "extraction_log_count": 0,
    }

    if JIEBA_USER_DICT_FILE.exists():
        lines = [l for l in JIEBA_USER_DICT_FILE.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        stats["jieba_user_dict_count"] = len(lines)

    if ROUTE_KEYWORDS_FILE.exists():
        try:
            data = json.loads(ROUTE_KEYWORDS_FILE.read_text(encoding="utf-8"))
            total = sum(len(v) for v in data.values())
            stats["route_keywords_count"] = total
            stats["route_keywords_by_route"] = {k: len(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            pass

    if EXTRACTED_KEYWORDS_FILE.exists():
        try:
            records = json.loads(EXTRACTED_KEYWORDS_FILE.read_text(encoding="utf-8"))
            stats["extraction_log_count"] = len(records)
        except (json.JSONDecodeError, OSError):
            pass

    return stats


def get_pending_review() -> Dict[str, Any]:
    """获取待审核的提取关键词"""
    from synonym_store import get_all_learned

    pending_synonyms = [
        item for item in get_all_learned()
        if item.get("source") == "import_extract" and not item.get("approved")
    ]

    return {
        "pending_synonyms": pending_synonyms,
        "pending_count": len(pending_synonyms),
    }
