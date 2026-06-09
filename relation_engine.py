"""佛教知识关联引擎：加载 relations.json 并提供跨实体关联查询。

用途：
1. doctrine 路由 → 补充教义之间的关系（四圣谛→八正道→十二因缘）
2. practice 路由 → 补充修行次第和方法关联
3. scripture 路由 → 补充经典与宗派、教义的关联
4. sect 路由 → 补充宗派之间的传承关系
5. concept 路由 → 补充概念之间的逻辑关系
"""
import json
import threading
from pathlib import Path
from typing import Any, List, Dict, Optional

from rag_runtime_config import RELATIONS_FILE, PRODUCT_ALIASES, PROJECT_ALIASES

_relations: Optional[Dict] = None
_lock = threading.Lock()

# 预建佛教主题别名倒排索引
_PROJECT_ALIAS_INV: Dict[str, str] = {}
for _pid, _aliases in PROJECT_ALIASES.items():
    for _a in _aliases:
        _PROJECT_ALIAS_INV[_a.lower()] = _pid

# 预建倒排索引
_idx_doctrine: Optional[Dict[str, List[Dict]]] = None    # 教义关联
_idx_practice: Optional[Dict[str, List[Dict]]] = None    # 修行关联
_idx_scripture: Optional[Dict[str, List[Dict]]] = None   # 经典关联
_idx_concept: Optional[Dict[str, List[Dict]]] = None     # 概念关联


def invalidate_relations_cache() -> None:
    """清除关联数据缓存（relations.json 更新后调用）"""
    global _relations, _idx_doctrine, _idx_practice, _idx_scripture, _idx_concept
    with _lock:
        _relations = None
        _idx_doctrine = None
        _idx_practice = None
        _idx_scripture = None
        _idx_concept = None


def _build_indices(data: Dict) -> None:
    """从 relations 数据构建倒排索引"""
    global _idx_doctrine, _idx_practice, _idx_scripture, _idx_concept

    # 教义关联
    idx_doc: Dict[str, List[Dict]] = {}
    for item in data.get("doctrine_relations", []):
        if not isinstance(item, dict):
            continue
        key = item.get("doctrine", "")
        if key:
            idx_doc.setdefault(key, []).append(item)
    _idx_doctrine = idx_doc

    # 修行关联
    idx_prac: Dict[str, List[Dict]] = {}
    for item in data.get("practice_relations", []):
        if not isinstance(item, dict):
            continue
        key = item.get("practice", "")
        if key:
            idx_prac.setdefault(key, []).append(item)
    _idx_practice = idx_prac

    # 经典关联
    idx_scr: Dict[str, List[Dict]] = {}
    for item in data.get("scripture_relations", []):
        if not isinstance(item, dict):
            continue
        key = item.get("scripture", "")
        if key:
            idx_scr.setdefault(key, []).append(item)
    _idx_scripture = idx_scr

    # 概念关联
    idx_con: Dict[str, List[Dict]] = {}
    for item in data.get("concept_relations", []):
        if not isinstance(item, dict):
            continue
        key = item.get("concept", "")
        if key:
            idx_con.setdefault(key, []).append(item)
    _idx_concept = idx_con


def _load() -> Dict:
    global _relations
    if _relations is not None:
        return _relations
    with _lock:
        if _relations is not None:
            return _relations
        if not RELATIONS_FILE.exists():
            _relations = {}
            return _relations
        try:
            data = json.loads(RELATIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            from rag_logger import log_error
            log_error("relation_engine", f"relations.json 加载失败: {exc}")
            _relations = {}
            return _relations
        _build_indices(data)
        _relations = data
    return _relations


def _product_label(pid: str) -> str:
    """知识领域 ID → 中文名"""
    aliases = PRODUCT_ALIASES.get(pid, [])
    return aliases[0] if aliases else pid


def _project_label(pid: str) -> str:
    """主题 ID → 中文名"""
    aliases = PROJECT_ALIASES.get(pid, [])
    return aliases[0] if aliases else pid


# ===== 公开查询接口 =====

def get_doctrine_relations(query: str, _data: Optional[Dict] = None) -> List[str]:
    """根据查询中的教义关键词，返回教义关联信息。"""
    if _data is None:
        _data = _load()
    lines = []
    q_lower = query.lower()
    for doctrine, items in (_idx_doctrine or {}).items():
        if doctrine in q_lower:
            for item in items:
                related = item.get("related", [])
                note = item.get("note", "")
                if related:
                    lines.append(f"【{doctrine}】相关教义：{'、'.join(related)}")
                if note:
                    lines.append(f"  {note}")
    return lines


def get_practice_relations(query: str, _data: Optional[Dict] = None) -> List[str]:
    """根据查询中的修行方法关键词，返回修行关联信息。"""
    if _data is None:
        _data = _load()
    lines = []
    q_lower = query.lower()
    for practice, items in (_idx_practice or {}).items():
        if practice in q_lower:
            for item in items:
                prerequisites = item.get("prerequisites", [])
                next_steps = item.get("next_steps", [])
                note = item.get("note", "")
                if prerequisites:
                    lines.append(f"【{practice}】前行基础：{'、'.join(prerequisites)}")
                if next_steps:
                    lines.append(f"【{practice}】进阶修行：{'、'.join(next_steps)}")
                if note:
                    lines.append(f"  {note}")
    return lines


def get_scripture_relations(query: str, _data: Optional[Dict] = None) -> List[str]:
    """根据查询中的经典关键词，返回经典关联信息。"""
    if _data is None:
        _data = _load()
    lines = []
    q_lower = query.lower()
    for scripture, items in (_idx_scripture or {}).items():
        if scripture in q_lower:
            for item in items:
                sects = item.get("sects", [])
                doctrines = item.get("doctrines", [])
                note = item.get("note", "")
                if sects:
                    lines.append(f"【{scripture}】相关宗派：{'、'.join(sects)}")
                if doctrines:
                    lines.append(f"【{scripture}】核心教义：{'、'.join(doctrines)}")
                if note:
                    lines.append(f"  {note}")
    return lines


def get_concept_relations(query: str, _data: Optional[Dict] = None) -> List[str]:
    """根据查询中的概念关键词，返回概念关联信息。"""
    if _data is None:
        _data = _load()
    lines = []
    q_lower = query.lower()
    for concept, items in (_idx_concept or {}).items():
        if concept in q_lower:
            for item in items:
                related = item.get("related", [])
                opposite = item.get("opposite", "")
                note = item.get("note", "")
                if related:
                    lines.append(f"【{concept}】关联概念：{'、'.join(related)}")
                if opposite:
                    lines.append(f"【{concept}】对治法门：{opposite}")
                if note:
                    lines.append(f"  {note}")
    return lines


def _enrich_doctrine(product_id: str, question: str, data: Dict) -> List[str]:
    return get_doctrine_relations(question, _data=data)


def _enrich_practice(product_id: str, question: str, data: Dict) -> List[str]:
    return get_practice_relations(question, _data=data)


def _enrich_scripture(product_id: str, question: str, data: Dict) -> List[str]:
    return get_scripture_relations(question, _data=data)


def _enrich_concept(product_id: str, question: str, data: Dict) -> List[str]:
    return get_concept_relations(question, _data=data)


def _enrich_sect(product_id: str, question: str, data: Dict) -> List[str]:
    # 宗派关联可从经典关联和教义关联中综合获取
    lines = get_scripture_relations(question, _data=data)
    lines.extend(get_doctrine_relations(question, _data=data))
    return lines


# 路由→enricher 映射
_ENRICH_DISPATCH: Dict[str, Any] = {
    "doctrine": _enrich_doctrine,
    "practice": _enrich_practice,
    "scripture": _enrich_scripture,
    "concept": _enrich_concept,
    "sect": _enrich_sect,
}


def enrich_answer(route: str, product_id: str, question: str) -> List[str]:
    """根据路由类型，从 relations.json 中提取补充信息。"""
    handler = _ENRICH_DISPATCH.get(route)
    if handler is None:
        return []
    data = _load()
    if not data:
        return []
    return handler(product_id, question, data)
