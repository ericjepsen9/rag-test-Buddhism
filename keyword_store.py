"""统一关键词存储：管理静态同义词的覆盖、消歧规则的持久化、LLM 扩充记录。

文件格式：JSON，存储在 data/keyword_overrides.json
结构：
{
  "synonym_overrides": {
    "新口语词": {"mapped_to": "规范形式", "action": "add|edit|delete", "ts": "..."}
  },
  "clarification_rules": {
    "触发词": {
      "options": [{"label": "...", "query": "...", "route": "..."}],
      "ts": "..."
    }
  },
  "llm_expansion_log": [
    {"ts": "...", "category": "synonym|clarification", "count": 5, "items": [...]}
  ]
}

线程安全：读写均加锁。
"""

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 输入清理：去除控制字符，防止注入
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(text: str, max_len: int = 500) -> str:
    """清理输入文本：去除控制字符并截断"""
    return _CONTROL_CHAR_RE.sub("", text.strip())[:max_len]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OVERRIDES_FILE = DATA_DIR / "keyword_overrides.json"

_lock = threading.Lock()


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> Dict[str, Any]:
    if not OVERRIDES_FILE.exists():
        return {"synonym_overrides": {}, "clarification_rules": {}, "llm_expansion_log": []}
    try:
        with OVERRIDES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # 确保结构完整
        data.setdefault("synonym_overrides", {})
        data.setdefault("clarification_rules", {})
        data.setdefault("llm_expansion_log", [])
        return data
    except (json.JSONDecodeError, OSError):
        return {"synonym_overrides": {}, "clarification_rules": {}, "llm_expansion_log": []}


def _save(data: Dict[str, Any]) -> None:
    _ensure_dir()
    tmp = OVERRIDES_FILE.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(OVERRIDES_FILE)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        from rag_logger import log_error
        log_error("keyword_store", f"保存覆盖文件失败: {e}",
                  meta={"path": str(OVERRIDES_FILE)})
        raise


# ============================================================
# 静态同义词覆盖（可在后台添加/编辑/删除静态同义词，无需重启）
# ============================================================

def get_synonym_overrides() -> Dict[str, Dict]:
    """返回所有静态同义词覆盖"""
    with _lock:
        return dict(_load()["synonym_overrides"])


def add_synonym_override(original: str, mapped_to: str) -> Dict[str, Any]:
    """添加或更新一条静态同义词覆盖"""
    original = _sanitize(original, max_len=200)
    mapped_to = _sanitize(mapped_to, max_len=200)
    if not original or not mapped_to:
        return {"ok": False, "error": "原始词和映射词不能为空"}
    if original == mapped_to:
        return {"ok": False, "error": "原始词和映射词不能相同"}

    with _lock:
        data = _load()
        data["synonym_overrides"][original] = {
            "mapped_to": mapped_to,
            "action": "add",
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        _save(data)
    return {"ok": True}


def delete_synonym_override(original: str) -> Dict[str, Any]:
    """删除一条静态同义词覆盖（标记为已删除，阻止静态表中的该条生效）"""
    original = original.strip()
    if not original:
        return {"ok": False, "error": "原始词不能为空"}
    with _lock:
        data = _load()
        if original in data["synonym_overrides"]:
            del data["synonym_overrides"][original]
        else:
            # 标记静态表中的词为"删除"
            data["synonym_overrides"][original] = {
                "mapped_to": "",
                "action": "delete",
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        _save(data)
    return {"ok": True}


def get_effective_synonyms() -> List[Dict[str, str]]:
    """获取最终生效的同义词列表（静态 + 覆盖合并后的结果）。

    返回: [{"original": "...", "mapped_to": "...", "source": "static|override|learned"}, ...]
    """
    from search_utils import _SYNONYM_MAP
    from synonym_store import get_all_learned

    with _lock:
        data = _load()
    overrides = data["synonym_overrides"]

    # 从静态表开始
    result = {}
    for k, v in _SYNONYM_MAP.items():
        result[k] = {"original": k, "mapped_to": v, "source": "static"}

    # 应用覆盖
    for k, ov in overrides.items():
        if ov.get("action") == "delete":
            result.pop(k, None)
        else:
            result[k] = {"original": k, "mapped_to": ov["mapped_to"], "source": "override"}

    # 合并已学习的词
    for item in get_all_learned():
        if item.get("approved") and item["original"] not in result:
            result[item["original"]] = {
                "original": item["original"],
                "mapped_to": item["mapped_to"],
                "source": "learned",
            }

    return sorted(result.values(), key=lambda x: x["original"])


# ============================================================
# 消歧规则管理
# ============================================================

def get_clarification_rules() -> Dict[str, Any]:
    """返回所有自定义消歧规则"""
    with _lock:
        return dict(_load()["clarification_rules"])


def save_clarification_rule(trigger: str, options: List[Dict[str, str]]) -> Dict[str, Any]:
    """保存一条消歧规则"""
    trigger = _sanitize(trigger, max_len=200)
    if not trigger:
        return {"ok": False, "error": "触发词不能为空"}
    if not options or len(options) < 1:
        return {"ok": False, "error": "至少需要一个选项"}
    # 校验选项格式并清理输入
    for opt in options:
        if not opt.get("label") or not opt.get("query"):
            return {"ok": False, "error": "每个选项需要 label 和 query"}
        opt["label"] = _sanitize(opt["label"], max_len=200)
        opt["query"] = _sanitize(opt["query"], max_len=500)

    with _lock:
        data = _load()
        data["clarification_rules"][trigger] = {
            "options": options,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        _save(data)
    return {"ok": True}


def delete_clarification_rule(trigger: str) -> Dict[str, Any]:
    """删除一条消歧规则"""
    trigger = trigger.strip()
    with _lock:
        data = _load()
        if trigger not in data["clarification_rules"]:
            return {"ok": False, "error": f"规则「{trigger}」不存在"}
        del data["clarification_rules"][trigger]
        _save(data)
    return {"ok": True}


# ============================================================
# LLM 扩充
# ============================================================

def save_llm_expansion_log(category: str, items: List[Dict]) -> None:
    """记录一次 LLM 扩充的结果"""
    with _lock:
        data = _load()
        data["llm_expansion_log"].append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "category": category,
            "count": len(items),
            "items": items[:50],  # 限制日志大小
        })
        # 只保留最近 50 条日志
        data["llm_expansion_log"] = data["llm_expansion_log"][-50:]
        _save(data)


def get_llm_expansion_log(limit: int = 20) -> List[Dict]:
    """获取最近的 LLM 扩充记录"""
    with _lock:
        data = _load()
    logs = data.get("llm_expansion_log", [])
    return list(reversed(logs[-limit:]))


def llm_expand_synonyms(category: str = "synonym", count: int = 20) -> Dict[str, Any]:
    """调用 LLM 基于已有词库生成新的同义词/消歧规则。

    参数:
        category: "synonym" 生成同义词, "clarification" 生成消歧规则
        count: 期望生成的词条数

    返回:
        {"ok": True, "items": [...], "applied": N} 或 {"ok": False, "error": "..."}
    """
    from rag_runtime_config import USE_OPENAI, OPENAI_MODEL

    if not USE_OPENAI:
        return {"ok": False, "error": "LLM 未启用，请先在模型切换中配置 API"}

    # 获取 LLM client
    client = None
    model = OPENAI_MODEL
    try:
        from llm_client import get_client, get_model, is_enabled
        if is_enabled("chat"):
            client = get_client("chat")
            model = get_model("chat") or OPENAI_MODEL
    except ImportError:
        pass
    if client is None:
        try:
            from rag_answer import _get_openai_client
            client = _get_openai_client()
        except Exception:
            pass
    if client is None:
        return {"ok": False, "error": "LLM client 不可用，请检查 API 配置"}

    if category == "synonym":
        return _llm_expand_synonyms_impl(client, model, count)
    elif category == "clarification":
        return _llm_expand_clarification_impl(client, model, count)
    else:
        return {"ok": False, "error": f"未知的扩充类型: {category}"}


def _llm_expand_synonyms_impl(client, model: str, count: int) -> Dict[str, Any]:
    """让 LLM 基于已有同义词表生成新的映射"""
    from search_utils import _SYNONYM_MAP

    # 取样已有词库作为示例（避免 prompt 太长）
    examples = []
    for k, v in list(_SYNONYM_MAP.items())[:30]:
        examples.append(f"「{k}」→「{v}」")
    example_text = "、".join(examples)

    prompt = f"""你是佛学知识库的词汇扩充助手。以下是已有的同义词映射示例：

{example_text}

请基于佛学领域知识，生成 {count} 条**新的**同义词映射（不要重复已有的）。
重点覆盖：
1. 通俗口语化表达（如"看破红尘"→"出离心"）
2. 经典/概念的常见异译或简称
3. 梵文/巴利文音译的不同版本
4. 网络流行用语中的佛学相关表达（如"佛系"→"随缘"）
5. 日常用语中的佛教来源词汇

输出格式（严格 JSON 数组）：
[{{"original": "口语/变体", "mapped_to": "规范术语"}}, ...]

只输出 JSON，不要其他文字。"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        raw = (resp.choices[0].message.content or "").strip()
        import re
        m = re.search(r'\[[\s\S]*\]', raw)
        if not m:
            return {"ok": False, "error": "LLM 返回格式异常，无法解析"}
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return {"ok": False, "error": "LLM 返回的 JSON 数组解析失败"}
        if not isinstance(items, list):
            return {"ok": False, "error": "LLM 返回格式异常：期望 JSON 数组"}

        # 过滤已存在的词
        existing = set(_SYNONYM_MAP.keys())
        new_items = [it for it in items
                     if it.get("original") and it.get("mapped_to")
                     and it["original"].strip() not in existing
                     and it["original"].strip() != it["mapped_to"].strip()]

        # 自动应用到学习词库（标记为 LLM 扩充，待审核）
        applied = 0
        from synonym_store import save_learned
        for it in new_items:
            try:
                save_learned(it["original"].strip(), it["mapped_to"].strip())
                applied += 1
            except Exception:
                pass

        save_llm_expansion_log("synonym", new_items)
        return {"ok": True, "items": new_items, "applied": applied, "total_generated": len(items)}
    except json.JSONDecodeError:
        return {"ok": False, "error": "LLM 返回的 JSON 解析失败"}
    except Exception as e:
        return {"ok": False, "error": f"LLM 调用失败: {str(e)}"}


def _llm_expand_clarification_impl(client, model: str, count: int) -> Dict[str, Any]:
    """让 LLM 基于已有消歧规则生成新的消歧触发词和选项"""
    from clarification_engine import _CLARIFICATION_RULES

    # 取样已有规则
    examples = []
    for trigger, options in list(_CLARIFICATION_RULES.items())[:5]:
        opts_str = " | ".join(o["label"] for o in options)
        examples.append(f"触发词「{trigger}」→ 选项: {opts_str}")
    example_text = "\n".join(examples)

    prompt = f"""你是佛学知识库的消歧引导扩充助手。当用户输入模糊查询时，系统会提供候选选项帮助用户精确提问。

已有的消歧规则示例：
{example_text}

请生成 {count} 条**新的**消歧规则（不要重复已有的触发词）。
每条规则包含一个触发词和 2-3 个候选选项。
重点覆盖佛学领域中容易产生歧义的口语查询。

输出格式（严格 JSON 数组）：
[{{
  "trigger": "触发词",
  "options": [
    {{"label": "选项描述", "query": "精确化的查询语句", "route": "路由名"}},
    ...
  ]
}}, ...]

可用的路由名: basic, concept, practice, scripture, school, precept, effect, history, person, ritual, combo, comparison

只输出 JSON，不要其他文字。"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000,
        )
        raw = (resp.choices[0].message.content or "").strip()
        import re
        m = re.search(r'\[[\s\S]*\]', raw)
        if not m:
            return {"ok": False, "error": "LLM 返回格式异常，无法解析"}
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return {"ok": False, "error": "LLM 返回的 JSON 数组解析失败"}
        if not isinstance(items, list):
            return {"ok": False, "error": "LLM 返回格式异常：期望 JSON 数组"}

        # 过滤已存在的触发词
        existing = set(_CLARIFICATION_RULES.keys())
        new_items = [it for it in items
                     if it.get("trigger") and it.get("options")
                     and it["trigger"].strip() not in existing]

        # 保存到自定义消歧规则
        applied = 0
        for it in new_items:
            result = save_clarification_rule(it["trigger"].strip(), it["options"])
            if result.get("ok"):
                applied += 1

        save_llm_expansion_log("clarification", new_items)
        return {"ok": True, "items": new_items, "applied": applied, "total_generated": len(items)}
    except json.JSONDecodeError:
        return {"ok": False, "error": "LLM 返回的 JSON 解析失败"}
    except Exception as e:
        return {"ok": False, "error": f"LLM 调用失败: {str(e)}"}
