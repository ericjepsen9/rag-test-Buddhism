"""
多 LLM 提供商客户端管理器。

支持为「对话」和「知识库整理」分别配置不同的 LLM 提供商 / 模型 / API Key。
所有 LLM 调用统一通过本模块获取 client，不再直接使用 rag_runtime_config 中的全局变量。
"""
import os
import threading
from typing import Optional, Dict, Any

_lock = threading.Lock()

# 两个用途的独立配置
# "chat"      — 用户对话（rag_answer, query_rewrite）
# "knowledge" — 知识库文档整理（import_knowledge）
_llm_configs: Dict[str, Dict[str, Any]] = {
    "chat": {
        "enabled": False,
        "provider": "",
        "model": "",
        "api_base": "",
        "api_key": "",      # 运行时存储，不持久化
    },
    "knowledge": {
        "enabled": False,
        "provider": "",
        "model": "",
        "api_base": "",
        "api_key": "",
    },
}

# 单例缓存：purpose -> OpenAI client
_clients: Dict[str, Any] = {"chat": None, "knowledge": None}
_clients_checked: Dict[str, bool] = {"chat": False, "knowledge": False}


def _init_from_legacy():
    """从旧版全局配置初始化（兼容迁移）。仅在首次加载时调用。"""
    from rag_runtime_config import USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    for purpose in ("chat", "knowledge"):
        cfg = _llm_configs[purpose]
        if not cfg["provider"]:
            cfg["enabled"] = USE_OPENAI
            cfg["model"] = OPENAI_MODEL
            cfg["api_base"] = OPENAI_API_BASE or ""
            cfg["api_key"] = key
            if OPENAI_API_BASE:
                base_lower = OPENAI_API_BASE.lower()
                if "deepseek" in base_lower:
                    cfg["provider"] = "deepseek"
                elif "minimax" in base_lower:
                    cfg["provider"] = "minimax"
                else:
                    cfg["provider"] = "custom"
            elif USE_OPENAI:
                cfg["provider"] = "openai"


def get_llm_config(purpose: str = "chat") -> Dict[str, Any]:
    """获取指定用途的 LLM 配置（只读副本）"""
    cfg = _llm_configs.get(purpose, _llm_configs["chat"])
    return {
        "enabled": cfg["enabled"],
        "provider": cfg["provider"],
        "model": cfg["model"],
        "api_base": cfg["api_base"],
        "api_key_set": bool(cfg["api_key"]),
    }


def get_all_llm_configs() -> Dict[str, Any]:
    """获取所有用途的 LLM 配置"""
    return {
        purpose: get_llm_config(purpose)
        for purpose in ("chat", "knowledge")
    }


def update_llm_config(purpose: str, *,
                       provider: str = "",
                       model: str = "",
                       api_base: Optional[str] = None,
                       api_key: str = "",
                       enabled: Optional[bool] = None) -> Dict[str, Any]:
    """更新指定用途的 LLM 配置，并重置对应的 client 缓存。"""
    if purpose not in _llm_configs:
        return {"error": f"未知用途: {purpose}，支持 chat / knowledge"}

    from rag_runtime_config import MODEL_PRESETS
    cfg = _llm_configs[purpose]

    with _lock:
        if provider:
            preset = MODEL_PRESETS.get(provider)
            if preset:
                if not model:
                    model = preset["default_model"]
                if api_base is None:
                    api_base = preset["api_base"]
            cfg["provider"] = provider
        if model:
            cfg["model"] = model
        if api_base is not None:
            cfg["api_base"] = api_base
        if api_key:
            cfg["api_key"] = api_key
        if enabled is not None:
            cfg["enabled"] = bool(enabled)
        # 重置 client 缓存
        _clients[purpose] = None
        _clients_checked[purpose] = False

    # 同步到旧版全局变量（保持向后兼容）
    _sync_to_legacy(purpose)
    # 持久化
    _persist_llm_configs()

    return {
        "ok": True,
        "purpose": purpose,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "api_base": cfg["api_base"],
        "enabled": cfg["enabled"],
    }


def get_client(purpose: str = "chat"):
    """获取指定用途的 OpenAI 兼容 client（单例缓存）。
    返回 None 表示未配置或不可用。
    """
    if purpose not in _llm_configs:
        purpose = "chat"

    # 快速路径：已检查过直接返回
    if _clients_checked[purpose]:
        return _clients[purpose]

    # 慢路径：加锁创建 client
    with _lock:
        if _clients_checked[purpose]:
            return _clients[purpose]

        cfg = _llm_configs[purpose]
        if not cfg["enabled"]:
            _clients_checked[purpose] = True
            return None

        api_key = cfg["api_key"] or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            _clients_checked[purpose] = True
            return None

        try:
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if cfg["api_base"]:
                kwargs["base_url"] = cfg["api_base"]
            _clients[purpose] = OpenAI(**kwargs)
        except Exception:
            _clients[purpose] = None
        _clients_checked[purpose] = True
        return _clients[purpose]


def get_model(purpose: str = "chat") -> str:
    """获取指定用途的模型名称"""
    return _llm_configs.get(purpose, _llm_configs["chat"])["model"]


def is_enabled(purpose: str = "chat") -> bool:
    """指定用途的 LLM 是否启用"""
    return _llm_configs.get(purpose, _llm_configs["chat"])["enabled"]


def reset_client(purpose: str = "chat"):
    """重置指定用途的 client 缓存（强制下次调用重新创建）"""
    with _lock:
        _clients[purpose] = None
        _clients_checked[purpose] = False


def reset_all_clients():
    """重置所有 client 缓存"""
    with _lock:
        for p in _clients:
            _clients[p] = None
            _clients_checked[p] = False


def _sync_to_legacy(purpose: str):
    """将 chat 配置同步回旧版全局变量，保持 rag_answer 等模块的兼容性。"""
    if purpose == "chat":
        import rag_runtime_config as _mod
        cfg = _llm_configs["chat"]
        _mod.USE_OPENAI = cfg["enabled"]
        _mod.OPENAI_MODEL = cfg["model"]
        _mod.OPENAI_API_BASE = cfg["api_base"] or None
        if cfg["api_key"]:
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
        # 重置 rag_answer 的旧版 client 缓存
        try:
            import rag_answer
            rag_answer._openai_client = None
            rag_answer._openai_client_checked = False
        except Exception:
            pass


def sync_from_legacy():
    """从旧版全局变量同步到 llm_client（旧 API 调用后调用此函数保持一致）。"""
    from rag_runtime_config import USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE
    cfg = _llm_configs["chat"]
    with _lock:
        cfg["enabled"] = USE_OPENAI
        cfg["model"] = OPENAI_MODEL
        cfg["api_base"] = OPENAI_API_BASE or ""
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            cfg["api_key"] = key
        _clients["chat"] = None
        _clients_checked["chat"] = False


def _persist_llm_configs():
    """持久化多 LLM 配置到文件"""
    import json
    from rag_runtime_config import BASE_DIR
    config_file = BASE_DIR / "data" / "llm_configs.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for purpose, cfg in _llm_configs.items():
        data[purpose] = {
            "enabled": cfg["enabled"],
            "provider": cfg["provider"],
            "model": cfg["model"],
            "api_base": cfg["api_base"],
            # api_key 不持久化到文件（安全考虑）
            "api_key_set": bool(cfg["api_key"]),
        }
    with _lock:
        tmp = config_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(config_file)


def load_persisted_llm_configs():
    """启动时加载持久化的多 LLM 配置"""
    import json
    from rag_runtime_config import BASE_DIR
    config_file = BASE_DIR / "data" / "llm_configs.json"
    if not config_file.exists():
        return False
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        loaded = False
        for purpose in ("chat", "knowledge"):
            if purpose in data:
                saved = data[purpose]
                cfg = _llm_configs[purpose]
                cfg["enabled"] = saved.get("enabled", False)
                cfg["provider"] = saved.get("provider", "")
                cfg["model"] = saved.get("model", "")
                cfg["api_base"] = saved.get("api_base", "")
                if purpose == "chat":
                    cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "").strip()
                elif purpose == "knowledge":
                    cfg["api_key"] = os.environ.get(
                        "KNOWLEDGE_LLM_API_KEY",
                        os.environ.get("OPENAI_API_KEY", "")
                    ).strip()
                loaded = True
        if loaded:
            _sync_to_legacy("chat")
            print(f"[INFO] 已加载多 LLM 配置: chat={_llm_configs['chat']['provider']}/{_llm_configs['chat']['model']}, "
                  f"knowledge={_llm_configs['knowledge']['provider']}/{_llm_configs['knowledge']['model']}")
        return loaded
    except Exception as e:
        print(f"[WARN] 加载多 LLM 配置失败: {e}")
        return False


# 模块加载时初始化
_loaded = load_persisted_llm_configs()
if not _loaded:
    _init_from_legacy()
