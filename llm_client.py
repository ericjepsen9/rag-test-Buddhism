"""
多 LLM 提供商客户端管理器。

支持为「对话」和「知识库整理」分别配置不同的 LLM 提供商 / 模型 / API Key。
所有 LLM 调用统一通过本模块获取 client，不再直接使用 rag_runtime_config 中的全局变量。
"""
import base64
import copy
import os
import threading
from typing import Optional, Dict, Any

_lock = threading.Lock()
_persist_lock = threading.Lock()

# 两个用途的独立配置
# "chat"      — 用户对话（rag_answer, query_rewrite）
# "knowledge" — 知识库文档整理（import_knowledge）
_llm_configs: Dict[str, Dict[str, Any]] = {
    "chat": {
        "enabled": False,
        "provider": "",
        "model": "",
        "api_base": "",
        "api_key": "",
        "model_format": "standard",  # "standard" or "litellm" (provider:model_id)
        "connection_verified": False,  # 测试连接成功后设为 True，配置变更时重置
    },
    "knowledge": {
        "enabled": False,
        "provider": "",
        "model": "",
        "api_base": "",
        "api_key": "",
        "model_format": "standard",  # "standard" or "litellm" (provider:model_id)
        "connection_verified": False,
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
                # 根据 api_base 猜测 provider
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
        "model_format": cfg.get("model_format", "standard"),
        "client_ready": _clients.get(purpose) is not None and _clients_checked.get(purpose, False),
        "connection_verified": cfg.get("connection_verified", False),
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
                       enabled: Optional[bool] = None,
                       model_format: str = "") -> Dict[str, Any]:
    """更新指定用途的 LLM 配置，并重置对应的 client 缓存。

    api_base: None 表示不更新，"" 表示清空（恢复默认）。
    """
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
        if model_format and model_format in ("standard", "litellm"):
            cfg["model_format"] = model_format
        # 配置变更，重置连接验证状态和 client 缓存
        cfg["connection_verified"] = False
        _clients[purpose] = None
        _clients_checked[purpose] = False
        # 同步到旧版全局变量（在锁内执行，防止并发更新竞争）
        _sync_to_legacy(purpose)

    # 持久化（锁外执行，避免 I/O 阻塞其他配置读取）
    _persist_llm_configs()

    return {
        "ok": True,
        "purpose": purpose,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "api_base": cfg["api_base"],
        "enabled": cfg["enabled"],
        "model_format": cfg.get("model_format", "standard"),
    }


def get_client(purpose: str = "chat"):
    """获取指定用途的 OpenAI 兼容 client（单例缓存）。
    返回 None 表示未配置或不可用。
    """
    if purpose not in _llm_configs:
        purpose = "chat"

    # 快速路径：已检查过直接返回（GIL 保证 dict 读安全）
    if _clients_checked[purpose]:
        return _clients[purpose]

    # 慢路径：加锁创建 client，防止并发重复创建
    with _lock:
        # double-check：另一个线程可能已经完成创建
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
            # 超时配置：连接 10s，读取 60s（LLM 生成可能较慢）
            _timeout = float(os.environ.get("LLM_CLIENT_TIMEOUT", "60"))
            kwargs["timeout"] = _timeout
            _clients[purpose] = OpenAI(**kwargs)
        except Exception as e:
            from rag_logger import log_error
            log_error("llm_client", f"OpenAI client ({purpose}) 初始化失败: {e}")
            _clients[purpose] = None
            # 不标记 checked，允许后续重试（可能是临时网络错误）
            return None
        _clients_checked[purpose] = True
        return _clients[purpose]


def get_model(purpose: str = "chat") -> str:
    """获取指定用途的模型名称。

    当 model_format 为 "litellm" 时，返回 "provider:model" 格式
    （适用于 LiteLLM 代理等需要 provider:model_id 格式的 API 端点）。
    """
    cfg = _llm_configs.get(purpose, _llm_configs["chat"])
    model = cfg["model"]
    if cfg.get("model_format") == "litellm" and model:
        # LiteLLM 模式：模型名本身已包含 provider 前缀（如 openai/xxx）
        # 仅当模型名中没有 "/" 也没有 ":" 时才拼接 provider
        if "/" not in model and ":" not in model and cfg["provider"] and cfg["provider"] != "custom":
            return f"{cfg['provider']}:{model}"
    return model


def is_enabled(purpose: str = "chat") -> bool:
    """指定用途的 LLM 是否启用"""
    return _llm_configs.get(purpose, _llm_configs["chat"])["enabled"]


def mark_connection_verified(purpose: str = "chat", verified: bool = True):
    """标记指定用途的连接验证状态（测试成功后调用）"""
    cfg = _llm_configs.get(purpose)
    if cfg:
        with _lock:
            cfg["connection_verified"] = verified
        _persist_llm_configs()


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
    """从旧版全局变量同步到 llm_client（旧 API 调用后调用此函数保持一致）。
    注意：不同步 enabled 字段。服务控制（start/stop）是运行时总开关，
    不应改变模型配置页面的 enabled 状态。enabled 只通过模型配置页面保存来修改。
    """
    from rag_runtime_config import OPENAI_MODEL, OPENAI_API_BASE
    cfg = _llm_configs["chat"]
    with _lock:
        cfg["model"] = OPENAI_MODEL
        cfg["api_base"] = OPENAI_API_BASE or ""
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            cfg["api_key"] = key
        _clients["chat"] = None
        _clients_checked["chat"] = False


def _persist_llm_configs():
    """持久化多 LLM 配置到文件。
    使用 _persist_lock 序列化写入，用 _lock 短暂持锁读取快照，
    避免并发 update_llm_config 导致后写入覆盖先写入的修改。"""
    import json
    from rag_runtime_config import BASE_DIR
    config_file = BASE_DIR / "data" / "llm_configs.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    # 短暂持锁读取配置快照
    with _lock:
        snapshot = copy.deepcopy(_llm_configs)
    data = {}
    for purpose, cfg in snapshot.items():
        raw_key = cfg["api_key"] or ""
        encoded_key = base64.b64encode(raw_key.encode()).decode() if raw_key else ""
        data[purpose] = {
            "enabled": cfg["enabled"],
            "provider": cfg["provider"],
            "model": cfg["model"],
            "api_base": cfg["api_base"],
            "api_key_set": bool(cfg["api_key"]),
            "api_key_enc": encoded_key,
            "model_format": cfg.get("model_format", "standard"),
            "connection_verified": cfg.get("connection_verified", False),
        }
    # 用独立的持久化锁序列化文件写入
    with _persist_lock:
        tmp = config_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.chmod(str(tmp), 0o600)
        except OSError:
            pass
        tmp.replace(config_file)
        try:
            os.chmod(str(config_file), 0o600)
        except OSError:
            pass


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
                cfg["model_format"] = saved.get("model_format", "standard")
                cfg["connection_verified"] = saved.get("connection_verified", False)
                # api_key: 优先从持久化文件恢复，其次从环境变量
                enc_key = saved.get("api_key_enc", "")
                if enc_key:
                    try:
                        cfg["api_key"] = base64.b64decode(enc_key).decode()
                    except Exception:
                        cfg["api_key"] = ""
                if not cfg["api_key"]:
                    if purpose == "chat":
                        cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "").strip()
                    elif purpose == "knowledge":
                        cfg["api_key"] = os.environ.get(
                            "KNOWLEDGE_LLM_API_KEY",
                            os.environ.get("OPENAI_API_KEY", "")
                        ).strip()
                loaded = True
        if loaded:
            # 同步 chat 到旧版全局变量
            _sync_to_legacy("chat")
            from rag_logger import log_event
            log_event("llm_config", "已加载多 LLM 配置",
                      meta={"chat": f"{_llm_configs['chat']['provider']}/{_llm_configs['chat']['model']}",
                            "knowledge": f"{_llm_configs['knowledge']['provider']}/{_llm_configs['knowledge']['model']}"})
        return loaded
    except Exception as e:
        from rag_logger import log_error
        log_error("llm_client", f"加载多 LLM 配置失败: {e}")
        return False


# 模块加载时初始化
_loaded = load_persisted_llm_configs()
if not _loaded:
    _init_from_legacy()
