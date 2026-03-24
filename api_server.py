import hashlib
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from media_router import find_media, invalidate_media_cache
from rag_answer import answer_question, invalidate_store_cache, get_last_route_product
from rag_logger import log_error, log_event, get_recent_qa, get_recent_misses, get_recent_errors

logger = logging.getLogger("rag-api")

# ===== 请求限流 =====
_ASK_RATE_LIMIT = os.environ.get("ASK_RATE_LIMIT", "60/minute")
_ADMIN_RATE_LIMIT = os.environ.get("ADMIN_RATE_LIMIT", "30/minute")
limiter = Limiter(key_func=get_remote_address)

# ===== 请求超时控制 =====
_ASK_TIMEOUT_SEC = int(os.environ.get("ASK_TIMEOUT_SEC", "30"))

# ===== 响应缓存 =====
_RESPONSE_CACHE: OrderedDict = OrderedDict()
_RESPONSE_CACHE_MAX = int(os.environ.get("RESPONSE_CACHE_MAX", "200"))
_RESPONSE_CACHE_TTL = int(os.environ.get("RESPONSE_CACHE_TTL", "300"))  # 秒
_response_cache_lock = threading.Lock()

# 每产品重建锁，防止并发 rebuild 导致文件损坏
_rebuild_locks: Dict[str, threading.Lock] = {}
_rebuild_locks_guard = threading.Lock()
from rag_runtime_config import KNOWLEDGE_DIR, SHARED_ENTITY_DIRS

# 预计算共享目录名集合，避免 health/admin_products 每次调用重建 set
_SHARED_DIR_NAMES = frozenset(SHARED_ENTITY_DIRS.values())

BASE_DIR = Path(__file__).resolve().parent
ADMIN_PAGE = BASE_DIR / "admin_page.html"
MOBILE_PAGE = BASE_DIR / "mobile.html"
CHAT_PAGE = BASE_DIR / "web" / "chat.html"

_startup_time: float = 0.0  # 服务启动时间戳（monotonic）

@asynccontextmanager
async def _lifespan(app):
    """启动时预热嵌入模型并预构建共享知识索引，避免首次查询延迟"""
    from rag_logger import log_event, log_error
    global _startup_time
    _startup_time = time.monotonic()
    log_event("startup", "服务启动中...", meta={
        "python": os.popen("python3 --version 2>&1").read().strip(),
        "pid": os.getpid(),
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "admin_key_set": bool(_ADMIN_API_KEY),
    })
    if not os.environ.get("SKIP_WARMUP"):
        try:
            from rag_answer import embed_query
            embed_query("预热查询")
            log_event("startup", "嵌入模型预热完成")
        except Exception as e:
            log_error("startup", f"模型预热失败: {e}")
        # 预构建共享知识索引（procedures/equipment/anatomy 等），
        # 避免首次跨实体查询时 30s+ 的冷启动延迟
        try:
            from rag_answer import _ensure_shared_store
            _ensure_shared_store()
            log_event("startup", "共享知识库索引预检完成")
        except Exception as e:
            log_error("startup", f"共享知识库预构建失败: {e}")
        # 预加载学习同义词到运行时，避免首次查询时的加载延迟
        try:
            from search_utils import reload_learned_synonyms
            count = reload_learned_synonyms()
            log_event("startup", f"学习同义词预加载完成: {count} 条")
        except Exception as e:
            log_error("startup", f"学习同义词预加载失败: {e}")
    yield
    # 优雅停机：清理资源
    log_event("shutdown", "服务正在关闭，刷新日志缓存...")
    try:
        from rag_answer import _search_pool
        _search_pool.shutdown(wait=True, cancel_futures=False)
        log_event("shutdown", "搜索线程池已关闭")
    except Exception as e:
        log_error("shutdown", f"搜索线程池关闭失败: {e}")

app = FastAPI(title="Buddhist Knowledge RAG API", version="1.0.0", lifespan=_lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"请求过于频繁，请稍后再试。限制: {exc.detail}"},
    )


app.add_middleware(
    CORSMiddleware,
    # 生产环境务必设置 CORS_ORIGINS 为具体域名（逗号分隔），如:
    # CORS_ORIGINS=https://example.com,https://admin.example.com
    # 默认 "*" 仅用于开发/测试
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


# ===== 请求追踪中间件 =====
@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """为每个请求生成唯一 trace_id，贯穿日志链路"""
    from rag_logger import set_trace_id, get_trace_id
    # 支持上游传入 trace_id（如 nginx/网关），否则自动生成
    incoming = request.headers.get("X-Trace-Id", "")
    trace_id = set_trace_id(incoming)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


# ===== 安全响应头中间件 =====
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """为所有响应添加安全头，防止 MIME 嗅探、点击劫持等"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # CSP：限制资源加载来源；inline script/style 因单页 HTML 架构需保留
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response

MAX_QUESTION_LEN = int(os.environ.get("MAX_QUESTION_LEN", "500"))
MAX_HISTORY_TOTAL_CHARS = int(os.environ.get("MAX_HISTORY_TOTAL_CHARS", "3000"))

# ===== 管理接口鉴权 =====
# 通过环境变量 ADMIN_API_KEY 设置管理密钥，未设置时管理接口不开放鉴权（向后兼容）
_ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "").strip()

# admin 页面本身（HTML）免鉴权，但所有 admin API 需要鉴权
_ADMIN_AUTH_EXEMPT = frozenset({"/admin"})


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    """管理接口鉴权中间件：对 /admin/* API 路径校验密钥"""
    path = request.url.path.rstrip("/")
    if path.startswith("/admin") and path not in _ADMIN_AUTH_EXEMPT:
        if not _ADMIN_API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"detail": "管理接口未配置鉴权密钥。请设置环境变量 ADMIN_API_KEY 后重启服务。"},
            )
    if path.startswith("/admin") and path not in _ADMIN_AUTH_EXEMPT and _ADMIN_API_KEY:
        auth = request.headers.get("authorization", "")
        key_ok = (
            auth.startswith("Bearer ") and auth[7:].strip() == _ADMIN_API_KEY
        )
        if not key_ok:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"detail": "管理接口需要鉴权：请设置 ADMIN_API_KEY 并在请求中携带 Authorization: Bearer <key>"},
            )
    return await call_next(request)


# ===== 请求日志中间件 =====
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """记录所有 API 请求的日志（非静态资源）。

    INFO 级别：记录 /ask、/admin/* 的状态码和耗时
    DEBUG 级别：额外记录请求体
    """
    path = request.url.path
    method = request.method

    # 跳过静态资源和健康检查（减少噪音）
    if path.startswith("/web/") or path in ("/favicon.ico", "/manifest.json", "/sw.js"):
        return await call_next(request)

    is_debug = logger.isEnabledFor(logging.DEBUG)
    is_api = path.startswith("/admin") or path in ("/ask", "/v1/chat/completions", "/health")

    # DEBUG 模式下捕获请求体
    body_text = ""
    if is_debug and method in ("POST", "PUT", "PATCH") and path.startswith("/admin"):
        try:
            body = await request.body()
            body_text = body.decode("utf-8", errors="replace")[:2000]
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
        except Exception:
            pass

    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 1)

    if is_api:
        status = response.status_code
        # 错误状态码始终记录（无论日志级别），方便排查
        if status >= 400:
            log_msg = f"[ERR] {method} {path} → {status} ({duration}ms)"
            if request.url.query:
                log_msg += f"  query: {request.url.query}"
            if body_text:
                log_msg += f"  body: {body_text[:300]}"
            logger.warning(log_msg)
        elif is_debug:
            log_msg = f"{method} {path} → {status} ({duration}ms)"
            if request.url.query:
                log_msg += f"  query: {request.url.query}"
            if body_text:
                log_msg += f"  body: {body_text[:500]}"
            logger.debug(log_msg)

    return response


# 输入清理：去除 HTML 标签和控制字符，防止注入
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_input(text: str) -> str:
    """清理用户输入：去除 HTML 标签和控制字符"""
    text = _HTML_TAG_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


# ===== 数据模型 =====

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(..., max_length=500)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LEN)
    mode: Literal["brief", "full"] = "brief"
    history: List[HistoryItem] = Field(default_factory=list, max_length=10)
    debug: bool = False


class MediaItem(BaseModel):
    title: str
    type: str
    url: str = ""


class ClarificationOption(BaseModel):
    label: str
    query: str
    route: str = ""


class ClarificationData(BaseModel):
    message: str
    options: List[ClarificationOption]
    fallback_option: Optional[ClarificationOption] = None


class AskResponse(BaseModel):
    ok: bool
    answer: str
    media: List[MediaItem] = []
    latency_ms: Optional[int] = None
    debug: Optional[Dict[str, Any]] = None
    needs_clarification: bool = False
    clarification: Optional[ClarificationData] = None


class RebuildRequest(BaseModel):
    product: str = Field(..., min_length=1, max_length=50)
    timeout_sec: int = Field(default=120, ge=10, le=600)


# ===== OpenAI 兼容数据模型 =====

_MODEL_NAME = os.environ.get("OPENAI_COMPAT_MODEL", "buddhism-rag")


class OAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class OAIChatRequest(BaseModel):
    model: str = _MODEL_NAME
    messages: List[OAIMessage] = Field(..., min_length=1)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


# ===== 问答接口 =====

# 健康检查缓存：避免频繁的文件系统探测（Kubernetes probes 等）
_health_cache: Dict[str, Any] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = float(os.environ.get("HEALTH_CACHE_TTL", "15.0"))
_health_lock = threading.Lock()


@app.get("/health")
def health():
    """健康检查：返回各产品索引状态、文档数、embedding模型状态及服务运行时间"""
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    # 快速路径：无锁读取（GIL 保护 dict 引用读取安全性）
    if _health_cache and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return _health_cache

    from rag_runtime_config import STORE_ROOT
    shared_names = _SHARED_DIR_NAMES
    products = []
    total_docs = 0
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir() or p.name in shared_names:
                continue
            store = STORE_ROOT / p.name
            index_path = store / "index.faiss"
            docs_path = store / "docs.jsonl"
            index_exists = index_path.exists()
            docs_exists = docs_path.exists()
            doc_count = 0
            index_size = 0
            if docs_exists:
                try:
                    doc_count = sum(1 for _ in docs_path.open("r", encoding="utf-8"))
                except OSError:
                    pass
            if index_exists:
                try:
                    index_size = index_path.stat().st_size
                except OSError:
                    pass
            total_docs += doc_count
            products.append({
                "name": p.name,
                "index_exists": index_exists,
                "docs_exists": docs_exists,
                "doc_count": doc_count,
                "index_size_bytes": index_size,
            })
    # 共享知识索引状态
    shared_store = STORE_ROOT / "_shared"
    shared_indexed = (shared_store / "index.faiss").exists() and (shared_store / "docs.jsonl").exists()
    shared_docs = 0
    if shared_indexed:
        try:
            shared_docs = sum(1 for _ in (shared_store / "docs.jsonl").open("r", encoding="utf-8"))
        except OSError:
            pass
    all_indexed = all(p["index_exists"] and p["docs_exists"] for p in products) if products else False

    # 依赖检查
    embedding_ready = False
    try:
        from rag_answer import _model
        embedding_ready = _model is not None
    except Exception as e:
        from rag_logger import log_error
        log_error("health", f"embedding 模型状态检查失败: {e}")

    result = {
        "status": "ok" if all_indexed else "degraded",
        "knowledge_exists": KNOWLEDGE_DIR.exists(),
        "products": products,
        "shared_knowledge_indexed": shared_indexed,
        "total_product_docs": total_docs,
        "shared_docs": shared_docs,
        "embedding_model_loaded": embedding_ready,
        "response_cache_size": len(_RESPONSE_CACHE),
        "uptime_seconds": int(now - _startup_time) if _startup_time else 0,
    }
    with _health_lock:
        _health_cache = result
        _health_cache_ts = now
    return result


@app.get("/chat")
def chat_page():
    """对话页面入口"""
    if not CHAT_PAGE.exists():
        raise HTTPException(status_code=404, detail="chat.html 不存在")
    return FileResponse(CHAT_PAGE, media_type="text/html",
                        headers={"Cache-Control": "public, max-age=300"})


@app.get("/sw.js")
def service_worker():
    """Service Worker 必须从根路径提供，以获得全站作用域"""
    sw_path = BASE_DIR / "web" / "sw.js"
    if not sw_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(sw_path, media_type="application/javascript",
                        headers={"Cache-Control": "no-cache",
                                 "Service-Worker-Allowed": "/"})


def _response_cache_get(key: str):
    """从响应缓存获取，返回 None 表示未命中"""
    with _response_cache_lock:
        entry = _RESPONSE_CACHE.get(key)
        if entry is None:
            return None
        cached_time, data = entry
        if time.monotonic() - cached_time > _RESPONSE_CACHE_TTL:
            _RESPONSE_CACHE.pop(key, None)
            return None
        _RESPONSE_CACHE.move_to_end(key)
        return data


def _response_cache_put(key: str, data):
    """写入响应缓存"""
    with _response_cache_lock:
        while len(_RESPONSE_CACHE) >= _RESPONSE_CACHE_MAX:
            try:
                _RESPONSE_CACHE.popitem(last=False)
            except KeyError:
                break
        _RESPONSE_CACHE[key] = (time.monotonic(), data)


@app.post("/ask", response_model=AskResponse)
@limiter.limit(_ASK_RATE_LIMIT)
def ask(request: Request, req: AskRequest):
    """RAG 问答主接口：接受用户问题，返回基于知识库的回答、相关媒体资源及调试信息"""
    t0 = time.monotonic()
    rw = None
    question = ""
    try:
        question = _sanitize_input(req.question)
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        logger.debug(f"ask: question={question[:200]}")

        # 响应缓存：相同 question+mode+history 组合在 TTL 内直接返回
        history_hash = hashlib.md5(str(req.history or []).encode()).hexdigest()[:8] if req.history else ""
        cache_key = f"{question}|{req.mode}|{history_hash}"
        cached = _response_cache_get(cache_key)
        if cached is not None and not req.debug:
            cached_resp = cached.copy()
            cached_resp["_cached"] = True
            return AskResponse(**cached_resp)
        history = [{"role": h.role, "content": _sanitize_input(h.content[:1000])} for h in req.history[-6:]]
        # 防御：限制历史总字符数，避免过大 payload 占用内存
        total_chars = sum(len(h.get("content", "")) for h in history)
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            # 从最早的消息裁剪，保留最近的对话（用累加代替 O(n) pop(0)）
            cum = 0
            trim_idx = 0
            excess = total_chars - MAX_HISTORY_TOTAL_CHARS
            for i, h in enumerate(history):
                cum += len(h.get("content", ""))
                if cum >= excess:
                    trim_idx = i + 1
                    break
            history = history[trim_idx:]
        # rewrite 只做一次，传给 answer_question 复用
        from query_rewrite import rewrite_query
        rw = rewrite_query(question, history=history)
        resolved_q = rw["original"]  # 上下文补全后的问题

        # 超时控制：复用 rag_answer 模块级线程池，避免每请求创建/销毁
        from concurrent.futures import TimeoutError as FuturesTimeout
        from rag_answer import _search_pool
        from rag_logger import get_trace_id, set_trace_id
        _tid = get_trace_id()

        _ctx = {}  # 跨线程共享容器

        def _run_with_trace():
            set_trace_id(_tid)  # 传播 trace_id 到工作线程
            ans = answer_question(question, req.mode, history=history, rewrite=rw)
            _ctx["route"], _ctx["product"] = get_last_route_product()
            return ans

        future = _search_pool.submit(_run_with_trace)
        try:
            answer = future.result(timeout=_ASK_TIMEOUT_SEC)
        except FuturesTimeout:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log_error("api_ask_timeout", f"请求超时 ({_ASK_TIMEOUT_SEC}s)",
                      meta={"question": question[:200], "latency_ms": latency_ms})
            return AskResponse(ok=False, answer=f"查询处理超时（{_ASK_TIMEOUT_SEC}秒），请简化问题后重试")
        latency_ms = int((time.monotonic() - t0) * 1000)
        route = _ctx.get("route", "")
        product_id = _ctx.get("product", "")
        if not route or not product_id:
            from rag_answer import detect_route, detect_product
            product_id = product_id or detect_product(resolved_q)
            route = route or detect_route(resolved_q)
        media = [MediaItem(**m) for m in find_media(resolved_q, product_id=product_id, route=route)]
        debug = None
        if req.debug:
            # 查询链路信息
            debug = {
                "trace_id": _tid,
                "question": question,
                "resolved_question": resolved_q if rw["context_resolved"] else None,
                "search_query": rw.get("search_query", ""),
                "mode": req.mode,
                "route": route,
                "product": product_id,
                "method": None,
                "expanded_query": rw["expanded"],
                "context_resolved": rw["context_resolved"],
                # 查询改写详情
                "llm_rewritten": rw.get("llm_rewritten") or None,
                "detected_routes": rw.get("detected_routes", []),
                "sub_questions": rw.get("sub_questions", []),
                "products": rw.get("products", []),
                "projects": rw.get("projects", []),
                "times": rw.get("times", []),
                "symptoms": rw.get("symptoms", []),
                # 特殊路径标记
                "is_chitchat": rw.get("is_chitchat", False),
                "is_offtopic": rw.get("is_offtopic", False),
                "needs_clarification": rw.get("needs_clarification", False),
                # 历史上下文
                "history_summary": rw.get("history_summary") or None,
                "history_pairs_count": len(rw.get("history_pairs", [])),
                "last_user_q": rw.get("last_user_q") or None,
                # 性能
                "latency_ms": latency_ms,
            }
        # 消歧引导：当查询模糊时，在回答同时提供候选选项
        needs_clarification = rw.get("needs_clarification", False)
        clarification_data = None
        if needs_clarification and rw.get("clarification"):
            cl = rw["clarification"]
            cl_options = [ClarificationOption(**o) for o in cl.get("options", [])]
            fb = cl.get("fallback_option")
            cl_fb = ClarificationOption(**fb) if fb else None
            clarification_data = ClarificationData(
                message=cl["message"],
                options=cl_options,
                fallback_option=cl_fb,
            )

        resp = AskResponse(
            ok=True,
            answer=answer,
            media=media,
            latency_ms=latency_ms,
            debug=debug,
            needs_clarification=needs_clarification,
            clarification=clarification_data,
        )
        logger.debug(f"ask: response_length={len(answer)}, latency_ms={latency_ms}")
        # 写入响应缓存（仅缓存成功且非 debug 的响应）
        if not req.debug and not needs_clarification:
            _response_cache_put(cache_key, {
                "ok": True, "answer": answer,
                "media": [m.model_dump() if hasattr(m, 'model_dump') else m.__dict__ for m in media],
                "latency_ms": latency_ms,
            })
        return resp
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        error_meta: Dict[str, Any] = {
            "question": question[:200],
            "latency_ms": latency_ms,
            "mode": req.mode,
        }
        # 尝试捕获已解析的上下文信息
        try:
            if rw:
                cached_route, cached_product = get_last_route_product()
                error_meta["route"] = cached_route or None
                error_meta["product"] = cached_product or None
        except Exception:
            pass
        log_error("api_ask", repr(e), meta=error_meta)
        # 根据异常类型返回更具体的错误信息
        err_str = str(e).lower()
        err_msg = "接口执行异常，请稍后重试"
        if "model" in err_str or "sentence" in err_str:
            err_msg = "嵌入模型未加载，请等待模型初始化完成后重试"
        elif "openai" in err_str or "api_key" in err_str:
            err_msg = "LLM 服务未配置，请先在管理后台配置 LLM 模型"
        elif "connect" in err_str or "connection" in err_str or "refused" in err_str:
            err_msg = "LLM 服务连接失败，请检查 API Base 地址和 LLM 服务是否正在运行"
        # 附加具体错误类型（便于管理员排查）
        err_detail = f"{type(e).__name__}: {str(e)[:200]}"
        logger.warning(f"ask error: {err_detail}")
        return AskResponse(
            ok=False,
            answer=err_msg,
        )


# ===== OpenAI 兼容接口 =====

def _oai_messages_to_question_and_history(messages: List[OAIMessage]):
    """将 OpenAI messages 格式转换为 question + history"""
    # 过滤掉 system 消息，提取 user/assistant 对话
    conv = [m for m in messages if m.role in ("user", "assistant")]
    if not conv:
        return "", []
    # 最后一条消息必须是 user 角色；如果是 assistant 则向前找最后一条 user 消息
    last = conv[-1]
    if last.role != "user":
        # 找最后一条 user 消息作为问题
        user_msgs = [m for m in conv if m.role == "user"]
        if not user_msgs:
            return "", []
        question = _sanitize_input(user_msgs[-1].content[:MAX_QUESTION_LEN])
        # 历史保留最后 user 消息之前的内容
        last_user_idx = len(conv) - 1 - conv[::-1].index(user_msgs[-1])
        history = [
            {"role": m.role, "content": _sanitize_input(m.content[:1000])}
            for m in conv[:last_user_idx]
        ][-6:]
    else:
        question = _sanitize_input(last.content[:MAX_QUESTION_LEN])
        history = [
            {"role": m.role, "content": _sanitize_input(m.content[:1000])}
            for m in conv[:-1]
        ][-6:]
    return question, history


def _build_oai_response(answer: str, model: str) -> Dict[str, Any]:
    """构建 OpenAI 兼容的响应格式"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


_RE_SENTENCE_SPLIT = re.compile(r'([。！？\n；;!?])')


def _split_into_stream_chunks(text: str) -> list:
    """将完整回答按句子边界拆分为流式分块。
    每个分块以句末标点结尾，给客户端逐句渐进显示体验。"""
    if not text:
        return [text] if text == "" else []
    parts = _RE_SENTENCE_SPLIT.split(text)
    chunks = []
    buf = ""
    for p in parts:
        buf += p
        if _RE_SENTENCE_SPLIT.fullmatch(p):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text]


def _build_oai_stream_chunk(content: str, model: str, chunk_id: str, finish: bool = False) -> str:
    """构建 SSE 格式的流式响应块"""
    import json
    if finish:
        delta = {}
        finish_reason = "stop"
    else:
        delta = {"role": "assistant", "content": content}
        finish_reason = None
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
@limiter.limit(_ASK_RATE_LIMIT)
def oai_chat_completions(request: Request, req: OAIChatRequest):
    """OpenAI 兼容接口：支持 messages 格式的对话请求，可选流式响应"""
    question, history = _oai_messages_to_question_and_history(req.messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    # 响应缓存（非流式且无历史时）
    cache_key = f"oai|{question}" if not history and not req.stream else None
    if cache_key:
        cached = _response_cache_get(cache_key)
        if cached is not None:
            return _build_oai_response(cached["answer"], req.model)

    t0 = time.monotonic()
    try:
        # 限制历史总字符数
        total_chars = sum(len(h.get("content", "")) for h in history)
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            cum = 0
            trim_idx = 0
            excess = total_chars - MAX_HISTORY_TOTAL_CHARS
            for i, h in enumerate(history):
                cum += len(h.get("content", ""))
                if cum >= excess:
                    trim_idx = i + 1
                    break
            history = history[trim_idx:]

        from query_rewrite import rewrite_query
        rw = rewrite_query(question, history=history)

        # 超时控制：复用 rag_answer 模块级线程池
        from concurrent.futures import TimeoutError as FuturesTimeout
        from rag_answer import _search_pool
        from rag_logger import get_trace_id, set_trace_id
        _oai_tid = get_trace_id()

        def _oai_run():
            set_trace_id(_oai_tid)
            return answer_question(question, "brief", history=history, rewrite=rw)

        future = _search_pool.submit(_oai_run)
        try:
            answer = future.result(timeout=_ASK_TIMEOUT_SEC)
        except FuturesTimeout:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log_error("oai_chat_timeout", f"请求超时 ({_ASK_TIMEOUT_SEC}s)",
                      meta={"question": question[:200], "latency_ms": latency_ms})
            answer = f"查询处理超时（{_ASK_TIMEOUT_SEC}秒），请简化问题后重试"

        # OAI 格式无自定义字段，将消歧选项追加到回答末尾
        if rw.get("needs_clarification") and rw.get("clarification"):
            from clarification_engine import format_clarification_text
            clarify_text = format_clarification_text(rw["clarification"])
            answer = answer + "\n\n---\n" + clarify_text

        # 写入缓存
        if cache_key:
            _response_cache_put(cache_key, {"answer": answer})
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log_error("oai_chat", repr(e), meta={"question": question[:200], "latency_ms": latency_ms})
        answer = "接口执行异常，请稍后重试"

    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def _generate():
            # 按句子/标点分块输出，模拟逐句流式体验
            for chunk_text in _split_into_stream_chunks(answer):
                yield _build_oai_stream_chunk(chunk_text, req.model, chunk_id)
            yield _build_oai_stream_chunk("", req.model, chunk_id, finish=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    return _build_oai_response(answer, req.model)


@app.get("/v1/models")
def oai_models():
    """OpenAI 兼容模型列表接口"""
    return {
        "object": "list",
        "data": [
            {
                "id": _MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
            }
        ],
    }


# ===== 管理接口 =====

@app.get("/admin")
def admin_page():
    """管理后台页面"""
    if not ADMIN_PAGE.exists():
        raise HTTPException(status_code=404, detail="admin_page.html 不存在")
    return FileResponse(ADMIN_PAGE, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/mobile")
def mobile_page():
    """手机端知识库快捷导入页面"""
    if not MOBILE_PAGE.exists():
        raise HTTPException(status_code=404, detail="mobile.html 不存在")
    return FileResponse(MOBILE_PAGE, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/admin/products")
def admin_products():
    shared_names = _SHARED_DIR_NAMES
    products = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir() or p.name in shared_names:
                continue
            products.append({
                "product": p.name,
                "files": sorted([x.name for x in p.iterdir() if x.is_file()]),
            })
    return {"products": products}


@app.post("/admin/rebuild")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_rebuild(request: Request, req: RebuildRequest):
    global _health_cache
    from build_faiss import build_for_product
    from rag_logger import log_event
    product = req.product.strip()
    log_event("admin_rebuild", f"开始重建索引: {product}", meta={"timeout_sec": req.timeout_sec})
    # 安全校验：产品名不得包含路径分隔符或特殊字符（防止路径遍历）
    if "/" in product or "\\" in product or ".." in product or not product:
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 规范化路径并确认仍在 KNOWLEDGE_DIR 下（防止 symlink 逃逸）
    product_dir = (KNOWLEDGE_DIR / product).resolve()
    knowledge_root = KNOWLEDGE_DIR.resolve()
    if not str(product_dir).startswith(str(knowledge_root) + "/"):
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 校验产品目录确实存在于知识库中
    if not product_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"产品 '{product}' 不存在")
    # 获取产品级锁，防止并发重建同一产品
    with _rebuild_locks_guard:
        if product not in _rebuild_locks:
            _rebuild_locks[product] = threading.Lock()
        lock = _rebuild_locks[product]
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"产品 '{product}' 正在重建中，请稍后重试")
    try:
        # 使用线程 + 超时，让前端传入的 timeout_sec 生效
        build_error = [None]
        def _do_build():
            try:
                build_for_product(product)
            except Exception as exc:
                build_error[0] = exc
        t = threading.Thread(target=_do_build, daemon=True)
        t.start()
        t.join(timeout=req.timeout_sec)
        if t.is_alive():
            # 超时：线程仍在运行，但已超出用户指定时限
            log_error("admin_rebuild", f"索引重建超时 ({req.timeout_sec}s)",
                      meta={"product": product})
            raise HTTPException(status_code=504,
                                detail=f"索引重建超时（{req.timeout_sec}秒），后台可能仍在运行")
        if build_error[0] is not None:
            raise build_error[0]
        # 重建后清除缓存，下次请求会加载新索引
        invalidate_store_cache(product)
        from search_utils import invalidate_bm25_cache
        invalidate_bm25_cache(product)
        with _health_lock:
            _health_cache = {}  # 索引变更后清除健康检查缓存
        # 同时清除关联数据和媒体缓存
        from relation_engine import invalidate_relations_cache
        invalidate_relations_cache()
        invalidate_media_cache(product)
        log_event("admin_rebuild", f"索引重建完成: {product}")
        return {"ok": True, "product": product}
    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_rebuild", repr(e), meta={"product": product})
        raise HTTPException(status_code=500, detail="索引重建失败，请查看服务器日志")
    finally:
        if not t.is_alive():
            lock.release()
        else:
            # 线程超时仍在运行：安排后台释放锁，防止永久死锁
            def _deferred_release():
                t.join()
                lock.release()
            threading.Thread(target=_deferred_release, daemon=True).start()


@app.post("/admin/rebuild_shared")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_rebuild_shared(request: Request):
    """重建共享知识索引（procedures、equipment、anatomy 等）"""
    global _health_cache
    from build_faiss import build_shared
    try:
        build_shared()
        invalidate_store_cache("_shared")
        from search_utils import invalidate_bm25_cache
        invalidate_bm25_cache("_shared")
        with _health_lock:
            _health_cache = {}  # 索引变更后清除健康检查缓存
        from relation_engine import invalidate_relations_cache
        invalidate_relations_cache()
        return {"ok": True, "store": "_shared"}
    except Exception as e:
        log_error("admin_rebuild_shared", repr(e))
        raise HTTPException(status_code=500, detail="共享索引重建失败，请查看服务器日志")


# ===== 词库管理接口 =====

def _reload_synonym_runtime():
    """刷新运行时同义词扩展表，使增删改操作立即生效于检索。"""
    try:
        from search_utils import reload_learned_synonyms
        reload_learned_synonyms()
    except Exception as e:
        log_error("synonym_reload", repr(e))


@app.get("/admin/synonyms/all")
def admin_synonyms_all():
    """返回完整词库：静态同义词 + LLM 学习到的同义词"""
    from synonym_store import get_all_synonyms_combined
    data = get_all_synonyms_combined()
    # 附加运行时状态
    try:
        from search_utils import _LEARNED_SYNONYM_DIRECT, _learned_loaded
        data["runtime_active_count"] = len(_LEARNED_SYNONYM_DIRECT)
        data["runtime_loaded"] = _learned_loaded
    except Exception:
        data["runtime_active_count"] = 0
        data["runtime_loaded"] = False
    return data


@app.post("/admin/synonyms/reload")
def admin_synonyms_reload():
    """手动刷新运行时同义词扩展表"""
    try:
        from search_utils import reload_learned_synonyms
        count = reload_learned_synonyms()
        return {"ok": True, "active_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/synonyms/export")
def admin_synonyms_export():
    """导出全部词库为 JSON（供下载）"""
    from synonym_store import get_all_synonyms_combined
    return get_all_synonyms_combined()


@app.post("/admin/synonyms/learned/approve")
def admin_synonyms_approve(original: str):
    """审核通过一条学习到的同义词"""
    from synonym_store import approve_learned
    if not original or not original.strip():
        raise HTTPException(status_code=400, detail="original 不能为空")
    ok = approve_learned(original.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该同义词")
    _reload_synonym_runtime()
    return {"ok": True, "original": original.strip()}


@app.delete("/admin/synonyms/learned")
def admin_synonyms_delete(original: str):
    """删除一条学习到的同义词"""
    from synonym_store import delete_learned
    if not original or not original.strip():
        raise HTTPException(status_code=400, detail="original 不能为空")
    ok = delete_learned(original.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该同义词")
    _reload_synonym_runtime()
    return {"ok": True, "deleted": original.strip()}


class SynonymAddRequest(BaseModel):
    original: str = Field(..., min_length=1, max_length=200)
    mapped_to: str = Field(..., min_length=1, max_length=200)


@app.post("/admin/synonyms/learned/add")
def admin_synonyms_add(req: SynonymAddRequest):
    """手动添加一条同义词映射"""
    from synonym_store import add_manual
    result = add_manual(req.original, req.mapped_to)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "添加失败"))
    _reload_synonym_runtime()
    return result


class SynonymBatchRequest(BaseModel):
    terms: List[str]


@app.post("/admin/synonyms/learned/batch-approve")
def admin_synonyms_batch_approve(req: SynonymBatchRequest):
    """批量审核通过多条同义词"""
    from synonym_store import batch_approve
    result = batch_approve(req.terms)
    _reload_synonym_runtime()
    return result


@app.post("/admin/synonyms/learned/batch-delete")
def admin_synonyms_batch_delete(req: SynonymBatchRequest):
    """批量删除多条同义词"""
    from synonym_store import batch_delete
    result = batch_delete(req.terms)
    _reload_synonym_runtime()
    return result


class SynonymImportItem(BaseModel):
    original: str = Field(..., min_length=1, max_length=200)
    mapped_to: str = Field(..., min_length=1, max_length=200)


class SynonymBatchImportRequest(BaseModel):
    items: List[SynonymImportItem] = Field(..., max_length=500)
    auto_approve: bool = False


@app.post("/admin/synonyms/import")
def admin_synonyms_import(req: SynonymBatchImportRequest):
    """批量导入同义词（支持 JSON 数组，可选自动审批）"""
    from synonym_store import save_learned, approve_learned
    added = 0
    skipped = 0
    for item in req.items:
        orig = item.original.strip()
        mapped = item.mapped_to.strip()
        if not orig or not mapped or orig == mapped:
            skipped += 1
            continue
        save_learned(orig, mapped)
        if req.auto_approve:
            approve_learned(orig)
        added += 1
    if added > 0:
        _reload_synonym_runtime()
    return {"ok": True, "added": added, "skipped": skipped, "total": len(req.items)}


# ===== 关键词综合管理接口 =====


@app.get("/admin/synonyms/test")
def admin_synonyms_test(query: str = Query(..., min_length=1, max_length=200)):
    """测试同义词扩展效果：输入查询词，返回扩展后的所有关键词"""
    from search_utils import expand_synonyms, normalize_text
    query = query.strip()
    expanded_str = expand_synonyms(query)
    normalized = normalize_text(query)
    # expand_synonyms 返回字符串如 "原查询 同义词1 同义词2"
    # 提取原查询之外新增的扩展词
    expanded_terms = expanded_str.split()
    query_terms = set(query.split())
    new_terms = sorted(t for t in expanded_terms if t not in query_terms)
    return {
        "query": query,
        "normalized": normalized,
        "expanded_terms": new_terms,
        "expanded_count": len(new_terms),
    }


@app.get("/admin/keywords/effective")
def admin_keywords_effective():
    """返回最终生效的所有同义词（静态+覆盖+学习 合并后）"""
    from keyword_store import get_effective_synonyms
    items = get_effective_synonyms()
    sources = {"static": 0, "override": 0, "learned": 0}
    for it in items:
        s = it.get("source", "static")
        sources[s] = sources.get(s, 0) + 1
    return {"items": items, "total": len(items), "sources": sources}


class SynonymOverrideRequest(BaseModel):
    original: str = Field(..., min_length=1, max_length=200)
    mapped_to: str = Field(..., min_length=1, max_length=200)


@app.post("/admin/keywords/synonym/override")
def admin_keyword_synonym_override(req: SynonymOverrideRequest):
    """添加/编辑一条同义词（可覆盖静态表中的词条）"""
    from keyword_store import add_synonym_override
    result = add_synonym_override(req.original, req.mapped_to)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "操作失败"))
    _reload_synonym_runtime()
    return result


@app.delete("/admin/keywords/synonym/override")
def admin_keyword_synonym_delete(original: str = Query(..., min_length=1, max_length=200)):
    """删除一条同义词覆盖（对静态词标记为删除）"""
    from keyword_store import delete_synonym_override
    result = delete_synonym_override(original.strip())
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "操作失败"))
    _reload_synonym_runtime()
    return result


# ===== 消歧规则管理接口 =====


@app.get("/admin/keywords/clarification")
def admin_clarification_rules():
    """获取所有消歧规则（静态 + 自定义合并）"""
    from clarification_engine import _CLARIFICATION_RULES, _get_merged_rules
    merged = _get_merged_rules()
    from keyword_store import get_clarification_rules
    custom_keys = set(get_clarification_rules().keys())
    items = []
    for trigger, options in merged.items():
        items.append({
            "trigger": trigger,
            "options": options,
            "source": "custom" if trigger in custom_keys else "static",
        })
    items.sort(key=lambda x: x["trigger"])
    return {
        "items": items,
        "total": len(items),
        "static_count": sum(1 for it in items if it["source"] == "static"),
        "custom_count": sum(1 for it in items if it["source"] == "custom"),
    }


class ClarificationRuleRequest(BaseModel):
    trigger: str = Field(..., min_length=1, max_length=200)
    options: List[Dict[str, str]] = Field(..., max_length=10)


def _invalidate_clarification_cache():
    """清除消歧规则合并缓存，使新规则立即生效"""
    try:
        from clarification_engine import _get_merged_rules
        import clarification_engine
        clarification_engine._merged_rules_cache = None
        clarification_engine._merged_rules_ts = 0
    except Exception:
        pass


@app.post("/admin/keywords/clarification")
def admin_clarification_save(req: ClarificationRuleRequest):
    """保存一条消歧规则"""
    from keyword_store import save_clarification_rule
    result = save_clarification_rule(req.trigger, req.options)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "操作失败"))
    _invalidate_clarification_cache()
    return result


@app.delete("/admin/keywords/clarification")
def admin_clarification_delete(trigger: str = Query(..., min_length=1, max_length=200)):
    """删除一条自定义消歧规则"""
    from keyword_store import delete_clarification_rule
    result = delete_clarification_rule(trigger.strip())
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "操作失败"))
    _invalidate_clarification_cache()
    return result


# ===== LLM 智能扩充接口 =====


class LLMExpandRequest(BaseModel):
    category: str = "synonym"  # synonym | clarification
    count: int = Field(default=20, ge=5, le=50)


@app.post("/admin/keywords/llm-expand")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_llm_expand(request: Request, req: LLMExpandRequest):
    """调用 LLM 基于已有词库智能生成新词条"""
    from keyword_store import llm_expand_synonyms
    result = llm_expand_synonyms(category=req.category, count=req.count)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "扩充失败"))
    _reload_synonym_runtime()
    return result


@app.get("/admin/keywords/llm-expand/log")
def admin_llm_expand_log(limit: int = 20):
    """获取 LLM 扩充历史记录"""
    from keyword_store import get_llm_expansion_log
    return {"items": get_llm_expansion_log(limit=min(max(1, limit), 50))}


@app.get("/admin/logs/qa")
def admin_logs_qa(limit: int = 20):
    return {"items": get_recent_qa(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/miss")
def admin_logs_miss(limit: int = 20):
    return {"items": get_recent_misses(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/error")
def admin_logs_error(limit: int = 20):
    return {"items": get_recent_errors(limit=min(max(1, limit), 100))}


# ===== DOC/DOCX 文本提取 =====

def _extract_docx_text(file_data: bytes) -> str:
    """从 .docx 文件提取纯文本。

    提取段落文本和表格内容，保留换行结构。
    Raises ImportError if python-docx not installed.
    """
    import io
    from docx import Document
    doc = Document(io.BytesIO(file_data))
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    # 提取表格内容
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_doc_text(file_data: bytes) -> str:
    """从 .doc (旧格式) 文件提取纯文本。

    尝试多种方法：antiword → soffice → OLE2 Word Document 流 → 多编码正则。
    """
    import subprocess
    import tempfile

    tmp_path = None

    def _write_tmp() -> str:
        nonlocal tmp_path
        if tmp_path is None:
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
        return tmp_path

    def _cleanup_tmp():
        nonlocal tmp_path
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            tmp_path = None

    try:
        # 方法1：使用 antiword（如已安装），强制 UTF-8 输出
        try:
            p = _write_tmp()
            result = subprocess.run(
                ["antiword", "-m", "UTF-8", p],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        # 方法2：使用 LibreOffice 转换
        try:
            import tempfile as _tf
            p = _write_tmp()
            out_dir = _tf.mkdtemp()
            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                 "--outdir", out_dir, p],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0:
                txt_name = Path(p).stem + ".txt"
                txt_path = os.path.join(out_dir, txt_name)
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    import shutil
                    shutil.rmtree(out_dir, ignore_errors=True)
                    if text:
                        return text
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        # 方法3：OLE2 解析 — 读取 WordDocument / 1Table 流中的文本
        try:
            import olefile
            import io
            ole = olefile.OleFileIO(io.BytesIO(file_data))
            if ole.exists("WordDocument"):
                word_stream = ole.openstream("WordDocument").read()
                # 尝试多种编码解码 Word 二进制流中的文本
                for enc in ("utf-16-le", "gbk", "gb2312", "gb18030", "big5"):
                    try:
                        decoded = word_stream.decode(enc, errors="ignore")
                        import re as _re
                        parts = _re.findall(
                            r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
                            r'\w\s，。！？、；：""''（）《》【】\-·]{6,}',
                            decoded)
                        if parts:
                            text = "\n".join(p.strip() for p in parts if p.strip())
                            if len(text) > 50:
                                ole.close()
                                return text
                    except (UnicodeDecodeError, LookupError):
                        continue
            ole.close()
        except ImportError:
            pass
        except Exception:
            pass

        # 方法4：粗略提取二进制中的可见文本（多编码尝试）
        text_parts: list[str] = []
        import re as _re
        for enc in ("utf-16-le", "gbk", "gb2312", "gb18030", "utf-8"):
            try:
                raw = file_data.decode(enc, errors="ignore")
                parts = _re.findall(
                    r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
                    r'\w\s，。！？、；：""''（）《》【】\-·]{10,}',
                    raw)
                if parts:
                    candidate = "\n".join(p.strip() for p in parts if p.strip())
                    if len(candidate) > len("\n".join(text_parts)):
                        text_parts = [candidate]
            except (UnicodeDecodeError, LookupError):
                continue
        return "\n".join(text_parts)
    finally:
        _cleanup_tmp()


# ===== 知识库文件管理接口 =====

# 安全校验：产品名只允许字母数字下划线横线
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\u4e00-\u9fff]+$")
# 允许的文件扩展名
_ALLOWED_EXTENSIONS = {".txt", ".json", ".doc", ".docx"}


def _validate_product_name(name: str) -> str:
    """校验并清理产品名"""
    name = name.strip()
    if not name or not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="非法产品名称：只允许字母、数字、下划线、横线、中文")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 路径遍历防护
    product_dir = (KNOWLEDGE_DIR / name).resolve()
    if not product_dir.is_relative_to(KNOWLEDGE_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法产品名称")
    return name


@app.get("/admin/knowledge/{product}")
def admin_knowledge_files(product: str):
    """列出某产品的知识库文件"""
    product = _validate_product_name(product)
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise HTTPException(status_code=404, detail=f"产品 '{product}' 不存在")
    files = []
    for f in sorted(pdir.iterdir()):
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": int(stat.st_mtime),
                "editable": f.suffix in _ALLOWED_EXTENSIONS,
            })
    return {"product": product, "files": files}


@app.get("/admin/knowledge/{product}/{filename}")
def admin_knowledge_read(product: str, filename: str):
    """读取知识库文件内容"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    fpath = KNOWLEDGE_DIR / product / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    # 路径遍历二次防护
    if not fpath.resolve().is_relative_to(KNOWLEDGE_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法路径")
    try:
        content = fpath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = fpath.read_text(encoding="utf-8-sig", errors="replace")
    mtime = fpath.stat().st_mtime
    return {"product": product, "filename": filename, "content": content,
            "size": len(content), "mtime": mtime}


class KnowledgeWriteRequest(BaseModel):
    content: str = Field(..., min_length=0, max_length=5_000_000)  # 5MB 上限
    expected_mtime: Optional[float] = None  # 乐观锁：上次读取时的 mtime


@app.put("/admin/knowledge/{product}/{filename}")
def admin_knowledge_write(product: str, filename: str, req: KnowledgeWriteRequest):
    """写入/更新知识库文件内容"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不允许的文件类型: {suffix}")
    pdir = KNOWLEDGE_DIR / product
    pdir.mkdir(parents=True, exist_ok=True)
    fpath = pdir / filename
    # 路径遍历防护
    if not fpath.resolve().is_relative_to((KNOWLEDGE_DIR / product).resolve()):
        raise HTTPException(status_code=400, detail="非法路径")
    # 乐观锁：检测并发编辑冲突
    if req.expected_mtime is not None and fpath.exists():
        current_mtime = fpath.stat().st_mtime
        if abs(current_mtime - req.expected_mtime) > 0.001:
            raise HTTPException(
                status_code=409,
                detail="文件已被其他人修改，请刷新后重试")
    # 原子写入
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    try:
        tmp.write_text(req.content, encoding="utf-8")
        os.replace(str(tmp), str(fpath))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="写入失败，请查看服务器日志")
    new_mtime = fpath.stat().st_mtime
    return {"ok": True, "product": product, "filename": filename,
            "size": len(req.content), "mtime": new_mtime}


@app.delete("/admin/knowledge/{product}/{filename}")
def admin_knowledge_delete(product: str, filename: str):
    """删除知识库文件"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    fpath = KNOWLEDGE_DIR / product / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    if not fpath.resolve().is_relative_to(KNOWLEDGE_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法路径")
    fpath.unlink()
    # 文件删除后清除相关缓存，避免搜索返回已删除文件的内容
    invalidate_store_cache(product)
    from search_utils import invalidate_bm25_cache
    invalidate_bm25_cache(product)
    invalidate_media_cache(product)
    return {"ok": True, "deleted": filename}


class CreateProductRequest(BaseModel):
    product: str = Field(..., min_length=1, max_length=50)


@app.post("/admin/knowledge/create_product")
def admin_create_product(req: CreateProductRequest):
    """创建新产品目录"""
    product = _validate_product_name(req.product)
    pdir = KNOWLEDGE_DIR / product
    if pdir.exists():
        raise HTTPException(status_code=409, detail=f"产品 '{product}' 已存在")
    pdir.mkdir(parents=True, exist_ok=True)
    # 创建空的 main.txt
    (pdir / "main.txt").write_text("", encoding="utf-8")
    return {"ok": True, "product": product}


@app.delete("/admin/knowledge/{product}")
def admin_delete_product(product: str):
    """删除产品目录（含所有文件）"""
    product = _validate_product_name(product)
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise HTTPException(status_code=404, detail=f"产品 '{product}' 不存在")
    import shutil
    shutil.rmtree(pdir)
    # 清理对应的索引
    from rag_runtime_config import STORE_ROOT
    store_dir = STORE_ROOT / product
    if store_dir.exists():
        shutil.rmtree(store_dir)
    invalidate_store_cache(product)
    from search_utils import invalidate_bm25_cache
    invalidate_bm25_cache(product)
    invalidate_media_cache(product)
    with _health_lock:
        _health_cache.clear()
    return {"ok": True, "deleted": product}


# ===== 文件上传接口 =====

@app.post("/admin/upload")
@limiter.limit(_ADMIN_RATE_LIMIT)
async def admin_upload(request: "Request"):
    """通用文件上传：支持上传 txt/json/doc/docx 文件到指定产品目录。
    doc/docx 文件会自动转为 txt 存储。
    Form fields: product (str), files (UploadFile[])
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail="解析表单失败")
    try:
        product = str(form.get("product", "")).strip()
        if not product:
            raise HTTPException(status_code=400, detail="缺少 product 字段")
        product = _validate_product_name(product)
        pdir = KNOWLEDGE_DIR / product
        pdir.mkdir(parents=True, exist_ok=True)
        uploaded = []
        errors = []
        for key in form:
            if key == "product":
                continue
            item = form[key]
            # UploadFile 对象
            if hasattr(item, "filename") and hasattr(item, "read"):
                fname = item.filename or ""
                if ".." in fname or "/" in fname or "\\" in fname:
                    errors.append({"file": fname, "error": "非法文件名"})
                    continue
                suffix = Path(fname).suffix.lower()
                if suffix not in _ALLOWED_EXTENSIONS:
                    errors.append({"file": fname, "error": f"不允许的文件类型: {suffix}"})
                    continue
                try:
                    content = await item.read()
                    if suffix == ".docx":
                        try:
                            text = _extract_docx_text(content)
                        except ImportError:
                            errors.append({"file": fname, "error": "服务器未安装 python-docx，无法处理 .docx"})
                            continue
                        except Exception as exc:
                            errors.append({"file": fname, "error": f"解析 .docx 失败: {exc}"})
                            continue
                        # 转存为同名 .txt 文件
                        fname = Path(fname).stem + ".txt"
                    elif suffix == ".doc":
                        try:
                            text = _extract_doc_text(content)
                        except Exception as exc:
                            errors.append({"file": fname, "error": f"解析 .doc 失败: {exc}"})
                            continue
                        if not text.strip():
                            errors.append({"file": fname, "error": "无法从 .doc 提取文本，建议转为 .docx 后重试"})
                            continue
                        fname = Path(fname).stem + ".txt"
                    else:
                        text = content.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = content.decode("utf-8-sig")
                    except Exception:
                        text = content.decode("gbk", errors="replace")
                fpath = pdir / fname
                if not fpath.resolve().is_relative_to(pdir.resolve()):
                    errors.append({"file": fname, "error": "非法文件路径"})
                    continue
                # 原子写入
                tmp = fpath.with_suffix(fpath.suffix + ".tmp")
                try:
                    tmp.write_text(text, encoding="utf-8")
                    os.replace(str(tmp), str(fpath))
                except Exception:
                    tmp.unlink(missing_ok=True)
                    raise
                uploaded.append({"file": fname, "size": len(text)})
        return {"ok": True, "product": product, "uploaded": uploaded, "errors": errors}
    finally:
        await form.close()


# ===== 批量上传：ZIP 包解压 =====

@app.post("/admin/upload_zip")
@limiter.limit(_ADMIN_RATE_LIMIT)
async def admin_upload_zip(request: "Request"):
    """上传 ZIP 包，自动解压到知识库。
    ZIP 内部结构：product_name/main.txt, product_name/faq.txt 等
    """
    import zipfile
    import io
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")
    form = await request.form()
    try:
        results = []
        for key in form:
            item = form[key]
            if not hasattr(item, "read"):
                continue
            fname = getattr(item, "filename", "") or ""
            if not fname.lower().endswith(".zip"):
                results.append({"file": fname, "error": "只支持 .zip 文件"})
                continue
            data = await item.read()
            # ZIP 炸弹防护：限制 ZIP 文件大小（100MB）和最大条目数
            _ZIP_MAX_SIZE = 100 * 1024 * 1024  # 100MB
            _ZIP_MAX_ENTRIES = 500
            _ZIP_MAX_EXTRACTED = 200 * 1024 * 1024  # 解压后总大小上限 200MB
            if len(data) > _ZIP_MAX_SIZE:
                results.append({"file": fname, "error": f"ZIP 文件过大（超过 {_ZIP_MAX_SIZE // 1024 // 1024}MB）"})
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    entries = zf.infolist()
                    if len(entries) > _ZIP_MAX_ENTRIES:
                        results.append({"file": fname,
                            "error": f"ZIP 条目过多（{len(entries)} > {_ZIP_MAX_ENTRIES}）"})
                        continue
                    total_uncompressed = sum(e.file_size for e in entries)
                    if total_uncompressed > _ZIP_MAX_EXTRACTED:
                        results.append({"file": fname,
                            "error": f"ZIP 解压后过大（超过 {_ZIP_MAX_EXTRACTED // 1024 // 1024}MB）"})
                        continue
                    for info in entries:
                        if info.is_dir():
                            continue
                        # 安全校验路径
                        parts = Path(info.filename).parts
                        if len(parts) < 2:
                            continue
                        product_name = parts[0]
                        # 保留子目录结构：parts[1:] 为产品目录内的相对路径
                        relative_path = Path(*parts[1:])
                        file_name = relative_path.name
                        if ".." in info.filename:
                            continue
                        suffix = Path(file_name).suffix.lower()
                        if suffix not in _ALLOWED_EXTENSIONS:
                            continue
                        try:
                            product_name = _validate_product_name(product_name)
                        except Exception:
                            continue
                        pdir = KNOWLEDGE_DIR / product_name
                        # 创建子目录（如果有的话）
                        dest_dir = pdir / relative_path.parent
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        content = zf.read(info.filename)
                        try:
                            text = content.decode("utf-8")
                        except UnicodeDecodeError:
                            text = content.decode("utf-8-sig", errors="replace")
                        # 原子写入
                        dest = dest_dir / file_name
                        if not dest.resolve().is_relative_to(pdir.resolve()):
                            continue
                        tmp = dest.with_suffix(dest.suffix + ".tmp")
                        try:
                            tmp.write_text(text, encoding="utf-8")
                            os.replace(str(tmp), str(dest))
                        except Exception:
                            tmp.unlink(missing_ok=True)
                            raise
                        results.append({"product": product_name, "file": file_name,
                                        "size": len(text)})
            except zipfile.BadZipFile:
                results.append({"file": fname, "error": "无效的 ZIP 文件"})
        return {"ok": True, "results": results}
    finally:
        await form.close()


# ===== 调试模式切换 =====

@app.post("/admin/debug")
def admin_toggle_debug(request: Request, enable: bool = True):
    """切换调试模式日志级别"""
    from rag_logger import log_event
    level = logging.DEBUG if enable else logging.INFO
    old_level = logging.getLevelName(logger.level) if logger.level else "NOT_SET"
    logger.setLevel(level)
    logging.getLogger().setLevel(level)  # root logger too
    new_level = "DEBUG" if enable else "INFO"
    log_event("admin_debug", f"日志级别切换: {old_level} → {new_level}")
    return {"ok": True, "level": new_level, "previous_level": old_level}


# ===== 运行时配置接口 =====

@app.get("/admin/config")
def admin_get_config():
    """获取所有可调参数"""
    from rag_runtime_config import get_tunable_config
    return get_tunable_config()


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any]


@app.post("/admin/config")
def admin_update_config(req: ConfigUpdateRequest):
    """热更新运行时参数"""
    from rag_runtime_config import update_tunable_config
    changed = update_tunable_config(req.updates)
    return {"ok": True, "changed": changed}


@app.get("/admin/config/model")
def admin_get_model_config():
    """获取当前模型配置"""
    from rag_runtime_config import get_model_config
    return get_model_config()


class ModelSwitchRequest(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str = ""
    api_base: str = ""
    api_key: str = ""


@app.post("/admin/config/model")
def admin_switch_model(req: ModelSwitchRequest):
    """切换模型提供商"""
    from rag_runtime_config import switch_model_provider
    result = switch_model_provider(req.provider, req.model, req.api_base, req.api_key)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ===== 多 LLM 提供商配置接口 =====

def _get_knowledge_model_name() -> str:
    """获取知识库 LLM 模型名，用于 API 响应"""
    try:
        from llm_client import get_model as _gm, is_enabled as _ie
        if _ie("knowledge"):
            m = _gm("knowledge")
            if m:
                return m
    except ImportError:
        pass
    from rag_runtime_config import OPENAI_MODEL
    return OPENAI_MODEL


@app.get("/admin/llm/configs")
def admin_get_llm_configs():
    """获取所有用途的 LLM 配置（chat / knowledge）"""
    try:
        from llm_client import get_all_llm_configs
        from rag_runtime_config import MODEL_PRESETS
        return {
            "ok": True,
            "configs": get_all_llm_configs(),
            "presets": MODEL_PRESETS,
        }
    except ImportError:
        from rag_runtime_config import get_model_config, MODEL_PRESETS
        cfg = get_model_config()
        return {
            "ok": True,
            "configs": {"chat": cfg, "knowledge": cfg},
            "presets": MODEL_PRESETS,
        }


class MultiLLMUpdateRequest(BaseModel):
    purpose: str = Field(..., pattern="^(chat|knowledge)$")
    provider: str = ""
    model: str = ""
    api_base: Optional[str] = None   # None=不更新, ""=清空
    api_key: str = ""
    enabled: Optional[bool] = None
    model_format: str = ""           # "standard" or "litellm" (provider:model_id)


@app.post("/admin/llm/configs")
def admin_update_llm_config(req: MultiLLMUpdateRequest):
    """更新指定用途的 LLM 配置"""
    from rag_logger import log_event
    log_event("admin_llm_config", f"更新 LLM 配置: purpose={req.purpose}", meta={
        "provider": req.provider, "model": req.model,
        "api_base": req.api_base or "", "enabled": req.enabled,
    })
    from llm_client import update_llm_config
    result = update_llm_config(
        req.purpose,
        provider=req.provider,
        model=req.model,
        api_base=req.api_base,
        api_key=req.api_key,
        enabled=req.enabled,
        model_format=req.model_format,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/admin/llm/test")
def admin_llm_test_purpose(purpose: str = "chat"):
    """测试指定用途的 LLM 连接"""
    from llm_client import get_client, get_model, is_enabled, get_llm_config
    if purpose not in ("chat", "knowledge"):
        raise HTTPException(status_code=400, detail="purpose 必须为 chat 或 knowledge")

    if not is_enabled(purpose):
        return JSONResponse(status_code=503,
            content={"ok": False, "error": f"{purpose} LLM 未启用"})

    cfg = get_llm_config(purpose)
    if not cfg["api_key_set"]:
        return JSONResponse(status_code=503,
            content={"ok": False, "error": f"{purpose} LLM 未设置 API Key"})

    t0 = time.monotonic()
    try:
        client = get_client(purpose)
        if client is None:
            return JSONResponse(status_code=503,
                content={"ok": False, "error": "LLM client 创建失败"})
        model_name = get_model(purpose)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "请回复OK"}],
            max_tokens=10,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        reply = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        # 测试成功，标记连接已验证（持久化，不受服务控制影响）
        from llm_client import mark_connection_verified
        mark_connection_verified(purpose, True)
        return {
            "ok": True,
            "purpose": purpose,
            "provider": cfg["provider"],
            "model": model_name,
            "api_base": cfg["api_base"] or "(default)",
            "reply": reply,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        # 测试失败，标记连接未验证
        from llm_client import mark_connection_verified
        mark_connection_verified(purpose, False)
        return JSONResponse(status_code=502,
            content={"ok": False, "error": str(e), "latency_ms": latency_ms})


# ===== 服务器 / 域名配置接口 =====

@app.get("/admin/config/server")
def admin_get_server_config():
    """获取服务器和域名访问配置"""
    from rag_runtime_config import get_server_config
    return get_server_config()


class ServerConfigRequest(BaseModel):
    updates: Dict[str, Any]


@app.post("/admin/config/server")
def admin_update_server_config(req: ServerConfigRequest):
    """更新服务器和域名配置"""
    from rag_runtime_config import update_server_config
    changed = update_server_config(req.updates)
    return {"ok": True, "changed": changed}


# ===== BGE-M3 嵌入模型控制接口 =====

@app.get("/admin/service/embedding")
def admin_embedding_status():
    """获取 BGE-M3 嵌入模型状态"""
    from rag_runtime_config import get_embedding_status
    return get_embedding_status()


@app.post("/admin/service/embedding/start")
def admin_embedding_start():
    """加载 BGE-M3 嵌入模型"""
    from rag_logger import log_event
    log_event("admin_service", "请求加载嵌入模型")
    from rag_runtime_config import start_embedding_model
    result = start_embedding_model()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "启动失败"))
    return result


@app.post("/admin/service/embedding/stop")
def admin_embedding_stop():
    """卸载 BGE-M3 嵌入模型"""
    from rag_logger import log_event
    log_event("admin_service", "请求卸载嵌入模型")
    from rag_runtime_config import stop_embedding_model
    result = stop_embedding_model()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "停止失败"))
    return result


# ===== LLM 服务控制接口 =====

@app.get("/admin/service/llm")
def admin_llm_status():
    """获取 LLM 服务状态"""
    from rag_runtime_config import get_llm_status
    return get_llm_status()


class LLMStartRequest(BaseModel):
    api_key: str = ""


@app.post("/admin/service/llm/start")
def admin_llm_start(req: LLMStartRequest):
    """启动 LLM 服务"""
    from rag_logger import log_event
    log_event("admin_service", "请求启动 LLM 服务")
    from rag_runtime_config import start_llm_service
    result = start_llm_service(req.api_key)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "启动失败"))
    return result


@app.post("/admin/service/llm/stop")
def admin_llm_stop():
    """停止 LLM 服务"""
    from rag_logger import log_event
    log_event("admin_service", "请求停止 LLM 服务")
    from rag_runtime_config import stop_llm_service
    return stop_llm_service()


@app.post("/admin/service/llm/test")
def admin_llm_test():
    """测试 LLM API 连接（发送一次简短请求验证连通性）"""
    from rag_runtime_config import USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE
    if not USE_OPENAI:
        return JSONResponse(status_code=503,
            content={"ok": False, "error": "LLM 未启用"})
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return JSONResponse(status_code=503,
            content={"ok": False, "error": "未设置 OPENAI_API_KEY"})
    t0 = time.monotonic()
    try:
        from openai import OpenAI
        kwargs = {"api_key": key}
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "请回复OK"}],
            max_tokens=10,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        reply = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "api_base": OPENAI_API_BASE or "(default)",
            "reply": reply,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return JSONResponse(status_code=502,
            content={"ok": False, "error": str(e), "latency_ms": latency_ms})


# ===== 缓存管理接口 =====

@app.get("/admin/cache")
def admin_cache_stats():
    """查看缓存统计"""
    from rag_answer import _embed_cache, _store_cache
    llm_cache_stats = {}
    try:
        from query_rewrite import _llm_rewrite_cache
        llm_cache_stats = _llm_rewrite_cache.stats
    except Exception as e:
        llm_cache_stats = {"error": str(e)}
    return {
        "response_cache": {
            "size": len(_RESPONSE_CACHE),
            "max_size": _RESPONSE_CACHE_MAX,
            "ttl_seconds": _RESPONSE_CACHE_TTL,
        },
        "embed_cache": {
            "size": len(_embed_cache),
            "max_size": 1024,
        },
        "store_cache": {
            "products_loaded": list(_store_cache.keys()),
        },
        "llm_rewrite_cache": llm_cache_stats,
    }


@app.post("/admin/cache/clear")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_cache_clear(request: Request):
    """清空所有缓存（响应缓存、嵌入缓存、索引缓存、LLM 改写缓存）"""
    from rag_logger import log_event
    log_event("admin_cache", "请求清空所有缓存")
    from rag_answer import _embed_cache, _store_cache
    cleared = {}
    with _response_cache_lock:
        cleared["response_cache"] = len(_RESPONSE_CACHE)
        _RESPONSE_CACHE.clear()
    cleared["embed_cache"] = len(_embed_cache)
    _embed_cache.clear()
    cleared["store_cache"] = len(_store_cache)
    _store_cache.clear()
    try:
        from query_rewrite import _llm_rewrite_cache
        with _llm_rewrite_cache._lock:
            cleared["llm_rewrite_cache"] = len(_llm_rewrite_cache._cache)
            _llm_rewrite_cache._cache.clear()
    except Exception:
        pass
    return {"ok": True, "cleared": cleared}


# ===== 系统完整运行状态 =====

@app.get("/admin/system_status")
def admin_system_status():
    """获取系统完整运行状态诊断信息，用于调试排查问题。

    返回所有子系统的健康状况、配置、版本信息。
    """
    import sys
    import platform
    from datetime import datetime as _dt
    status = {
        "timestamp": _dt.now().isoformat(timespec="seconds"),
        "uptime_seconds": int(time.monotonic() - _startup_time) if _startup_time else 0,
        "python": {
            "version": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "subsystems": {},
        "config": {},
        "errors": [],
    }

    # ---- 1. 嵌入模型状态 ----
    try:
        from rag_answer import _model
        model_loaded = _model is not None
        model_name = ""
        if _model is not None:
            model_name = getattr(_model, "model_card_data", {}) or ""
            # 尝试获取模型路径
            try:
                from rag_answer import EMBED_MODEL_NAME
                model_name = EMBED_MODEL_NAME
            except ImportError:
                pass
        status["subsystems"]["embedding"] = {
            "status": "ok" if model_loaded else "not_loaded",
            "model_loaded": model_loaded,
            "model_name": str(model_name),
        }
    except Exception as e:
        status["subsystems"]["embedding"] = {"status": "error", "error": str(e)}
        status["errors"].append(f"embedding: {e}")

    # ---- 2. LLM 配置状态（多用途） ----
    try:
        from llm_client import get_all_llm_configs, is_enabled, get_client
        configs = get_all_llm_configs()
        llm_status = {}
        for purpose in ("chat", "knowledge"):
            cfg = configs.get(purpose, {})
            enabled = is_enabled(purpose)
            client_ok = False
            if enabled:
                try:
                    c = get_client(purpose)
                    client_ok = c is not None
                except Exception:
                    pass
            llm_status[purpose] = {
                "enabled": enabled,
                "provider": cfg.get("provider", ""),
                "model": cfg.get("model", ""),
                "api_base": cfg.get("api_base", ""),
                "api_key_set": cfg.get("api_key_set", False),
                "client_created": client_ok,
                "connection_verified": cfg.get("connection_verified", False),
            }
        status["subsystems"]["llm"] = {"status": "ok", "purposes": llm_status}
    except ImportError:
        # 无 llm_client 模块，回退到旧配置
        try:
            from rag_runtime_config import USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE
            status["subsystems"]["llm"] = {
                "status": "legacy",
                "enabled": USE_OPENAI,
                "model": OPENAI_MODEL,
                "api_base": OPENAI_API_BASE,
                "api_key_set": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            }
        except Exception as e:
            status["subsystems"]["llm"] = {"status": "error", "error": str(e)}
            status["errors"].append(f"llm: {e}")

    # ---- 3. 知识库状态 ----
    try:
        from rag_runtime_config import STORE_ROOT
        products = []
        if KNOWLEDGE_DIR.exists():
            shared_names = _SHARED_DIR_NAMES
            for p in sorted(KNOWLEDGE_DIR.iterdir()):
                if not p.is_dir() or p.name in shared_names:
                    continue
                store = STORE_ROOT / p.name
                index_ok = (store / "index.faiss").exists()
                docs_ok = (store / "docs.jsonl").exists()
                doc_count = 0
                file_count = sum(1 for f in p.iterdir() if f.is_file())
                if docs_ok:
                    try:
                        doc_count = sum(1 for _ in (store / "docs.jsonl").open("r", encoding="utf-8"))
                    except OSError:
                        pass
                products.append({
                    "name": p.name,
                    "knowledge_files": file_count,
                    "index_built": index_ok,
                    "docs_indexed": doc_count,
                })
        shared_store = STORE_ROOT / "_shared"
        shared_ok = (shared_store / "index.faiss").exists()
        shared_docs = 0
        if shared_ok:
            try:
                shared_docs = sum(1 for _ in (shared_store / "docs.jsonl").open("r", encoding="utf-8"))
            except OSError:
                pass
        status["subsystems"]["knowledge"] = {
            "status": "ok" if products else "empty",
            "products": products,
            "shared_index_built": shared_ok,
            "shared_docs": shared_docs,
            "knowledge_dir": str(KNOWLEDGE_DIR),
            "store_dir": str(STORE_ROOT),
        }
    except Exception as e:
        status["subsystems"]["knowledge"] = {"status": "error", "error": str(e)}
        status["errors"].append(f"knowledge: {e}")

    # ---- 4. 缓存状态 ----
    try:
        from rag_answer import _embed_cache, _store_cache
        cache_info = {
            "response_cache_size": len(_RESPONSE_CACHE),
            "response_cache_max": _RESPONSE_CACHE_MAX,
            "embed_cache_size": len(_embed_cache),
            "store_cache_products": list(_store_cache.keys()),
        }
        try:
            from query_rewrite import _llm_rewrite_cache
            cache_info["llm_rewrite_cache"] = _llm_rewrite_cache.stats
        except Exception:
            cache_info["llm_rewrite_cache"] = "unavailable"
        status["subsystems"]["cache"] = cache_info
    except Exception as e:
        status["subsystems"]["cache"] = {"error": str(e)}

    # ---- 5. 同义词系统 ----
    try:
        from synonym_store import get_all_synonyms_combined
        syn = get_all_synonyms_combined()
        status["subsystems"]["synonyms"] = {
            "static_count": len(syn.get("static", {})),
            "learned_count": len(syn.get("learned", {})),
        }
    except Exception as e:
        status["subsystems"]["synonyms"] = {"error": str(e)}

    # ---- 6. 日志状态 ----
    try:
        from rag_logger import QA_LOG, MISS_LOG, ERROR_LOG, EVENT_LOG, LOG_DIR
        log_info = {}
        for name, path in [("qa", QA_LOG), ("miss", MISS_LOG), ("error", ERROR_LOG), ("event", EVENT_LOG)]:
            if path.exists():
                st = path.stat()
                log_info[name] = {"size_bytes": st.st_size, "exists": True}
            else:
                log_info[name] = {"size_bytes": 0, "exists": False}
        status["subsystems"]["logs"] = {
            "log_dir": str(LOG_DIR),
            "files": log_info,
        }
    except Exception as e:
        status["subsystems"]["logs"] = {"error": str(e)}

    # ---- 7. 运行时配置 ----
    try:
        from rag_runtime_config import get_tunable_config
        status["config"]["tunable"] = get_tunable_config()
    except Exception as e:
        status["config"]["tunable"] = {"error": str(e)}

    # ---- 8. 线程池状态 ----
    try:
        from rag_answer import _search_pool
        pool = _search_pool
        status["subsystems"]["thread_pool"] = {
            "max_workers": getattr(pool, "_max_workers", "unknown"),
            "threads_alive": len([t for t in getattr(pool, "_threads", []) if t.is_alive()]),
        }
    except Exception as e:
        status["subsystems"]["thread_pool"] = {"error": str(e)}

    # ---- 9. 管理鉴权状态 ----
    status["subsystems"]["auth"] = {
        "admin_api_key_set": bool(_ADMIN_API_KEY),
    }

    # ---- 10. 速率限制配置 ----
    status["config"]["rate_limits"] = {
        "ask": _ASK_RATE_LIMIT,
        "admin": _ADMIN_RATE_LIMIT,
        "ask_timeout_sec": _ASK_TIMEOUT_SEC,
    }

    # ---- 11. 日志级别 ----
    status["config"]["log_level"] = logging.getLevelName(logger.level) if logger.level else "NOT_SET"
    status["config"]["root_log_level"] = logging.getLevelName(logging.getLogger().level)

    # ---- 12. 最近错误摘要 ----
    try:
        recent_errors = get_recent_errors(limit=5)
        status["recent_errors"] = recent_errors
    except Exception:
        status["recent_errors"] = []

    # 总体状态判断
    has_errors = bool(status["errors"])
    emb_ok = status["subsystems"].get("embedding", {}).get("model_loaded", False)
    llm_ok = False
    llm_sub = status["subsystems"].get("llm", {})
    if "purposes" in llm_sub:
        llm_ok = any(p.get("enabled") for p in llm_sub["purposes"].values())
    elif llm_sub.get("enabled"):
        llm_ok = True
    knowledge_ok = bool(status["subsystems"].get("knowledge", {}).get("products"))

    if has_errors:
        overall = "error"
    elif emb_ok and llm_ok and knowledge_ok:
        overall = "healthy"
    elif emb_ok and knowledge_ok:
        overall = "degraded_no_llm"
    elif emb_ok:
        overall = "degraded_no_knowledge"
    else:
        overall = "degraded"

    status["overall_status"] = overall
    return status


# ===== 系统统计接口 =====

@app.get("/admin/stats")
def admin_stats():
    """获取系统统计信息"""
    from rag_runtime_config import STORE_ROOT
    products_info = []
    if KNOWLEDGE_DIR.exists():
        shared_names = _SHARED_DIR_NAMES
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir() or p.name in shared_names:
                continue
            info = {"name": p.name, "files": [], "total_size": 0}
            for f in p.iterdir():
                if f.is_file():
                    size = f.stat().st_size
                    info["files"].append({"name": f.name, "size": size})
                    info["total_size"] += size
            # 索引状态
            store = STORE_ROOT / p.name
            info["index_exists"] = (store / "index.faiss").exists()
            info["docs_count"] = 0
            docs_path = store / "docs.jsonl"
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    info["docs_count"] = sum(1 for _ in f)
            products_info.append(info)
    # 共享知识
    shared_store = STORE_ROOT / "_shared"
    shared_docs = 0
    if (shared_store / "docs.jsonl").exists():
        with open(shared_store / "docs.jsonl", "r", encoding="utf-8") as f:
            shared_docs = sum(1 for _ in f)
    return {
        "products": products_info,
        "shared_docs": shared_docs,
        "shared_indexed": (shared_store / "index.faiss").exists(),
    }


# ===== 知识库导入接口（LLM 自动整理） =====

# 导入锁，防止并发导入
_import_locks: Dict[str, threading.Lock] = {}
_import_locks_guard = threading.Lock()


class ImportKnowledgeRequest(BaseModel):
    """通过文本内容导入知识库"""
    type: str = Field(..., description="知识类型: product/procedure/equipment/material/anatomy/indication/complication/course/script")
    id: str = Field(default="", description="实体ID（目录名），product/procedure/equipment/material 必填")
    content: str = Field(..., min_length=10, description="原始文档文本内容")
    build: bool = Field(default=True, description="导入后自动构建索引")
    dry_run: bool = Field(default=False, description="仅预览，不写入文件")


@app.post("/admin/import_knowledge")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_import_knowledge(request: Request, req: ImportKnowledgeRequest):
    """通过 LLM 自动将原始文档整理为结构化知识库文件。

    用法：POST JSON 请求体，content 字段放原始文档文本。
    LLM 会自动整理为 main.txt / faq.txt / alias.txt 等结构化文件。
    """
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, _generate_knowledge,
        _write_knowledge_files,
    )

    entity_type = req.type.strip()
    entity_id = req.id.strip()

    # 校验类型
    if entity_type not in _ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的类型: {entity_type}，可选: {', '.join(_ENTITY_TYPES.keys())}")

    # 校验 ID
    _, is_single = _ENTITY_TYPES[entity_type]
    if not is_single and not entity_id:
        raise HTTPException(
            status_code=400,
            detail=f"类型 '{entity_type}' 需要提供 id 参数")

    # 安全校验 ID
    if entity_id:
        if ".." in entity_id or "/" in entity_id or "\\" in entity_id:
            raise HTTPException(status_code=400, detail="非法 ID")
        if not _SAFE_NAME_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="ID 只允许字母、数字、下划线、横线、中文")

    raw_text = req.content.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="文档内容不能为空")

    logger.debug(f"import_knowledge: type={entity_type}, id={entity_id}, content_length={len(raw_text)}, dry_run={getattr(req, 'dry_run', False)}")

    # 获取导入锁
    lock_key = f"{entity_type}:{entity_id or '_single'}"
    with _import_locks_guard:
        if lock_key not in _import_locks:
            _import_locks[lock_key] = threading.Lock()
        lock = _import_locks[lock_key]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="该知识正在导入中，请稍后重试")

    try:
        # 检查 LLM 是否可用（优先检查知识库专用 LLM，回退到全局）
        try:
            from llm_client import is_enabled as _llm_is_enabled
            knowledge_llm_ok = _llm_is_enabled("knowledge")
        except ImportError:
            knowledge_llm_ok = False
        if not knowledge_llm_ok:
            from rag_runtime_config import USE_OPENAI
            if not USE_OPENAI:
                raise HTTPException(
                    status_code=400,
                    detail="知识库 LLM 未启用，请先在管理后台「模型切换」中配置知识库整理用的 LLM")
        from rag_runtime_config import OPENAI_MODEL

        # 调用 LLM 整理
        client = _get_openai_client()
        result = _generate_knowledge(client, raw_text, entity_type, entity_id)

        # 写入文件
        out_dir = _write_knowledge_files(result, entity_type, entity_id,
                                          dry_run=req.dry_run)

        # 构建索引
        built_index = False
        if req.build and not req.dry_run:
            try:
                if entity_type == "product":
                    from build_faiss import build_for_product
                    build_for_product(entity_id)
                    invalidate_store_cache(entity_id)
                else:
                    from build_faiss import build_shared
                    build_shared()
                    invalidate_store_cache("_shared")
                # 清除健康检查缓存
                with _health_lock:
                    global _health_cache
                    _health_cache = {}
                built_index = True
            except Exception as e:
                log_error("import_build_index", repr(e),
                          meta={"type": entity_type, "id": entity_id})

        # 构建响应：索引构建失败时明确告知
        response = {
            "ok": True if (not req.build or req.dry_run or built_index) else False,
            "type": entity_type,
            "id": entity_id,
            "dry_run": req.dry_run,
            "output_dir": str(out_dir),
            "built_index": built_index,
            "model": _get_knowledge_model_name(),
            "files_generated": {},
        }
        if req.build and not req.dry_run and not built_index:
            response["warning"] = "知识文件已写入，但索引构建失败，查询可能返回旧结果。请手动重建索引。"
        if result.get("main_txt"):
            response["files_generated"]["main.txt"] = len(result["main_txt"])
        if result.get("faq_txt"):
            response["files_generated"]["faq.txt"] = len(result["faq_txt"])
        if result.get("alias_txt"):
            response["files_generated"]["alias.txt"] = len(result["alias_txt"])

        # 别名注册提示
        aliases = result.get("product_aliases") or result.get("entity_aliases") or []
        name = result.get("product_name") or result.get("entity_name") or entity_id
        if aliases:
            alias_config_key = {
                "product": "PRODUCT_ALIASES",
                "procedure": "PROCEDURE_ALIASES",
                "equipment": "EQUIPMENT_ALIASES",
                "material": "MATERIAL_ALIASES",
            }.get(entity_type)
            if alias_config_key:
                response["alias_hint"] = {
                    "config_key": alias_config_key,
                    "entity_id": entity_id,
                    "name": name,
                    "aliases": aliases,
                }

        # dry_run 时返回预览内容
        if req.dry_run:
            response["preview"] = {}
            if result.get("main_txt"):
                txt = result["main_txt"]
                response["preview"]["main_txt"] = txt[:3000] + (f"\n...(省略 {len(txt)-3000} 字)" if len(txt) > 3000 else "")
            if result.get("faq_txt"):
                txt = result["faq_txt"]
                response["preview"]["faq_txt"] = txt[:2000] + (f"\n...(省略 {len(txt)-2000} 字)" if len(txt) > 2000 else "")
            if result.get("alias_txt"):
                response["preview"]["alias_txt"] = result["alias_txt"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_import_knowledge", repr(e),
                  meta={"type": entity_type, "id": entity_id})
        raise HTTPException(status_code=500, detail="导入失败，请查看服务器日志")
    finally:
        lock.release()


class RefineKnowledgeRequest(BaseModel):
    """修订知识库内容"""
    type: str = Field(..., description="知识类型")
    id: str = Field(default="", description="实体ID")
    feedback: str = Field(..., min_length=2, description="用户修改意见")
    current: dict = Field(..., description="当前整理结果，包含 main_txt/faq_txt/alias_txt")
    raw_text: str = Field(default="", description="可选，原始文档供 LLM 参考")


class CommitKnowledgeRequest(BaseModel):
    """正式写入知识库"""
    type: str = Field(..., description="知识类型")
    id: str = Field(default="", description="实体ID")
    content: dict = Field(..., description="最终确认的内容，包含 main_txt/faq_txt/alias_txt")
    build: bool = Field(default=True, description="是否自动构建索引")


@app.post("/admin/import_knowledge/refine")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_import_knowledge_refine(request: Request, req: RefineKnowledgeRequest):
    """根据用户反馈修订知识库内容（LLM 修订）。"""
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, refine_knowledge,
    )

    entity_type = req.type.strip()
    entity_id = req.id.strip()

    if entity_type not in _ENTITY_TYPES:
        raise HTTPException(status_code=400,
            detail=f"不支持的类型: {entity_type}")

    _, is_single = _ENTITY_TYPES[entity_type]
    if not is_single and not entity_id:
        raise HTTPException(status_code=400,
            detail=f"类型 '{entity_type}' 需要提供 id 参数")

    if entity_id:
        if ".." in entity_id or "/" in entity_id or "\\" in entity_id:
            raise HTTPException(status_code=400, detail="非法 ID")
        if not _SAFE_NAME_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="ID 只允许字母、数字、下划线、横线、中文")

    lock_key = f"refine:{entity_type}:{entity_id or '_single'}"
    with _import_locks_guard:
        if lock_key not in _import_locks:
            _import_locks[lock_key] = threading.Lock()
        lock = _import_locks[lock_key]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="正在修订中，请稍后重试")

    try:
        # 检查知识库 LLM 是否可用
        try:
            from llm_client import is_enabled as _llm_is_enabled
            knowledge_llm_ok = _llm_is_enabled("knowledge")
        except ImportError:
            knowledge_llm_ok = False
        if not knowledge_llm_ok:
            from rag_runtime_config import USE_OPENAI
            if not USE_OPENAI:
                raise HTTPException(status_code=400,
                    detail="知识库 LLM 未启用，请先在管理后台配置知识库整理用的 LLM")

        client = _get_openai_client()
        result = refine_knowledge(
            client,
            current=req.current,
            feedback=req.feedback,
            raw_text=req.raw_text,
            entity_type=entity_type,
        )

        response = {
            "ok": True,
            "type": entity_type,
            "id": entity_id,
            "model": _get_knowledge_model_name(),
            "preview": {},
            "files_generated": {},
        }

        if result.get("main_txt"):
            response["preview"]["main_txt"] = result["main_txt"]
            response["files_generated"]["main.txt"] = len(result["main_txt"])
        if result.get("faq_txt"):
            response["preview"]["faq_txt"] = result["faq_txt"]
            response["files_generated"]["faq.txt"] = len(result["faq_txt"])
        if result.get("alias_txt"):
            response["preview"]["alias_txt"] = result["alias_txt"]
            response["files_generated"]["alias.txt"] = len(result["alias_txt"])

        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_import_refine", repr(e),
                  meta={"type": entity_type, "id": entity_id})
        raise HTTPException(status_code=500, detail="修订失败，请查看服务器日志")
    finally:
        lock.release()


@app.post("/admin/import_knowledge/commit")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_import_knowledge_commit(request: Request, req: CommitKnowledgeRequest):
    """将审阅确认的内容正式写入知识库文件并建索引。"""
    from import_knowledge import (
        _ENTITY_TYPES, _write_knowledge_files,
    )

    entity_type = req.type.strip()
    entity_id = req.id.strip()

    if entity_type not in _ENTITY_TYPES:
        raise HTTPException(status_code=400,
            detail=f"不支持的类型: {entity_type}")

    _, is_single = _ENTITY_TYPES[entity_type]
    if not is_single and not entity_id:
        raise HTTPException(status_code=400,
            detail=f"类型 '{entity_type}' 需要提供 id 参数")

    if entity_id:
        if ".." in entity_id or "/" in entity_id or "\\" in entity_id:
            raise HTTPException(status_code=400, detail="非法 ID")
        if not _SAFE_NAME_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="ID 只允许字母、数字、下划线、横线、中文")

    content = req.content
    if not content.get("main_txt"):
        raise HTTPException(status_code=400, detail="内容不能为空（至少需要 main_txt）")

    lock_key = f"commit:{entity_type}:{entity_id or '_single'}"
    with _import_locks_guard:
        if lock_key not in _import_locks:
            _import_locks[lock_key] = threading.Lock()
        lock = _import_locks[lock_key]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="正在写入中，请稍后重试")

    try:
        out_dir = _write_knowledge_files(content, entity_type, entity_id,
                                          dry_run=False)

        built_index = False
        if req.build:
            try:
                from search_utils import invalidate_bm25_cache
                if entity_type == "product":
                    from build_faiss import build_for_product
                    build_for_product(entity_id)
                    invalidate_store_cache(entity_id)
                    invalidate_bm25_cache(entity_id)
                    invalidate_media_cache(entity_id)
                else:
                    from build_faiss import build_shared
                    build_shared()
                    invalidate_store_cache("_shared")
                    invalidate_bm25_cache("_shared")
                with _health_lock:
                    global _health_cache
                    _health_cache = {}
                built_index = True
            except Exception as e:
                log_error("commit_build_index", repr(e),
                          meta={"type": entity_type, "id": entity_id})

        return {
            "ok": True,
            "type": entity_type,
            "id": entity_id,
            "output_dir": str(out_dir),
            "built_index": built_index,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_import_commit", repr(e),
                  meta={"type": entity_type, "id": entity_id})
        raise HTTPException(status_code=500, detail="写入失败，请查看服务器日志")
    finally:
        lock.release()


@app.post("/admin/extract_file_text")
@limiter.limit(_ADMIN_RATE_LIMIT)
async def admin_extract_file_text(request: Request):
    """从上传文件中提取纯文本（不导入，仅提取）。

    支持 .txt .md .pdf .doc .docx，返回提取后的文本供前端预览编辑。
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")

    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"解析表单失败: {exc}")

    try:
        file_item = form.get("file")
        if not file_item or not hasattr(file_item, "read"):
            raise HTTPException(status_code=400, detail="缺少 file 文件")

        fname = getattr(file_item, "filename", "") or "upload.txt"
        suffix = Path(fname).suffix.lower()
        allowed = {".txt", ".md", ".pdf", ".doc", ".docx"}
        if suffix not in allowed:
            return JSONResponse(content={
                "ok": False, "detail": f"不支持的文件类型: {suffix}，支持: {', '.join(sorted(allowed))}"
            })

        file_data = await file_item.read()
        if not file_data:
            return JSONResponse(content={"ok": False, "detail": "文件内容为空"})

        raw_text = ""
        if suffix == ".pdf":
            try:
                import pdfplumber
                import io
                text_parts = []
                with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                raw_text = "\n\n".join(text_parts)
            except ImportError:
                return JSONResponse(content={"ok": False, "detail": "服务器未安装 pdfplumber，无法处理 PDF"})
        elif suffix == ".docx":
            try:
                raw_text = _extract_docx_text(file_data)
            except ImportError:
                return JSONResponse(content={"ok": False, "detail": "服务器未安装 python-docx，无法处理 .docx"})
            except Exception as exc:
                return JSONResponse(content={"ok": False, "detail": f"解析 .docx 失败: {exc}"})
        elif suffix == ".doc":
            try:
                raw_text = _extract_doc_text(file_data)
            except Exception as exc:
                return JSONResponse(content={"ok": False, "detail": f"解析 .doc 失败: {exc}"})
            if not raw_text.strip():
                return JSONResponse(content={"ok": False, "detail": "无法从 .doc 提取文本，建议转为 .docx 后重试"})
        else:
            # txt / md
            for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
                try:
                    raw_text = file_data.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                raw_text = file_data.decode("utf-8", errors="replace")

        if not raw_text.strip():
            return JSONResponse(content={"ok": False, "detail": "提取到的文本内容为空"})

        return {
            "ok": True,
            "filename": fname,
            "format": suffix,
            "text": raw_text,
            "length": len(raw_text),
        }
    finally:
        await form.close()


@app.post("/admin/import_knowledge_file")
async def admin_import_knowledge_file(request: "Request"):
    """通过文件上传导入知识库（LLM 自动整理）。

    Form fields:
        - type: 知识类型 (product/procedure/equipment/material/...)
        - id: 实体ID（product/procedure/equipment/material 必填）
        - build: "1" 导入后自动建索引（默认 "1"）
        - dry_run: "1" 仅预览（默认 "0"）
        - file: 上传的文件（支持 .txt .md .pdf .doc .docx）
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")

    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail="解析表单失败")

    try:
        entity_type = str(form.get("type", "")).strip()
        entity_id = str(form.get("id", "")).strip()
        build = str(form.get("build", "1")).strip().lower() in ("1", "true", "yes")
        dry_run = str(form.get("dry_run", "0")).strip().lower() in ("1", "true", "yes")

        if not entity_type:
            raise HTTPException(status_code=400, detail="缺少 type 字段")

        # 读取上传的文件
        file_item = form.get("file")
        if not file_item or not hasattr(file_item, "read"):
            raise HTTPException(status_code=400, detail="缺少 file 文件")

        fname = getattr(file_item, "filename", "") or "upload.txt"
        suffix = Path(fname).suffix.lower()

        file_data = await file_item.read()

        if suffix == ".pdf":
            try:
                import pdfplumber
                import io
                text_parts = []
                with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                raw_text = "\n\n".join(text_parts)
            except ImportError:
                raise HTTPException(status_code=400, detail="服务器未安装 pdfplumber，无法处理 PDF")
        elif suffix == ".docx":
            try:
                raw_text = _extract_docx_text(file_data)
            except ImportError:
                raise HTTPException(status_code=400, detail="服务器未安装 python-docx，无法处理 .docx")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"解析 .docx 失败: {exc}")
        elif suffix == ".doc":
            try:
                raw_text = _extract_doc_text(file_data)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"解析 .doc 失败: {exc}")
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="无法从 .doc 提取文本，建议转为 .docx 后重试")
        else:
            # txt / md 等文本文件
            for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
                try:
                    raw_text = file_data.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                raw_text = file_data.decode("utf-8", errors="replace")

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="上传的文件内容为空")

        # 构造请求，复用 JSON 接口逻辑
        import_req = ImportKnowledgeRequest(
            type=entity_type,
            id=entity_id,
            content=raw_text,
            build=build,
            dry_run=dry_run,
        )
        return admin_import_knowledge(request, import_req)
    finally:
        await form.close()


def _extract_article_text(html: str):
    """从 HTML 中提取文章标题和正文纯文本。

    优先使用 BeautifulSoup（如已安装），否则用正则做基础提取。
    """
    title = ""
    text = ""

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # 提取标题
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.title:
            title = soup.title.get_text(strip=True)

        # 移除不需要的标签
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # 微信文章特有结构
        article = soup.find(id="js_content") or soup.find(class_="rich_media_content")
        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            # 通用提取：找最大的文本块
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)

    except ImportError:
        # 无 BeautifulSoup，用正则基础提取
        import re as _re

        # 标题
        m = _re.search(r'<title[^>]*>(.*?)</title>', html, _re.DOTALL | _re.IGNORECASE)
        if m:
            title = m.group(1).strip()

        # og:title
        m = _re.search(r'property="og:title"\s+content="([^"]*)"', html)
        if m:
            title = m.group(1).strip()

        # 微信正文区域
        m = _re.search(r'id="js_content"[^>]*>(.*?)</div>\s*</div>', html, _re.DOTALL)
        if m:
            raw = m.group(1)
        else:
            raw = _re.sub(r'<script[^>]*>.*?</script>', '', html, flags=_re.DOTALL | _re.IGNORECASE)
            raw = _re.sub(r'<style[^>]*>.*?</style>', '', raw, flags=_re.DOTALL | _re.IGNORECASE)

        # 去除 HTML 标签
        raw = _re.sub(r'<br\s*/?>', '\n', raw, flags=_re.IGNORECASE)
        raw = _re.sub(r'<p[^>]*>', '\n', raw, flags=_re.IGNORECASE)
        raw = _re.sub(r'<[^>]+>', '', raw)
        # 去除 HTML 实体
        raw = _re.sub(r'&nbsp;', ' ', raw)
        raw = _re.sub(r'&[a-zA-Z]+;', '', raw)
        text = _re.sub(r'\n{3,}', '\n\n', raw).strip()

    # 清理多余空行
    lines = [l.strip() for l in text.split("\n")]
    text = re.sub(r'\n{3,}', '\n\n', "\n".join(lines)).strip()

    return title, text


class FetchUrlRequest(BaseModel):
    """通过 URL 抓取网页正文"""
    url: str = Field(..., description="要抓取的网页 URL")


@app.post("/admin/fetch_url")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_fetch_url(request: Request, req: FetchUrlRequest):
    """抓取网页正文内容，返回提取后的纯文本。

    主要用于抓取微信公众号文章等网页内容。
    """
    import requests as http_requests
    from urllib.parse import urlparse

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="请提供完整的 URL（以 http:// 或 https:// 开头）")

    # 安全校验：只允许抓取 HTTP(S) 链接
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="仅支持 HTTP/HTTPS 链接")

    # SSRF 防护：禁止请求内网 IP
    import ipaddress
    import socket
    hostname = parsed.hostname or ""
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                raise HTTPException(status_code=400, detail="不允许访问内网地址")
    except socket.gaierror:
        raise HTTPException(status_code=400, detail=f"无法解析域名: {hostname}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://mp.weixin.qq.com/",
    }

    try:
        resp = http_requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except http_requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"抓取失败: {e}")

    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text

    # 提取正文
    title, text = _extract_article_text(html)

    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="未能提取到有效正文内容，可能被反爬拦截。请尝试手动复制粘贴。")

    # 提取媒体
    media = _extract_article_media(html)

    logger.debug(f"fetch_url: url={url}, content_length={len(text)}, media_count={len(media)}")
    return {"title": title, "content": text, "url": url, "length": len(text), "media": media}


# ===== 自动抓取 + 导入一站式接口 =====

class AutoImportItem(BaseModel):
    """单个 URL 导入项"""
    url: str = Field(..., description="要抓取的网页 URL")
    type: str = Field(default="", description="知识类型（留空则自动推断）")
    id: str = Field(default="", description="实体ID（留空则从标题自动生成）")


class AutoImportRequest(BaseModel):
    """自动抓取 + 导入请求"""
    urls: List[str] = Field(default=[], description="URL 列表（简化用法）")
    items: List[AutoImportItem] = Field(default=[], description="详细导入项（可指定每个 URL 的类型和 ID）")
    type: str = Field(default="doctrine", description="全局默认知识类型")
    build: bool = Field(default=True, description="全部导入完成后自动构建索引")
    dry_run: bool = Field(default=False, description="仅预览，不写入文件")


def _fetch_url_content(url: str) -> dict:
    """抓取 URL 并提取正文（内部复用，含 SSRF 防护）。

    Returns: {"title": str, "content": str, "url": str}
    Raises: ValueError on failure.
    """
    import requests as http_requests
    from urllib.parse import urlparse
    import ipaddress
    import socket

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL 必须以 http:// 或 https:// 开头")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("仅支持 HTTP/HTTPS 链接")

    # SSRF 防护
    hostname = parsed.hostname or ""
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                raise ValueError("不允许访问内网地址")
    except socket.gaierror:
        raise ValueError(f"无法解析域名: {hostname}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        resp = http_requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except http_requests.RequestException as e:
        raise ValueError(f"抓取失败: {e}")

    resp.encoding = resp.apparent_encoding or "utf-8"
    title, text = _extract_article_text(resp.text)

    if not text or len(text.strip()) < 20:
        raise ValueError("未能提取到有效正文（可能被反爬拦截）")

    return {"title": title, "content": text, "url": url}


def _auto_detect_type(title: str, content: str) -> str:
    """根据标题和内容自动推断知识类型。"""
    sample = (title + " " + content[:500]).lower()
    # 按优先级匹配
    if any(kw in sample for kw in ("经", "论", "律", "偈", "品第")):
        return "scripture"
    if any(kw in sample for kw in ("禅修", "打坐", "修行", "念佛", "持咒", "观想", "止观")):
        return "practice"
    if any(kw in sample for kw in ("仪轨", "法会", "供养", "早课", "晚课")):
        return "ritual"
    if any(kw in sample for kw in ("宗派", "禅宗", "净土宗", "天台", "华严宗", "密宗", "传承")):
        return "sect"
    if any(kw in sample for kw in ("法师", "大师", "祖师", "高僧", "上人")):
        return "master"
    if any(kw in sample for kw in ("历史", "朝代", "传入", "佛教史")):
        return "history"
    return "doctrine"


def _title_to_id(title: str) -> str:
    """从标题生成安全的实体 ID。"""
    import re as _re
    # 提取中文/英文/数字
    cleaned = _re.sub(r"[^\w\u4e00-\u9fff]", "_", title.strip())
    cleaned = _re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        import hashlib
        cleaned = "auto_" + hashlib.md5(title.encode()).hexdigest()[:8]
    return cleaned[:50]


@app.post("/admin/auto_import")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_auto_import(request: Request, req: AutoImportRequest):
    """一站式自动导入：抓取 URL → LLM 整理 → 写入知识库 → 建索引。

    支持批量 URL，每个 URL 独立处理，部分失败不影响其他。
    """
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, _generate_knowledge,
        _write_knowledge_files,
    )

    # 合并 urls（简化用法）和 items（详细用法）
    all_items: List[AutoImportItem] = list(req.items)
    for u in req.urls:
        all_items.append(AutoImportItem(url=u, type="", id=""))

    if not all_items:
        raise HTTPException(status_code=400, detail="请提供至少一个 URL")
    if len(all_items) > 20:
        raise HTTPException(status_code=400, detail="单次最多 20 个 URL")

    results = []
    success_count = 0
    need_build_products = set()

    for item in all_items:
        entry = {"url": item.url, "status": "pending", "error": None}
        try:
            # 1. 抓取
            fetched = _fetch_url_content(item.url)
            entry["title"] = fetched["title"]
            entry["content_length"] = len(fetched["content"])

            # 2. 确定类型和 ID
            entity_type = (item.type or req.type or "").strip()
            if not entity_type:
                entity_type = _auto_detect_type(fetched["title"], fetched["content"])
            if entity_type not in _ENTITY_TYPES:
                entry["status"] = "failed"
                entry["error"] = f"不支持的类型: {entity_type}"
                results.append(entry)
                continue

            _, is_single = _ENTITY_TYPES[entity_type]
            entity_id = item.id.strip()
            if not entity_id and not is_single:
                entity_id = _title_to_id(fetched["title"])
            # 安全校验
            if entity_id and not _SAFE_NAME_RE.match(entity_id):
                entity_id = _title_to_id(entity_id)

            entry["type"] = entity_type
            entry["id"] = entity_id

            if req.dry_run:
                entry["status"] = "dry_run"
                results.append(entry)
                continue

            # 3. LLM 整理
            client = _get_openai_client()
            result = _generate_knowledge(client, fetched["content"], entity_type, entity_id)
            entry["files_generated"] = {}
            if result.get("main_txt"):
                entry["files_generated"]["main.txt"] = len(result["main_txt"])
            if result.get("faq_txt"):
                entry["files_generated"]["faq.txt"] = len(result["faq_txt"])
            if result.get("alias_txt"):
                entry["files_generated"]["alias.txt"] = len(result["alias_txt"])

            # 4. 写入知识库
            out_dir = _write_knowledge_files(result, entity_type, entity_id, dry_run=False)
            entry["output_dir"] = str(out_dir)

            # 记录需要重建索引的产品
            if entity_type == "product":
                need_build_products.add(entity_id)
            else:
                need_build_products.add("_shared")

            entry["status"] = "ok"
            success_count += 1

        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            log_error("auto_import", repr(e), meta={"url": item.url})

        results.append(entry)

    # 5. 统一建索引（避免每个 URL 重复构建）
    built_index = False
    if req.build and not req.dry_run and success_count > 0:
        try:
            if "_shared" in need_build_products:
                from build_faiss import build_shared
                build_shared()
                invalidate_store_cache("_shared")
            for pid in need_build_products:
                if pid != "_shared":
                    from build_faiss import build_for_product
                    build_for_product(pid)
                    invalidate_store_cache(pid)
            with _health_lock:
                global _health_cache
                _health_cache = {}
            built_index = True
        except Exception as e:
            log_error("auto_import_build", repr(e))

    return {
        "ok": success_count > 0,
        "total": len(all_items),
        "success": success_count,
        "failed": len(all_items) - success_count,
        "built_index": built_index,
        "dry_run": req.dry_run,
        "results": results,
    }


class BatchPatternRequest(BaseModel):
    """URL 模板批量导入请求"""
    url_pattern: str = Field(..., description="URL 模板，用 {n} 表示数字变量，如 https://example.com/lesson-{n}/")
    start: int = Field(..., description="起始数字")
    end: int = Field(..., description="结束数字（包含）")
    type: str = Field(default="doctrine", description="知识类型")
    build: bool = Field(default=True, description="全部导入完成后自动构建索引")
    dry_run: bool = Field(default=False, description="仅预览生成的 URL 列表，不实际导入")
    delay: float = Field(default=2.0, description="每次抓取间隔秒数（避免被封）")


@app.post("/admin/batch_pattern_import")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_batch_pattern_import(request: Request, req: BatchPatternRequest):
    """根据 URL 模板和数字范围批量导入。

    例如 url_pattern="https://example.com/di-{n}-ke/", start=1, end=201
    会自动生成 201 个 URL 并逐一抓取导入。
    """
    if "{n}" not in req.url_pattern:
        raise HTTPException(status_code=400, detail="url_pattern 必须包含 {n} 占位符")
    if req.start > req.end:
        raise HTTPException(status_code=400, detail="start 不能大于 end")
    if req.end - req.start + 1 > 500:
        raise HTTPException(status_code=400, detail="单次最多 500 个 URL")

    urls = [req.url_pattern.replace("{n}", str(i)) for i in range(req.start, req.end + 1)]

    if req.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "total": len(urls),
            "urls": urls,
        }

    # 实际导入
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, _generate_knowledge,
        _write_knowledge_files,
    )
    import time as _time

    results = []
    success_count = 0
    need_build_products = set()

    for idx, url in enumerate(urls):
        entry = {"url": url, "index": idx + 1, "status": "pending", "error": None}
        try:
            # 抓取
            fetched = _fetch_url_content(url)
            entry["title"] = fetched["title"]
            entry["content_length"] = len(fetched["content"])

            # 确定类型和 ID
            entity_type = req.type or "doctrine"
            if entity_type not in _ENTITY_TYPES:
                entry["status"] = "failed"
                entry["error"] = f"不支持的类型: {entity_type}"
                results.append(entry)
                continue

            _, is_single = _ENTITY_TYPES[entity_type]
            entity_id = _title_to_id(fetched["title"])
            entry["type"] = entity_type
            entry["id"] = entity_id

            # LLM 整理
            client = _get_openai_client()
            result = _generate_knowledge(client, fetched["content"], entity_type, entity_id)
            entry["files_generated"] = {}
            if result.get("main_txt"):
                entry["files_generated"]["main.txt"] = len(result["main_txt"])
            if result.get("faq_txt"):
                entry["files_generated"]["faq.txt"] = len(result["faq_txt"])
            if result.get("alias_txt"):
                entry["files_generated"]["alias.txt"] = len(result["alias_txt"])

            # 写入知识库
            out_dir = _write_knowledge_files(result, entity_type, entity_id, dry_run=False)
            entry["output_dir"] = str(out_dir)

            if entity_type == "product":
                need_build_products.add(entity_id)
            else:
                need_build_products.add("_shared")

            entry["status"] = "ok"
            success_count += 1

        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            log_error("batch_pattern_import", repr(e), meta={"url": url})

        results.append(entry)

        # 间隔避免被封
        if idx < len(urls) - 1 and req.delay > 0:
            _time.sleep(req.delay)

    # 统一建索引
    built_index = False
    if req.build and success_count > 0:
        try:
            if "_shared" in need_build_products:
                from build_faiss import build_shared
                build_shared()
                invalidate_store_cache("_shared")
            for pid in need_build_products:
                if pid != "_shared":
                    from build_faiss import build_for_product
                    build_for_product(pid)
                    invalidate_store_cache(pid)
            with _health_lock:
                global _health_cache
                _health_cache = {}
            built_index = True
        except Exception as e:
            log_error("batch_pattern_build", repr(e))

    return {
        "ok": success_count > 0,
        "total": len(urls),
        "success": success_count,
        "failed": len(urls) - success_count,
        "built_index": built_index,
        "results": results,
    }


class CrawlImportRequest(BaseModel):
    """爬取目录页并批量导入"""
    start_url: str = Field(..., description="起始页 URL（系统会自动发现同系列所有链接）")
    url_must_contain: str = Field(default="", description="URL 必须包含的关键字（过滤无关链接）")
    type: str = Field(default="doctrine", description="知识类型")
    build: bool = Field(default=True, description="完成后自动构建索引")
    dry_run: bool = Field(default=False, description="仅发现链接，不实际导入")
    delay: float = Field(default=2.0, description="每次抓取间隔秒数")
    max_pages: int = Field(default=500, description="最多抓取页数")


def _crawl_discover_links(start_url: str, url_must_contain: str = "",
                           max_pages: int = 500, delay: float = 1.0) -> list[str]:
    """从起始页爬取，自动发现同系列的所有文章链接。

    策略：
    1. 先抓起始页，提取页面中所有同域链接
    2. 用 url_must_contain 过滤出同系列链接
    3. 对发现的链接继续爬取目录/导航页以发现更多链接
    4. 去重排序后返回
    """
    import requests as http_requests
    from urllib.parse import urlparse, urljoin
    from bs4 import BeautifulSoup
    import re as _re
    import time as _time

    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    discovered: dict[str, str] = {}  # url -> title
    visited: set[str] = set()
    to_visit: list[str] = [start_url]

    # 也尝试抓 sitemap
    sitemap_urls = [
        f"{parsed_start.scheme}://{base_domain}/sitemap.xml",
        f"{parsed_start.scheme}://{base_domain}/sitemap_index.xml",
        f"{parsed_start.scheme}://{base_domain}/wp-sitemap.xml",
        f"{parsed_start.scheme}://{base_domain}/wp-sitemap-posts-post-1.xml",
        f"{parsed_start.scheme}://{base_domain}/wp-sitemap-posts-page-1.xml",
    ]
    for sm_url in sitemap_urls:
        try:
            resp = http_requests.get(sm_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                # 提取 sitemap 中的 URL
                for m in _re.finditer(r'<loc>\s*(https?://[^<]+)\s*</loc>', resp.text):
                    u = m.group(1).strip()
                    if url_must_contain and url_must_contain not in u:
                        continue
                    if urlparse(u).netloc == base_domain:
                        discovered[u] = ""
                        if u not in visited:
                            to_visit.append(u)
        except Exception:
            pass

    # BFS 爬取发现链接
    pages_fetched = 0
    while to_visit and pages_fetched < min(max_pages, 10):
        # 只爬少量页面来发现链接，不需要全部爬
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = http_requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            if resp.status_code != 200:
                continue
            resp.encoding = resp.apparent_encoding or "utf-8"
            pages_fetched += 1

            soup = BeautifulSoup(resp.text, "html.parser")

            # 提取所有链接
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full_url = urljoin(url, href)
                # 清理 fragment 和 query
                full_url = full_url.split("#")[0].split("?")[0]
                if not full_url.startswith("http"):
                    continue
                if urlparse(full_url).netloc != base_domain:
                    continue
                if url_must_contain and url_must_contain not in full_url:
                    continue

                title = (a.get_text(strip=True) or "")[:100]
                if full_url not in discovered:
                    discovered[full_url] = title

            if pages_fetched < 5 and delay > 0:
                _time.sleep(delay)

        except Exception:
            continue

    # 排序：尝试按数字排序
    def _sort_key(url: str):
        nums = _re.findall(r'(\d+)', url.split("/")[-2] if url.endswith("/") else url.split("/")[-1])
        return [int(n) for n in nums] if nums else [0]

    sorted_urls = sorted(discovered.keys(), key=_sort_key)
    return sorted_urls


@app.post("/admin/crawl_import")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_crawl_import(request: Request, req: CrawlImportRequest):
    """自动爬取目录发现所有链接，然后批量导入知识库。

    1. 从起始 URL 爬取，发现所有同系列文章链接
    2. 逐一抓取正文 → LLM 整理 → 写入知识库
    3. 统一构建索引
    """
    if not req.start_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="请提供完整 URL")

    # 第1步：发现链接
    try:
        all_urls = _crawl_discover_links(
            req.start_url,
            url_must_contain=req.url_must_contain,
            max_pages=req.max_pages,
            delay=req.delay,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"爬取发现链接失败: {e}")

    if not all_urls:
        return {"ok": False, "detail": "未发现任何匹配的链接", "total": 0, "urls": []}

    if req.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "total": len(all_urls),
            "urls": all_urls,
        }

    # 第2步：逐一导入
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, _generate_knowledge,
        _write_knowledge_files,
    )
    import time as _time

    results = []
    success_count = 0
    need_build_products = set()

    for idx, url in enumerate(all_urls):
        entry = {"url": url, "index": idx + 1, "status": "pending", "error": None}
        try:
            fetched = _fetch_url_content(url)
            entry["title"] = fetched["title"]
            entry["content_length"] = len(fetched["content"])

            entity_type = req.type or "doctrine"
            if entity_type not in _ENTITY_TYPES:
                entry["status"] = "failed"
                entry["error"] = f"不支持的类型: {entity_type}"
                results.append(entry)
                continue

            _, is_single = _ENTITY_TYPES[entity_type]
            entity_id = _title_to_id(fetched["title"])
            entry["type"] = entity_type
            entry["id"] = entity_id

            client = _get_openai_client()
            result = _generate_knowledge(client, fetched["content"], entity_type, entity_id)
            entry["files_generated"] = {}
            if result.get("main_txt"):
                entry["files_generated"]["main.txt"] = len(result["main_txt"])
            if result.get("faq_txt"):
                entry["files_generated"]["faq.txt"] = len(result["faq_txt"])
            if result.get("alias_txt"):
                entry["files_generated"]["alias.txt"] = len(result["alias_txt"])

            out_dir = _write_knowledge_files(result, entity_type, entity_id, dry_run=False)
            entry["output_dir"] = str(out_dir)

            if entity_type == "product":
                need_build_products.add(entity_id)
            else:
                need_build_products.add("_shared")

            entry["status"] = "ok"
            success_count += 1

        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            log_error("crawl_import", repr(e), meta={"url": url})

        results.append(entry)

        if idx < len(all_urls) - 1 and req.delay > 0:
            _time.sleep(req.delay)

    # 第3步：建索引
    built_index = False
    if req.build and success_count > 0:
        try:
            if "_shared" in need_build_products:
                from build_faiss import build_shared
                build_shared()
                invalidate_store_cache("_shared")
            for pid in need_build_products:
                if pid != "_shared":
                    from build_faiss import build_for_product
                    build_for_product(pid)
                    invalidate_store_cache(pid)
            with _health_lock:
                global _health_cache
                _health_cache = {}
            built_index = True
        except Exception as e:
            log_error("crawl_import_build", repr(e))

    return {
        "ok": success_count > 0,
        "total": len(all_urls),
        "success": success_count,
        "failed": len(all_urls) - success_count,
        "built_index": built_index,
        "results": results,
    }


def _extract_article_media(html: str) -> list:
    """从 HTML 中提取文章的图片和视频链接。"""
    import re as _re
    media = []
    seen = set()

    # --- 图片提取 ---
    # 微信图片：data-src（懒加载）和 src
    for attr in ("data-src", "src"):
        for m in _re.finditer(rf'<img[^>]+{attr}="([^"]+)"', html, _re.IGNORECASE):
            url = m.group(1).strip()
            if not url or url in seen:
                continue
            # 只保留真实图片链接（排除 data:, icon, emoji 等小图）
            if url.startswith("data:"):
                continue
            if any(skip in url for skip in ["emoji", "icon", "/s?__biz=", "res.wx.qq.com/t/wx_fed"]):
                continue
            seen.add(url)
            # 尝试提取 alt 作为描述
            alt_m = _re.search(r'alt="([^"]*)"', m.group(0))
            alt = alt_m.group(1) if alt_m else ""
            media.append({"type": "image", "url": url, "alt": alt})

    # --- 视频提取 ---
    # 微信文章视频：mpvideo（iframe 或 data-mpvid）
    for m in _re.finditer(r'data-mpvid="([^"]+)"', html):
        vid = m.group(1)
        if vid and vid not in seen:
            seen.add(vid)
            media.append({"type": "video", "url": f"https://mp.weixin.qq.com/mp/readtemplate?t=pages/video_player_tmpl&vid={vid}", "alt": ""})

    # video/source 标签
    for m in _re.finditer(r'<(?:video|source)[^>]+src="([^"]+)"', html, _re.IGNORECASE):
        url = m.group(1).strip()
        if url and url not in seen:
            seen.add(url)
            media.append({"type": "video", "url": url, "alt": ""})

    # iframe 嵌入视频（腾讯视频等）
    for m in _re.finditer(r'<iframe[^>]+src="([^"]*(?:v\.qq\.com|mp\.weixin)[^"]*)"', html, _re.IGNORECASE):
        url = m.group(1).strip()
        if url and url not in seen:
            seen.add(url)
            media.append({"type": "video", "url": url, "alt": ""})

    return media


@app.get("/admin/proxy_media")
@limiter.limit("120/minute")
def admin_proxy_media(request: Request, url: str = Query(..., description="要代理的媒体 URL")):
    """代理微信图片/视频，绕过防盗链（去除 Referer）。

    用于在手机端预览微信公众号文章中的图片。
    """
    import requests as http_requests
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="仅支持 HTTP/HTTPS")

    # 安全：只代理已知的图片/视频域名
    allowed_hosts = {"mmbiz.qpic.cn", "mmecoa.qpic.cn", "mmbiz.qlogo.cn",
                     "mp.weixin.qq.com", "mpvideo.qpic.cn", "wx1.sinaimg.cn",
                     "wx2.sinaimg.cn", "wx3.sinaimg.cn", "wx4.sinaimg.cn"}
    if parsed.hostname not in allowed_hosts:
        raise HTTPException(status_code=403, detail=f"不允许代理该域名: {parsed.hostname}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Referer": "",  # 空 Referer 绕过防盗链
    }

    try:
        resp = http_requests.get(url, headers=headers, timeout=15, stream=True)
        resp.raise_for_status()
    except http_requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"获取媒体失败: {e}")

    content_type = resp.headers.get("content-type", "application/octet-stream")

    def _stream():
        for chunk in resp.iter_content(chunk_size=65536):
            yield chunk

    return StreamingResponse(
        _stream(),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"}
    )


class SaveMediaRequest(BaseModel):
    """保存选中的媒体到 media.json"""
    product: str = Field(..., description="产品ID")
    media: list = Field(..., description="媒体列表，每项包含 title/type/url/keywords/routes")


@app.post("/admin/save_media")
@limiter.limit(_ADMIN_RATE_LIMIT)
def admin_save_media(request: Request, req: SaveMediaRequest):
    """将审核通过的媒体追加到产品的 media.json。"""
    import json

    product = req.product.strip()
    if not product or not re.match(r'^[\w\-\u4e00-\u9fff]+$', product):
        raise HTTPException(status_code=400, detail="无效的产品ID")

    logger.debug(f"save_media: product={product}, media_count={len(req.media)}")
    product_dir = KNOWLEDGE_DIR / product
    if not product_dir.is_dir():
        product_dir.mkdir(parents=True, exist_ok=True)

    media_file = product_dir / "media.json"

    # 读取现有 media.json
    existing = []
    if media_file.exists():
        try:
            existing = json.loads(media_file.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    # 去重：按 url 去重
    existing_urls = {item.get("url") for item in existing if isinstance(item, dict)}
    added = 0
    for item in req.media:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        if item["url"] in existing_urls:
            continue
        # 标准化格式
        entry = {
            "title": item.get("title", ""),
            "type": item.get("type", "image"),
            "url": item["url"],
            "routes": item.get("routes", []),
            "keywords": item.get("keywords", []),
        }
        existing.append(entry)
        existing_urls.add(item["url"])
        added += 1

    # 原子写入：先写临时文件再 rename，防止进程崩溃时数据损坏
    tmp = media_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(media_file))

    # 清除缓存
    invalidate_media_cache(product)

    return {"ok": True, "added": added, "total": len(existing)}


# ==================== crawl_v2 API ====================

class CrawlV2DiscoverRequest(BaseModel):
    start_url: str = Field(..., description="起始页 URL")
    url_must_contain: str = Field(default="", description="URL 过滤关键字")
    max_discovery_pages: int = Field(default=30, description="最多爬取几个页面来发现链接")
    delay: float = Field(default=1.0, description="发现阶段抓取间隔秒数")


class CrawlV2StartRequest(BaseModel):
    start_url: str = Field(..., description="起始页 URL")
    url_must_contain: str = Field(default="", description="URL 过滤关键字")
    type: str = Field(default="doctrine", description="知识类型")
    delay: float = Field(default=2.0, description="每次抓取间隔秒数")
    build: bool = Field(default=True, description="完成后自动构建索引")
    urls: Optional[list] = Field(default=None, description="手动指定 URL 列表（跳过发现阶段）")


@app.post("/admin/crawl_v2/discover")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_discover(request: Request, req: CrawlV2DiscoverRequest):
    """发现链接（预览，不实际导入）"""
    if not req.start_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="请提供完整 URL")

    from crawl_v2 import SiteAdapter
    adapter = SiteAdapter(
        req.start_url,
        url_must_contain=req.url_must_contain,
        max_discovery_pages=req.max_discovery_pages,
        delay=req.delay,
    )
    try:
        found = adapter.discover()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"链接发现失败: {e}")

    return {"ok": True, "total": len(found), "urls": found}


@app.post("/admin/crawl_v2/start")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_start(request: Request, req: CrawlV2StartRequest):
    """启动爬取导入任务（后台运行，返回 job_id）"""
    if not req.start_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="请提供完整 URL")

    from crawl_v2 import CrawlJob, SiteAdapter, set_active_job

    job = CrawlJob()
    job.start_url = req.start_url
    job.url_must_contain = req.url_must_contain
    job.entity_type = req.type
    job.delay = req.delay

    # 如果传入了 urls，直接用；否则先发现
    if req.urls:
        job.urls = [u if isinstance(u, dict) else {"url": u, "title": ""} for u in req.urls]
    else:
        adapter = SiteAdapter(req.start_url, url_must_contain=req.url_must_contain)
        try:
            job.urls = adapter.discover()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"链接发现失败: {e}")

    if not job.urls:
        return {"ok": False, "detail": "未发现任何链接"}

    job.save()
    set_active_job(job)

    # 后台线程执行
    def _run():
        try:
            job.run(build=req.build)
        except Exception as e:
            job.status = "failed"
            job.save()
            job.emit_sse("failed", {"detail": f"任务异常终止: {e}"})
            logger.error("crawl_v2 job failed: %s", e)
        finally:
            # 任务完成后不立即清理 active_jobs，保留供 SSE 查询
            pass

    t = threading.Thread(target=_run, daemon=True, name=f"crawl_v2_{job.job_id}")
    t.start()

    return {
        "ok": True,
        "job_id": job.job_id,
        "total": len(job.urls),
        "urls": job.urls,
    }


@app.get("/admin/crawl_v2/jobs")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_list_jobs(request: Request):
    """列出所有爬取任务"""
    from crawl_v2 import CrawlJob
    return {"ok": True, "jobs": CrawlJob.list_all()}


@app.get("/admin/crawl_v2/jobs/{job_id}")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_get_job(request: Request, job_id: str):
    """获取任务详情"""
    from crawl_v2 import CrawlJob, get_active_job
    # 优先从活跃任务取（有实时状态）
    job = get_active_job(job_id)
    if not job:
        try:
            job = CrawlJob.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")

    return {
        "ok": True,
        "job_id": job.job_id,
        "status": job.status,
        "total": len(job.urls),
        "completed": len(job.completed),
        "failed": len(job.failed),
        "urls": job.urls,
        "results": job.results,
        "built_index": job.built_index,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@app.get("/admin/crawl_v2/jobs/{job_id}/progress")
async def crawl_v2_progress(request: Request, job_id: str):
    """SSE 实时进度推送"""
    from crawl_v2 import get_active_job, CrawlJob
    import asyncio

    job = get_active_job(job_id)
    if not job:
        # 尝试加载已完成的任务
        try:
            finished = CrawlJob.load(job_id)
            # 直接返回最终状态
            async def _done_gen():
                yield f"data: {json.dumps({'event': 'done', 'total': len(finished.urls), 'completed': len(finished.completed), 'failed': len(finished.failed), 'built_index': finished.built_index}, ensure_ascii=False)}\n\n"
            return StreamingResponse(_done_gen(), media_type="text/event-stream")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")

    q = job.add_sse_listener()

    async def _generate():
        try:
            # 先推送当前状态摘要
            yield f"data: {json.dumps({'event': 'connected', 'job_id': job_id, 'status': job.status, 'total': len(job.urls), 'completed': len(job.completed), 'failed': len(job.failed)}, ensure_ascii=False)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                while q:
                    msg = q.pop(0)
                    yield f"data: {msg}\n\n"
                    parsed = json.loads(msg)
                    if parsed.get("event") in ("done", "failed"):
                        return
                # 检查任务是否已在队列外终止（如线程崩溃）
                if job.status in ("done", "failed"):
                    yield f"data: {json.dumps({'event': job.status, 'total': len(job.urls), 'completed': len(job.completed), 'failed': len(job.failed), 'built_index': job.built_index, 'detail': '任务已结束'}, ensure_ascii=False)}\n\n"
                    return
                await asyncio.sleep(0.3)
        finally:
            job.remove_sse_listener(q)

    return StreamingResponse(_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/admin/crawl_v2/jobs/{job_id}/pause")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_pause(request: Request, job_id: str):
    """暂停任务"""
    from crawl_v2 import get_active_job
    job = get_active_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="活跃任务不存在")
    if job.status != "running":
        return {"ok": False, "detail": f"任务状态为 {job.status}，无法暂停"}
    job._stop_flag = True
    job._pause_event.clear()
    return {"ok": True, "detail": "暂停信号已发送"}


@app.post("/admin/crawl_v2/jobs/{job_id}/resume")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_resume(request: Request, job_id: str):
    """断点续传：继续执行暂停/中断的任务"""
    from crawl_v2 import CrawlJob, get_active_job, set_active_job

    job = get_active_job(job_id)
    if not job:
        try:
            job = CrawlJob.load(job_id)
            set_active_job(job)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")

    if job.status == "running":
        return {"ok": False, "detail": "任务正在运行中"}

    # 重置暂停状态
    job._stop_flag = False
    job._pause_event.set()
    job.status = "running"
    job.save()

    def _run():
        try:
            job.run(build=True)
        except Exception as e:
            job.status = "failed"
            job.save()
            job.emit_sse("failed", {"detail": f"任务异常终止: {e}"})
            logger.error("crawl_v2 resume failed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name=f"crawl_v2_resume_{job.job_id}")
    t.start()

    remaining = len(job.urls) - len(job.completed)
    return {"ok": True, "detail": f"已恢复，剩余 {remaining} 项", "job_id": job.job_id}


@app.post("/admin/crawl_v2/jobs/{job_id}/retry_failed")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_retry_failed(request: Request, job_id: str):
    """重试所有失败项"""
    from crawl_v2 import CrawlJob, get_active_job, set_active_job

    job = get_active_job(job_id)
    if not job:
        try:
            job = CrawlJob.load(job_id)
            set_active_job(job)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")

    if job.status == "running":
        return {"ok": False, "detail": "任务正在运行中"}

    if not job.failed:
        return {"ok": False, "detail": "没有失败项需要重试"}

    # 清除失败记录，让 run() 重新处理这些索引
    failed_indices = list(job.failed.keys())
    job.failed.clear()
    # 清除对应的 results
    job.results = [r for r in job.results if r.get("index") not in failed_indices]
    job._stop_flag = False
    job._pause_event.set()
    job.save()

    def _run():
        try:
            job.run(build=True)
        except Exception as e:
            job.status = "failed"
            job.save()
            job.emit_sse("failed", {"detail": f"任务异常终止: {e}"})
            logger.error("crawl_v2 retry failed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name=f"crawl_v2_retry_{job.job_id}")
    t.start()

    return {"ok": True, "detail": f"正在重试 {len(failed_indices)} 个失败项", "job_id": job.job_id}


@app.post("/admin/crawl_v2/jobs/{job_id}/stop")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_stop(request: Request, job_id: str):
    """强制终止任务（标记为 done，不再继续处理剩余项）"""
    from crawl_v2 import CrawlJob, get_active_job

    job = get_active_job(job_id)
    if not job:
        try:
            job = CrawlJob.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")

    if job.status not in ("running", "paused", "pending"):
        return {"ok": False, "detail": f"任务状态为 {job.status}，无需终止"}

    job._stop_flag = True
    job._pause_event.set()  # 唤醒暂停中的线程使其退出
    job.status = "done"
    job.save()
    job.emit_sse("done", {
        "total": len(job.urls),
        "completed": len(job.completed),
        "failed": len(job.failed),
        "built_index": job.built_index,
    })

    return {
        "ok": True,
        "detail": f"任务已终止，已完成 {len(job.completed)}/{len(job.urls)}",
        "completed": len(job.completed),
        "total": len(job.urls),
    }


@app.post("/admin/crawl_v2/jobs/{job_id}/restart")
@limiter.limit(_ADMIN_RATE_LIMIT)
def crawl_v2_restart(request: Request, job_id: str):
    """重新开始任务：清除所有进度，从头重新处理全部 URL"""
    from crawl_v2 import CrawlJob, get_active_job, set_active_job

    job = get_active_job(job_id)
    if not job:
        try:
            job = CrawlJob.load(job_id)
            set_active_job(job)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")

    if job.status == "running":
        return {"ok": False, "detail": "任务正在运行中，请先暂停或终止"}

    # 清除所有进度
    total = len(job.urls)
    job.completed.clear()
    job.failed.clear()
    job.results.clear()
    job.built_index = False
    job._stop_flag = False
    job._pause_event.set()
    job.save()

    def _run():
        try:
            job.run(build=True)
        except Exception as e:
            job.status = "failed"
            job.save()
            job.emit_sse("failed", {"detail": f"任务异常终止: {e}"})
            logger.error("crawl_v2 restart failed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name=f"crawl_v2_restart_{job.job_id}")
    t.start()

    return {"ok": True, "detail": f"任务已重新开始，共 {total} 项", "job_id": job.job_id, "total": total}


# ===== 静态文件（必须放在所有路由之后，避免拦截 API 路径）=====
_web_dir = BASE_DIR / "web"
if _web_dir.is_dir():
    app.mount("/web", StaticFiles(directory=str(_web_dir)), name="web_static")
