import os
import re
import sys
import threading
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from media_router import find_media, invalidate_media_cache
from rag_logger import log_qa, log_error, get_recent_qa, get_recent_misses, get_recent_errors
from rag_runtime_config import KNOWLEDGE_DIR, STORE_ROOT

# 每产品重建锁，防止并发 rebuild 导致文件损坏
_rebuild_locks: Dict[str, threading.Lock] = {}
_rebuild_locks_guard = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent
INDEX_PAGE = BASE_DIR / "index.html"
ADMIN_PAGE = BASE_DIR / "admin_page.html"
CHAT_PAGE = BASE_DIR / "web" / "chat.html"
BUILD_SCRIPT = BASE_DIR / "build_faiss.py"
PYTHON_EXE = sys.executable

logger = logging.getLogger("rag_api")

app = FastAPI(title="Buddhist Knowledge RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 启动预热 =====

@app.on_event("startup")
def _warmup_models():
    """启动时预热嵌入模型并预构建共享知识索引，避免首次查询延迟"""
    if os.environ.get("SKIP_WARMUP"):
        return
    try:
        from rag_answer import embed_query
        embed_query("预热查询")
        logger.info("嵌入模型预热完成")
    except Exception as e:
        logger.warning(f"模型预热失败: {e}")
    # 预构建共享知识索引，避免首次查询时的冷启动延迟
    try:
        from rag_answer import get_model
        get_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Model preload failed (will retry on first query): {e}")


# ===== 输入清理 =====

MAX_QUESTION_LEN = 500
MAX_HISTORY_TOTAL_CHARS = 3000  # 防止历史内容过大

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


class AskResponse(BaseModel):
    ok: bool
    answer: str
    media: List[MediaItem] = []
    route: str = ""
    latency_ms: Optional[int] = None
    debug: Optional[Dict[str, Any]] = None


class RebuildRequest(BaseModel):
    product: str = Field(default="buddhism", min_length=1, max_length=50)
    timeout_sec: int = Field(default=180, ge=10, le=1800)


class KnowledgeWriteRequest(BaseModel):
    content: str = Field(..., min_length=0)


class CreateProductRequest(BaseModel):
    product: str = Field(..., min_length=1, max_length=50)


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any]


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


# ===== 健康检查 =====

_health_cache: Dict[str, Any] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = 15.0  # 秒
_health_lock = threading.Lock()


@app.get("/health")
def health():
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    # 快速路径：无锁读取（GIL 保护 dict 引用读取安全性）
    if _health_cache and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return _health_cache

    products = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir():
                continue
            store = STORE_ROOT / p.name
            products.append({
                "name": p.name,
                "index_exists": (store / "index.faiss").exists(),
                "docs_exists": (store / "docs.jsonl").exists(),
            })
    all_indexed = all(p["index_exists"] and p["docs_exists"] for p in products) if products else False
    result = {
        "status": "ok" if all_indexed else "degraded",
        "knowledge_exists": KNOWLEDGE_DIR.exists(),
        "products": products,
    }
    with _health_lock:
        _health_cache = result
        _health_cache_ts = now
    return result


# ===== 页面路由 =====

@app.get("/")
def root():
    return FileResponse(str(INDEX_PAGE))


@app.get("/chat")
def chat_page():
    if not CHAT_PAGE.exists():
        raise HTTPException(status_code=404, detail="chat.html 不存在")
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.get("/admin")
def admin_page():
    if not ADMIN_PAGE.exists():
        raise HTTPException(status_code=404, detail="admin_page.html 不存在")
    return FileResponse(ADMIN_PAGE)


# ===== 问答接口 =====

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = _sanitize_input(req.question)
    if not question:
        return AskResponse(ok=False, answer="请输入问题。", route="")

    start_ms = time.monotonic()
    try:
        from rag_answer import answer_question, detect_route

        # 处理历史消息：取最近 6 条，限制总字符数
        history = [{"role": h.role, "content": _sanitize_input(h.content[:1000])} for h in req.history[-6:]]
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

        route = detect_route(question)
        answer = answer_question(question, req.mode)
        product_id = "buddhism"
        media = [MediaItem(**m) for m in find_media(question, product_id=product_id, route=route)]
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)

        # 记录 QA 日志（与原始 RAG 架构一致）
        hit = bool(answer and "未找到" not in answer and "未覆盖" not in answer)
        log_qa(
            question=question,
            answer=answer,
            hit=hit,
            latency_ms=elapsed_ms,
            meta={"route": route, "mode": req.mode},
        )

        debug = None
        if req.debug:
            debug = {
                "question": question,
                "mode": req.mode,
                "route": route,
                "product": product_id,
                "latency_ms": elapsed_ms,
                "history_count": len(history),
            }
        return AskResponse(
            ok=True,
            answer=answer,
            media=media,
            route=route,
            latency_ms=elapsed_ms,
            debug=debug,
        )
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
        logger.exception("Error answering question")
        log_error("api_ask", repr(e), meta={"question": question[:200], "latency_ms": elapsed_ms})
        return AskResponse(ok=False, answer=f"处理异常：{e}", route="")


# ===== OpenAI 兼容接口 =====

def _oai_messages_to_question_and_history(messages: List[OAIMessage]):
    """将 OpenAI messages 格式转换为 question + history"""
    conv = [m for m in messages if m.role in ("user", "assistant")]
    if not conv:
        return "", []
    last = conv[-1]
    if last.role != "user":
        user_msgs = [m for m in conv if m.role == "user"]
        if not user_msgs:
            return "", []
        question = _sanitize_input(user_msgs[-1].content[:MAX_QUESTION_LEN])
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


def _build_oai_stream_chunk(content: str, model: str, chunk_id: str, finish: bool = False) -> str:
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
def oai_chat_completions(req: OAIChatRequest):
    question, history = _oai_messages_to_question_and_history(req.messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

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

        from rag_answer import answer_question
        answer = answer_question(question, "brief")
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log_error("oai_chat", repr(e), meta={"question": question[:200], "latency_ms": latency_ms})
        answer = "接口执行异常，请稍后重试"

    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def _generate():
            yield _build_oai_stream_chunk(answer, req.model, chunk_id)
            yield _build_oai_stream_chunk("", req.model, chunk_id, finish=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    return _build_oai_response(answer, req.model)


@app.get("/v1/models")
def oai_models():
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

# 安全校验：产品名只允许字母数字下划线横线中文
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\u4e00-\u9fff]+$")
_ALLOWED_EXTENSIONS = {".txt", ".json"}


def _validate_product_name(name: str) -> str:
    """校验并清理产品名"""
    name = name.strip()
    if not name or not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="非法产品名称：只允许字母、数字、下划线、横线、中文")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 路径遍历防护
    product_dir = (KNOWLEDGE_DIR / name).resolve()
    if not str(product_dir).startswith(str(KNOWLEDGE_DIR.resolve()) + "/"):
        raise HTTPException(status_code=400, detail="非法产品名称")
    return name


@app.get("/admin/products")
def admin_products():
    products = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir():
                continue
            products.append({
                "product": p.name,
                "files": sorted([x.name for x in p.iterdir() if x.is_file()]),
            })
    return {"products": products}


@app.post("/admin/rebuild")
def admin_rebuild(req: RebuildRequest):
    global _health_cache
    from build_faiss import build_for_product
    product = req.product.strip()
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
        build_for_product(product)
        # 重建后清除缓存，下次请求会加载新索引
        try:
            from rag_answer import _store_cache, _store_mtime
            _store_cache.pop(product, None)
            _store_mtime.pop(product, None)
        except ImportError:
            pass
        with _health_lock:
            _health_cache = {}  # 索引变更后清除健康检查缓存
        # 同时清除关联数据和媒体缓存
        try:
            from relation_engine import invalidate_relations_cache
            invalidate_relations_cache()
        except ImportError:
            pass
        invalidate_media_cache(product)
        return {"ok": True, "product": product}
    except Exception as e:
        log_error("admin_rebuild", repr(e), meta={"product": product})
        raise HTTPException(status_code=500, detail="索引重建失败，请查看服务器日志")
    finally:
        lock.release()


@app.get("/admin/config")
def admin_get_config():
    """获取所有可调参数"""
    from rag_runtime_config import get_tunable_config
    return get_tunable_config()


@app.post("/admin/config/update")
def admin_update_config(req: ConfigUpdateRequest):
    """热更新运行时参数"""
    from rag_runtime_config import update_tunable_config
    changed = update_tunable_config(req.updates)
    return {"ok": True, "changed": changed}


@app.get("/admin/logs/qa")
def admin_logs_qa(limit: int = 20):
    return {"items": get_recent_qa(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/miss")
def admin_logs_miss(limit: int = 20):
    return {"items": get_recent_misses(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/error")
def admin_logs_error(limit: int = 20):
    return {"items": get_recent_errors(limit=min(max(1, limit), 100))}


# ===== 词库管理接口 =====

@app.get("/admin/synonyms/all")
def admin_synonyms_all():
    """返回完整词库：静态同义词 + LLM 学习到的同义词"""
    from synonym_store import get_all_synonyms_combined
    return get_all_synonyms_combined()


@app.get("/admin/synonyms/learned")
def admin_synonyms_learned():
    """返回所有 LLM 学习到的同义词映射"""
    from synonym_store import get_all_learned
    return {"items": get_all_learned()}


@app.post("/admin/synonyms/learned/approve")
def admin_synonyms_approve(original: str):
    """审核通过一条学习到的同义词"""
    from synonym_store import approve_learned
    if not original or not original.strip():
        raise HTTPException(status_code=400, detail="original 不能为空")
    ok = approve_learned(original.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该同义词")
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
    return {"ok": True, "deleted": original.strip()}


# ===== 知识库文件管理接口 =====

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
    if not str(fpath.resolve()).startswith(str(KNOWLEDGE_DIR.resolve()) + "/"):
        raise HTTPException(status_code=400, detail="非法路径")
    try:
        content = fpath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = fpath.read_text(encoding="utf-8-sig", errors="replace")
    return {"product": product, "filename": filename, "content": content,
            "size": len(content)}


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
    if not str(fpath.resolve()).startswith(str((KNOWLEDGE_DIR / product).resolve())):
        raise HTTPException(status_code=400, detail="非法路径")
    # 原子写入
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    try:
        tmp.write_text(req.content, encoding="utf-8")
        os.replace(str(tmp), str(fpath))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="写入失败，请查看服务器日志")
    return {"ok": True, "product": product, "filename": filename,
            "size": len(req.content)}


@app.delete("/admin/knowledge/{product}/{filename}")
def admin_knowledge_delete(product: str, filename: str):
    """删除知识库文件"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    fpath = KNOWLEDGE_DIR / product / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    if not str(fpath.resolve()).startswith(str(KNOWLEDGE_DIR.resolve()) + "/"):
        raise HTTPException(status_code=400, detail="非法路径")
    fpath.unlink()
    return {"ok": True, "deleted": filename}


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
    store_dir = STORE_ROOT / product
    if store_dir.exists():
        shutil.rmtree(store_dir)
    try:
        from rag_answer import _store_cache, _store_mtime
        _store_cache.pop(product, None)
        _store_mtime.pop(product, None)
    except ImportError:
        pass
    invalidate_media_cache(product)
    return {"ok": True, "deleted": product}


# ===== 文件上传接口 =====

@app.post("/admin/upload")
async def admin_upload(request: Request):
    """通用文件上传：支持上传 txt/json 文件到指定产品目录。
    Form fields: product (str), files (UploadFile[])
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")
    try:
        form = await request.form()
    except Exception:
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
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = content.decode("utf-8-sig")
                    except Exception:
                        text = content.decode("gbk", errors="replace")
                fpath = pdir / fname
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


@app.post("/admin/upload_zip")
async def admin_upload_zip(request: Request):
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
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        parts = Path(info.filename).parts
                        if len(parts) < 2:
                            continue
                        product_name = parts[0]
                        file_name = parts[-1]
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
                        pdir.mkdir(parents=True, exist_ok=True)
                        content = zf.read(info.filename)
                        try:
                            text = content.decode("utf-8")
                        except UnicodeDecodeError:
                            text = content.decode("utf-8-sig", errors="replace")
                        dest = pdir / file_name
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


# ===== 启动入口 =====

if __name__ == "__main__":
    import uvicorn
    from rag_runtime_config import SERVER_HOST, SERVER_PORT
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
