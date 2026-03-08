import sys
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from media_router import find_media

BASE_DIR = Path(__file__).resolve().parent
INDEX_PAGE = BASE_DIR / "index.html"

logger = logging.getLogger("rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载模型，避免首次请求卡顿"""
    logger.info("Preloading BGE-M3 model...")
    try:
        from rag_answer import get_model
        get_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Model preload failed (will retry on first query): {e}")
    yield


app = FastAPI(title="Buddhist Knowledge RAG API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    mode: Literal["brief", "full"] = "brief"
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
    debug: Optional[Dict[str, Any]] = None


@app.get("/")
def root():
    return FileResponse(str(INDEX_PAGE))


@app.get("/health")
def health():
    from rag_answer import _model
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        return AskResponse(ok=False, answer="请输入问题。", route="")
    try:
        from rag_answer import answer_question, detect_route
        route = detect_route(req.question)
        answer = answer_question(req.question, req.mode)
        media = [MediaItem(**m) for m in find_media(req.question)]
        debug = None
        if req.debug:
            debug = {"route": route, "mode": req.mode, "question": req.question}
        return AskResponse(
            ok=True,
            answer=answer,
            media=media,
            route=route,
            debug=debug,
        )
    except Exception as e:
        logger.exception("Error answering question")
        return AskResponse(ok=False, answer=f"处理异常：{e}", route="")
