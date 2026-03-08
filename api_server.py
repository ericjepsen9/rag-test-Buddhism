import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from media_router import find_media

BASE_DIR = Path(__file__).resolve().parent
RAG_SCRIPT = BASE_DIR / "rag_answer.py"
ANSWER_FILE = BASE_DIR / "answer.txt"
PYTHON_EXE = sys.executable

app = FastAPI(title="Medical Aesthetics RAG API", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    mode: Literal["brief", "full"] = "brief"
    timeout_sec: int = 90
    debug: bool = False


class MediaItem(BaseModel):
    title: str
    type: str
    url: str = ""


class AskResponse(BaseModel):
    ok: bool
    answer: str
    media: List[MediaItem] = []
    stderr: str = ""
    debug: Optional[Dict[str, Any]] = None


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=enc).strip()
        except Exception:
            pass
    return ""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_script_exists": RAG_SCRIPT.exists(),
        "answer_file_exists": ANSWER_FILE.exists(),
        "python_exe": PYTHON_EXE,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    cmd = [PYTHON_EXE, str(RAG_SCRIPT), req.question, req.mode]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5, req.timeout_sec),
            env=env,
        )
        answer = read_text(ANSWER_FILE)
        media = [MediaItem(**m) for m in find_media(req.question)]
        debug = None
        if req.debug:
            debug = {"cmd": cmd, "stdout": (proc.stdout or "")[:2000], "stderr": (proc.stderr or "")[:2000]}
        return AskResponse(
            ok=(proc.returncode == 0),
            answer=answer,
            media=media,
            stderr=(proc.stderr or "")[:2000],
            debug=debug,
        )
    except Exception as e:
        return AskResponse(ok=False, answer="接口执行异常", stderr=repr(e))
