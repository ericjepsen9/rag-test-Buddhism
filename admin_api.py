from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_logger import get_recent_errors, get_recent_misses, get_recent_qa, log_error

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
BUILD_SCRIPT = BASE_DIR / "build_faiss.py"
ADMIN_PAGE = BASE_DIR / "admin_page.html"
PYTHON_EXE = sys.executable

app = FastAPI(title="RAG Admin API", version="1.0.0")


class RebuildRequest(BaseModel):
    product: str = Field(..., description="product id，例如 feiluoao")
    timeout_sec: int = Field(180, ge=10, le=1800)


@app.get("/")
def root() -> FileResponse:
    if not ADMIN_PAGE.exists():
        raise HTTPException(status_code=404, detail="admin_page.html 不存在")
    return FileResponse(ADMIN_PAGE)


@app.get("/admin/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "base_dir": str(BASE_DIR),
        "knowledge_exists": KNOWLEDGE_DIR.exists(),
        "build_script_exists": BUILD_SCRIPT.exists(),
        "python_exe": PYTHON_EXE,
    }


@app.get("/admin/products")
def list_products() -> Dict[str, List[Dict[str, Any]]]:
    products: List[Dict[str, Any]] = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir():
                continue
            products.append(
                {
                    "product": p.name,
                    "files": sorted([x.name for x in p.iterdir() if x.is_file()]),
                }
            )
    return {"products": products}


@app.post("/admin/rebuild")
def rebuild(req: RebuildRequest) -> Dict[str, Any]:
    if not BUILD_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="build_faiss.py 不存在")

    cmd = [PYTHON_EXE, str(BUILD_SCRIPT), "--product", req.product]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=req.timeout_sec,
            env=env,
            shell=False,
        )
        return {
            "ok": proc.returncode == 0,
            "cmd": cmd,
            "return_code": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
        }
    except Exception as e:
        log_error("admin_rebuild", repr(e), meta={"product": req.product})
        raise HTTPException(status_code=500, detail=repr(e))


@app.get("/admin/logs/qa")
def logs_qa(limit: int = 20) -> Dict[str, Any]:
    return {"items": get_recent_qa(limit=limit)}


@app.get("/admin/logs/miss")
def logs_miss(limit: int = 20) -> Dict[str, Any]:
    return {"items": get_recent_misses(limit=limit)}


@app.get("/admin/logs/error")
def logs_error(limit: int = 20) -> Dict[str, Any]:
    return {"items": get_recent_errors(limit=limit)}
