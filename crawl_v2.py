"""
crawl_v2 — 网站自动爬取导入系统 (v2)

特性：
- SiteAdapter 可扩展的链接发现（sitemap + 导航 + 翻页追踪）
- CrawlJob 后台任务 + JSON 持久化 + 断点续传
- SSE 实时进度推送
- 暂停/继续/重试失败项
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin

logger = logging.getLogger("crawl_v2")

DATA_DIR = Path(__file__).resolve().parent / "data" / "crawl_jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


# ==================== SiteAdapter ====================

class SiteAdapter:
    """通用网站适配器：自动发现目录链接。

    发现策略（按优先级）：
    1. sitemap.xml 系列
    2. 起始页上的同系列链接
    3. 翻页追踪（下一课/pagination）
    """

    def __init__(self, start_url: str, url_must_contain: str = "",
                 max_discovery_pages: int = 30, delay: float = 1.0):
        self.start_url = start_url
        self.url_must_contain = url_must_contain
        self.max_discovery_pages = max_discovery_pages
        self.delay = delay
        parsed = urlparse(start_url)
        self.scheme = parsed.scheme
        self.domain = parsed.netloc

    def _get(self, url: str, timeout: int = 15):
        import requests
        return requests.get(url, headers=_HTTP_HEADERS, timeout=timeout, allow_redirects=True)

    def _matches_filter(self, url: str) -> bool:
        if self.url_must_contain and self.url_must_contain not in url:
            return False
        return urlparse(url).netloc == self.domain

    def discover(self) -> list[dict]:
        """返回 [{"url": ..., "title": ...}, ...] 按课序排序"""
        found: dict[str, str] = {}  # url -> title

        # 策略1：sitemap
        self._discover_from_sitemap(found)

        # 策略2：BFS 爬取起始页和发现的页面
        self._discover_from_pages(found)

        # 去重、排序
        results = [{"url": u, "title": t} for u, t in found.items()]
        results.sort(key=lambda x: self._sort_key(x["url"]))
        return results

    def _discover_from_sitemap(self, found: dict[str, str]):
        sitemap_urls = [
            f"{self.scheme}://{self.domain}/sitemap.xml",
            f"{self.scheme}://{self.domain}/sitemap_index.xml",
            f"{self.scheme}://{self.domain}/wp-sitemap.xml",
            f"{self.scheme}://{self.domain}/wp-sitemap-posts-post-1.xml",
            f"{self.scheme}://{self.domain}/wp-sitemap-posts-page-1.xml",
        ]
        for sm_url in sitemap_urls:
            try:
                resp = self._get(sm_url, timeout=10)
                if resp.status_code != 200:
                    continue
                text = resp.text

                # 如果是 sitemap index，提取子 sitemap
                sub_sitemaps = re.findall(r'<loc>\s*(https?://[^<]+sitemap[^<]*)\s*</loc>', text)
                for sub_url in sub_sitemaps:
                    try:
                        sub_resp = self._get(sub_url, timeout=10)
                        if sub_resp.status_code == 200:
                            for m in re.finditer(r'<loc>\s*(https?://[^<]+)\s*</loc>', sub_resp.text):
                                u = m.group(1).strip()
                                if self._matches_filter(u):
                                    found.setdefault(u, "")
                    except Exception:
                        pass

                # 直接提取当前 sitemap 的 URL
                for m in re.finditer(r'<loc>\s*(https?://[^<]+)\s*</loc>', text):
                    u = m.group(1).strip()
                    if self._matches_filter(u):
                        found.setdefault(u, "")
            except Exception:
                pass

    def _discover_from_pages(self, found: dict[str, str]):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return

        visited: set[str] = set()
        to_visit: list[str] = [self.start_url]

        # 也把已发现的前几个和最后几个加入待访问（目录链接常在导航中）
        known = list(found.keys())
        if known:
            to_visit.extend(known[:3])
            to_visit.extend(known[-3:])

        pages_fetched = 0
        while to_visit and pages_fetched < self.max_discovery_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = self._get(url)
                if resp.status_code != 200:
                    continue
                resp.encoding = resp.apparent_encoding or "utf-8"
                pages_fetched += 1

                soup = BeautifulSoup(resp.text, "html.parser")
                new_links = []

                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    full_url = urljoin(url, href).split("#")[0].split("?")[0]
                    if not full_url.startswith("http"):
                        continue
                    if not self._matches_filter(full_url):
                        continue
                    title = (a.get_text(strip=True) or "")[:100]
                    if full_url not in found:
                        found[full_url] = title
                        new_links.append(full_url)

                # 翻页追踪：查找"下一课"、"下一页"、pagination 链接
                next_links = self._find_next_page_links(soup, url)
                for nl in next_links:
                    if nl not in visited:
                        to_visit.append(nl)

                # 如果发现了很多新链接，不需要继续深爬
                if len(found) > 50 and pages_fetched > 5:
                    break

                if self.delay > 0 and pages_fetched < self.max_discovery_pages:
                    time.sleep(self.delay)

            except Exception:
                continue

    def _find_next_page_links(self, soup, current_url: str) -> list[str]:
        """查找翻页/下一课链接"""
        results = []
        next_patterns = re.compile(
            r'(下一[课页篇章节]|next|older|后一|»|›)', re.IGNORECASE
        )
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            classes = " ".join(a.get("class", []))
            rel = " ".join(a.get("rel", []))
            if next_patterns.search(text) or "next" in classes or "next" in rel:
                href = urljoin(current_url, a["href"]).split("#")[0].split("?")[0]
                if href.startswith("http") and urlparse(href).netloc == self.domain:
                    results.append(href)

        # pagination: <nav class="pagination">
        for nav in soup.find_all(["nav", "div"], class_=re.compile(r'paginat|page-nav', re.I)):
            for a in nav.find_all("a", href=True):
                href = urljoin(current_url, a["href"]).split("#")[0].split("?")[0]
                if href.startswith("http") and self._matches_filter(href):
                    results.append(href)

        return results

    @staticmethod
    def _sort_key(url: str):
        last_part = url.rstrip("/").split("/")[-1]
        nums = re.findall(r'(\d+)', last_part)
        return [int(n) for n in nums] if nums else [0]


# ==================== CrawlJob ====================

class CrawlJob:
    """爬取任务，支持持久化和断点续传"""

    def __init__(self, job_id: str = None):
        self.job_id = job_id or f"cj_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.status = "pending"  # pending | running | paused | done | failed
        self.start_url = ""
        self.url_must_contain = ""
        self.entity_type = "doctrine"
        self.delay = 2.0
        self.urls: list[dict] = []  # [{"url": ..., "title": ...}, ...]
        self.completed: list[int] = []  # 已完成的索引
        self.failed: dict[int, str] = {}  # 索引 -> 错误
        self.results: list[dict] = []  # 详细结果
        self.created_at = time.time()
        self.updated_at = time.time()
        self.built_index = False

        # 运行时状态（不持久化）
        self._sse_listeners: list = []
        self._sse_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 默认不暂停
        self._stop_flag = False

    @property
    def file_path(self) -> Path:
        return DATA_DIR / f"{self.job_id}.json"

    def save(self):
        """持久化到 JSON"""
        self.updated_at = time.time()
        data = {
            "job_id": self.job_id,
            "status": self.status,
            "start_url": self.start_url,
            "url_must_contain": self.url_must_contain,
            "entity_type": self.entity_type,
            "delay": self.delay,
            "urls": self.urls,
            "completed": self.completed,
            "failed": {str(k): v for k, v in self.failed.items()},
            "results": self.results,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "built_index": self.built_index,
        }
        tmp = self.file_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(self.file_path))

    @classmethod
    def load(cls, job_id: str) -> "CrawlJob":
        fp = DATA_DIR / f"{job_id}.json"
        if not fp.exists():
            raise FileNotFoundError(f"任务不存在: {job_id}")
        data = json.loads(fp.read_text(encoding="utf-8"))
        job = cls(data["job_id"])
        job.status = data["status"]
        job.start_url = data["start_url"]
        job.url_must_contain = data.get("url_must_contain", "")
        job.entity_type = data.get("entity_type", "doctrine")
        job.delay = data.get("delay", 2.0)
        job.urls = data.get("urls", [])
        job.completed = data.get("completed", [])
        job.failed = {int(k): v for k, v in data.get("failed", {}).items()}
        job.results = data.get("results", [])
        job.created_at = data.get("created_at", 0)
        job.updated_at = data.get("updated_at", 0)
        job.built_index = data.get("built_index", False)
        return job

    @classmethod
    def list_all(cls) -> list[dict]:
        jobs = []
        for f in sorted(DATA_DIR.glob("cj_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                jobs.append({
                    "job_id": data["job_id"],
                    "status": data["status"],
                    "start_url": data["start_url"],
                    "total": len(data.get("urls", [])),
                    "completed": len(data.get("completed", [])),
                    "failed": len(data.get("failed", {})),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "built_index": data.get("built_index", False),
                })
            except Exception:
                pass
        return jobs

    def emit_sse(self, event: str, data: dict):
        """向所有 SSE 监听者推送事件"""
        msg = json.dumps({"event": event, **data}, ensure_ascii=False)
        with self._sse_lock:
            dead = []
            for q in self._sse_listeners:
                try:
                    q.append(msg)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._sse_listeners.remove(q)

    def add_sse_listener(self) -> list:
        """注册一个 SSE 监听队列"""
        q: list = []
        with self._sse_lock:
            self._sse_listeners.append(q)
        return q

    def remove_sse_listener(self, q: list):
        with self._sse_lock:
            if q in self._sse_listeners:
                self._sse_listeners.remove(q)

    def run(self, build: bool = True):
        """在当前线程中执行导入任务"""
        logger.info("任务 %s 开始执行，共 %d 个URL", self.job_id, len(self.urls))
        from import_knowledge import (
            _ENTITY_TYPES, _get_openai_client, _generate_knowledge,
            _write_knowledge_files,
        )
        # 延迟导入 api_server 中的函数
        from api_server import _fetch_url_content, _title_to_id
        logger.info("任务 %s 模块导入完成", self.job_id)

        self.status = "running"
        self.save()
        self.emit_sse("started", {"total": len(self.urls)})

        need_build = set()
        entity_type = self.entity_type or "doctrine"
        if entity_type not in _ENTITY_TYPES:
            self.status = "failed"
            self.save()
            self.emit_sse("error", {"detail": f"不支持的类型: {entity_type}"})
            return

        for idx, item in enumerate(self.urls):
            # 跳过已完成的
            if idx in self.completed:
                continue

            # 暂停检查
            self._pause_event.wait()
            if self._stop_flag:
                self.status = "paused"
                self.save()
                self.emit_sse("paused", {"current": idx, "total": len(self.urls)})
                return

            url = item["url"]
            entry = {"url": url, "index": idx, "status": "importing", "error": None}

            self.emit_sse("progress", {
                "current": idx,
                "total": len(self.urls),
                "completed": len(self.completed),
                "url": url,
                "title": item.get("title", ""),
                "phase": "fetching",
            })

            try:
                # 抓取
                fetched = _fetch_url_content(url)
                entry["title"] = fetched["title"]
                entry["content_length"] = len(fetched["content"])

                self.emit_sse("progress", {
                    "current": idx,
                    "total": len(self.urls),
                    "completed": len(self.completed),
                    "url": url,
                    "title": fetched["title"],
                    "phase": "llm_processing",
                })

                # LLM 整理
                _, is_single = _ENTITY_TYPES[entity_type]
                entity_id = _title_to_id(fetched["title"])
                entry["type"] = entity_type
                entry["id"] = entity_id

                client = _get_openai_client()
                result = _generate_knowledge(client, fetched["content"], entity_type, entity_id)

                entry["files_generated"] = {}
                for key in ("main_txt", "faq_txt", "alias_txt"):
                    if result.get(key):
                        entry["files_generated"][key.replace("_txt", ".txt")] = len(result[key])

                out_dir = _write_knowledge_files(result, entity_type, entity_id, dry_run=False)
                entry["output_dir"] = str(out_dir)

                if entity_type == "product":
                    need_build.add(entity_id)
                else:
                    need_build.add("_shared")

                entry["status"] = "ok"
                self.completed.append(idx)

                # 从 failed 移除（如果是重试）
                self.failed.pop(idx, None)

            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = str(e)
                self.failed[idx] = str(e)
                logger.error("任务 %s 导入失败: url=%s err=%s", self.job_id, url, e, exc_info=True)

            self.results.append(entry)
            self.save()

            self.emit_sse("item_done", {
                "current": idx,
                "total": len(self.urls),
                "completed": len(self.completed),
                "failed": len(self.failed),
                "status": entry["status"],
                "title": entry.get("title", ""),
                "url": url,
                "error": entry.get("error"),
            })

            if idx < len(self.urls) - 1 and self.delay > 0:
                time.sleep(self.delay)

        # 建索引
        if build and self.completed:
            self.emit_sse("building_index", {"total": len(self.urls), "completed": len(self.completed)})
            try:
                from build_faiss import build_shared, build_for_product
                from rag_answer import invalidate_store_cache
                if "_shared" in need_build:
                    build_shared()
                    invalidate_store_cache("_shared")
                for pid in need_build:
                    if pid != "_shared":
                        build_for_product(pid)
                        invalidate_store_cache(pid)
                self.built_index = True
            except Exception as e:
                logger.error("任务 %s 建索引失败: %s", self.job_id, e, exc_info=True)

        logger.info("任务 %s 执行完毕: 成功=%d 失败=%d", self.job_id, len(self.completed), len(self.failed))
        self.status = "done"
        self.save()
        self.emit_sse("done", {
            "total": len(self.urls),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "built_index": self.built_index,
        })


# ==================== 全局任务管理 ====================

_active_jobs: dict[str, CrawlJob] = {}
_jobs_lock = threading.Lock()


def get_active_job(job_id: str) -> Optional[CrawlJob]:
    with _jobs_lock:
        return _active_jobs.get(job_id)


def set_active_job(job: CrawlJob):
    with _jobs_lock:
        _active_jobs[job.job_id] = job


def remove_active_job(job_id: str):
    with _jobs_lock:
        _active_jobs.pop(job_id, None)
