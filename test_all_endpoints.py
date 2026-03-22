#!/usr/bin/env python3
"""
RAG 系统 API 端点综合测试脚本
Comprehensive test script for all key API endpoints of the RAG system.

用法 / Usage:
    python test_all_endpoints.py --key YOUR_ADMIN_KEY
    python test_all_endpoints.py --key YOUR_KEY --test-url "https://example.com/article"
    python test_all_endpoints.py --host 192.168.1.10 --port 9000 --key YOUR_KEY
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# ============================================================
# 终端颜色定义 - ANSI escape codes for colored terminal output
# ============================================================

class Colors:
    """终端颜色常量"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    """给文本添加终端颜色"""
    return f"{color}{text}{Colors.RESET}"


# ============================================================
# 测试结果数据结构 - Data structure for individual test results
# ============================================================

class TestResult:
    """单个测试用例的结果记录"""

    def __init__(self, step: int, name: str, method: str, endpoint: str):
        self.step = step
        self.name = name
        self.method = method
        self.endpoint = endpoint
        self.status = "SKIP"
        self.status_code: Optional[int] = None
        self.request_body: Optional[str] = None
        self.response_body: Optional[str] = None
        self.duration_ms: float = 0.0
        self.error_message: Optional[str] = None
        self.skip_reason: Optional[str] = None

    def mark_pass(self, status_code: int, response_body: str, duration_ms: float):
        self.status = "PASS"
        self.status_code = status_code
        self.response_body = _truncate(response_body, 2000)
        self.duration_ms = duration_ms

    def mark_fail(self, error_message: str, status_code: Optional[int] = None,
                  response_body: Optional[str] = None, duration_ms: float = 0.0):
        self.status = "FAIL"
        self.error_message = error_message
        self.status_code = status_code
        self.response_body = _truncate(response_body, 2000) if response_body else None
        self.duration_ms = duration_ms

    def mark_skip(self, reason: str):
        self.status = "SKIP"
        self.skip_reason = reason

    @property
    def status_colored(self) -> str:
        if self.status == "PASS":
            return colored("PASS", Colors.GREEN)
        elif self.status == "FAIL":
            return colored("FAIL", Colors.RED)
        else:
            return colored("SKIP", Colors.YELLOW)

    @property
    def status_plain(self) -> str:
        return self.status


def _truncate(text: str, max_len: int) -> str:
    if text and len(text) > max_len:
        return text[:max_len] + f"\n... [truncated, total {len(text)} chars]"
    return text


# ============================================================
# API 测试运行器 - Main test runner class
# ============================================================

class RAGApiTester:
    """RAG 系统 API 综合测试器"""

    def __init__(self, host: str, port: int, api_key: str, test_url: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.test_url = test_url
        self.results: List[TestResult] = []
        self.step_counter = 0
        self.discovered_products: List[str] = []
        self.fetched_media: List[Dict] = []
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _next_step(self) -> int:
        self.step_counter += 1
        return self.step_counter

    def _request(self, method: str, path: str, json_body: Optional[Dict] = None,
                 params: Optional[Dict] = None, timeout: int = 60) -> TestResult:
        step = self._next_step()
        result = TestResult(step=step, name="", method=method, endpoint=path)
        if json_body is not None:
            result.request_body = _truncate(json.dumps(json_body, ensure_ascii=False, indent=2), 1000)
        url = f"{self.base_url}{path}"
        try:
            start = time.time()
            resp = requests.request(method=method, url=url, headers=self.headers,
                                    json=json_body, params=params, timeout=timeout)
            elapsed_ms = (time.time() - start) * 1000
            try:
                body_text = resp.text
            except Exception:
                body_text = "<unable to decode response body>"
            if 200 <= resp.status_code < 300:
                result.mark_pass(resp.status_code, body_text, elapsed_ms)
            else:
                result.mark_fail(f"HTTP {resp.status_code}", resp.status_code, body_text, elapsed_ms)
        except requests.exceptions.ConnectionError as exc:
            result.mark_fail(f"Connection error: {exc}")
        except requests.exceptions.Timeout:
            result.mark_fail(f"Request timed out after {timeout}s")
        except Exception as exc:
            result.mark_fail(f"Unexpected error: {type(exc).__name__}: {exc}")
        return result

    def _run_test(self, name: str, method: str, path: str,
                  json_body: Optional[Dict] = None, params: Optional[Dict] = None,
                  skip_reason: Optional[str] = None, timeout: int = 60) -> TestResult:
        if skip_reason:
            step = self._next_step()
            result = TestResult(step=step, name=name, method=method, endpoint=path)
            result.mark_skip(skip_reason)
            self.results.append(result)
            self._print_result(result)
            return result
        step_preview = self.step_counter + 1
        print(f"\n{Colors.CYAN}[{step_preview}]{Colors.RESET} {Colors.BOLD}{name}{Colors.RESET}")
        print(f"    {Colors.DIM}{method} {path}{Colors.RESET}", end="", flush=True)
        result = self._request(method, path, json_body=json_body, params=params, timeout=timeout)
        result.name = name
        self.results.append(result)
        self._print_result_inline(result)
        return result

    def _print_result_inline(self, result: TestResult):
        duration_str = f"{result.duration_ms:.0f}ms"
        code_str = f"[{result.status_code}]" if result.status_code else ""
        print(f"  -> {result.status_colored} {code_str} ({duration_str})")
        if result.status == "FAIL" and result.error_message:
            print(f"    {Colors.RED}Error: {result.error_message}{Colors.RESET}")

    def _print_result(self, result: TestResult):
        print(f"\n{Colors.CYAN}[{result.step}]{Colors.RESET} {Colors.BOLD}{result.name}{Colors.RESET}")
        print(f"    {Colors.DIM}{result.method} {result.endpoint}{Colors.RESET}")
        if result.skip_reason:
            print(f"    -> {result.status_colored} ({result.skip_reason})")
        else:
            self._print_result_inline(result)

    def _get_json(self, result: TestResult) -> Optional[Any]:
        if result.status != "PASS" or not result.response_body:
            return None
        try:
            return json.loads(result.response_body.split("\n... [truncated")[0])
        except (json.JSONDecodeError, ValueError):
            return None

    # --------------------------------------------------------
    # 测试用例组
    # --------------------------------------------------------

    def test_health(self):
        self._run_test("Health Check", "GET", "/health")

    def test_stats(self):
        self._run_test("Admin Stats", "GET", "/admin/stats")

    def test_products(self):
        result = self._run_test("Products List", "GET", "/admin/products")
        data = self._get_json(result)
        if data:
            if isinstance(data, list):
                self.discovered_products = [
                    (p.get("name") or p.get("product") or p.get("id") or str(p))
                    if isinstance(p, dict) else str(p) for p in data
                ]
            elif isinstance(data, dict):
                items = data.get("products") or data.get("items") or data.get("data") or []
                if isinstance(items, list):
                    self.discovered_products = [
                        (p.get("name") or p.get("product") or p.get("id") or str(p))
                        if isinstance(p, dict) else str(p) for p in items
                    ]
        if self.discovered_products:
            print(f"    {Colors.DIM}Discovered entries: {self.discovered_products}{Colors.RESET}")

    def test_config(self):
        self._run_test("Config (General)", "GET", "/admin/config")
        self._run_test("Config (Model)", "GET", "/admin/config/model")

    def test_services(self):
        self._run_test("Service: Embedding", "GET", "/admin/service/embedding")
        self._run_test("Service: LLM", "GET", "/admin/service/llm")

    def test_llm_configs(self):
        self._run_test("LLM Configs", "GET", "/admin/llm/configs")

    def test_knowledge(self):
        if not self.discovered_products:
            self._run_test("Knowledge List", "GET", "/admin/knowledge/{product}",
                           skip_reason="No entries discovered from previous test")
            return
        product = self.discovered_products[0]
        self._run_test(f"Knowledge List (entry={product})", "GET", f"/admin/knowledge/{product}")

    def test_synonyms(self):
        self._run_test("Synonyms (All)", "GET", "/admin/synonyms/all")

    def test_ask(self):
        body = {"question": "什么是四圣谛"}
        if self.discovered_products:
            body["product"] = self.discovered_products[0]
        self._run_test("Ask (Search/QA)", "POST", "/ask", json_body=body, timeout=120)

    def test_fetch_url(self):
        if not self.test_url:
            self._run_test("Fetch URL", "POST", "/admin/fetch_url",
                           skip_reason="No --test-url provided")
            return
        body = {"url": self.test_url}
        result = self._run_test("Fetch URL", "POST", "/admin/fetch_url",
                                json_body=body, timeout=120)
        data = self._get_json(result)
        if data and isinstance(data, dict):
            self.fetched_media = data.get("media") or data.get("images") or []

    def test_proxy_media(self):
        if not self.fetched_media:
            self._run_test("Proxy Media", "GET", "/admin/proxy_media",
                           skip_reason="No media available from fetch_url result")
            return
        first_media = self.fetched_media[0]
        media_url = first_media if isinstance(first_media, str) else (
            first_media.get("url") or first_media.get("src") or "")
        if not media_url:
            self._run_test("Proxy Media", "GET", "/admin/proxy_media",
                           skip_reason="Could not extract media URL from fetch_url response")
            return
        self._run_test("Proxy Media", "GET", "/admin/proxy_media", params={"url": media_url})

    def test_save_media(self):
        body = {"media_list": [], "product": self.discovered_products[0] if self.discovered_products else "test"}
        self._run_test("Save Media (empty list)", "POST", "/admin/save_media", json_body=body)

    def test_import_knowledge(self):
        body = {
            "product": self.discovered_products[0] if self.discovered_products else "test",
            "content": "This is a test knowledge entry for dry-run import validation.",
            "title": "API Test - Dry Run",
            "dry_run": True,
        }
        self._run_test("Import Knowledge (dry_run=true)", "POST", "/admin/import_knowledge",
                        json_body=body, timeout=120)

    def test_llm_test(self):
        self._run_test("LLM Test", "POST", "/admin/llm/test", json_body={}, timeout=120)

    def test_cache(self):
        self._run_test("Cache Status", "GET", "/admin/cache")

    def test_logs(self):
        self._run_test("Logs: QA", "GET", "/admin/logs/qa")
        self._run_test("Logs: Miss", "GET", "/admin/logs/miss")
        self._run_test("Logs: Error", "GET", "/admin/logs/error")

    def test_rebuild(self):
        if not self.discovered_products:
            self._run_test("Index Rebuild", "POST", "/admin/rebuild",
                           skip_reason="No entries available; skipping rebuild")
            return
        product = self.discovered_products[0]
        self._run_test(f"Index Rebuild (entry={product})", "POST", "/admin/rebuild",
                        json_body={"product": product}, timeout=300)

    # --------------------------------------------------------
    # 主执行流程
    # --------------------------------------------------------

    def run_all(self):
        banner = "RAG System - Comprehensive API Endpoint Tests"
        print(f"\n{'=' * 60}")
        print(colored(f"  {banner}", Colors.BOLD))
        print(f"{'=' * 60}")
        print(f"  Base URL : {self.base_url}")
        print(f"  API Key  : {self.api_key[:8]}{'*' * (len(self.api_key) - 8) if len(self.api_key) > 8 else ''}")
        print(f"  Test URL : {self.test_url or '(not provided)'}")
        print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        test_groups = [
            ("Health Check", self.test_health),
            ("Stats", self.test_stats),
            ("Products", self.test_products),
            ("Config", self.test_config),
            ("Services", self.test_services),
            ("LLM Configs", self.test_llm_configs),
            ("Knowledge", self.test_knowledge),
            ("Synonyms", self.test_synonyms),
            ("Search/Ask", self.test_ask),
            ("Fetch URL", self.test_fetch_url),
            ("Proxy Media", self.test_proxy_media),
            ("Save Media", self.test_save_media),
            ("Import Knowledge", self.test_import_knowledge),
            ("LLM Test", self.test_llm_test),
            ("Cache", self.test_cache),
            ("Logs", self.test_logs),
            ("Index Rebuild", self.test_rebuild),
        ]

        for group_name, test_fn in test_groups:
            print(f"\n{Colors.BOLD}--- {group_name} ---{Colors.RESET}")
            try:
                test_fn()
            except Exception as exc:
                print(colored(f"  Unexpected error in test group '{group_name}': {exc}", Colors.RED))

        self._generate_report()

    def _generate_report(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        print(f"\n{'=' * 60}")
        print(colored("  TEST SUMMARY", Colors.BOLD))
        print(f"{'=' * 60}")
        print(f"  Total  : {total}")
        print(f"  Passed : {colored(str(passed), Colors.GREEN)}")
        print(f"  Failed : {colored(str(failed), Colors.RED)}")
        print(f"  Skipped: {colored(str(skipped), Colors.YELLOW)}")
        print(f"{'=' * 60}")

        if failed == 0:
            print(colored("\n  All executed tests passed!\n", Colors.GREEN))
        else:
            print(colored(f"\n  {failed} test(s) failed. See details below.\n", Colors.RED))

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("  RAG SYSTEM API TEST REPORT")
        report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"  Base URL : {self.base_url}")
        report_lines.append(f"  Test URL : {self.test_url or '(not provided)'}")
        report_lines.append("=" * 70)
        report_lines.append("")
        report_lines.append(f"SUMMARY: Total={total}  Pass={passed}  Fail={failed}  Skip={skipped}")
        report_lines.append("")
        report_lines.append("-" * 70)

        for r in self.results:
            report_lines.append("")
            report_lines.append(f"[{r.step}] {r.status_plain} | {r.name}")
            report_lines.append(f"    Method   : {r.method} {r.endpoint}")
            report_lines.append(f"    Status   : HTTP {r.status_code}" if r.status_code else f"    Status   : N/A")
            report_lines.append(f"    Duration : {r.duration_ms:.0f}ms")
            if r.skip_reason:
                report_lines.append(f"    Skipped  : {r.skip_reason}")
            if r.error_message:
                report_lines.append(f"    Error    : {r.error_message}")
            if r.request_body:
                report_lines.append(f"    Request  :")
                for line in r.request_body.splitlines():
                    report_lines.append(f"        {line}")
            if r.response_body:
                report_lines.append(f"    Response :")
                for line in r.response_body.splitlines():
                    report_lines.append(f"        {line}")
            report_lines.append("-" * 70)

        report_text = "\n".join(report_lines) + "\n"
        report_path = "test_report.txt"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"  Report saved to: {Colors.CYAN}{report_path}{Colors.RESET}")
        except IOError as exc:
            print(colored(f"  Failed to save report: {exc}", Colors.RED))

        print(f"\n{'=' * 60}")
        print(colored("  DETAILED RESULTS", Colors.BOLD))
        print(f"{'=' * 60}")
        for r in self.results:
            duration_str = f"{r.duration_ms:.0f}ms"
            print(f"\n  [{r.step}] {r.status_colored} | {r.name}")
            print(f"       {r.method} {r.endpoint}  ({duration_str})")
            if r.status_code:
                print(f"       HTTP {r.status_code}")
            if r.skip_reason:
                print(f"       {colored('Reason: ' + r.skip_reason, Colors.YELLOW)}")
            if r.error_message:
                print(f"       {colored('Error: ' + r.error_message, Colors.RED)}")
        print(f"\n{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG 系统 API 端点综合测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="localhost", help="API server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    parser.add_argument("--key", required=True, help="Admin API key (ADMIN_API_KEY)")
    parser.add_argument("--test-url", default=None, help="Optional article URL for fetch_url test")
    return parser.parse_args()


def main():
    args = parse_args()
    tester = RAGApiTester(host=args.host, port=args.port, api_key=args.key, test_url=args.test_url)
    tester.run_all()
    failed = sum(1 for r in tester.results if r.status == "FAIL")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
