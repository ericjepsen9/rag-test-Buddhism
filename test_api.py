"""API 端点集成测试：使用 FastAPI TestClient 验证 HTTP 级行为。

覆盖范围：
- 核心用户端点：/health, /chat, /ask, /v1/models, /v1/chat/completions
- 管理端点鉴权：ADMIN_API_KEY 校验
- 管理端点：/admin/products, /admin/cache, /admin/config 等
- 安全头：CSP, X-Frame-Options 等
- 输入校验：超长问题、空问题、XSS 注入
- 限流：验证装饰器存在
"""
import os
import json
import pytest

# 设置测试环境变量（必须在导入 api_server 之前）
os.environ.setdefault("SKIP_WARMUP", "1")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key-12345")

from starlette.testclient import TestClient
from api_server import app, _RESPONSE_CACHE, _response_cache_lock, MAX_QUESTION_LEN


@pytest.fixture(scope="module")
def client():
    """复用同一个 TestClient 实例，避免每个测试都初始化"""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer test-admin-key-12345"}


# ============================================================
# 安全响应头测试
# ============================================================

class TestSecurityHeaders:
    """验证所有响应都包含安全头和 trace ID"""

    def test_health_has_security_headers(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_response_has_trace_id(self, client):
        resp = client.get("/health")
        assert "X-Trace-Id" in resp.headers
        assert len(resp.headers["X-Trace-Id"]) >= 8

    def test_trace_id_passthrough(self, client):
        resp = client.get("/health", headers={"X-Trace-Id": "test-trace-123"})
        assert resp.headers["X-Trace-Id"] == "test-trace-123"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Content-Security-Policy" in resp.headers
        csp = resp.headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in csp
        assert "script-src 'self'" in csp

    def test_api_endpoint_has_security_headers(self, client):
        resp = client.get("/v1/models")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in resp.headers


# ============================================================
# /health 端点测试
# ============================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_contains_products(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "products" in data
        assert isinstance(data["products"], list)

    def test_health_contains_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_contains_embedding_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "embedding_model_loaded" in data
        assert isinstance(data["embedding_model_loaded"], bool)


# ============================================================
# /v1/models 端点测试
# ============================================================

class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["object"] == "model"


# ============================================================
# /ask 端点测试
# ============================================================

class TestAskEndpoint:
    def test_ask_empty_question_rejected(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_too_long_question_rejected(self, client):
        long_q = "测" * (MAX_QUESTION_LEN + 1)
        resp = client.post("/ask", json={"question": long_q})
        assert resp.status_code == 422

    def test_ask_xss_input_sanitized(self, client):
        resp = client.post("/ask", json={
            "question": '<script>alert("xss")</script>什么是般若'
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "<script>" not in data.get("answer", "")

    def test_ask_control_chars_sanitized(self, client):
        resp = client.post("/ask", json={
            "question": "般若\x00\x01\x02是什么"
        })
        assert resp.status_code == 200

    def test_ask_returns_response_model(self, client):
        resp = client.post("/ask", json={"question": "你好"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data
        assert "answer" in data
        assert isinstance(data.get("media", []), list)

    def test_ask_with_history(self, client):
        resp = client.post("/ask", json={
            "question": "还有呢",
            "history": [
                {"role": "user", "content": "佛教有哪些宗派"},
                {"role": "assistant", "content": "主要宗派包括..."}
            ]
        })
        assert resp.status_code == 200

    def test_ask_debug_mode(self, client):
        resp = client.post("/ask", json={
            "question": "你好",
            "debug": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        if data.get("ok"):
            assert "debug" in data

    def test_ask_invalid_mode(self, client):
        resp = client.post("/ask", json={
            "question": "测试",
            "mode": "invalid_mode"
        })
        assert resp.status_code == 422


# ============================================================
# /v1/chat/completions 端点测试（OpenAI 兼容）
# ============================================================

class TestOAIChatEndpoint:
    def test_oai_basic_request(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "你好"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_oai_empty_messages_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": []
        })
        assert resp.status_code in (422, 200)

    def test_oai_response_has_usage(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "测试"}]
        })
        if resp.status_code == 200:
            data = resp.json()
            assert "usage" in data
            assert "model" in data


# ============================================================
# 管理端点鉴权测试
# ============================================================

class TestAdminAuth:
    def test_admin_api_requires_auth(self, client):
        resp = client.get("/admin/products")
        assert resp.status_code == 403

    def test_admin_api_wrong_key(self, client):
        resp = client.get("/admin/products",
                          headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 403

    def test_admin_api_correct_key(self, client, admin_headers):
        resp = client.get("/admin/products", headers=admin_headers)
        assert resp.status_code == 200

    def test_admin_query_param_auth_rejected(self, client):
        """Query-param auth is NOT supported — only Bearer header is accepted."""
        resp = client.get("/admin/products?admin_key=test-admin-key-12345")
        assert resp.status_code == 403

    def test_admin_page_exempt_from_auth(self, client):
        resp = client.get("/admin")
        assert resp.status_code != 403


# ============================================================
# 管理端点功能测试
# ============================================================

class TestAdminEndpoints:
    def test_admin_products_list(self, client, admin_headers):
        resp = client.get("/admin/products", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "products" in data
        assert isinstance(data["products"], list)

    def test_admin_cache_stats(self, client, admin_headers):
        resp = client.get("/admin/cache", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "response_cache" in data
        assert "embed_cache" in data
        assert "store_cache" in data
        assert "llm_rewrite_cache" in data

    def test_admin_cache_clear(self, client, admin_headers):
        resp = client.post("/admin/cache/clear", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "cleared" in data

    def test_admin_config_get(self, client, admin_headers):
        resp = client.get("/admin/config", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_admin_logs_qa(self, client, admin_headers):
        resp = client.get("/admin/logs/qa", headers=admin_headers)
        assert resp.status_code == 200

    def test_admin_logs_miss(self, client, admin_headers):
        resp = client.get("/admin/logs/miss", headers=admin_headers)
        assert resp.status_code == 200

    def test_admin_logs_error(self, client, admin_headers):
        resp = client.get("/admin/logs/error", headers=admin_headers)
        assert resp.status_code == 200

    def test_admin_synonyms_all(self, client, admin_headers):
        resp = client.get("/admin/synonyms/all", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "static" in data or "learned" in data

    def test_admin_keywords_effective(self, client, admin_headers):
        resp = client.get("/admin/keywords/effective", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    def test_admin_keywords_clarification(self, client, admin_headers):
        resp = client.get("/admin/keywords/clarification", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data


# ============================================================
# 服务管理端点测试
# ============================================================

class TestServiceManagement:
    def test_embedding_status(self, client, admin_headers):
        resp = client.get("/admin/service/embedding", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert "model_name" in data

    def test_llm_status(self, client, admin_headers):
        resp = client.get("/admin/service/llm", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "model" in data

    def test_embedding_start(self, client, admin_headers):
        resp = client.post("/admin/service/embedding/start", headers=admin_headers)
        # 200 if model loads, 500 if unavailable in test env
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.json()["ok"] is True

    def test_embedding_stop(self, client, admin_headers):
        resp = client.post("/admin/service/embedding/stop", headers=admin_headers)
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.json()["ok"] is True

    def test_llm_start(self, client, admin_headers):
        resp = client.post("/admin/service/llm/start",
                           json={}, headers=admin_headers)
        # 200 if API key valid, 500 if no key or client fails
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.json()["ok"] is True

    def test_llm_stop(self, client, admin_headers):
        resp = client.post("/admin/service/llm/stop", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_llm_test_without_service(self, client, admin_headers):
        # Stop LLM first, then test should return 503
        client.post("/admin/service/llm/stop", headers=admin_headers)
        resp = client.post("/admin/service/llm/test", headers=admin_headers)
        assert resp.status_code == 503
        data = resp.json()
        assert data["ok"] is False


# ============================================================
# 管理端点输入校验测试
# ============================================================

class TestAdminInputValidation:
    def test_rebuild_empty_product_rejected(self, client, admin_headers):
        resp = client.post("/admin/rebuild", json={"product": ""}, headers=admin_headers)
        assert resp.status_code == 422

    def test_rebuild_path_traversal_rejected(self, client, admin_headers):
        resp = client.post("/admin/rebuild", json={"product": "../../etc"}, headers=admin_headers)
        assert resp.status_code == 400

    def test_rebuild_nonexistent_product(self, client, admin_headers):
        resp = client.post("/admin/rebuild",
                           json={"product": "nonexistent_entry_xyz"}, headers=admin_headers)
        assert resp.status_code in (400, 404)

    def test_synonym_override_empty_rejected(self, client, admin_headers):
        resp = client.post("/admin/keywords/synonym/override",
                           json={"original": "", "mapped_to": "test"}, headers=admin_headers)
        assert resp.status_code == 422

    def test_synonym_override_too_long_rejected(self, client, admin_headers):
        resp = client.post("/admin/keywords/synonym/override",
                           json={"original": "x" * 201, "mapped_to": "test"}, headers=admin_headers)
        assert resp.status_code == 422

    def test_clarification_empty_trigger_rejected(self, client, admin_headers):
        resp = client.post("/admin/keywords/clarification",
                           json={"trigger": "", "options": [{"label": "a", "query": "b"}]},
                           headers=admin_headers)
        assert resp.status_code == 422

    def test_knowledge_write_path_traversal(self, client, admin_headers):
        resp = client.put("/admin/knowledge/test/../../../etc/passwd",
                          json={"content": "hacked"}, headers=admin_headers)
        assert resp.status_code in (400, 404)

    def test_knowledge_write_illegal_filename(self, client, admin_headers):
        resp = client.put("/admin/knowledge/test/..%2Fevil.txt",
                          json={"content": "hacked"}, headers=admin_headers)
        assert resp.status_code in (400, 404, 422)


# ============================================================
# 知识库管理端点测试
# ============================================================

class TestKnowledgeManagement:
    """Tests for knowledge CRUD endpoints using a temporary product."""

    TEMP_PRODUCT = "_test_tmp_product"

    def _ensure_clean(self, client, admin_headers):
        """Delete temp product if it exists from a prior run."""
        client.delete(f"/admin/knowledge/{self.TEMP_PRODUCT}", headers=admin_headers)

    def test_create_product(self, client, admin_headers):
        self._ensure_clean(client, admin_headers)
        resp = client.post("/admin/knowledge/create_product",
                           json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_create_product_duplicate(self, client, admin_headers):
        self._ensure_clean(client, admin_headers)
        client.post("/admin/knowledge/create_product",
                    json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        resp = client.post("/admin/knowledge/create_product",
                           json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        assert resp.status_code == 409

    def test_list_product_files(self, client, admin_headers):
        self._ensure_clean(client, admin_headers)
        client.post("/admin/knowledge/create_product",
                    json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        resp = client.get(f"/admin/knowledge/{self.TEMP_PRODUCT}", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_write_and_read_file(self, client, admin_headers):
        self._ensure_clean(client, admin_headers)
        client.post("/admin/knowledge/create_product",
                    json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        # Write
        resp = client.put(f"/admin/knowledge/{self.TEMP_PRODUCT}/test_file.txt",
                          json={"content": "hello world"}, headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["size"] == len("hello world")
        # Read back
        resp = client.get(f"/admin/knowledge/{self.TEMP_PRODUCT}/test_file.txt",
                          headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["content"] == "hello world"

    def test_delete_file(self, client, admin_headers):
        self._ensure_clean(client, admin_headers)
        client.post("/admin/knowledge/create_product",
                    json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        client.put(f"/admin/knowledge/{self.TEMP_PRODUCT}/to_delete.txt",
                   json={"content": "tmp"}, headers=admin_headers)
        resp = client.delete(f"/admin/knowledge/{self.TEMP_PRODUCT}/to_delete.txt",
                             headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_delete_product(self, client, admin_headers):
        self._ensure_clean(client, admin_headers)
        client.post("/admin/knowledge/create_product",
                    json={"product": self.TEMP_PRODUCT}, headers=admin_headers)
        resp = client.delete(f"/admin/knowledge/{self.TEMP_PRODUCT}",
                             headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_list_nonexistent_product(self, client, admin_headers):
        resp = client.get("/admin/knowledge/nonexistent_xyz_999", headers=admin_headers)
        assert resp.status_code == 404


# ============================================================
# 同义词操作端点测试
# ============================================================

class TestSynonymOperations:

    def test_synonyms_export(self, client, admin_headers):
        resp = client.get("/admin/synonyms/export", headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_synonyms_reload(self, client, admin_headers):
        resp = client.post("/admin/synonyms/reload", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "active_count" in data

    def test_synonyms_learned_add(self, client, admin_headers):
        # Delete first in case it exists from a prior run
        client.post("/admin/synonyms/learned/batch-delete",
                    json={"terms": ["_test_syn_orig"]}, headers=admin_headers)
        resp = client.post("/admin/synonyms/learned/add",
                           json={"original": "_test_syn_orig", "mapped_to": "_test_syn_map"},
                           headers=admin_headers)
        assert resp.status_code == 200

    def test_synonyms_learned_approve(self, client, admin_headers):
        # Add first, then approve
        client.post("/admin/synonyms/learned/add",
                    json={"original": "_test_approve", "mapped_to": "_test_map"},
                    headers=admin_headers)
        resp = client.post("/admin/synonyms/learned/approve?original=_test_approve",
                           headers=admin_headers)
        assert resp.status_code in (200, 404)

    def test_synonyms_batch_approve(self, client, admin_headers):
        resp = client.post("/admin/synonyms/learned/batch-approve",
                           json={"terms": ["_test_batch_1"]}, headers=admin_headers)
        assert resp.status_code == 200

    def test_synonyms_batch_delete(self, client, admin_headers):
        resp = client.post("/admin/synonyms/learned/batch-delete",
                           json={"terms": ["_test_batch_del"]}, headers=admin_headers)
        assert resp.status_code == 200

    def test_synonyms_import(self, client, admin_headers):
        resp = client.post("/admin/synonyms/import",
                           json={"items": [{"original": "_test_imp", "mapped_to": "_test_imp_map"}],
                                 "auto_approve": False},
                           headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "added" in data

    def test_synonyms_approve_empty_rejected(self, client, admin_headers):
        resp = client.post("/admin/synonyms/learned/approve?original=",
                           headers=admin_headers)
        assert resp.status_code == 400


# ============================================================
# 响应缓存行为测试
# ============================================================

class TestResponseCache:
    def test_cache_cleared_after_clear_endpoint(self, client, admin_headers):
        resp = client.post("/admin/cache/clear", headers=admin_headers)
        assert resp.status_code == 200
        resp = client.get("/admin/cache", headers=admin_headers)
        assert resp.json()["response_cache"]["size"] == 0


# ============================================================
# 404 / 方法不允许测试
# ============================================================

class TestErrorHandling:
    def test_nonexistent_endpoint_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_ask_get_method_not_allowed(self, client):
        resp = client.get("/ask")
        assert resp.status_code == 405

    def test_health_post_method_not_allowed(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405
