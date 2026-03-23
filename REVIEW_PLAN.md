# 全面检查计划

## 一、前后端接口一致性检查

### 1.1 后端有路由但前端（admin_page.html）未调用的孤立接口
| 接口 | 方法 | api_server.py 行号 | 说明 |
|------|------|-------------------|------|
| `/admin/debug` | POST | 1549 | 调试开关，前端无入口 |
| `/admin/config` | GET/POST | 1560/1571 | 通用配置读写，前端未使用 |
| `/admin/config/model` | GET/POST | 1579/1593 | 模型切换，前端未使用 |
| `/admin/config/server` | GET/POST | 1722/1733 | 服务器配置，前端未使用 |
| `/admin/config/nginx` | POST | 1749 | Nginx 配置生成，前端无入口 |
| `/admin/cache` | GET | 1857 | 缓存统计，前端无入口 |
| `/admin/cache/clear` | POST | 1884 | 清空缓存，前端无入口 |
| `/admin/import_knowledge_file` | POST | 2310 | 单文件导入，前端未调用 |
| `/admin/synonyms/learned` | GET | 946 | 仅学习词列表（前端用 /all 代替） |
| `/admin/synonyms/learned` | PUT | 1000 | 编辑同义词，前端无编辑 UI |

### 1.2 mobile.html 覆盖检查
- mobile.html 调用的 5 个接口均存在后端路由 ✓
- `loadProducts()` 失败时静默 `.catch(() => {})`，用户无感知 — 建议加 toast

### 1.3 需要确认的问题
- [ ] 孤立接口是否为预留 API？若不需要，考虑清理或在前端补 UI
- [ ] `/admin/cache` 和 `/admin/cache/clear` 是否应在系统 Tab 加缓存管理卡片？
- [ ] `/admin/config/*` 系列是否应在系统 Tab 呈现？

---

## 二、测试覆盖缺口

### 2.1 完全无测试的 admin 接口（26 个）
**服务管理（Phase 7A/7B）— 0/5 覆盖：**
- `/admin/service/embedding/start` POST
- `/admin/service/embedding/stop` POST
- `/admin/service/llm/start` POST
- `/admin/service/llm/stop` POST
- `/admin/service/llm/test` POST

**LLM 配置 — 0/1 覆盖：**
- `/admin/llm/configs` POST（写入）

**知识库管理 — 0/6 覆盖：**
- `/admin/knowledge/{product}` GET/DELETE
- `/admin/knowledge/{product}/{filename}` GET/PUT/DELETE
- `/admin/knowledge/create_product` POST

**导入流程 — 0/3 覆盖：**
- `/admin/import_knowledge/refine` POST
- `/admin/import_knowledge/commit` POST
- `/admin/import_knowledge_file` POST

**同义词操作 — 0/7 覆盖：**
- `/admin/synonyms/export` GET
- `/admin/synonyms/import` POST
- `/admin/synonyms/learned/add` POST
- `/admin/synonyms/learned/approve` POST
- `/admin/synonyms/learned/batch-approve` POST
- `/admin/synonyms/learned/batch-delete` POST
- `/admin/synonyms/reload` POST

**其他 — 0/4 覆盖：**
- `/admin/rebuild_shared` POST
- `/admin/upload` POST
- `/admin/upload_zip` POST
- `/admin/keywords/llm-expand` POST

### 2.2 测试引用了不存在的功能
- `test_api.py` 中 `test_admin_query_param_auth`（约 L226）测试 query param 认证 `?admin_key=...`，但 `admin_auth_middleware` 仅支持 `Authorization: Bearer` header — **测试与实现不符**

### 2.3 断言质量
- 所有现有测试函数均包含 assert ✓ 无空断言

---

## 三、代码质量检查

### 3.1 错误处理
- [ ] `api_server.py` 所有 admin 端点的异常是否统一返回格式？
- [ ] `/admin/service/embedding/start` 超时风险（模型加载可能很慢），是否需要异步化？
- [ ] LLM test 端点每次新建 OpenAI client 而非复用已缓存 client — 是否合理？

### 3.2 安全检查
- [ ] 所有 `/admin/*` 路由是否均受 `admin_auth_middleware` 保护？
- [ ] 文件上传是否有路径遍历防护（`../` 攻击）？
- [ ] `admin_knowledge_write` 是否限制文件大小？
- [ ] `admin_fetch_url` 是否有 SSRF 防护？

### 3.3 设计不对称
- `stop_llm_service()` 持久化 `{"use_openai": False}` 到磁盘
- `start_llm_service()` **不**持久化 — 重启后 LLM 默认关闭
- 建议：start 时也 `_persist_overrides({"use_openai": True})`

---

## 四、前端 UI 完整性

### 4.1 系统 Tab 缺失功能
- [ ] 缓存管理（统计 + 清空按钮）— 后端已有 `/admin/cache` 和 `/admin/cache/clear`
- [ ] 调试开关 — 后端已有 `/admin/debug`
- [ ] 服务器配置（端口/host 等）— 后端已有 `/admin/config/server`

### 4.2 mobile.html 细节
- [ ] `loadProducts()` 静默失败，建议加错误提示
- [ ] fetch 请求无超时机制

---

## 五、执行优先级

| 优先级 | 类别 | 具体项 | 工作量 |
|--------|------|--------|--------|
| P0 | 测试 | 修复 `test_admin_query_param_auth` 测试与实现不符 | 小 |
| P0 | 安全 | 检查文件上传路径遍历防护 | 小 |
| P1 | 设计 | `start_llm_service` 添加持久化 | 小 |
| P1 | 前端 | 系统 Tab 加缓存管理卡片 | 中 |
| P1 | 测试 | 补充服务管理端点测试（5 个） | 中 |
| P2 | 前端 | 系统 Tab 加调试开关/服务器配置 | 中 |
| P2 | 测试 | 补充知识库管理端点测试（6 个） | 中 |
| P2 | 测试 | 补充同义词操作端点测试（7 个） | 中 |
| P3 | 清理 | 评估孤立接口是否保留或删除 | 小 |
| P3 | 前端 | mobile.html 错误处理增强 | 小 |
