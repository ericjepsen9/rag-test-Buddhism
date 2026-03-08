# RAG 后台增强包

## 包含内容
- `admin_api.py`：后台管理接口
- `admin_page.html`：后台管理页面
- `rag_logger.py`：问答/未命中/异常日志模块
- `templates/knowledge_product_template/`：标准知识库录入模板

## 启动方式
1. 将这些文件复制到你的项目根目录。
2. 确保已安装 FastAPI / Uvicorn。
3. 启动：
   `uvicorn admin_api:app --host 0.0.0.0 --port 8010`
4. 浏览器打开：
   `http://127.0.0.1:8010/`

## 说明
- 这个增强包不会直接改你的 `rag_answer.py`，主要是补管理、日志和模板。
- 若你希望自动记录问答，需要在 `rag_answer.py` 或 `api_server.py` 里调用 `rag_logger.log_qa(...)`。
