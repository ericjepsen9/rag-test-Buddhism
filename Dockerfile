FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（先复制 requirements 利用缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY *.py ./
COPY web/ ./web/
COPY admin_page.html ./

# 数据目录（运行时挂载或构建时复制）
# 知识库和索引建议通过 volume 挂载：
#   -v /path/to/knowledge:/app/knowledge
#   -v /path/to/stores:/app/stores
#   -v /path/to/data:/app/data
#   -v /path/to/logs:/app/logs
RUN mkdir -p knowledge stores data logs

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
