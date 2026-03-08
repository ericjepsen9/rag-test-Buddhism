# 最终增强生产版（全功能整合）

## 已包含功能
- 问题改写（Query Rewrite）
- 向量检索 + 关键词检索融合（Hybrid Search）
- 多实体识别（多产品 / 多项目 / 时间词 / 症状词）
- 章节优先回答（直接从 knowledge/main.txt 抽章节，避免回答缺段）
- 未命中兜底模板
- 证据引用（来源文件 / 段落 / 类型）
- 风险类自动追加“仅供参考，需医生评估”
- OpenAI 开关（默认关闭）
- brief / full 模式
- FastAPI 接口
- 媒体路由（返回相关图片/视频条目）
- 回归测试脚本

## 文件职责
- rag_runtime_config.py：统一配置入口
- search_utils.py：文本清洗 / section 抽取 / hybrid 合并 / 多问拆分
- query_rewrite.py：问题改写与多实体识别
- answer_formatter.py：结构化输出模板
- build_faiss.py：重建单产品索引
- rag_answer.py：主回答入口
- api_server.py：HTTP 接口
- rag_media_config.py / media_router.py / media.json：图文视频返回
- run_regression.py / regression_cases.json：回归测试
- knowledge/feiluoao/*：示例知识库

## 替换顺序
1. 覆盖代码文件到项目根目录
2. 覆盖 knowledge/<产品>/main.txt, faq.txt, alias.txt
3. 运行：python build_faiss.py --product feiluoao
4. 测试：python rag_answer.py "非罗奥 术后护理" brief
5. 启动接口：uvicorn api_server:app --host 0.0.0.0 --port 8000

## 启动命令
```powershell
cd C:\Users\ericj\bge-m3-test
.\.venv\Scripts\Activate.ps1
python build_faiss.py --product feiluoao
python rag_answer.py "非罗奥 怎么验真伪" full
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 测试命令
```powershell
python rag_answer.py "非罗奥 怎么验真伪" full
python rag_answer.py "禁忌人群" full
python rag_answer.py "注射深度 0.8mm" full
python rag_answer.py "非罗奥 术后护理" brief
python run_regression.py
```
