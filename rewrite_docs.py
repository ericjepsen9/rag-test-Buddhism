import json
from pathlib import Path

p = Path("faiss_store") / "docs.jsonl"
p.parent.mkdir(parents=True, exist_ok=True)

docs = [
    {"id": "doc_0", "text": "佛教讲缘起与无常。", "meta": {"tag": "buddhism"}},
    {"id": "doc_1", "text": "向量数据库可以做相似度检索。", "meta": {"tag": "vector_db"}},
    {"id": "doc_2", "text": "RAG=检索增强生成：先检索相关资料，再由模型生成答案。", "meta": {"tag": "rag"}},
    {"id": "doc_3", "text": "FAISS 是一个用于高效相似度搜索与向量聚类的库。", "meta": {"tag": "faiss"}},
    {"id": "doc_4", "text": "医美护肤知识库需要：项目、成分、禁忌、不良反应与护理。", "meta": {"tag": "aesthetic"}},
]

with p.open("w", encoding="utf-8", newline="\n") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("Rewrote:", p.resolve())
