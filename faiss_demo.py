import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

docs = [
    "佛教讲缘起与无常。",
    "向量数据库可以做相似度检索。",
    "bge-m3 是一个多语言 embedding 模型。",
    "RTX 4050 可以加速向量生成。"
]

emb = model.encode(docs, batch_size=1, max_length=256)["dense_vecs"]
emb = np.asarray(emb, dtype="float32")

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(emb)
index.add(emb)

query = "我想用向量检索做搜索"
qv = model.encode([query], batch_size=1, max_length=256)["dense_vecs"]
qv = np.asarray(qv, dtype="float32")
faiss.normalize_L2(qv)

k = 3
scores, ids = index.search(qv, k)

print("Query:", query)
for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
    print(f"{rank}. score={s:.4f} doc={docs[i]}")
