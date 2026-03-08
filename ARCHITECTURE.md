\# 系统架构



docs/\*.txt

&nbsp; ↓ 切块

docs.jsonl

&nbsp; ↓ embedding (bge-m3)

向量

&nbsp; ↓ FAISS

index.faiss

&nbsp; ↓ 检索

TopK chunks

&nbsp; ↓ 策略

\- 电话 → 正则抽取

\- 普通问答 → LLM

&nbsp; ↓

answer.txt



