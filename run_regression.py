"""
回归测试：验证 RAG 系统对标准问题的回答质量。
使用 in-process 调用（与 api_server 一致），避免 subprocess 开销。
"""
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CASES = json.loads((BASE_DIR / "regression_cases.json").read_text(encoding="utf-8"))

# in-process 导入
from rag_answer import answer_question

ok = 0
total = len(CASES)
total_time = 0.0

for case in CASES:
    q = case["q"]
    note = case.get("note", "")
    start = time.time()
    try:
        ans = answer_question(q, "brief")
    except Exception as e:
        elapsed = time.time() - start
        total_time += elapsed
        print(f"[ERROR] {q} ({elapsed:.1f}s) {note} — {e}")
        continue
    elapsed = time.time() - start
    total_time += elapsed
    passed = any(x in ans for x in case.get("expect_any", []))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {q} ({elapsed:.1f}s, {len(ans)} chars) {note}")
    if not passed:
        expected = case.get("expect_any", [])
        print(f"  Expected one of: {expected}")
        print(f"  Got: {ans[:500]}")
    ok += 1 if passed else 0

print(f"\nPassed {ok}/{total} ({ok/total*100:.0f}%) in {total_time:.1f}s total")
