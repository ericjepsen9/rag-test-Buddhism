import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CASES = json.loads((BASE_DIR / "regression_cases.json").read_text(encoding="utf-8"))
TIMEOUT = 60  # 每个用例最多 60 秒

ok = 0
total = len(CASES)
total_time = 0.0

for case in CASES:
    q = case["q"]
    note = case.get("note", "")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(BASE_DIR / "rag_answer.py"), q, "brief"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        total_time += elapsed
        print(f"[TIMEOUT] {q} ({elapsed:.1f}s) {note}")
        continue
    elapsed = time.time() - start
    total_time += elapsed
    ans = (proc.stdout or "").strip()
    passed = any(x in ans for x in case.get("expect_any", []))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {q} ({elapsed:.1f}s, {len(ans)} chars) {note}")
    if not passed:
        expected = case.get("expect_any", [])
        print(f"  Expected one of: {expected}")
        print(f"  Got: {ans[:500]}")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:200]}")
    ok += 1 if passed else 0

print(f"\nPassed {ok}/{total} ({ok/total*100:.0f}%) in {total_time:.1f}s total")
