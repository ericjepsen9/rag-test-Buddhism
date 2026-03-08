import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CASES = json.loads((BASE_DIR / "regression_cases.json").read_text(encoding="utf-8"))

ok = 0
total = len(CASES)
for case in CASES:
    q = case["q"]
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "rag_answer.py"), q, "brief"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ans = (proc.stdout or "").strip()
    passed = any(x in ans for x in case.get("expect_any", []))
    print(f"[{'PASS' if passed else 'FAIL'}] {q}")
    if not passed:
        print(f"  stdout: {ans[:300]}")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:200]}")
    ok += 1 if passed else 0

print(f"\nPassed {ok}/{total}")
