"""综合测试：覆盖所有佛学领域的路由检测 + 答案内容验证

用法：
  python run_comprehensive_test.py              # 仅路由检测
  python run_comprehensive_test.py --full       # 路由 + 答案内容
  python run_comprehensive_test.py --category 教义  # 指定分类
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
_RAW = json.loads((BASE_DIR / "test_comprehensive.json").read_text(encoding="utf-8"))
CASES = [c for c in _RAW if "q" in c]


def test_route_detection(cases, verbose=True):
    from rag_answer import detect_route
    stats = defaultdict(lambda: {"ok": 0, "total": 0, "fails": []})
    ok = total = 0
    for case in cases:
        expected = case.get("route")
        if not expected:
            continue
        total += 1
        cat = case.get("category", "未分类")
        actual = detect_route(case["q"])
        passed = actual == expected
        stats[cat]["total"] += 1
        if passed:
            ok += 1
            stats[cat]["ok"] += 1
        else:
            stats[cat]["fails"].append(
                f"    {case['q'][:40]}  expected={expected}  got={actual}")
            if verbose:
                print(f"  [FAIL] {case['q'][:40]}  expected={expected}  got={actual}")

    print("\n  --- 路由检测分类统计 ---")
    for cat, s in sorted(stats.items()):
        rate = s["ok"] / s["total"] * 100 if s["total"] else 0
        m = "OK" if s["ok"] == s["total"] else "FAIL"
        print(f"  {m} {cat}: {s['ok']}/{s['total']} ({rate:.0f}%)")
        for f in s["fails"]:
            print(f)
    print(f"\n  路由总计: {ok}/{total} ({ok/total*100:.1f}%)")
    return ok, total


def test_answer_content(cases, verbose=True):
    from rag_answer import answer_question
    stats = defaultdict(lambda: {"ok": 0, "total": 0, "fails": []})
    ok = total = 0
    for case in cases:
        total += 1
        cat = case.get("category", "未分类")
        q = case["q"]
        ans = answer_question(q, "brief")
        expected = case.get("expect_any", [])
        passed = any(x in ans for x in expected)
        stats[cat]["total"] += 1
        if passed:
            ok += 1
            stats[cat]["ok"] += 1
        else:
            stats[cat]["fails"].append(f"    {q}")
            if verbose:
                print(f"  [FAIL] {q}")
                print(f"    Expected any of: {expected}")
                print(f"    Got: {ans[:200]}")

    print("\n  --- 答案内容分类统计 ---")
    for cat, s in sorted(stats.items()):
        rate = s["ok"] / s["total"] * 100 if s["total"] else 0
        m = "OK" if s["ok"] == s["total"] else "FAIL"
        print(f"  {m} {cat}: {s['ok']}/{s['total']} ({rate:.0f}%)")
        for f in s["fails"]:
            print(f)
    print(f"\n  答案总计: {ok}/{total} ({ok/total*100:.1f}%)")
    return ok, total


def main():
    full = "--full" in sys.argv
    cat_filter = None
    for i, a in enumerate(sys.argv):
        if a == "--category" and i + 1 < len(sys.argv):
            cat_filter = sys.argv[i + 1]

    cases = CASES
    if cat_filter:
        cases = [c for c in CASES if cat_filter in c.get("category", "")]
        print(f"过滤分类: '{cat_filter}', 匹配 {len(cases)} 条\n")

    print("=" * 60)
    print("佛学综合测试 (Comprehensive Buddhism Knowledge Test)")
    print(f"测试用例: {len(cases)} 条")
    print("=" * 60)

    total_ok, total_all = 0, 0

    print("\n[1] 路由检测测试")
    print("-" * 40)
    ok, t = test_route_detection(cases)
    total_ok += ok
    total_all += t

    if full:
        print("\n[2] 答案内容测试")
        print("-" * 40)
        ok, t = test_answer_content(cases)
        total_ok += ok
        total_all += t
    else:
        print("\n[2] 答案内容测试 (跳过, 使用 --full 启用)")

    print("\n" + "=" * 60)
    rate = total_ok / total_all * 100 if total_all else 0
    print(f"总计: {total_ok}/{total_all} 通过 ({rate:.1f}%)")
    print("=" * 60)
    sys.exit(0 if rate >= 80 else 1)


if __name__ == "__main__":
    main()
