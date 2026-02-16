#!/usr/bin/env python3
"""
maasv Retrieval Quality Validation

Runs 10 test queries against the memory store and checks that expected
keywords appear in the top-5 results. This is the standard gate for
retrieval changes — Phase 1 baseline was 9/10.

Usage (from the doris project directory, with its venv active):
    cd /Users/macmini/Projects/doris
    python /Users/macmini/Projects/maasv/scripts/validate_retrieval.py

Must be run with Doris's Python environment (maasv + doris on sys.path).
"""

import sys
import os

# Ensure both maasv and doris are importable
sys.path.insert(0, "/Users/macmini/Projects/maasv")
sys.path.insert(0, "/Users/macmini/Projects/doris")

# Set working directory to doris (config.py expects it)
os.chdir("/Users/macmini/Projects/doris")


SEARCH_QUALITY_TESTS = [
    {"query": "Adam's wife", "expected_contains": ["Gabby"]},
    {"query": "Adam's children", "expected_contains": ["Levi", "Dani"]},
    {"query": "Doris architecture", "expected_contains": ["FastAPI"]},
    {"query": "Hudson Valley", "expected_contains": ["second home"]},
    {"query": "TerryAnn", "expected_contains": ["Medicare"]},
    {"query": "birthday January", "expected_contains": ["Levi"]},
    {"query": "birthday November", "expected_contains": ["Dani"]},
    {"query": "mac mini", "expected_contains": ["M4"]},
    {"query": "dog", "expected_contains": ["Billi"]},
    {"query": "Upper West Side", "expected_contains": ["Manhattan"]},
]


def run_validation(verbose: bool = False):
    """Run all 10 test queries and report pass/fail."""
    # Initialize maasv through Doris's bridge
    from maasv_bridge import init_maasv
    init_maasv()

    from maasv.core.store import find_similar_memories

    passed = 0
    failed = 0
    results = []

    for i, test in enumerate(SEARCH_QUALITY_TESTS):
        query = test["query"]
        expected = test["expected_contains"]

        memories = find_similar_memories(query, limit=5)
        all_content = " ".join(m.get("content", "") for m in memories).lower()

        missing = [kw for kw in expected if kw.lower() not in all_content]

        if missing:
            status = "FAIL"
            failed += 1
        else:
            status = "PASS"
            passed += 1

        results.append({
            "test": i + 1,
            "query": query,
            "status": status,
            "missing": missing,
            "result_count": len(memories),
        })

        marker = "\u2705" if status == "PASS" else "\u274c"
        print(f"  {marker} Test {i+1:2d}: \"{query}\" -> {status}", end="")
        if missing:
            print(f"  (missing: {', '.join(missing)})")
        else:
            print()

        if verbose:
            for j, mem in enumerate(memories):
                content = mem.get("content", "")[:100]
                acc = mem.get("access_count", 0)
                cat = mem.get("category", "?")
                print(f"         [{j+1}] ({cat}, acc={acc}) {content}")

    print()
    print(f"  Result: {passed}/{passed + failed}")
    print()

    return passed, failed, results


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print()
    print("=" * 60)
    print("  maasv Retrieval Quality Validation")
    print("=" * 60)
    print()

    passed, failed, _ = run_validation(verbose=verbose)

    if failed == 0:
        print("  All tests passed!")
    elif failed <= 1:
        print(f"  {failed} test(s) failed (acceptable if test 3 = Doris architecture)")
    else:
        print(f"  {failed} test(s) failed — regression detected")

    sys.exit(0 if failed <= 1 else 1)
