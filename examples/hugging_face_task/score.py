#!/usr/bin/env python3
"""
Compute pass@1 and other statistics from graded results in output/.

Usage:
    python score.py                  # score all results in output/
    python score.py path/to/output   # score results in a custom directory
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_grades(output_dir: Path) -> list[dict]:
    results = []
    for grades_file in sorted(output_dir.glob("*/grades.json")):
        slug = grades_file.parent.name
        try:
            data = json.loads(grades_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: skipping {slug}: {e}", file=sys.stderr)
            continue

        scoring = data.get("scoring_results", {})
        final_score = scoring.get("final_score")
        if final_score is None:
            print(f"WARNING: skipping {slug}: no final_score", file=sys.stderr)
            continue

        method = scoring.get("scoring_method_result_values", {})
        verifiers = data.get("verifier_results", [])

        results.append({
            "slug": slug,
            "final_score": final_score,
            "task_score": method.get("task_score", final_score),
            "universal_penalty": method.get("universal_penalty", 0.0),
            "capped_penalty": method.get("capped_penalty", 0.0),
            "task_verifier_count": method.get("task_verifier_count", len(verifiers)),
            "verifier_scores": [v.get("score", 0.0) for v in verifiers],
            "status": data.get("grading_run_status", "unknown"),
        })
    return results


def compute_stats(results: list[dict]):
    n = len(results)
    if n == 0:
        print("No graded results found.")
        return

    scores = [r["final_score"] for r in results]
    task_scores = [r["task_score"] for r in results]
    passed = [s for s in scores if s == 1.0]
    partial = [s for s in scores if 0.0 < s < 1.0]
    failed = [s for s in scores if s == 0.0]

    # Verifier-level stats
    all_verifier_scores = []
    for r in results:
        all_verifier_scores.extend(r["verifier_scores"])

    total_verifiers = len(all_verifier_scores)
    verifiers_passed = sum(1 for v in all_verifier_scores if v == 1.0)

    # Score distribution buckets
    buckets = defaultdict(int)
    for s in scores:
        if s == 0.0:
            buckets["0%"] += 1
        elif s < 0.25:
            buckets["1-24%"] += 1
        elif s < 0.50:
            buckets["25-49%"] += 1
        elif s < 0.75:
            buckets["50-74%"] += 1
        elif s < 1.0:
            buckets["75-99%"] += 1
        else:
            buckets["100%"] += 1

    sorted_scores = sorted(scores)
    median = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    print("=" * 60)
    print(f"  Results Summary ({n} tasks)")
    print("=" * 60)
    print()
    print(f"  pass@1:            {len(passed) / n:.4f}  ({len(passed)}/{n})")
    print(f"  Mean score:        {sum(scores) / n:.4f}")
    print(f"  Median score:      {median:.4f}")
    print(f"  Mean task score:   {sum(task_scores) / n:.4f}")
    print()
    print(f"  Fully passed:      {len(passed):>4}  ({len(passed)/n*100:.1f}%)")
    print(f"  Partially passed:  {len(partial):>4}  ({len(partial)/n*100:.1f}%)")
    print(f"  Fully failed:      {len(failed):>4}  ({len(failed)/n*100:.1f}%)")
    print()
    print("  Score distribution:")
    for bucket in ["0%", "1-24%", "25-49%", "50-74%", "75-99%", "100%"]:
        count = buckets[bucket]
        bar = "#" * (count * 40 // n) if n > 0 else ""
        print(f"    {bucket:>6}: {count:>4}  {bar}")
    print()
    print(f"  Verifiers: {verifiers_passed}/{total_verifiers} passed ({verifiers_passed/total_verifiers*100:.1f}%)" if total_verifiers else "  Verifiers: none")
    print()

    # Penalty stats
    penalties = [r["capped_penalty"] for r in results if r["capped_penalty"] > 0]
    if penalties:
        print(f"  Tasks with penalties: {len(penalties)}/{n}")
        print(f"  Mean penalty (when applied): {sum(penalties)/len(penalties):.4f}")
        print()

    # Bottom/top tasks
    by_score = sorted(results, key=lambda r: r["final_score"])
    print("  Bottom 10 tasks:")
    for r in by_score[:10]:
        print(f"    {r['final_score']:.4f}  {r['slug']}")
    print()
    print("  Top 10 tasks:")
    for r in by_score[-10:]:
        print(f"    {r['final_score']:.4f}  {r['slug']}")
    print()
    print("=" * 60)


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output"
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = load_grades(output_dir)
    compute_stats(results)


if __name__ == "__main__":
    main()
