#!/usr/bin/env python3
"""
Batch run all tasks with parallel workers.

Usage:
    uv run python batch_run.py                    # 32 workers
    uv run python batch_run.py --workers 8
    uv run python batch_run.py --resume           # skip done tasks
    ORCHESTRATOR_MODEL=gemini/gemini-3-pro-preview uv run python batch_run.py
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOCKER_DIR = Path(os.environ.get("DOCKER_IMAGES_DIR", "/mnt/nvme/images/docker/"))
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_tasks():
    tasks = []
    for world_dir in sorted(DOCKER_DIR.iterdir()):
        tasks_dir = world_dir / "tasks"
        if not tasks_dir.is_dir():
            continue
        for f in sorted(tasks_dir.glob("*.json")):
            if (world_dir / "runner_configs" / f.stem).is_dir():
                tasks.append(f.stem)
    return tasks


def preload_images():
    log("Pre-loading Docker images...")
    for world_dir in sorted(DOCKER_DIR.iterdir()):
        tar = world_dir / "image.tar"
        if not tar.exists():
            continue
        log(f"  {world_dir.name}")
        subprocess.run(["docker", "load", "-i", str(tar)], capture_output=True)
    log("Done loading images.")


def run_task(slug, max_retries=3):
    out = OUTPUT_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SKIP_DOCKER_LOAD"] = "1"
    env["DOCKER_IMAGES_DIR"] = str(DOCKER_DIR)
    start = time.time()
    for attempt in range(1, max_retries + 1):
        with open(out / "run.log", "w") as lf:
            rc = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "main.py"), slug],
                env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=1800,
            ).returncode
        if rc == 0 or attempt == max_retries:
            break
        log(f"Retrying {slug} (attempt {attempt + 1}/{max_retries})")
        time.sleep(2 * attempt)
    elapsed = round(time.time() - start, 1)
    score = None
    grades = out / "grades.json"
    if grades.exists():
        score = json.loads(grades.read_text()).get("scoring_results", {}).get("final_score")
    return slug, rc, score, elapsed


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-preload", action="store_true")
    args = p.parse_args()

    tasks = discover_tasks()
    if args.resume:
        before = len(tasks)
        tasks = [s for s in tasks if not (OUTPUT_DIR / s / "trajectory.json").exists()]
        log(f"Resume: {before - len(tasks)} done, {len(tasks)} remaining")

    log(f"{len(tasks)} tasks, {args.workers} workers")
    if not tasks:
        return

    if not args.skip_preload:
        preload_images()
    else:
        log("Skipping image preload (already loaded)")

    done, failed, scores = 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_task, slug): slug
            for slug in tasks
        }
        for f in as_completed(futures):
            slug, rc, score, elapsed = f.result()
            done += 1
            if rc != 0:
                failed += 1
            if score is not None:
                scores.append(score)
            tag = "OK" if rc == 0 else "FAIL"
            log(f"[{done}/{len(tasks)}] {tag} {slug} ({elapsed}s, score={score})")

    avg = sum(scores) / len(scores) if scores else 0
    log("=" * 60)
    log(f"DONE: {len(tasks)} tasks, {failed} failed, avg score={avg:.4f}")
    log(f"Output: {OUTPUT_DIR}")
    log("=" * 60)

    (OUTPUT_DIR / "batch_results.json").write_text(json.dumps({
        "total": len(tasks), "failed": failed,
        "scored": len(scores), "avg_score": round(avg, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
