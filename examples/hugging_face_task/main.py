#!/usr/bin/env python3
"""
Run a task using a pre-built Docker image from /home/ubuntu/docker/.

Usage:
    ./main.py                                           # Use default task slug
    ./main.py world221-tr-01-9ba58a61                   # Run by task slug
    ./main.py /home/ubuntu/docker/investment-banking-world-221--(world_f83f49b3776b4b5e870c36091f7e2b0b)  # Run by world dir (first task)
"""

import atexit
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import uuid
import zipfile
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(os.environ.get("EXAMPLE_DIR", Path(__file__).parent))
ARCHIPELAGO_DIR = Path(os.environ.get("ARCHIPELAGO_DIR", EXAMPLE_DIR.parent.parent))
AGENTS_DIR = Path(os.environ.get("AGENTS_DIR", ARCHIPELAGO_DIR / "agents"))
GRADING_DIR = Path(os.environ.get("GRADING_DIR", ARCHIPELAGO_DIR / "grading"))

DOCKER_IMAGES_DIR = Path(os.environ.get("DOCKER_IMAGES_DIR", "/home/ubuntu/docker"))
LOCAL_DATA_DIR = Path(os.environ.get("LOCAL_DATA_DIR", "/mnt/nvme/data"))

DEFAULT_TASK_SLUG = "world221-tr-01-9ba58a61"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_world_and_slug(selector: str) -> tuple[Path, str]:
    """Resolve selector to (world_dir, task_slug).

    Selector can be:
      - A task slug like "world221-tr-01-9ba58a61"
      - A path to a world directory
    """
    # If it's a directory path, use first task
    if os.path.isdir(selector):
        world_dir = Path(selector)
        slugs = [f.stem for f in sorted((world_dir / "tasks").glob("*.json"))]
        if not slugs:
            log(f"ERROR: No tasks found in {world_dir}/tasks/")
            sys.exit(1)
        return world_dir, slugs[0]

    # Otherwise treat as task slug — search all world dirs
    task_slug = selector
    for entry in sorted(DOCKER_IMAGES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        config_dir = entry / "runner_configs" / task_slug
        if config_dir.is_dir():
            return entry, task_slug

    log(f"ERROR: Task slug '{task_slug}' not found in any world under {DOCKER_IMAGES_DIR}")
    sys.exit(1)


def load_docker_image(world_dir: Path) -> str:
    """Load image.tar and return the image reference.

    If SKIP_DOCKER_LOAD=1 is set (e.g. by batch_run.py after pre-loading),
    just read the tag from image.tar metadata without loading again.
    """
    image_tar = world_dir / "image.tar"

    if os.environ.get("SKIP_DOCKER_LOAD") == "1":
        # Image already loaded; extract the tag from the tar manifest
        with tarfile.open(str(image_tar), "r") as tf:
            manifest = json.loads(tf.extractfile("manifest.json").read())
        tags = manifest[0].get("RepoTags", [])
        if tags:
            log(f"Using pre-loaded image: {tags[0]}")
            return tags[0]
        # Fallback to image ID
        config = manifest[0].get("Config", "")
        image_id = config.replace(".json", "")
        log(f"Using pre-loaded image ID: sha256:{image_id}")
        return f"sha256:{image_id}"

    log(f"Loading docker image from {image_tar}...")
    result = subprocess.run(
        ["docker", "load", "-i", str(image_tar)], capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"ERROR: docker load failed: {result.stderr}")
        sys.exit(1)

    for line in result.stdout.strip().splitlines():
        m = re.match(r"Loaded image(?:\s+ID)?: (.+)", line)
        if m:
            return m.group(1)

    log(f"ERROR: Could not parse image ref from: {result.stdout}")
    sys.exit(1)


def get_container_ip(container_name: str) -> str:
    """Get the container's IP address on the Docker bridge network."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container_name],
        capture_output=True, text=True,
    )
    ip = result.stdout.strip()
    if result.returncode != 0 or not ip:
        log("ERROR: Could not get container IP")
        sys.exit(1)
    return ip


def start_container(image_ref: str, task_slug: str, container_name: str, env_file: Path | None) -> str:
    """Start the pre-built docker container. Returns the environment base URL."""
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", container_name,
    ]
    if env_file and env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
    cmd.extend([
        image_ref,
        "bash", "-c",
        "sed -i \""
        "s/TERRAPIN_OFFLINE='0'/TERRAPIN_OFFLINE='1'/;"
        "s/FMP_OFFLINE_MODE='false'/FMP_OFFLINE_MODE='true'/;"
        "s/EDGAR_OFFLINE_MODE='false'/EDGAR_OFFLINE_MODE='true'/;"
        "\" /app/tools/start.sh && exec /app/tools/start.sh \"$@\"",
        "--", task_slug,
    ])

    log(f"Starting container for task {task_slug}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ERROR: Failed to start container: {result.stderr}")
        sys.exit(1)

    env_url = f"http://{get_container_ip(container_name)}:8000"

    log("Waiting for health...")
    start = time.time()
    while time.time() - start < 300:
        # Fail fast if the container has exited (--rm removes it)
        probe = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            log("ERROR: Container exited (already removed by --rm)")
            log("ERROR: Environment failed to start")
            sys.exit(1)
        try:
            if httpx.get(f"{env_url}/health", timeout=5).status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(1)
    else:
        subprocess.run(["docker", "logs", container_name])
        log("ERROR: Environment failed to start")
        sys.exit(1)

    log("Waiting for MCP startup...")
    start = time.time()
    while time.time() - start < 300:
        logs = subprocess.run(
            ["docker", "logs", container_name], capture_output=True, text=True
        )
        if "Startup complete!" in logs.stdout or "Startup complete!" in logs.stderr:
            break
        time.sleep(1)

    log("Environment ready")
    return env_url


def snapshot(url: str, out_path: Path) -> Path:
    """Capture a snapshot and return the zip path."""
    with httpx.stream("POST", f"{url}/data/snapshot") as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return tar_gz_to_zip(out_path)


def tar_gz_to_zip(tar_gz_path: Path) -> Path:
    """Convert tar.gz to zip for grading."""
    stem = tar_gz_path.stem
    if stem.endswith(".tar"):
        stem = stem[:-4]
    zip_path = tar_gz_path.parent / f"{stem}.zip"
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        zf.writestr(member.name, f.read())
    return zip_path


def main():
    selector = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK_SLUG
    world_dir, task_slug = find_world_and_slug(selector)

    trajectory_id = f"hf_{task_slug}_{uuid.uuid4().hex[:8]}"
    grading_run_id = f"gr_{uuid.uuid4().hex[:8]}"
    output_dir = EXAMPLE_DIR / "output" / task_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log(f"Task slug: {task_slug}")
    log(f"World dir: {world_dir}")
    log("=" * 60)

    image_ref = load_docker_image(world_dir)
    container_name = f"hf_task_{task_slug}_{os.getpid()}"
    env_file = world_dir / ".env"

    atexit.register(lambda: (
        log(f"Stopping container {container_name}..."),
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True),
    ))

    env_url = start_container(image_ref, task_slug, container_name, env_file)

    # Capture initial snapshot
    log("Capturing initial snapshot...")
    initial_zip = snapshot(env_url, output_dir / "initial_snapshot.tar.gz")

    # Load task data from local files
    log("Loading task data from local files...")
    slug_suffix = task_slug.split("-")[-1]
    task_dir = None
    for entry in sorted(LOCAL_DATA_DIR.joinpath("tasks").iterdir()):
        if not entry.is_dir():
            continue
        # Directory names contain task_id in parens, e.g. "name--(task_bb48b8b3...)"
        m = re.search(r"\(task_([0-9a-f]+)\)", entry.name)
        if m and m.group(1).startswith(slug_suffix):
            task_dir = entry
            break
    if not task_dir:
        log(f"ERROR: Could not find local task matching slug suffix '{slug_suffix}'")
        sys.exit(1)
    with open(task_dir / "task.json") as f:
        task = json.load(f)
    log(f"Task: {task['task_name']}")

    # Generate initial messages from local task prompt
    # System prompt from agents/runner/agents/react_toolbelt_agent/README.md
    system_prompt = """You are an AI assistant that completes tasks by reasoning and using tools.

## Be Efficient

You have a LIMITED token budget. Conserve tokens at every step:
- Keep reasoning to 1-2 sentences max. No preamble, no recaps.
- Request only the data you need. Avoid dumping entire files or databases.
- Combine related tool calls when possible.
- Remove tools from your toolbelt when no longer needed (`toolbelt_remove_tool`).
- Do NOT repeat information from previous steps. Refer to it briefly.
- Skip todos for simple tasks — go straight to tool calls and `final_answer`.

## Tools

**Always Available (Meta-Tools):**
- `todo_write` - Task planning: create/update todos. Takes `todos` array [{id, content, status}] and `merge` boolean.
- `toolbelt_list_tools` / `toolbelt_inspect_tool` / `toolbelt_add_tool` / `toolbelt_remove_tool` - Tool management
- `final_answer` - Submit your answer (status: completed/blocked/failed)

**Domain Tools:** Use `toolbelt_list_tools` to discover, then `toolbelt_add_tool` to add them.

## Workflow

1. Discover: Use `toolbelt_list_tools` to find relevant tools
2. Execute: Add only the tools you need, get the data, solve the task
3. Complete: Call `final_answer` as soon as you have the answer

## Rules

- For complex multi-step tasks, use `todo_write` to plan. For simple tasks, skip it.
- Update todo status with `todo_write`: set `in_progress` when starting, `completed` when done
- Show your work for calculations (briefly)
- `final_answer` is rejected if todos are incomplete
"""
    initial_messages = [
        {"role": "system", "content": system_prompt},
    ] + [
        {"role": m["role"], "content": m["content"]}
        for m in task["task_prompt_messages"]
    ]
    with open(output_dir / "initial_messages.json", "w") as f:
        json.dump(initial_messages, f, indent=2)

    # Load orchestrator config
    with open(EXAMPLE_DIR / "orchestrator_config.json") as f:
        orchestrator_config = json.load(f)

    orchestrator_model = os.environ.get("ORCHESTRATOR_MODEL") or orchestrator_config["model"]

    trajectory_file = output_dir / "trajectory.json"

    # Run agent
    log("Running agent...")
    agent_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "runner.main",
        "--trajectory-id",
        trajectory_id,
        "--initial-messages",
        str(output_dir / "initial_messages.json"),
        "--mcp-gateway-url",
        f"{env_url}/mcp/",
        "--agent-config",
        str(EXAMPLE_DIR / "agent_config.json"),
        "--orchestrator-model",
        orchestrator_model,
        "--output",
        str(trajectory_file),
    ]

    # Add extra args if present
    if orchestrator_config.get("extra_args"):
        extra_args_file = output_dir / "orchestrator_extra_args.json"
        with open(extra_args_file, "w") as f:
            json.dump(orchestrator_config["extra_args"], f)
        agent_cmd.extend(["--orchestrator-extra-args", str(extra_args_file)])

    result = subprocess.run(agent_cmd, cwd=AGENTS_DIR)
    if result.returncode != 0:
        log(f"WARNING: Agent exited with code {result.returncode}")

    agent_status = None
    if trajectory_file.exists():
        with open(trajectory_file) as f:
            trajectory = json.load(f)
            agent_status = trajectory.get("status")
            log(f"Agent status: {agent_status}")

    # Save final snapshot
    log("Saving final snapshot...")
    final_zip = snapshot(env_url, output_dir / "final_snapshot.tar.gz")
    log(f"Saved: {final_zip}")

    # Run grading if agent completed
    if agent_status != "completed":
        log(f"Skipping grading (agent status: {agent_status})")
    else:
        log("Running grading...")

        # Generate verifiers from local task data
        verifiers = [
            {
                "verifier_id": v["verifier_id"],
                "verifier_version": v.get("verifier_version", 1),
                "world_id": task["world_id"],
                "task_id": task["task_id"],
                "eval_config_id": "ec_output_llm",
                "verifier_values": {
                    "criteria": v["config_input"]["criteria"],
                    "is_primary_objective": i == 0,
                },
                "verifier_index": v.get("verifier_index", i),
                "verifier_dependencies": v.get("verifier_dependencies"),
            }
            for i, v in enumerate(task.get("task_verifiers", []))
        ]
        with open(output_dir / "verifiers.json", "w") as f:
            json.dump(verifiers, f, indent=2)

        grades_file = output_dir / "grades.json"

        grading_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "runner.main",
            "--grading-run-id",
            grading_run_id,
            "--trajectory-id",
            trajectory_id,
            "--initial-snapshot",
            str(initial_zip),
            "--final-snapshot",
            str(final_zip),
            "--trajectory",
            str(trajectory_file),
            "--grading-settings",
            str(EXAMPLE_DIR / "grading_settings.json"),
            "--verifiers",
            str(output_dir / "verifiers.json"),
            "--eval-configs",
            str(EXAMPLE_DIR / "eval_configs.json"),
            "--scoring-config",
            str(EXAMPLE_DIR / "scoring_config.json"),
            "--output",
            str(grades_file),
        ]

        result = subprocess.run(grading_cmd, cwd=GRADING_DIR)
        if result.returncode != 0:
            log(f"WARNING: Grading exited with code {result.returncode}")

        if grades_file.exists():
            with open(grades_file) as f:
                grades = json.load(f)
            log("=" * 60)
            log("GRADING RESULTS")
            log("=" * 60)
            log(f"Status: {grades.get('grading_run_status')}")
            log(f"Final Score: {grades.get('scoring_results', {}).get('final_score')}")
            for vr in grades.get("verifier_results", []):
                log(f"  - {vr.get('verifier_id')}: {vr.get('score')}")

    log("=" * 60)
    log("DONE")
    log(f"Output: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
