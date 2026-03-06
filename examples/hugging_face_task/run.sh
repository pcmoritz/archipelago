#!/bin/bash
#
# Run a task using a pre-built Docker image from /home/ubuntu/docker/.
#
# Usage:
#   cd archipelago/examples/hugging_face_task
#   ./run.sh                              # Run default task
#   ./run.sh world221-tr-01-9ba58a61      # Run by task slug
#   ./run.sh /home/ubuntu/docker/...      # Run by world directory (first task)
#
# Prerequisites:
#   - Docker running
#   - LLM API key set in agents/.env
#   - .env file in the world directory with required secrets
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIPELAGO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export EXAMPLE_DIR="$SCRIPT_DIR"
export AGENTS_DIR="$ARCHIPELAGO_DIR/agents"
export GRADING_DIR="$ARCHIPELAGO_DIR/grading"

echo "============================================================"
echo "DOCKER IMAGE TASK"
echo "============================================================"
echo "Example dir:     $EXAMPLE_DIR"
echo "Archipelago dir: $ARCHIPELAGO_DIR"
echo "============================================================"

# Install agent dependencies
echo "Installing agent dependencies..."
cd "$AGENTS_DIR"
uv sync

# Install httpx for health checks
uv pip install -q httpx

# Install grading dependencies
echo "Installing grading dependencies..."
cd "$GRADING_DIR"
uv sync

# Run main.py with any arguments passed to this script
cd "$AGENTS_DIR" && uv run python "$EXAMPLE_DIR/main.py" "$@"
