#!/bin/bash
set -euo pipefail
EXCLUDES=(
    --exclude=target/
    --exclude=.git/
    --exclude=.claude/
    --exclude=.worktrees/
    --exclude='mutants.out*'
    --exclude=data/*/source/
    --exclude=data/*/onnx/
    --exclude=data/*/.venv/
    --exclude=data/*/venv/
)
echo "pushing to murdock (dry-run)"
rsync -azn --delete "${EXCLUDES[@]}" ./ "murdock:/home/desmond/tensorrt/"
