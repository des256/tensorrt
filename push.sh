#!/bin/bash
set -euo pipefail
EXCLUDES=(
    --exclude=target/
    --exclude=.git/
    --exclude=.claude/
    --exclude=.worktrees/
    --exclude=data/llama3-3b/source/
    --exclude=data/llama3-3b/ckpt/
    --exclude=data/llama3-3b/engine/
)
echo "pushing to murdock"
rsync -az --delete "${EXCLUDES[@]}" ./ "murdock:/home/desmond/tensorrt/"
