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
    --exclude=data/parakeet/source/
    --exclude=data/parakeet/ckpt/
    --exclude=data/parakeet/engine/
    --exclude=data/moonshine/source/
    --exclude=data/moonshine/ckpt/
    --exclude=data/moonshine/engine/
)
echo "pushing to murdock"
rsync -az --delete "${EXCLUDES[@]}" ./ "murdock:/home/desmond/tensorrt/"
echo "done."
