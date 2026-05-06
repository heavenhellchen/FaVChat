#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

pip install -e ./src/qwen-vl-utils
pip install -e ./src/r1-v
pip install wandb==0.18.3 tensorboardx torchvision flash-attn --no-build-isolation
pip install vllm==0.7.2 nltk rouge_score deepspeed
