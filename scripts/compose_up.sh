#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
  echo "GPU detected; using NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
else
  export NVIDIA_VISIBLE_DEVICES=""
  export NVIDIA_DRIVER_CAPABILITIES=""
  echo "No GPU detected; using CPU fallback"
fi

docker compose -f "${repo_root}/docker-compose.yml" up -d --build
