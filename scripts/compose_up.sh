#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

use_gpu=false

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if [[ -c /dev/nvidia0 ]]; then
    use_gpu=true
  fi
fi

if [[ "${use_gpu}" == "true" ]]; then
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
  echo "GPU detected; enabling GPU devices"
  docker compose -f "${repo_root}/docker-compose.yml" -f "${repo_root}/docker-compose.gpu.yml" up -d --build
else
  export NVIDIA_VISIBLE_DEVICES=""
  export NVIDIA_DRIVER_CAPABILITIES=""
  echo "GPU not available or CDI missing; using CPU fallback"
  docker compose -f "${repo_root}/docker-compose.yml" up -d --build
fi
