#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN gen_proto: would compile proto/kanvis/segmentation/v1/segmentation.proto"
  echo "PASS gen_proto"
  exit 0
fi

mkdir -p kanvis/segmentation/v1
touch kanvis/__init__.py kanvis/segmentation/__init__.py kanvis/segmentation/v1/__init__.py

"$PYTHON" -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/kanvis/segmentation/v1/segmentation.proto

test -f kanvis/segmentation/v1/segmentation_pb2.py
test -f kanvis/segmentation/v1/segmentation_pb2_grpc.py
echo "PASS gen_proto"
