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

STUB_PY="kanvis/segmentation/v1/segmentation_pb2.py"
STUB_GRPC="kanvis/segmentation/v1/segmentation_pb2_grpc.py"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN gen_proto: would compile proto/kanvis/segmentation/v1/segmentation.proto"
  echo "PASS gen_proto"
  exit 0
fi

if ! "$PYTHON" -c "import grpc_tools.protoc" >/dev/null 2>&1; then
  if [[ -f "$STUB_PY" && -f "$STUB_GRPC" ]]; then
    echo "SKIP gen_proto: grpc_tools missing; using committed stubs"
    echo "PASS gen_proto"
    exit 0
  fi
  echo "FAIL No module named grpc_tools. Install with:" >&2
  echo "  python3 -m pip install -r requirements-service.txt" >&2
  echo "  # Debian/Ubuntu: python3 -m pip install --break-system-packages grpcio-tools" >&2
  exit 1
fi

mkdir -p kanvis/segmentation/v1
touch kanvis/__init__.py kanvis/segmentation/__init__.py kanvis/segmentation/v1/__init__.py

"$PYTHON" -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/kanvis/segmentation/v1/segmentation.proto

test -f "$STUB_PY"
test -f "$STUB_GRPC"
echo "PASS gen_proto"
