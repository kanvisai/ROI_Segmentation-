#!/usr/bin/env bash
# HealthCheck (+ optional ApplySegmentation) against TARGET.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${TARGET:-localhost:50100}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
DRY_RUN="${DRY_RUN:-1}"
SESSION_ID="${SESSION_ID:-smoke-session}"
REQUEST_ID="${REQUEST_ID:-smoke-$(date +%s)}"

echo "smoke_grpc target=$TARGET dry_run=$DRY_RUN"
if [[ "$DRY_RUN" == "1" && -z "${FORCE_RPC:-}" && -z "${SOURCE_REF:-}" && -z "${CHUNK_STEM:-}" ]]; then
  echo "DRY_RUN smoke_grpc: skip RPC (set FORCE_RPC=1 or SOURCE_REF to call $TARGET)"
  echo "PASS smoke_grpc"
  exit 0
fi
if [[ -n "${SOURCE_REF:-}" || -n "${CHUNK_STEM:-}" ]]; then
  "$PYTHON" "$ROOT/tools/grpc_smoke_client.py" \
    --target "$TARGET" \
    --request-id "$REQUEST_ID" \
    --session-id "$SESSION_ID" \
    --source-ref "${SOURCE_REF:-}" \
    --pose-chunk-ref "${POSE_REF:-${POSE_CHUNK_REF:-}}" \
    --image-chunk-ref "${IMAGE_CHUNK_REF:-}" \
    --span-spec "${SPAN_SPEC:-single}" \
    --dry-run "$DRY_RUN" \
    --persist "${PERSIST:-0}" \
    --bucket "${BUCKET:-}" \
    --retail-id "${RETAIL_ID:-}" \
    --store-id "${STORE_ID:-}" \
    --camera-id "${CAMERA_ID:-}" \
    --chunk-stem "${CHUNK_STEM:-}"
else
  "$PYTHON" "$ROOT/tools/grpc_smoke_client.py" \
    --target "$TARGET" \
    --request-id "$REQUEST_ID" \
    --session-id "$SESSION_ID" \
    --health-only
fi
echo "PASS smoke_grpc"
