#!/usr/bin/env bash
# End-to-end ApplySegmentation against a fixed MinIO session URI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${TARGET:-localhost:50100}"
SOURCE_REF="${SOURCE_REF:?s3 session prefix with trailing slash}"
POSE_REF="${POSE_REF:?e.g. .../chunk_001/poses_full.npy}"
SESSION_ID="${SESSION_ID:-smoke-session}"
REQUEST_ID="${REQUEST_ID:-smoke-$(date +%s)}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
DRY_RUN="${DRY_RUN:-1}"

echo "smoke_minio target=$TARGET dry_run=$DRY_RUN"
"$PYTHON" "$ROOT/tools/grpc_smoke_client.py" \
  --target "$TARGET" \
  --request-id "$REQUEST_ID" \
  --session-id "$SESSION_ID" \
  --source-ref "$SOURCE_REF" \
  --pose-chunk-ref "$POSE_REF" \
  --span-spec "${SPAN_SPEC:-single}" \
  --dry-run "$DRY_RUN"

echo "PASS smoke_minio"
