#!/usr/bin/env bash
# Offline runner: local folder layout, no MinIO writes.
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
SESSION_ROOT="${SESSION_ROOT:-$ROOT/fixtures/session_001}"
CHUNK_STEM="${CHUNK_STEM:-chunk_001}"
OUTPUT="${OUTPUT:-/tmp/kanvis-segmentation-offline/tracking.json}"
IMAGE="${IMAGE:-kanvis-segmentation:gpu}"
DRY_RUN="${DRY_RUN:-1}"
USE_DOCKER="${USE_DOCKER:-0}"

if [[ ! -d "$SESSION_ROOT" ]]; then
  echo "FAIL missing session root $SESSION_ROOT" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" && "$USE_DOCKER" != "1" ]]; then
  echo "DRY_RUN offline: session=$SESSION_ROOT chunk=$CHUNK_STEM output=$OUTPUT"
  echo "PASS run_local_offline"
  exit 0
fi

mkdir -p "$(dirname "$OUTPUT")"

if [[ "$USE_DOCKER" == "1" ]]; then
  docker run --rm --gpus all \
    -v "$SESSION_ROOT:/fixtures/session_001:ro" \
    -e DRY_RUN="$DRY_RUN" \
    "$IMAGE" \
    python3 tools/run_offline_segmentation.py \
      --local-session-root /fixtures/session_001 \
      --chunk-stem "$CHUNK_STEM" \
      --output /tmp/tracking.json
  echo "PASS run_local_offline (docker)"
  exit 0
fi

"$PYTHON" "$ROOT/tools/run_offline_segmentation.py" \
  --local-session-root "$SESSION_ROOT" \
  --chunk-stem "$CHUNK_STEM" \
  --output "$OUTPUT"

if command -v jq >/dev/null 2>&1; then
  jq -e '.final_state' "$OUTPUT" >/dev/null
fi
echo "PASS run_local_offline"
