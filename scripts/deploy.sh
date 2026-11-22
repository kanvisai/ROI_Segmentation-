#!/usr/bin/env bash
# Pull/build and docker compose up -d for kanvis-segmentation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.segmentation.yml}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "FAIL missing $COMPOSE_FILE" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN compose -f $COMPOSE_FILE config"
  docker compose -f "$COMPOSE_FILE" config >/dev/null
  echo "PASS deploy (dry-run, no MinIO writes, stack not started)"
  exit 0
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  bash "$ROOT/scripts/build.sh"
fi

docker compose -f "$COMPOSE_FILE" up -d
echo "PASS deploy"
