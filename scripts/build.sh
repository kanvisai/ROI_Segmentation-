#!/usr/bin/env bash
# Build kanvis-segmentation Docker image (GPU default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${TARGET:-gpu}"
DRY_RUN="${DRY_RUN:-0}"

case "$TARGET" in
  gpu) TAG="${IMAGE_TAG:-kanvis-segmentation:gpu}" ;;
  cpu) TAG="${IMAGE_TAG:-kanvis-segmentation:cpu}" ;;
  *)
    echo "FAIL unknown TARGET=$TARGET (gpu|cpu)" >&2
    exit 1
    ;;
esac

if [[ "$TARGET" == "gpu" ]]; then
  ALSO_LATEST="${ALSO_LATEST:-1}"
else
  ALSO_LATEST="${ALSO_LATEST:-0}"
fi

echo "build target=$TARGET tag=$TAG"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN docker build -f Dockerfile --target $TARGET -t $TAG ."
  echo "PASS build"
  exit 0
fi

bash "$ROOT/scripts/gen_proto.sh"
docker build \
  -f Dockerfile \
  --target "$TARGET" \
  -t "$TAG" \
  .
if [[ "$ALSO_LATEST" == "1" ]]; then
  docker tag "$TAG" kanvis-segmentation:latest
fi
echo "PASS build $TAG"
