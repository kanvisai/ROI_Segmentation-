#!/usr/bin/env python3
"""Run SAM3 segmentation on a local session folder (no MinIO)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-session-root", type=Path, required=True)
    parser.add_argument("--chunk-stem", default="chunk_001")
    parser.add_argument("--span-spec", default="single")
    parser.add_argument("--output", type=Path, default=Path("/tmp/tracking.json"))
    parser.add_argument("--review-output", type=Path, default=None)
    args = parser.parse_args()

    from kanvis_segmentation.config import load_settings
    from kanvis_segmentation.pipeline_runner import run_local_session

    settings = load_settings()
    result = run_local_session(
        args.local_session_root,
        args.chunk_stem,
        span_spec=args.span_spec,
        settings=settings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.tracking, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    review_path = args.review_output or args.output.with_name("roi_fp_review.json")
    review_path.write_text(
        json.dumps(result.review, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"status": result.status, "output": str(args.output), "review": str(review_path)}))
    if result.status == "error":
        print(f"FAIL {result.error_message}", file=sys.stderr)
        return 1
    if "final_state" not in result.tracking:
        print("FAIL tracking missing final_state", file=sys.stderr)
        return 1
    print("PASS offline segmentation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
