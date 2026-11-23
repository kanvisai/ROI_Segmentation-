#!/usr/bin/env python3
"""Minimal ApplySegmentation client for kanvis-segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="gRPC smoke client for ApplySegmentation")
    parser.add_argument("--target", default="localhost:50100")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--pose-chunk-ref", default="")
    parser.add_argument("--image-chunk-ref", default="")
    parser.add_argument("--span-spec", default="single")
    parser.add_argument("--dry-run", type=int, default=1)
    parser.add_argument("--persist", type=int, default=0)
    parser.add_argument("--bucket", default="")
    parser.add_argument("--retail-id", default="")
    parser.add_argument("--store-id", default="")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--chunk-stem", default="")
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()

    import grpc
    from kanvis.segmentation.v1 import segmentation_pb2, segmentation_pb2_grpc

    channel = grpc.insecure_channel(args.target)
    stub = segmentation_pb2_grpc.SegmentationServiceStub(channel)
    health = stub.HealthCheck(segmentation_pb2.HealthCheckRequest(), timeout=10)
    print(json.dumps({"ready": health.ready, "message": health.message}))
    if args.health_only:
        if not health.ready:
            print("FAIL HealthCheck not ready", file=sys.stderr)
            return 1
        print("PASS health")
        return 0
    req = segmentation_pb2.SegmentationRequest(
        request_id=args.request_id,
        session_id=args.session_id,
        source_ref=args.source_ref,
        pose_chunk_ref=args.pose_chunk_ref,
        image_chunk_ref=args.image_chunk_ref,
        span_spec=args.span_spec,
        dry_run=bool(args.dry_run),
        persist_to_minio=bool(args.persist) and not bool(args.dry_run),
        bucket=args.bucket,
        retail_id=args.retail_id,
        store_id=args.store_id,
        camera_id=args.camera_id,
        chunk_stem=args.chunk_stem,
    )
    resp = stub.ApplySegmentation(req, timeout=600)
    review = json.loads(resp.roi_fp_review_json or "{}")
    tracking = json.loads(resp.tracking_json or "{}")
    print(
        json.dumps(
            {
                "status": resp.status,
                "error_message": resp.error_message,
                "tracking_ref": resp.tracking_ref,
                "final_state": tracking.get("final_state"),
                "hands_visible": tracking.get("hands_visible"),
                "span_spec": tracking.get("span_spec"),
                "p1_motion_stem": tracking.get("p1_motion_stem"),
                "overlap": tracking.get("overlap"),
                "frames_used": tracking.get("frames_used"),
                "roi_fp_review": review,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if resp.status not in ("ok", "skipped"):
        print(f"FAIL ApplySegmentation status={resp.status}", file=sys.stderr)
        return 1
    if review.get("schema_version") != "c2_roi_fp_review_v1":
        print("FAIL roi_fp_review schema", file=sys.stderr)
        return 1
    print("PASS ApplySegmentation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
