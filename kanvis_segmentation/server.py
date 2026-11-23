"""gRPC SegmentationService on port 50100."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import traceback
from concurrent import futures
from pathlib import Path
from time import perf_counter

from .config import load_settings
from .contracts import summarize_tracking_for_c3, whitelist_roi_fp_review
from .pipeline_runner import SegmentationResult, load_local_bundle, run_sam3_on_bundle
from .refs import job_from_fields

try:
    from prometheus_client import Counter, Histogram, start_http_server
except ImportError:  # pragma: no cover
    Counter = Histogram = start_http_server = None  # type: ignore[misc, assignment]


_REQ = Counter("kanvis_segmentation_requests_total", "ApplySegmentation calls", ["status"]) if Counter else None
_LAT = Histogram("kanvis_segmentation_latency_seconds", "ApplySegmentation latency") if Histogram else None


class SegmentationLogic:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self._ready = False
        self._ready_msg = "loading models"
        self._lock = threading.Lock()
        self._detectores = None
        from kanvis.segmentation.v1 import segmentation_pb2

        self._pb2 = segmentation_pb2

    def load_models(self) -> None:
        from pipeline_manos import cargar_detectores
        from .pipeline_runner import _make_args

        args = _make_args(self.settings, Path(self.settings.work_dir) / "boot")
        self._detectores = cargar_detectores(args)
        self._ready = True
        self._ready_msg = "ready"

    def health(self):
        return self._pb2.HealthCheckResponse(ready=self._ready, message=self._ready_msg)

    def apply(self, request):
        t0 = perf_counter()
        try:
            result = self._apply(request)
            if _REQ is not None:
                _REQ.labels(status=result.status).inc()
            return self._to_proto(request.request_id, result)
        except Exception as exc:
            traceback.print_exc()
            if _REQ is not None:
                _REQ.labels(status="error").inc()
            return self._pb2.SegmentationResponse(
                request_id=request.request_id,
                status="error",
                error_message=str(exc) or exc.__class__.__name__,
                tracking_json="{}",
                roi_fp_review_json=json.dumps(summarize_tracking_for_c3({})),
            )
        finally:
            if _LAT is not None:
                _LAT.observe(perf_counter() - t0)

    def _to_proto(self, request_id: str, result: SegmentationResult):
        return self._pb2.SegmentationResponse(
            request_id=request_id,
            status=result.status,
            error_message=result.error_message or "",
            tracking_json=json.dumps(result.tracking, ensure_ascii=False, default=str),
            roi_fp_review_json=json.dumps(
                whitelist_roi_fp_review(result.review), ensure_ascii=False, default=str
            ),
            tracking_ref=result.tracking_ref or "",
        )

    def _apply(self, request) -> SegmentationResult:
        if not self._ready:
            raise RuntimeError(self._ready_msg)
        job = job_from_fields(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            bucket=request.bucket,
            retail_id=request.retail_id,
            store_id=request.store_id,
            camera_id=request.camera_id,
            chunk_stem=request.chunk_stem,
            source_ref=request.source_ref,
            pose_chunk_ref=request.pose_chunk_ref,
            image_chunk_ref=request.image_chunk_ref,
            span_spec=request.span_spec or "single",
            persist_to_minio=bool(request.persist_to_minio),
            dry_run=bool(request.dry_run),
        )
        self.settings.work_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="seg-", dir=str(self.settings.work_dir)))
        try:
            from .minio_io import materialize_job, s3_client, upload_bytes

            client = s3_client(self.settings)
            session_root = materialize_job(client, job, work)
            bundle = load_local_bundle(
                session_root, job.chunk_stems, self.settings, span=job.extra
            )
            with self._lock:
                result = run_sam3_on_bundle(
                    bundle, self.settings, work_dir=work / "run", detectores=self._detectores
                )
            if job.persist and result.status in {"ok", "skipped"}:
                key = f"{job.chunk_prefix(job.anchor_chunk)}roi_tracking.json"
                body = json.dumps(result.tracking, ensure_ascii=False, default=str).encode("utf-8")
                upload_bytes(client, job.bucket, key, body)
                result.tracking_ref = f"s3://{job.bucket}/{key}"
            return result
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _servicer_class(logic: SegmentationLogic):
    from kanvis.segmentation.v1 import segmentation_pb2_grpc

    class Bound(segmentation_pb2_grpc.SegmentationServiceServicer):
        def HealthCheck(self, request, context):
            return logic.health()

        def ApplySegmentation(self, request, context):
            return logic.apply(request)

    return Bound()


def serve(settings=None) -> None:
    import grpc
    from kanvis.segmentation.v1 import segmentation_pb2_grpc

    settings = settings or load_settings()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    if start_http_server is not None:
        start_http_server(settings.metrics_port)
        print(f"  metrics :{settings.metrics_port}/metrics")
    logic = SegmentationLogic(settings)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    segmentation_pb2_grpc.add_SegmentationServiceServicer_to_server(_servicer_class(logic), server)
    bind = f"{settings.host}:{settings.port}"
    server.add_insecure_port(bind)
    server.start()
    print(f"  gRPC {bind}")
    try:
        logic.load_models()
        print(f"  {logic._ready_msg}")
    except Exception as exc:
        logic._ready = False
        logic._ready_msg = f"model load failed: {exc}"
        print(f"  FAIL {logic._ready_msg}")
        traceback.print_exc()
    server.wait_for_termination()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
