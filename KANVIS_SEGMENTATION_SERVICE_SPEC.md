# Kanvis Segmentation Service (`kanvis-segmentation`) — Integration Specification

**Audience:** Engineer / coding agent owning the SAM3 / ROI project (today embedded or vendored under `roi_segmentation/`).  
**Goal:** Extract all SAM3 hand–object segmentation into a **standalone Docker service** that Kanvis C2 or C3 can call over **gRPC**. The service reads frames from **MinIO**, runs the existing SAM3 pipeline, and returns a **compact** tracking artifact equivalent to today’s `roi_tracking.json` plus a **C3-friendly** `roi_fp_review` summary.

**Language:** English (handoff doc).  
**Consumer repos:** `kanvis-ai-infserver` (C2 gateway), Kanvis C3 orchestrator.

---

## 1. Problem and success criteria

### 1.1 Today

- SAM3 runs **inside** the C2 inference gateway container (`ApplySam3Review` on port **50052**).
- C2 loads JPEGs from MinIO, invokes `sam3_mano_objeto` / ROI stack locally, writes `chunk_XXX/roi_tracking.json`, merges `roi_fp_review` into `p1.json`.
- Any change to the SAM3 repo forces C2 image rebuilds and coupled releases.

### 1.2 Target

| Requirement | Detail |
|-------------|--------|
| **Decoupling** | SAM3 lives only in **`kanvis-segmentation`** (separate repo + image). |
| **Same I/O contract** | Given the same MinIO session/chunk inputs, **outputs match** what C2 produced before (modulo intentional schema version bumps). |
| **Callers** | C2 **or** C3 may call segmentation directly; C2 may become a thin proxy. |
| **Operations** | Deploy via **`.sh` scripts**, smoke tests against **online or offline** MinIO paths, documented ports. |

### 1.3 Acceptance

1. `ApplySegmentation` gRPC against a known pilot session/chunk returns `status=ok` and valid JSON bodies.
2. `roi_fp_review` validates against `c2_roi_fp_review_v1` allowed keys.
3. Slim `tracking` payload is persistable as `roi_tracking.json` under the chunk prefix in MinIO (optional flag).
4. Multi-person frames: **session subject anchor** behavior matches Kanvis reference (`docs/SAM3_MULTI_PERSON_SESSION_ANCHOR.md` in `kanvis-ai-infserver`).
5. Agent provides **automated tests** runnable inside Docker with a documented MinIO URI or local fixture path.

---

## 2. Architecture

```text
                    ┌─────────────────┐
                    │  C3 orchestrator│
                    │  (optional)     │
                    └────────┬────────┘
                             │ gRPC :50100
              ┌──────────────┼──────────────┐
              │              ▼              │
              │   kanvis-segmentation       │
              │   (SAM3 / ROI stack)       │
              │              │              │
              │              │ S3 API      │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │     MinIO       │
                    │  kanvis-chunks  │
                    └─────────────────┘
                             ▲
              ┌──────────────┴──────────────┐
              │   C2 inference gateway      │
              │   (proxy or direct C3)      │
              │   :50052                    │
              └─────────────────────────────┘
```

**Principle:** Segmentation is **stateless** per request except for optional temp disk. Session truth is in **MinIO** (frames, poses, meta).

---

## 3. MinIO layout (must implement)

Standard session prefix (pilot example):

```text
s3://{bucket}/sessions/{retail_id}/{store_id}/{camera_id}/{session_id}/
  chunk_001/
    frames/frame_000.jpg …
    frames_meta.json
    poses_full.npy          # preferred for session anchor
    poses.npy               # fallback
    meta.json
  chunk_002/
    …
  p1.json                   # C2 motion; segmentation may read, must not destroy
  meta.json
  clip_buffer.mp4           # optional
```

**Resolving a job from tenant + session + chunk:**

| Input field | Example |
|-------------|---------|
| `bucket` | `kanvis-chunks` |
| `retail_id` | `cliente-demo` |
| `store_id` | `b1597c1f-6045-4d61-a147-75e1711aef1f` |
| `camera_id` | `416a662e-c64c-47ff-8a28-e2d2eac5bfde` |
| `session_id` | `8acfc86274ba4dd79892b18c65ac8d6c` |
| `chunk_stem` | `chunk_003` |

**Session root URI:**

```text
s3://kanvis-chunks/sessions/{retail_id}/{store_id}/{camera_id}/{session_id}/
```

**Frames for chunk `chunk_003`:**

```text
…/chunk_003/frames/
```

The service MUST accept either:

- **Structured fields** (`bucket`, `retail_id`, …), or  
- **Legacy refs** (same as C2 today): `source_ref` + `pose_chunk_ref` + optional `image_chunk_ref`.

Compatibility rule: if `pose_chunk_ref` is `…/chunk_003/poses_full.npy`, derive `chunk_stem=chunk_003` and `source_ref` = parent session directory (trailing `/`).

---

## 4. gRPC API (new service — do not use C2 port 50052)

### 4.1 Port allocation (Kanvis pilot — avoid conflicts)

| Service | Port | Notes |
|---------|------|--------|
| C2 inference gRPC | **50052** | `InferenceService` — do **not** bind here |
| C2 Prometheus | **9108** | |
| Triton | **8000–8002** | |
| MinIO | **9000** | |
| **kanvis-segmentation gRPC** | **50100** | **Default for this service** |
| **kanvis-segmentation metrics** | **9109** | Prometheus `/metrics` optional |

Env overrides: `KANVIS_SEGMENTATION_LISTEN=0.0.0.0:50100`, `KANVIS_SEGMENTATION_METRICS_PORT=9109`.

### 4.2 Proposed proto (`kanvis/segmentation/v1/segmentation.proto`)

Place proto in the **segmentation repo**; consumers copy or use `buf` / shared submodule later.

```protobuf
syntax = "proto3";
package kanvis.segmentation.v1;

service SegmentationService {
  rpc ApplySegmentation(SegmentationRequest) returns (SegmentationResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

message SegmentationRequest {
  string request_id = 1;
  string trace_id = 2;
  string session_id = 3;

  // --- Option A: structured tenant path ---
  string bucket = 10;
  string retail_id = 11;
  string store_id = 12;
  string camera_id = 13;
  // Session UUID (hex, no dashes allowed in pilot paths)
  // session_id field 3 duplicates this when using structured path.

  // Chunk anchor: "chunk_003"
  string chunk_stem = 14;

  // --- Option B: legacy Kanvis refs (same semantics as Sam3ReviewRequest) ---
  string source_ref = 4;        // session prefix, trailing slash
  string pose_chunk_ref = 5;    // e.g. .../chunk_003/poses_full.npy
  string image_chunk_ref = 6;   // optional .../chunk_003/frames/

  // Span: "", "single", "with_prev", "with_next", "chunk_002,chunk_003"
  string span_spec = 7;

  // If true, write roi_tracking.json under chunk anchor in MinIO
  bool persist_to_minio = 8;

  // Dry-run: compute only, no S3 writes (mirrors x-kanvis-sam3-persist=0)
  bool dry_run = 9;
}

message SegmentationResponse {
  string request_id = 1;
  // ok | error | skipped | pending
  string status = 2;
  string error_message = 3;

  // Full slim tracking JSON (see §5.1) — UTF-8 JSON string
  string tracking_json = 4;

  // C3 summary (see §5.2) — schema c2_roi_fp_review_v1
  string roi_fp_review_json = 5;

  // s3://…/chunk_XXX/roi_tracking.json if persist_to_minio && !dry_run
  string tracking_ref = 6;
}

message HealthCheckRequest {}
message HealthCheckResponse {
  bool ready = 1;
  string message = 2;
}
```

**Parity with existing C2 RPC:** Field meanings MUST match `Sam3ReviewRequest` / `Sam3ReviewResponse` in `orchestrator.proto` (`ApplySam3Review`). C2 can forward requests 1:1 after a thin adapter.

---

## 5. Output contracts

### 5.1 `tracking_json` / MinIO `roi_tracking.json`

- Pipeline today writes an internal `tracking.json` with optional **`--summary`** mode (`C2_SAM3_SUMMARY=true`).
- For MinIO persistence, C2 applies `slim_tracking_payload_for_storage()`:
  - Drops bloated `identifiers_yaml` catalog; keep path in `parameters.identifiers_yaml` if needed.
- **Your service** should produce the **same slim shape** C2 already stores as:

```text
{session_prefix}/chunk_{NNN}/roi_tracking.json
```

**Minimum fields** (non-exhaustive; must match your current summary output):

- `final_state` — e.g. `in_hand`, `inside_container`, `disappeared`, `no_hands`, `no_data`, …
- `hands_visible` — boolean
- Fields needed to derive C3 confidences (phone, container), as today in `summarize_tracking_for_c3()`.

**Reference implementation (read-only for you):**

- `kanvis-ai-infserver/scripts/c2_sam3_roi.py` — `slim_tracking_payload_for_storage`, `summarize_tracking_for_c3`
- `kanvis-ai-infserver/scripts/c2_inference_service.py` — `_persist_roi_tracking`

### 5.2 `roi_fp_review_json` (C3)

Schema version: **`c2_roi_fp_review_v1`**.

**Allowed keys only** (reject or strip others):

- `schema_version`
- `final_state`
- `hands_visible`
- `cell_phone_confidence`
- `inside_container_confidence`
- `container`

This is what C3 merges into `p1.json` under the anchor chunk (e.g. `chunk_003.roi_fp_review`).

### 5.3 RPC `status`

| status | When |
|--------|------|
| `ok` | Segmentation ran; review has usable state |
| `skipped` | Wrist gate / no hands / policy (`hands_visible=false`, `final_state=no_hands`, …) |
| `error` | Missing frames, I/O, model failure |
| `pending` | Optional async mode only (default: synchronous) |

---

## 6. Session anchor (multi-person frames)

**Mandatory** for production parity with Kanvis C1/C2.

Before SAM3, filter YOLO persons/hands to the **session subject** using C1 `poses_full.npy` wrists per frame alignment (`frames_meta.json` / `frames_origin`).

Full algorithm: **`kanvis-ai-infserver/docs/SAM3_MULTI_PERSON_SESSION_ANCHOR.md`**.

Your service must load, for every chunk in `span_spec`:

- `frames/` + `frames_meta.json`
- `poses_full.npy` (fallback `poses_compat.npy` / `poses.npy`)

If you do not implement anchoring, outputs **will diverge** from C2 on busy shop floors.

---

## 7. Environment variables (segmentation container)

| Variable | Default | Purpose |
|----------|---------|---------|
| `KANVIS_SEGMENTATION_LISTEN` | `0.0.0.0:50100` | gRPC bind |
| `KANVIS_SEGMENTATION_METRICS_PORT` | `9109` | Prometheus |
| `SEGMENTATION_S3_ENDPOINT_URL` | — | MinIO endpoint (`http://minio:9000` in compose) |
| `SEGMENTATION_S3_ACCESS_KEY_ID` | — | |
| `SEGMENTATION_S3_SECRET_ACCESS_KEY` | — | |
| `SEGMENTATION_S3_REGION` | `us-east-1` | |
| `SEGMENTATION_S3_SECURE` | `false` | |
| `SEGMENTATION_SAM3_SUMMARY` | `true` | Compact tracking (parity `C2_SAM3_SUMMARY`) |
| `SEGMENTATION_SAM3_MIN_WRIST_VISIBLE_FRAMES` | `3` | Wrist gate |
| `SEGMENTATION_SAM3_WRIST_MATCH_MAX_PX` | `140` | Anchor match radius |
| `SEGMENTATION_IDENTIFIERS_YAML` | — | Catalog path inside image |
| `SEGMENTATION_WORK_DIR` | `/tmp/kanvis-segmentation` | Scratch (ephemeral) |

Align names with existing C2 `C2_SAM3_*` where behavior is identical.

---

## 8. Docker image and compose (deliverables)

### 8.1 Image name

`kanvis-segmentation:latest` (GPU variant: `kanvis-segmentation:gpu`).

**Requirements:**

- Python 3.10+ (or project baseline)
- SAM3 / ROI dependencies bundled **inside this image only**
- No dependency on C2 gateway image

### 8.2 Example `docker-compose.segmentation.yml` (snippet for your repo)

```yaml
services:
  kanvis-segmentation:
    image: kanvis-segmentation:gpu
  # build: .
    environment:
      KANVIS_SEGMENTATION_LISTEN: "0.0.0.0:50100"
      SEGMENTATION_S3_ENDPOINT_URL: "http://kanvis-monitoring-minio-1:9000"
      SEGMENTATION_S3_ACCESS_KEY_ID: "minioadmin"
      SEGMENTATION_S3_SECRET_ACCESS_KEY: "minioadmin"
    ports:
      - "50100:50100"
      - "9109:9109"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Connect to the **same Docker network** as MinIO and C2/C3, or publish `50100` on the pilot host firewall.

---

## 9. Shell scripts (required deliverables)

Provide in the **segmentation repo**:

| Script | Purpose |
|--------|---------|
| `scripts/build.sh` | Build Docker image (GPU/CPU targets) |
| `scripts/deploy.sh` | Pull/build + `docker compose up -d` |
| `scripts/smoke_grpc.sh` | Call `ApplySegmentation` with env-based target |
| `scripts/smoke_minio.sh` | End-to-end against a fixed session URI |
| `scripts/run_local_offline.sh` | No MinIO; local folder layout |

All scripts must:

- Exit non-zero on failure
- Print explicit `PASS` / `FAIL`
- Support `DRY_RUN=1` (no MinIO writes)

### 9.1 Example smoke (online MinIO)

```bash
#!/usr/bin/env bash
# scripts/smoke_minio.sh — run INSIDE container or from host with grpcurl/python client

set -euo pipefail

TARGET="${TARGET:-localhost:50100}"
SOURCE_REF="${SOURCE_REF:?s3 session prefix with trailing slash}"
POSE_REF="${POSE_REF:?e.g. .../chunk_001/poses_full.npy}"
SESSION_ID="${SESSION_ID:-smoke-session}"
REQUEST_ID="smoke-$(date +%s)"

python3 tools/grpc_smoke_client.py \
  --target "$TARGET" \
  --request-id "$REQUEST_ID" \
  --session-id "$SESSION_ID" \
  --source-ref "$SOURCE_REF" \
  --pose-chunk-ref "$POSE_REF" \
  --span-spec "${SPAN_SPEC:-single}" \
  --dry-run "${DRY_RUN:-1}"

echo "PASS smoke_minio"
```

**Pilot example URIs** (replace with a real session from `mc find`):

```bash
export SOURCE_REF="s3://kanvis-chunks/sessions/cliente-demo/b1597c1f-6045-4d61-a147-75e1711aef1f/416a662e-c64c-47ff-8a28-e2d2eac5bfde/8acfc86274ba4dd79892b18c65ac8d6c/"
export POSE_REF="${SOURCE_REF}chunk_001/poses_full.npy"
export TARGET="wudalkot.ddns.net:50100"
bash scripts/smoke_minio.sh
```

### 9.2 Example offline / fixture

Directory layout (mount as volume):

```text
/fixtures/session_001/
  chunk_001/frames/frame_*.jpg
  chunk_001/frames_meta.json
  chunk_001/poses_full.npy
  chunk_001/meta.json
```

```bash
docker run --rm -v "$PWD/fixtures:/fixtures:ro" kanvis-segmentation:gpu \
  python3 tools/run_offline_segmentation.py \
    --local-session-root /fixtures/session_001 \
    --chunk-stem chunk_001 \
    --output /tmp/tracking.json

test -f /tmp/tracking.json && jq -e '.final_state' /tmp/tracking.json
```

---

## 10. gRPC smoke client (template for your agent)

Ship `tools/grpc_smoke_client.py` in the segmentation repo (analogous to `kanvis-ai-infserver/scripts/smoke_c2_sam3_grpc_client.py`):

```python
#!/usr/bin/env python3
"""Minimal ApplySegmentation client."""
import argparse
import json
import grpc
# import segmentation_pb2, segmentation_pb2_grpc  # generated from your proto

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="localhost:50100")
    p.add_argument("--request-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--source-ref", required=True)
    p.add_argument("--pose-chunk-ref", required=True)
    p.add_argument("--span-spec", default="single")
    p.add_argument("--dry-run", type=int, default=1)
    args = p.parse_args()
    # channel = grpc.insecure_channel(args.target)
    # stub = SegmentationServiceStub(channel)
    # resp = stub.ApplySegmentation(SegmentationRequest(...))
    # assert resp.status in ("ok", "skipped")
    # json.loads(resp.roi_fp_review_json)
    print("Implement with generated stubs")

if __name__ == "__main__":
    main()
```

---

## 11. Testing checklist (agent must execute and document)

- [ ] **Unit:** `tracking_json` slimming; `roi_fp_review` key whitelist.
- [ ] **Unit:** Session anchor picks correct person when 2+ YOLO detections (synthetic or recorded frames).
- [ ] **Integration:** Read frames from MinIO path; temp dir cleaned on exit.
- [ ] **Integration:** `dry_run=true` → no `roi_tracking.json` in bucket.
- [ ] **Integration:** `persist_to_minio=true` → object at `…/chunk_XXX/roi_tracking.json`.
- [ ] **Parity:** Same session/chunk as historical C2 `roi_tracking.json` → `final_state` and `hands_visible` match (allow float tolerance on confidences).
- [ ] **Load:** 1 concurrent request stable; document GPU memory.
- [ ] **Health:** `HealthCheck` returns `ready=true` after models loaded.

Record results in `docs/TEST_REPORT.md` in the segmentation repo with session IDs and timestamps.

---

## 12. Future wiring in Kanvis C2 (for your awareness)

C2 will eventually:

1. Set `C2_SAM3_REMOTE_TARGET=kanvis-segmentation:50100` (name TBD).
2. `ApplySam3Review` forwards to `ApplySegmentation` instead of in-process ROI.
3. Optionally merge `roi_fp_review` into `p1.json` exactly as today.

**You do not need to implement C2 changes** in this task; deliver a **standalone** service that passes §11.

---

## 13. References in `kanvis-ai-infserver`

| Topic | Path |
|-------|------|
| SAM3 session anchor | `docs/SAM3_MULTI_PERSON_SESSION_ANCHOR.md` |
| C2 SAM3 integration | `scripts/c2_sam3_roi.py`, `scripts/c2_roi_session_anchor.py` |
| gRPC today | `proto/orchestrator.proto` → `ApplySam3Review` |
| C2 smoke client | `scripts/smoke_c2_sam3_grpc_client.py` |
| Pilot ports C2 | `deploy/docker-compose.yml` (`50052`, `9108`) |

---

## 14. Agent handoff summary

**Build `kanvis-segmentation`:**

1. gRPC on **50100**, metrics **9109**.
2. Load `chunk_XXX/frames` (+ poses/meta) from MinIO using tenant path or legacy refs.
3. Run existing SAM3 pipeline with **session anchor** and **summary** tracking.
4. Return `tracking_json` + `roi_fp_review_json`; optional persist `roi_tracking.json`.
5. Ship **Docker**, **`deploy.sh`**, **`smoke_minio.sh`**, **offline runner**, and **TEST_REPORT** after green smokes.

When this is done, Kanvis C2/C3 only change **endpoint configuration**, not SAM3 internals.