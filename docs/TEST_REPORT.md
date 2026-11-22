# kanvis-segmentation — test report

**When:** 22 Nov 2026, Europe/Madrid (UTC+1)  
**Host:** local workspace `ROI_Segmentation-`  
**Python:** `.venv/bin/python` (3.12)

## Unit tests

Command: `.venv/bin/python -m pytest tests -q`

| Case | Result |
|------|--------|
| `tracking_json` slimming (drop `identifiers_yaml` catalog) | PASS |
| `roi_fp_review` key whitelist `c2_roi_fp_review_v1` | PASS |
| RPC status `skipped` on `no_hands` | PASS |
| Legacy `pose_chunk_ref` → `chunk_stem` + session prefix | PASS |
| Structured tenant path | PASS |
| `span_spec` with_prev / with_next | PASS |
| Session anchor picks YOLO person nearest C1 wrists | PASS |
| Wrist gate (min frames / fraction) | PASS |
| `dry_run=true` forces `persist=false` | PASS |

**10 passed.**

## Scripts (DRY_RUN)

| Script | Command | Result |
|--------|---------|--------|
| `scripts/gen_proto.sh` | `bash scripts/gen_proto.sh` | PASS (stubs in `kanvis/segmentation/v1/`) |
| `scripts/build.sh` | `DRY_RUN=1 bash scripts/build.sh` | PASS (no image build) |
| `scripts/deploy.sh` | `DRY_RUN=1 bash scripts/deploy.sh` | compose config check |
| `scripts/run_local_offline.sh` | `DRY_RUN=1 bash scripts/run_local_offline.sh` | PASS (no SAM3 / no MinIO write) |
| `scripts/smoke_minio.sh` | requires live gRPC + `SOURCE_REF` | not run against piloto MinIO in this pass |

## Integration not executed on this pass

These need the GPU image up and a real (or mocked) MinIO:

- Read frames from MinIO; temp dir cleaned on exit
- `persist_to_minio=true` → `…/chunk_XXX/roi_tracking.json`
- Parity vs historical C2 `roi_tracking.json` (`final_state` / `hands_visible`)
- Load test (1 concurrent request) / GPU memory
- `HealthCheck` `ready=true` after models loaded (requires `python -m kanvis_segmentation.server` with CUDA)

Offline fixture path exists at `fixtures/session_001/chunk_001/` (3 JPEG + `poses_full.npy` + `frames_meta.json`).

## How to run remaining smokes

```bash
bash scripts/build.sh                 # TARGET=gpu
bash scripts/deploy.sh                # SKIP_BUILD=1 if image already tagged
TARGET=localhost:50100 bash scripts/smoke_grpc.sh

export SOURCE_REF="s3://kanvis-chunks/sessions/<retail>/<store>/<camera>/<session>/"
export POSE_REF="${SOURCE_REF}chunk_001/poses_full.npy"
DRY_RUN=1 bash scripts/smoke_minio.sh
```
