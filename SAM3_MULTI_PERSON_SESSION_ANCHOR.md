# SAM3 / ROI: Selecting One Person in Multi-Person Frames (Session Anchor)

**Audience:** Engineers implementing or porting the C2 ROI + SAM3 path (hand–object review) who do **not** yet have single-subject filtering in their fork.

**Role:** Read this document as a **computer vision and pose-tracking expert**. Your deliverable is a **fully functional** integration that, when a frame contains **multiple people**, runs SAM3 (and downstream hand logic) **only for the session’s target subject**, **ignoring all other detections**.

**Status in Kanvis `kanvis-ai-infserver`:** Implemented in `scripts/c2_roi_session_anchor.py`, wired from `scripts/c2_inference_service.py` (`ApplySam3Review`) via `scripts/c2_sam3_roi.py` (`run_roi_pipeline_on_bundle`).

---

## 1. Problem statement

### 1.1 What goes wrong without anchoring

Retail theft pipelines often trigger SAM3 / hand-in-ROI review on a **short frame span** (one chunk or chunk + neighbor). The ROI stack typically:

1. Runs **person detection + pose** (e.g. YOLO-pose) on each frame → **N ≥ 1 persons**.
2. Runs **hand / palm** detection and **SAM3** segmentation on hands and objects.

If **N > 1**, the pipeline may:

- Segment **the wrong person’s** hands or a bystander’s interaction with a shelf.
- Produce **`ok` / `in_hand`** states that are **not** about the customer associated with the **motion / pose session** used for robbery scoring.
- Increase false positives and break trust in C3 orchestration (HAR, VLM, alerts).

You need a deterministic rule: **“this MinIO session / this Infer request refers to subject S; only S may drive SAM3.”**

### 1.2 What “tracking” means here

| Meaning | Required? |
|--------|-----------|
| Persistent **track ID** across minutes (ReID, multi-camera) | Optional product feature; **not** what the reference C2 code implements today. |
| **Per-frame** choice of which YOLO person matches the **C1 session subject** | **Yes** — minimum bar for SAM3 correctness. |
| **Temporal smoothing** (same ID across frames in the chunk) | **Strongly recommended** for production; reference code re-selects each frame (see §5.4). |

---

## 2. System context (C1 → C2 → C3)

```text
C1 (edge / continuous pose)
  └─ MinIO chunk: frames/, frames_meta.json, poses_full.npy, valid_mask.npy, meta.json
       │
       ▼
C2 Infer (motion TCN) ──► p1.json (robbery_score, robbery_alarm, …)
       │
       ▼
C2 ApplySam3Review (gRPC) ──► ROI pipeline + SAM3 ──► tracking.json ──► p1 roi_fp_review
       │
       ▼
C3 orchestrator (HAR, VLM, business rules)
```

**Design principle:** **C1 owns “who is the subject of this session/chunk”** via pose export and per-frame wrist metadata. **C2 must not invent a different subject** at SAM3 time. C2 **filters** third-party ROI detections to match C1.

---

## 3. Reference solution (Kanvis infserver)

### 3.1 High-level flow

1. **Load frame bundle** for the requested span (`chunk_only`, `with_next`, etc.): BGR frames, dimensions, per-frame `frames_meta` rows, `frame_chunk_names`, `frames_origin` (index into chunk pose timeline).
2. **Wrist gate (span-level):** If C1 `frames_meta` shows insufficient wrist visibility across the span, **skip SAM3** entirely (`status: skipped`, `final_state: no_hands`). No point running SAM on backs / empty wrists.
3. **Load `poses_full.npy`** for each chunk in the span from the same `source_ref` as the request.
4. **Install a frame hook** before ROI `AnalizadorEscena.analizar` runs:
   - YOLO still detects all persons.
   - **Filter** `contexto.personas` to **one** person and **filter hands** to that person’s id.
5. Run **`procesar_video`** (SAM3 backend) on the filtered context only.
6. Persist **`tracking.json`**, summarize into **`roi_fp_review`** on `p1.json` for C3.

Entry points:

- `c2_inference_service._run_sam3_review_sync`
- `c2_sam3_roi.run_roi_pipeline_on_bundle(..., session_anchor=...)`
- `c2_roi_session_anchor.install_session_person_filter`

### 3.2 Session anchor data structure

`SessionRoiAnchor`:

- `bundle` — merged `ChunkFramesBundle` (frames + meta + alignment fields).
- `poses_by_chunk` — `dict[chunk_name, poses_full array (T, 8, 2)]`.
- `frame_idx` — cursor incremented after each analyzed frame.

For frame `i`:

1. Resolve `chunk = frame_chunk_names[i]`, `source_index = frames_origin[i]`.
2. Read wrists from `poses_by_chunk[chunk][source_index]` at keypoint slots **4 and 5** (GuadIA order, COCO wrists 9–10 in the 8-KP export).
3. Convert normalized **[0,1]** coords to **pixels** using `bundle.width` / `bundle.height` when `max(|x|,|y|) ≤ 1.5`.

### 3.3 Per-frame person selection algorithm

**Input:** `contexto.personas` from ROI YOLO-pose (each with `keypoints`, shape ≥ 11×3 COCO-style).

**Input (session):** list of wrist points `session_pts` from C1 `poses_full` (0, 1, or 2 wrists).

**Score each persona:**

```text
For each YOLO person P:
  For each session wrist S in session_pts:
    Find best YOLO wrist keypoint (indices 9, 10) with conf >= yolo_conf (default 0.2)
    Compute distance d = ||P_wrist - S|| in pixels
  If any d <= WRIST_MATCH_MAX_PX (default 140):
    score(P) = sum over matched wrists of (-d)   # closer is better
  Else:
    score(P) = -inf

Pick argmax score(P).
If best score is -inf:
  Clear personas and manos (no safe match).
Else:
  contexto.personas = [best P]
  contexto.manos = [hands with persona_id == index of best P]
```

**Fallback when `session_pts` is empty** but `frames_meta` still claims wrist visibility: score by **sum of YOLO wrist confidences** only (weak; prefer fixing C1 alignment).

**When meta says no wrist visibility and no session points:** clear all personas (forces no-hand path).

Constants (env):

| Variable | Default | Role |
|----------|---------|------|
| `C2_SAM3_WRIST_MATCH_MAX_PX` | `140` | Max pixel distance to accept a YOLO person as the session subject |
| `C2_SAM3_MIN_WRIST_CONF` / `C2_VLM_SKIP_IF_MAX_WRIST_CONF_BELOW` | `0.35` | Wrist visibility in `frames_meta` |
| `C2_SAM3_MIN_WRIST_VISIBLE_FRAMES` | `3` | Min frames with visible wrists in span to run SAM3 |
| `C2_SAM3_MIN_WRIST_VISIBLE_FRACTION` | `0.10` | Min fraction of span frames with visible wrists |

Implementation: `filter_contexto_to_session_person`, `_persona_wrist_score`, `session_wrists_px_from_pose` in `c2_roi_session_anchor.py`.

### 3.4 Integration mechanism (monkey-patch)

ROI code calls `AnalizadorEscena.analizar(frame)` internally. The reference implementation **wraps** that method:

```python
def _wrapped_analizar(self, frame):
    ctx = original(self, frame)
    filter_contexto_to_session_person(ctx, anchor)
    anchor.frame_idx += 1
    return ctx
```

After `procesar_video` finishes, the original method is restored (`PersonFilterInstall.uninstall()`).

**Why patch instead of forking ROI:** Keeps the vendor/submodule ROI repository unchanged; all session logic lives in C2.

### 3.5 Span-level wrist gate (before SAM3)

Even with per-frame filtering, running SAM on spans where **C1 never saw wrists** (e.g. person facing away) wastes GPU and creates bogus states.

`evaluate_sam3_wrist_gate(bundle)` counts frames where `frames_meta` passes `frame_has_wrist_visibility` (left/right wrist visible with conf ≥ threshold). Requires:

- `visible_frames >= C2_SAM3_MIN_WRIST_VISIBLE_FRAMES`, and  
- `visible_frames / total_frames >= C2_SAM3_MIN_WRIST_VISIBLE_FRACTION`.

If gate fails → skip SAM3; review payload includes `session_person_anchor: "wrist_c1"` and skip reason.

---

## 4. C1 data contract (required for a working anchor)

Your C1 (or equivalent) export **must** provide:

| Artifact | Purpose |
|----------|---------|
| `chunk_NNN/poses_full.npy` | `(T, 8, 2)` float32, wrists at KP indices 4–5, coords normalized or pixel-consistent |
| `chunk_NNN/frames_meta.json` (or v1 bundle meta) | Per sampled frame: `left_wrist_visible`, `right_wrist_visible`, confidences |
| `frames/` + alignment | `frames_origin[i]` maps bundle frame `i` → row index in that chunk’s pose array |
| `frame_chunk_names[i]` | Which chunk owns bundle frame `i` when merging spans |

**Strongly recommended (not yet mandatory in reference C2):**

| Field | Purpose |
|-------|---------|
| `meta.json` → `track_id` / `person_id` / `dominant_user_index` | Explicit subject id when C1 already solved multi-track offline |
| Per-frame `person_bbox` or `yolo_track_id` in meta | Lets you match by IoU or id instead of pure wrist distance |

Without alignment (`frames_origin`, chunk names), wrist positions will be taken from the **wrong time index** and SAM3 will lock onto the wrong body.

---

## 5. Limitations of the reference approach

1. **No cross-frame track continuity:** Each frame independently picks the closest YOLO person to C1 wrists. A person swapping places or occlusion can flip the selected index frame-to-frame.
2. **Depends on C1 pose quality:** Wrong or jittery C1 wrists → wrong YOLO association (especially when two people stand close).
3. **2D distance only:** No depth / homography; 140 px is a tunable heuristic.
4. **Single subject only:** Sessions explicitly multi-customer (e.g. family at counter) are out of scope; product should enforce `max_persons_in_frame=1` or define which track is “dominant.”
5. **Dual detector stack:** C1 pose (GuadIA / export pipeline) vs ROI YOLO-pose — wrists must correspond in **the same image coordinate system** (resolution, letterboxing).

---

## 6. Alternative and improved approaches

Evaluate these as a CV expert; combine where useful.

### 6.1 C1 explicit subject ID (recommended upgrade)

**Idea:** C1 already picks `user_0` / `track_id` for motion export. Propagate to `meta.json` and optionally per-frame `track_id` in `frames_meta`.

**C2 change:** If present, filter YOLO persons by matching `track_id` or highest IoU with C1 bbox — wrists become a **sanity check**, not the only signal.

**Pros:** Stable across frames; minimal ambiguity when tracks don’t cross.  
**Cons:** Requires C1 multi-person tracker and contract versioning.

### 6.2 Temporal tracker on top of per-frame association

**Idea:** After wrist-based pick, run a lightweight **IoU / OKS tracker** (SORT, ByteTrack-style on bboxes) on the **selected** person only; on later frames, prefer the track id that was chosen initially, using wrist match as correction.

**Pros:** Reduces identity swaps when two people are close.  
**Cons:** Extra code in C2 or ROI; need failure recovery when track is lost.

### 6.3 Single detector source (pose consistency)

**Idea:** Use **one** pose model for both C1 export and ROI (same weights, same letterbox), or warp C1 keypoints into ROI inference coordinates via known resize metadata in `meta.json`.

**Pros:** Removes cross-model wrist mismatch.  
**Cons:** Couples C1 and ROI release cycles.

### 6.4 ReID embedding

**Idea:** At session start, embed the dominant person (OSNet, CLIP crop); for each frame pick the detection with highest similarity.

**Pros:** Works when wrists occluded but face/torso visible.  
**Cons:** Latency, GPU, need gallery from first good frame.

### 6.5 Geometric / zone priors

**Idea:** “Customer” is in checkout ROI polygon; filter detections whose hip center lies in zone.

**Pros:** Simple stores with fixed cameras.  
**Cons:** Fails for wide FOV or wandering subjects; use as **prior**, not sole rule.

### 6.6 Ranking for production

| Priority | Approach |
|----------|----------|
| P0 (ship now) | Wrist gate + per-frame wrist distance filter (**reference implementation**) |
| P1 | C1 `track_id` / bbox in meta + IoU match |
| P2 | Temporal smoothing tracker on selected bbox |
| P3 | Unified pose model / ReID for hard scenes |

---

## 7. Implementation guide (port to another repository)

**Goal:** SAM3 runs as if only one person exists in every frame — the session subject.

### 7.1 Steps

1. **Define `ChunkFramesBundle`** (or equivalent) with: `frames_bgr`, `width`, `height`, `frames_origin`, `frame_chunk_names`, `frame_meta_rows`, `chunk_name`, `video_fps`.
2. **Implement `load_poses_for_chunks(source_ref, chunk_names)`** reading `poses_full.npy` from object storage.
3. **Implement `SessionRoiAnchor`** + `session_wrists_px_from_pose` with the **same KP index convention** as your C1 export (document indices in your repo).
4. **Implement `evaluate_sam3_wrist_gate`** using your C1 wrist visibility flags (or compute from pose confidences if meta missing).
5. **Hook ROI scene analysis** at the narrowest point where `personas` and `manos` are available **before** SAM3 consumes them — reference uses `AnalizadorEscena.analizar` wrap; your fork may use a cleaner dependency-injected filter.
6. **Always uninstall / restore** hooks after the video loop to avoid leaking global state across gRPC requests.
7. **Write review JSON** with explicit fields: `session_person_anchor: "wrist_c1"`, `wrists_meta_summary`, skip reasons.
8. **Unit tests:** synthetic poses + fake personas with keypoints at known offsets; assert filter keeps correct index.
9. **Integration test:** two-person still image or clip; assert `tracking.json` only reflects one hand stream.

### 7.2 Pseudocode (framework-agnostic)

```python
class SessionAnchor:
    def wrists_px(self, frame_index: int) -> list[tuple[float, float]]:
        ...

def select_person_index(persons, wrists_px, max_dist_px=140) -> int | None:
    best_i, best_score = None, float("-inf")
    for i, p in enumerate(persons):
        s = wrist_match_score(p.keypoints, wrists_px, max_dist_px)
        if s > best_score:
            best_i, best_score = i, s
    return best_i if best_score > float("-inf") else None

def filter_scene_context(ctx, anchor: SessionAnchor):
    i = anchor.frame_index
    wrists = anchor.wrists_px(i)
    if not wrists and not meta_has_wrists(anchor, i):
        ctx.persons = []
        ctx.hands = []
        return
    j = select_person_index(ctx.persons, wrists)
    if j is None:
        ctx.persons = []
        ctx.hands = []
        return
    ctx.persons = [ctx.persons[j]]
    ctx.hands = [h for h in ctx.hands if h.person_index == j]
    anchor.frame_index += 1
```

### 7.3 Acceptance criteria

- [ ] Multi-person frame: SAM3 masks / hand states refer to the **same individual** as C1 `poses_full` for that timestamp (manual review on ≥ 5 clips).
- [ ] Single-person pilot clips: behavior unchanged vs unfiltered pipeline.
- [ ] Span with zero wrist visibility: SAM3 **skipped**, no `ok+in_hand` from random bystander.
- [ ] Concurrent gRPC reviews: no cross-session contamination (hook restored per call).
- [ ] Documented env thresholds and tuning procedure for `WRIST_MATCH_MAX_PX`.

---

## 8. Testing and debugging

| Symptom | Likely cause |
|---------|----------------|
| Wrong person segmented | `frames_origin` mismatch; wrong chunk in `frame_chunk_names`; C1/C2 resolution mismatch |
| Empty personas every frame | All distances > `WRIST_MATCH_MAX_PX`; C1 wrists NaN; YOLO wrist conf < 0.2 |
| SAM always skipped | Wrist gate thresholds too strict; `frames_meta` not populated |
| Identity flip mid-chunk | Expected with per-frame-only logic; add temporal tracker (§6.2) |

**Debug outputs to log per frame (optional):** `frame_idx`, `session_pts`, chosen `persona_index`, best distance, number of YOLO persons.

**MinIO checks:** Compare overlay of C1 wrists (projected) vs YOLO wrists on a saved JPEG for one frame.

---

## 9. Relation to motion / guardrails

SAM3 subject selection is **orthogonal** to motion guardrails (`c2_motion_c1_guardrails.py`) and `robbery_alarm`. Motion uses accumulated `poses_full` + `valid_mask`; SAM3 uses **RGB frames** + the **same** C1 pose anchor for **who** to segment. Both should agree on the subject via C1 exports.

---

## 10. Summary

- **Problem:** Multi-person frames break SAM3 unless you restrict to the **session customer**.
- **Reference Kanvis solution:** C1 wrist positions from `poses_full` + per-frame YOLO-pose association inside ROI, with a **span-level wrist visibility gate** before SAM3.
- **Not true long-term tracking:** Re-select each frame; upgrade with C1 `track_id` + temporal smoothing for production hardening.
- **Your task as implementer:** Port the hook + anchor + gate, validate alignment with C1, tune `WRIST_MATCH_MAX_PX`, and extend with track id from C1 when available.

---

## 11. File map (Kanvis reference)

| File | Responsibility |
|------|----------------|
| `scripts/c2_roi_session_anchor.py` | Anchor, wrist gate, person filter, hook install |
| `scripts/c2_sam3_roi.py` | ROI pipeline driver, optional `session_anchor` |
| `scripts/c2_inference_service.py` | `ApplySam3Review`, bundle load, persist review |
| `scripts/c2_chunk_frames.py` | Frame bundle load/merge (alignment fields) |

For pose / motion contract (separate from SAM3): `docs/MENSAJE_C2_POSE_KANVIS_V1.md`.

---

*Document version: 2026-03 (piloto Kanvis continuous v2, IAModel full pose export).*
