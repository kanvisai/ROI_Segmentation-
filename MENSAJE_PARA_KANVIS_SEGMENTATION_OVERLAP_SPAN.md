# Mensaje para kanvis-segmentation (SAM3) — `span_spec` solape half-half + integración C3

**Audiencia:** equipo `kanvis-segmentation` (servicio GPU SAM3, gRPC `:50100`).  
**Origen contrato:** `kanvis-orchestrator/proto/segmentation.proto` + este documento.  
**Estado C3:** implementado en `main` / piloto wudalkot; depende de que SAM3 entienda el nuevo `span_spec`.

---

## 1. Contexto piloto (continuo v2)

Pipeline por sesión en MinIO (`kanvis-chunks/sessions/{retail}/{store}/{camera}/{session_id}/`):

| Orden | Etapa | C3 / servicio |
|-------|--------|----------------|
| 0 | Prefiltro muñecas | C3 (`frames_meta.json` C1 + gate por chunk) |
| 1 | Motion | C2 `Infer` → claves `p1.json`: `chunk_NNN` y solape `chunk_NNN_MMM` |
| 2 | HAR_v2 pocket | C3, mismas claves p1; poses solape = mitad final N + mitad inicial M |
| 3 | SAM3 ROI | **kanvis-segmentation** `ApplySegmentation` |

C3 **ya no** usa C2 `:50052` para SAM3 si `KANVIS_SEGMENTATION_GRPC_TARGET` está definido (p. ej. `kanvis-pilot.ai:50100`).

---

## 2. Problema a resolver

Para claves de **solape** (`chunk_001_002`), motion y HAR evalúan una ventana de **~8 s con stride 4 s**:

- **Frames:** última mitad del chunk `chunk_001` (p. ej. frames 13–24 si hay 24 JPEGs) + primera mitad de `chunk_002` (1–12).
- **Poses:** misma lógica en `poses_full.npy` (concatenación; ver `build_slide_window_poses` en HAR_v2).

Hasta ahora SAM3 recibía `span_spec=with_next` + refs del ancla `chunk_001/frames/` → **carpetas completas** N y N+1, **sin** alineación half-half.

**Objetivo:** que SAM3 procese **exactamente la misma ventana temporal** que motion/HAR en solape.

---

## 3. RPC `ApplySegmentation` (campos que C3 envía)

Proto (`segmentation.proto`):

```protobuf
message SegmentationRequest {
  string request_id = 1;
  string trace_id = 2;
  string session_id = 3;
  string source_ref = 4;           // s3://kanvis-chunks/sessions/.../session_id/
  string pose_chunk_ref = 5;       // relativo o absoluto al pose del ANCLA
  string image_chunk_ref = 6;      // relativo o absoluto a frames/ del ANCLA
  string span_spec = 7;            // ← clave del comportamiento
  bool persist_to_minio = 8;
  bool dry_run = 9;
  string bucket = 10;
  string retail_id = 11;
  string store_id = 12;
  string camera_id = 13;
  string chunk_stem = 14;          // stem lógico del candidato p1
}
```

### 3.1 Convenciones de refs (C3)

- `source_ref`: raíz sesión con barra final.
- `pose_chunk_ref` / `image_chunk_ref`: refs del **chunk ancla** (primer índice del par), típico:
  - `chunk_001/poses_full.npy`
  - `chunk_001/frames/`
- **`chunk_stem`:** stem del candidato en `p1.json`:
  - chunk simple → `chunk_001`
  - solape → `chunk_001_002`
- **`span_spec`:** ver sección 4.

C3 rellena `retail_id`, `store_id`, `camera_id` desde la sesión.

---

## 4. Valores de `span_spec` (contrato)

### 4.1 Existentes (mantener)

| `span_spec` | Significado |
|-------------|-------------|
| `single` | Solo chunk ancla (último chunk, o cierre forzado). |
| `with_next` | Chunk ancla **completo** + chunk N+1 **completo** (carpetas `frames/` y poses). |
| `with_prev` | (si lo soportáis) ancla + anterior. |
| `next` | Solo N+1. |
| `chunk_002,chunk_003` | Lista explícita de stems. |

### 4.2 **Nuevo — obligatorio para piloto v2**

Formato exacto que C3 emite:

```text
overlap_half:{left}:{right}:ratio={ratio}
```

Ejemplo:

```text
overlap_half:chunk_001:chunk_002:ratio=0.5000
```

- `{left}`, `{right}`: stems de chunk (3 dígitos).
- `{ratio}`: float `0.0–1.0`, por defecto **0.5** (8 s ventana, 4 s stride → 50 % solape).

#### Semántica que debe implementar SAM3

Dado `source_ref` y el par `(left, right)`:

1. **Listar / ordenar** JPEGs en:
   - `{source_ref}{left}/frames/`
   - `{source_ref}{right}/frames/`
   (misma convención de nombres que C1, orden temporal lexicográfico o por índice en `frames_meta.json` si existe).

2. **Selección de frames:**
   - `n_left = len(frames_left)`, `n_right = len(frames_right)`
   - `k_left = max(1, round(n_left * ratio))`
   - `k_right = max(1, round(n_right * ratio))`
   - Secuencia final: `frames_left[-k_left:]` **concatenado** con `frames_right[:k_right]` (orden temporal).

3. **Poses (tracking / alineación):**
   - Cargar `poses_full.npy` de `left` y `right`.
   - Misma ventana: `poses_left[-k_left:]` + `poses_right[:k_right]`.
   - Debe coincidir con HAR_v2 (`build_slide_window_poses`, ratio 0.5 por defecto en C3).

4. **Salida MinIO / `roi_fp_review`:**
   - Escribir resultado ROI en la clave **ancla** del p1: `chunk_001` (no en `chunk_001_002` como directorio físico; puede ser campo en `p1.json` bajo `chunk_001`, igual que C2 legacy).
   - Incluir en JSON de revisión, si es útil:
     - `p1_motion_stem`: `chunk_001_002`
     - `span_spec` usado
     - `frames_used`: `{ "left": [indices], "right": [indices] }` (opcional, debug)

5. **Gate muñecas en span:** antes del modelo, aplicar gate C1 sobre **todos** los `frames_meta` de frames usados en la ventana (misma filosofía que C2: sin muñecas → `status=skipped`, `hands_visible: false`).

6. **`dry_run`:** mismo comportamiento sin `persist_to_minio` (como hoy en smoke C3).

### 4.3 Cuándo C3 envía cada modo

| Candidato p1 | `chunk_stem` (RPC) | `span_spec` típico | `pose/image` refs base |
|--------------|-------------------|-------------------|-------------------------|
| `chunk_003` (solo chunk) | `chunk_003` | `with_next` | `chunk_003/...` |
| `chunk_001_002` (solape) | `chunk_001_002` | `overlap_half:chunk_001:chunk_002:ratio=0.5000` | `chunk_001/...` (ancla) |
| último chunk / sin N+1 | ancla | `single` | ancla |

C3 espera que **chunk `right` esté materializado** en MinIO antes de llamar (frames + meta + poses de `chunk_002`).

---

## 5. Ejemplo de request completo (solape)

```json
{
  "request_id": "...",
  "trace_id": "...",
  "session_id": "48773107eec74d46ab4222c0ab2d8e3d",
  "source_ref": "s3://kanvis-chunks/sessions/cliente-demo/.../48773107.../",
  "pose_chunk_ref": "chunk_001/poses_full.npy",
  "image_chunk_ref": "chunk_001/frames/",
  "span_spec": "overlap_half:chunk_001:chunk_002:ratio=0.5000",
  "chunk_stem": "chunk_001_002",
  "retail_id": "cliente-demo",
  "store_id": "...",
  "camera_id": "...",
  "persist_to_minio": true,
  "dry_run": false
}
```

Con **24 frames** por chunk → usar **12 + 12** frames (índices 1-based: 13–24 de chunk_001, 1–12 de chunk_002).

---

## 6. Smoke / verificación mutua

Desde host piloto (C3):

```bash
# Solape manual (cuando existan chunk_001 y chunk_002 en MinIO)
SMOKE_SAM3_SPAN_SPEC='overlap_half:chunk_001:chunk_002:ratio=0.5000' \
  ./scripts/smoke_pilot_chunk.sh --store ... --camera ... --session ... --chunk 001 --sam3-only
```

Tras implementar SAM3:

- `HealthCheck` → `ready=true`
- `ApplySegmentation` con el `span_spec` anterior → `status=ok` (o `skipped` si gate manos)
- Logs SAM3 deben mostrar conteo de frames `k_left`, `k_right` y stems.

---

## 7. Checklist implementación SAM3

- [x] Parser `span_spec` con prefijo `overlap_half:`.
- [x] Descarga MinIO multi-chunk (prefix `frames/`, `poses_full.npy`).
- [x] Selección half-half con `ratio` (default 0.5, tolerancia `round` como C3).
- [x] Pipeline interno (vídeo lógico o lista de paths) con longitud = `k_left + k_right`.
- [x] `roi_fp_review_json` en ancla; referencia a `chunk_stem` del solape.
- [x] Gate muñecas en ventana compuesta (`frames_meta`).
- [x] Compatibilidad: `with_next`, `single`, `dry_run`, `persist_to_minio`.
- [x] README / spec actualizado con ejemplos de este doc.
- [ ] Despliegue en piloto (`kanvis-pilot.ai:50100` o contenedor en red C3).

---

## 8. Coordinación despliegue

1. SAM3 implementa y despliega en wudalkot.
2. C3: `git pull` + `./deploy.sh` (target gRPC ya resuelto).
3. Sesión real con motion en solape + HAR `hand_to_body` → log C3 `c2_sam3_review_scheduled` con `span_spec=overlap_half:...`.
4. Validar `roi_fp_review` en `p1.json` del ancla.

---

## 9. Referencias en kanvis-orchestrator

| Tema | Ruta |
|------|------|
| Cálculo `span_spec` | `app/application/sam3_span_spec.py` |
| Cliente gRPC | `app/adapters/outbound/segmentation_client.py` |
| Proto | `proto/segmentation.proto` |
| HAR solape poses | `app/application/har_actions.py` → `build_slide_window_poses` |
| Red / deploy C3 | `docs/MENSAJE_PARA_KANVIS_SEGMENTATION_RED_C3.md` |
| Flujo motion+SAM3 | `docs/MENSAJE_PARA_C3_C2_BIN_FULL_Y_SAM3.md` |

---

## 10. Preguntas frecuentes

**¿Por qué no subir un MP4 pre-cortado desde C3?**  
Para no duplicar lógica ni storage; SAM3 ya lee MinIO; el contrato `overlap_half` fija la semántica.

**¿Y si `ratio` no es 0.5?**  
C3 puede cambiarlo si cambian `KANVIS_HAR_WINDOW_SECONDS` / `KANVIS_HAR_STRIDE_SECONDS`; SAM3 debe leer el valor del string.

**¿`chunk_stem` con solape sin carpeta física `chunk_001_002/`?**  
Correcto: es clave lógica p1; los artefactos viven bajo `chunk_001/` y `chunk_002/`.

**¿Unificación con C2 `ApplySam3Review`?**  
Mismo significado de `span_spec`; C3 ya no llama C2 para SAM3 en piloto si segmentation target está definido.

---

*Documento generado para handoff agente kanvis-segmentation ↔ C3. Versión alineada con commit que incluye `sam3_span_spec.py` y envío de `chunk_stem` en gRPC.*