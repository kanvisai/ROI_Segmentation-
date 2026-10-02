# Seguimiento del objeto en la mano

Repositorio de experimentación: seis tecnologías segmentan y siguen el objeto que una persona lleva en la mano, sobre los mismos clips, con la **misma detección previa** y el **mismo layout de salida**. Sirve para comparar backends y para integrar el pipeline en otro sistema.

Siempre ejecutar con el intérprete del venv local: `.venv/bin/python` (CPython 3.12, PyTorch 2.10 + CUDA 12.8). No usar el `python` del sistema ni un conda ajeno: carece de Pose26 / Ultralytics de este proyecto.

## Contrato para integrar el código

Un agente o un servicio externo puede reutilizar el repo entero o solo el pipeline común. Restricciones que hay que respetar:

| Tema | Contrato |
|---|---|
| Python | 3.11 o superior. El venv del repo es 3.12.13. |
| GPU | NVIDIA con CUDA 12+. PyTorch informa CUDA 12.8. Sin GPU los scripts abortan. |
| Intérprete | `.venv/bin/python` desde la raíz del repo. |
| Vídeos | Carpeta `videos/` (gitignored). Extensiones: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. |
| Pesos | Carpeta `pesos/` (gitignored salvo que alguien deje un `.gitkeep`). |
| Terceros | `third_party/` (gitignored): clones de Cutie, XMem/Track-Anything, HOIST, VISOR-HOS, envoltorio nvcc. |
| Salidas | `salidas/<tecnologia>/<stem_del_clip>/` más `salidas/comparativa/`. |
| Lado de la mano | Izquierda y derecha son las **de la persona**, no de la cámara. |
| VISOR | El script existe; **no entra en la comparativa de calidad** (entrenado en egocéntrico). |
| GPU ajena | No matar procesos CUDA de otros usuarios. |

Punto de entrada mínimo para un backend nuevo (ver [Añadir un backend](#añadir-un-backend)):

```python
from pipeline_manos import (
    anadir_argumentos_comunes,
    cargar_detectores,
    comprobar_entorno,
    procesar_video,
    resolver_videos,
    seguir_por_deteccion,  # redetectar cada frame; o pasar un `seguir` propio
)
```

---

## Qué hace el sistema

1. Localiza muñecas con **YOLO26n-pose**.
2. Recorta una ventana alrededor de cada palma (`--padding-muneca`, `--crop-min`).
3. Dentro de ese recorte busca el objeto (YOLO26n-seg + MobileSAM opcional).
4. Puntúa agarre contra las dos manos (`izquierda` / `derecha` / `ambas`).
5. Detecta identificadores de escena en el **frame completo** según `identifiers.yaml` (YOLO cerrado + YOLOE open-vocab).
6. El backend **sigue o redetecta** la máscara del objeto. Si las muñecas no superan `hands.min_confidence`, ese frame no ejecuta el modelo de objeto.
7. El último frame visible decide si el objeto acaba dentro de un contenedor (`ends_in_container`, `final_container`).
8. Escribe vídeo anotado, `tracking.json`, `summary.txt`, `metrics.json` y una fila en la comparativa. Si nunca hay muñecas, `final_state` es `no_hands`.

Cutie y Track-Anything **propagan** la semilla. YOLO26-seg, SAM 3 y HOIST **redetectan** cada frame (`seguir_por_deteccion`). VISOR no usa este pipeline: corre su propio HOS + Faster R-CNN.

---

## Mapa del repositorio

```
ROI_Segmentation-/
├── pipeline_manos.py              # Hub: detección, YAML, muestreo --fps, JSON, vídeo
├── salida_comun.py                # Carpetas, Medidor RAM/VRAM, comparativa.csv/.md/.json
├── identifiers.yaml               # Catálogo de escena + hands.min_confidence (inglés)
├── identificadores.yaml           # Copia de compatibilidad para --identificadores antiguo
├── detector_palma/                # Pose, recorte, YOLO-seg, MobileSAM, score de agarre
│   ├── personas.py                # AnalizadorEscena (YOLO-pose), dibujar_pose cintura↑
│   ├── deteccion.py               # DetectorUltralytics, GeneradorMascarasSAM
│   ├── agarre.py                  # puntuar_agarre, anotar_agarre, es_voluminoso
│   ├── geometria.py               # cajas, IoU, centroides
│   ├── tipos.py                   # Instancia
│   └── utils.py                   # importaciones perezosas
├── yolo26_mano_objeto.py          # Backend: redetección YOLO (el más barato)
├── cutie_mano_objeto.py           # Backend: Cutie (propagación)
├── track_anything_mano_objeto.py  # Backend: XMem
├── sam3_mano_objeto.py            # Backend: pipeline común + utilidades SAM 3 / VISOR
├── hoist_mano_objeto.py           # Backend: pipeline común (HOIST ya no sustituye detección)
├── visor_mano_objeto.py           # Backend aparte (EPIC-KITCHENS HOS); no comparativa de calidad
├── requirements.txt               # Base (SAM 3, ultralytics, opencv, pyyaml, …)
├── requirements-cutie.txt
├── requirements-track-anything.txt
├── requirements-hoist.txt
├── Conclusiones.md                # Ranking empírico de las seis tecnologías
├── scripts.txt                    # Recetas de lanzamiento usadas en el laboratorio
├── pesos/                         # Checkpoints YOLO / YOLOE / MobileSAM (no en git)
├── videos/                        # Clips de entrada (no en git)
├── salidas/                       # Resultados (no en git)
└── third_party/                   # Clones y operadores CUDA (no en git)
```

`sam3_mano_objeto.py` sigue exportando utilidades que VISOR (y restos de HOIST) importan: `info_video`, `anotar_video`, `linea_temporal`, `comprobar_entorno`, etc. El seguimiento del objeto en SAM 3 **ya no** usa el prompt de texto de SAM 3 a distancia de cámara: la máscara sale del recorte de muñeca del pipeline común.

---

## Arquitectura

```
vídeo --leer_frames(--fps)--> frames BGR + indices_origen
        |
        v
cargar_detectores(args)
  AnalizadorEscena (yolo26n-pose)
  DetectorUltralytics (yolo26n-seg)
  GeneradorMascarasSAM (mobile_sam.pt, si existe y no --sin-sam)
  DetectorIdentificadores (YAML → YOLO COCO + YOLOE)
        |
        v
detectar_objetos()  → contextos por frame + pistas (máscaras semilla)
seguir(frames, pistas, contextos, args)  → list[Objeto]   # lo único que cambia el backend
seguir_contenedores()
descartar_accesorios / quedarse_con_agarrados / descartar_contenedores_en_mano
        |
        v
escribir_salidas() → anotado.mp4, frames/, tracking.json, summary.txt, metrics.json
publicar_comparativa() → salidas/comparativa/
```

Funciones clave de `pipeline_manos.py` para un integrador:

| Función | Papel |
|---|---|
| `anadir_argumentos_comunes(parser)` | CLI compartida (`--videos`, `--video`, `--fps`, YAML, umbrales…). |
| `resolver_videos(args)` | Un archivo (`--video`) o todos los de `--videos`. |
| `cargar_catalogo(ruta)` / `Identificador` | Lectura de `identificadores.yaml`. |
| `cargar_detectores(args)` | Devuelve `(escena, detector, sam, identificadores)`. |
| `leer_frames(video, max_frames, fps_proceso)` | `(frames, fps, indices_origen)`. |
| `offsets_muestreo` / `describir_muestreo` / `frame_origen` | Muestreo `--fps` y remap a índices del vídeo original. |
| `detectar_objetos(frames, escena, detector, sam, args)` | Pistas semilla. |
| `seguir_por_deteccion(...)` | Backend nulo: la máscara de cada frame es la del detector. |
| `procesar_video(detectores, video, args, tecnologia, backend, modelo, seguir, extra=None)` | Orquesta un clip. |
| `escribir_salidas(...)` | JSON + vídeo + comparativa. |
| `Objeto`, `Vista` | Tipos de la traza. |

`salida_comun.py`:

- `carpeta_de(raiz, tecnologia, video)` → `salidas/<tecnologia>/<stem>/`. Si `raiz.name` ya es la tecnología, no duplica el nivel.
- `Medidor(backend)` con `with medidor.etapa("segmentacion"):` y `medidor.cerrar(nframes)`.
- `publicar_comparativa(...)` reescribe **solo** la fila `(clip, backend)` y deja el resto.

Callback `seguir` que debe implementar un backend de propagación:

```python
def seguir(frames: list[np.ndarray], pistas: list[dict], contextos, args) -> list[Objeto]:
    ...
```

Cada pista tiene `id`, `primer_frame`, `ultimo_frame`, `mascaras: dict[int, np.ndarray]`, `agarres`. Tras propagar, llamar a `trazar(objeto, contextos, pista, args)` para rellenar `objeto.vistas`. Cutie fusiona identidades con `fusionar_objetos`.

---

## Entorno

### Hardware y runtime comprobados

- Python 3.12 en `.venv`
- PyTorch 2.10.0+cu128, torchvision 0.25.0
- CUDA runtime 12.8 (PyTorch). El `nvcc` del sistema puede ser 13.2; ver HOIST.
- GPU de laboratorio: RTX 5090. YOLO26 cabe en GPU modesta (~200 MB VRAM). Cutie ~1.6 GB. XMem / SAM 3 / HOIST pueden pedir 12–16 GB.

`comprobar_entorno()` (en `sam3_mano_objeto.py`, reexportado por el pipeline) exige Python ≥ 3.11, torch ≥ 2.0 y CUDA ≥ 12.

### Crear el venv (base, suficiente para YOLO26 y el pipeline)

Desde la raíz del repo:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

PyTorch **no** va en `requirements.txt`: si se resuelve junto al índice CUDA, `uv` no encuentra `iopath` (dependencia de SAM 3). Orden: torch primero, luego el resto.

`requirements.txt` instala, entre otros:

- `sam3` desde `git+https://github.com/facebookresearch/sam3.git` (la primera vez clona facebookresearch/sam3)
- `numpy>=1.26,<2`
- `opencv-python-headless>=4.10`
- `ultralytics` (YOLO26-pose, YOLO26-seg, YOLOE, MobileSAM)
- `pyyaml`, `Pillow`, `einops`, `pycocotools`, `psutil`, `tqdm`

Paquetes extra por backend: `requirements-cutie.txt`, `requirements-track-anything.txt`, `requirements-hoist.txt`.

### Pesos en `pesos/`

Ultralytics descarga solo el que falte la primera vez que se instancia el modelo, **si** la ruta por defecto coincide con un nombre conocido. Este repo apunta a archivos locales; hay que tenerlos en `pesos/`:

| Archivo | Uso | Origen típico |
|---|---|---|
| `yolo26n-pose.pt` | Muñecas / esqueleto | Ultralytics YOLO26-pose |
| `yolo26n-seg.pt` | Objeto en el recorte + clases COCO del YAML | Ultralytics YOLO26-seg |
| `yoloe-26m-seg.pt` | Identificadores open-vocab (carrito, cesto, bolsa) | Ultralytics YOLOE |
| `mobileclip2_b.ts` | Encoder de texto de YOLOE (junto a `yoloe-26m-seg.pt`) | Lo trae Ultralytics al cargar YOLOE |
| `mobile_sam.pt` | Máscaras extra en la palma | Ultralytics MobileSAM |

Si falta `mobile_sam.pt`, el pipeline sigue con YOLO-seg y escribe `Sin MobileSAM`. Si falta `yoloe-26m-seg.pt`, solo se detectan las clases COCO del YAML. Si faltan pose o seg, el proceso aborta.

SAM 3.1 (Hugging Face `facebook/sam3.1`) solo hace falta si se usa el paquete `sam3` / APIs que aún lo cargan. Hay que aceptar la licencia y `hf auth login` (token de lectura). El script `sam3_mano_objeto.py` **como experimento de objeto-en-mano** usa el pipeline común, no el prompt de texto a distancia.

---

## Tecnologías

Orden empírico en `Conclusiones.md` (43 clips): **YOLO26-seg > Cutie > Track-Anything > HOIST-Former > SAM 3 > VISOR**. VISOR quedó a cobertura 0 en tercera persona.

### YOLO26-seg — `yolo26_mano_objeto.py`

Sin propagador. Cada frame se redetecta en el recorte de muñeca. Es el backend de integración más simple y el de menos VRAM.

```bash
source .venv/bin/activate
python yolo26_mano_objeto.py --videos videos --salida salidas
python yolo26_mano_objeto.py --video videos/3_clip_3.mp4 --salida salidas
```

Salida: `salidas/yolo26/<clip>/`. Fila comparativa: `YOLO26-seg`.

### Cutie — `cutie_mano_objeto.py`

No detecta: propaga la máscara semilla. `--salto-max` evita saltos a estanterías; `--frames-sin-mano` corta cuando el objeto lleva mucho tiempo sin mano.

```bash
source .venv/bin/activate
uv pip install -r requirements-cutie.txt
git clone --depth 1 https://github.com/hkchengrex/Cutie.git third_party/Cutie
uv pip install --no-deps -e third_party/Cutie
python cutie_mano_objeto.py --videos videos --salida salidas
```

La primera ejecución descarga `cutie-base-mega.pth` en `third_party/Cutie/weights/`. Lado interno por defecto 720 px (`--resolucion-interna`; el paper usa 480). Extra CLI: `--repo`, `--hueco-resembra` (3), `--salto-max` (120), `--frames-sin-mano` (25).

Salida: `salidas/cutie/<clip>/`. Fila: `Cutie`.

### Track-Anything (XMem) — `track_anything_mano_objeto.py`

Igual que Cutie en detección. XMem resiembra a los `--hueco-resembra` frames (8) si la máscara se vacía, sin borrar la memoria larga.

```bash
source .venv/bin/activate
uv pip install -r requirements-track-anything.txt
git clone --depth 1 https://github.com/gaomingqi/Track-Anything.git third_party/Track-Anything
mkdir -p third_party/Track-Anything/checkpoints
# Colocar XMem-s012.pth desde https://github.com/hkchengrex/XMem/releases
python track_anything_mano_objeto.py --videos videos --salida salidas
```

`--resolucion-interna` por defecto 1080; `480` es el yaml oficial. Extra CLI: `--repo`, `--pesos`, `--hueco-resembra`, `--salto-max`.

Salida: `salidas/track_anything/<clip>/`. Fila: `Track-Anything`.

### SAM 3 — `sam3_mano_objeto.py`

Usa el pipeline común (`seguir_por_deteccion`). Umbrales de agarre un poco más estrictos por defecto (`min_detecciones=3`, `contacto_min=0.12`, `area_max_persona=0.02`, …) para no pegar bolsos de hombro u objetos de balda al objeto de la palma.

```bash
source .venv/bin/activate
hf auth login   # solo si se va a descargar facebook/sam3.1
python sam3_mano_objeto.py --videos videos --salida salidas
```

Extra CLI: `--version {sam3.1,sam3}`, `--lote`. Salida: `salidas/sam3/<clip>/`. Fila: `SAM 3`.

### HOIST-Former — `hoist_mano_objeto.py`

El script oficial pedía Python 3.8 / PyTorch 1.9. Aquí corre con el PyTorch 2.10 + CUDA 12.8 del `.venv`. **La detección de objeto-en-mano es la del pipeline común**; HOIST ya no sustituye muñecas ni contenedores.

Compilar detectron2 y el operador Deformable Attention es lo más frágil. En esta máquina `nvcc` 12.8 no enlaza contra la glibc actual y `nvcc` 13.2 lo rechaza PyTorch por versión mayor. `third_party/nvcc-wrap` informa 12.8 y llama al 13.2:

```bash
source .venv/bin/activate
uv pip install -r requirements-hoist.txt
git clone --depth 1 https://github.com/SupreethN/HOISTFormer.git third_party/HOISTFormer
export CUDA_HOME="$PWD/third_party/nvcc-wrap"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
cd third_party/HOISTFormer/mask2former/modeling/pixel_decoder/ops
FORCE_CUDA=1 python setup.py build install
cd ../../../../../..
mkdir -p third_party/HOISTFormer/pretrained_models
gdown 1yLXCqt8mQroqQYDnncMvJ3HaOKMln8Xs -O third_party/HOISTFormer/pretrained_models/trained_model.pth
python hoist_mano_objeto.py --videos videos --salida salidas
```

Si el operador CUDA no está, `asegurar_fallback_cuda` parchea HOIST para caer a la implementación PyTorch. Extra CLI: `--repo`, `--pesos`.

Salida: `salidas/hoist/<clip>/`. Fila: `HOIST-Former`.

### VISOR-HOS — `visor_mano_objeto.py`

Pipeline **propio**: Faster R-CNN (persona) + PointRend HOS (mano, contacto, lado, objeto). Entrenado en EPIC-KITCHENS egocéntrico; en tercera persona suele no pasar el umbral 0.5. Comparte `--fps`, `--video` y el layout de salida, no `identificadores.yaml`.

```bash
source .venv/bin/activate
git clone --depth 1 https://github.com/epic-kitchens/VISOR-HOS.git third_party/VISOR-HOS
mkdir -p third_party/VISOR-HOS/checkpoints
# model_final_hos.pth, enlace del README de VISOR-HOS
# detectron2: el mismo que HOIST
python visor_mano_objeto.py --videos videos --salida salidas
```

Salida: `salidas/visor/<clip>/`. Fila: `VISOR`.

---

## CLI común (pipeline)

Todos salvo VISOR llaman a `anadir_argumentos_comunes`. VISOR replica `--videos`, `--video`, `--salida`, `--fps`, `--max-frames`.

### Entrada y salida

| Flag | Default | Significado |
|---|---|---|
| `--videos` | `videos` | Carpeta de clips. |
| `--video` | ninguno | Un solo archivo; ignora `--videos`. |
| `--salida` | `salidas` | Raíz de resultados. |
| `--identificadores` | `identifiers.yaml` | Catálogo de escena + `hands.min_confidence`. |
| `--mediapipe` | off | Esqueleto MediaPipe (Pose + Hands) en lugar de YOLO-pose. |
| `--fps N` | omitido = frame a frame | Ver [Muestreo `--fps`](#muestreo---fps). |
| `--max-frames` | ninguno | Corta la lectura del vídeo de origen (antes del muestreo). |
| `--sin-sam` | off | No usa MobileSAM. |
| `--sin-contenedores` | off | Ignora el YAML. |
| `--sin-frames` | off | No vuelca JPEG sueltos. |
| `--sin-pose` | off | No pinta el esqueleto (cintura, brazos, muñecas, palmas; sin piernas ni abanico de dedos ni textos). |
| `--mediapipe` | off | MediaPipe Pose + Hands instead of YOLO-pose. More reliable empty-hand rejection. Wrist visibility uses `hands.min_confidence` (default 0.20). |

### Detección de palma

| Flag | Default | Significado |
|---|---|---|
| `--modelo-pose` | `pesos/yolo26n-pose.pt` | |
| `--modelo-objeto` | `pesos/yolo26n-seg.pt` | |
| `--padding-muneca` | `1.2` | Semilado del recorte = 1.2 × radio del brazo. |
| `--crop-min` | `96` | Semilado mínimo en píxeles. |
| `--factor-palma` | `0.85` | Radio de **aceptación**: el objeto tiene que estar pegado a la palma, no basta con caer en el recorte. |
| `--conf-pose` / `--conf-objeto` / `--conf-objeto-crop` | 0.25 / 0.25 / 0.12 | |
| `--imgsz` | 640 | |
| `--area-min` / `--area-max-px` / `--area-max-persona` | 80 / 6000 / 0.05 | Filtro de tamaño (evita brazos, baldas, carritos como “objeto en mano”). |
| `--relacion-max` / `--relleno-min` | 3.0 / 0.3 | Forma compacta. |
| `--umbral-agarre` / `--contacto-min` | 0.35 / 0.10 | Score y contacto palma/dedos. |
| `--min-detecciones` | 2 | Pistas más cortas se tiran. |
| `--hueco-max` / `--distancia-max` | 45 / 180 | Asociación temporal de detecciones. |
| `--frames-traspaso` / `--hueco-parpadeo` | 3 / 3 | Eventos de cambio de mano y huecos. |

### Identificadores de escena

| Flag | Default | Significado |
|---|---|---|
| `--modelo-contenedores` | `pesos/yoloe-26m-seg.pt` | |
| `--conf-contenedor` / `--imgsz-contenedor` | 0.2 / 960 | |
| `--cada-contenedor` | 5 | Inferir 1 de cada N frames procesados. |
| `--dentro-min` | 0.6 | Fracción del objeto dentro del ROI destino. |
| `--factor-contenedor` | 2.5 | Tamaño relativo objeto vs contenedor. |
| `--frames-fin` | 5 | Últimos frames visibles para `contenedor_final`. |

Prueba corta recomendada:

```bash
.venv/bin/python yolo26_mano_objeto.py \
  --video videos/3_clip_3.mp4 --fps 2 --max-frames 30 \
  --sin-frames --sin-contenedores --sin-sam --salida salidas
```

---

## Muestreo `--fps`

`--fps N` pide **N frames por cada segundo de vídeo**, no un stride uniforme.

`rate = round(fps_del_video)`. Offsets dentro de cada segundo: `{int(i * rate / N) for i in range(N)}` (0-index). Ejemplos con vídeo a **15 fps**:

| Parámetro | Offsets 0-index | Frames 1-index |
|---|---|---|
| `--fps 1` | 0 | 1 |
| `--fps 2` | 0, 7 | 1 y 8 |
| `--fps 3` | 0, 5, 10 | 1, 6 y 11 |

Si `--fps` se omite, o si `N >= rate`, el procesamiento es **frame a frame**.

El seguimiento interno usa índices 0…M−1 de los frames **procesados**. JSON, HUD y nombres `frames/frame_XXXXX.jpg` se remapean al índice **original** del archivo (`t = orig / fps_video`). El `anotado.mp4` se escribe a `N` fps cuando hay muestreo, para conservar la duración.

Consola, `summary.txt` y `tracking.json["sampling"]` declaran siempre una de estas tres:

- `--fps was not set. Processing ran frame by frame.`
- `--fps N was set. The video is … (rate R): each second uses frames …`
- `--fps N was set, equal to or greater than the video rate (R fps). Processing ran frame by frame.`

`--max-frames` limita frames de **origen** leídos, y después se aplica el muestreo.

---

## Scene identifiers (`identifiers.yaml`)

Immediate detection on the **full frame**. Keys and comments are in English. Category ids (`carrito`, `cesto`, …) stay stable so `cesto_2` does not change.

```yaml
hands:
  min_confidence: 0.20

identifiers:
  celular:
    active: true
    ultralytics_id: 67
    name: cell phone
    destination: false
    accessory: false
    prompts:
      - cell phone
      - mobile phone
```

| Field | Effect |
|---|---|
| `active` | `false` disables the class. |
| `ultralytics_id` | COCO shortcut. The detector resolves by `name`. |
| `name` | Ultralytics English name. |
| `prompts` | YOLOE text. Required if the class is not in COCO. |
| `destination` | The handheld object may end inside this ROI (green). |
| `accessory` | Shoulder bag / backpack: not merged with the grasped object. |
| `hands.min_confidence` | Below this, wrists/hands are treated as not visible and the object model is skipped. |

Ids Ultralytics COCO (contiguos 0–79), no los ids “paper” de COCO: `24` backpack, `26` handbag, `66` **keyboard**, `67` **cell phone**. El móvil es **67**. Carrito, cesto y bolsa no están en COCO: solo YOLOE.

Por defecto en este repo: carrito, cesto, bolso, mochila, bolsa y celular activos. Destinos en verde; el resto (p. ej. celular) en cian.

---

## Salidas

Por clip y tecnología, en `salidas/<tecnologia>/<stem>/`:

| Archivo | Contenido |
|---|---|
| `anotado.mp4` | Máscaras de objeto, caja + etiqueta de mano (`IZQ`/`DER`/`AMBAS`), ROI de identificadores, esqueleto cintura↑, HUD `frame <origen> t=…`. |
| `frames/frame_XXXXX.jpg` | Mismo overlay; `XXXXX` es el índice original. `--sin-frames` lo omite. |
| `tracking.json` | Full trace (schema below). |
| `summary.txt` | Same events in English, plus the `--fps` sentence. |
| `metrics.json` | `processed_frames`, `mean_fps`, `ms_per_frame`, RAM/VRAM peaks, CPU, `stages`. |

### `tracking.json` (shared pipeline)

Header: `video`, `model`, `width`, `height`, `frames`, `source_frames`, `fps`, `output_fps`, `sampling`, `pose_backend`, `hands_visible`, `final_state`, `parameters`, `identifiers_yaml`.

`sampling`:

```json
{
  "specified": true,
  "fps_parameter": 2,
  "video_fps": 15.0,
  "integer_rate": 15,
  "source_frames": 150,
  "processed_frames": 20,
  "offsets_in_second": [0, 7],
  "description": "--fps 2 was set. …"
}
```

`containers[]`: `id`, `category`, `prompt`, scores, `first_frame` / `last_frame` (source), `frames_seen`, `destination`, `accessory`, `last_roi`, `rois` frame→`[x1,y1,x2,y2]`.

`objects[]`:

- `id`, `frames_visible`, `frames_in_hand`, `first_frame`, `last_frame`, `reseeds`
- `hands`: `{left, right, both, no_hand}`
- `dominant_hand`
- `last_roi`, `last_center`, `last_visible_frame`
- `final_state`: `in_hand` \| `released` \| `disappeared` \| `inside_container` \| `no_data` \| `no_hands`
- `ends_in_container` (bool)
- `final_container`: `{id, category, roi, fraction_inside, center_inside}` or `null`
- `events[]`: `appears`, `disappears`, `reappears`, `releases`, `regrasps`, `hand_change` (`from`/`to`)
- `timeline[]`: `frame`, `processed_frame`, `t`, `hand`, `roi`, `center`, `area`, `grip_score`, `hand_scores`, `origin` (`detection` \| `propagation`)

VISOR escribe un JSON distinto (`eventos`, `linea_temporal` estilo SAM 3 original) más el bloque `muestreo`.

### Comparativa

`salidas/comparativa/comparativa.csv` (también `.md` y `.json`). Una fila por `(clip, backend)`. Columnas: `clip`, `backend`, `ok`, `frames`, `fps`, `ms_frame`, `ms_segmentacion`, `ram_pico_mb`, `vram_pico_mb`, `cpu_pct`, `objetos`, `desapariciones`, `frames_con_objeto`, `codigo_salida`, `error`, `carpeta`.

Cada pasada sustituye su fila y **no borra** las demás. Columnas añadidas a mano en el CSV se heredan.

Backends escritos hoy: `YOLO26-seg`, `Cutie`, `Track-Anything`, `HOIST-Former`, `SAM 3`, `VISOR`.

---

## Añadir un backend

1. Crear `mi_backend_mano_objeto.py` en la raíz.
2. `anadir_argumentos_comunes(parser)` y, si hace falta, flags propios (`--repo`, `--resolucion-interna`, …).
3. `comprobar_entorno()` + `cargar_detectores(args)`.
4. Definir `seguir(frames, pistas, contextos, args) -> list[Objeto]`:
   - Redetección: reutilizar `seguir_por_deteccion`.
   - Propagación: sembrar con `pista["mascaras"][pista["primer_frame"]]`, rellenar `objeto.mascaras`, llamar `trazar`, opcionalmente `fusionar_objetos`.
5. Por cada vídeo de `resolver_videos(args)`:

```python
procesar_video(
    detectores, video, args,
    tecnologia="mi_backend",   # carpeta salidas/mi_backend/
    backend="MiBackend",       # columna comparativa
    modelo="MiBackend …",      # campo JSON / resumen
    seguir=seguir,
    extra={"propagator": "…"}, # merged into tracking.json
)
```

6. Do not change the `objects[]` / `containers[]` schema if a consumer already parses `tracking.json`.
7. Documentar pesos y `third_party/` en este README.

Para un servicio que solo quiera ROI final + contenedor, leer `tracking.json` → `objects[i].last_roi`, `ends_in_container`, `final_container`. No hace falta el vídeo anotado (`--sin-frames`).

Importar detección suelta (sin `procesar_video`):

```python
from pipeline_manos import cargar_detectores, detectar_objetos, leer_frames

frames, fps, origen = leer_frames(ruta, max_frames=None, fps_proceso=2)
escena, detector, sam, _ids = cargar_detectores(args)
contextos, pistas = detectar_objetos(frames, escena, detector, sam, args)
```

`detector_palma` es usable por su cuenta (`AnalizadorEscena`, `DetectorUltralytics`, `puntuar_agarre`) si otro sistema ya tiene su propio tracker.

---

## Recetas de laboratorio

`scripts.txt` guarda comandos reales. Patrón habitual: `--videos videos --identificadores identificadores.yaml --salida salidas`. Para un clip:

```bash
.venv/bin/python sam3_mano_objeto.py \
  --video videos/3_clip_2.mp4 \
  --identificadores identificadores.yaml \
  --salida salidas
```

---

## Vídeos de prueba (locales, no en git)

- `videos/clip.mp4`: objeto en la mano izquierda y traspaso a la derecha.
- `videos/clip_2.mp4`: misma tarea, segundo clip.
- En el laboratorio hay series `N_clip_M.mp4` (p. ej. `3_clip_3.mp4`) usadas en la comparativa de 43 clips.

---

## Problemas frecuentes

- **`python` no es el venv.** Pose26 / YOLO26 fallan. Usar `.venv/bin/python` o `source .venv/bin/activate`.
- **Celular id 66.** En Ultralytics 66 es `keyboard`. Usar `nombre: cell phone` (id 67). El código resuelve por nombre.
- **Ruido en SAM 3 / recorte grande.** El recorte de muñeca es amplio (`padding-muneca` 1.2); la aceptación es `factor-palma` 0.85 sobre el radio del brazo. No ampliar el factor sin refiltrar agarre.
- **Objeto fusionado con bolso de hombro.** `accesorio: true` en el YAML y `descartar_accesorios` en el pipeline.
- **HOIST no compila.** Hace falta el truco `third_party/nvcc-wrap` o el fallback PyTorch de atención deformable.
- **Cutie `No está el código`.** Falta el clone en `third_party/Cutie`.
- **YOLOE silencioso.** Falta `pesos/yoloe-26m-seg.pt`; carrito/cesto/bolsa no saldrán, bolso/mochila/celular sí si están en COCO.
- **`--fps` y clips que no van a 15 fps.** Los ejemplos 1/8 y 1/6/11 son para rate 15. Un clip a 12.333 fps redondea a 12; `--fps 2` toma offsets 0 y 6 (frames 1 y 7).
- **Duración del anotado.** Con `--fps N` el writer usa N fps. Sin `--fps`, usa el fps del archivo.

100DOH y detectores mano-objeto de esa generación no entran: piden CUDA/PyTorch anteriores a los de esta máquina.
