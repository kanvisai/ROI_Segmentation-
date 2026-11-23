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
| Vídeos | Carpeta `videos/` (gitignored). Extensiones: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. El mp4 es **opcional** si se pasa `list[np.ndarray]` (`frames=` / `args.frames`). |
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
    procesar_lote,          # varios clips o frames=list[np.ndarray]
    procesar_video,         # un clip (mp4 o arrays)
    resolver_videos,
    seguir_por_deteccion,   # redetectar cada frame; o pasar un `seguir` propio
)
```

---

## Qué hace el sistema

1. Localiza personas con **YOLO26n-pose** y **ancla al sujeto de la sesión** (alcance a la balda + tamaño). En el resto de frames se le sigue por IoU / muñecas / color; las demás detecciones no entran a SAM. MediaPipe Holistic, con `--mediapipe`, solo corre en ese recorte. Si se pierde el sujeto, ese frame queda vacío (no se salta a un transeúnte). `--sin-ancla-persona` desactiva el filtro.
2. Recorta una ventana alrededor de cada palma (`--padding-muneca`, `--crop-min`). Si el brazo alcanza una estantería (palma fuera del bbox), el recorte se aprieta para no comerse la balda.
3. Dentro de ese recorte busca el objeto (YOLO26n-seg + MobileSAM opcional). No se busca en una palma abierta que cuelga al costado. Un puño que vuelve al torso sí se busca.
4. Puntúa agarre contra las dos manos (`izquierda` / `derecha` / `ambas`). Dos recortes de la misma pieza se fusionan en un solo objeto. Un bulto de ropa en la cadera **no** se fusiona con un agarre posterior de la otra mano.
5. Detecta identificadores de escena en el **frame completo** según `identifiers.yaml` (YOLO cerrado + YOLOE open-vocab). El bolsillo no sale de YOLO: se sintetiza con las caderas de la pose (`body: true`).
6. El backend **sigue o redetecta** la máscara del objeto. Si las muñecas no superan `hands.min_confidence` (0.50) o, con `--mediapipe`, los **dedos** no llegan a `hands.min_hand_confidence` (0.30), esa palma no ejecuta SAM. Un salto L↔R entre frames consecutivos no se asocia al mismo objeto.
7. El último frame visible decide el depósito y el `final_state` (ver [Estado final](#estado-final-del-objeto)). Mueble primero; bolsillo solo con gesto; bolso/mochila no son destino de reserva.
8. Escribe vídeo anotado, `tracking.json`, `summary.txt`, `metrics.json` y una fila en la comparativa. Si nunca hay muñecas, `final_state` es `no_hands`. Un clip que falle no aborta el lote (`procesar_lote`).

Cutie y Track-Anything **propagan** la semilla. YOLO26-seg, SAM 3 y HOIST **redetectan** cada frame (`seguir_por_deteccion`). VISOR no usa este pipeline: corre su propio HOS + Faster R-CNN.

---

## Mapa del repositorio

```
ROI_Segmentation-/
├── pipeline_manos.py              # Hub: detección, YAML, muestreo --fps, JSON, vídeo
├── salida_comun.py                # Carpetas, Medidor RAM/VRAM, comparativa.csv/.md/.json
├── identifiers.yaml               # Catálogo de escena + hands.min_confidence (claves YAML en inglés)
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
├── scripts/                       # build.sh, deploy.sh, smoke_*.sh, run_local_offline.sh
├── kanvis_segmentation/           # Servicio gRPC standalone (puerto 50100)
├── proto/kanvis/segmentation/v1/  # segmentation.proto
├── docker-compose.segmentation.yml
├── fixtures/session_001/          # Layout local tipo MinIO para smoke offline
├── pesos/                         # Checkpoints YOLO / YOLOE / MobileSAM (no en git)
├── videos/                        # Clips de entrada (no en git)
├── fp_agosto2026/                 # Clips locales de depuración (no en git; ver .gitignore)
├── salidas/                       # Resultados (no en git)
└── third_party/                   # Clones y operadores CUDA (no en git)
```

`sam3_mano_objeto.py` sigue exportando utilidades que VISOR (y restos de HOIST) importan: `info_video`, `anotar_video`, `linea_temporal`, `comprobar_entorno`, etc. El seguimiento del objeto en SAM 3 **ya no** usa el prompt de texto de SAM 3 a distancia de cámara: la máscara sale del recorte de muñeca del pipeline común.

---

## Arquitectura

```
mp4 --leer_frames(--fps)--> frames BGR + indices_origen
  o
list[np.ndarray] + frames_origin + fps/width/height (enteros)
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
  # AnalizadorEscena ancla 1 persona; palma vacía / ropa / balda se tiran aquí
seguir(...) o seguir_por_deteccion()  → list[Objeto]   # lo único que cambia el backend
seguir_contenedores()                 # YOLO/YOLOE + bolsillo de pose; cose/ancla accesorios
recortar saltos a bolsa o cadera / sembrar celular junto a la palma
completar_deposito()                  # sigue el objeto si la mano entra en un cesto
extender_si_sigue_en_mano / rellenar_huecos_en_mano
quedarse_con_agarrados                # tira bultos estáticos de cadera; se reejecuta tras fusionar
fusionar_objetos                      # no pega un bulto de cadera con un agarre de la otra mano
descartar_falsos_bolsillo             # si hay cesto, no se deja un fantasma en el bolsillo
        |
        v
escribir_salidas() → anotado.mp4, frames/, tracking.json, summary.txt, metrics.json
publicar_comparativa() → salidas/comparativa/
```

Funciones clave de `pipeline_manos.py` para un integrador:

| Función | Papel |
|---|---|
| `anadir_argumentos_comunes(parser)` | CLI compartida (`--videos`, `--video`, `--fps`, `--width`, `--height`, `--frames-origin`, YAML…). |
| `resolver_videos(args)` | Un archivo (`--video`), todos los de `--videos`, o un stem ficticio si hay `args.frames`. |
| `preparar_frames(frames, frames_origin, fps, width, height)` | Valida `list[np.ndarray]` BGR, índices enteros y `fps`/`width`/`height` enteros. |
| `cargar_catalogo(ruta)` / `Identificador` | Lectura de `identifiers.yaml` (acepta claves en inglés o español). |
| `cargar_detectores(args)` | Devuelve `(escena, detector, sam, identificadores)`. Con `--mediapipe` la escena usa Holistic. |
| `leer_frames(video, max_frames, fps_proceso)` | `(frames, fps, indices_origen)`. |
| `offsets_muestreo` / `describir_muestreo` / `frame_origen` | Muestreo `--fps` y remap a índices del vídeo original. |
| `detectar_objetos(frames, escena, detector, sam, args)` | Pistas semilla. |
| `seguir_por_deteccion(...)` | Backend nulo: la máscara de cada frame es la del detector. |
| `fusionar_objetos(...)` | Une dos pistas de la misma pieza (p. ej. las dos manos sobre una botella). No fusiona un bulto en la cadera con un agarre posterior. |
| `quedarse_con_agarrados(...)` | Se queda con el objeto principal; descarta palmas vacías y bultos de ropa en la cadera. |
| `estado_final(...)` | `in_hand` / `disappeared` / `inside_container` / `released` / `no_data`. El clip puede ser `no_hands` si no hay muñecas. |
| `contenedor_final(...)` | Destino del último frame: mueble primero, bolsillo solo con gesto. |
| `procesar_video(...)` | Orquesta un clip (mp4 o arrays). Kwargs: `frames`, `frames_origin`, `fps`, `width`, `height`. |
| `procesar_lote(...)` | Varios clips o una lista de arrays; un fallo no aborta el resto. Devuelve cuántos fallaron. |
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

Cada pista tiene `id`, `primer_frame`, `ultimo_frame`, `mascaras: dict[int, np.ndarray]`, `agarres`. Tras propagar, llamar a `trazar(objeto, contextos, pista, args)` para rellenar `objeto.vistas`. Cutie y el pipeline común fusionan identidades con `fusionar_objetos`.

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

Usa el pipeline común (`seguir_por_deteccion`). Umbrales de agarre un poco más estrictos por defecto (`min_detecciones=3`, `contacto_min=0.12`, `area_max_persona=0.02`, `distancia_max=280`, …) para no pegar bolsos de hombro u objetos de balda al objeto de la palma. Receta de laboratorio: `--identificadores identifiers.yaml --mediapipe --fps 3`. Si un clip es más rápido (traspaso corto, depósito en cesto), `--fps half` o `--fps third`.

```bash
source .venv/bin/activate
hf auth login   # solo si se va a descargar facebook/sam3.1
python sam3_mano_objeto.py --videos videos --identificadores identifiers.yaml --mediapipe --fps 3 --salida salidas
python sam3_mano_objeto.py --video videos/6_clip_2.mp4 --identificadores identifiers.yaml --mediapipe --fps 3 --salida salidas
python sam3_mano_objeto.py --video videos/multi_pers_clip_1.mp4 --identificadores identificadores.yaml --mediapipe --fps 3 --salida salidas
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

Pipeline **propio**: Faster R-CNN (persona) + PointRend HOS (mano, contacto, lado, objeto). Entrenado en EPIC-KITCHENS egocéntrico; en tercera persona suele no pasar el umbral 0.5. Comparte `--fps`, `--video` y el layout de salida, no `identifiers.yaml`.

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
| `--video` | ninguno | Un solo mp4 (opcional si hay `--carpeta-frames` o `frames`). Con arrays o carpeta de JPEG solo nombra la carpeta de salida. Si se pasa varias veces, argparse se queda con el **último**. |
| `--carpeta-frames` | ninguno | Carpeta de JPEG/PNG. Sustituye al mp4 (todas las imágenes, sin muestreo). Hace falta `--fps` entero. El número del nombre (`frame_000.jpg` → 0) es el índice de origen. Si la ruta no tiene imágenes pero sí subcarpetas con JPEG (p. ej. `videos/`), se procesa cada una. Lo usan SAM 3, YOLO26, Cutie, Track-Anything y HOIST. |
| `--subcarpetas-frames` | off | Con `--videos DIR`, procesa también cada subcarpeta que contenga JPEG/PNG (además de los mp4). Hace falta `--fps` entero para esas carpetas. |
| `--frames` | ninguno | **API Python**, no flag de shell: `list[np.ndarray]` BGR (`args.frames` o `frames=`). Si está definida, el mp4 no hace falta. |
| `--frames-origin` | `0…N-1` | `list[int]`: índice de origen de cada array. Misma longitud que `--frames`. |
| `--width` / `--height` | se infieren del array | Enteros ≥ 1. Con arrays son el tamaño; si se pasan deben coincidir. Con mp4, si se pasan deben coincidir con el archivo. |
| `--salida` | `salidas` | Raíz de resultados. |
| `--identificadores` | `identifiers.yaml` | Catálogo de escena + `hands.min_confidence`. |
| `--mediapipe` | off | MediaPipe Holistic (pose + manos) en el recorte de persona, en lugar de YOLO-pose. Rechaza mejor la mano vacía. El umbral de muñecas es `hands.min_confidence` (0.50). |
| `--fps N\|half\|third` | omitido = frame a frame (solo mp4) | Entero, `half` o `third`. Con mp4: muestreo (`half`/`third` = la mitad o un tercio del fps del archivo). Con `--carpeta-frames` / `--frames`: fps entero de la línea temporal (**obligatorio**). Ver [Muestreo `--fps`](#muestreo---fps). |
| `--max-frames` | ninguno | Corta la lectura del vídeo de origen (antes del muestreo). No aplica a arrays en memoria. |
| `--sin-sam` | off | No usa MobileSAM. |
| `--sin-contenedores` | off | Ignora el YAML. |
| `--sin-frames` | off | No vuelca JPEG sueltos. |
| `--sin-pose` | off | No pinta el esqueleto (cintura, brazos, muñecas, palmas; sin piernas ni textos). |
| `--sin-ancla-persona` | off | SAM en todas las personas del frame. Por defecto se sigue a un único sujeto. |
| `--ancla-persona` | `auto` | Quién sembrar: `auto`, `0`/`1`/`2`… (izquierda→derecha), `izquierda`, `derecha`, `centro`, `altura`, `X,Y` (clic) o `x1,y1,x2,y2` (muñecas izq/der, matching C1). |
| `--ancla-listar` | off | Solo YOLO-pose: escribe `ancla_candidatos.jpg` con un número sobre cada persona y sale. Luego `--ancla-persona N`. |
| `--ancla-distancia` | `140` | Distancia máxima (px) entre muñecas para reidentificar al sujeto (`C2_SAM3_WRIST_MATCH_MAX_PX`). |
| `--summary` | off | `tracking.json` compacto: cabecera, eventos, `last_roi` y contenedor final. Sin `timeline[]`, sin `containers[].rois` y sin `frames_origin`. `summary.txt` no cambia. |

### Detección de palma

| Flag | Default | Significado |
|---|---|---|
| `--modelo-pose` | `pesos/yolo26n-pose.pt` | |
| `--modelo-objeto` | `pesos/yolo26n-seg.pt` | |
| `--padding-muneca` | `1.2` | Semilado del recorte = 1.2 × radio del brazo. |
| `--crop-min` | `96` | Semilado mínimo en píxeles. |
| `--factor-palma` | `0.85` | Radio de **aceptación**: el objeto tiene que estar pegado a la palma, no basta con caer en el recorte. |
| `--conf-pose` / `--conf-objeto` / `--conf-objeto-crop` | 0.25 / 0.25 / 0.12 | SAM 3 usa `conf_objeto_crop` 0.15. |
| `--imgsz` | 640 | |
| `--area-min` / `--area-max-px` / `--area-max-persona` | 80 / 6000 / 0.05 | Filtro de tamaño (evita brazos, baldas, carritos como “objeto en mano”). SAM 3 usa 5000 / 0.02. |
| `--relacion-max` / `--relleno-min` | 3.0 / 0.3 | Forma compacta. |
| `--umbral-agarre` / `--contacto-min` | 0.35 / 0.10 | Score y contacto palma/dedos. SAM 3 usa `contacto_min` 0.12. |
| `--min-detecciones` | 2 | Pistas más cortas se tiran. SAM 3 sube a 3. |
| `--hueco-max` / `--distancia-max` | 45 / 180 | Asociación temporal de detecciones. SAM 3 sube `distancia_max` a 280. |
| `--frames-traspaso` / `--hueco-parpadeo` | 3 / 3 | Eventos de cambio de mano y huecos. |

### Identificadores de escena

| Flag | Default | Significado |
|---|---|---|
| `--modelo-contenedores` | `pesos/yoloe-26m-seg.pt` | |
| `--conf-contenedor` / `--imgsz-contenedor` | 0.2 / 960 | |
| `--cada-contenedor` | 5 | Inferir 1 de cada N frames procesados. |
| `--dentro-min` | 0.6 | Fracción del objeto dentro del ROI destino (mueble). |
| `--factor-contenedor` | 2.5 | Tamaño relativo objeto vs contenedor. |
| `--frames-fin` | 5 | Últimos frames visibles para decidir un mueble. El bolsillo no usa este margen: exige gesto. |

Prueba corta recomendada:

```bash
.venv/bin/python yolo26_mano_objeto.py \
  --video videos/3_clip_3.mp4 --fps 2 --max-frames 30 \
  --sin-frames --sin-contenedores --sin-sam --salida salidas
```

---

## Muestreo `--fps`

`--fps` acepta un **entero** ≥ 1 o las palabras **`half`** / **`third`**. El significado depende de la entrada:

| Entrada | `--fps N` / `half` / `third` | Si se omite |
|---|---|---|
| mp4 | **N frames por cada segundo de vídeo**, no un stride uniforme. `half` = `round(fps)/2`, `third` = `round(fps)/3` (un vídeo a 12 fps: `half` ≡ 6, `third` ≡ 4) | Frame a frame |
| `--carpeta-frames` / `list[np.ndarray]` | Fps **entero** de la **línea temporal** (obligatorio). No se vuelve a muestrear. `half`/`third` no valen | Error |

### mp4

`rate = round(fps_del_video)`. Offsets dentro de cada segundo: `{int(i * rate / N) for i in range(N)}` (0-index). Ejemplos con vídeo a **15 fps**:

| Parámetro | Offsets 0-index | Frames 1-index |
|---|---|---|
| `--fps 1` | 0 | 1 |
| `--fps 2` | 0, 7 | 1 y 8 |
| `--fps 3` | 0, 5, 10 | 1, 6 y 11 |
| `--fps half` (rate 12) | ≡ `--fps 6` | 6 muestras/s |
| `--fps third` (rate 12) | ≡ `--fps 4` | 4 muestras/s |

Si `N >= rate`, el procesamiento es **frame a frame**.

El seguimiento interno usa índices 0…M−1 de los frames **procesados**. JSON, HUD y nombres `frames/<video>_frame_XXXXX.jpg` se remapean al índice **original** del archivo (`t = orig / fps_video`). El `anotado.mp4` se escribe a `N` fps cuando hay muestreo, para conservar la duración.

`--max-frames` limita frames de **origen** leídos, y después se aplica el muestreo.

### Carpeta de JPEG (`--carpeta-frames`)

Sustituye al mp4 en SAM 3, YOLO26-seg, Cutie, Track-Anything y HOIST. Se leen todos los `.jpg`/`.jpeg`/`.png` de la carpeta (no se vuelve a muestrear). `--fps` es el reloj `t = origen / fps` (entero; `half`/`third` no valen). El número del nombre es el índice de origen (`frame_012.jpg` → 12). `--video` es opcional y solo nombra `salidas/<tecnologia>/<stem>/`.

```bash
.venv/bin/python sam3_mano_objeto.py \
  --carpeta-frames videos/a_clip_r12_frames_chunk_002 \
  --identificadores identificadores.yaml \
  --mediapipe --fps 12 \
  --salida salidas

# Todas las subcarpetas de frames bajo videos/ (y también los mp4):
.venv/bin/python sam3_mano_objeto.py \
  --videos videos --subcarpetas-frames \
  --identificadores identifiers.yaml --mediapipe --fps 12 \
  --salida salidas

# Solo subcarpetas de frames (sin mp4):
.venv/bin/python sam3_mano_objeto.py \
  --carpeta-frames videos \
  --identificadores identifiers.yaml --mediapipe --fps 12 \
  --salida salidas
```

### Arrays en memoria (`--frames`)

No se puede pasar `list[np.ndarray]` por la shell (usa `--carpeta-frames` si los JPEG ya están en disco). En Python:

```python
from pipeline_manos import (
    anadir_argumentos_comunes, cargar_detectores, procesar_video, seguir_por_deteccion,
)
import argparse

parser = argparse.ArgumentParser()
anadir_argumentos_comunes(parser)
args = parser.parse_args([
    "--salida", "salidas",
    "--fps", "15",           # entero: fps de la línea temporal
    "--width", "1280",       # entero
    "--height", "720",       # entero
    "--frames-origin", "0", "5", "10",
    "--video", "clip_memoria",  # opcional: stem de salidas/<tecnologia>/clip_memoria/
])
args.frames = frames_bgr          # list[np.ndarray], HxWx3 uint8
# equivalente a kwargs:
detectores = cargar_detectores(args)
procesar_video(
    detectores, None, args, "yolo26", "deteccion", "yolo26n-seg",
    seguir_por_deteccion,
    frames=frames_bgr,           # list[np.ndarray]
    frames_origin=[0, 5, 10],    # list[int], misma longitud
    fps=15,                      # int
    width=1280,                  # int
    height=720,                  # int
)
```

`width` y `height` deben ser enteros. Si se omiten, se leen del primer array y se comprueba que todos coincidan. Si se pasan, tienen que coincidir con los arrays (o con el mp4, si la entrada es archivo).

`frames_origin` alinea cada array con el índice original (`t = origen / fps`). Si se omite, se usa `0…N-1`. El `anotado.mp4` usa un `output_fps` que conserva la duración del tramo de origen: `(N-1) * fps / (origen[-1] - origen[0])`. Ejemplo: orígenes `0, 5, 10` y `--fps 15` → `output_fps` 3.

`procesar_lote(..., frames=..., frames_origin=..., fps=..., width=..., height=...)` acepta los mismos kwargs (un solo clip).

Consola, `summary.txt` y `tracking.json["sampling"]` declaran una de estas:

- `--fps was not set. Processing ran frame by frame.`
- `--fps N was set. The video is … (rate R): each second uses frames …`
- `--fps N was set, equal to or greater than the video rate (R fps). Processing ran frame by frame.`
- `--frames was set (N arrays). Timeline fps is F. No mp4 sampling was applied.`

---

## Identificadores de escena (`identifiers.yaml`)

Detección inmediata en el **frame completo**. Los campos del YAML van en inglés (`active`, `destination`, `accessory`, `body`). Las claves de categoría (`carrito`, `cesto`, `bolsillo`, …) se mantienen estables para que un `cesto_2` no cambie de nombre entre versiones.

```yaml
hands:
  min_confidence: 0.50
  min_hand_confidence: 0.30

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

| Campo | Efecto |
|---|---|
| `active` | `false` desactiva la clase. |
| `ultralytics_id` | Atajo COCO. El detector resuelve por `name`. |
| `name` | Nombre inglés de Ultralytics. |
| `prompts` | Texto YOLOE. Obligatorio si la clase no está en COCO. |
| `destination` | El objeto en la mano **puede** acabar dentro de este ROI (verde). |
| `accessory` | Bolso/mochila: se sigue sobre la persona, no se fusiona con el objeto agarrado y **no** es destino de reserva. |
| `body` | ROI a partir de la pose (bolsillo en cadera/muslo). No se detecta con YOLO. |
| `hands.min_confidence` | Por debajo, esa **muñeca** no se busca. Si no hay ninguna, `final_state` es `no_hands`. |
| `hands.min_hand_confidence` | Con `--mediapipe`: confianza de **palma/dedos** (no la muñeca). Por debajo de 0.30 no se lanza SAM en esa mano. |

Ids Ultralytics COCO (contiguos 0–79), no los ids “paper” de COCO: `24` backpack, `26` handbag, `66` **keyboard**, `67` **cell phone**. El móvil es **67**. Carrito, cesto y bolsa no están en COCO: solo YOLOE.

Por defecto en este repo: carrito, cesto, bolso, mochila, bolsa, celular y bolsillo activos. Destinos en verde; el resto (p. ej. celular) en cian. El bolsillo se estima con las caderas de la pose (COCO 11/12). MediaPipe también descarta máscaras que son la mano vacía (los landmarks cubren la máscara). Si dos palmas de MediaPipe salen a menos de 28 px, se conservan las muñecas de la pose.

Filtros extra (no son flags): se tiran móviles débiles o que no están junto a una palma; cestos/bolsas en el borde de la imagen con score bajo (p. ej. packs de latas); accesorios de 1 muestra lejos de la persona (ROI congelado en el pasillo). El bolso/mochila se **ancla** a la persona cuando sí está sobre ella, para que el ROI no se quede quieto.

---

## Persona de la sesión (varias personas en el frame)

SAM 3 y el resto del pipeline común no deben segmentar las manos de un transeúnte. El ancla vive en `detector_palma/ancla_sesion.py` (`AnclaPersonaSesion`) y se aplica **después** de YOLO-pose y **antes** de MediaPipe / SAM, en cada `AnalizadorEscena.analizar`. Se reinicia al empezar cada clip (`detectar_objetos` → `reset_ancla`), para no mezclar sujetos entre vídeos.

Sin `poses_full.npy` de C1 (este repo no exporta GuadIA), el sujeto se elige así:

1. **Semilla** (primer frame con personas): por defecto (`--ancla-persona auto`) gana quien **alcanza la balda** y está más cerca de la cámara. `izquierda` / `derecha` solo cubren los extremos. Con tres o más, usa el índice `0`/`1`/`2`… de izquierda a derecha, `centro`, `altura`, un clic `X,Y`, o las **muñecas** `x1,y1,x2,y2` (matching C1: YOLO a ≤ `--ancla-distancia`). `--ancla-listar` escribe `ancla_candidatos.jpg` e imprime `munecas=x,y x,y` para copiarlas.
2. **Seguimiento**: IoU del bbox (≥ 0.28 bloquea; ≥ 0.12 recupera), distancia de muñecas ≤ `--ancla-distancia` (140 px) e histograma HSV del torso.
3. **Pérdida**: si nadie encaja, `personas` y `manos` se vacían. SAM no corre sobre otra persona. Un `--fps` bajo (p. ej. 3) puede perder al sujeto unos frames; no se reasigna al transeúnte.

`tracking.json` guarda `session_person_anchor` (`temporal_iou` u `off`), `seed_frame`, `seed_criterion`, `seed_index`, `frames_locked`, `frames_lost`, `max_candidates`. El overlay solo pinta el esqueleto del sujeto.

Clips de prueba: `videos/multi_pers_clip_{1,2,3}.mp4` y carpetas `videos/multi_pers_clip_*_frames_chunk00N`. El contrato C1 (muñecas de `poses_full.npy`) está descrito en `SAM3_MULTI_PERSON_SESSION_ANCHOR.md`; aquí el equivalente autónomo es el tracker temporal (§6.2 de ese documento).

```bash
# 1) Ver quién es 0, 1, 2… y copiar muñecas (JPEG numerado, sin SAM)
.venv/bin/python sam3_mano_objeto.py \
  --carpeta-frames videos/multi_pers_clip_1_frames_chunk001 \
  --identificadores identificadores.yaml --fps 12 --ancla-listar --salida salidas
# 2) Seguir a la persona del medio (o la que sea: 0 izquierda, 1, 2…)
.venv/bin/python sam3_mano_objeto.py \
  --carpeta-frames videos/multi_pers_clip_1_frames_chunk001 \
  --identificadores identificadores.yaml --mediapipe --fps 12 \
  --ancla-persona 1 --salida salidas
# 3) Una persona, sembrando con sus muñecas (izq, der) del primer frame:
.venv/bin/python sam3_mano_objeto.py \
  --carpeta-frames videos/6_clip_r7_frames --identificadores identificadores.yaml \
  --mediapipe --fps 12 \
  --ancla-persona 645.4,553.5,942.5,491.4 --salida salidas
```

---

## Objeto en la mano: qué se acepta y qué se tira

El recorte de muñeca es amplio a propósito (estantería, puño, traspaso). La aceptación es estricta:

- **Confianza de muñeca.** `hands.min_confidence: 0.50` en el YAML. Por debajo no se lanza SAM/YOLO en esa palma. Si **ninguna** muñeca supera el umbral en todo el clip (de espaldas desde el primer frame), `final_state: no_hands` y no hay objetos.
- **Confianza de la mano (dedos).** Con `--mediapipe`, `hands.min_hand_confidence: 0.30`. Palma abierta que cuelga o score de dedos bajo junto al muslo: no se lanza SAM. Un **puño cerrado** o un brazo que alcanza la estantería sí se busca aunque Hands no puntúe los dedos (el artículo robado no tiene por qué ser un móvil).
- **Mercancía genérica.** El recorte de palma (YOLO-seg + SAM) es el detector principal. `botella` y `celular` no se buscan ni se pintan en el frame completo (ni en JPEG ni en `anotado.mp4`): las baldas no se etiquetan. Solo cuenta lo que está en el recorte de muñeca. Un pack rectangular en la mano (caja, cartón) se acepta aunque esté junto al torso.
- **Sin persona / sin manos.** No se lanza SAM ni YOLO de identificadores. El overlay no pinta botellas de góndola ni `phone` en el HUD.
- **No es el antebrazo.** Una máscara que sigue el codo→muñeca o que cae hacia el codo (manga) se tira. No es un artículo.
- **Sin teleport L↔R.** En frames consecutivos el mismo objeto no puede pasar de izquierda a derecha (ni al revés) salvo un traspaso real (`ambas` o las dos palmas puntúan). Un identificador `celular` tampoco salta de una mano a la otra en un frame.
- **Una persona.** El ancla de sesión elige un sujeto y lo sigue (IoU / muñecas). No se usa “el más alto” en cada frame. `--sin-ancla-persona` lo desactiva.
- **Palma abierta colgando** (brazo vertical junto al muslo): no se busca objeto. Un puño o una palma que **vuelve al torso** después de alcanzar la estantería sí. Un pack rectangular pegado a la palma en la cadera no se tira como ropa.
- **Ropa / muslo / puño vacío.** Palma abierta cuya máscara es la mano: se tira. Un puño cerrado en un alcance o al volver al cuerpo se conserva (ocultación). Un SAM pegado a una palma que cuelga se trata como puño vacío, no como artículo.
- **Balda en el recorte.** Si el brazo está lejos del torso, un blob más lejos de la palma que la propia palma (cartel, góndola) se rechaza o se recorta al disco de la palma.
- **Bulto estático en la cadera.** Una mancha de ropa o de puño vacío que no se aleja del torso (persona quieta, manos libres) no es un agarre. Se filtra **después** de fusionar pistas, para no unir dos fantasmas cortos en uno largo. Excepción: el artículo real que acaba escondido a la espalda o dentro del ROI de la persona.
- **No pegar cadera con agarre.** Un bulto en la cadera de una mano no se asocia ni se fusiona con un alcance posterior de la otra (o de la misma cuando el brazo sí se aleja).
- **No arrastrar el ROI.** `extender_si_sigue_en_mano` no traduce la caja si la palma vuelve pegada al torso (evita un fantasma de cadera), salvo un pack rectangular que sigue en la misma mano.

---

## Estado final del objeto

`final_state` (inglés en JSON y `summary.txt`) se escribe en la cabecera del clip y en cada objeto. El del clip se decide así, en este orden:

| Condición | `final_state` |
|---|---|
| Nunca hay muñecas por encima de `hands.min_confidence` | `no_hands` |
| No queda ningún objeto en mano | `no_data` |
| El último ROI está en un **bolsillo** (con gesto) | `disappeared` (`ends_in_container` true; `final_container` es el bolsillo) |
| El último ROI está en **cesto / carrito / bolsa** de depósito | `inside_container` |
| Se siguió en la palma y esa mano se esconde **a la espalda** (codo abierto, no un brazo colgando) | `disappeared` |
| Se siguió en la palma y después las mismas palmas están a la vista vacías (abiertas, colgando o libres en pose) | `disappeared` |
| Se cogió algo (brazo lejos del torso), el último ROI entra en la persona y las manos quedan libres | `disappeared` |
| El último frame visible sigue en la mano, y la persona se va o se da la vuelta **con el objeto** | `in_hand` |
| El objeto queda visible lejos del cuerpo (estantería) | `released` |
| Estuvo en la mano y deja de verse sin las condiciones de arriba | `disappeared` |

Un brazo **colgando al costado** no cuenta como “detrás del tronco”. Si la persona se gira de espaldas **después** de esconder el artículo en el cuerpo, sigue siendo `disappeared`, no `in_hand`.

---

## Depósito: mueble vs bolsillo

`contenedor_final` mira el último frame visible del objeto, en este orden:

1. **Mueble** (`cesto`, `carrito`, bolsa que no es la que ya se lleva en la cadera). Basta solape o que la mano entre después en ese ROI.
2. **Bolsillo** solo si, a la vez:
   - el torso está erguido (no es agacharse al cesto);
   - hay un gesto de meter la mano: la palma baja respecto a la cadera e entra en el ROI del mismo lado;
   - esa mano no está ocupada por un objeto que no cabe en el bolsillo (bolsa de compra);
   - la persona no sale de escena ni está de espaldas (sin cara visible no se afirma el depósito);
   - el objeto no acaba de estar lejos del cuerpo (estantería).
3. **Bolso / mochila** (`accessory: true`): se detectan y se pintan si van sobre la persona, pero no se usan como destino si el objeto no llega de verdad. Una mochila al fondo del pasillo no “traga” lo que se deja en la estantería.

Si el objeto acaba en el **bolsillo**, o se estaba siguiendo desde las manos y deja de verse en las palmas (o se oculta en el ROI de la persona / a la espalda), `final_state` es `disappeared` (el `final_container` del bolsillo se conserva). Cesto/carrito/bolsa siguen siendo `inside_container`. Si el último frame del objeto sigue en la mano y la persona se va o se da la vuelta **sosteniéndolo**, `final_state` es `in_hand`. Si lo deja visible en una estantería, es `released`.

En el vídeo anotado el ROI del bolsillo **solo se pinta** cuando el gesto está activo. No se pinta el bolsillo del lado cuya mano sostiene un objeto, ni si el ROI del bolsillo solapa ese objeto.

---

## Salidas

Por clip y tecnología, en `salidas/<tecnologia>/<stem>/`:

| Archivo | Contenido |
|---|---|
| `anotado.mp4` | Máscaras de objeto, caja + etiqueta de mano (`L`/`R`/`BOTH`/`FREE`) y score, ROI de identificadores con confianza, esqueleto cintura↑. HUD: `frame`, `t=`, objetos en mano y, si hay, `phone 0.73` / `inside 0.89`. |
| `frames/<video>_frame_XXXXX.jpg` | Mismo overlay; `XXXXX` es el índice original. `--sin-frames` lo omite. Incluye las mismas confianzas. |
| `tracking.json` | Traza completa (esquema abajo). Con `--summary`: sin `timeline[]` ni `rois` por frame. Campos en inglés. |
| `summary.txt` | Los mismos eventos en inglés, más la frase de `--fps`. |
| `metrics.json` | `processed_frames`, `mean_fps`, `ms_per_frame`, picos RAM/VRAM, CPU, `stages`. |

### `tracking.json` (pipeline común)

Cabecera: `video`, `model`, `width`, `height`, `frames`, `source_frames`, `fps`, `output_fps`, `sampling`, `pose_backend` (`yolo-pose` o `mediapipe`), `hands_visible` (bool), `final_state` (del clip), `inside_container_confidence` / `cell_phone_confidence` si aplican, `session_person_anchor`, `seed_frame`, `frames_locked`, `frames_lost`, `max_candidates`, `parameters`, `identifiers_yaml`.

`final_state` del clip: si no hay muñecas, `no_hands`. Si hay muñecas pero ningún objeto, `no_data`. Si hay varios objetos, gana el de mayor prioridad (`inside_container` > `disappeared` > `in_hand` > `released`). Un puño que sigue `in_hand` no tapa la ocultación. Si el clip es `inside_container`, la cabecera incluye `inside_container_confidence` (el del objeto ganador). `parameters.hand_min_confidence` y `parameters.mediapipe` reflejan el YAML y `--mediapipe`.

Con mp4, `fps` es el del archivo (float) y `parameters.fps` es el `--fps` de muestreo (o `null`). Con arrays, `fps` y `parameters.fps` son el entero de la línea temporal; `parameters.input` es `"frames"`; `parameters.frames_origin` es la `list[int]`; `parameters.width` / `parameters.height` son enteros. `width` y `height` de la cabecera también son enteros.

`sampling` (mp4):

```json
{
  "specified": true,
  "input": "video",
  "fps_parameter": 2,
  "video_fps": 15.0,
  "integer_rate": 15,
  "source_frames": 150,
  "processed_frames": 20,
  "offsets_in_second": [0, 7],
  "description": "--fps 2 was set. …"
}
```

`sampling` (arrays): `"input": "frames"`, `fps_parameter` entero, `source_indices` igual a `frames_origin`, y `description` en inglés (`--frames was set … Timeline fps is …`).

`containers[]`: `id`, `category`, `prompt`, scores, `first_frame` / `last_frame` (origen), `frames_seen`, `destination`, `accessory`, `body`, `side` (`left`/`right` en bolsillos; si no, `null`), `last_roi`, `rois` frame→`[x1,y1,x2,y2]`. Si `category` es `celular`, también `confidence` (0–1).

`objects[]`:

- `id`, `frames_visible`, `frames_in_hand`, `first_frame`, `last_frame`, `reseeds`
- `hands`: `{left, right, both, no_hand}`
- `dominant_hand`
- `last_roi`, `last_center`, `last_visible_frame`
- `final_state`: `in_hand` \| `released` \| `disappeared` \| `inside_container` \| `no_data` \| `no_hands`
- `inside_container_confidence` (0–1, solo si `final_state` es `inside_container`): solape del ROI + centro dentro + score del identificador
- `cell_phone_confidence` (0–1, si hay identificador `celular` que solapa el objeto): score YOLO + estabilidad + cercanía a la mano
- `ends_in_container` (bool)
- `final_container`: `{id, category, roi, fraction_inside, center_inside, score, confidence}` o `null`
- `events[]`: `appears`, `disappears`, `reappears`, `releases`, `regrasps`, `hand_change` (`from`/`to`)
- `timeline[]`: `frame`, `processed_frame`, `t`, `hand`, `roi`, `center`, `area`, `grip_score`, `hand_scores`, `origin` (`detection` \| `propagation`)

Con `--summary` el mismo archivo lleva `"summary": true`, omite `objects[].timeline` y `containers[].rois`, y sustituye `parameters.frames_origin` / `sampling.source_indices` por un recuento (y `source_index_span`). Siguen `events[]`, `last_roi`, `final_state`, `inside_container_confidence` (si hay depósito), `cell_phone_confidence` (si hay móvil) y `final_container` (con `confidence`). `summary.txt` también escribe esas confianzas. `containers[]` de un `celular` incluye `confidence`.

`hand` en JSON es `left` / `right` / `both` / `no_hand`. En el overlay se abrevia `L` / `R` / `BOTH` / `FREE`. Izquierda y derecha son las **de la persona**.

VISOR escribe un JSON distinto (`eventos`, `linea_temporal` estilo SAM 3 original) más el bloque `muestreo`.

### Comparativa

`salidas/comparativa/comparativa.csv` (también `.md` y `.json`). Una fila por `(clip, backend)`. Columnas: `clip`, `backend`, `ok`, `frames`, `fps`, `ms_frame`, `ms_segmentacion`, `ram_pico_mb`, `vram_pico_mb`, `cpu_pct`, `objetos`, `desapariciones`, `frames_con_objeto`, `codigo_salida`, `error`, `carpeta`.

Cada pasada sustituye su fila y **no borra** las demás. Columnas añadidas a mano en el CSV se heredan. En `comparativa.md`, `ok` se escribe `yes` / `NO`.

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
    extra={"propagator": "…"}, # se fusiona en tracking.json
)
```

6. No cambiar el esquema de `objects[]` / `containers[]` si un consumidor ya parsea `tracking.json` (`side` y `body` en contenedores son aditivos).
7. Documentar pesos y `third_party/` en este README.

Para varios clips, `procesar_lote(detectores, videos, args, tecnologia, backend, modelo, seguir)` llama a `procesar_video` y no aborta el resto si uno falla.

Para un servicio que solo quiera ROI final + contenedor, `--summary` (y opcionalmente `--sin-frames`). Leer `tracking.json` → `objects[i].last_roi`, `ends_in_container`, `final_container`.

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

`scripts.txt` guarda comandos reales. Receta de laboratorio (SAM 3, MediaPipe, 3 muestras por segundo):

```bash
.venv/bin/python sam3_mano_objeto.py \
  --video videos/3_clip_2.mp4 \
  --identificadores identifiers.yaml \
  --mediapipe --fps 3 \
  --salida salidas
```

`--fps 3` es el valor por defecto en esos clips. Usar `--fps half` o `--fps third` si el gesto es más denso (traspaso corto, depósito en cesto). Varios `--video` no se acumulan: gana el último. Para un lote, `--videos videos`. Para una carpeta de JPEG, `--carpeta-frames videos/6_clip_r9_frames --fps 12` (sin `--video`).

---

## Vídeos de prueba (locales, no en git)

- `videos/clip.mp4`: objeto en la mano izquierda y traspaso a la derecha.
- `videos/clip_2.mp4`: misma tarea, segundo clip.
- En el laboratorio hay series `N_clip_M.mp4` (p. ej. `3_clip_3.mp4`, `6_clip_2.mp4`) usadas en la comparativa de 43 clips, y clips `6_clip_r*.mp4` / `fp_*.mp4` de depuración (palma vacía, bolsillo, espalda, estantería).
- `videos/multi_pers_clip_*.mp4` y `videos/multi_pers_clip_*_frames_chunk00N`: varias personas en el pasillo; el ancla debe seguir a un único sujeto.
- `fp_agosto2026/fp_*.mp4`: clips de depuración (depósito en cesto vs bolsillo, salida de escena, accesorios lejanos).

---

## Problemas frecuentes

- **`python` no es el venv.** Pose26 / YOLO26 fallan. Usar `.venv/bin/python` o `source .venv/bin/activate`.
- **Celular id 66.** En Ultralytics 66 es `keyboard`. Usar `name: cell phone` (id 67). El código resuelve por nombre.
- **Móvil fantasma.** Un `celular` débil pegado al objeto de la estantería se descarta. Hace falta score alto y palma cerca.
- **Ruido en SAM 3 / recorte grande.** El recorte de muñeca es amplio (`padding-muneca` 1.2, `crop-min` 96); la aceptación es `factor-palma` 0.85 sobre el radio de la palma (tope 88 px). Si el brazo alcanza la góndola, el recorte se reduce. No ampliar el factor sin refiltrar agarre.
- **Objeto fantasma en la cadera (manos vacías, persona quieta).** SAM se come el pantalón, el puño o la manga. El pipeline tira palmas abiertas que cuelgan, máscaras que son la mano o el antebrazo, bultos estáticos en la cadera y no pega ese bulto con un agarre posterior. No bajar `hands.min_confidence` (0.50). `min_hand_confidence` (0.30) ya no bloquea un puño en un alcance.
- **Celular que salta de mano en un frame.** No es un traspaso: o la pose L/R está invertida o YOLO mezcló dos detecciones. El rastreador no asocia L↔R en hueco ≤ 2 frames.
- **SAM se come la estantería al alargar el brazo.** Un blob más lejos del torso que la palma se rechaza o se recorta al disco de la palma. El agarre real queda en la mano, no en el cartel.
- **Objeto fusionado con bolso de hombro.** `accessory: true` en el YAML y `descartar_accesorios` en el pipeline. El ROI del bolso se ancla a la persona; una detección lejana de 1 frame se tira.
- **Mochila lejana como destino.** Ya no. `accessory: true` no es depósito de reserva: dejar un artículo en la estantería no lo mete en una mochila al fondo del pasillo.
- **Bolsillo vs cesto.** Agacharse al cesto no cuenta como meter la mano en el bolsillo. Gana el mueble. El ROI del bolsillo no se pinta si esa mano sostiene el objeto.
- **Persona que sale o se da la vuelta.** Si el último evidencia sigue en la mano: `in_hand`. Bolsillo, pérdida en las palmas, ocultación en el ROI de la persona o mano a la espalda: `disappeared`, no `inside_container`. Un brazo colgando no es “a la espalda”.
- **Muñecas nunca visibles.** De espaldas o por debajo de 0.50: `final_state: no_hands`, sin objetos. Si antes se vio objeto+manos y luego se gira **sosteniéndolo**, es `in_hand`, no `no_hands`.
- **Varias personas / SAM en el transeúnte.** `izquierda`/`derecha` no bastan si hay tres o más. `--ancla-listar` numera a todas (izquierda→derecha) en `ancla_candidatos.jpg`; relanza con `--ancla-persona 0|1|2`, `X,Y` o `x1,y1,x2,y2` (muñecas). `--sin-ancla-persona` desactiva el filtro.
- **Cruce de personas.** Si el sujeto queda tapado, el ancla no hereda el esqueleto del otro: ese frame queda vacío y se recupera cuando YOLO vuelve a ver al sujeto (no al transeúnte).
- **Packs de latas como cesto/bolsa.** YOLOE en el borde inferior se filtra por score, tamaño y posición.
- **HOIST no compila.** Hace falta el truco `third_party/nvcc-wrap` o el fallback PyTorch de atención deformable.
- **Cutie `No está el código`.** Falta el clone en `third_party/Cutie`.
- **YOLOE silencioso.** Falta `pesos/yoloe-26m-seg.pt`; carrito/cesto/bolsa no saldrán, bolso/mochila/celular sí si están en COCO.
- **`--fps` y clips que no van a 15 fps.** Los ejemplos 1/8 y 1/6/11 son para rate 15. Un clip a 12.333 fps redondea a 12; `--fps 2` toma offsets 0 y 6 (frames 1 y 7).
- **Duración del anotado.** Con mp4 y `--fps N` el writer usa N fps. Sin `--fps`, usa el fps del archivo. Con arrays, usa el `output_fps` derivado de `frames_origin` y el fps entero de la línea temporal.
- **`--frames` no es un flag de shell.** Hay que asignar `args.frames` o el kwarg `frames=` con `list[np.ndarray]`. `--fps`, `--width` y `--height` sí van por CLI, y tienen que ser enteros.

100DOH y detectores mano-objeto de esa generación no entran: piden CUDA/PyTorch anteriores a los de esta máquina.

---

## Servicio Kanvis (`kanvis-segmentation`)

Contenedor Docker autónomo: C2/C3 llaman a SAM 3 por **gRPC**. Lee frames de MinIO, corre el pipeline de este repo y devuelve `tracking_json` + `roi_fp_review` (esquema `c2_roi_fp_review_v1`). Especificación de contrato: `KANVIS_SEGMENTATION_SERVICE_SPEC.md`.

### Puertos

| Servicio | Puerto | Variable |
|----------|--------|----------|
| gRPC `SegmentationService` | **50100** | `KANVIS_SEGMENTATION_LISTEN` (default `0.0.0.0:50100`) |
| Prometheus `/metrics` | **9109** | `KANVIS_SEGMENTATION_METRICS_PORT` |

No se usa el 50052 (C2). El gRPC de este servicio es **50100**.

### Qué hace una petición

1. Resuelve la sesión por campos tenant (`bucket`, `retail_id`, `store_id`, `camera_id`, `session_id`, `chunk_stem`) o por refs C2 (`source_ref`, `pose_chunk_ref`, `image_chunk_ref`). `pose_chunk_ref` / `image_chunk_ref` pueden ser relativos al `source_ref` (`chunk_001/poses_full.npy`).
2. Baja `chunk_XXX/frames/`, `frames_meta.json` y `poses_full.npy` (si falta: `poses_compat.npy` / `poses.npy`). Con `span_spec=overlap_half:chunk_001:chunk_002:ratio=0.5000` usa la **última mitad** de `chunk_001` y la **primera mitad** de `chunk_002` (24 JPEGs → 12+12). `with_next` sigue concatenando carpetas completas.
3. Ancla al sujeto de la sesión (muñecas C1) y corre SAM 3. El gate de muñecas mira solo los `frames_meta` de la ventana.
4. Responde `tracking_json` (compacto, con `p1_motion_stem` / `frames_used` en solape) y `roi_fp_review_json`. Con `persist_to_minio=true` y `dry_run=false` escribe `…/chunk_XXX/roi_tracking.json` en el **ancla** (`chunk_001`, no `chunk_001_002/`).

RPC: `ApplySegmentation`, `HealthCheck`. Proto: `proto/kanvis/segmentation/v1/segmentation.proto`.

### Variables de entorno

| Variable | Default | Uso |
|----------|---------|-----|
| `KANVIS_SEGMENTATION_LISTEN` | `0.0.0.0:50100` | Bind gRPC |
| `KANVIS_SEGMENTATION_METRICS_PORT` | `9109` | Prometheus |
| `SEGMENTATION_S3_ENDPOINT_URL` | `http://host.docker.internal:9000` | MinIO publicado en el host (`:9000`). Si el servicio y MinIO comparten red Docker, usa el hostname del contenedor MinIO. |
| `SEGMENTATION_S3_ACCESS_KEY_ID` | — | |
| `SEGMENTATION_S3_SECRET_ACCESS_KEY` | — | |
| `SEGMENTATION_S3_REGION` | `us-east-1` | |
| `SEGMENTATION_S3_SECURE` | `false` | TLS |
| `SEGMENTATION_SAM3_SUMMARY` | `true` | `tracking.json` compacto |
| `SEGMENTATION_SAM3_MIN_WRIST_VISIBLE_FRAMES` | `3` | Umbral de muñecas para no lanzar SAM |
| `SEGMENTATION_SAM3_WRIST_MATCH_MAX_PX` | `140` | Radio del ancla de sesión |
| `SEGMENTATION_IDENTIFIERS_YAML` | `identifiers.yaml` | Catálogo dentro de la imagen |
| `SEGMENTATION_WORK_DIR` | `/tmp/kanvis-segmentation` | Scratch |

### Scripts

Todos salen distinto de cero si fallan, imprimen `PASS` / `FAIL` y con `DRY_RUN=1` no escriben en MinIO.

| Script | Función |
|--------|---------|
| `scripts/build.sh` | Construye `kanvis-segmentation:gpu` (`TARGET=cpu` para CPU) |
| `scripts/deploy.sh` | Build + `docker compose -f docker-compose.segmentation.yml up -d` |
| `scripts/smoke_grpc.sh` | HealthCheck en `localhost:50100` (`FORCE_RPC=1` fuerza la llamada) |
| `scripts/smoke_minio.sh` | `ApplySegmentation` contra `SOURCE_REF` + `POSE_REF` |
| `scripts/run_local_offline.sh` | Fixture local `fixtures/session_001/` (sin MinIO) |
| `scripts/gen_proto.sh` | Regenera stubs gRPC |

```bash
# Imagen GPU y arranque (publica 50100 y 9109)
bash scripts/build.sh
bash scripts/deploy.sh

# Health
TARGET=localhost:50100 FORCE_RPC=1 bash scripts/smoke_grpc.sh

# MinIO (DRY_RUN=1 no escribe roi_tracking.json)
export SOURCE_REF="s3://kanvis-chunks/sessions/<retail>/<store>/<camera>/<session>/"
export POSE_REF="${SOURCE_REF}chunk_001/poses_full.npy"
export TARGET="localhost:50100"
DRY_RUN=1 bash scripts/smoke_minio.sh

# Solape C3 (mitad final de chunk_001 + mitad inicial de chunk_002)
SPAN_SPEC='overlap_half:chunk_001:chunk_002:ratio=0.5000' \
CHUNK_STEM=chunk_001_002 \
DRY_RUN=1 bash scripts/smoke_minio.sh

# Offline, sin MinIO
SESSION_ROOT=fixtures/session_001 DRY_RUN=0 bash scripts/run_local_offline.sh
```

Compose: `docker-compose.segmentation.yml` (mapeo `50100:50100` y `9109:9109`). Cliente: `tools/grpc_smoke_client.py`. Runner local: `tools/run_offline_segmentation.py`. Tests: `.venv/bin/python -m pytest tests -q`. Informe: `docs/TEST_REPORT.md`.

La imagen GPU instala `libegl1` (MediaPipe Holistic) y reinstala `torchvision` desde el índice cu128 (Ultralytics deja un wheel sin `torchvision::nms`). MinIO se alcanza por `host.docker.internal:9000` (puerto publicado en el host), no por el hostname `minio` de otra red Compose.
