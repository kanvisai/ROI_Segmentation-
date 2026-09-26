# Seguimiento del objeto en la mano

Seis experimentos, el mismo directorio y el mismo `.venv`:

- `sam3_mano_objeto.py` — [SAM 3](https://github.com/facebookresearch/sam3). El checkpoint está restringido en Hugging Face.
- `hoist_mano_objeto.py` — [HOIST-Former](https://github.com/SupreethN/HOISTFormer) (CVPR 2024). Segmenta y sigue el objeto cogido con la mano. El ROI de la persona lo aporta Faster R-CNN.
- `cutie_mano_objeto.py` — [Cutie](https://github.com/hkchengrex/Cutie). Propaga la máscara del objeto en la mano. La máscara inicial sale del detector de palma.
- `track_anything_mano_objeto.py` — [Track-Anything](https://github.com/gaomingqi/Track-Anything). Propaga esa misma máscara con XMem.
- `visor_mano_objeto.py` — [VISOR](https://epic-kitchens.github.io/VISOR/) (EPIC-KITCHENS). Segmenta la mano y el objeto en contacto.
- `yolo26_mano_objeto.py` — [YOLO26-seg](https://docs.ultralytics.com/models/yolo26). Segmenta las instancias de cada frame y se queda con el objeto en la mano.

## HOIST-Former

El código oficial pide Python 3.8 y PyTorch 1.9. Este script lo ejecuta con el PyTorch 2.10 y CUDA 12.8 del `.venv`, sin bajar de versión.

Desde este directorio, con el entorno ya creado para SAM 3:

```bash
source .venv/bin/activate
uv pip install -r requirements-hoist.txt
git clone --depth 1 https://github.com/SupreethN/HOISTFormer.git third_party/HOISTFormer
```

PyTorch del venv es CUDA 12.8. El `nvcc` 12.8 de esta máquina no compila contra la glibc actual, y el `nvcc` 13.2 del sistema lo rechaza PyTorch por la versión mayor. En `third_party/nvcc-wrap` hay un envoltorio que informa 12.8 y llama al `nvcc` 13.2, con `include` y `lib64` enlazados a `/usr/local/cuda-13.2`. Con eso:

```bash
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

`third_party/` no se sube a git. Los vídeos siguen en `videos/`, también fuera del repositorio.

HOIST ve el vídeo en ventanas de 4 frames (con 2 de solape para no perder la identidad) a resolución completa. El yaml oficial baja el lado corto a 360 px; en estos clips de 1080p el objeto desaparece y la puntuación cae por debajo del umbral, así que el script mantiene 1080×1920. Devuelve la máscara del objeto y las de las manos. Faster R-CNN marca a la persona; el ROI es esa caja con un 35 % de padding. Un objeto solo cuenta si cae dentro. La mano izquierda o derecha sale de YOLO-pose (`pesos/yolo26n-pose.pt`): el lado es el de la persona, no el de la imagen. El agarre suma palma, dedos y cercanía. Si el objeto sigue viéndose pero ya no está en la mano, el estado es suelto y se guardan las dos últimas posiciones. Un cambio de mano espera 3 frames. Un hueco de 3 frames o menos no se cuenta como desaparición.

Resultados en `salidas/hoist/<clip>/`: `anotado.mp4`, `seguimiento.json`, `resumen.txt` y `metricas.json`. El vídeo pinta el ROI en amarillo, las manos en azul (izquierda) y naranja (derecha), y el objeto en magenta. Una cruz marca la última posición si deja de verse. Al terminar se actualiza `salidas/comparativa/comparativa.csv`.

Si la GPU se queda corta, baja la ventana: `--ventana 2`. Para una prueba de unos segundos: `--max-frames 24`. Si la persona está de espaldas: `--de-espaldas`.

## Cutie

Cutie no detecta el objeto: lo sigue. El detector de palma pone la primera máscara y Cutie la propaga hasta el final del clip. Si la máscara se vacía, o crece muy por encima de la semilla, ese frame no cuenta; a los 3 frames el detector vuelve a sembrar y Cutie sigue. La persona sale de Faster R-CNN y la mano, de YOLO-pose, igual que en HOIST.

```bash
source .venv/bin/activate
uv pip install -r requirements-cutie.txt
git clone --depth 1 https://github.com/hkchengrex/Cutie.git third_party/Cutie
uv pip install --no-deps -e third_party/Cutie
python cutie_mano_objeto.py --videos videos --salida salidas
```

La primera ejecución descarga `cutie-base-mega.pth` en `third_party/Cutie/weights/` (también está en `.gitignore` vía `third_party/`). El lado interno por defecto es 720 px; el paper usa 480. Con `--resolucion-interna 480` va más rápido y puede perder el objeto pequeño. `--resolucion-interna -1` deja el frame a 1080p. `--hueco-resembra -1` no vuelve a sembrar. `--max-frames 24` sirve para una prueba corta.

Resultados en `salidas/cutie/<clip>/`: `anotado.mp4`, `seguimiento.json`, `resumen.txt` y `metricas.json`. La fila de la comparativa se llama `Cutie`.

## Track-Anything

La parte que sigue el objeto es XMem. Segment Anything, en el repositorio original, sirve para marcar el primer frame a mano; aquí esa máscara la pone el detector de palma, igual que en Cutie. Si se vacía o crece muy por encima de la semilla, a los 3 frames se resiembra.

```bash
source .venv/bin/activate
uv pip install -r requirements-track-anything.txt
git clone --depth 1 https://github.com/gaomingqi/Track-Anything.git third_party/Track-Anything
mkdir -p third_party/Track-Anything/checkpoints
# XMem-s012.pth, desde https://github.com/hkchengrex/XMem/releases
python track_anything_mano_objeto.py --videos videos --salida salidas
```

XMem trabaja a 1080 px de lado menor para no perder el objeto pequeño. Una máscara que se vacía o se hincha no entra en la memoria; si faltan 8 frames, el detector de palma corrige sin borrar el histórico. `--resolucion-interna 480` es el valor del yaml oficial. Los resultados van a `salidas/track_anything/<clip>/` y la fila de la comparativa se llama `Track-Anything`.

## VISOR (EPIC-KITCHENS)

VISOR-HOS predice, en cada frame, la mano, si está en contacto, el lado (izquierda o derecha) y el objeto al que apunta ese contacto. El modelo está entrenado con vídeo de cocina en primera persona.

```bash
source .venv/bin/activate
git clone --depth 1 https://github.com/epic-kitchens/VISOR-HOS.git third_party/VISOR-HOS
mkdir -p third_party/VISOR-HOS/checkpoints
# model_final_hos.pth, enlace del README de VISOR-HOS
python visor_mano_objeto.py --videos videos --salida salidas
```

Hace falta detectron2, el mismo que usa HOIST. En estos clips la persona se ve de frente y la mano ocupa pocos píxeles: VISOR, entrenado en primera persona, puede no pasar el umbral de 0.5. Los resultados van a `salidas/visor/<clip>/` y la fila de la comparativa se llama `VISOR`.

## YOLO26-seg

YOLO26 segmenta cada frame. La pose (`pesos/yolo26n-pose.pt`) da la persona y el lado de la mano. El objeto pequeño que no entra en una clase COCO lo completa MobileSAM sobre la palma. No se descartan los objetos grandes.

```bash
source .venv/bin/activate
python yolo26_mano_objeto.py --videos videos --salida salidas
```

Los pesos ya están en `pesos/yolo26n-seg.pt`. Los resultados van a `salidas/yolo26/<clip>/` y la fila de la comparativa se llama `YOLO26-seg`.

## SAM 3

Primer experimento de segmentación: localizar a la persona, ampliar su región de interés y seguir el objeto que lleva en las manos. El modelo es [SAM 3](https://github.com/facebookresearch/sam3) (checkpoint **SAM 3.1**, Object Multiplex). Acepta un texto en inglés y devuelve la máscara de cada instancia a lo largo del vídeo, con una identidad estable.

100DOH y otros detectores de mano-objeto de esa generación no entran: piden CUDA y PyTorch anteriores a los de esta máquina.

## Qué hace el script

`sam3_mano_objeto.py` recorre cada vídeo de `videos/` en tres pasadas:

1. **Persona** (`person`). Se queda con la instancia de mayor área y construye un ROI. El padding por defecto es el 35 % del ancho y del alto de la caja, para que un objeto grande no quede fuera.
2. **Manos** (`hand`). Solo se conservan las que caen dentro de ese ROI. Si la persona mira a la cámara, la mano que está a la derecha de la imagen es su mano izquierda.
3. **Objeto** (`object held in the hand`, o el texto que indiques). Se queda con la máscara que toca una mano dilatada y cae dentro del ROI. Si SAM 3 cambia el identificador al pasar el objeto de una mano a la otra, el script intenta unir esos tramos.

Por cada frame anota si el objeto está visible. Cuando deja de verse, guarda la última posición (frame, centro, caja y mano). Si vuelve a verse, lo marca como reaparición. Si la mano asociada cambia, lo marca como un pase.

## Requisitos

- Python 3.12 (el `venv` del proyecto; hace falta 3.11 o superior)
- PyTorch 2.10 con CUDA 12.8
- GPU NVIDIA. El checkpoint pesa varios GB y la inferencia de vídeo va en GPU
- Acceso aceptado a los pesos en Hugging Face: [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)

## Preparar el entorno

Desde este directorio:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

PyTorch se instala aparte, desde el índice de CUDA 12.8. El resto (incluido SAM 3) sale de PyPI. La primera vez clona `facebookresearch/sam3`.

Los pesos no son públicos hasta aceptar la licencia. Con el entorno activado:

```bash
hf auth login
```

El token tiene que ser de lectura. La descarga ocurre sola la primera vez que se ejecuta el script.

## Ejecutar

Coloca los vídeos en `videos/` (esa carpeta está en `.gitignore` y no se sube al repositorio) y lanza:

```bash
source .venv/bin/activate
python sam3_mano_objeto.py --videos videos --salida salidas
```

Un solo archivo:

```bash
python sam3_mano_objeto.py --videos videos/clip.mp4
```

Si el prompt genérico no agarra el objeto, nómbralo en inglés:

```bash
python sam3_mano_objeto.py --videos videos/clip.mp4 --prompt-objeto bottle
```

Otros ajustes útiles:

| Opción | Efecto |
| --- | --- |
| `--padding 0.35` | Holgura del ROI respecto a la caja de la persona |
| `--de-espaldas` | Invierte izquierda/derecha. Por defecto la persona está de frente |
| `--umbral 0.5` | Probabilidad mínima de SAM 3 |
| `--version sam3` | Checkpoint de noviembre 2025, en lugar de SAM 3.1 |
| `--lote 4` | Frames por tanda. Si CUDA se queda sin memoria, baja a `--lote 2` |
| `--umbral-agarre 0.35` | Puntuación de palma y dedos para dar el objeto por cogido |
| `--sin-pose` | No usa el esqueleto. El lado de la mano sale de la imagen |
| `--dilatacion-mano 28` | Píxeles extra alrededor de la mano para dar por cogido el objeto |

En una GPU de 32 GiB el lote de 16 frames de SAM 3.1 no entra si hay otros procesos usando la tarjeta. El script usa 4 y activa `expandable_segments` para que PyTorch no fragmente la memoria.

## Resultados

Para cada vídeo, en `salidas/sam3/<clip>/`:

- `anotado.mp4` — persona en verde, ROI en amarillo, manos en azul (izquierda) y naranja (derecha), objeto en magenta. Si el objeto no está visible, una cruz marca la última posición.
- `seguimiento.json` — ROI, manos y objeto en cada frame, más la lista de eventos (`aparece`, `desaparece`, `reaparece`, `cambio_de_mano`).
- `resumen.txt` — los mismos eventos en texto.
- `metricas.json` — frames, FPS, RAM y VRAM de esa pasada.
- `frames/frame_00000.jpg` — cada frame anotado por separado, con el mismo dibujo que el vídeo.

Toda la comparativa vive en un único `salidas/comparativa/comparativa.csv` (también `.md` y `.json`), con una fila por clip y tecnología: `clip`, `backend`, `ok`, `frames`, `fps`, `ms_frame`, `ms_segmentacion`, `ram_pico_mb`, `vram_pico_mb`, `cpu_pct`, `objetos`, `desapariciones`, `frames_con_objeto`, `codigo_salida`, `error`, `carpeta`. Cada pasada reescribe su propia fila (`clip` + `backend`) y deja intactas las demás.

En `seguimiento.json`, `linea_temporal[].objeto.ultima_posicion` es la última detección válida. No cambia mientras el objeto sigue oculto.

El objeto en la mano no sale del prompt de texto de SAM 3 ni de la máscara nativa de HOIST: a esta distancia esos modelos no lo marcan. Lo segmenta YOLO sobre la palma y MobileSAM (`pesos/yolo26n-seg.pt`, `pesos/yolo26n-pose.pt` y `pesos/mobile_sam.pt`). SAM 3 y HOIST siguen aportando la persona, las manos y el ROI. Cutie y Track-Anything (XMem) usan esa misma máscara como semilla y la propagan.

SAM 3 no publica una pista nueva hasta confirmarla durante unos 15 frames (hotstart). El primer frame marcado puede ir un poco por detrás del momento real en el que el objeto entra en plano.

## Vídeos de prueba

- `videos/clip.mp4`: una persona lleva un objeto en la mano izquierda y luego lo pasa a la derecha.
- `videos/clip_2.mp4`: segundo clip de la misma tarea.
