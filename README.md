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

El objeto en la mano, los identificadores de escena (`identificadores.yaml`) y el JSON son los del pipeline común (`pipeline_manos.py`): muñecas con YOLO26n-pose, recorte con padding, ambas manos, detección inmediata de carrito/cesto/bolso/mochila/bolsa/celular, y el último ROI del objeto con `acaba_en_contenedor`.

```bash
python hoist_mano_objeto.py --videos videos --salida salidas
python hoist_mano_objeto.py --video videos/3_clip_3.mp4 --salida salidas
```

Resultados en `salidas/hoist/<clip>/`. La fila de la comparativa se llama `HOIST-Former`.

## Cutie

Cutie no detecta el objeto: lo sigue. Toda la detección es previa y funciona así:

1. **Muñecas con YOLO26n-pose.** De cada muñeca se toma la palma proyectada y un abanico de puntos de agarre.
2. **ROI de muñeca.** Alrededor de cada palma se recorta una ventana con `--padding-muneca` (1.2 veces el radio del brazo, mínimo 96 px de semilado) y solo dentro de ese recorte se buscan objetos, con YOLO26n-seg a confianza baja y MobileSAM guiado por los puntos de la palma. Un filtro de forma (`--relacion-max 3.0`, `--relleno-min 0.3`) descarta brazos y baldas.
3. **Izquierda, derecha o ambas.** Cada candidato se puntúa contra las dos manos. Si las dos superan `--umbral-agarre` y `--contacto-min`, el objeto queda marcado como sostenido con `ambas`.
4. **Propagación con Cutie.** Cada objeto se siembra y se sigue hasta el final. `--salto-max` impide que la máscara salte a una estantería y `--frames-sin-mano` corta la propagación cuando el objeto lleva mucho tiempo sin mano, de forma que el último ROI siga siendo fiable.
5. **Identificadores de escena (`identificadores.yaml`).** Se detectan en el frame completo. Las clases que existen en COCO (bolso 26, mochila 24, celular 67) salen de YOLO26-seg con el id de [Ultralytics](https://docs.ultralytics.com/datasets/detect/coco/); carrito, cesto y bolsa se preguntan por texto a YOLOE. Pon `activo: false` para apagar una clase. `--identificadores otra.yaml` usa otro archivo.
6. **Destino final.** Con los últimos `--frames-fin` frames visibles se decide si el objeto acabó dentro del ROI de alguno de esos contenedores.

```bash
source .venv/bin/activate
uv pip install -r requirements-cutie.txt
git clone --depth 1 https://github.com/hkchengrex/Cutie.git third_party/Cutie
uv pip install --no-deps -e third_party/Cutie
python cutie_mano_objeto.py --videos videos --salida salidas
python cutie_mano_objeto.py --video videos/3_clip_3.mp4 --salida salidas
```

`--videos` toma una carpeta y `--video` un único archivo. La primera ejecución descarga `cutie-base-mega.pth` en `third_party/Cutie/weights/`, y `yoloe-26m-seg.pt` más su encoder de texto `mobileclip2_b.ts` en `pesos/`. El lado interno por defecto es 720 px; el paper usa 480. `--sin-contenedores` salta los identificadores de escena, `--sin-sam` salta MobileSAM y `--sin-frames` no vuelca los frames sueltos. `--max-frames 24` sirve para una prueba corta.

## Identificadores de escena

El archivo `identificadores.yaml` en la raíz activa o desactiva la detección inmediata (en todo el frame, sin esperar a la mano) de carrito, cesto, bolso, mochila, bolsa y celular.

```yaml
celular:
  activo: true
  ultralytics_id: 67
  nombre: cell phone
mochila:
  activo: false
```

Los ids coinciden con las clases COCO de Ultralytics ([documentación](https://docs.ultralytics.com/datasets/detect/coco/)): `24` backpack, `26` handbag, `66` keyboard, `67` cell phone. El móvil es **67** en Ultralytics; el 66 es el teclado. El detector resuelve por `nombre`, así que encuentra `cell phone` aunque el número cambie. Carrito, cesto y bolsa no están en COCO y se preguntan por texto a YOLOE.

`destino: true` marca si el objeto de la mano puede acabar dentro (cesto, carrito, bolso). `accesorio: true` evita confundir el bolso al hombro o la mochila con lo que lleva en la palma. Los destinos se pintan en verde; el celular y el resto, en cian.

El vídeo anotado pinta las máscaras de los objetos, su caja con la mano (`IZQ`, `DER`, `AMBAS`), los ROI de los identificadores y el esqueleto YOLO de la cintura hacia arriba (caderas, torso, brazos, muñecas y palmas). `--sin-pose` lo oculta.

Resultados en `salidas/cutie/<clip>/`: `anotado.mp4`, `frames/`, `seguimiento.json`, `resumen.txt` y `metricas.json`. La fila de la comparativa se llama `Cutie`. `seguimiento.json` lleva la traza completa:

- `contenedores`: id, categoría, prompt que lo encontró, score, primer y último frame, ROI por frame y `roi_ultimo`.
- `objetos`: por objeto, `manos` con el reparto de frames entre izquierda, derecha, ambas y sin mano, `mano_predominante`, `ultimo_roi`, `ultimo_centro`, `ultimo_frame_visible`, `estado_final`, `acaba_en_contenedor`, `contenedor_final` con la fracción del objeto dentro, la lista de `eventos` y la `linea_temporal` frame a frame con ROI, centro, área, mano y score de agarre.

## Track-Anything

La parte que sigue el objeto es XMem. La semilla y los contenedores salen del pipeline común (muñecas, recorte, ambas manos, YOLOE). Si la máscara se vacía o se hincha, a los 8 frames se resiembra sin borrar la memoria larga.

```bash
source .venv/bin/activate
uv pip install -r requirements-track-anything.txt
git clone --depth 1 https://github.com/gaomingqi/Track-Anything.git third_party/Track-Anything
mkdir -p third_party/Track-Anything/checkpoints
# XMem-s012.pth, desde https://github.com/hkchengrex/XMem/releases
python track_anything_mano_objeto.py --videos videos --salida salidas
python track_anything_mano_objeto.py --video videos/3_clip_3.mp4 --salida salidas
```

XMem trabaja a 1080 px de lado menor. `--resolucion-interna 480` es el valor del yaml oficial. Los resultados van a `salidas/track_anything/<clip>/` con el mismo `seguimiento.json` que Cutie. La fila de la comparativa se llama `Track-Anything`.

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

Usa el pipeline común sin propagador: cada frame se redetecta en el recorte de la muñeca. Los identificadores de escena salen de `identificadores.yaml`.

```bash
source .venv/bin/activate
python yolo26_mano_objeto.py --videos videos --salida salidas
python yolo26_mano_objeto.py --video videos/3_clip_3.mp4 --salida salidas
```

Los pesos ya están en `pesos/`. Los resultados van a `salidas/yolo26/<clip>/` con el mismo JSON que Cutie. La fila de la comparativa se llama `YOLO26-seg`.

## SAM 3

Primer experimento de segmentación: localizar a la persona, ampliar su región de interés y seguir el objeto que lleva en las manos. El modelo es [SAM 3](https://github.com/facebookresearch/sam3) (checkpoint **SAM 3.1**, Object Multiplex). Acepta un texto en inglés y devuelve la máscara de cada instancia a lo largo del vídeo, con una identidad estable.

100DOH y otros detectores de mano-objeto de esa generación no entran: piden CUDA y PyTorch anteriores a los de esta máquina.

## Qué hace el script

`sam3_mano_objeto.py` usa el mismo pipeline que Cutie para el objeto en la mano (muñecas, recorte, ambas manos, identificadores de escena y JSON). El prompt de texto de SAM 3 no marca el objeto a esta distancia; la máscara sale del recorte de la muñeca.

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
python sam3_mano_objeto.py --video videos/3_clip_3.mp4 --salida salidas
```

## Resultados

Para cada vídeo, en `salidas/sam3/<clip>/`:

- `anotado.mp4` — máscaras de objeto, ROI de identificadores y pose YOLO de la cintura hacia arriba.
- `seguimiento.json` — traza completa: `objetos[].ultimo_roi`, `linea_temporal`, `acaba_en_contenedor` y `contenedor_final`.
- `resumen.txt` — los mismos eventos en texto.
- `metricas.json` — frames, FPS, RAM y VRAM de esa pasada.
- `frames/frame_00000.jpg` — cada frame anotado por separado.

Toda la comparativa vive en un único `salidas/comparativa/comparativa.csv` (también `.md` y `.json`), con una fila por clip y tecnología: `clip`, `backend`, `ok`, `frames`, `fps`, `ms_frame`, `ms_segmentacion`, `ram_pico_mb`, `vram_pico_mb`, `cpu_pct`, `objetos`, `desapariciones`, `frames_con_objeto`, `codigo_salida`, `error`, `carpeta`. Cada pasada reescribe su propia fila (`clip` + `backend`) y deja intactas las demás.

Cutie, Track-Anything, YOLO26-seg, SAM 3 y HOIST comparten `pipeline_manos.py`. Cutie y Track-Anything propagan la semilla; los otros redetectan cada frame. VISOR no entra en este pipeline.

## Vídeos de prueba

- `videos/clip.mp4`: una persona lleva un objeto en la mano izquierda y luego lo pasa a la derecha.
- `videos/clip_2.mp4`: segundo clip de la misma tarea.
