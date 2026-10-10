"""Pipeline común: muñecas, objetos en la mano, contenedores y traza JSON.

Lo usan Cutie, Track-Anything, YOLO26-seg, SAM 3 y HOIST. Cada script solo
cambia cómo se sigue la máscara del objeto; la detección y la salida son las mismas.

  1. YOLO26n-pose localiza las muñecas.
  2. Alrededor de cada palma se recorta una ventana con padding amplio y solo
     ahí se buscan objetos (YOLO26n-seg y MobileSAM).
  3. Cada candidato se puntúa contra las dos manos. Si las dos superan el umbral,
     el objeto queda como sostenido con «ambas».
  4. El YAML ``identificadores.yaml`` activa o desactiva la detección inmediata
     (frame completo) de carrito, cesto, bolso, mochila, bolsa y celular. Las
     clases que existen en COCO usan el id de Ultralytics; el resto, YOLOE.
  5. El JSON guarda la traza completa y si el último ROI acabó dentro de un
     contenedor. El vídeo anotado pinta solo objetos y contenedores.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

from detector_palma.agarre import anotar_agarre, es_voluminoso, mascara_es_la_mano, puntuar_agarre
from detector_palma.geometria import caja_desde_mascara, iou_cajas
from detector_palma.personas import (
    CADERA_DER,
    CADERA_IZQ,
    ContextoEscena,
    HOMBRO_DER,
    HOMBRO_IZQ,
    dibujar_pose,
)
from detector_palma.tipos import Instancia
from salida_comun import Medidor, carpeta_de, publicar_comparativa

RAIZ = Path(__file__).resolve().parent
PESOS = RAIZ / "pesos"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
YAML_IDENTIFICADORES = (
    RAIZ / "identifiers.yaml" if (RAIZ / "identifiers.yaml").is_file()
    else RAIZ / "identificadores.yaml"
)

MANO_EN = {"izquierda": "left", "derecha": "right", "ambas": "both", "sin_mano": "no_hand"}
MANO_HUD = {"izquierda": "L", "derecha": "R", "ambas": "BOTH", "sin_mano": "FREE"}
ESTADO_EN = {
    "en_mano": "in_hand",
    "suelto": "released",
    "desaparecido": "disappeared",
    "dentro_de_contenedor": "inside_container",
    "sin_datos": "no_data",
    "sin_manos": "no_hands",
}
EVENTO_EN = {
    "aparece": "appears",
    "desaparece": "disappears",
    "reaparece": "reappears",
    "suelta": "releases",
    "reagarre": "regrasps",
    "cambio_de_mano": "hand_change",
}


def _mano_en(valor):
    if valor is None:
        return None
    return MANO_EN.get(valor, valor)
COLOR_OBJETO = [
    (255, 0, 255),
    (0, 255, 255),
    (0, 165, 255),
    (255, 255, 0),
    (180, 105, 255),
]
COLOR_CONTENEDOR = (0, 230, 0)
COLOR_IDENTIFICADOR = (0, 200, 255)


@dataclass
class Identificador:
    """Una clase del YAML: se detecta en el frame completo si está activa."""

    clave: str
    activo: bool = True
    ultralytics_id: int | None = None
    nombre: str | None = None
    prompts: list[str] = field(default_factory=list)
    destino: bool = False
    accesorio: bool = False
    cuerpo: bool = False


@dataclass
class ConfigManos:
    """Umbral de visibilidad de muñecas / manos leído del YAML."""

    min_confidence: float = 0.20


def _aviso(mensaje: str) -> None:
    print(f"  [aviso] {mensaje}", file=sys.stderr)


def _vaciar_cuda() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def cargar_catalogo(ruta: Path) -> tuple[list[Identificador], ConfigManos]:
    """Lee identifiers.yaml (o identificadores.yaml). Acepta claves en inglés o español."""
    if not ruta.is_file():
        raise SystemExit(f"Missing identifiers YAML {ruta}")
    try:
        datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"YAML inválido {ruta}: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"No se pudo leer {ruta}: {exc}") from exc
    if not isinstance(datos, dict):
        raise SystemExit(f"{ruta} must be a YAML mapping.")
    bloque = datos.get("identifiers") or datos.get("identificadores")
    if not isinstance(bloque, dict) or not bloque:
        raise SystemExit(f"{ruta} has no ``identifiers`` map.")
    manos_raw = datos.get("hands") or datos.get("manos") or {}
    if not isinstance(manos_raw, dict):
        manos_raw = {}
    min_conf = manos_raw.get("min_confidence", manos_raw.get("min_confianza", 0.20))
    try:
        min_conf = float(min_conf)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"hands.min_confidence inválido en {ruta}: {min_conf!r}") from exc
    min_conf = min(1.0, max(0.0, min_conf))
    config_manos = ConfigManos(min_confidence=min_conf)
    catalogo: list[Identificador] = []
    for clave, raw in bloque.items():
        if not isinstance(raw, dict):
            raise SystemExit(f"Entry «{clave}» in {ruta} is not a map.")
        uid = raw.get("ultralytics_id")
        prompts = [str(p).strip() for p in (raw.get("prompts") or []) if str(p).strip()]
        nombre = raw.get("name", raw.get("nombre"))
        if nombre is not None:
            nombre = str(nombre).strip() or None
        catalogo.append(
            Identificador(
                clave=str(clave),
                activo=bool(raw.get("active", raw.get("activo", True))),
                ultralytics_id=int(uid) if uid is not None and uid != "" else None,
                nombre=nombre,
                prompts=prompts,
                destino=bool(raw.get("destination", raw.get("destino", False))),
                accesorio=bool(raw.get("accessory", raw.get("accesorio", False))),
                cuerpo=bool(raw.get("body", raw.get("cuerpo", False))),
            )
        )
    return catalogo, config_manos


def _id_ultralytics(nombres: dict, ident: Identificador) -> int | None:
    """Resuelve el id COCO por el nombre de Ultralytics; el id del YAML es un atajo."""
    if ident.nombre:
        buscado = ident.nombre.lower()
        for clave, valor in nombres.items():
            if str(valor).lower() == buscado:
                return int(clave)
    if ident.ultralytics_id is not None:
        return int(ident.ultralytics_id)
    return None


@dataclass
class Vista:
    """Lo que se sabe de un objeto en un frame."""

    frame: int
    caja: tuple[int, int, int, int]
    centro: tuple[float, float]
    area: int
    mano: str | None = None
    score: float = 0.0
    scores: dict = field(default_factory=dict)
    origen: str = "propagacion"
    persona_id: int = -1


@dataclass
class Objeto:
    """Un objeto sostenido, con su máscara por frame y su traza."""

    id: int
    semillas: dict[int, np.ndarray] = field(default_factory=dict)
    mascaras: dict[int, np.ndarray] = field(default_factory=dict)
    vistas: dict[int, Vista] = field(default_factory=dict)
    resiembras: int = 0
    centro: tuple[float, float] | None = None
    ultimo_frame: int = -1


def comprobar_entorno() -> None:
    from sam3_mano_objeto import comprobar_entorno as _comprobar

    _comprobar()


def info_video(ruta: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(ruta))
    try:
        if not cap.isOpened():
            raise SystemExit(f"No se puede abrir el vídeo {ruta}")
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps <= 1e-3:
        fps = 25.0
    return ancho, alto, nframes, fps


def listar_videos(ruta: Path) -> list[Path]:
    if ruta.is_file():
        return [ruta]
    if not ruta.exists():
        raise SystemExit(f"No existe la carpeta de vídeos {ruta.resolve()}")
    videos = [
        p for p in sorted(ruta.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        raise SystemExit(f"No hay vídeos en {ruta}")
    return videos


def parsear_fps(valor):
    """Acepta un entero ≥ 1, ``half`` o ``third``."""
    if valor is None:
        return None
    if isinstance(valor, (int, np.integer)) and not isinstance(valor, bool):
        n = int(valor)
        if n < 1:
            raise SystemExit("--fps must be an integer >= 1, or half/third")
        return n
    texto = str(valor).strip().lower()
    if texto in ("half", "third"):
        return texto
    if texto.isdigit():
        n = int(texto)
        if n < 1:
            raise SystemExit("--fps must be an integer >= 1, or half/third")
        return n
    raise SystemExit("--fps must be an integer >= 1, or half/third")


def resolver_fps_proceso(fps_proceso, fps_video: float) -> int | None:
    """``half``/``third`` → N frames por segundo de vídeo (12 fps → 6 / 4)."""
    if fps_proceso is None:
        return None
    if fps_proceso in ("half", "third"):
        rate = max(1, int(round(float(fps_video) or 25.0)))
        return max(1, rate // (2 if fps_proceso == "half" else 3))
    return int(fps_proceso)


def offsets_muestreo(fps_video: float, fps_proceso) -> tuple[set[int] | None, int]:
    """Offsets dentro de cada segundo de vídeo. ``None`` = todos los frames."""
    rate = max(1, int(round(float(fps_video) or 25.0)))
    n = resolver_fps_proceso(fps_proceso, fps_video)
    if n is None:
        return None, rate
    if n < 1:
        raise SystemExit("--fps must be an integer >= 1, or half/third")
    if n >= rate:
        return None, rate
    return {int(i * rate / n) for i in range(n)}, rate


def describir_muestreo(
    fps_proceso,
    fps_video: float,
    n_origen: int,
    n_procesados: int,
    desde_frames: bool = False,
    indices_origen: list[int] | None = None,
) -> dict:
    """Texto y metadatos de si --fps se usó, si llegaron arrays, o el análisis fue frame a frame."""
    if desde_frames:
        fps_entero = int(round(float(fps_video)))
        indices = list(indices_origen) if indices_origen is not None else None
        origen_txt = ""
        if indices:
            muestra = ", ".join(str(i) for i in indices[:8])
            if len(indices) > 8:
                muestra += ", …"
            origen_txt = f" Source indices: {muestra}."
        return {
            "specified": True,
            "input": "frames",
            "fps_parameter": (
                fps_proceso if fps_proceso in ("half", "third")
                else int(fps_proceso) if fps_proceso is not None else fps_entero
            ),
            "video_fps": fps_entero,
            "integer_rate": fps_entero,
            "source_frames": n_origen,
            "processed_frames": n_procesados,
            "source_indices": indices,
            "description": (
                f"--frames was set ({n_procesados} arrays). "
                f"Timeline fps is {fps_entero}. No mp4 sampling was applied.{origen_txt}"
            ),
        }
    offsets, rate = offsets_muestreo(fps_video, fps_proceso)
    n = resolver_fps_proceso(fps_proceso, fps_video)
    etiqueta = (
        f"{fps_proceso} ({n})" if fps_proceso in ("half", "third") and n is not None
        else (str(int(fps_proceso)) if fps_proceso is not None else None)
    )
    base = {
        "input": "video",
        "video_fps": round(float(fps_video), 4),
        "integer_rate": rate,
        "source_frames": n_origen,
        "processed_frames": n_procesados,
        "fps_resolved": n,
    }
    if fps_proceso is None:
        return {
            **base,
            "specified": False,
            "fps_parameter": None,
            "description": (
                "--fps was not set. Processing ran frame by frame."
            ),
        }
    if offsets is None:
        return {
            **base,
            "specified": True,
            "fps_parameter": fps_proceso if fps_proceso in ("half", "third") else int(fps_proceso),
            "description": (
                f"--fps {etiqueta} was set, equal to or greater than the video "
                f"rate ({rate} fps). Processing ran frame by frame."
            ),
        }
    ejemplo = ", ".join(str(i + 1) for i in sorted(offsets))
    descripcion = (
        f"--fps {etiqueta} was set. The video is {fps_video:.3f} fps "
        f"(rate {rate}): each second uses frames {ejemplo} "
        f"({n_procesados} of {n_origen} frames)."
    )
    return {
        **base,
        "specified": True,
        "fps_parameter": fps_proceso if fps_proceso in ("half", "third") else int(fps_proceso),
        "offsets_in_second": sorted(offsets),
        "description": descripcion,
    }


def frame_origen(indice: int, indices_origen: list[int] | None) -> int:
    if not indices_origen or indice < 0 or indice >= len(indices_origen):
        return indice
    return int(indices_origen[indice])


def leer_frames(
    video: Path,
    max_frames: int | None,
    fps_proceso: int | None = None,
) -> tuple[list[np.ndarray], float, list[int]]:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise SystemExit(f"No se puede abrir {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        offsets, rate = offsets_muestreo(fps, fps_proceso)
        frames: list[np.ndarray] = []
        origen: list[int] = []
        indice = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if offsets is None or (indice % rate) in offsets:
                frames.append(frame)
                origen.append(indice)
            indice += 1
            if max_frames is not None and indice >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise SystemExit(f"{video} no tiene frames")
    return frames, fps, origen


def _es_entero(valor) -> bool:
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, bool)


def _entero_obligatorio(valor, nombre: str) -> int:
    if valor is None:
        raise SystemExit(f"--{nombre} is required when passing --frames")
    if not _es_entero(valor):
        raise SystemExit(f"--{nombre} must be an integer, got {type(valor).__name__}")
    n = int(valor)
    if n < 1:
        raise SystemExit(f"--{nombre} must be an integer >= 1")
    return n


def _entero_indice(valor, nombre: str) -> int:
    if not _es_entero(valor):
        raise SystemExit(f"{nombre} must be an integer, got {type(valor).__name__}")
    n = int(valor)
    if n < 0:
        raise SystemExit(f"{nombre} must be an integer >= 0")
    return n


def preparar_frames(
    frames,
    frames_origin=None,
    fps=None,
    width=None,
    height=None,
) -> tuple[list[np.ndarray], list[int], int, int, int]:
    """Valida arrays en memoria: BGR uint8, fps/ancho/alto enteros, índices de origen."""
    if not isinstance(frames, (list, tuple)) or not frames:
        raise SystemExit("--frames must be a non-empty list of numpy arrays")
    normalizados: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise SystemExit(f"--frames[{i}] must be HxW or HxWxC (1, 3 or 4 channels)")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        normalizados.append(np.ascontiguousarray(arr))
    alto_f, ancho_f = int(normalizados[0].shape[0]), int(normalizados[0].shape[1])
    ancho = _entero_obligatorio(ancho_f if width is None else width, "width")
    alto = _entero_obligatorio(alto_f if height is None else height, "height")
    if width is not None and ancho != ancho_f:
        raise SystemExit(f"--width ({ancho}) does not match frames ({ancho_f})")
    if height is not None and alto != alto_f:
        raise SystemExit(f"--height ({alto}) does not match frames ({alto_f})")
    for i, arr in enumerate(normalizados):
        if int(arr.shape[0]) != alto or int(arr.shape[1]) != ancho:
            raise SystemExit(
                f"--frames[{i}] is {arr.shape[1]}x{arr.shape[0]}, expected {ancho}x{alto}"
            )
    fps_entero = _entero_obligatorio(fps, "fps")
    if frames_origin is None:
        origen = list(range(len(normalizados)))
    else:
        if not isinstance(frames_origin, (list, tuple)):
            raise SystemExit("--frames-origin must be a list of integers")
        if len(frames_origin) != len(normalizados):
            raise SystemExit("--frames-origin length must match --frames")
        origen = [
            _entero_indice(idx, f"--frames-origin[{i}]")
            for i, idx in enumerate(frames_origin)
        ]
    return normalizados, origen, fps_entero, ancho, alto


def _fps_salida_frames(fps: float, indices_origen: list[int]) -> float:
    if len(indices_origen) >= 2:
        span = int(indices_origen[-1]) - int(indices_origen[0])
        if span > 0:
            return max(1.0, (len(indices_origen) - 1) * float(fps) / span)
    return float(fps)


def entrada_en_memoria(args, frames=None) -> bool:
    return frames is not None or getattr(args, "frames", None) is not None


def resolver_videos(args) -> list[Path]:
    if entrada_en_memoria(args):
        nombre = args.video if getattr(args, "video", None) is not None else Path("frames")
        return [Path(nombre)]
    if args.video is not None:
        if not args.video.is_file():
            raise SystemExit(f"No existe el vídeo {args.video}")
        return [args.video]
    return listar_videos(args.videos)


# --------------------------------------------------------------------------- #
# Detección alrededor de la muñeca
# --------------------------------------------------------------------------- #


def cargar_detectores(args):
    """Pose para las muñecas, seg y MobileSAM para el objeto, identificadores del YAML."""
    from detector_palma.deteccion import DetectorUltralytics, GeneradorMascarasSAM
    from detector_palma.personas import AnalizadorEscena

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    for ruta in (args.modelo_pose, args.modelo_objeto):
        if not Path(ruta).is_file():
            raise SystemExit(f"Falta el modelo {ruta}")
    catalogo, config_manos = cargar_catalogo(args.identificadores)
    args.catalogo = catalogo
    args.config_manos = config_manos
    activos = [item for item in catalogo if item.activo]
    estrategia = "mediapipe" if getattr(args, "mediapipe", False) else "pose"
    escena = AnalizadorEscena(
        estrategia=estrategia,
        modelo_pose=str(args.modelo_pose),
        dispositivo=dispositivo,
        padding_persona=args.padding,
        radio_mano=args.radio_mano,
        conf=args.conf_pose,
        imgsz=args.imgsz,
        conf_mano=config_manos.min_confidence,
    )
    detector = DetectorUltralytics(
        modelo=str(args.modelo_objeto),
        dispositivo=dispositivo,
        conf=args.conf_objeto,
        imgsz=args.imgsz,
    )
    sam = None
    if not args.sin_sam:
        ruta_sam = PESOS / "mobile_sam.pt"
        if ruta_sam.is_file():
            sam = GeneradorMascarasSAM(str(ruta_sam), dispositivo, args.imgsz, 0.2)
        else:
            print(f"  Sin MobileSAM: falta {ruta_sam}")
    identificadores = None
    if not args.sin_contenedores and activos:
        identificadores = DetectorIdentificadores(
            args.modelo_contenedores, detector, dispositivo,
            args.conf_contenedor, args.imgsz_contenedor, activos,
        )
        print(
            "  Identificadores activos: "
            + ", ".join(item.clave for item in activos)
        )
    elif not activos:
        print("  Ningún identificador activo en el YAML.")
    print(
        f"  Detectors on {dispositivo}: {estrategia} + seg + "
        f"{'MobileSAM' if sam else 'no SAM'} "
        f"(hand min_confidence={config_manos.min_confidence:.2f})"
    )
    return escena, detector, sam, identificadores


def caja_muneca(mano, padding: float, minimo: int, ancho: int, alto: int):
    """Ventana centrada en la palma, con padding relativo al tamaño del brazo."""
    palma = mano.palma or mano.punto
    radio = max(float(minimo), padding * max(mano.radio, mano.radio_palma * 3.0))
    cx, cy = float(palma[0]), float(palma[1])
    return (
        int(max(0, round(cx - radio))),
        int(max(0, round(cy - radio))),
        int(min(ancho, round(cx + radio))),
        int(min(alto, round(cy + radio))),
        radio,
    )


def _dentro_de_caja(centro, caja, margen: float = 0.0) -> bool:
    if centro is None or caja is None:
        return False
    x, y = centro
    x1, y1, x2, y2 = caja
    return x1 - margen <= x <= x2 + margen and y1 - margen <= y <= y2 + margen


def _radio_palma_objeto(mano, args) -> float:
    """Radio de aceptación: el objeto tiene que estar en la palma, no solo en el recorte."""
    return max(24.0, float(mano.radio) * args.factor_palma)


def _pegado_a_palma(inst: Instancia, mano, args) -> bool:
    """True si el centro o la caja tocan la palma, la muñeca o las yemas."""
    radio = _radio_palma_objeto(mano, args)
    palma = mano.palma or mano.punto
    puntos = [palma, mano.punto, *(mano.puntos_agarre or [])]
    cx, cy = float(inst.centro[0]), float(inst.centro[1])
    for px, py in puntos:
        if float(np.hypot(cx - float(px), cy - float(py))) <= radio:
            return True
        if _dentro_de_caja((float(px), float(py)), inst.caja, margen=10.0):
            return True
    return False


def _mano_lejos_del_torso(mano, contexto) -> bool:
    """True si la palma no está pegada al cuerpo (brazo extendido, no un bulto al costado)."""
    if contexto is None or not getattr(contexto, "personas", None):
        return True
    persona = None
    tid = getattr(mano, "persona_id", -1)
    if tid >= 0:
        persona = next((p for p in contexto.personas if getattr(p, "track_id", -1) == tid), None)
    if persona is None:
        persona = max(contexto.personas, key=lambda p: p.altura)
    palma = mano.palma or mano.punto
    pc = _centro_caja(persona.caja)
    return float(np.hypot(palma[0] - pc[0], palma[1] - pc[1])) > 0.30 * max(persona.altura, 40.0)


def candidatos_en_munecas(frame, contexto, detector, sam, args) -> list[Instancia]:
    """Objetos compactos pegados a cada palma. El recorte solo limita la búsqueda."""
    alto, ancho = frame.shape[:2]
    area_persona = 0.0
    if contexto.personas:
        persona = max(contexto.personas, key=lambda p: p.altura)
        x1, y1, x2, y2 = persona.caja
        area_persona = float(max(1, x2 - x1) * max(1, y2 - y1))
    area_max = args.area_max_px
    if area_persona:
        area_max = min(area_max, max(float(args.area_min), args.area_max_persona * area_persona))
    candidatos: list[Instancia] = []
    umbral_mano = float(getattr(contexto, "conf_dibujo", 0.20) or 0.20)
    for mano in contexto.manos:
        if float(mano.confianza) < umbral_mano:
            continue
        if getattr(mano, "abierta", False) and _mano_lejos_del_torso(mano, contexto):
            continue
        x1, y1, x2, y2, _radio = caja_muneca(
            mano, args.padding_muneca, args.crop_min, ancho, alto
        )
        if (x2 - x1) < 48 or (y2 - y1) < 48:
            continue
        locales = detector.detectar_en_caja(frame, (x1, y1, x2, y2), args.conf_objeto_crop)
        if sam is not None:
            locales.extend(_sam_en_palma(sam, detector, frame, mano, (x1, y1, x2, y2), args))
        for inst in locales:
            if inst.es_persona or inst.area < args.area_min or inst.area > area_max:
                continue
            if not _dentro_de_caja(inst.centro, (x1, y1, x2, y2)):
                continue
            if not _pegado_a_palma(inst, mano, args):
                continue
            if not _forma_de_objeto(inst, args):
                continue
            if es_voluminoso(inst, contexto) or _cubre_torso(inst, contexto):
                continue
            if _parece_brazo_o_suelo(inst, contexto):
                continue
            if mascara_es_la_mano(inst, mano):
                continue
            candidatos.append(inst)
    return _sin_duplicados(candidatos)


def _forma_de_objeto(inst: Instancia, args) -> bool:
    """Descarta brazos y baldas: el objeto sostenido es compacto y poco alargado."""
    x1, y1, x2, y2 = inst.caja
    ancho, alto = max(1, x2 - x1), max(1, y2 - y1)
    if max(ancho, alto) / min(ancho, alto) > args.relacion_max:
        return False
    return inst.area / float(ancho * alto) >= args.relleno_min


def _sam_en_palma(sam, detector, frame, mano, caja, args) -> list[Instancia]:
    """MobileSAM en la palma: solo la mancha compacta más cercana, no la estantería."""
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = caja
    recorte = frame[y1:y2, x1:x2]
    if recorte.size == 0:
        return []
    palma = mano.palma or mano.punto
    puntos = [(palma[0] - x1, palma[1] - y1)]
    for punto in list(mano.puntos_agarre or [])[:3]:
        puntos.append((punto[0] - x1, punto[1] - y1))
    area_max = min(
        float(args.area_max_px),
        max(120.0, float(np.pi * (mano.radio_palma * 2.6) ** 2)),
    )
    mejor, mejor_dist = None, 1e9
    for inst in sam.segmentar(recorte, puntos=puntos):
        proyectada = detector.reproyectar_desde_recorte(inst, caja, ancho, alto)
        if proyectada is None:
            continue
        if proyectada.area < args.area_min or proyectada.area > area_max:
            continue
        if not _pegado_a_palma(proyectada, mano, args) or not _forma_de_objeto(proyectada, args):
            continue
        dist = float(np.hypot(proyectada.centro[0] - palma[0], proyectada.centro[1] - palma[1]))
        if dist < mejor_dist:
            mejor, mejor_dist = proyectada, dist
    if mejor is None:
        return []
    mejor.etiqueta = "objeto"
    return [mejor]


def _solapan(a: np.ndarray, b: np.ndarray, umbral: float = 0.5) -> bool:
    """Dos máscaras son el mismo objeto si una está contenida en la otra."""
    interseccion = int(np.count_nonzero(a & b))
    if interseccion == 0:
        return False
    menor = min(int(a.sum()), int(b.sum()))
    return interseccion / max(1, menor) >= umbral


def _sin_duplicados(candidatos: list[Instancia], iou_max: float = 0.55) -> list[Instancia]:
    """El mismo objeto lo pueden ver el modelo de segmentación y SAM a la vez."""
    ordenados = sorted(candidatos, key=lambda i: i.area)
    salida: list[Instancia] = []
    for inst in ordenados:
        if any(
            iou_cajas(inst.caja, ya.caja) > iou_max or _solapan(inst.mascara, ya.mascara)
            for ya in salida
        ):
            continue
        salida.append(inst)
    return salida


# --------------------------------------------------------------------------- #
# Agarre: izquierda, derecha o ambas
# --------------------------------------------------------------------------- #


def puntuar_manos(mascara: np.ndarray, caja, contexto, args) -> dict:
    """Misma puntuación de palma/dedos que la versión anterior de SAM 3."""
    inst = Instancia(mascara=mascara, caja=caja)
    if es_voluminoso(inst, contexto):
        return {"mano": None, "score": 0.0, "scores": {}}
    score, lado = puntuar_agarre(inst, contexto)
    detalle = inst.extra.get("detalle_agarre") or {}
    if lado == "ambas":
        contacto = max((d.get("contacto") or 0.0) for d in detalle.values()) if detalle else 0.0
    else:
        contacto = float((detalle.get(lado) or {}).get("contacto") or 0.0)
    mano_lado = next((m for m in contexto.manos if m.lado == lado), None)
    es_mano = bool(mano_lado is not None and mascara_es_la_mano(inst, mano_lado))
    en_mano = bool(lado) and score >= args.umbral_agarre and contacto >= args.contacto_min and not es_mano
    return {
        "mano": lado if en_mano else None,
        "score": float(score),
        "scores": inst.scores_mano or {},
    }


# --------------------------------------------------------------------------- #
# Identificadores de escena (YAML + Ultralytics / YOLOE)
# --------------------------------------------------------------------------- #


class DetectorIdentificadores:
    """Detecta en el frame completo las clases activas del YAML.

    Las que tienen nombre COCO salen de YOLO26-seg con ``classes=[id]``.
    Las demás (y un refuerzo por texto) salen de YOLOE.
    """

    def __init__(self, pesos_yoloe, yolo, dispositivo: str, conf: float, imgsz: int,
                 identificadores: list[Identificador]):
        self.identificadores = identificadores
        self.dispositivo = dispositivo
        self.conf = conf
        self.imgsz = imgsz
        self.yolo = yolo
        nombres = getattr(yolo, "nombres_clase", {}) or {}
        self.por_id: dict[int, Identificador] = {}
        self.por_texto: dict[str, Identificador] = {}
        ids: list[int] = []
        textos: list[str] = []
        for ident in identificadores:
            if ident.cuerpo:
                print(f"    {ident.clave}: ROI from pose (hips)")
                continue
            uid = _id_ultralytics(nombres, ident)
            if uid is not None:
                self.por_id[uid] = ident
                ids.append(uid)
                print(
                    f"    {ident.clave}: Ultralytics id {uid} "
                    f"({nombres.get(uid, ident.nombre or '?')})"
                )
            for prompt in ident.prompts:
                clave = prompt.lower()
                self.por_texto[clave] = ident
                textos.append(prompt)
            if ident.nombre:
                self.por_texto[ident.nombre.lower()] = ident
        self.ids_yolo = sorted(set(ids))
        self.textos = textos
        self.yoloe = None
        if textos:
            ruta_yoloe = Path(pesos_yoloe)
            if not ruta_yoloe.is_file():
                print(f"  Sin YOLOE: falta {pesos_yoloe}. Solo clases COCO del YAML.")
            else:
                try:
                    from ultralytics import YOLOE

                    self.yoloe = YOLOE(str(ruta_yoloe))
                    self.yoloe.set_classes(textos, self.yoloe.get_text_pe(textos))
                except Exception as exc:
                    self.yoloe = None
                    _aviso(f"No se pudo cargar YOLOE ({exc}). Solo clases COCO del YAML.")

    def detectar(self, frame) -> list[dict]:
        salida: list[dict] = []
        if self.yolo is not None and self.ids_yolo:
            try:
                detectadas = self.yolo.detectar_clases(frame, self.ids_yolo, self.conf)
            except Exception as exc:
                _aviso(f"YOLO de identificadores falló: {exc}")
                detectadas = []
            for inst in detectadas:
                ident = self.por_id.get(int(inst.extra.get("clase_id", -1)))
                if ident is None:
                    ident = self.por_texto.get((inst.etiqueta or "").lower())
                if ident is None:
                    continue
                salida.append(_det_identificador(ident, inst.etiqueta, inst.score, inst.caja))
        if self.yoloe is not None:
            salida.extend(self._detectar_yoloe(frame))
        alto, ancho = frame.shape[:2]
        salida = [d for d in salida if not _deteccion_identificador_invalida(d, ancho, alto)]
        return _fusionar_contenedores(salida)

    def _detectar_yoloe(self, frame) -> list[dict]:
        alto, ancho = frame.shape[:2]
        try:
            resultados = self.yoloe.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.dispositivo,
                retina_masks=True,
                verbose=False,
            )
        except Exception as exc:
            _aviso(f"YOLOE falló ({exc}). Se desactiva para el resto del clip.")
            self.yoloe = None
            _vaciar_cuda()
            return []
        if not resultados or resultados[0].boxes is None:
            return []
        resultado = resultados[0]
        nombres = resultado.names or getattr(self.yoloe, "names", {}) or {}
        salida: list[dict] = []
        for caja in resultado.boxes:
            try:
                cls_id = int(caja.cls)
                texto = str(nombres.get(cls_id, nombres.get(str(cls_id), cls_id))).lower()
            except Exception:
                continue
            ident = self.por_texto.get(texto)
            if ident is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho, x2), min(alto, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                score = float(caja.conf)
            except Exception:
                score = 0.0
            salida.append(
                _det_identificador(ident, texto, score, (x1, y1, x2, y2))
            )
        return salida

    def detectar_cuerpo(self, contexto) -> list[dict]:
        """Bolsillos y similares: no hay clase COCO, se estiman con la cadera."""
        if contexto is None:
            return []
        salida: list[dict] = []
        for ident in self.identificadores:
            if not ident.cuerpo or not ident.activo:
                continue
            if ident.clave in ("bolsillo", "pocket"):
                salida.extend(_cajas_bolsillo(contexto, ident))
        return salida


def _det_identificador(ident: Identificador, texto: str, score: float, caja, lado: str | None = None) -> dict:
    det = {
        "categoria": ident.clave,
        "texto": texto,
        "score": float(score),
        "caja": caja,
        "destino": ident.destino,
        "accesorio": ident.accesorio,
        "cuerpo": ident.cuerpo,
    }
    if lado:
        det["lado"] = lado
    return det


def _clip_caja(caja, ancho: int, alto: int):
    x1, y1, x2, y2 = caja
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(ancho, int(round(x2)))
    y2 = min(alto, int(round(y2)))
    if x2 - x1 < 12 or y2 - y1 < 16:
        return None
    return (x1, y1, x2, y2)


def _cajas_bolsillo(contexto, ident: Identificador) -> list[dict]:
    """ROI pequeño de bolsillo: del lado de la cintura hacia el muslo, ~tamaño de palma."""
    ancho, alto = int(contexto.ancho), int(contexto.alto)
    umbral = min(0.15, float(getattr(contexto, "conf_dibujo", 0.20) or 0.20))
    radio_mano = None
    for mano in getattr(contexto, "manos", []) or []:
        r = float(getattr(mano, "radio_palma", None) or mano.radio or 0.0)
        if r > 0:
            radio_mano = r if radio_mano is None else max(radio_mano, r)
    candidatas = []
    for persona in getattr(contexto, "personas", []) or []:
        altura = max(float(persona.altura), 1.0)
        if altura < 90:
            continue
        tid = getattr(persona, "track_id", -1)
        if not any(
            m.persona_id == tid and m.lado in ("izquierda", "derecha")
            for m in (getattr(contexto, "manos", None) or [])
        ):
            continue
        px1, py1, px2, py2 = persona.caja
        if altura < 0.32 * alto and py1 < 0.12 * alto:
            continue
        candidatas.append(persona)
    if candidatas:
        max_h = max(p.altura for p in candidatas)
        candidatas = [p for p in candidatas if p.altura >= 0.55 * max_h]
    salida: list[dict] = []
    for persona in candidatas:
        altura = max(float(persona.altura), 1.0)
        px1, py1, px2, py2 = persona.caja
        kp = persona.keypoints
        tiene_cadera = kp is not None and (
            (len(kp) > CADERA_IZQ and float(kp[CADERA_IZQ][2]) >= umbral)
            or (len(kp) > CADERA_DER and float(kp[CADERA_DER][2]) >= umbral)
        )
        if py2 >= alto - 10 and not tiene_cadera:
            continue
        torso_cx = 0.5 * (px1 + px2)
        candidatos = []
        for lado, idx in (("izquierda", CADERA_IZQ), ("derecha", CADERA_DER)):
            hx = hy = conf = None
            if kp is not None and len(kp) > idx:
                hx, hy, conf = float(kp[idx][0]), float(kp[idx][1]), float(kp[idx][2])
                if conf < umbral:
                    hx = hy = None
            if hx is None:
                # De cara a cámara: la cadera izquierda de la persona está a la derecha de la imagen.
                hx = px1 + 0.72 * (px2 - px1) if lado == "izquierda" else px1 + 0.28 * (px2 - px1)
                hy = py2 - 0.38 * altura
            # Pose a veces deja la cadera en el pecho: el bolsillo real está más abajo.
            if hy < py1 + 0.45 * altura:
                hy = py1 + 0.70 * altura
            candidatos.append((lado, hx, hy))
        for lado, hx, hy in candidatos:
            if radio_mano:
                w = max(16.0, min(2.1 * radio_mano, 0.10 * altura))
                h = max(22.0, min(2.6 * radio_mano, 0.14 * altura))
            else:
                w = max(16.0, 0.075 * altura)
                h = max(22.0, 0.11 * altura)
            outward = 0.03 * altura if hx >= torso_cx else -0.03 * altura
            cx = hx + outward
            # Empieza en la cintura y baja al muslo; no cubre el torso.
            y1 = hy - 0.015 * altura
            y2 = y1 + h
            if hy > alto - 12:
                continue
            caja = _clip_caja((cx - w / 2.0, y1, cx + w / 2.0, y2), ancho, alto)
            if caja is None:
                continue
            texto = "left pocket" if lado == "izquierda" else "right pocket"
            salida.append(_det_identificador(ident, texto, 0.6, caja, lado=lado))
    return salida


DetectorContenedores = DetectorIdentificadores


def _area_caja(caja) -> float:
    if caja is None:
        return 0.0
    x1, y1, x2, y2 = caja
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def _centro_caja(caja) -> tuple[float, float]:
    x1, y1, x2, y2 = caja
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diagonal(caja) -> float:
    x1, y1, x2, y2 = caja
    return float(np.hypot(max(1, x2 - x1), max(1, y2 - y1)))


def _cajas_cercanas(caja_a, caja_b, holgura: float = 48.0) -> bool:
    """YOLO y YOLOE a menudo parten el mismo bolso en dos cajas que apenas se solapan."""
    if caja_a is None or caja_b is None:
        return False
    if iou_cajas(caja_a, caja_b) >= 0.15:
        return True
    xa1, ya1, xa2, ya2 = caja_a
    xb1, yb1, xb2, yb2 = caja_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    if ix2 > ix1 and iy2 > iy1:
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(1, xa2 - xa1) * max(1, ya2 - ya1))
        area_b = float(max(1, xb2 - xb1) * max(1, yb2 - yb1))
        if inter / min(area_a, area_b) >= 0.35:
            return True
    ca, cb = _centro_caja(caja_a), _centro_caja(caja_b)
    dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    return dist <= 0.28 * (_diagonal(caja_a) + _diagonal(caja_b)) + holgura


def _fusionar_contenedores(detecciones: list[dict], iou_max: float = 0.55) -> list[dict]:
    """Varios prompts ven el mismo objeto (cart y trolley): se queda el de más score."""
    salida: list[dict] = []
    for det in sorted(detecciones, key=lambda d: -d["score"]):
        if any(
            ya["categoria"] == det["categoria"]
            and (
                iou_cajas(det["caja"], ya["caja"]) > iou_max
                or (
                    (det.get("accesorio") or ya.get("accesorio"))
                    and _cajas_cercanas(det["caja"], ya["caja"])
                )
            )
            for ya in salida
        ):
            continue
        salida.append(det)
    return salida


def _misma_pista(
    caja_a, caja_b, iou_min: float, factor_centro: float,
    hueco: int = 1, accesorio: bool = False,
) -> bool:
    """El contenedor se mueve con la persona: el IoU solo no basta para reconocerlo."""
    if iou_cajas(caja_a, caja_b) >= iou_min:
        return True
    ca, cb = _centro_caja(caja_a), _centro_caja(caja_b)
    distancia = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    base = factor_centro * 0.5 * (_diagonal(caja_a) + _diagonal(caja_b))
    paso = (90.0 if accesorio else 24.0) * max(1, hueco)
    return distancia <= base + paso


def _indice_persona(caja, contexto, factor: float = 2.4):
    if contexto is None or not getattr(contexto, "personas", None):
        return None
    c = _centro_caja(caja)
    mejor, mejor_d = None, 1e9
    for i, persona in enumerate(contexto.personas):
        pc = _centro_caja(persona.caja_padded)
        dist = float(np.hypot(c[0] - pc[0], c[1] - pc[1]))
        radio = factor * max(persona.altura, 60.0)
        if dist <= radio or iou_cajas(caja, persona.caja_padded) > 0:
            if dist < mejor_d:
                mejor, mejor_d = i, dist
    return mejor


def _cerca_de_persona(caja, contexto, factor: float = 2.4) -> bool:
    return _indice_persona(caja, contexto, factor) is not None


def _caderas_persona(persona) -> list[tuple[str, float, float]]:
    """Caderas visibles o, si faltan, una estimación a ambos lados de la cintura."""
    if persona is None:
        return []
    ph = max(float(persona.altura), 1.0)
    pw = max(1.0, float(persona.caja[2] - persona.caja[0]))
    halladas: dict[str, tuple[float, float]] = {}
    kp = getattr(persona, "keypoints", None)
    if kp is not None and len(kp) > CADERA_DER:
        for idx, lado in ((CADERA_IZQ, "izquierda"), (CADERA_DER, "derecha")):
            if float(kp[idx][2]) >= 0.12:
                halladas[lado] = (float(kp[idx][0]), float(kp[idx][1]))
    estim = {
        "izquierda": (persona.caja[0] + 0.72 * pw, persona.caja[1] + 0.62 * ph),
        "derecha": (persona.caja[0] + 0.28 * pw, persona.caja[1] + 0.62 * ph),
    }
    return [(lado, *halladas.get(lado, estim[lado])) for lado in ("izquierda", "derecha")]


def _accesorio_en_persona(caja, contexto, holgura: float = 0.35) -> bool:
    """True si el bolso/mochila está sobre la persona, no un ROI lejano congelado."""
    if caja is None or contexto is None or not getattr(contexto, "personas", None):
        return False
    for persona in contexto.personas:
        if iou_cajas(caja, persona.caja) >= 0.02:
            return True
        if _fraccion_caja_dentro(caja, persona.caja_padded) >= 0.25:
            return True
        cc, pc = _centro_caja(caja), _centro_caja(persona.caja)
        ancho_p = max(1.0, float(persona.caja[2] - persona.caja[0]))
        alto_p = max(1.0, float(persona.altura))
        if abs(cc[0] - pc[0]) <= holgura * ancho_p + 24.0 and (
            persona.caja[1] - 24.0 <= cc[1] <= persona.caja[1] + 0.90 * alto_p
        ):
            return True
        for _lado, hx, hy in _caderas_persona(persona):
            if abs(cc[0] - hx) <= 0.32 * ancho_p + 16.0 and abs(cc[1] - hy) <= 0.24 * alto_p + 16.0:
                return True
    return False


def _persona_sale_de_escena(contexto, margen: int = 10) -> bool:
    if contexto is None:
        return True
    if not getattr(contexto, "personas", None):
        return True
    persona = max(contexto.personas, key=lambda p: p.altura)
    x1, y1, x2, y2 = persona.caja
    ancho, alto = int(contexto.ancho), int(contexto.alto)
    alto_p = max(1.0, float(y2 - y1))
    margen_lado = max(margen, int(0.045 * ancho))
    sale_lado = x1 <= margen_lado or x2 >= ancho - margen_lado
    sale_abajo = y2 >= alto - margen and alto_p < 0.82 * alto
    sale_arriba = y1 <= margen and alto_p < 0.45 * alto
    return sale_lado or sale_abajo or sale_arriba


def _de_espaldas(contexto) -> bool:
    """Sin cara visible: de espaldas, no se puede afirmar un depósito en el bolsillo."""
    if contexto is None or not getattr(contexto, "personas", None):
        return True
    persona = max(contexto.personas, key=lambda p: p.altura)
    kp = persona.keypoints
    if kp is None or len(kp) < 1:
        return True
    nariz = float(kp[0][2])
    ojos = 0.0
    orejas = 0.0
    if len(kp) > 2:
        ojos = max(float(kp[1][2]), float(kp[2][2]))
    if len(kp) > 4:
        orejas = max(float(kp[3][2]), float(kp[4][2]))
    if nariz < 0.18 and ojos < 0.18:
        return True
    return orejas >= 0.30 and nariz < 0.22 and ojos < 0.22


def _deteccion_identificador_invalida(det: dict, ancho: int, alto: int) -> bool:
    """Cajas de refrescos en la esquina, móviles minúsculos o bolsos fantasma."""
    caja = det.get("caja")
    if caja is None:
        return True
    x1, y1, x2, y2 = caja
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    area = float(w * h)
    frame_area = max(1.0, float(ancho * alto))
    borde = x1 <= 6 or y1 <= 6 or x2 >= ancho - 6 or y2 >= alto - 6
    cat = det.get("categoria")
    score = float(det.get("score") or 0.0)
    if cat in ("cesto", "bolsa"):
        if score < 0.42 and borde:
            return True
        if y1 > 0.78 * alto and (x1 < 0.20 * ancho or x2 > 0.80 * ancho):
            return True
        if borde and area < 0.055 * frame_area:
            return True
    if cat == "celular":
        if score < 0.32:
            return True
        if area > 0.025 * frame_area:
            return True
        if area < 40:
            return True
    if cat in ("bolso", "mochila"):
        if score < 0.30:
            return True
        if y2 >= alto - 6 and (x1 <= 8 or x2 >= ancho - 8) and area < 0.02 * frame_area:
            return True
    return False


def _contexto_en(contextos, indice):
    if not contextos or indice < 0 or indice >= len(contextos):
        return None
    return contextos[indice]


def seguir_contenedores(frames, detector, cada: int, iou_min: float = 0.3,
                        factor_centro: float = 0.8, contextos=None) -> list[dict]:
    """Rastrea los contenedores por IoU y cercanía para tener un id estable y su último ROI."""
    if detector is None:
        return []
    cada = max(1, int(cada))
    indices = set(range(0, len(frames), cada))
    indices.add(len(frames) - 1)
    if contextos is not None:
        for i, ctx in enumerate(contextos):
            if getattr(ctx, "personas", None):
                indices.add(i)
    pistas: list[dict] = []
    for indice in sorted(indices):
        contexto = _contexto_en(contextos, indice)
        try:
            detecciones = detector.detectar(frames[indice])
        except Exception as exc:
            _aviso(f"frame {indice}: identificadores fallaron ({exc})")
            detecciones = []
        if contexto is not None and hasattr(detector, "detectar_cuerpo"):
            try:
                detecciones.extend(detector.detectar_cuerpo(contexto))
            except Exception as exc:
                _aviso(f"frame {indice}: bolsillo/pose falló ({exc})")
        for det in detecciones:
            mejor, mejor_s = None, -1.0
            misma_caja = False
            for pista in pistas:
                if pista["categoria"] != det["categoria"]:
                    continue
                accesorio = bool(
                    det.get("accesorio")
                    or pista.get("accesorio")
                    or det.get("cuerpo")
                    or pista.get("cuerpo")
                )
                if det.get("lado") and pista.get("lado") and det["lado"] != pista["lado"]:
                    continue
                if indice == pista["ultimo_frame"]:
                    if accesorio and _cajas_cercanas(
                        det["caja"], pista["cajas"].get(indice, pista["caja"])
                    ):
                        if det["score"] > (pista["scores"][-1] if pista["scores"] else -1.0):
                            pista["caja"] = det["caja"]
                            pista["cajas"][indice] = det["caja"]
                            if pista["scores"]:
                                pista["scores"][-1] = det["score"]
                            else:
                                pista["scores"].append(det["score"])
                        misma_caja = True
                    continue
                hueco = indice - pista["ultimo_frame"]
                if not _misma_pista(
                    det["caja"], pista["caja"], iou_min, factor_centro, hueco, accesorio
                ):
                    if not (
                        accesorio
                        and hueco > 0
                        and _indice_persona(det["caja"], contexto) is not None
                        and _indice_persona(
                            pista["caja"], _contexto_en(contextos, pista["ultimo_frame"])
                        ) == _indice_persona(det["caja"], contexto)
                    ):
                        continue
                iou = iou_cajas(det["caja"], pista["caja"])
                score = iou + 1.0 / (1.0 + hueco)
                if score > mejor_s:
                    mejor, mejor_s = pista, score
            if misma_caja:
                continue
            if mejor is not None:
                mejor["caja"] = det["caja"]
                mejor["ultimo_frame"] = indice
                mejor["scores"].append(det["score"])
                mejor["cajas"][indice] = det["caja"]
                mejor["accesorio"] = mejor.get("accesorio", False) or bool(det.get("accesorio"))
                mejor["cuerpo"] = mejor.get("cuerpo", False) or bool(det.get("cuerpo"))
                if det.get("lado"):
                    mejor["lado"] = det["lado"]
            else:
                pistas.append(
                    {
                        "id": f"{det['categoria']}_{len(pistas) + 1}",
                        "categoria": det["categoria"],
                        "texto": det["texto"],
                        "caja": det["caja"],
                        "cajas": {indice: det["caja"]},
                        "scores": [det["score"]],
                        "primer_frame": indice,
                        "ultimo_frame": indice,
                        "destino": bool(det.get("destino", False)),
                        "accesorio": bool(det.get("accesorio", False)),
                        "cuerpo": bool(det.get("cuerpo", False)),
                        "lado": det.get("lado"),
                    }
                )
    pistas = _coser_accesorios(pistas, contextos)
    pistas = _unir_contenedores(pistas)
    pistas = _limpiar_identificadores(pistas, contextos, frames[0].shape if frames else (0, 0))
    pistas = _estabilizar_muebles(pistas)
    pistas = _anclar_accesorio_a_persona(pistas, contextos)
    pistas = _anclar_celular_a_mano(pistas, contextos)
    pistas = _cortar_mueble_si_atraviesa(pistas, contextos)
    for numero, pista in enumerate(pistas, start=1):
        pista["id"] = f"{pista['categoria']}_{numero}"
        pista["score_medio"] = round(float(np.mean(pista["scores"])), 4)
        pista["score_max"] = round(float(np.max(pista["scores"])), 4)
        pista["frames_visto"] = len(pista["cajas"])
    print(
        f"  Identificadores: "
        + (", ".join(f"{p['id']} ({p['score_max']:.2f})" for p in pistas) or "ninguno")
    )
    return pistas


def _celular_junto_a_persona(pista: dict, contextos) -> bool:
    """El móvil junto al torso o al borde del bbox (persona recortada por abajo)."""
    if not contextos:
        return False
    votos, total = 0, 0
    for frame, caja in (pista.get("cajas") or {}).items():
        ctx = _contexto_en(contextos, int(frame))
        if ctx is None or not getattr(ctx, "personas", None):
            continue
        total += 1
        persona = max(ctx.personas, key=lambda p: p.altura)
        if iou_cajas(caja, persona.caja) >= 0.01:
            votos += 1
            continue
        cc, pc = _centro_caja(caja), _centro_caja(persona.caja)
        ancho_p = max(1.0, float(persona.caja[2] - persona.caja[0]))
        alto_p = max(1.0, float(persona.altura))
        if abs(cc[0] - pc[0]) <= 0.85 * ancho_p + 28.0 and abs(cc[1] - pc[1]) <= 0.85 * alto_p + 36.0:
            votos += 1
    return total > 0 and votos / total >= 0.4


def _celular_junto_a_mano(pista: dict, contextos) -> bool:
    if not contextos:
        return False
    votos, total = 0, 0
    for frame, caja in (pista.get("cajas") or {}).items():
        ctx = _contexto_en(contextos, int(frame))
        if ctx is None or not getattr(ctx, "manos", None):
            continue
        total += 1
        if any(_mano_cerca_caja(m, caja, margen=36.0) for m in ctx.manos):
            votos += 1
    return total > 0 and votos / total >= 0.5


def _limpiar_identificadores(pistas: list[dict], contextos, forma) -> list[dict]:
    alto, ancho = (int(forma[0]), int(forma[1])) if forma is not None and len(forma) >= 2 else (0, 0)
    vivos = []
    for pista in pistas:
        cat = pista.get("categoria")
        score = max(pista.get("scores") or [0.0])
        n = len(pista.get("cajas") or {})
        if cat == "celular":
            if score < 0.32:
                continue
            if n < 2 and score < 0.40:
                continue
            if not _celular_junto_a_mano(pista, contextos) and not _celular_junto_a_persona(
                pista, contextos
            ):
                continue
        if cat == "carrito" and n < 1:
            continue
        if cat == "carrito" and n < 2 and score < 0.22:
            continue
        if cat == "cesto" and n < 2 and score < 0.45:
            continue
        if cat in ("cesto", "bolsa") and ancho and alto:
            caja = pista.get("caja")
            if caja is not None and _deteccion_identificador_invalida(
                {"categoria": cat, "score": score, "caja": caja}, ancho, alto
            ):
                continue
        if pista.get("accesorio") and contextos:
            en_persona = False
            for frame, caja in (pista.get("cajas") or {}).items():
                if _accesorio_en_persona(caja, _contexto_en(contextos, int(frame))):
                    en_persona = True
                    break
            if not en_persona and (score < 0.40 or n < 3):
                continue
        vivos.append(pista)
    return vivos


def _filtrar_celular_fantasma(contenedores: list[dict], objetos: list[Objeto]) -> list[dict]:
    """Un móvil débil pegado al objeto de la estantería no es un segundo artículo."""
    vivos = []
    for pista in contenedores:
        if pista.get("categoria") != "celular":
            vivos.append(pista)
            continue
        score = max(pista.get("scores") or [0.0])
        if score < 0.32:
            print(f"  {pista['id']} se descarta: móvil poco fiable ({score:.2f})")
            continue
        solapa = False
        for frame, caja in (pista.get("cajas") or {}).items():
            for objeto in objetos:
                vista = objeto.vistas.get(int(frame))
                if vista is None:
                    continue
                if iou_cajas(caja, vista.caja) >= 0.12 or _fraccion_caja_dentro(caja, vista.caja) >= 0.3:
                    solapa = True
                    break
            if solapa:
                break
        if solapa:
            vivos.append(pista)
            continue
        vivos.append(pista)
    return vivos


def _lado_cadera_accesorio(pista: dict, contextos) -> str | None:
    """Lado de cintura de las primeras detecciones; no se cambia si el ROI deriva."""
    votos = []
    for frame, caja in sorted((pista.get("cajas") or {}).items()):
        ctx = _contexto_en(contextos, int(frame))
        if ctx is None or not getattr(ctx, "personas", None) or not _accesorio_en_persona(caja, ctx):
            continue
        persona = max(ctx.personas, key=lambda p: p.altura)
        cc = _centro_caja(caja)
        caderas = _caderas_persona(persona)
        if not caderas:
            continue
        lado, _, _ = min(caderas, key=lambda p: (p[1] - cc[0]) ** 2 + (p[2] - cc[1]) ** 2)
        votos.append(lado)
    if not votos:
        return None
    primeros = votos[: max(3, len(votos) // 2)]
    return "izquierda" if primeros.count("izquierda") >= primeros.count("derecha") else "derecha"


def _caja_en_cadera(persona, ancho: float, alto: float, lado: str | None, marco_w: int, marco_h: int):
    caderas = _caderas_persona(persona)
    if not caderas:
        return None
    chosen = next((c for c in caderas if lado and c[0] == lado), None) or caderas[0]
    _, hx, hy = chosen
    ph = max(float(persona.altura), 1.0)
    pc = _centro_caja(persona.caja)
    nx = hx + 0.02 * (1 if hx >= pc[0] else -1) * ph
    ny = hy + 0.06 * ph
    return _clip_caja((nx - ancho / 2.0, ny - alto / 2.0, nx + ancho / 2.0, ny + alto / 2.0), marco_w, marco_h)


def _es_bolso_cintura(pista: dict, contextos) -> bool:
    """Riñonera / bolso a la cadera, no el de hombro que cuelga al caminar."""
    rels = []
    for frame, caja in (pista.get("cajas") or {}).items():
        ctx = _contexto_en(contextos, int(frame))
        if ctx is None or not getattr(ctx, "personas", None) or not _accesorio_en_persona(caja, ctx):
            continue
        persona = max(ctx.personas, key=lambda p: p.altura)
        cy = 0.5 * (caja[1] + caja[3])
        rel = (cy - float(persona.caja[1])) / max(float(persona.altura), 1.0)
        w, h = float(caja[2] - caja[0]), float(caja[3] - caja[1])
        if w < 0.55 * h or h > 0.40 * max(float(persona.altura), 1.0):
            continue
        rels.append(rel)
    if len(rels) < 3:
        return False
    med = float(np.median(rels))
    return 0.48 <= med <= 0.82


def _anclar_accesorio_a_persona(pistas: list[dict], contextos) -> list[dict]:
    """El bolso/mochila se mueve con la persona; no se deja el ROI congelado en el pasillo."""
    if not contextos:
        return pistas
    for pista in pistas:
        if not pista.get("accesorio"):
            continue
        cajas = pista.get("cajas") or {}
        if not cajas:
            continue
        offsets = []
        on_person = []
        for frame, caja in cajas.items():
            ctx = _contexto_en(contextos, int(frame))
            if ctx is None or caja is None:
                continue
            idx = _indice_persona(caja, ctx, 1.5)
            if idx is None or not _accesorio_en_persona(caja, ctx):
                continue
            persona = ctx.personas[idx]
            pc = _centro_caja(persona.caja)
            cc = _centro_caja(caja)
            item = (
                int(frame),
                cc[0] - pc[0],
                cc[1] - pc[1],
                float(caja[2] - caja[0]),
                float(caja[3] - caja[1]),
            )
            offsets.append(item)
            on_person.append(item)
        if not offsets:
            malas = [
                f for f, caja in list(cajas.items())
                if _contexto_en(contextos, int(f)) is not None
                and not _accesorio_en_persona(caja, _contexto_en(contextos, int(f)))
            ]
            for f in malas:
                cajas.pop(f, None)
            if cajas:
                uf = max(cajas)
                pista["ultimo_frame"] = uf
                pista["caja"] = cajas[uf]
                pista["primer_frame"] = min(cajas)
            continue
        offsets.sort()
        w_med = float(np.median([item[3] for item in on_person]))
        h_med = float(np.median([item[4] for item in on_person]))
        lado = _lado_cadera_accesorio(pista, contextos) if _es_bolso_cintura(pista, contextos) else None
        ultimo_bueno = offsets[-1][0]
        for i, ctx in enumerate(contextos):
            if not getattr(ctx, "personas", None):
                continue
            persona = max(ctx.personas, key=lambda p: p.altura)
            actual = cajas.get(i)
            en_borde = False
            lejos = False
            otra_cadera = False
            if actual is not None:
                en_borde = (
                    actual[3] >= int(ctx.alto) - 8
                    and _area_caja(actual) < 0.04 * float(ctx.ancho) * float(ctx.alto)
                )
                lejos = not _accesorio_en_persona(actual, ctx)
                if pista.get("categoria") == "mochila" and getattr(ctx, "personas", None):
                    pc = _centro_caja(persona.caja)
                    cc = _centro_caja(actual)
                    ph = max(float(persona.altura), 1.0)
                    pw = max(1.0, float(persona.caja[2] - persona.caja[0]))
                    if abs(cc[1] - (persona.caja[1] + 0.22 * ph)) > 0.22 * ph:
                        lejos = True
                    if abs(cc[0] - pc[0]) > 0.42 * pw:
                        lejos = True
                if lado and pista.get("categoria") in ("bolso", "mochila"):
                    cc = _centro_caja(actual)
                    caderas = _caderas_persona(persona)
                    pref = next((c for c in caderas if c[0] == lado), None)
                    otra = next((c for c in caderas if c[0] != lado), None)
                    if pref is not None and otra is not None:
                        dpref = (cc[0] - pref[1]) ** 2 + (cc[1] - pref[2]) ** 2
                        dotra = (cc[0] - otra[1]) ** 2 + (cc[1] - otra[2]) ** 2
                        if dotra + 24.0 ** 2 < dpref:
                            otra_cadera = True
            if lado and pista.get("categoria") == "bolso" and _es_bolso_cintura(pista, contextos):
                otra_cadera = True
            necesita = actual is None or en_borde or lejos or otra_cadera
            if not necesita:
                continue
            if actual is None and (i < offsets[0][0] or i > ultimo_bueno + 8):
                continue
            ref = None
            for item in reversed(offsets):
                if item[0] <= i:
                    ref = item
                    break
            if ref is None:
                continue
            if actual is None and i - ref[0] > 24:
                continue
            _, dx, dy, w, h = ref
            w, h = w_med or w, h_med or h
            if lado and pista.get("categoria") == "bolso":
                nueva = _caja_en_cadera(persona, w, h, lado, ctx.ancho, ctx.alto)
            elif pista.get("categoria") == "mochila":
                pc = _centro_caja(persona.caja)
                ph = max(float(persona.altura), 1.0)
                nx, ny = pc[0] + 0.15 * dx, persona.caja[1] + 0.22 * ph
                nueva = _clip_caja(
                    (nx - w / 2.0, ny - h / 2.0, nx + w / 2.0, ny + h / 2.0),
                    ctx.ancho, ctx.alto,
                )
            else:
                pc = _centro_caja(persona.caja)
                nx, ny = pc[0] + dx, pc[1] + dy
                nueva = _clip_caja(
                    (nx - w / 2.0, ny - h / 2.0, nx + w / 2.0, ny + h / 2.0),
                    ctx.ancho, ctx.alto,
                )
            if nueva is None:
                if actual is not None and (en_borde or lejos):
                    cajas.pop(i, None)
                continue
            if not _accesorio_en_persona(nueva, ctx) and iou_cajas(nueva, persona.caja) < 0.02:
                if actual is not None and (en_borde or lejos):
                    cajas.pop(i, None)
                continue
            cajas[i] = nueva
        if pista.get("categoria") == "mochila":
            for frame in list(cajas):
                ctx = _contexto_en(contextos, int(frame))
                if ctx is None or _accesorio_en_persona(cajas[frame], ctx):
                    continue
                cajas.pop(frame, None)
        if cajas:
            uf = max(cajas)
            pista["ultimo_frame"] = uf
            pista["caja"] = cajas[uf]
            pista["primer_frame"] = min(cajas)
    return pistas


def _anclar_celular_a_mano(pistas: list[dict], contextos) -> list[dict]:
    """Un móvil visto una vez junto a la palma se arrastra con esa mano hasta el bolsillo."""
    if not contextos:
        return pistas
    for pista in pistas:
        if pista.get("categoria") != "celular":
            continue
        cajas = pista.get("cajas") or {}
        if not cajas:
            continue
        lados: dict[str, int] = {}
        medidas = []
        for frame, caja in cajas.items():
            ctx = _contexto_en(contextos, int(frame))
            if ctx is None or not getattr(ctx, "manos", None):
                continue
            medidas.append((float(caja[2] - caja[0]), float(caja[3] - caja[1])))
            cc = _centro_caja(caja)
            candidatas = []
            for mano in ctx.manos:
                if mano.lado not in ("izquierda", "derecha"):
                    continue
                palma = mano.palma or mano.punto
                dist = float(np.hypot(palma[0] - cc[0], palma[1] - cc[1]))
                if _mano_cerca_caja(mano, caja, margen=48.0):
                    candidatas.append((dist, mano.lado))
            if candidatas:
                lado_c = min(candidatas)[1]
                lados[lado_c] = lados.get(lado_c, 0) + 1
        if not lados or not medidas:
            continue
        lado = max(lados, key=lados.get)
        w = float(np.median([m[0] for m in medidas]))
        h = float(np.median([m[1] for m in medidas]))
        primer, ultimo = min(cajas), max(cajas)
        for i in range(max(0, primer - 2), min(len(contextos), ultimo + 12)):
            if i in cajas:
                continue
            ctx = contextos[i]
            mano = _mano_del_lado(ctx, lado)
            if mano is None:
                continue
            palma = mano.palma or mano.punto
            nueva = _clip_caja(
                (palma[0] - w / 2.0, palma[1] - h / 2.0, palma[0] + w / 2.0, palma[1] + h / 2.0),
                ctx.ancho, ctx.alto,
            )
            if nueva is None:
                continue
            if not _mano_cerca_caja(mano, nueva, margen=36.0):
                continue
            cajas[i] = nueva
        if cajas:
            uf = max(cajas)
            pista["ultimo_frame"] = uf
            pista["caja"] = cajas[uf]
            pista["primer_frame"] = min(cajas)
            pista["frames_visto"] = len(cajas)
    return pistas


def _cortar_mueble_si_atraviesa(pistas: list[dict], contextos) -> list[dict]:
    """Un cesto/carrito congelado no sigue pintándose si la persona cruza esa zona."""
    if not contextos:
        return pistas
    for pista in pistas:
        if pista.get("cuerpo"):
            continue
        if pista.get("accesorio"):
            cajas = pista.get("cajas") or {}
            if not cajas:
                continue
            ultimo = max(cajas)
            caja = cajas[ultimo]
            for i in range(ultimo + 1, len(contextos)):
                ctx = contextos[i]
                if not getattr(ctx, "personas", None):
                    continue
                persona = max(ctx.personas, key=lambda p: p.altura)
                if iou_cajas(caja, persona.caja) < 0.02 and _fraccion_caja_dentro(caja, persona.caja) < 0.08:
                    pista["hold_max"] = 0
                    pista["ultimo_frame"] = ultimo
                    break
            continue
        if pista.get("categoria") not in ("cesto", "carrito", "bolsa"):
            continue
        cajas = pista.get("cajas") or {}
        if not cajas:
            continue
        ultimo = max(cajas)
        caja = cajas[ultimo]
        for i in range(ultimo + 1, len(contextos)):
            ctx = contextos[i]
            if not getattr(ctx, "personas", None):
                continue
            if any(
                iou_cajas(caja, p.caja) >= 0.12 or _fraccion_caja_dentro(caja, p.caja) >= 0.28
                for p in ctx.personas
            ):
                pista["hold_max"] = 0
                pista["ultimo_frame"] = ultimo
                break
    return pistas


def _estabilizar_muebles(pistas: list[dict]) -> list[dict]:
    """Un cesto parado no cambia de tamaño de un frame a otro."""
    for pista in pistas:
        if pista.get("categoria") not in ("cesto", "carrito") or pista.get("accesorio"):
            continue
        cajas = pista.get("cajas") or {}
        if len(cajas) < 3:
            continue
        ws = [float(c[2] - c[0]) for c in cajas.values()]
        hs = [float(c[3] - c[1]) for c in cajas.values()]
        w_med, h_med = float(np.median(ws)), float(np.median(hs))
        if w_med < 8 or h_med < 8:
            continue
        for frame, caja in list(cajas.items()):
            w, h = float(caja[2] - caja[0]), float(caja[3] - caja[1])
            if abs(w - w_med) <= 0.28 * w_med and abs(h - h_med) <= 0.28 * h_med:
                continue
            cx, cy = _centro_caja(caja)
            nueva = (
                int(round(cx - w_med / 2.0)),
                int(round(cy - h_med / 2.0)),
                int(round(cx + w_med / 2.0)),
                int(round(cy + h_med / 2.0)),
            )
            cajas[frame] = nueva
        if cajas:
            uf = max(cajas)
            pista["caja"] = cajas[uf]
    return pistas


def _hueco_pistas(a: dict, b: dict) -> int:
    if not (a["ultimo_frame"] < b["primer_frame"] or b["ultimo_frame"] < a["primer_frame"]):
        return 0
    return min(abs(b["primer_frame"] - a["ultimo_frame"]), abs(a["primer_frame"] - b["ultimo_frame"]))


def _accesorios_unibles(a: dict, b: dict, contextos=None) -> bool:
    if a["categoria"] != b["categoria"]:
        return False
    if a.get("lado") and b.get("lado") and a["lado"] != b["lado"]:
        return False
    area_a = max(1.0, _diagonal(a["caja"]) ** 2)
    area_b = max(1.0, _diagonal(b["caja"]) ** 2)
    if max(area_a, area_b) > 12.0 * min(area_a, area_b):
        return False
    hueco = _hueco_pistas(a, b)
    for fa, ca in a["cajas"].items():
        for fb, cb in b["cajas"].items():
            salto = abs(int(fa) - int(fb))
            if salto <= 10 and _cajas_cercanas(ca, cb):
                return True
            if salto <= 16 and _misma_pista(ca, cb, 0.3, 0.8, max(1, salto), True):
                return True
    if hueco > 24:
        return False
    pa = _indice_persona(a["caja"], _contexto_en(contextos, a["ultimo_frame"]))
    pb = _indice_persona(b["caja"], _contexto_en(contextos, b["ultimo_frame"]))
    if pa is None:
        pa = _indice_persona(a["caja"], _contexto_en(contextos, a["primer_frame"]))
    if pb is None:
        pb = _indice_persona(b["caja"], _contexto_en(contextos, b["primer_frame"]))
    return pa is not None and pa == pb


def _fundir_pista(destino: dict, pista: dict) -> None:
    score_in = max(pista["scores"]) if pista["scores"] else 0.0
    score_ya = max(destino["scores"]) if destino["scores"] else 0.0
    for frame, caja in pista["cajas"].items():
        if frame not in destino["cajas"] or score_in > score_ya:
            destino["cajas"][frame] = caja
    destino["scores"].extend(pista["scores"])
    destino["destino"] = destino.get("destino", False) or pista.get("destino", False)
    destino["accesorio"] = destino.get("accesorio", False) or pista.get("accesorio", False)
    destino["cuerpo"] = destino.get("cuerpo", False) or pista.get("cuerpo", False)
    if pista.get("lado") and not destino.get("lado"):
        destino["lado"] = pista["lado"]
    destino["primer_frame"] = min(destino["primer_frame"], pista["primer_frame"])
    if pista["ultimo_frame"] > destino["ultimo_frame"]:
        destino["ultimo_frame"] = pista["ultimo_frame"]
        destino["caja"] = pista["caja"]
        destino["texto"] = pista.get("texto", destino.get("texto"))
    elif score_in > score_ya:
        destino["caja"] = destino["cajas"].get(
            destino["ultimo_frame"], destino.get("caja", pista.get("caja"))
        )
        destino["texto"] = pista.get("texto", destino.get("texto"))


def _coser_accesorios(pistas: list[dict], contextos=None) -> list[dict]:
    """Un bolso o mochila que camina con la persona no debe abrir un id nuevo."""
    fijas = [p for p in pistas if not p.get("accesorio") and not p.get("cuerpo")]
    moviles = [p for p in pistas if p.get("accesorio") or p.get("cuerpo")]
    if not moviles:
        return pistas
    moviles.sort(key=lambda p: (-max(p["scores"] or [0.0]), p["primer_frame"]))
    unidas: list[dict] = []
    for pista in moviles:
        destino = next((ya for ya in unidas if _accesorios_unibles(ya, pista, contextos)), None)
        if destino is None:
            unidas.append(pista)
            continue
        _fundir_pista(destino, pista)
    return fijas + unidas


def _interpolar_caja(a, b, fa: int, fb: int, f: int):
    if fb == fa:
        return tuple(int(v) for v in a)
    t = (f - fa) / float(fb - fa)
    return tuple(int(round(x + t * (y - x))) for x, y in zip(a, b))


def _unir_contenedores(pistas: list[dict], iou_min: float = 0.4) -> list[dict]:
    """El mismo mueble puede salir como carrito y como cesto: gana el de más score."""
    pistas = sorted(pistas, key=lambda p: -max(p["scores"] or [0.0]))
    unidas: list[dict] = []
    for pista in pistas:
        destino = None
        for ya in unidas:
            if not _compatibles_para_unir(pista, ya):
                continue
            solapes = []
            for frame, caja in pista["cajas"].items():
                rival = caja_contenedor(ya, frame)
                solapes.append(
                    max(
                        iou_cajas(caja, rival),
                        _fraccion_caja_dentro(caja, rival),
                        _fraccion_caja_dentro(rival, caja),
                    )
                )
            if solapes and float(np.mean(solapes)) >= max(iou_min, 0.7):
                destino = ya
                break
        if destino is None:
            unidas.append(pista)
            continue
        for frame, caja in pista["cajas"].items():
            destino["cajas"].setdefault(frame, caja)
        destino["scores"].extend(pista["scores"])
        destino["destino"] = destino.get("destino", False) or pista.get("destino", False)
        destino["accesorio"] = destino.get("accesorio", False) or pista.get("accesorio", False)
        destino["cuerpo"] = destino.get("cuerpo", False) or pista.get("cuerpo", False)
        if pista.get("lado") and not destino.get("lado"):
            destino["lado"] = pista["lado"]
        destino["primer_frame"] = min(destino["primer_frame"], pista["primer_frame"])
        if pista["ultimo_frame"] > destino["ultimo_frame"]:
            destino["ultimo_frame"] = pista["ultimo_frame"]
            destino["caja"] = pista["caja"]
    return unidas


def _compatibles_para_unir(a: dict, b: dict) -> bool:
    """Carrito y cesto pueden ser el mismo mueble. Un celular no se fusiona con el cesto."""
    if a.get("lado") and b.get("lado") and a["lado"] != b["lado"]:
        return False
    if a["categoria"] == b["categoria"]:
        return True
    return {a["categoria"], b["categoria"]} <= {"carrito", "cesto"}


HOLD_MUEBLE = 8
HOLD_ACCESORIO = 6
HOLD_BOLSILLO = 4


def caja_contenedor(pista: dict, frame: int, hold: int | None = None):
    """ROI del contenedor: interpolado entre detecciones, o un hold corto.

    Nunca se pinta antes de la primera detección. ``hold is None`` arrastra
    el último ROI unos pocos frames (no indefinidamente).
    """
    cajas = pista.get("cajas") or {}
    if not cajas:
        return None
    if frame in cajas:
        return cajas[frame]
    conocidos = sorted(cajas)
    primer, ultimo = conocidos[0], conocidos[-1]
    if frame < primer:
        return None
    antes = [f for f in conocidos if f <= frame]
    despues = [f for f in conocidos if f >= frame]
    if antes and despues:
        fa, fb = max(antes), min(despues)
        return _interpolar_caja(cajas[fa], cajas[fb], fa, fb, frame)
    if not antes:
        return None
    cercano = max(antes)
    hold_pista = pista.get("hold_max")
    if hold_pista is not None:
        hold_eff = int(hold_pista)
    elif hold is None:
        hold_eff = HOLD_MUEBLE
    else:
        hold_eff = int(hold)
    if abs(frame - cercano) > hold_eff:
        return None
    return cajas[cercano]


def _fraccion_caja_dentro(caja, contenedor) -> float:
    if caja is None or contenedor is None:
        return 0.0
    x1, y1, x2, y2 = caja
    area = float(max(1, x2 - x1) * max(1, y2 - y1))
    cx1, cy1, cx2, cy2 = contenedor
    ix1, iy1 = max(x1, cx1), max(y1, cy1)
    ix2, iy2 = min(x2, cx2), min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1)) / area


def _objeto_en_contenedor(vista: Vista, caja_cont) -> tuple[float, bool]:
    """Fracción del ROI del objeto dentro del contenedor, y si el centro cae dentro."""
    if caja_cont is None:
        return 0.0, False
    fraccion = _fraccion_caja_dentro(vista.caja, caja_cont)
    centro_dentro = _dentro_de_caja(vista.centro, caja_cont)
    return fraccion, centro_dentro


def _es_rinonera(pista: dict | None, contextos=None, frame: int | None = None) -> bool:
    """Bolso a la cintura (riñonera), no al hombro: sí es destino de depósito."""
    if not pista or pista.get("categoria") != "bolso" or not pista.get("accesorio"):
        return False
    f = int(frame if frame is not None else pista.get("ultimo_frame") or 0)
    caja = caja_contenedor(pista, f, hold=HOLD_ACCESORIO)
    if caja is None:
        return False
    ctx = _contexto_en(contextos, f) if contextos is not None else None
    if ctx is None or not getattr(ctx, "personas", None):
        w, h = caja[2] - caja[0], caja[3] - caja[1]
        return w >= h and h <= 90
    if not _accesorio_en_persona(caja, ctx):
        return False
    persona = max(ctx.personas, key=lambda p: p.altura)
    cy = 0.5 * (caja[1] + caja[3])
    rel = (cy - float(persona.caja[1])) / max(float(persona.altura), 1.0)
    return 0.38 <= rel <= 0.92


def _mueble_sobre_cuerpo(pista: dict, caja, contexto) -> bool:
    """Carrito/cesto dibujado en la espalda/tronco: no es el mueble del pasillo.

    Un cesto de verdad solapa las piernas y baja al suelo; eso no cuenta.
    """
    if caja is None or contexto is None or not getattr(contexto, "personas", None):
        return False
    if pista.get("categoria") not in ("cesto", "carrito"):
        return False
    if _torso_vertical(contexto) < 0.26:
        return False
    persona = max(contexto.personas, key=lambda p: p.altura)
    x1, y1, x2, y2 = persona.caja
    h = max(1.0, float(y2 - y1))
    cc = _centro_caja(caja)
    cy = cc[1]
    alto_caja = caja[3] - caja[1]
    if caja[3] >= float(contexto.alto) - 8:
        return False
    if caja[3] >= y2 + 0.06 * h and alto_caja >= 0.45 * h:
        return False
    frac = _fraccion_caja_dentro(caja, persona.caja)
    iou = iou_cajas(caja, persona.caja)
    # Carrito dibujado en la espalda: compacto, centrado y sin bajar a los pies.
    if x1 <= cc[0] <= x2 and y1 <= cy <= y2 and alto_caja < 0.88 * h:
        if caja[3] <= y1 + 0.88 * h and (frac >= 0.22 or iou >= 0.10):
            return True
    if cy > y1 + 0.82 * h:
        return False
    return frac >= 0.35 or iou >= 0.18


def _cubre_torso(inst: Instancia, contexto) -> bool:
    """Máscara que se come el pecho/cintura: no es el objeto en la palma.

    Un objeto pequeño *dentro* del torso (bolsillo, riñonera) es válido; se
    rechaza cuando el bbox del objeto envuelve una fracción grande del tronco.
    """
    if contexto is None or not getattr(contexto, "personas", None):
        return False
    persona = max(contexto.personas, key=lambda p: p.altura)
    x1, y1, x2, y2 = persona.caja
    w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    torso = (x1 + 0.08 * w, y1 + 0.12 * h, x2 - 0.08 * w, y1 + 0.72 * h)
    if _fraccion_caja_dentro(torso, inst.caja) >= 0.42:
        return True
    area_t = _area_caja(torso)
    if area_t > 0 and _area_caja(inst.caja) >= 0.50 * area_t:
        return True
    return False


def _parece_brazo_o_suelo(inst: Instancia, contexto) -> bool:
    """Losa del pasillo o el brazo entero no son un objeto en la palma."""
    if inst is None or contexto is None:
        return False
    x1, y1, x2, y2 = inst.caja
    w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    if max(w, h) / min(w, h) > 2.85 and max(w, h) > 70:
        return True
    if max(w, h) / min(w, h) > 2.2 and min(w, h) < 18:
        return True
    if not getattr(contexto, "personas", None):
        return False
    persona = max(contexto.personas, key=lambda p: p.altura)
    if inst.centro[1] > float(persona.caja[3]) + 12:
        return True
    if inst.centro[1] < float(persona.caja[1]) - 6:
        return True
    if inst.centro[1] > 0.82 * float(contexto.alto):
        palma_cerca = any(
            float(np.hypot(
                float((m.palma or m.punto)[0]) - inst.centro[0],
                float((m.palma or m.punto)[1]) - inst.centro[1],
            )) < 40.0
            for m in (contexto.manos or [])
        )
        if not palma_cerca:
            return True
    if h > 0.38 * max(float(persona.altura), 1.0) and w < 0.45 * h:
        return True
    return False


def _destinos_deposito(contenedores, incluir_cuerpo: bool = False, frame: int | None = None,
                      contextos=None) -> list[dict]:
    """Cesto/carrito/bolsa útil. El bolso al hombro no; la riñonera sí si el objeto llega."""
    destinos = [p for p in contenedores if p.get("destino", False)]
    if not incluir_cuerpo:
        destinos = [p for p in destinos if not _es_bolsillo(p)]
    destinos = [p for p in destinos if not _es_bolsa_en_cadera(p, contenedores, frame)]
    filtrados = []
    for p in destinos:
        if p.get("accesorio") and not _es_rinonera(p, contextos, frame):
            continue
        filtrados.append(p)
    return filtrados


def _mano_en_caja(contexto, caja, margen: float = 40.0) -> bool:
    if contexto is None or caja is None:
        return False
    for mano in getattr(contexto, "manos", []) or []:
        palma = mano.palma or mano.punto
        if _dentro_de_caja(palma, caja, margen=margen):
            return True
        if _dentro_de_caja(mano.punto, caja, margen=margen):
            return True
    return False


def _ambas_manos_cerca_vista(contexto, vista, margen: float = 56.0) -> bool:
    if contexto is None or vista is None:
        return False
    lados = set()
    for mano in getattr(contexto, "manos", []) or []:
        if mano.lado not in ("izquierda", "derecha"):
            continue
        palma = mano.palma or mano.punto
        if _dentro_de_caja(palma, vista.caja, margen=margen):
            lados.add(mano.lado)
            continue
        if float(np.hypot(palma[0] - vista.centro[0], palma[1] - vista.centro[1])) <= margen:
            lados.add(mano.lado)
    return lados >= {"izquierda", "derecha"}


def _es_bolsillo(pista: dict | None) -> bool:
    if not pista:
        return False
    return bool(pista.get("cuerpo") or pista.get("categoria") in ("bolsillo", "pocket"))


def _es_bulto_cadera(pista: dict | None) -> bool:
    """Bolsa/bolso/mochila: no caben en un bolsillo de pantalón."""
    if not pista:
        return False
    return pista.get("categoria") in ("bolsa", "bolso", "mochila")


def _solapan_cadera(caja_a, caja_b) -> bool:
    if caja_a is None or caja_b is None:
        return False
    if iou_cajas(caja_a, caja_b) >= 0.08:
        return True
    if _fraccion_caja_dentro(caja_a, caja_b) >= 0.22 or _fraccion_caja_dentro(caja_b, caja_a) >= 0.22:
        return True
    ca, cb = _centro_caja(caja_a), _centro_caja(caja_b)
    sep_x = 0.42 * ((caja_a[2] - caja_a[0]) + (caja_b[2] - caja_b[0]))
    sep_y = 0.50 * ((caja_a[3] - caja_a[1]) + (caja_b[3] - caja_b[1]))
    return abs(ca[0] - cb[0]) <= sep_x and abs(ca[1] - cb[1]) <= sep_y


def _cabe_en_bolsillo(caja_obj, caja_bolsillo) -> bool:
    """Una bolsa de compra no entra en el bolsillo: el bbox es mayor o igual que el ROI."""
    if caja_obj is None or caja_bolsillo is None:
        return True
    ao, ap = _area_caja(caja_obj), _area_caja(caja_bolsillo)
    if ap <= 1 or ao >= 0.9 * ap:
        return False
    wo, ho = caja_obj[2] - caja_obj[0], caja_obj[3] - caja_obj[1]
    wp, hp = caja_bolsillo[2] - caja_bolsillo[0], caja_bolsillo[3] - caja_bolsillo[1]
    return wo < 1.15 * wp and ho < 1.25 * hp


def _bolsillo_ocupado_por_bolsa(pista: dict, frame: int, contenedores) -> bool:
    """Descarta el ROI de bolsillo del lado donde ya hay una bolsa demasiado grande."""
    if not _es_bolsillo(pista) or not contenedores:
        return False
    caja_p = caja_contenedor(pista, frame, hold=4)
    if caja_p is None:
        return False
    for otro in contenedores:
        if not _es_bulto_cadera(otro):
            continue
        caja_o = caja_contenedor(otro, frame, hold=6)
        if caja_o is None or not _solapan_cadera(caja_p, caja_o):
            continue
        if otro.get("categoria") == "bolsa" or not _cabe_en_bolsillo(caja_o, caja_p):
            return True
        if iou_cajas(caja_p, caja_o) >= 0.18:
            return True
    return False


def _es_bolsa_en_cadera(pista: dict, contenedores, frame: int | None = None) -> bool:
    """Bolsa pegada al muslo: se lleva en la mano, no es un contenedor de depósito."""
    if not _es_bulto_cadera(pista) or not contenedores:
        return False
    if frame is not None:
        frames = [int(frame)]
    else:
        frames = sorted(pista.get("cajas") or [pista.get("ultimo_frame", 0)])[-6:]
    for f in frames:
        caja = caja_contenedor(pista, int(f), hold=6)
        if caja is None:
            continue
        for bolsillo in contenedores:
            if not _es_bolsillo(bolsillo):
                continue
            caja_p = caja_contenedor(bolsillo, int(f), hold=4)
            if caja_p is None or not _solapan_cadera(caja, caja_p):
                continue
            if pista.get("categoria") == "bolsa" or not _cabe_en_bolsillo(caja, caja_p):
                return True
    return False


def _filtrar_bolsillos_ocupados(detecciones: list[dict]) -> list[dict]:
    grandes = [d for d in detecciones if _es_bulto_cadera(d)]
    if not grandes:
        return detecciones
    vivos = []
    for det in detecciones:
        if not _es_bolsillo(det):
            vivos.append(det)
            continue
        ocupado = False
        for g in grandes:
            if not _solapan_cadera(det.get("caja"), g.get("caja")):
                continue
            if g.get("categoria") == "bolsa" or not _cabe_en_bolsillo(g.get("caja"), det.get("caja")):
                ocupado = True
                break
        if not ocupado:
            vivos.append(det)
    return vivos


def _mano_del_lado(contexto, lado: str | None):
    if contexto is None or not lado:
        return None
    for mano in getattr(contexto, "manos", []) or []:
        if mano.lado == lado:
            return mano
    return None


def _media_puntos(kp, indices, umbral: float):
    xs, ys = [], []
    for i in indices:
        if kp is None or i >= len(kp):
            continue
        if float(kp[i][2]) < umbral:
            continue
        xs.append(float(kp[i][0]))
        ys.append(float(kp[i][1]))
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _torso_vertical(contexto) -> float:
    """1 = erguida, ~0 = agachada sobre un cesto. El bolsillo no vale si está muy baja."""
    if contexto is None:
        return 0.5
    umbral = 0.12
    mejores = []
    for persona in getattr(contexto, "personas", []) or []:
        altura = max(float(persona.altura), 1.0)
        kp = persona.keypoints
        hombros = _media_puntos(kp, (HOMBRO_IZQ, HOMBRO_DER), umbral)
        caderas = _media_puntos(kp, (CADERA_IZQ, CADERA_DER), umbral)
        if hombros is not None and caderas is not None:
            mejores.append(float(np.clip((caderas[1] - hombros[1]) / altura, 0.0, 1.0)))
            continue
        x1, y1, x2, y2 = persona.caja
        ancho = max(1.0, float(x2 - x1))
        mejores.append(float(np.clip(altura / (1.7 * ancho), 0.0, 1.0)))
    return max(mejores) if mejores else 0.5


def _lado_pista(pista: dict | None) -> str | None:
    if not pista:
        return None
    if pista.get("lado") in ("izquierda", "derecha"):
        return pista["lado"]
    texto = str(pista.get("texto") or pista.get("prompt") or "").lower()
    if "left" in texto:
        return "izquierda"
    if "right" in texto:
        return "derecha"
    return None


def _y_cadera(contexto, lado: str | None):
    if contexto is None or not getattr(contexto, "personas", None):
        return None
    persona = max(contexto.personas, key=lambda p: p.altura)
    idx = CADERA_IZQ if lado == "izquierda" else CADERA_DER
    kp = persona.keypoints
    if kp is not None and len(kp) > idx and float(kp[idx][2]) >= 0.12:
        return float(kp[idx][1])
    return float(persona.caja[3]) - 0.38 * max(float(persona.altura), 1.0)


def _mano_hacia_mueble(palma, p0, contenedores, frame: int) -> bool:
    """La mano baja hacia el cesto: no es un gesto de bolsillo."""
    if not contenedores or palma is None or p0 is None:
        return False
    for pista in contenedores:
        if pista.get("categoria") not in ("cesto", "carrito") or pista.get("accesorio"):
            continue
        caja = caja_contenedor(pista, frame)
        if caja is None:
            continue
        cx = 0.5 * (float(caja[0]) + float(caja[2]))
        cy = 0.5 * (float(caja[1]) + float(caja[3]))
        d0 = float(np.hypot(float(p0[0]) - cx, float(p0[1]) - cy))
        d1 = float(np.hypot(float(palma[0]) - cx, float(palma[1]) - cy))
        if d1 + 24.0 < d0 and d1 < 280.0:
            return True
    return False


def _gesto_mano_en_bolsillo(contextos, frame: int, pista: dict, ventana: int = 10,
                            contenedores=None) -> bool:
    """True solo si esa mano se acerca al bolsillo del mismo lado, sin agacharse al cesto."""
    if not _es_bolsillo(pista) or not contextos:
        return False
    if contenedores and _bolsillo_ocupado_por_bolsa(pista, frame, contenedores):
        return False
    ctx = _contexto_en(contextos, frame)
    if ctx is None or _torso_vertical(ctx) < 0.22:
        return False
    caja = caja_contenedor(pista, frame, hold=HOLD_BOLSILLO)
    if caja is None:
        return False
    lado = _lado_pista(pista)
    mano = _mano_del_lado(ctx, lado)
    if mano is None:
        mano = next(
            (m for m in (ctx.manos or []) if _dentro_de_caja(m.palma or m.punto, caja, 24.0)),
            None,
        )
    if mano is None:
        return False
    palma = mano.palma or mano.punto
    if not _dentro_de_caja(palma, caja, margen=10.0):
        return False
    hip_pt = _y_cadera(ctx, lado or mano.lado)
    if hip_pt is not None and abs(float(palma[1]) - float(hip_pt)) > 0.28 * max(
        float(getattr(max(ctx.personas, key=lambda p: p.altura), "altura", 1.0)) if ctx.personas else 1.0,
        1.0,
    ):
        return False
    lado_mano = mano.lado
    hip = _y_cadera(ctx, lado or lado_mano)
    inicio = max(0, int(frame) - int(ventana))
    for i in range(inicio, int(frame)):
        ctx_i = contextos[i]
        mano_i = _mano_del_lado(ctx_i, lado_mano)
        if mano_i is None:
            continue
        p0 = mano_i.palma or mano_i.punto
        if _mano_hacia_mueble(palma, p0, contenedores, frame):
            return False
        caja_i = caja_contenedor(pista, i, hold=4)
        if caja_i is None:
            continue
        fuera = not _dentro_de_caja(p0, caja_i, margen=6.0)
        hip_i = _y_cadera(ctx_i, lado or lado_mano)
        if hip is None or hip_i is None:
            continue
        mas_alta = (float(p0[1]) - hip_i) + 40.0 < (float(palma[1]) - hip)
        if fuera and mas_alta and float(np.hypot(p0[0] - palma[0], p0[1] - palma[1])) >= 40.0:
            return True
    return False


def _bolsillo_activo(pista: dict, indice: int, contextos, contenedores=None) -> bool:
    if contextos is None:
        return False
    if contenedores and _bolsillo_ocupado_por_bolsa(pista, indice, contenedores):
        return False
    ctx = _contexto_en(contextos, indice)
    if ctx is None or _torso_vertical(ctx) < 0.22:
        return False
    return _gesto_mano_en_bolsillo(contextos, indice, pista, contenedores=contenedores)


def _deposito_en_bolsillo(vista: Vista, pista: dict, contextos, contenedores=None) -> bool:
    """El objeto acaba en el bolsillo solo con torso erguido y gesto de meter la mano."""
    if not _es_bolsillo(pista) or contextos is None or vista is None:
        return False
    if contenedores and _bolsillo_ocupado_por_bolsa(pista, vista.frame, contenedores):
        return False
    ctx = _contexto_en(contextos, vista.frame)
    if ctx is None or _torso_vertical(ctx) < 0.22:
        return False
    if vista.mano is None:
        return False
    caja = caja_contenedor(pista, vista.frame, hold=HOLD_BOLSILLO)
    if caja is None:
        return False
    cerca = (
        _dentro_de_caja(vista.centro, caja, margen=8.0)
        or _fraccion_caja_dentro(vista.caja, caja) >= 0.40
    )
    if not cerca:
        return False
    if vista.mano is not None and pista.get("lado") and vista.mano != pista.get("lado") and vista.mano != "ambas":
        return False
    if _gesto_mano_en_bolsillo(contextos, vista.frame, pista, contenedores=contenedores):
        return True
    if (
        _fraccion_caja_dentro(vista.caja, caja) >= 0.45
        and _cabe_en_bolsillo(vista.caja, caja)
        and _torso_vertical(ctx) >= 0.26
    ):
        return True
    inicio = max(0, int(vista.frame) - 4)
    fin = min(len(contextos), int(vista.frame) + 2)
    return any(
        _gesto_mano_en_bolsillo(contextos, f, pista, contenedores=contenedores)
        for f in range(inicio, fin)
    )


def _destinos_para_frame(contenedores, contextos, frame: int | None) -> list[dict]:
    destinos = list(_destinos_deposito(contenedores, frame=frame, contextos=contextos))
    ctx = _contexto_en(contextos, int(frame)) if contextos is not None and frame is not None else None
    destinos = [p for p in destinos if not _mueble_sobre_cuerpo(p, caja_contenedor(p, int(frame) if frame is not None else 0), ctx)]
    if contextos is None or frame is None:
        return destinos
    vistos = {id(p) for p in destinos}
    for pista in _destinos_deposito(contenedores, incluir_cuerpo=True, frame=frame, contextos=contextos):
        if id(pista) in vistos:
            continue
        if _es_bolsillo(pista) and _bolsillo_activo(pista, int(frame), contextos, contenedores):
            destinos.append(pista)
            vistos.add(id(pista))
        elif _es_rinonera(pista, contextos, frame) and id(pista) not in vistos:
            destinos.append(pista)
            vistos.add(id(pista))
    return destinos


def _objeto_lejos_del_cuerpo(vistas: list[Vista], contextos, ultimas: int = 6) -> bool:
    """True si el objeto estuvo en una estantería u otro sitio lejos del torso."""
    if not vistas or not contextos:
        return False
    for vista in list(vistas)[-ultimas:]:
        ctx = _contexto_en(contextos, vista.frame)
        if ctx is None or not getattr(ctx, "personas", None):
            continue
        persona = max(ctx.personas, key=lambda p: p.altura)
        if _fraccion_caja_dentro(vista.caja, persona.caja_padded) < 0.12:
            return True
        pc = _centro_caja(persona.caja)
        if float(np.hypot(vista.centro[0] - pc[0], vista.centro[1] - pc[1])) > 0.55 * max(persona.altura, 80.0):
            return True
    return False


def contenedor_final(vistas: list[Vista], contenedores: list[dict], args,
                    contextos=None) -> dict | None:
    """Último frame del objeto: primero mueble (cesto), bolsillo solo con gesto."""
    if not vistas or not contenedores:
        return None
    destinos = _destinos_deposito(contenedores, frame=vistas[-1].frame, contextos=contextos)
    ultima = vistas[-1]
    ctx_u = _contexto_en(contextos, ultima.frame)
    destinos = [
        p for p in destinos
        if not _mueble_sobre_cuerpo(p, caja_contenedor(p, ultima.frame), ctx_u)
    ]
    cerca_carro = False
    for p in destinos:
        if p.get("categoria") not in ("cesto", "carrito"):
            continue
        caja_m = caja_contenedor(p, ultima.frame)
        frac_m, centro_m = _objeto_en_contenedor(ultima, caja_m)
        if centro_m or frac_m >= 0.15 or _dentro_de_caja(ultima.centro, caja_m, margen=56.0):
            cerca_carro = True
            break
    if cerca_carro:
        destinos = [p for p in destinos if p.get("categoria") != "bolso"]
    if contextos is not None:
        inicio_g = max(0, ultima.frame - 6)
        fin_g = min(len(contextos), ultima.frame + 3)
        gesto = False
        for f in range(inicio_g, fin_g):
            for b in contenedores:
                if _es_bolsillo(b) and _gesto_mano_en_bolsillo(
                    contextos, f, b, contenedores=contenedores
                ):
                    gesto = True
                    break
            if gesto:
                break
        if gesto:
            cerca_mueble = False
            for p in destinos:
                if p.get("categoria") not in ("cesto", "carrito"):
                    continue
                caja_m = caja_contenedor(p, ultima.frame)
                frac_m, centro_m = _objeto_en_contenedor(ultima, caja_m)
                if centro_m or frac_m >= 0.18 or _dentro_de_caja(ultima.centro, caja_m, margen=48.0):
                    cerca_mueble = True
                    break
            if not cerca_mueble:
                destinos = [
                    p for p in destinos
                    if p.get("categoria") not in ("cesto", "carrito")
                ]
    umbral = float(args.dentro_min)
    umbral_suave = min(umbral, 0.35)
    margen_px = 80.0

    def candidato(pista, caja, fraccion, centro_dentro, frames_eval=1):
        if caja is None:
            return None
        return {
            "id": pista["id"],
            "category": pista["categoria"],
            "roi": list(caja),
            "fraction_inside": round(float(fraccion), 4),
            "center_inside": bool(centro_dentro),
            "frames_evaluated": frames_eval,
            "score": pista["score_max"],
        }

    def mano_despues(pista) -> bool:
        if contextos is None:
            return False
        for idx in range(ultima.frame, len(contextos)):
            ctx = contextos[idx]
            caja = caja_contenedor(pista, idx)
            if _mueble_sobre_cuerpo(pista, caja, ctx):
                continue
            if _mano_en_caja(ctx, caja):
                return True
        return False

    mejor_dentro = None
    mejor_cerca = None
    for pista in destinos:
        caja = caja_contenedor(pista, ultima.frame)
        if caja is None:
            continue
        if pista.get("accesorio"):
            if ctx_u is None or not _mano_en_caja(ctx_u, caja, margen=18.0):
                continue
            if (
                iou_cajas(ultima.caja, caja) >= 0.28
                and _area_caja(ultima.caja) >= 0.40 * _area_caja(caja)
            ):
                continue
        fraccion, centro_dentro = _objeto_en_contenedor(ultima, caja)
        cerca = centro_dentro or _dentro_de_caja(ultima.centro, caja, margen=margen_px)
        item = candidato(pista, caja, fraccion, centro_dentro)
        if item is None:
            continue
        if centro_dentro or fraccion >= umbral_suave:
            if mejor_dentro is None or item["fraction_inside"] > mejor_dentro["fraction_inside"]:
                mejor_dentro = item
        elif cerca:
            if mejor_cerca is None or item["fraction_inside"] > mejor_cerca["fraction_inside"]:
                mejor_cerca = item | {"_pista": pista}

    if mejor_dentro is not None:
        return mejor_dentro
    if mejor_cerca is not None and mano_despues(mejor_cerca.pop("_pista")):
        mejor_cerca["center_inside"] = True
        mejor_cerca["fraction_inside"] = max(mejor_cerca["fraction_inside"], 0.5)
        return mejor_cerca

    ultimas = vistas[-max(1, args.frames_fin) :]
    mejor = None
    for pista in destinos:
        fracs = []
        for vista in ultimas:
            caja = caja_contenedor(pista, vista.frame)
            fraccion, centro_dentro = _objeto_en_contenedor(vista, caja)
            fracs.append(1.0 if centro_dentro else fraccion)
        if not fracs:
            continue
        media = float(np.mean(fracs))
        votos = sum(1 for f in fracs if f >= umbral_suave) / len(fracs)
        if votos < 0.5:
            continue
        caja = caja_contenedor(pista, ultima.frame)
        item = candidato(pista, caja, media, _dentro_de_caja(ultima.centro, caja), len(fracs))
        if item is None:
            continue
        if mejor is None or item["fraction_inside"] > mejor["fraction_inside"]:
            mejor = item
    if mejor is not None:
        return mejor

    if contextos is not None:
        for pista in destinos:
            caja_u = caja_contenedor(pista, ultima.frame)
            cerca = (
                _dentro_de_caja(ultima.centro, caja_u, margen=120.0)
                or _fraccion_caja_dentro(ultima.caja, caja_u) >= 0.15
            )
            if not cerca or not mano_despues(pista):
                continue
            caja = caja_contenedor(pista, len(contextos) - 1)
            item = candidato(
                pista, caja,
                max(_fraccion_caja_dentro(ultima.caja, caja), 0.5),
                True,
                max(1, len(contextos) - ultima.frame),
            )
            if item is not None:
                return item

    if contextos is None:
        return None
    if ultima.mano is None or sum(1 for v in vistas[-3:] if v.mano is None) >= 2:
        return None
    if _objeto_lejos_del_cuerpo(vistas, contextos):
        return None
    bolsillos = [
        p for p in _destinos_deposito(
            contenedores, incluir_cuerpo=True, frame=ultima.frame, contextos=contextos
        )
        if _es_bolsillo(p) and not _bolsillo_ocupado_por_bolsa(p, ultima.frame, contenedores)
    ]
    mejor_bolsillo = None
    for vista in reversed(vistas[-3:]):
        ctx_v = _contexto_en(contextos, vista.frame)
        if ctx_v is not None and _persona_sale_de_escena(ctx_v):
            continue
        for pista in bolsillos:
            if _bolsillo_ocupado_por_bolsa(pista, vista.frame, contenedores):
                continue
            if not _deposito_en_bolsillo(vista, pista, contextos, contenedores):
                continue
            caja = caja_contenedor(pista, vista.frame, hold=4)
            fraccion = _fraccion_caja_dentro(vista.caja, caja)
            if vista.mano is not None and fraccion < 0.55:
                continue
            item = candidato(
                pista, caja,
                max(fraccion, 0.45),
                True,
                1,
            )
            if item is None:
                continue
            if mejor_bolsillo is None or item["fraction_inside"] > mejor_bolsillo["fraction_inside"]:
                mejor_bolsillo = item
        if mejor_bolsillo is not None:
            return mejor_bolsillo
    for pista in bolsillos:
        caja = caja_contenedor(pista, ultima.frame, hold=HOLD_BOLSILLO)
        if caja is None or ctx_u is None:
            continue
        if not _mano_en_caja(ctx_u, caja, margen=28.0):
            continue
        celular = any(
            p.get("categoria") == "celular"
            and (
                iou_cajas(ultima.caja, caja_contenedor(p, ultima.frame, hold=6) or (0, 0, 0, 0)) >= 0.12
                or _fraccion_caja_dentro(ultima.caja, caja_contenedor(p, ultima.frame, hold=6)) >= 0.30
            )
            for p in contenedores
        )
        cerca = _dentro_de_caja(ultima.centro, caja, margen=28.0) or _fraccion_caja_dentro(ultima.caja, caja) >= 0.22
        if celular and cerca:
            item = candidato(pista, caja, max(_fraccion_caja_dentro(ultima.caja, caja), 0.45), True, 1)
            if item is not None:
                return item
    return None


# --------------------------------------------------------------------------- #
# Pistas de objeto y propagación
# --------------------------------------------------------------------------- #


def objetos_del_frame(frame, contexto, detector, sam, args) -> list[tuple[Instancia, dict]]:
    """Como máximo un objeto por mano, y solo si hay contacto real de palma o dedos."""
    candidatos = candidatos_en_munecas(frame, contexto, detector, sam, args)
    anotados = [
        inst
        for inst in anotar_agarre(
            candidatos,
            contexto,
            umbral=args.umbral_agarre,
            area_minima=max(1e-6, args.area_min / float(max(1, contexto.ancho * contexto.alto))),
        )
        if inst.en_mano
    ]
    evaluados = []
    for inst in anotados:
        agarre = {
            "mano": inst.mano_asociada or None,
            "score": float(inst.score_agarre),
            "scores": inst.scores_mano or {},
        }
        if agarre["mano"] is not None:
            evaluados.append((inst, agarre))
    evaluados.sort(key=lambda par: (-par[1]["score"], par[0].area))
    elegidos: list[tuple[Instancia, dict]] = []
    ocupadas: set[str] = set()
    for inst, agarre in evaluados:
        lados = {"izquierda", "derecha"} if agarre["mano"] == "ambas" else {agarre["mano"]}
        if lados & ocupadas:
            continue
        if any(_solapan(inst.mascara, ya.mascara, 0.3) for ya, _ in elegidos):
            continue
        elegidos.append((inst, agarre))
        ocupadas |= lados
    fusionados: list[tuple[Instancia, dict]] = []
    usados: set[int] = set()
    for i, (inst, agr) in enumerate(elegidos):
        if i in usados:
            continue
        mano = agr["mano"]
        for j, (otro, agr_o) in enumerate(elegidos):
            if j <= i or j in usados:
                continue
            dist = float(
                np.hypot(inst.centro[0] - otro.centro[0], inst.centro[1] - otro.centro[1])
            )
            mismo = (
                dist < 56.0
                or iou_cajas(inst.caja, otro.caja) >= 0.15
                or _solapan(inst.mascara, otro.mascara, 0.15)
            )
            if not mismo:
                continue
            usados.add(j)
            if otro.area < 0.55 * inst.area:
                inst, agr = otro, agr_o
            if {mano, agr_o["mano"]} >= {"izquierda", "derecha"} or "ambas" in (mano, agr_o["mano"]):
                mano = "ambas"
        agr = dict(agr)
        agr["mano"] = mano
        fusionados.append((inst, agr))
        usados.add(i)
    return fusionados


def detectar_objetos(frames, escena, detector, sam, args):
    """Recorre el vídeo y agrupa en pistas los objetos vistos en las manos."""
    contextos: list = []
    pistas: list[dict] = []
    total = 0
    for indice, frame in enumerate(frames):
        try:
            contexto = escena.analizar(frame)
        except Exception as exc:
            _aviso(f"frame {indice}: pose/manos falló ({exc})")
            alto, ancho = frame.shape[:2]
            contexto = ContextoEscena(ancho=ancho, alto=alto)
        contextos.append(contexto)
        if not contexto.manos:
            if indice % 20 == 0:
                print(f"    frame {indice}: wrists below min_confidence, object model skipped")
            continue
        try:
            hallados = objetos_del_frame(frame, contexto, detector, sam, args)
        except Exception as exc:
            _aviso(f"frame {indice}: detección en muñeca falló ({exc})")
            _vaciar_cuda()
            continue
        for inst, agarre in hallados:
            total += 1
            centro = inst.centro
            mejor, mejor_d = None, 1e9
            for pista in pistas:
                if indice - pista["ultimo_frame"] > args.hueco_max:
                    continue
                if indice == pista["ultimo_frame"]:
                    continue
                dist = float(
                    np.hypot(centro[0] - pista["centro"][0], centro[1] - pista["centro"][1])
                )
                if dist < mejor_d:
                    mejor, mejor_d = pista, dist
            mano_nueva = agarre["mano"]
            if mejor is not None and mejor_d <= args.distancia_max:
                mano_prev = (mejor["agarres"].get(mejor["ultimo_frame"]) or {}).get("mano")
                cambio_lejano = (
                    mano_prev in ("izquierda", "derecha")
                    and mano_nueva in ("izquierda", "derecha")
                    and mano_prev != mano_nueva
                    and mejor_d > 80.0
                )
                area_prev = int(mejor["mascaras"][mejor["ultimo_frame"]].sum())
                salto_area = (
                    area_prev > 0
                    and max(area_prev, inst.area) > 2.6 * min(area_prev, inst.area)
                )
                dy = abs(float(centro[1]) - float(mejor["centro"][1]))
                # Con las dos manos visibles, acercarse al cesto baja el objeto >140 px.
                salto_vertical = dy > (220.0 if (mano_prev and mano_nueva) else 140.0)
            else:
                cambio_lejano = False
                salto_area = False
                salto_vertical = False
            if (
                mejor is not None
                and mejor_d <= args.distancia_max
                and not cambio_lejano
                and not salto_area
                and not salto_vertical
            ):
                mejor["centro"] = centro
                mejor["ultimo_frame"] = indice
                mejor["mascaras"][indice] = inst.mascara
                mejor["agarres"][indice] = agarre
            else:
                pistas.append(
                    {
                        "id": len(pistas) + 1,
                        "centro": centro,
                        "primer_frame": indice,
                        "ultimo_frame": indice,
                        "mascaras": {indice: inst.mascara},
                        "agarres": {indice: agarre},
                    }
                )
        if indice % 20 == 0:
            print(f"    frame {indice}: {len(pistas)} pistas, {total} detecciones en mano")
    def _pista_corta_al_salir(pista: dict) -> bool:
        if len(pista["mascaras"]) < 1:
            return False
        ultimo = int(pista["ultimo_frame"])
        agarre = pista["agarres"].get(ultimo) or {}
        if not agarre.get("mano"):
            return False
        ctx = contextos[ultimo] if 0 <= ultimo < len(contextos) else None
        if ctx is None:
            return False
        if _persona_sale_de_escena(ctx) or _de_espaldas(ctx):
            return True
        if getattr(ctx, "personas", None):
            persona = max(ctx.personas, key=lambda p: p.altura)
            if persona.caja[3] >= int(ctx.alto) - 10:
                return True
        return not any(
            bool(getattr(contextos[i], "manos", None))
            for i in range(ultimo + 1, len(contextos))
        )

    sale_mas_tarde = any(
        bool(getattr(ctx, "personas", None))
        and (
            _persona_sale_de_escena(ctx)
            or max(ctx.personas, key=lambda p: p.altura).caja[3] >= int(ctx.alto) - 10
        )
        for ctx in contextos
    )
    pistas = [
        p for p in pistas
        if len(p["mascaras"]) >= args.min_detecciones
        or _pista_corta_al_salir(p)
        or (sale_mas_tarde and len(p["mascaras"]) >= 1)
    ]
    pistas = fusionar_pistas(pistas)
    pistas = unir_tramos(pistas, args)
    for numero, pista in enumerate(pistas, start=1):
        pista["id"] = numero
    print(f"  Manos: {total} detecciones en mano, {len(pistas)} objetos")
    return contextos, pistas


def fusionar_pistas(pistas: list[dict], comunes_min: int = 2, iou_min: float = 0.3) -> list[dict]:
    """Dos pistas que ven la misma máscara en los mismos frames son un solo objeto."""
    fusionadas: list[dict] = []
    for pista in sorted(pistas, key=lambda p: (p["primer_frame"], -len(p["mascaras"]))):
        destino = None
        for ya in fusionadas:
            comunes = set(pista["mascaras"]) & set(ya["mascaras"])
            if len(comunes) < comunes_min:
                continue
            solapes = [
                1.0 if _solapan(pista["mascaras"][f], ya["mascaras"][f], 0.3) else 0.0
                for f in comunes
            ]
            if float(np.mean(solapes)) >= iou_min:
                destino = ya
                break
        if destino is None:
            fusionadas.append(pista)
            continue
        for frame, mascara in pista["mascaras"].items():
            destino["mascaras"].setdefault(frame, mascara)
            destino["agarres"].setdefault(frame, pista["agarres"][frame])
        destino["primer_frame"] = min(destino["primer_frame"], pista["primer_frame"])
        if pista["ultimo_frame"] > destino["ultimo_frame"]:
            destino["ultimo_frame"] = pista["ultimo_frame"]
            destino["centro"] = pista["centro"]
    return fusionadas


def _centro_mascara(mascara) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mascara)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def unir_tramos(pistas: list[dict], args) -> list[dict]:
    """El mismo objeto se parte al caminar: se empalman tramos que no se solapan."""
    ordenadas = sorted(pistas, key=lambda p: (p["primer_frame"], -len(p["mascaras"])))
    unidas: list[dict] = []
    for pista in ordenadas:
        destino = None
        mejor_d = 1e9
        for ya in unidas:
            if pista["primer_frame"] <= ya["ultimo_frame"]:
                continue
            hueco = pista["primer_frame"] - ya["ultimo_frame"]
            if hueco > args.hueco_max:
                continue
            c_ya = _centro_mascara(ya["mascaras"][ya["ultimo_frame"]])
            c_nueva = _centro_mascara(pista["mascaras"][pista["primer_frame"]])
            if c_ya is None or c_nueva is None:
                continue
            dist = float(np.hypot(c_nueva[0] - c_ya[0], c_nueva[1] - c_ya[1]))
            permitido = args.distancia_max + 4.0 * hueco
            mano_ya = (ya["agarres"].get(ya["ultimo_frame"]) or {}).get("mano")
            mano_nueva = (pista["agarres"].get(pista["primer_frame"]) or {}).get("mano")
            if (
                mano_ya in ("izquierda", "derecha")
                and mano_nueva in ("izquierda", "derecha")
                and mano_ya != mano_nueva
                and dist > args.distancia_max * 0.55
            ):
                continue
            area_ya = int(ya["mascaras"][ya["ultimo_frame"]].sum())
            area_nueva = int(pista["mascaras"][pista["primer_frame"]].sum())
            if min(area_ya, area_nueva) <= 0:
                continue
            if max(area_ya, area_nueva) > 2.4 * min(area_ya, area_nueva):
                continue
            if mano_ya is None and dist > 56.0:
                continue
            tope_dist = 200.0 if (mano_ya and mano_nueva) else 140.0
            if dist > tope_dist:
                continue
            if dist <= permitido and dist < mejor_d:
                destino, mejor_d = ya, dist
        if destino is None:
            unidas.append(pista)
            continue
        for frame, mascara in pista["mascaras"].items():
            destino["mascaras"].setdefault(frame, mascara)
            destino["agarres"].setdefault(frame, pista["agarres"][frame])
        if pista["ultimo_frame"] > destino["ultimo_frame"]:
            destino["ultimo_frame"] = pista["ultimo_frame"]
            destino["centro"] = pista["centro"]
    return unidas


def descartar_contenedores_en_mano(contenedores, objetos, args) -> list[dict]:
    """Un bolso o una bolsa en la mano es el objeto seguido, no su contenedor."""
    vivos = []
    for pista in contenedores:
        if pista.get("categoria") == "celular":
            vivos.append(pista)
            continue
        total = max(1, len(pista["cajas"]))
        coincidencias = 0
        for frame, caja in pista["cajas"].items():
            area_cont = float(max(1, caja[2] - caja[0]) * max(1, caja[3] - caja[1]))
            for objeto in objetos:
                vista = objeto.vistas.get(frame)
                if vista is None:
                    continue
                area_obj = float(
                    max(1, vista.caja[2] - vista.caja[0]) * max(1, vista.caja[3] - vista.caja[1])
                )
                if (
                    iou_cajas(caja, vista.caja) >= 0.35
                    and area_cont < args.factor_contenedor * area_obj
                ):
                    coincidencias += 1
                    break
        # Frente a todas las detecciones del identificador: un solape tardío
        # (el objeto se acerca al bolso) no debe borrar el accesorio.
        if coincidencias / total >= 0.5:
            print(f"  {pista['id']} se descarta: es el objeto que lleva en la mano")
            continue
        vivos.append(pista)
    return vivos


def descartar_accesorios(objetos: list[Objeto], contenedores: list[dict]) -> list[Objeto]:
    """El bolso al hombro o la mochila (accesorio: true en el YAML) no son el objeto agarrado."""
    accesorios = [c for c in contenedores if c.get("accesorio")]
    if not accesorios:
        return objetos
    vivos: list[Objeto] = []
    for objeto in objetos:
        # Solo las primeras vistas: acercarse al bolso al final es un depósito, no identidad.
        primeras = list(objeto.vistas.items())[:4]
        votos = 0
        comparados = 0
        for frame, vista in primeras:
            for accesorio in accesorios:
                caja = caja_contenedor(accesorio, frame, hold=HOLD_ACCESORIO)
                if caja is None:
                    continue
                comparados += 1
                if _area_caja(vista.caja) < 0.55 * _area_caja(caja):
                    continue
                if (
                    iou_cajas(vista.caja, caja) >= 0.25
                    or _fraccion_caja_dentro(vista.caja, caja) >= 0.45
                ):
                    votos += 1
                    break
        if comparados and votos / comparados >= 0.5:
            print(f"  Objeto {objeto.id} se descarta: es un bolso o mochila, no va en la palma")
            continue
        vivos.append(objeto)
    for numero, objeto in enumerate(vivos, start=1):
        objeto.id = numero
    return vivos


def _vista_es_bolsa(vista: Vista, contenedores) -> bool:
    if vista is None or not contenedores:
        return False
    for pista in contenedores:
        if not _es_bulto_cadera(pista):
            continue
        caja = caja_contenedor(pista, vista.frame, hold=6)
        if caja is None:
            continue
        if iou_cajas(vista.caja, caja) >= 0.30 or _fraccion_caja_dentro(vista.caja, caja) >= 0.45:
            return True
        if _area_caja(vista.caja) >= 0.85 * _area_caja(caja) and _solapan_cadera(vista.caja, caja):
            return True
    return False


def recortar_objeto_si_salta_a_bolsa(objetos: list[Objeto], contenedores) -> list[Objeto]:
    """Si el móvil salta a la bolsa (bbox mucho más grande), se queda la traza anterior al salto."""
    for objeto in objetos:
        frames = sorted(objeto.vistas)
        if len(frames) < 2:
            continue
        corte = None
        for prev_f, f in zip(frames, frames[1:]):
            prev, vista = objeto.vistas[prev_f], objeto.vistas[f]
            salto = vista.area > 2.6 * max(1, prev.area)
            if salto or (_vista_es_bolsa(vista, contenedores) and not _vista_es_bolsa(prev, contenedores)):
                corte = f
                break
        if corte is None:
            continue
        for f in [k for k in objeto.vistas if k >= corte]:
            del objeto.vistas[f]
            objeto.mascaras.pop(f, None)
        if objeto.vistas:
            ultimo = max(objeto.vistas)
            objeto.ultimo_frame = ultimo
            objeto.centro = objeto.vistas[ultimo].centro
    return [o for o in objetos if o.vistas]


def _vista_en_mueble(vista: Vista, contenedores, contextos=None) -> bool:
    if vista is None or not contenedores:
        return False
    ctx = _contexto_en(contextos, vista.frame) if contextos is not None else None
    for pista in contenedores:
        if pista.get("categoria") not in ("cesto", "carrito", "bolsa") or pista.get("accesorio"):
            continue
        caja = caja_contenedor(pista, vista.frame, hold=HOLD_MUEBLE)
        if caja is None:
            continue
        if _mueble_sobre_cuerpo(pista, caja, ctx):
            continue
        if _dentro_de_caja(vista.centro, caja, margen=24.0) or _fraccion_caja_dentro(vista.caja, caja) >= 0.2:
            return True
    return False


def _quitar_vistas_desviadas(objetos: list[Objeto], contenedores=None, contextos=None) -> list[Objeto]:
    """Quita un frame intermedio que salta a la cadera/bolso y vuelve al objeto."""
    for objeto in objetos:
        frames = sorted(objeto.vistas)
        if len(frames) < 3:
            continue
        # Varias pasadas: un salto al bolso puede «justificar» el salto a la cadera.
        while True:
            frames = sorted(objeto.vistas)
            if len(frames) < 3:
                break
            borrar = []
            for a, b, c in zip(frames, frames[1:], frames[2:]):
                va, vb, vc = objeto.vistas[a], objeto.vistas[b], objeto.vistas[c]
                if _vista_en_mueble(vb, contenedores, contextos):
                    continue
                dac = float(np.hypot(vc.centro[0] - va.centro[0], vc.centro[1] - va.centro[1]))
                dab = float(np.hypot(vb.centro[0] - va.centro[0], vb.centro[1] - va.centro[1]))
                dbc = float(np.hypot(vc.centro[0] - vb.centro[0], vc.centro[1] - vb.centro[1]))
                t = (b - a) / float(max(1, c - a))
                pred = (
                    va.centro[0] + t * (vc.centro[0] - va.centro[0]),
                    va.centro[1] + t * (vc.centro[1] - va.centro[1]),
                )
                dpred = float(np.hypot(vb.centro[0] - pred[0], vb.centro[1] - pred[1]))
                if dpred > 80.0 and dab > 70.0 and dab + dbc > 1.55 * max(1.0, dac):
                    borrar.append(b)
            if not borrar:
                break
            for f in borrar:
                objeto.vistas.pop(f, None)
                objeto.mascaras.pop(f, None)
        if objeto.vistas:
            ultimo = max(objeto.vistas)
            objeto.ultimo_frame = ultimo
            objeto.centro = objeto.vistas[ultimo].centro
    return [o for o in objetos if o.vistas]


def recortar_cola_si_salta(objetos: list[Objeto], contenedores=None, contextos=None) -> list[Objeto]:
    """Tras soltar en la estantería no se continúa con un bulto de la cadera."""
    for objeto in objetos:
        frames = sorted(objeto.vistas)
        if len(frames) < 3:
            continue
        diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
        paso = int(np.median(diffs)) if diffs else 1
        corte = None
        for prev_f, f in zip(frames, frames[1:]):
            prev, vista = objeto.vistas[prev_f], objeto.vistas[f]
            dist = float(
                np.hypot(vista.centro[0] - prev.centro[0], vista.centro[1] - prev.centro[1])
            )
            if (f - prev_f) > max(3, 2 * paso) and dist > 80.0:
                n_despues = sum(1 for k in frames if k >= f)
                if n_despues <= 2:
                    corte = f
                    break
            salto_area = vista.area > 2.4 * max(1, prev.area)
            if _vista_en_mueble(vista, contenedores, contextos) and not salto_area:
                continue
            if prev.mano is None and dist > 56.0:
                corte = f
                break
            if (
                dist > 140.0
                and prev.mano in ("izquierda", "derecha")
                and vista.mano in ("izquierda", "derecha")
                and prev.mano != vista.mano
                and not _vista_en_mueble(vista, contenedores, contextos)
            ):
                ctx_s = _contexto_en(contextos, f) if contextos is not None else None
                if ctx_s is not None and _ambas_manos_cerca_vista(ctx_s, vista):
                    objeto.vistas[f] = replace(vista, mano="ambas")
                    continue
                posteriores = [k for k in frames if k >= f][:6]
                if any(
                    _vista_en_mueble(objeto.vistas[k], contenedores, contextos)
                    for k in posteriores
                ):
                    objeto.vistas[f] = replace(vista, mano=prev.mano)
                    continue
                corte = f
                break
            # Caminar hacia la cámara mueve el ROI >120 px; no cortar si sigue en la mano
            # o entra en un cesto/carrito real. El teletransporte sin mano sí se corta.
            if (
                dist > 120.0
                and vista.mano is None
                and not _vista_en_mueble(vista, contenedores, contextos)
            ):
                corte = f
                break
            if salto_area and (dist > 48.0 or vista.area > 3.0 * max(1, prev.area)):
                ctx = _contexto_en(contextos, f) if contextos is not None else None
                cubre = False
                if ctx is not None:
                    cubre = _cubre_torso(SimpleNamespace(caja=vista.caja), ctx)
                if vista.mano is None or cubre:
                    corte = f
                    break
        if corte is None:
            continue
        for f in [k for k in objeto.vistas if k >= corte]:
            del objeto.vistas[f]
            objeto.mascaras.pop(f, None)
        if objeto.vistas:
            ultimo = max(objeto.vistas)
            objeto.ultimo_frame = ultimo
            objeto.centro = objeto.vistas[ultimo].centro
    return [o for o in objetos if o.vistas]


def recortar_si_vuelve_al_cuerpo(objetos: list[Objeto], contenedores, contextos) -> list[Objeto]:
    """Si el artículo deja la mano y esta vuelve a la cadera, no se sigue la palma vacía."""
    if not objetos or contextos is None:
        return objetos
    for objeto in objetos:
        frames = sorted(objeto.vistas)
        if len(frames) < 4:
            continue
        if any(_vista_en_mueble(objeto.vistas[f], contenedores, contextos) for f in frames):
            continue
        ancla = None
        for f in frames[: max(2, len(frames) // 2)]:
            vista = objeto.vistas[f]
            ctx = _contexto_en(contextos, f)
            if ctx is None or not getattr(ctx, "personas", None):
                continue
            persona = max(ctx.personas, key=lambda p: p.altura)
            ph = max(float(persona.altura), 1.0)
            lado_m = vista.mano if vista.mano in ("izquierda", "derecha") else None
            hip = _y_cadera(ctx, lado_m)
            caderas = _caderas_persona(persona)
            hx = next((c[1] for c in caderas if lado_m and c[0] == lado_m), None)
            if hx is None and caderas:
                hx = min(caderas, key=lambda c: abs(c[1] - vista.centro[0]))[1]
            if hip is None:
                continue
            dist_cadera = float(np.hypot(
                (vista.centro[0] - hx) if hx is not None else 0.0,
                vista.centro[1] - float(hip),
            ))
            if dist_cadera > 0.22 * ph:
                ancla = (f, float(hip), ph, vista)
                break
        if ancla is None:
            continue
        corte = None
        for f in frames:
            if f <= ancla[0]:
                continue
            vista = objeto.vistas[f]
            ctx = _contexto_en(contextos, f)
            if ctx is None or not getattr(ctx, "personas", None):
                continue
            if _vista_en_mueble(vista, contenedores, contextos):
                continue
            persona = max(ctx.personas, key=lambda p: p.altura)
            ph = max(float(persona.altura), 1.0)
            lado_m = vista.mano if vista.mano in ("izquierda", "derecha") else None
            hip = _y_cadera(ctx, lado_m)
            caderas = _caderas_persona(persona)
            hx = next((c[1] for c in caderas if lado_m and c[0] == lado_m), None)
            if hx is None and caderas:
                hx = min(caderas, key=lambda c: abs(c[1] - vista.centro[0]))[1]
            if hip is None:
                continue
            dist_cadera = float(np.hypot(
                (vista.centro[0] - hx) if hx is not None else 0.0,
                vista.centro[1] - float(hip),
            ))
            if dist_cadera > 0.16 * ph:
                continue
            if vista.mano in ("izquierda", "derecha", "ambas"):
                continue
            if any(
                _deposito_en_bolsillo(vista, p, contextos, contenedores)
                for p in contenedores if _es_bolsillo(p)
            ):
                continue
            if any(
                _es_rinonera(p, contextos, f) and _mano_en_caja(ctx, caja_contenedor(p, f), 36.0)
                for p in contenedores
            ):
                continue
            if any(p.get("categoria") == "celular" for p in contenedores) and any(
                _mano_en_caja(ctx, caja_contenedor(b, f, hold=4), 18.0)
                for b in contenedores if _es_bolsillo(b)
            ):
                continue
            corte = f
            break
        if corte is None:
            continue
        for k in [i for i in objeto.vistas if i >= corte]:
            objeto.vistas.pop(k, None)
            objeto.mascaras.pop(k, None)
        if objeto.vistas:
            ultimo = max(objeto.vistas)
            objeto.ultimo_frame = ultimo
            objeto.centro = objeto.vistas[ultimo].centro
    return [o for o in objetos if o.vistas]


def descartar_objetos_bolsa(objetos: list[Objeto], contenedores) -> list[Objeto]:
    """La bolsa que ya se lleva no es el objeto pequeño que se guarda en el bolsillo."""
    if not contenedores:
        return objetos
    vivos = []
    for objeto in objetos:
        if not objeto.vistas:
            continue
        votos = sum(1 for v in objeto.vistas.values() if _vista_es_bolsa(v, contenedores))
        if votos / len(objeto.vistas) >= 0.5:
            print(f"  Objeto {objeto.id} se descarta: es la bolsa, no cabe en un bolsillo")
            continue
        frames = sorted(objeto.vistas)
        if len(frames) >= 3:
            huecos = [b - a for a, b in zip(frames, frames[1:])]
            if huecos and max(huecos) >= 6:
                despues = [objeto.vistas[f] for f in frames if f >= frames[len(frames)//2]]
                if despues and sum(1 for v in despues if _vista_es_bolsa(v, contenedores)) >= 0.5 * len(despues):
                    print(f"  Objeto {objeto.id} se descarta: reaparece como bolsa, no como artículo")
                    continue
        vivos.append(objeto)
    for numero, objeto in enumerate(vivos, start=1):
        objeto.id = numero
    return vivos


def descartar_objetos_que_son_mueble(objetos: list[Objeto], contenedores) -> list[Objeto]:
    """El asa del cesto o el carrito no es el artículo que se guarda."""
    if not contenedores:
        return objetos
    vivos = []
    for objeto in objetos:
        if not objeto.vistas:
            continue
        votos = 0
        for vista in objeto.vistas.values():
            for pista in contenedores:
                if pista.get("categoria") not in ("cesto", "carrito"):
                    continue
                caja = caja_contenedor(pista, vista.frame, hold=HOLD_MUEBLE)
                if caja is None:
                    continue
                if (
                    _fraccion_caja_dentro(vista.caja, caja) >= 0.55
                    or iou_cajas(vista.caja, caja) >= 0.25
                ) and _area_caja(vista.caja) > 0.32 * _area_caja(caja):
                    votos += 1
                    break
        if votos / len(objeto.vistas) >= 0.45:
            print(f"  Objeto {objeto.id} se descarta: es el cesto/carrito, no el artículo")
            continue
        vivos.append(objeto)
    for numero, objeto in enumerate(vivos, start=1):
        objeto.id = numero
    return vivos


def componente_cercana(mascara, centro, area_min: int, area_max: int, salto_max: float = 120.0):
    """Cutie a veces hincha la máscara o la salta a una estantería. Se queda la mancha útil."""
    n, etiquetas, stats, centros = cv2.connectedComponentsWithStats(
        mascara.astype(np.uint8), connectivity=8
    )
    mejor, mejor_d = None, 1e9
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_min or area > area_max:
            continue
        cx, cy = float(centros[i, 0]), float(centros[i, 1])
        dist = 0.0 if centro is None else float(np.hypot(cx - centro[0], cy - centro[1]))
        if dist < mejor_d:
            mejor, mejor_d = i, dist
    if mejor is None:
        return None
    # Un objeto no se teletransporta: si la mancha más cercana está lejos, se ha perdido.
    if centro is not None and mejor_d > salto_max:
        return None
    return etiquetas == mejor


def objetos_desde_pistas(pistas, contextos, args) -> list[Objeto]:
    """Convierte las pistas del detector en objetos, sin propagador de vídeo."""
    objetos: list[Objeto] = []
    for pista in pistas:
        objeto = Objeto(id=pista["id"], semillas=pista["mascaras"], mascaras=dict(pista["mascaras"]))
        if objeto.mascaras:
            ultimo = max(objeto.mascaras)
            ys, xs = np.nonzero(objeto.mascaras[ultimo])
            objeto.centro = (float(xs.mean()), float(ys.mean()))
            objeto.ultimo_frame = ultimo
        trazar(objeto, contextos, pista, args)
        objetos.append(objeto)
    return fusionar_objetos(objetos)


def _mano_cerca_caja(mano, caja, margen: float = 48.0) -> bool:
    palma = mano.palma or mano.punto
    return _dentro_de_caja(palma, caja, margen=margen) or _dentro_de_caja(mano.punto, caja, margen=margen)


def _mascara_de_caja(alto: int, ancho: int, caja) -> np.ndarray:
    mascara = np.zeros((alto, ancho), dtype=bool)
    if caja is None:
        return mascara
    x1, y1, x2, y2 = caja
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(ancho, int(x2)), min(alto, int(y2))
    if x2 > x1 and y2 > y1:
        mascara[y1:y2, x1:x2] = True
    return mascara


def _objeto_en_bolsillo(objeto: Objeto, contenedores, contextos=None) -> bool:
    if not objeto.vistas or not contenedores:
        return False
    for vista in objeto.vistas.values():
        for pista in contenedores:
            if not _es_bolsillo(pista):
                continue
            if _deposito_en_bolsillo(vista, pista, contextos, contenedores):
                return True
    return False


def _cerca_de_bolsillo_valido(objeto: Objeto, contenedores, contextos=None) -> bool:
    """El móvil junto al bolsillo libre (p. ej. frame 100) no se descarta por pocas detecciones."""
    if not objeto.vistas or not contenedores:
        return False
    for vista in objeto.vistas.values():
        if contextos is not None:
            ctx = _contexto_en(contextos, vista.frame)
            if ctx is not None and _torso_vertical(ctx) < 0.22:
                continue
        for pista in contenedores:
            if not _es_bolsillo(pista) or _bolsillo_ocupado_por_bolsa(pista, vista.frame, contenedores):
                continue
            caja = caja_contenedor(pista, vista.frame, hold=4)
            if caja is None:
                continue
            if _dentro_de_caja(vista.centro, caja, margen=12.0) or _fraccion_caja_dentro(vista.caja, caja) >= 0.40:
                return True
    return False


def sembrar_objetos_desde_identificadores(contenedores, contextos, args) -> list[Objeto]:
    """Un celular pegado a la palma es el objeto en la mano si el recorte de muñeca no lo vio."""
    pistas = []
    for cont in contenedores:
        if cont.get("categoria") != "celular":
            continue
        mascaras = {}
        agarres = {}
        if float(cont.get("score_max") or 0.0) < 0.32:
            continue
        for frame, caja in (cont.get("cajas") or {}).items():
            ctx = _contexto_en(contextos, frame)
            if ctx is None or not ctx.manos:
                continue
            junto_mano = any(_mano_cerca_caja(m, caja, margen=48.0) for m in (ctx.manos or []))
            junto_persona = bool(ctx.personas) and (
                iou_cajas(caja, max(ctx.personas, key=lambda p: p.altura).caja) >= 0.01
                or any(_mano_cerca_caja(m, caja, margen=72.0) for m in (ctx.manos or []))
            )
            if not junto_mano and not junto_persona:
                continue
            mascara = _mascara_de_caja(ctx.alto, ctx.ancho, caja)
            if not np.any(mascara):
                continue
            mascaras[int(frame)] = mascara
            cc = _centro_caja(caja)
            candidatas = []
            for m in ctx.manos:
                if not _mano_cerca_caja(m, caja):
                    continue
                palma = m.palma or m.punto
                candidatas.append((float(np.hypot(palma[0] - cc[0], palma[1] - cc[1])), m.lado))
            mano = min(candidatas)[1] if candidatas else None
            agarres[int(frame)] = {
                "mano": mano,
                "score": float(cont.get("score_max") or 0.4),
                "scores": {mano: 0.5} if mano else {},
            }
        if not mascaras:
            continue
        frames_ord = sorted(mascaras)
        caja_ref = cont.get("caja")
        if caja_ref is None:
            caja_ref = next(iter((cont.get("cajas") or {}).values()), (0, 0, 1, 1))
        pistas.append(
            {
                "id": len(pistas) + 1,
                "centro": _centro_caja(caja_ref),
                "primer_frame": frames_ord[0],
                "ultimo_frame": frames_ord[-1],
                "mascaras": mascaras,
                "agarres": agarres,
            }
        )
    if not pistas:
        return []
    print(f"  Objetos sembrados desde identificador celular: {len(pistas)}")
    return objetos_desde_pistas(pistas, contextos, args)


def fusionar_objetos(objetos: list[Objeto], comunes_min: int = 1, solape_min: float = 0.25):
    """Dos manos sobre la misma botella (o dos recortes) dejan un solo objeto."""
    def en_mano(objeto: Objeto) -> int:
        return sum(1 for v in objeto.vistas.values() if v.mano)

    def _mismo_objeto(a: Objeto, b: Objeto) -> bool:
        comunes = set(a.vistas) & set(b.vistas)
        if comunes:
            cerca = 0
            for f in comunes:
                va, vb = a.vistas[f], b.vistas[f]
                dist = float(np.hypot(va.centro[0] - vb.centro[0], va.centro[1] - vb.centro[1]))
                if dist < 64.0 or iou_cajas(va.caja, vb.caja) >= 0.12:
                    cerca += 1
                elif f in a.mascaras and f in b.mascaras and _solapan(a.mascaras[f], b.mascaras[f], 0.2):
                    cerca += 1
            if cerca / len(comunes) < solape_min:
                return False
            manos_a = {v.mano for v in a.vistas.values() if v.mano in ("izquierda", "derecha")}
            manos_b = {v.mano for v in b.vistas.values() if v.mano in ("izquierda", "derecha")}
            if manos_a and manos_b and manos_a.isdisjoint(manos_b):
                areas_a = [v.area for v in a.vistas.values()]
                areas_b = [v.area for v in b.vistas.values()]
                if areas_a and areas_b:
                    ra, rb = float(np.median(areas_a)), float(np.median(areas_b))
                    if min(ra, rb) > 0 and max(ra, rb) > 2.2 * min(ra, rb):
                        return False
            return True
        if not a.vistas or not b.vistas:
            return False
        fa_a, la_a = min(a.vistas), max(a.vistas)
        fa_b, la_b = min(b.vistas), max(b.vistas)
        if la_a < fa_b:
            va, vb = a.vistas[la_a], b.vistas[fa_b]
            salto = abs(int(fa_b) - int(la_a))
        elif la_b < fa_a:
            va, vb = a.vistas[fa_a], b.vistas[la_b]
            salto = abs(int(fa_a) - int(la_b))
        else:
            return False
        if salto > 12:
            return False
        dist = float(np.hypot(va.centro[0] - vb.centro[0], va.centro[1] - vb.centro[1]))
        if salto <= 8:
            return dist < 80.0
        return dist < 160.0

    supervivientes: list[Objeto] = []
    for objeto in sorted(objetos, key=lambda o: (-en_mano(o), -len(o.vistas))):
        duplicado = False
        for ya in supervivientes:
            if _mismo_objeto(objeto, ya):
                duplicado = True
                for frame, vista in objeto.vistas.items():
                    if frame in ya.vistas:
                        prev = ya.vistas[frame]
                        if vista.area < 0.55 * max(1, prev.area):
                            ya.vistas[frame] = vista
                            if frame in objeto.mascaras:
                                ya.mascaras[frame] = objeto.mascaras[frame]
                            prev = ya.vistas[frame]
                        manos = {prev.mano, vista.mano} - {None}
                        if {"izquierda", "derecha"} <= manos or "ambas" in manos:
                            ya.vistas[frame] = replace(prev, mano="ambas")
                    else:
                        vecinos = [f for f in ya.vistas if abs(int(f) - int(frame)) <= 2]
                        if vecinos:
                            dmin = min(
                                float(np.hypot(
                                    vista.centro[0] - ya.vistas[f].centro[0],
                                    vista.centro[1] - ya.vistas[f].centro[1],
                                ))
                                for f in vecinos
                            )
                            if dmin > 140.0:
                                continue
                        ya.vistas[frame] = vista
                    if frame in objeto.mascaras:
                        ya.mascaras.setdefault(frame, objeto.mascaras[frame])
                print(f"  Objeto {objeto.id} se une al {ya.id}: mismo objeto a dos manos")
                break
        if not duplicado:
            supervivientes.append(objeto)
    supervivientes.sort(key=lambda o: min(o.vistas) if o.vistas else 0)
    for numero, objeto in enumerate(supervivientes, start=1):
        objeto.id = numero
    return supervivientes


def trazar(objeto: Objeto, contextos, pista: dict, args) -> None:
    """Rellena la traza por frame: caja, centro, área y mano que lo sostiene."""
    for idx in sorted(objeto.mascaras):
        if idx < 0 or idx >= len(contextos):
            continue
        mascara = objeto.mascaras[idx]
        caja = caja_desde_mascara(mascara)
        if caja is None:
            continue
        agarre = puntuar_manos(mascara, caja, contextos[idx], args)
        ys, xs = np.nonzero(mascara)
        persona_id = -1
        lado = agarre["mano"] if agarre["mano"] in ("izquierda", "derecha") else None
        for mano in getattr(contextos[idx], "manos", []) or []:
            if lado and mano.lado == lado and mano.persona_id >= 0:
                persona_id = int(mano.persona_id)
                break
            if agarre["mano"] == "ambas" and mano.persona_id >= 0:
                persona_id = int(mano.persona_id)
        objeto.vistas[idx] = Vista(
            frame=idx,
            caja=caja,
            centro=(float(xs.mean()), float(ys.mean())),
            area=int(mascara.sum()),
            mano=agarre["mano"],
            score=round(agarre["score"], 4),
            scores=agarre["scores"],
            origen="deteccion" if idx in pista["mascaras"] else "propagacion",
            persona_id=persona_id,
        )


def _registrar_mascara(objeto: Objeto, indice: int, mascara, contextos, args,
                       origen: str = "deteccion") -> None:
    if indice < 0 or indice >= len(contextos):
        return
    caja = caja_desde_mascara(mascara)
    if caja is None:
        return
    agarre = puntuar_manos(mascara, caja, contextos[indice], args)
    ys, xs = np.nonzero(mascara)
    persona_id = -1
    ctx = contextos[indice]
    lado = agarre["mano"] if agarre["mano"] in ("izquierda", "derecha") else None
    for mano in getattr(ctx, "manos", []) or []:
        if lado and mano.lado == lado and mano.persona_id >= 0:
            persona_id = int(mano.persona_id)
            break
        if agarre["mano"] == "ambas" and mano.persona_id >= 0:
            persona_id = int(mano.persona_id)
    objeto.mascaras[indice] = mascara
    objeto.vistas[indice] = Vista(
        frame=indice,
        caja=caja,
        centro=(float(xs.mean()), float(ys.mean())),
        area=int(mascara.sum()),
        mano=agarre["mano"],
        score=round(agarre["score"], 4),
        scores=agarre["scores"],
        origen=origen,
        persona_id=persona_id,
    )
    objeto.ultimo_frame = max(objeto.ultimo_frame, indice)
    objeto.centro = (float(xs.mean()), float(ys.mean()))


def _detectar_en_deposito(frame, contexto, detector, sam, caja, args, area_ref, centro_ref,
                         estricto_area: bool = False):
    """Busca un objeto compacto en el contenedor o en la mano que está dentro."""
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = caja
    caja_e = (
        max(0, int(x1) - 24),
        max(0, int(y1) - 24),
        min(ancho, int(x2) + 24),
        min(alto, int(y2) + 24),
    )
    candidatos = list(detector.detectar_en_caja(frame, caja_e, args.conf_objeto_crop))
    for mano in contexto.manos:
        palma = mano.palma or mano.punto
        if not _dentro_de_caja(palma, caja, margen=50):
            continue
        rec = caja_muneca(mano, args.padding_muneca, args.crop_min, ancho, alto)
        recorte = rec[:4]
        candidatos.extend(detector.detectar_en_caja(frame, recorte, args.conf_objeto_crop))
        if sam is not None:
            candidatos.extend(_sam_en_palma(sam, detector, frame, mano, recorte, args))
    mejor, mejor_s = None, 1e9
    area_ref = max(1.0, float(area_ref))
    for inst in candidatos:
        if inst.es_persona or inst.area < args.area_min:
            continue
        if inst.area > max(float(args.area_max_px), 4.0 * area_ref):
            continue
        if inst.area > 5.0 * area_ref:
            continue
        if estricto_area and inst.area > 2.6 * area_ref:
            continue
        if _cubre_torso(inst, contexto):
            continue
        if inst.area < 0.2 * area_ref:
            continue
        if es_voluminoso(inst, contexto) or _cubre_torso(inst, contexto) or not _forma_de_objeto(inst, args):
            continue
        en_destino = _dentro_de_caja(inst.centro, caja, margen=30)
        en_mano = any(_pegado_a_palma(inst, m, args) for m in contexto.manos)
        if not en_destino and not en_mano:
            continue
        dist = float(np.hypot(inst.centro[0] - centro_ref[0], inst.centro[1] - centro_ref[1]))
        score = dist + 0.002 * abs(inst.area - area_ref)
        if score < mejor_s:
            mejor, mejor_s = inst, score
    return mejor


def _alinear_aproximacion_a_mueble(objetos, contenedores, contextos):
    """Entre la última mano y el cesto no se acepta un salto al bolso/cadera."""
    if not objetos or not contenedores:
        return objetos
    for objeto in objetos:
        frames = sorted(objeto.vistas)
        en_mueble = [
            f for f in frames
            if _vista_en_mueble(objeto.vistas[f], contenedores, contextos)
        ]
        if not en_mueble:
            continue
        primero = min(en_mueble)
        previos = []
        for f in frames:
            if f >= primero or not objeto.vistas[f].mano:
                continue
            if _vista_es_bolsa(objeto.vistas[f], contenedores):
                continue
            accesorio = False
            for pista in contenedores:
                if not (pista.get("accesorio") or _es_bolsillo(pista)):
                    continue
                caja_a = caja_contenedor(pista, f, hold=6)
                if caja_a is None:
                    continue
                if (
                    iou_cajas(objeto.vistas[f].caja, caja_a) >= 0.20
                    or _fraccion_caja_dentro(objeto.vistas[f].caja, caja_a) >= 0.35
                ):
                    accesorio = True
                    break
            if not accesorio:
                previos.append(f)
        if not previos:
            continue
        ultimo_mano = max(previos)
        va, vc = objeto.vistas[ultimo_mano], objeto.vistas[primero]
        span = float(max(1, primero - ultimo_mano))
        for f in [k for k in objeto.vistas if ultimo_mano < k < primero]:
            vb = objeto.vistas[f]
            t = (f - ultimo_mano) / span
            pred = (
                va.centro[0] + t * (vc.centro[0] - va.centro[0]),
                va.centro[1] + t * (vc.centro[1] - va.centro[1]),
            )
            if float(np.hypot(vb.centro[0] - pred[0], vb.centro[1] - pred[1])) > 72.0:
                objeto.vistas.pop(f, None)
                objeto.mascaras.pop(f, None)
        if objeto.vistas:
            ultimo = max(objeto.vistas)
            objeto.ultimo_frame = ultimo
            objeto.centro = objeto.vistas[ultimo].centro
    return objetos


def rellenar_huecos_en_mano(frames, objetos, contextos, detector, sam, args, contenedores=None):
    """Si el objeto sigue en la mano, un frame suelto no debe romper el seguimiento."""
    if detector is None or not objetos:
        return objetos
    for objeto in objetos:
        if len(objeto.vistas) < 2:
            continue
        primero, ultimo = min(objeto.vistas), max(objeto.vistas)
        for indice in range(int(primero) + 1, int(ultimo)):
            if indice in objeto.vistas:
                continue
            if indice >= len(frames) or indice >= len(contextos):
                continue
            prev = objeto.vistas[max(f for f in objeto.vistas if f < indice)]
            if prev.mano is None:
                continue
            if indice - int(prev.frame) > args.hueco_parpadeo + 2:
                continue
            ctx = contextos[indice]
            if not getattr(ctx, "manos", None):
                continue
            siguientes = [f for f in objeto.vistas if f > indice]
            nxt = objeto.vistas[min(siguientes)] if siguientes else None
            if nxt is None:
                continue
            hueco = int(nxt.frame) - int(prev.frame)
            hacia_mueble = _vista_en_mueble(nxt, contenedores, contextos)
            if hueco > (8 if hacia_mueble or nxt.mano else 2):
                continue
            t = (indice - int(prev.frame)) / float(hueco)
            pred = (
                prev.centro[0] + t * (nxt.centro[0] - prev.centro[0]),
                prev.centro[1] + t * (nxt.centro[1] - prev.centro[1]),
            )
            mejor = None
            if not hacia_mueble:
                try:
                    hallados = objetos_del_frame(frames[indice], ctx, detector, sam, copy(args))
                except Exception as exc:
                    _aviso(f"frame {indice}: hueco en mano falló ({exc})")
                    hallados = []
                mejor, mejor_d = None, 1e9
                for inst, _agr in hallados:
                    if _cubre_torso(inst, ctx):
                        continue
                    dist = float(np.hypot(inst.centro[0] - pred[0], inst.centro[1] - pred[1]))
                    if dist < mejor_d:
                        mejor, mejor_d = inst, dist
                if mejor is not None and (mejor_d > 72.0 or (
                    prev.area > 0 and mejor.area > 3.5 * max(1, prev.area)
                )):
                    mejor = None
            if mejor is not None:
                _registrar_mascara(objeto, indice, mejor.mascara, contextos, args)
            else:
                caja = _interpolar_caja(prev.caja, nxt.caja, int(prev.frame), int(nxt.frame), indice)
                mascara = _mascara_de_caja(int(ctx.alto), int(ctx.ancho), caja)
                if int(mascara.sum()) <= 0:
                    continue
                _registrar_mascara(objeto, indice, mascara, contextos, args, origen="propagacion")
            vista = objeto.vistas.get(indice)
            if vista is not None and vista.mano is None and not hacia_mueble:
                objeto.vistas.pop(indice, None)
                objeto.mascaras.pop(indice, None)
    return objetos


def _mano_en_cadera_actual(contexto, lado: str | None) -> bool:
    if contexto is None or not getattr(contexto, "personas", None) or not lado:
        return False
    persona = max(contexto.personas, key=lambda p: p.altura)
    hip = _y_cadera(contexto, lado)
    mano = _mano_del_lado(contexto, lado)
    if hip is None or mano is None:
        return False
    palma = mano.palma or mano.punto
    return abs(float(palma[1]) - float(hip)) <= 0.20 * max(float(persona.altura), 1.0)


def _cerrar_si_entra_bolsillo(objetos: list[Objeto], contenedores, contextos) -> list[Objeto]:
    """Si el móvil entra en el bolsillo, no se sigue arrastrando el ROI fuera de él."""
    if not objetos or not contenedores or contextos is None:
        return objetos
    for objeto in objetos:
        if not objeto.vistas:
            continue
        if not _objeto_es_celular(objeto, contenedores):
            continue
        corte = None
        for f in sorted(objeto.vistas):
            for pista in contenedores:
                if not _es_bolsillo(pista):
                    continue
                if _gesto_mano_en_bolsillo(contextos, f, pista, contenedores=contenedores):
                    corte = f
                    break
            if corte is not None:
                break
        if corte is None:
            continue
        for k in [i for i in objeto.vistas if i > corte + 2]:
            objeto.vistas.pop(k, None)
            objeto.mascaras.pop(k, None)
        if objeto.vistas:
            ultimo = max(objeto.vistas)
            objeto.ultimo_frame = ultimo
            objeto.centro = objeto.vistas[ultimo].centro
    return [o for o in objetos if o.vistas]


def extender_si_sigue_en_mano(frames, objetos, contextos, detector, sam, args, contenedores=None):
    """Si el objeto sigue visible en la misma palma, no se corta la traza al fallar un frame."""
    if detector is None or not objetos:
        return objetos
    extra = max(12, int(getattr(args, "hueco_parpadeo", 3) or 3) + 8)
    for objeto in objetos:
        if len(objeto.vistas) < 2:
            continue
        ultima = max(objeto.vistas)
        vista = objeto.vistas[ultima]
        if vista.mano not in ("izquierda", "derecha", "ambas") or vista.area < 80:
            continue
        if _vista_en_mueble(vista, contenedores, contextos):
            continue
        lado = vista.mano if vista.mano in ("izquierda", "derecha") else None
        for indice in range(int(ultima) + 1, min(len(frames), int(ultima) + extra + 1)):
            if indice >= len(contextos) or indice in objeto.vistas:
                continue
            ctx = contextos[indice]
            if not getattr(ctx, "manos", None):
                break
            if lado and any(
                getattr(m, "abierta", False) and m.lado == lado for m in ctx.manos
            ):
                break
            try:
                hallados = objetos_del_frame(frames[indice], ctx, detector, sam, copy(args))
            except Exception as exc:
                _aviso(f"frame {indice}: extensión en mano falló ({exc})")
                break
            mejor, mejor_d = None, 1e9
            for inst, agr in hallados:
                if _cubre_torso(inst, ctx) or _parece_brazo_o_suelo(inst, ctx):
                    continue
                mano_agr = agr.get("mano")
                if lado and mano_agr not in (lado, "ambas"):
                    continue
                dist = float(np.hypot(inst.centro[0] - vista.centro[0], inst.centro[1] - vista.centro[1]))
                if vista.area > 0 and inst.area > 3.2 * max(1, vista.area):
                    continue
                if dist < mejor_d:
                    mejor, mejor_d = inst, dist
            if mejor is not None and mejor_d <= 96.0:
                _registrar_mascara(objeto, indice, mejor.mascara, contextos, args)
                vista = objeto.vistas.get(indice, vista)
                continue
            mano = _mano_del_lado(ctx, lado) if lado else None
            if mano is None or getattr(mano, "abierta", False):
                break
            palma = mano.palma or mano.punto
            dx = float(palma[0] - vista.centro[0])
            dy = float(palma[1] - vista.centro[1])
            if float(np.hypot(dx, dy)) > 72.0:
                break
            x1, y1, x2, y2 = vista.caja
            nueva = _clip_caja((x1 + dx, y1 + dy, x2 + dx, y2 + dy), ctx.ancho, ctx.alto)
            if nueva is None:
                break
            mascara = _mascara_de_caja(int(ctx.alto), int(ctx.ancho), nueva)
            if int(mascara.sum()) <= 0:
                break
            _registrar_mascara(objeto, indice, mascara, contextos, args, origen="propagacion")
            vista = objeto.vistas.get(indice, vista)
    return objetos


def _parece_mano_vacia(objeto: Objeto, contextos) -> bool:
    """Palma o manga sin artículo: no se inventa un objeto para llevarlo al bolsillo."""
    if not objeto.vistas:
        return True
    votos = 0
    for frame, vista in objeto.vistas.items():
        ctx = _contexto_en(contextos, int(frame)) if contextos is not None else None
        w = max(1.0, float(vista.caja[2] - vista.caja[0]))
        h = max(1.0, float(vista.caja[3] - vista.caja[1]))
        if vista.area < 160 and max(w, h) / min(w, h) > 2.0:
            votos += 1
            continue
        if ctx is None or not getattr(ctx, "manos", None):
            continue
        mano = _mano_del_lado(ctx, vista.mano) if vista.mano in ("izquierda", "derecha") else None
        if mano is None:
            continue
        mascara = objeto.mascaras.get(int(frame))
        if mascara is None:
            continue
        inst = SimpleNamespace(caja=vista.caja, mascara=mascara, area=vista.area, centro=vista.centro)
        if mascara_es_la_mano(inst, mano):
            votos += 1
    return votos / max(1, len(objeto.vistas)) >= 0.5


def completar_deposito(frames, objetos, contenedores, contextos, detector, sam, args):
    """Si la mano entra en un cesto/carrito (o bolsillo con gesto), sigue el objeto."""
    hay_mueble = bool(_destinos_deposito(contenedores))
    hay_bolsillo = any(_es_bolsillo(p) for p in contenedores)
    if (not hay_mueble and not hay_bolsillo) or not objetos or detector is None:
        return objetos
    activos = [o for o in objetos if o.vistas]
    for indice, frame in enumerate(frames):
        if indice >= len(contextos):
            break
        ctx = contextos[indice]
        destinos = _destinos_para_frame(contenedores, contextos, indice)
        if not destinos:
            continue
        pares = []
        for p in destinos:
            caja = caja_contenedor(p, indice)
            if caja is None:
                continue
            if _mueble_sobre_cuerpo(p, caja, ctx):
                continue
            if _es_bolsillo(p):
                if _persona_sale_de_escena(ctx):
                    continue
                if not _gesto_mano_en_bolsillo(contextos, indice, p, contenedores=contenedores):
                    continue
            pares.append((p, caja))
        if not any(
            _mano_en_caja(ctx, caja, margen=36 if _es_rinonera(p, contextos, indice) else 50)
            for p, caja in pares
        ):
            continue
        pendientes = []
        for objeto in activos:
            if indice in objeto.vistas:
                continue
            ultima = max(objeto.vistas)
            extra_hueco = 12
            if indice < ultima or indice - ultima > args.hueco_parpadeo + extra_hueco:
                continue
            pendientes.append(objeto)
        if not pendientes:
            continue
        for pista, caja in pares:
            if caja is None:
                continue
            def _dist_obj(objeto, caja=caja, indice=indice):
                vista = objeto.vistas[max(f for f in objeto.vistas if f <= indice)]
                cx = 0.5 * (caja[0] + caja[2])
                cy = 0.5 * (caja[1] + caja[3])
                return float(np.hypot(vista.centro[0] - cx, vista.centro[1] - cy))

            pendientes.sort(key=_dist_obj)
            objeto = pendientes[0]
            vista = objeto.vistas[max(f for f in objeto.vistas if f <= indice)]
            try:
                inst = _detectar_en_deposito(
                    frame, ctx, detector, sam, caja, args, vista.area, vista.centro,
                    estricto_area=_es_bolsillo(pista) or _es_rinonera(pista, contextos, indice),
                )
            except Exception as exc:
                _aviso(f"frame {indice}: depósito falló ({exc})")
                continue
            rinonera = _es_rinonera(pista, contextos, indice)
            if inst is None and (rinonera or _es_bolsillo(pista)) and _mano_en_caja(ctx, caja, margen=28.0):
                t = min(1.0, (indice - ultima) / 6.0)
                cx = vista.centro[0] + t * (0.5 * (caja[0] + caja[2]) - vista.centro[0])
                cy = vista.centro[1] + t * (0.5 * (caja[1] + caja[3]) - vista.centro[1])
                ow = max(12.0, (vista.caja[2] - vista.caja[0]) * (1.0 - 0.35 * t))
                oh = max(16.0, (vista.caja[3] - vista.caja[1]) * (1.0 - 0.35 * t))
                nueva = _clip_caja((cx - ow / 2.0, cy - oh / 2.0, cx + ow / 2.0, cy + oh / 2.0), ctx.ancho, ctx.alto)
                mascara = _mascara_de_caja(int(ctx.alto), int(ctx.ancho), nueva)
                if int(mascara.sum()) > 0:
                    _registrar_mascara(objeto, indice, mascara, contextos, args, origen="propagacion")
                    pendientes.pop(0)
                    if not pendientes:
                        break
                continue
            if inst is None:
                continue
            dist = float(
                np.hypot(inst.centro[0] - vista.centro[0], inst.centro[1] - vista.centro[1])
            )
            limite = float(args.distancia_max)
            if _es_bolsillo(pista):
                limite = min(limite, 90.0)
            elif rinonera:
                limite = min(limite, 220.0)
            if dist > limite:
                continue
            _registrar_mascara(objeto, indice, inst.mascara, contextos, args)
            pendientes.pop(0)
            if not pendientes:
                break
    for objeto in activos:
        if not objeto.vistas:
            continue
        ultima = max(objeto.vistas)
        vista = objeto.vistas[ultima]
        for indice in range(int(ultima) + 1, min(len(frames), int(ultima) + 8)):
            if indice >= len(contextos) or indice in objeto.vistas:
                continue
            ctx = contextos[indice]
            destinos = _destinos_para_frame(contenedores, contextos, indice)
            rinoneras = [
                (p, caja_contenedor(p, indice))
                for p in destinos if _es_rinonera(p, contextos, indice)
            ]
            rinoneras = [(p, c) for p, c in rinoneras if c is not None and _mano_en_caja(ctx, c, 36.0)]
            if not rinoneras:
                continue
            _, caja = rinoneras[0]
            t = min(1.0, (indice - ultima) / 5.0)
            cx = vista.centro[0] + t * (0.5 * (caja[0] + caja[2]) - vista.centro[0])
            cy = vista.centro[1] + t * (0.5 * (caja[1] + caja[3]) - vista.centro[1])
            ow = max(12.0, (vista.caja[2] - vista.caja[0]) * (1.0 - 0.3 * t))
            oh = max(16.0, (vista.caja[3] - vista.caja[1]) * (1.0 - 0.3 * t))
            nueva = _clip_caja((cx - ow / 2.0, cy - oh / 2.0, cx + ow / 2.0, cy + oh / 2.0), ctx.ancho, ctx.alto)
            mascara = _mascara_de_caja(int(ctx.alto), int(ctx.ancho), nueva)
            if int(mascara.sum()) <= 0:
                continue
            _registrar_mascara(objeto, indice, mascara, contextos, args, origen="propagacion")
            vista = objeto.vistas.get(indice, vista)
    return objetos


# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #


def eventos_de(objeto: Objeto, fps: float, args) -> list[dict]:
    """Aparece, cambia de mano, suelta, desaparece y reaparece."""
    vistas = [objeto.vistas[i] for i in sorted(objeto.vistas)]
    if not vistas:
        return []
    eventos: list[dict] = []

    def añadir(tipo: str, vista: Vista, extra: dict | None = None) -> None:
        evento = {
            "tipo": tipo,
            "frame": vista.frame,
            "t": round(vista.frame / fps, 4) if fps else None,
            "mano": vista.mano,
            "centro": [round(vista.centro[0], 2), round(vista.centro[1], 2)],
            "roi": list(vista.caja),
        }
        if extra:
            evento.update(extra)
        eventos.append(evento)

    añadir("aparece", vistas[0])
    mano = vistas[0].mano
    candidata, repeticiones = None, 0
    anterior = vistas[0]
    for vista in vistas[1:]:
        if vista.frame - anterior.frame > args.hueco_parpadeo:
            añadir("desaparece", anterior)
            añadir("reaparece", vista)
            mano = vista.mano
            candidata, repeticiones = None, 0
            anterior = vista
            continue
        if vista.mano != mano:
            if candidata == vista.mano:
                repeticiones += 1
            else:
                candidata, repeticiones = vista.mano, 1
            if repeticiones >= args.frames_traspaso:
                if candidata is None:
                    añadir("suelta", vista, {"mano_anterior": mano})
                elif mano is None:
                    añadir("reagarre", vista)
                else:
                    añadir("cambio_de_mano", vista, {"desde": mano, "hasta": candidata})
                mano = candidata
                candidata, repeticiones = None, 0
        else:
            candidata, repeticiones = None, 0
        anterior = vista
    return eventos


def estado_final(objeto: Objeto, nframes: int, contenedor: dict | None, args,
                 contextos=None) -> str:
    del nframes
    vistas = [objeto.vistas[i] for i in sorted(objeto.vistas)]
    if not vistas:
        return "no_data"
    ultima = vistas[-1]
    if contenedor is not None:
        return "inside_container"
    sale = False
    if contextos is not None:
        for ctx in contextos[max(0, ultima.frame) :]:
            if not getattr(ctx, "personas", None):
                continue
            persona = max(ctx.personas, key=lambda p: p.altura)
            if (
                _persona_sale_de_escena(ctx)
                or _de_espaldas(ctx)
                or persona.caja[3] >= int(ctx.alto) - 10
            ):
                sale = True
                break
    if ultima.mano is not None or (sale and any(v.mano for v in vistas)):
        return "in_hand"
    return "released"


def _final_state_video(informes: list[dict]) -> str:
    if not informes:
        return "no_data"
    rango = {
        "inside_container": 0,
        "in_hand": 1,
        "released": 2,
        "disappeared": 3,
        "no_data": 4,
        "no_hands": 5,
    }
    return min(
        informes,
        key=lambda info: (
            rango.get(info.get("final_state"), 9),
            -(info.get("last_visible_frame") or 0),
        ),
    )["final_state"]


# --------------------------------------------------------------------------- #
# Vídeo anotado
# --------------------------------------------------------------------------- #


def _etiqueta_mano(mano: str | None) -> str:
    return MANO_HUD.get(mano or "sin_mano", "FREE")


def _overlay_objeto(objeto: Objeto, indice: int, contenedores, contextos=None) -> tuple:
    """Máscara del frame o, si ya está en un destino (no accesorio), la última vista en el ROI."""
    vista = objeto.vistas.get(indice)
    mascara = objeto.mascaras.get(indice)
    if vista is not None and mascara is not None:
        return vista, mascara
    destinos = _destinos_para_frame(contenedores, contextos, indice)
    if not destinos:
        return None, None
    anteriores = [f for f in objeto.vistas if f <= indice]
    if not anteriores:
        return None, None
    ultimo = max(anteriores)
    if indice - ultimo > 8:
        return None, None
    vista = objeto.vistas[ultimo]
    mascara = objeto.mascaras.get(ultimo)
    if mascara is None:
        return None, None
    for pista in destinos:
        caja = caja_contenedor(pista, indice)
        fraccion, centro_dentro = _objeto_en_contenedor(vista, caja)
        if centro_dentro or fraccion >= 0.35:
            return vista, mascara
    return None, None


def anotar(frames, destino: Path, objetos: list[Objeto], contenedores, fps_salida: float,
           fps_video: float, indices_origen: list[int] | None,
           carpeta_frames: Path | None, alfa: float,
           contextos=None, dibujar_esqueleto: bool = True,
           prefijo_frame: str | None = None) -> int:
    """Máscaras de objeto, ROI de identificadores y pose YOLO de la cintura hacia arriba."""
    if not frames:
        return 0
    alto, ancho = frames[0].shape[:2]
    escritor = cv2.VideoWriter(
        str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps_salida or 25.0, (ancho, alto)
    )
    if not escritor.isOpened():
        _aviso(f"no se pudo crear el vídeo {destino}; se escriben solo frames si aplica")
        escritor = None
    if carpeta_frames is not None:
        carpeta_frames.mkdir(parents=True, exist_ok=True)
    escritos = 0
    try:
        for indice, original in enumerate(frames):
            frame = original.copy()
            orig = frame_origen(indice, indices_origen)
            ctx_i = contextos[indice] if contextos is not None and indice < len(contextos) else None
            manos_ocupadas = set()
            for objeto in objetos:
                vista_o = objeto.vistas.get(indice)
                if vista_o and vista_o.mano in ("izquierda", "derecha", "ambas"):
                    if vista_o.mano == "ambas":
                        manos_ocupadas.update(("izquierda", "derecha"))
                    else:
                        manos_ocupadas.add(vista_o.mano)
            for pista in contenedores:
                if pista.get("accesorio") or pista.get("cuerpo"):
                    hold = HOLD_ACCESORIO
                elif pista.get("destino"):
                    hold = HOLD_MUEBLE
                else:
                    hold = 0
                caja = caja_contenedor(pista, indice, hold=hold)
                if caja is None:
                    continue
                if (
                    _mueble_sobre_cuerpo(pista, caja, ctx_i)
                    and indice not in (pista.get("cajas") or {})
                ):
                    continue
                if _es_bolsillo(pista) and (
                    _bolsillo_ocupado_por_bolsa(pista, indice, contenedores)
                    or not _bolsillo_activo(pista, indice, contextos, contenedores)
                    or (_lado_pista(pista) in manos_ocupadas)
                    or (ctx_i is not None and _persona_sale_de_escena(ctx_i))
                    or any(
                        vo is not None
                        and vo.mano is not None
                        and (
                            iou_cajas(vo.caja, caja) >= 0.04
                            or _fraccion_caja_dentro(vo.caja, caja) >= 0.12
                        )
                        for objeto in objetos
                        for vo in (objeto.vistas.get(indice),)
                    )
                ):
                    continue
                if pista.get("accesorio") and ctx_i is not None and not _accesorio_en_persona(caja, ctx_i):
                    continue
                x1, y1, x2, y2 = caja
                color = COLOR_CONTENEDOR if pista.get("destino") else COLOR_IDENTIFICADOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, pista["id"], (x1 + 4, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
                )
            for objeto in objetos:
                vista, mascara = _overlay_objeto(objeto, indice, contenedores, contextos)
                if vista is None or mascara is None:
                    continue
                if mascara.ndim != 2 or mascara.shape[:2] != frame.shape[:2]:
                    continue
                color = COLOR_OBJETO[(objeto.id - 1) % len(COLOR_OBJETO)]
                try:
                    frame[mascara] = (
                        (1.0 - alfa) * frame[mascara] + alfa * np.array(color, dtype=np.float32)
                    ).astype(np.uint8)
                except Exception as exc:
                    _aviso(f"frame {orig}: overlay de objeto {objeto.id} falló ({exc})")
                x1, y1, x2, y2 = vista.caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"obj {objeto.id} {_etiqueta_mano(vista.mano)} {vista.score:.2f}",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )
            if dibujar_esqueleto and contextos is not None and indice < len(contextos):
                try:
                    dibujar_pose(frame, contextos[indice])
                except Exception as exc:
                    _aviso(f"frame {orig}: esqueleto falló ({exc})")
            cv2.rectangle(frame, (0, 0), (ancho, 34), (0, 0, 0), -1)
            activos = sum(
                1 for o in objetos
                if _overlay_objeto(o, indice, contenedores)[0] is not None
            )
            t = orig / fps_video if fps_video else 0.0
            cv2.putText(
                frame,
                f"frame {orig}  t={t:.2f}s  handheld objects: {activos}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
            )
            if escritor is not None:
                escritor.write(frame)
            if carpeta_frames is not None:
                prefijo = prefijo_frame or "frame"
                cv2.imwrite(
                    str(carpeta_frames / f"{prefijo}_frame_{orig:05d}.jpg"),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
            escritos += 1
    finally:
        if escritor is not None:
            escritor.release()
    return escritos


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #


def informe_objeto(objeto: Objeto, fps: float, nframes: int, contenedores, args,
                   indices_origen: list[int] | None = None, contextos=None) -> dict:
    vistas = [objeto.vistas[i] for i in sorted(objeto.vistas)]
    contenedor = contenedor_final(vistas, contenedores, args, contextos)
    manos = {"izquierda": 0, "derecha": 0, "ambas": 0, "sin_mano": 0}
    for vista in vistas:
        manos[vista.mano or "sin_mano"] += 1

    def orig(idx: int) -> int:
        return frame_origen(idx, indices_origen)

    linea = [
        {
            "frame": orig(v.frame),
            "processed_frame": v.frame,
            "t": round(orig(v.frame) / fps, 4) if fps else None,
            "hand": _mano_en(v.mano),
            "roi": list(v.caja),
            "center": [round(v.centro[0], 2), round(v.centro[1], 2)],
            "area": v.area,
            "grip_score": v.score,
            "hand_scores": {_mano_en(k) or k: val for k, val in (v.scores or {}).items()},
            "origin": "detection" if v.origen == "deteccion" else "propagation",
        }
        for v in vistas
    ]
    en_mano = [v for v in vistas if v.mano]
    eventos = eventos_de(objeto, fps, args)
    eventos_en = []
    for evento in eventos:
        interno = evento["frame"]
        item = {
            "type": EVENTO_EN.get(evento["tipo"], evento["tipo"]),
            "frame": orig(interno),
            "processed_frame": interno,
            "t": round(orig(interno) / fps, 4) if fps else None,
            "hand": _mano_en(evento.get("mano")),
            "center": evento.get("centro"),
            "roi": evento.get("roi"),
        }
        if "desde" in evento:
            item["from"] = _mano_en(evento["desde"])
        if "hasta" in evento:
            item["to"] = _mano_en(evento["hasta"])
        if "mano_anterior" in evento:
            item["previous_hand"] = _mano_en(evento["mano_anterior"])
        eventos_en.append(item)
    manos_en = {
        "left": manos["izquierda"],
        "right": manos["derecha"],
        "both": manos["ambas"],
        "no_hand": manos["sin_mano"],
    }
    dominante = max(manos, key=manos.get) if vistas else None
    return {
        "id": objeto.id,
        "frames_visible": len(vistas),
        "frames_in_hand": len(en_mano),
        "first_frame": orig(vistas[0].frame) if vistas else None,
        "last_frame": orig(vistas[-1].frame) if vistas else None,
        "reseeds": objeto.resiembras,
        "hands": manos_en,
        "dominant_hand": _mano_en(dominante),
        "last_roi": list(vistas[-1].caja) if vistas else None,
        "last_center": (
            [round(vistas[-1].centro[0], 2), round(vistas[-1].centro[1], 2)] if vistas else None
        ),
        "last_visible_frame": orig(vistas[-1].frame) if vistas else None,
        "final_state": estado_final(objeto, nframes, contenedor, args, contextos),
        "ends_in_container": contenedor is not None,
        "final_container": contenedor,
        "person_id": (pid + 1) if (pid := _persona_id_objeto(objeto)) >= 0 else None,
        "events": eventos_en,
        "timeline": linea,
    }


def _persona_id_objeto(objeto: Objeto) -> int:
    votos: dict[int, int] = {}
    for vista in objeto.vistas.values():
        pid = int(getattr(vista, "persona_id", -1))
        if pid >= 0:
            votos[pid] = votos.get(pid, 0) + 1
    if not votos:
        return -1
    return max(votos, key=votos.get)


def informes_personas(contextos, informes: list[dict], objetos: list[Objeto]) -> list[dict]:
    """Solo personas con muñecas visibles. Una entrada por track estable."""
    stats: dict[int, dict] = {}
    for ctx in contextos or []:
        for mano in getattr(ctx, "manos", []) or []:
            if mano.persona_id < 0 or mano.lado not in ("izquierda", "derecha"):
                continue
            item = stats.setdefault(int(mano.persona_id), {"frames": 0, "left": 0, "right": 0})
            item["frames"] += 1
            item["left" if mano.lado == "izquierda" else "right"] += 1
    salida = []
    tope = max((d["frames"] for d in stats.values()), default=0)
    for tid, dato in sorted(stats.items()):
        if dato["frames"] < 2:
            continue
        if tope and dato["frames"] < max(2, int(0.18 * tope)):
            continue
        objs = [
            info for obj, info in zip(objetos, informes)
            if _persona_id_objeto(obj) == tid
        ]
        salida.append({
            "id": tid + 1,
            "hands_visible": True,
            "frames_with_wrists": dato["frames"],
            "object_ids": [info["id"] for info in objs],
            "final_state": _final_state_video(objs) if objs else "no_object",
        })
    return salida


def resumen(video: Path, informes: list[dict], contenedores, nframes: int, modelo: str,
            yaml_identificadores: Path | None = None, muestreo: dict | None = None,
            final_state: str | None = None, pose_backend: str = "yolo-pose",
            personas: list[dict] | None = None) -> str:
    lineas = [
        f"Video: {video}",
        f"Processed frames: {nframes}",
        f"Model: {modelo}. Wrists come from {pose_backend}; the object from the wrist "
        f"crop; scene identifiers from the YAML "
        f"({yaml_identificadores or YAML_IDENTIFICADORES}).",
        (muestreo or {}).get(
            "description",
            "--fps was not set. Processing ran frame by frame.",
        ),
        "",
    ]
    if contenedores:
        lineas.append("Scene identifiers:")
        for pista in contenedores:
            extra = []
            if pista.get("destino"):
                extra.append("destination")
            if pista.get("accesorio"):
                extra.append("accessory")
            if pista.get("cuerpo"):
                extra.append("body")
            sufijo = f" [{', '.join(extra)}]" if extra else ""
            roi = list(pista["caja"]) if pista.get("caja") is not None else []
            lineas.append(
                f"  {pista['id']}: ROI {roi}, max score {pista['score_max']:.2f}, "
                f"seen in {pista['frames_visto']} samples{sufijo}"
            )
    else:
        lineas.append("No active identifier from the YAML was detected.")
    lineas.append("")
    if not informes:
        if final_state == "no_hands":
            lineas.append("No wrists were visible. final_state: no_hands")
        else:
            lineas.append("No handheld object.")
        return "\n".join(lineas) + "\n"
    for info in informes:
        manos = info["hands"]
        lineas.append(
            f"Object {info['id']}: visible in {info['frames_visible']} frames, "
            f"in hand in {info['frames_in_hand']} "
            f"(left {manos['left']}, right {manos['right']}, both {manos['both']})"
        )
        for evento in info["events"]:
            detalle = ""
            if evento["type"] == "hand_change":
                detalle = f" from {evento.get('from')} to {evento.get('to')}"
            lineas.append(
                f"  frame {evento['frame']} ({evento['t']} s): {evento['type']}{detalle}"
            )
        lineas.append(f"  last ROI: {info['last_roi']} at frame {info['last_frame']}")
        if info["final_container"]:
            cont = info["final_container"]
            lineas.append(
                f"  ends inside {cont['id']} ({cont['category']}), ROI {cont['roi']}, "
                f"{100 * cont['fraction_inside']:.0f} % of the object inside"
            )
        else:
            lineas.append("  does not end inside any container")
        lineas.append(f"  final_state: {info['final_state']}")
    if personas:
        lineas.append("")
        lineas.append(f"People with visible wrists: {len(personas)}")
        for persona in personas:
            ids = persona.get("object_ids") or []
            lineas.append(
                f"  person {persona['id']}: objects {ids or 'none'}, "
                f"final_state {persona.get('final_state')}"
            )
    lineas.append(
        "\nLeft and right are the person's hands. «both» means both hands beat the "
        "grip threshold on the same frame."
    )
    return "\n".join(lineas) + "\n"


# --------------------------------------------------------------------------- #
# Proceso compartido
# --------------------------------------------------------------------------- #


def anadir_argumentos_comunes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--videos", type=Path, default=Path("videos"), help="Carpeta de vídeos.")
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Un único mp4 (opcional si args.frames / frames= es una list[np.ndarray]). "
             "Con arrays en memoria solo nombra la carpeta de salida.",
    )
    parser.add_argument("--salida", type=Path, default=Path("salidas"))
    parser.add_argument(
        "--width", type=int, default=None,
        help="Ancho en píxeles; debe ser un entero. Con --frames es el de los arrays "
             "(si se omite se infiere). Con mp4, si se pasa debe coincidir con el archivo.",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Alto en píxeles; debe ser un entero. Con --frames es el de los arrays "
             "(si se omite se infiere). Con mp4, si se pasa debe coincidir con el archivo.",
    )
    parser.add_argument(
        "--frames-origin", dest="frames_origin", nargs="+", type=int, default=None, metavar="I",
        help="Índices de origen (enteros) de cada array en --frames. Misma longitud que la lista de frames.",
    )
    parser.set_defaults(frames=None)  # list[np.ndarray] BGR; API Python, no flag de shell
    parser.add_argument("--modelo-pose", type=Path, default=PESOS / "yolo26n-pose.pt")
    parser.add_argument("--modelo-objeto", type=Path, default=PESOS / "yolo26n-seg.pt")
    parser.add_argument("--modelo-contenedores", type=Path, default=PESOS / "yoloe-26m-seg.pt")
    parser.add_argument(
        "--identificadores", type=Path, default=YAML_IDENTIFICADORES,
        help="YAML de identificadores (identifiers.yaml). Claves en inglés.",
    )
    parser.add_argument(
        "--mediapipe", action="store_true",
        help="Usa el esqueleto MediaPipe (Pose + Hands) en lugar de YOLO-pose. "
             "El umbral de muñecas es hands.min_confidence en el YAML.",
    )
    parser.add_argument("--sin-sam", action="store_true", help="No usar MobileSAM en la palma.")
    parser.add_argument(
        "--sin-contenedores", action="store_true",
        help="No detectar identificadores de escena (ignora el YAML).",
    )
    parser.add_argument("--sin-frames", action="store_true", help="No volcar los frames sueltos.")
    parser.add_argument(
        "--sin-pose", action="store_true",
        help="No dibujar el esqueleto YOLO (cintura, brazos y manos) en el vídeo anotado.",
    )
    parser.add_argument("--padding-muneca", type=float, default=1.2,
                        help="Padding del recorte de muñeca, relativo al radio del brazo.")
    parser.add_argument("--crop-min", type=int, default=96,
                        help="Semilado mínimo en píxeles del recorte de muñeca.")
    parser.add_argument("--radio-mano", type=float, default=0.22)
    parser.add_argument("--padding", type=float, default=0.35, help="Padding del ROI de persona.")
    parser.add_argument("--conf-pose", type=float, default=0.25)
    parser.add_argument("--conf-objeto", type=float, default=0.25)
    parser.add_argument("--conf-objeto-crop", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--area-min", type=int, default=80)
    parser.add_argument("--area-max-px", type=int, default=6000)
    parser.add_argument("--area-max-persona", type=float, default=0.05)
    parser.add_argument("--relacion-max", type=float, default=3.0)
    parser.add_argument("--relleno-min", type=float, default=0.3)
    parser.add_argument("--factor-palma", type=float, default=0.85,
                        help="Radio máximo del objeto respecto al radio del brazo. "
                             "El recorte puede ser más grande; solo se acepta lo pegado a la palma.")
    parser.add_argument("--umbral-agarre", type=float, default=0.35)
    parser.add_argument("--contacto-min", type=float, default=0.10)
    parser.add_argument("--min-detecciones", type=int, default=2)
    parser.add_argument("--hueco-max", type=int, default=45)
    parser.add_argument("--distancia-max", type=float, default=180.0)
    parser.add_argument("--frames-traspaso", type=int, default=3)
    parser.add_argument("--hueco-parpadeo", type=int, default=3)
    parser.add_argument("--conf-contenedor", type=float, default=0.2)
    parser.add_argument("--imgsz-contenedor", type=int, default=960)
    parser.add_argument("--cada-contenedor", type=int, default=5)
    parser.add_argument("--dentro-min", type=float, default=0.6)
    parser.add_argument("--factor-contenedor", type=float, default=2.5)
    parser.add_argument("--frames-fin", type=int, default=5)
    parser.add_argument("--alfa", type=float, default=0.55)
    parser.add_argument(
        "--fps", type=parsear_fps, default=None,
        help="Entero ≥ 1, o half/third. Con mp4: frames a procesar por cada segundo "
             "de vídeo (a 12 fps, --fps half ≡ 6 y --fps third ≡ 4; a 15 fps, "
             "--fps 3 toma 1, 6 y 11). Con --frames: fps entero de la línea "
             "temporal (obligatorio). Si se omite en mp4, se procesa frame a frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None)


def _objeto_es_celular(objeto: Objeto, contenedores) -> bool:
    if not contenedores or not objeto.vistas:
        return False
    for pista in contenedores:
        if pista.get("categoria") != "celular":
            continue
        for frame, vista in objeto.vistas.items():
            caja_c = caja_contenedor(pista, int(frame), hold=6)
            if caja_c is None:
                continue
            if iou_cajas(vista.caja, caja_c) >= 0.12 or _fraccion_caja_dentro(vista.caja, caja_c) >= 0.35:
                return True
    return False


def _mayoria_mano_abierta(objeto: Objeto, contextos) -> bool:
    if not objeto.vistas or contextos is None:
        return False
    votos, n = 0, 0
    for frame, vista in objeto.vistas.items():
        ctx = _contexto_en(contextos, int(frame))
        if ctx is None or vista.mano not in ("izquierda", "derecha"):
            continue
        mano = _mano_del_lado(ctx, vista.mano)
        if mano is None:
            continue
        n += 1
        if getattr(mano, "abierta", False):
            votos += 1
    return n >= 2 and votos / n >= 0.45


def quedarse_con_agarrados(objetos: list[Objeto], args, contenedores=None, contextos=None) -> list[Objeto]:
    """Se queda con el objeto principal y, si existe, otro realmente sostenido."""
    def en_mano(objeto: Objeto) -> int:
        return sum(1 for vista in objeto.vistas.values() if vista.mano)

    candidatos = []
    for objeto in objetos:
        n = en_mano(objeto)
        if not objeto.vistas:
            continue
        en_bolsillo = _objeto_en_bolsillo(objeto, contenedores, contextos)
        cerca_bolsillo = _cerca_de_bolsillo_valido(objeto, contenedores, contextos)
        ultima = objeto.vistas[max(objeto.vistas)]
        ctx_u = _contexto_en(contextos, ultima.frame) if contextos is not None else None
        sale_con_objeto = bool(
            ultima.mano
            and ctx_u is not None
            and (_persona_sale_de_escena(ctx_u) or _de_espaldas(ctx_u))
        )
        sale_clip = contextos is not None and any(
            bool(getattr(ctx, "personas", None))
            and (
                _persona_sale_de_escena(ctx)
                or max(ctx.personas, key=lambda p: p.altura).caja[3] >= int(ctx.alto) - 10
            )
            for ctx in contextos
        )
        es_celular = _objeto_es_celular(objeto, contenedores)
        if _parece_mano_vacia(objeto, contextos) and not es_celular:
            continue
        if not es_celular and objeto.vistas:
            areas = [v.area for v in objeto.vistas.values()]
            en_cabeza = 0
            for frame, vista in objeto.vistas.items():
                ctx_v = _contexto_en(contextos, int(frame)) if contextos is not None else None
                if ctx_v is None or not getattr(ctx_v, "personas", None):
                    continue
                persona = max(ctx_v.personas, key=lambda p: p.altura)
                if (vista.centro[1] - float(persona.caja[1])) < 0.16 * max(float(persona.altura), 1.0):
                    en_cabeza += 1
            if areas and float(np.median(areas)) < 220 and en_cabeza / len(objeto.vistas) >= 0.5:
                continue
        if n < args.min_detecciones and not es_celular:
            if not ((sale_con_objeto or sale_clip) and n >= 2 and ultima.mano):
                continue
        if n / len(objeto.vistas) < 0.5 and not sale_con_objeto and not es_celular:
            continue
        if not es_celular:
            solidos = 0
            for frame, vista in objeto.vistas.items():
                ctx_v = _contexto_en(contextos, int(frame)) if contextos is not None else None
                if ctx_v is None or not getattr(ctx_v, "personas", None):
                    if vista.area >= 250:
                        solidos += 1
                    continue
                persona = max(ctx_v.personas, key=lambda p: p.altura)
                rel = (vista.centro[1] - float(persona.caja[1])) / max(float(persona.altura), 1.0)
                if vista.area >= 250 and rel >= 0.16:
                    solidos += 1
            if solidos < 1:
                continue
        candidatos.append(objeto)
    if not candidatos:
        return []
    def mano_dom(objeto: Objeto) -> str | None:
        votos = {"izquierda": 0, "derecha": 0, "ambas": 0}
        for vista in objeto.vistas.values():
            if vista.mano in votos:
                votos[vista.mano] += 1
        return max(votos, key=votos.get) if any(votos.values()) else None

    def _acaba_en_mueble(objeto: Objeto) -> int:
        if not contenedores:
            return 0
        dest = contenedor_final(
            [objeto.vistas[i] for i in sorted(objeto.vistas)],
            contenedores, args, contextos,
        )
        return int(bool(dest and dest.get("category") in ("cesto", "carrito", "bolsa", "bolso", "bolsillo")))

    candidatos.sort(
        key=lambda o: (
            int(_objeto_es_celular(o, contenedores)),
            en_mano(o),
            _acaba_en_mueble(o),
            len(o.vistas),
        ),
        reverse=True,
    )
    celulares = [o for o in candidatos if _objeto_es_celular(o, contenedores)]
    if celulares:
        fin_cel = max(max(o.vistas) for o in celulares)
        candidatos = [
            o for o in candidatos
            if _objeto_es_celular(o, contenedores)
            or min(o.vistas) <= fin_cel + 2
        ]
        if not candidatos:
            candidatos = list(celulares)
    principal = candidatos[0]
    umbral = max(args.min_detecciones, int(0.35 * en_mano(principal)))
    mano_ppal = mano_dom(principal)
    ppal_mueble = _acaba_en_mueble(principal)
    vivos = [principal]
    for objeto in candidatos[1:]:
        es_cel = _objeto_es_celular(objeto, contenedores)
        empieza_antes = min(objeto.vistas) + 6 < min(principal.vistas)
        if ppal_mueble and not _acaba_en_mueble(objeto) and not es_cel and not empieza_antes:
            continue
        otra_mano = mano_dom(objeto) in ("izquierda", "derecha") and mano_dom(objeto) != mano_ppal
        if ppal_mueble and en_mano(objeto) < umbral and not es_cel and not otra_mano:
            continue
        if en_mano(objeto) >= umbral or es_cel or (
            otra_mano and en_mano(objeto) >= args.min_detecciones
        ):
            vivos.append(objeto)
    for numero, objeto in enumerate(vivos, start=1):
        objeto.id = numero
    return vivos


def descartar_falsos_bolsillo(objetos: list[Objeto], contenedores, contextos, args) -> list[Objeto]:
    """Si un objeto acaba en el cesto, el fantasma de la otra mano no va al bolsillo."""
    if len(objetos) < 2:
        return objetos
    destinos = [
        contenedor_final(
            [o.vistas[i] for i in sorted(o.vistas)], contenedores, args, contextos
        )
        for o in objetos
    ]
    hay_mueble = any(
        d and d.get("category") in ("cesto", "carrito", "bolsa") for d in destinos
    )
    if not hay_mueble:
        return objetos
    muebles = [
        o for o, d in zip(objetos, destinos)
        if d and d.get("category") in ("cesto", "carrito", "bolsa") and o.vistas
    ]
    vivos = []
    for objeto, dest in zip(objetos, destinos):
        if dest and dest.get("category") in ("bolsillo", "pocket") and objeto.vistas:
            contemporaneo = False
            fin_p = max(objeto.vistas)
            ini_p = min(objeto.vistas)
            for otro in muebles:
                if fin_p < min(otro.vistas) - 8 or ini_p > max(otro.vistas) + 8:
                    continue
                contemporaneo = True
                break
            if contemporaneo:
                print(f"  Objeto {objeto.id} se descarta: el depósito real es el cesto, no el bolsillo")
                continue
        vivos.append(objeto)
    if not vivos:
        return objetos
    for numero, objeto in enumerate(vivos, start=1):
        objeto.id = numero
    return vivos


def seguir_por_deteccion(frames, pistas, contextos, args) -> list[Objeto]:
    """Sin propagador: la máscara de cada frame es la del detector de muñeca."""
    del frames
    return objetos_desde_pistas(pistas, contextos, args)


def escribir_salidas(
    video: Path,
    args,
    destino: Path,
    objetos: list[Objeto],
    contenedores: list[dict],
    contextos,
    frames,
    indices_origen: list[int],
    metricas: dict,
    fps: float,
    nframes: int,
    nframes_origen: int,
    ancho: int,
    alto: int,
    tecnologia: str,
    backend: str,
    modelo: str,
    extra: dict | None = None,
) -> None:
    extra = extra or {}
    muestreo = extra.get("sampling") or describir_muestreo(args.fps, fps, nframes_origen, nframes)
    informes = [
        informe_objeto(o, fps, nframes, contenedores, args, indices_origen, contextos)
        for o in objetos
    ]
    if extra.get("input") == "frames":
        fps_salida = _fps_salida_frames(fps, indices_origen)
    else:
        offsets, _rate = offsets_muestreo(fps, args.fps)
        n_fps = resolver_fps_proceso(args.fps, fps)
        fps_salida = float(n_fps) if offsets is not None and n_fps is not None else fps
    anotado = destino / "anotado.mp4"
    print(f"  Escribiendo {anotado}")
    escritos = anotar(
        frames, anotado, objetos, contenedores, fps_salida, fps, indices_origen,
        destino / "frames" if not args.sin_frames else None, args.alfa,
        contextos, dibujar_esqueleto=not args.sin_pose,
        prefijo_frame=video.stem,
    )
    if not args.sin_frames:
        print(f"  {escritos} frames en {destino / 'frames'}")

    def orig(idx: int) -> int:
        return frame_origen(idx, indices_origen)

    if extra.get("final_state") is None:
        extra["final_state"] = _final_state_video(informes)
    pose_backend = "mediapipe" if getattr(args, "mediapipe", False) else "yolo-pose"
    traza = {
        "video": str(video),
        "model": modelo,
        "width": int(ancho),
        "height": int(alto),
        "frames": nframes,
        "source_frames": nframes_origen,
        "fps": int(fps) if extra.get("input") == "frames" else fps,
        "output_fps": fps_salida,
        "sampling": muestreo,
        "pose_backend": pose_backend,
        "hands_visible": extra.get("hands_visible", True),
        "final_state": extra.get("final_state"),
        "parameters": {
            "input": extra.get("input", "video"),
            "wrist_padding": args.padding_muneca,
            "crop_min": args.crop_min,
            "palm_factor": args.factor_palma,
            "grip_threshold": args.umbral_agarre,
            "min_contact": args.contacto_min,
            "container_conf": args.conf_contenedor,
            "container_every": args.cada_contenedor,
            "inside_min": args.dentro_min,
            "container_factor": args.factor_contenedor,
            "identifiers_yaml": str(args.identificadores),
            "hand_min_confidence": getattr(getattr(args, "config_manos", None), "min_confidence", None),
            "mediapipe": bool(getattr(args, "mediapipe", False)),
            "fps": int(fps) if extra.get("input") == "frames" else args.fps,
            "width": int(ancho),
            "height": int(alto),
            "frames_origin": list(indices_origen) if extra.get("input") == "frames" else None,
        },
        "identifiers_yaml": [
            {
                "key": item.clave,
                "active": item.activo,
                "ultralytics_id": item.ultralytics_id,
                "name": item.nombre,
                "prompts": item.prompts,
                "destination": item.destino,
                "accessory": item.accesorio,
                "body": item.cuerpo,
            }
            for item in getattr(args, "catalogo", [])
        ],
        "containers": [
            {
                "id": p["id"],
                "category": p["categoria"],
                "prompt": p["texto"],
                "mean_score": p["score_medio"],
                "max_score": p["score_max"],
                "first_frame": orig(p["primer_frame"]),
                "last_frame": orig(p["ultimo_frame"]),
                "frames_seen": p["frames_visto"],
                "destination": bool(p.get("destino", False)),
                "accessory": bool(p.get("accesorio", False)),
                "body": bool(p.get("cuerpo", False)),
                "side": _mano_en(p.get("lado")),
                "last_roi": list(p["caja"]) if p.get("caja") is not None else [],
                "rois": {str(orig(f)): list(c) for f, c in sorted(p["cajas"].items())},
            }
            for p in contenedores
        ],
        "objects": informes,
        "people": extra.get("people") or informes_personas(contextos, informes, objetos),
    }
    if extra:
        for clave, valor in extra.items():
            if clave not in traza or traza[clave] is None:
                traza[clave] = valor
    (destino / "tracking.json").write_text(
        json.dumps(traza, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    texto = resumen(
        video, informes, contenedores, nframes, modelo, args.identificadores, muestreo,
        final_state=traza.get("final_state"), pose_backend=pose_backend,
        personas=traza.get("people"),
    )
    (destino / "summary.txt").write_text(texto, encoding="utf-8")
    (destino / "metrics.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    frames_con_objeto = len({i for o in objetos for i in o.vistas})
    desapariciones = sum(
        1 for info in informes for e in info["events"] if e["type"] == "disappears"
    )
    ruta = publicar_comparativa(
        args.salida, tecnologia, video.stem, backend, destino, metricas,
        len(objetos), desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def procesar_video(
    detectores,
    video: Path | None,
    args,
    tecnologia: str,
    backend: str,
    modelo: str,
    seguir,
    extra: dict | None = None,
    *,
    frames: list[np.ndarray] | None = None,
    frames_origin: list[int] | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    """Procesa un clip (mp4 o arrays en memoria). Devuelve True si las salidas se escribieron."""
    destino = None
    frames_proc: list = []
    medidor = Medidor(tecnologia)
    extra = dict(extra or {})
    if video is None:
        video = Path(getattr(args, "video", None) or "frames")
    else:
        video = Path(video)
    try:
        frames_mem = frames if frames is not None else getattr(args, "frames", None)
        if frames_mem is not None:
            origen_mem = (
                frames_origin if frames_origin is not None
                else getattr(args, "frames_origin", None)
            )
            fps_mem = fps if fps is not None else getattr(args, "fps", None)
            ancho_mem = width if width is not None else getattr(args, "width", None)
            alto_mem = height if height is not None else getattr(args, "height", None)
            frames_proc, indices_origen, fps, ancho, alto = preparar_frames(
                frames_mem, origen_mem, fps_mem, ancho_mem, alto_mem,
            )
            nframes_origen = max(indices_origen) + 1 if indices_origen else len(frames_proc)
            extra["input"] = "frames"
            muestreo = describir_muestreo(
                fps, fps, nframes_origen, len(frames_proc),
                desde_frames=True, indices_origen=indices_origen,
            )
        else:
            if getattr(args, "frames_origin", None) is not None and frames_origin is None:
                raise SystemExit("--frames-origin requires --frames (list[np.ndarray])")
            if not video.is_file():
                raise SystemExit(
                    "Pass an mp4 (--video) or --frames as list[np.ndarray] "
                    "with --fps, --width and --height as integers"
                )
            if width is not None or getattr(args, "width", None) is not None:
                _entero_obligatorio(width if width is not None else args.width, "width")
            if height is not None or getattr(args, "height", None) is not None:
                _entero_obligatorio(height if height is not None else args.height, "height")
            ancho, alto, nframes_archivo, fps_archivo = info_video(video)
            frames_proc, fps, indices_origen = leer_frames(video, args.max_frames, args.fps)
            fps = fps_archivo or fps
            nframes_origen = nframes_archivo
            if args.max_frames is not None:
                nframes_origen = min(nframes_origen, args.max_frames)
            extra["input"] = "video"
            muestreo = describir_muestreo(args.fps, fps, nframes_origen, len(frames_proc))
            if width is not None or getattr(args, "width", None) is not None:
                ancho_pedido = int(width if width is not None else args.width)
                if ancho_pedido != int(ancho):
                    raise SystemExit(f"--width ({ancho_pedido}) does not match the mp4 ({ancho})")
            if height is not None or getattr(args, "height", None) is not None:
                alto_pedido = int(height if height is not None else args.height)
                if alto_pedido != int(alto):
                    raise SystemExit(f"--height ({alto_pedido}) does not match the mp4 ({alto})")
        extra["sampling"] = muestreo
        print(f"\n=== {video.name}: {ancho}x{alto}, {len(frames_proc)} frames procesados, {fps:.3f} fps ===")
        print(f"  {muestreo['description']}")
        destino = carpeta_de(args.salida, tecnologia, video)
        escena, detector, sam, det_contenedores = detectores
        with medidor.etapa("segmentacion"):
            contextos, pistas = detectar_objetos(frames_proc, escena, detector, sam, args)
            manos_visibles = any(bool(ctx.manos) for ctx in contextos)
            extra["hands_visible"] = manos_visibles
            if not manos_visibles:
                extra["final_state"] = "no_hands"
                objetos = []
                contenedores = seguir_contenedores(
                    frames_proc, det_contenedores, args.cada_contenedor, contextos=contextos
                )
                print("  No wrists visible above min_confidence. Object model was not applied.")
            else:
                try:
                    objetos = seguir(frames_proc, pistas, contextos, args)
                except Exception as exc:
                    _aviso(f"{video.name}: el seguidor de máscara falló ({exc})")
                    traceback.print_exc()
                    objetos = objetos_desde_pistas(pistas, contextos, args) if pistas else []
                contenedores = seguir_contenedores(
                    frames_proc, det_contenedores, args.cada_contenedor, contextos=contextos
                )
                objetos = recortar_objeto_si_salta_a_bolsa(objetos, contenedores)
                objetos = descartar_objetos_que_son_mueble(objetos, contenedores)
                objetos = _quitar_vistas_desviadas(objetos, contenedores, contextos)
                objetos = recortar_cola_si_salta(objetos, contenedores, contextos)
                objetos = recortar_si_vuelve_al_cuerpo(objetos, contenedores, contextos)
                objetos = descartar_objetos_bolsa(objetos, contenedores)
                sembrados = sembrar_objetos_desde_identificadores(
                    contenedores, contextos, args
                )
                if sembrados:
                    objetos = fusionar_objetos(list(objetos) + list(sembrados))
                objetos = completar_deposito(
                    frames_proc, objetos, contenedores, contextos, detector, sam, args
                )
                objetos = _cerrar_si_entra_bolsillo(objetos, contenedores, contextos)
                objetos = extender_si_sigue_en_mano(
                    frames_proc, objetos, contextos, detector, sam, args, contenedores
                )
                objetos = _cerrar_si_entra_bolsillo(objetos, contenedores, contextos)
                objetos = _alinear_aproximacion_a_mueble(objetos, contenedores, contextos)
                objetos = _quitar_vistas_desviadas(objetos, contenedores, contextos)
                objetos = rellenar_huecos_en_mano(
                    frames_proc, objetos, contextos, detector, sam, args, contenedores
                )
                objetos = recortar_cola_si_salta(objetos, contenedores, contextos)
                objetos = recortar_si_vuelve_al_cuerpo(objetos, contenedores, contextos)
                objetos = descartar_accesorios(objetos, contenedores)
                objetos = quedarse_con_agarrados(objetos, args, contenedores, contextos)
                objetos = fusionar_objetos(objetos)
                objetos = _alinear_aproximacion_a_mueble(objetos, contenedores, contextos)
                objetos = _quitar_vistas_desviadas(objetos, contenedores, contextos)
                objetos = rellenar_huecos_en_mano(
                    frames_proc, objetos, contextos, detector, sam, args, contenedores
                )
                objetos = descartar_falsos_bolsillo(objetos, contenedores, contextos, args)
                contenedores = descartar_contenedores_en_mano(contenedores, objetos, args)
                contenedores = _filtrar_celular_fantasma(contenedores, objetos)
        escribir_salidas(
            video, args, destino, objetos, contenedores, contextos, frames_proc, indices_origen,
            medidor.cerrar(len(frames_proc)),
            fps, len(frames_proc), nframes_origen, ancho, alto, tecnologia, backend, modelo, extra,
        )
        return True
    except KeyboardInterrupt:
        raise
    except (Exception, SystemExit) as exc:
        mensaje = str(exc) or exc.__class__.__name__
        _aviso(f"{video.name} falló: {mensaje}")
        traceback.print_exc()
        if destino is None:
            try:
                destino = carpeta_de(args.salida, tecnologia, video)
            except Exception:
                destino = args.salida / tecnologia / Path(video).stem
        try:
            publicar_comparativa(
                args.salida, tecnologia, video.stem, backend, destino,
                medidor.cerrar(len(frames_proc)), 0, 0, 0, ok=False, codigo=1, error=mensaje,
            )
        except Exception as extra_exc:
            _aviso(f"no se pudo actualizar la comparativa ({extra_exc})")
        return False
    finally:
        _vaciar_cuda()


def procesar_lote(
    detectores,
    videos: list[Path] | None,
    args,
    tecnologia: str,
    backend: str,
    modelo: str,
    seguir,
    extra: dict | None = None,
    *,
    frames: list[np.ndarray] | None = None,
    frames_origin: list[int] | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> int:
    """Procesa varios clips o una lista de arrays. Un fallo no aborta el resto."""
    if frames is not None:
        args.frames = frames
        if frames_origin is not None:
            args.frames_origin = frames_origin
        if fps is not None:
            args.fps = fps
        if width is not None:
            args.width = width
        if height is not None:
            args.height = height
        videos = resolver_videos(args)
    elif not videos:
        videos = resolver_videos(args)
    fallos = 0
    for video in videos:
        if not procesar_video(
            detectores, video, args, tecnologia, backend, modelo, seguir, extra,
            frames=frames, frames_origin=frames_origin, fps=fps, width=width, height=height,
        ):
            fallos += 1
    return fallos

