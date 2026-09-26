#!/usr/bin/env python3
"""Segmenta a la persona, su ROI con padding y el objeto que lleva en la mano.

Usa SAM 3 (checkpoint SAM 3.1, Object Multiplex) para:
  1. Segmentar y seguir a la persona.
  2. Construir un ROI con padding alrededor de su caja, para no recortar
     objetos grandes que sobresalen del cuerpo.
  3. Segmentar las manos dentro de ese ROI y el objeto descrito por texto.
  4. Seguir el objeto, marcar cuándo está visible y guardar la última
     posición antes de que desaparezca. Si cambia de mano, lo registra.

Los prompts de texto van en inglés: es el idioma con el que se entrenó SAM 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Hay que fijarlo antes de que PyTorch cree el asignador de CUDA.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np

from salida_comun import Medidor, carpeta_de, publicar_comparativa

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
PESOS = Path(__file__).resolve().parent / "pesos"


def comprobar_entorno() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit(
            f"Se necesita Python 3.11 o superior (hay {sys.version.split()[0]})."
        )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "No está instalado torch. Activa .venv y sigue el README."
        ) from exc

    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 0):
        raise SystemExit(f"Se necesita PyTorch 2 o superior (hay {torch.__version__}).")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible para PyTorch.")
    cuda = torch.version.cuda or ""
    major = int(cuda.split(".")[0]) if cuda else 0
    if major < 12:
        raise SystemExit(f"Se necesita CUDA 12 o superior (PyTorch informa {cuda}).")
    nombre = torch.cuda.get_device_name(0)
    print(
        f"Entorno: Python {sys.version.split()[0]}, "
        f"torch {torch.__version__}, CUDA {cuda}, GPU {nombre}"
    )


def listar_videos(ruta: Path) -> list[Path]:
    if ruta.is_file():
        return [ruta]
    videos = [
        p
        for p in sorted(ruta.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        raise SystemExit(f"No hay vídeos en {ruta}")
    return videos


def a_numpy(valor):
    import torch

    if isinstance(valor, torch.Tensor):
        return valor.detach().cpu().numpy()
    return np.asarray(valor)


def parsear_salidas(outputs: dict) -> list[dict]:
    """Normaliza el diccionario de SAM 3 a máscaras en resolución original."""
    if not outputs or "out_obj_ids" not in outputs:
        return []
    ids = a_numpy(outputs["out_obj_ids"]).astype(np.int64).reshape(-1)
    if ids.size == 0:
        return []
    mascaras = a_numpy(outputs["out_binary_masks"])
    if mascaras.dtype != np.bool_:
        mascaras = mascaras > 0
    cajas = a_numpy(outputs["out_boxes_xywh"]).astype(np.float32)
    if "out_probs" in outputs:
        probs = a_numpy(outputs["out_probs"]).astype(np.float32).reshape(-1)
    else:
        probs = np.ones(len(ids), dtype=np.float32)

    instancias = []
    for i, obj_id in enumerate(ids):
        mascara = np.asarray(mascaras[i])
        if mascara.ndim == 3:
            mascara = mascara[0]
        mascara = mascara.astype(bool)
        if int(mascara.sum()) == 0:
            continue
        x, y, w, h = (float(v) for v in cajas[i][:4])
        instancias.append(
            {
                "id": int(obj_id),
                "mascara": mascara,
                "prob": float(probs[i]) if i < len(probs) else 1.0,
                "caja_xywh_norm": (x, y, w, h),
            }
        )
    return instancias


def caja_de_mascara(mascara: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mascara)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def centro_de_mascara(mascara: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mascara)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def expandir_caja(
    caja: tuple[int, int, int, int],
    ratio: float,
    ancho: int,
    alto: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = caja
    pad_x = ratio * max(1, x1 - x0 + 1)
    pad_y = ratio * max(1, y1 - y0 + 1)
    return (
        int(max(0, np.floor(x0 - pad_x))),
        int(max(0, np.floor(y0 - pad_y))),
        int(min(ancho - 1, np.ceil(x1 + pad_x))),
        int(min(alto - 1, np.ceil(y1 + pad_y))),
    )


def fraccion_dentro_de_caja(
    mascara: np.ndarray, caja: tuple[int, int, int, int]
) -> float:
    area = int(mascara.sum())
    if area == 0:
        return 0.0
    x0, y0, x1, y1 = caja
    return float(mascara[y0 : y1 + 1, x0 : x1 + 1].sum()) / area


def dilatar(mascara: np.ndarray, pixeles: int) -> np.ndarray:
    if pixeles <= 0:
        return mascara
    k = pixeles * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mascara.astype(np.uint8), kernel).astype(bool)


def solape(a: np.ndarray, b: np.ndarray) -> float:
    area = int(a.sum())
    if area == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / area


def info_video(ruta: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise SystemExit(f"No se puede abrir el vídeo {ruta}")
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 1e-3:
        fps = 25.0
    return ancho, alto, nframes, fps


def ajustar_lote(predictor, lote: int) -> None:
    """SAM 3.1 procesa el vídeo en tandas. Con 16 frames la RTX 5090 se queda corta."""
    modelo = getattr(predictor, "model", None)
    if modelo is None:
        return
    for nombre in ("batched_grounding_batch_size", "postprocess_batch_size"):
        if hasattr(modelo, nombre):
            setattr(modelo, nombre, lote)
    print(f"Tanda de inferencia: {lote} frames.")


def construir_predictor(version: str, umbral: float, lote: int):
    try:
        from sam3.model_builder import build_sam3_predictor
    except ImportError as exc:
        raise SystemExit(
            "No está instalado el paquete sam3. Activa .venv y sigue el README."
        ) from exc

    print(f"Cargando SAM {version} (la primera vez descarga el checkpoint)...")
    try:
        kwargs = dict(
            version=version,
            compile=False,
            async_loading_frames=False,
        )
        if version == "sam3.1":
            kwargs.update(use_fa3=False, use_rope_real=True, warm_up=False)
        predictor = build_sam3_predictor(**kwargs)
    except Exception as exc:
        texto = f"{exc}\n{traceback.format_exc()}"
        pista = ""
        bajo = texto.lower()
        if any(k in bajo for k in ("401", "403", "gated", "authorized", "access", "token")):
            repo = "facebook/sam3.1" if version == "sam3.1" else "facebook/sam3"
            pista = (
                "\nEl checkpoint de SAM 3 está restringido en Hugging Face.\n"
                f"1. Entra en https://huggingface.co/{repo} y acepta la licencia.\n"
                "2. Crea un token de lectura en https://huggingface.co/settings/tokens\n"
                "3. En este entorno ejecuta: hf auth login\n"
            )
        raise SystemExit(f"No se pudo cargar SAM 3.\n{exc}{pista}") from exc

    if hasattr(predictor, "default_output_prob_thresh"):
        predictor.default_output_prob_thresh = umbral
    _adaptar_init_state(predictor)
    ajustar_lote(predictor, lote)
    return predictor


def _adaptar_init_state(predictor) -> None:
    """El predictor envía offload_state_to_cpu y SAM 3.1 no lo declara."""
    import inspect

    original = predictor.model.init_state
    parametros = inspect.signature(original).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parametros.values()):
        return
    validos = set(parametros)

    def init_state(*args, **kwargs):
        return original(*args, **{clave: valor for clave, valor in kwargs.items() if clave in validos})

    predictor.model.init_state = init_state


def detecciones_palma(
    fuente, nframes: int
) -> tuple[dict[int, list[Deteccion]], dict[int, list[dict]]] | None:
    """El objeto en la mano sale de YOLO y MobileSAM sobre la palma.

    SAM 3 con un texto genérico y HOIST a esta distancia no devuelven esa
    máscara. Los pesos están en ``pesos/``.
    """
    try:
        from detector_palma.agarre import anotar_agarre
        from detector_palma.deteccion import DetectorUltralytics
        from detector_palma.personas import AnalizadorEscena
    except ImportError as exc:
        print(f"  No se pudo usar el detector de palma ({exc}).")
        return None
    pesos_yolo = PESOS / "yolo26n-seg.pt"
    pesos_pose = PESOS / "yolo26n-pose.pt"
    if not pesos_yolo.is_file() or not pesos_pose.is_file():
        print(f"  Faltan yolo26n-seg.pt o yolo26n-pose.pt en {PESOS}.")
        return None

    import torch

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Objeto en la palma: YOLO-seg + MobileSAM ({dispositivo})")
    detector = DetectorUltralytics(
        modelo=str(pesos_yolo), dispositivo=dispositivo, conf=0.25, imgsz=640
    )
    analizador = AnalizadorEscena(
        estrategia="pose",
        modelo_pose=str(pesos_pose),
        dispositivo=dispositivo,
        padding_persona=0.35,
        radio_mano=0.22,
        conf=0.25,
        imgsz=640,
    )
    if isinstance(fuente, (str, Path)):
        cap = cv2.VideoCapture(str(fuente))
        if not cap.isOpened():
            return None
        frames = []
        while len(frames) < nframes:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    else:
        frames = list(fuente)[:nframes]

    por_id: dict[int, list[Deteccion]] = {}
    por_frame: dict[int, list[dict]] = {}
    activos: list[tuple[int, tuple[float, float], int]] = []
    siguiente = 1
    en_mano = 0
    for indice, frame in enumerate(frames):
        contexto = analizador.analizar(frame)
        candidatos = anotar_agarre(
            detector.detectar(frame, contexto=contexto),
            contexto,
            umbral=0.35,
            area_minima=0.0002,
            area_maxima=0.02,
        )
        for inst in candidatos:
            if not inst.en_mano or inst.mano_asociada not in ("izquierda", "derecha"):
                continue
            mejor_i = None
            mejor_d = 1e9
            for i, (oid, centro, frame_prev) in enumerate(activos):
                if indice - frame_prev > 45:
                    continue
                dist = float(np.hypot(inst.centro[0] - centro[0], inst.centro[1] - centro[1]))
                if dist < mejor_d:
                    mejor_d, mejor_i = dist, i
            if mejor_i is not None and mejor_d < 180:
                oid = activos[mejor_i][0]
                activos[mejor_i] = (oid, inst.centro, indice)
            else:
                oid = siguiente
                siguiente += 1
                activos.append((oid, inst.centro, indice))
            por_id.setdefault(oid, []).append(
                Deteccion(
                    frame=indice,
                    id_sam3=oid,
                    centro=inst.centro,
                    bbox=inst.caja,
                    prob=float(inst.score_agarre),
                    area=inst.area,
                    mano=inst.mano_asociada,
                    en_mano=True,
                    score_agarre=float(inst.score_agarre),
                )
            )
            por_frame.setdefault(indice, []).append(
                {
                    "id": oid,
                    "mascara": inst.mascara,
                    "prob": float(inst.score_agarre),
                    "caja_xywh_norm": (0.0, 0.0, 1.0, 1.0),
                }
            )
            en_mano += 1
    for dets in por_id.values():
        dets.sort(key=lambda d: d.frame)
    print(f"  Palma: {en_mano} detecciones en mano, {len(por_id)} pistas")
    if not por_id:
        return None
    return por_id, por_frame


def seguir_concepto(predictor, video: Path, texto: str, umbral: float) -> dict[int, list[dict]]:
    """Una pasada de SAM 3: el prompt de texto se propaga a todo el vídeo."""
    print(f"  Prompt «{texto}»")
    alta = predictor.handle_request(
        dict(
            type="start_session",
            resource_path=str(video),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
    )
    session_id = alta["session_id"]
    por_frame: dict[int, list[dict]] = {}
    try:
        predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=texto,
                output_prob_thresh=umbral,
            )
        )
        flujo = predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=0,
                output_prob_thresh=umbral,
            )
        )
        for respuesta in flujo:
            por_frame[int(respuesta["frame_index"])] = parsear_salidas(
                respuesta["outputs"]
            )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        raise SystemExit(
            "CUDA se quedó sin memoria en SAM 3. "
            "Vuelve a lanzarlo con --lote 4 o --lote 2, "
            "o cierra otros programas que estén usando la GPU.\n"
            f"{exc}"
        ) from exc
    finally:
        predictor.handle_request(
            dict(type="close_session", session_id=session_id, run_gc_collect=True)
        )
    n_ids = {inst["id"] for insts in por_frame.values() for inst in insts}
    print(f"    {len(por_frame)} frames con salida, {len(n_ids)} identidades")
    return por_frame


def elegir_persona(por_frame: dict[int, list[dict]]) -> int | None:
    areas: dict[int, list[int]] = {}
    for insts in por_frame.values():
        for inst in insts:
            areas.setdefault(inst["id"], []).append(int(inst["mascara"].sum()))
    if not areas:
        return None
    return max(areas, key=lambda i: float(np.median(areas[i])))


def instancia(por_frame: dict[int, list[dict]], frame: int, obj_id: int) -> dict | None:
    for inst in por_frame.get(frame, []):
        if inst["id"] == obj_id:
            return inst
    return None


def lado_anatomico(
    orden: int,
    total: int,
    cx_mano: float,
    cx_persona: float | None,
    de_frente: bool,
) -> str:
    """Mano izquierda/derecha de la persona.

    Con una sola mano se compara contra el centro del cuerpo. Con varias,
    la mitad izquierda de la imagen y la derecha se reparten por su orden.
    Si la persona mira a la cámara, su izquierda queda a la derecha de la imagen.
    """
    if total <= 1 and cx_persona is not None:
        en_lado_izquierdo_imagen = cx_mano < cx_persona
    else:
        en_lado_izquierdo_imagen = orden < total / 2
    if de_frente:
        return "derecha" if en_lado_izquierdo_imagen else "izquierda"
    return "izquierda" if en_lado_izquierdo_imagen else "derecha"


_MUNECA_IZQ, _MUNECA_DER = 9, 10
_CODO_IZQ, _CODO_DER = 7, 8
_PESO_PROXIMIDAD, _PESO_PALMA, _PESO_DEDOS, _PESO_ENCUADRE = 0.30, 0.35, 0.25, 0.10
_MIN_CONTACTO_AGARRE = 0.10
_MARGEN_DEDO_PX = 10


def ruta_pose_por_defecto() -> Path:
    return PESOS / "yolo26n-pose.pt"


def analizar_pose(
    fuente,
    nframes: int,
    modelo_pose: Path,
    conf: float = 0.25,
    imgsz: int = 640,
) -> list[list[dict]] | None:
    """Muñecas, palma y abanico de dedos. El lado es el de la persona, no el de la imagen."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  Sin ultralytics: la mano se estima por la posición en la imagen.")
        return None
    if not Path(modelo_pose).is_file():
        print(f"  No está el modelo de pose en {modelo_pose}.")
        return None
    import torch

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = YOLO(str(modelo_pose))
    print(f"  Pose: {Path(modelo_pose).name} en {dispositivo}")
    if isinstance(fuente, (str, Path)):
        cap = cv2.VideoCapture(str(fuente))
        if not cap.isOpened():
            print(f"  No se puede abrir {fuente} para la pose.")
            return None
        frames = []
        while len(frames) < nframes:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    else:
        frames = list(fuente)[:nframes]

    por_frame: list[list[dict]] = []
    con_mano = 0
    for frame in frames:
        por_frame.append(_pose_de_frame(modelo, frame, conf, imgsz, dispositivo))
        if por_frame[-1]:
            con_mano += 1
    while len(por_frame) < nframes:
        por_frame.append([])
    print(f"  Pose: manos en {con_mano} de {nframes} frames")
    return por_frame


def _pose_de_frame(modelo, frame, conf: float, imgsz: int, dispositivo: str) -> list[dict]:
    resultados = modelo.predict(
        frame, conf=conf, imgsz=imgsz, device=dispositivo, verbose=False
    )
    if not resultados or resultados[0].boxes is None or len(resultados[0].boxes) == 0:
        return []
    resultado = resultados[0]
    kps = None
    if resultado.keypoints is not None and resultado.keypoints.data is not None:
        kps = resultado.keypoints.data.cpu().numpy()
    mejor = None
    mejor_area = -1.0
    for i in range(len(resultado.boxes)):
        x1, y1, x2, y2 = resultado.boxes.xyxy[i].cpu().numpy().tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= mejor_area or kps is None or i >= len(kps):
            continue
        manos = _manos_de_keypoints(kps[i], max(1.0, y2 - y1))
        if not manos:
            continue
        mejor_area = area
        mejor = manos
    return mejor or []


def _manos_de_keypoints(kp: np.ndarray, altura: float, radio_rel: float = 0.22) -> list[dict]:
    radio = max(20.0, radio_rel * altura)
    manos = []
    for indice_muneca, indice_codo, lado in (
        (_MUNECA_IZQ, _CODO_IZQ, "izquierda"),
        (_MUNECA_DER, _CODO_DER, "derecha"),
    ):
        if indice_muneca >= len(kp) or float(kp[indice_muneca][2]) < 0.3:
            continue
        muneca = (float(kp[indice_muneca][0]), float(kp[indice_muneca][1]))
        dx, dy = 0.0, 1.0
        if indice_codo < len(kp) and float(kp[indice_codo][2]) >= 0.3:
            vx = muneca[0] - float(kp[indice_codo][0])
            vy = muneca[1] - float(kp[indice_codo][1])
            norma = float(np.hypot(vx, vy))
            if norma > 1.0:
                dx, dy = vx / norma, vy / norma
        palma = (muneca[0] + 0.28 * radio * dx, muneca[1] + 0.28 * radio * dy)
        radio_palma = max(10.0, radio * 0.38)
        perp_x, perp_y = -dy, dx
        puntos = [palma]
        for paso in (-0.55, -0.28, 0.0, 0.28, 0.55):
            puntos.append(
                (
                    palma[0] + 0.55 * radio * dx + paso * radio_palma * perp_x,
                    palma[1] + 0.55 * radio * dy + paso * radio_palma * perp_y,
                )
            )
        manos.append(
            {
                "lado": lado,
                "centro": muneca,
                "palma": palma,
                "radio": radio,
                "radio_palma": radio_palma,
                "puntos_agarre": puntos,
                "confianza": float(kp[indice_muneca][2]),
            }
        )
    return manos


def alinear_lados(manos_mascara: list[dict], manos_pose: list[dict]) -> None:
    """La máscara hereda el lado anatómico de la muñeca más cercana."""
    if not manos_pose:
        return
    libres = list(manos_pose)
    for mano in sorted(manos_mascara, key=lambda m: m["centro"][0]):
        if not libres:
            break
        mejor = min(
            libres,
            key=lambda p: float(
                np.hypot(p["centro"][0] - mano["centro"][0], p["centro"][1] - mano["centro"][1])
            ),
        )
        dist = float(
            np.hypot(mejor["centro"][0] - mano["centro"][0], mejor["centro"][1] - mano["centro"][1])
        )
        if dist <= max(120.0, mejor["radio"] * 1.8):
            mano["lado"] = mejor["lado"]
            libres.remove(mejor)


def _en_roi(punto: tuple[float, float], roi, margen: float = 80.0) -> bool:
    if roi is None:
        return True
    x, y = punto
    return roi[0] - margen <= x <= roi[2] + margen and roi[1] - margen <= y <= roi[3] + margen


def pose_en_roi(manos_pose: list[dict] | None, roi) -> list[dict]:
    if not manos_pose:
        return []
    return [m for m in manos_pose if _en_roi(m["palma"], roi)]


def _distancia_a_caja(punto: tuple[float, float], mascara: np.ndarray, caja) -> float:
    x, y = punto
    alto, ancho = mascara.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < alto and 0 <= xi < ancho and mascara[yi, xi]:
        return 0.0
    x1, y1, x2, y2 = caja
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def _solape_disco(mascara: np.ndarray, cx: float, cy: float, radio: float) -> float:
    alto, ancho = mascara.shape[:2]
    radio_i = max(4, int(round(radio)))
    cxi, cyi = int(round(cx)), int(round(cy))
    x1, y1 = max(0, cxi - radio_i), max(0, cyi - radio_i)
    x2, y2 = min(ancho, cxi + radio_i + 1), min(alto, cyi + radio_i + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ys, xs = np.ogrid[y1:y2, x1:x2]
    disco = (xs - cxi) ** 2 + (ys - cyi) ** 2 <= radio_i * radio_i
    total = int(np.count_nonzero(disco))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(mascara[y1:y2, x1:x2] & disco)) / total


def _contacto_puntos(mascara: np.ndarray, puntos: list[tuple[float, float]]) -> float:
    if not puntos:
        return 0.0
    alto, ancho = mascara.shape[:2]
    toques = 0
    for x, y in puntos:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < ancho and 0 <= yi < alto):
            continue
        if mascara[yi, xi]:
            toques += 1
            continue
        x1, y1 = max(0, xi - _MARGEN_DEDO_PX), max(0, yi - _MARGEN_DEDO_PX)
        x2, y2 = min(ancho, xi + _MARGEN_DEDO_PX + 1), min(alto, yi + _MARGEN_DEDO_PX + 1)
        if np.any(mascara[y1:y2, x1:x2]):
            toques += 1
    return toques / len(puntos)


def _como_pose(mano: dict) -> dict:
    x1, y1, x2, y2 = mano["bbox"]
    radio = max(20.0, 0.5 * max(x2 - x1, y2 - y1))
    return {
        "lado": mano["lado"],
        "centro": mano["centro"],
        "palma": mano["centro"],
        "radio": radio,
        "radio_palma": max(10.0, radio * 0.45),
        "puntos_agarre": [mano["centro"]],
        "confianza": mano.get("prob", 1.0),
        "mascara": mano.get("mascara"),
        "dilatada": mano.get("dilatada"),
    }


def evaluar_agarre(
    mascara: np.ndarray,
    caja: tuple[int, int, int, int],
    manos_pose: list[dict],
    manos_mascara: list[dict],
    umbral: float,
) -> dict:
    """Puntuación de palma, dedos y cercanía. No descarta objetos grandes."""
    fuentes = manos_pose or [_como_pose(m) for m in manos_mascara]
    if not fuentes:
        return {"en_mano": False, "lado": None, "score": 0.0, "scores": {}}
    por_lado_mascara = {m["lado"]: m for m in manos_mascara}
    scores: dict[str, float] = {}
    contactos: dict[str, float] = {}
    for mano in fuentes:
        palma = mano["palma"]
        radio_palma = max(8.0, float(mano["radio_palma"]))
        radio = max(radio_palma, float(mano["radio"]))
        distancia = _distancia_a_caja(palma, mascara, caja)
        proximidad = float(np.clip(1.0 - distancia / max(1.0, radio), 0.0, 1.0))
        contacto_palma = _solape_disco(mascara, palma[0], palma[1], radio_palma)
        gemela = por_lado_mascara.get(mano["lado"])
        if gemela is not None and gemela.get("dilatada") is not None:
            contacto_palma = max(contacto_palma, solape(mascara, gemela["dilatada"]))
        palma_boost = min(1.0, contacto_palma * 3.0)
        dedos = _contacto_puntos(mascara, mano.get("puntos_agarre") or [])
        score = (
            _PESO_PROXIMIDAD * proximidad
            + _PESO_PALMA * palma_boost
            + _PESO_DEDOS * dedos
            + _PESO_ENCUADRE
        )
        lado = mano["lado"]
        if lado not in scores or score > scores[lado]:
            scores[lado] = score
            contactos[lado] = max(palma_boost, dedos)
    lado = max(scores, key=scores.get)
    score = scores[lado]
    en_mano = score >= umbral and contactos[lado] >= _MIN_CONTACTO_AGARRE
    return {
        "en_mano": en_mano,
        "lado": lado if en_mano else lado,
        "score": score,
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


@dataclass
class Deteccion:
    frame: int
    id_sam3: int
    centro: tuple[float, float]
    bbox: tuple[int, int, int, int]
    prob: float
    area: int
    mano: str | None = None
    en_mano: bool = False
    score_agarre: float = 0.0
    scores_mano: dict = field(default_factory=dict)


@dataclass
class Cadena:
    detecciones: list[Deteccion] = field(default_factory=list)

    @property
    def ultimo(self) -> Deteccion:
        return self.detecciones[-1]

    @property
    def primero(self) -> Deteccion:
        return self.detecciones[0]


def asociar_objetos(
    nframes: int,
    personas: dict[int, list[dict]],
    persona_id: int,
    manos: dict[int, list[dict]],
    objetos: dict[int, list[dict]],
    ancho: int,
    alto: int,
    padding: float,
    dilatacion: int,
    contacto_min: float,
    roi_min: float,
    de_frente: bool,
    manos_pose: list[list[dict]] | None = None,
    umbral_agarre: float = 0.35,
) -> tuple[dict[int, dict], dict[int, list[Deteccion]]]:
    """Devuelve el estado por frame (ROI y manos) y las detecciones de objeto en mano."""
    estado: dict[int, dict] = {}
    por_id: dict[int, list[Deteccion]] = {}

    for frame in range(nframes):
        persona = instancia(personas, frame, persona_id)
        roi = None
        centro_persona = None
        if persona is not None:
            caja = caja_de_mascara(persona["mascara"])
            if caja is not None:
                roi = expandir_caja(caja, padding, ancho, alto)
                centro_persona = centro_de_mascara(persona["mascara"])

        manos_frame = []
        if roi is not None:
            candidatas = []
            for inst in manos.get(frame, []):
                centro = centro_de_mascara(inst["mascara"])
                if centro is None:
                    continue
                dentro = (
                    roi[0] <= centro[0] <= roi[2] and roi[1] <= centro[1] <= roi[3]
                )
                if not dentro and fraccion_dentro_de_caja(inst["mascara"], roi) < 0.3:
                    continue
                candidatas.append((centro[0], inst, centro))
            candidatas.sort(key=lambda t: t[0])
            for orden, (_, inst, centro) in enumerate(candidatas):
                caja = caja_de_mascara(inst["mascara"])
                if caja is None:
                    continue
                cx_persona = centro_persona[0] if centro_persona is not None else None
                manos_frame.append(
                    {
                        "id": inst["id"],
                        "lado": lado_anatomico(
                            orden, len(candidatas), centro[0], cx_persona, de_frente
                        ),
                        "centro": centro,
                        "bbox": caja,
                        "mascara": inst["mascara"],
                        "dilatada": dilatar(inst["mascara"], dilatacion),
                        "prob": inst["prob"],
                    }
                )

        pose_frame = pose_en_roi(
            manos_pose[frame] if manos_pose is not None and frame < len(manos_pose) else None,
            roi,
        )
        alinear_lados(manos_frame, pose_frame)
        estado[frame] = {
            "roi": roi,
            "centro_persona": centro_persona,
            "persona_visible": persona is not None,
            "manos": manos_frame,
            "pose": pose_frame,
        }

        if roi is None:
            continue
        for inst in objetos.get(frame, []):
            if fraccion_dentro_de_caja(inst["mascara"], roi) < roi_min:
                continue
            caja = caja_de_mascara(inst["mascara"])
            centro = centro_de_mascara(inst["mascara"])
            if caja is None or centro is None:
                continue
            agarre = evaluar_agarre(
                inst["mascara"], caja, pose_frame, manos_frame, umbral_agarre
            )
            # El prompt del objeto a veces devuelve la propia mano o el cuerpo.
            solape_piel = max(
                (solape(inst["mascara"], mano["mascara"]) for mano in manos_frame),
                default=0.0,
            )
            if solape_piel > 0.65:
                continue
            if persona is not None:
                area_persona = int(persona["mascara"].sum())
                if area_persona > 0 and int(inst["mascara"].sum()) > 0.45 * area_persona:
                    continue
            det = Deteccion(
                frame=frame,
                id_sam3=inst["id"],
                centro=centro,
                bbox=caja,
                prob=inst["prob"],
                area=int(inst["mascara"].sum()),
                mano=agarre["lado"],
                en_mano=agarre["en_mano"],
                score_agarre=agarre["score"],
                scores_mano=agarre["scores"],
            )
            por_id.setdefault(inst["id"], []).append(det)

    for dets in por_id.values():
        dets.sort(key=lambda d: d.frame)
    return estado, por_id


def encadenar(
    por_id: dict[int, list[Deteccion]],
    hueco_max: int,
    distancia_max: float,
) -> Cadena | None:
    """Une identidades de SAM 3 que son el mismo objeto al cambiar de mano."""
    if not por_id:
        return None
    tramos = [dets for dets in por_id.values() if dets]
    tramos.sort(key=lambda dets: (dets[0].frame, -len(dets)))

    usados: set[int] = set()
    semilla = max(tramos, key=lambda dets: (sum(1 for d in dets if d.en_mano), len(dets)))
    cadena = Cadena(detecciones=list(semilla))
    usados.add(id(semilla))

    def encaja(anterior: Deteccion, siguiente: Deteccion) -> tuple[int, float] | None:
        hueco = siguiente.frame - anterior.frame
        if hueco < 1 or hueco > hueco_max:
            return None
        dist = float(
            np.hypot(
                siguiente.centro[0] - anterior.centro[0],
                siguiente.centro[1] - anterior.centro[1],
            )
        )
        cambia_mano = (
            siguiente.mano is not None
            and anterior.mano is not None
            and siguiente.mano != anterior.mano
        )
        if dist > distancia_max and not cambia_mano:
            return None
        return hueco, dist

    cambio = True
    while cambio:
        cambio = False
        mejor = None
        mejor_clave = None
        prepender = False
        for dets in tramos:
            if id(dets) in usados:
                continue
            adelante = encaja(cadena.ultimo, dets[0])
            atras = encaja(dets[-1], cadena.primero)
            for clave, al_principio in ((adelante, False), (atras, True)):
                if clave is None:
                    continue
                if mejor_clave is None or clave < mejor_clave:
                    mejor_clave = clave
                    mejor = dets
                    prepender = al_principio
        if mejor is not None:
            if prepender:
                cadena.detecciones = list(mejor) + cadena.detecciones
            else:
                cadena.detecciones.extend(mejor)
            usados.add(id(mejor))
            cambio = True
    return cadena


def _vista(det: Deteccion, fps: float, mano: str | None) -> dict:
    return {
        "frame": det.frame,
        "tiempo_s": round(det.frame / fps, 4),
        "centro_xy": [round(det.centro[0], 2), round(det.centro[1], 2)],
        "bbox_xyxy": list(det.bbox),
        "mano": mano,
        "id_sam3": det.id_sam3,
        "probabilidad": round(det.prob, 4),
        "area_px": det.area,
        "score_agarre": round(det.score_agarre, 4),
    }


def linea_temporal(
    nframes: int,
    fps: float,
    estado: dict[int, dict],
    cadena: Cadena | None,
    frames_traspaso: int = 3,
    frames_suelte: int = 2,
) -> tuple[list[dict], list[dict], dict | None]:
    """Visible en la mano, suelto (sigue en escena) u oculto. El traspaso espera varios frames."""
    visibles = {}
    if cadena is not None:
        for det in cadena.detecciones:
            visibles[det.frame] = det

    frames = []
    eventos = []
    ultima = None
    ultima_en_mano = None
    empezado = False
    estaba_visible = False
    en_mano_estable = False
    mano_estable = None
    mano_candidata = None
    frames_nueva = 0
    frames_sin_agarre = 0
    suelto_confirmado = False

    for frame in range(nframes):
        det = visibles.get(frame)
        en_mano_raw = bool(det is not None and det.en_mano and det.mano in ("izquierda", "derecha"))
        if not empezado and not en_mano_raw:
            det_util = None
            visible = False
            en_mano = False
            mano_frame = None
            estado_obj = "ausente"
        else:
            if en_mano_raw:
                empezado = True
            det_util = det if empezado else None
            visible = det_util is not None
            if en_mano_raw:
                frames_sin_agarre = 0
                if mano_estable is None:
                    mano_estable = det.mano
                    frames_nueva = 0
                    mano_candidata = None
                elif det.mano == mano_estable:
                    frames_nueva = 0
                    mano_candidata = None
                else:
                    if mano_candidata != det.mano:
                        mano_candidata = det.mano
                        frames_nueva = 1
                    else:
                        frames_nueva += 1
                    if frames_nueva >= frames_traspaso:
                        eventos.append(
                            {
                                "tipo": "cambio_de_mano",
                                "frame": frame,
                                "tiempo_s": round(frame / fps, 4),
                                "desde": mano_estable,
                                "hacia": det.mano,
                                "centro_xy": [round(det.centro[0], 2), round(det.centro[1], 2)],
                                "bbox_xyxy": list(det.bbox),
                                "id_sam3": det.id_sam3,
                            }
                        )
                        mano_estable = det.mano
                        mano_candidata = None
                        frames_nueva = 0
                mano_frame = mano_estable
                if suelto_confirmado:
                    suelto_confirmado = False
                    eventos.append(
                        {
                            "tipo": "reagarre",
                            "frame": frame,
                            "tiempo_s": round(frame / fps, 4),
                            "mano": mano_frame,
                            "centro_xy": [round(det.centro[0], 2), round(det.centro[1], 2)],
                            "bbox_xyxy": list(det.bbox),
                            "id_sam3": det.id_sam3,
                        }
                    )
                elif not estaba_visible:
                    eventos.append(
                        {
                            "tipo": "reaparece" if any(e["tipo"] == "aparece" for e in eventos) else "aparece",
                            "frame": frame,
                            "tiempo_s": round(frame / fps, 4),
                            "centro_xy": [round(det.centro[0], 2), round(det.centro[1], 2)],
                            "bbox_xyxy": list(det.bbox),
                            "mano": mano_frame,
                            "id_sam3": det.id_sam3,
                        }
                    )
                en_mano = True
                en_mano_estable = True
                estado_obj = "en_mano"
            elif visible:
                frames_sin_agarre += 1
                mano_frame = mano_estable
                if en_mano_estable and frames_sin_agarre >= frames_suelte:
                    en_mano_estable = False
                    suelto_confirmado = True
                    eventos.append(
                        {
                            "tipo": "suelta",
                            "frame": frame,
                            "tiempo_s": round(frame / fps, 4),
                            "mano": mano_frame,
                            "centro_xy": [round(det.centro[0], 2), round(det.centro[1], 2)],
                            "bbox_xyxy": list(det.bbox),
                            "ultima_en_mano": ultima_en_mano,
                            "id_sam3": det.id_sam3,
                        }
                    )
                    en_mano = False
                    estado_obj = "suelto"
                elif en_mano_estable:
                    en_mano = True
                    estado_obj = "en_mano"
                else:
                    en_mano = False
                    if not estaba_visible:
                        eventos.append(
                            {
                                "tipo": "reaparece_suelto",
                                "frame": frame,
                                "tiempo_s": round(frame / fps, 4),
                                "mano": mano_frame,
                                "centro_xy": [round(det.centro[0], 2), round(det.centro[1], 2)],
                                "bbox_xyxy": list(det.bbox),
                                "id_sam3": det.id_sam3,
                            }
                        )
                    estado_obj = "suelto"
            else:
                mano_frame = mano_estable
                en_mano = False
                if estaba_visible and ultima is not None:
                    eventos.append(
                        {
                            "tipo": "desaparece",
                            "frame": frame,
                            "tiempo_s": round(frame / fps, 4),
                            "ultima_posicion": ultima,
                            "ultima_en_mano": ultima_en_mano,
                        }
                    )
                en_mano_estable = False
                suelto_confirmado = False
                estado_obj = "oculto" if ultima is not None else "ausente"

            if visible and det_util is not None:
                ultima = _vista(det_util, fps, mano_frame)
                if en_mano_raw:
                    ultima_en_mano = ultima

        roi = estado[frame]["roi"]
        frames.append(
            {
                "frame": frame,
                "tiempo_s": round(frame / fps, 4),
                "persona_visible": estado[frame]["persona_visible"],
                "roi_xyxy": list(roi) if roi is not None else None,
                "manos": [
                    {
                        "id_sam3": m["id"],
                        "lado": m["lado"],
                        "centro_xy": [round(m["centro"][0], 2), round(m["centro"][1], 2)],
                        "bbox_xyxy": list(m["bbox"]),
                        "probabilidad": round(m["prob"], 4),
                    }
                    for m in estado[frame]["manos"]
                ],
                "pose": [
                    {
                        "lado": p["lado"],
                        "muneca_xy": [round(p["centro"][0], 1), round(p["centro"][1], 1)],
                        "palma_xy": [round(p["palma"][0], 1), round(p["palma"][1], 1)],
                    }
                    for p in estado[frame].get("pose") or []
                ],
                "objeto": {
                    "visible": visible,
                    "en_mano": en_mano,
                    "estado": estado_obj,
                    "id_sam3": det_util.id_sam3 if det_util else None,
                    "centro_xy": (
                        [round(det_util.centro[0], 2), round(det_util.centro[1], 2)]
                        if det_util
                        else None
                    ),
                    "bbox_xyxy": list(det_util.bbox) if det_util else None,
                    "mano": mano_frame,
                    "probabilidad": round(det_util.prob, 4) if det_util else None,
                    "area_px": det_util.area if det_util else 0,
                    "score_agarre": round(det_util.score_agarre, 4) if det_util else None,
                    "ultima_posicion": ultima,
                    "ultima_en_mano": ultima_en_mano,
                },
            }
        )
        estaba_visible = visible

    if ultima is not None:
        ultima = dict(ultima)
        ultima["en_mano"] = en_mano_estable
        ultima["ultima_en_mano"] = ultima_en_mano
    return frames, eventos, ultima


def _fuente(tamano: int):
    from PIL import ImageFont

    for ruta in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(ruta).is_file():
            return ImageFont.truetype(ruta, tamano)
    return ImageFont.load_default()


def pintar_mascara(imagen: np.ndarray, mascara: np.ndarray, color: tuple[int, int, int], alfa: float) -> None:
    if mascara.shape[:2] != imagen.shape[:2]:
        mascara = cv2.resize(
            mascara.astype(np.uint8),
            (imagen.shape[1], imagen.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if not mascara.any():
        return
    mezcla = imagen[mascara].astype(np.float32) * (1 - alfa) + np.array(color, np.float32) * alfa
    imagen[mascara] = mezcla.astype(np.uint8)


def anotar_video(
    video: Path,
    destino: Path,
    personas: dict[int, list[dict]],
    persona_id: int,
    objetos: dict[int, list[dict]],
    estado: dict[int, dict],
    frames_json: list[dict],
    fps: float,
    carpeta_frames: Path | None = None,
) -> None:
    from PIL import Image, ImageDraw

    if carpeta_frames is not None:
        carpeta_frames.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"No se puede leer {video} para anotar")
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(destino),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ancho, alto),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"No se pudo crear el vídeo {destino}")

    fuente = _fuente(28)
    fuente_peq = _fuente(22)
    indice = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if indice >= len(frames_json):
            break
        info = frames_json[indice]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        persona = instancia(personas, indice, persona_id)
        if persona is not None:
            pintar_mascara(rgb, persona["mascara"], (40, 180, 70), 0.28)
        for mano in estado[indice]["manos"]:
            color = (40, 140, 255) if mano["lado"] == "izquierda" else (255, 150, 30)
            pintar_mascara(rgb, mano["mascara"], color, 0.45)

        for pose in info.get("pose") or []:
            px, py = pose["palma_xy"]
            color = (40, 140, 255) if pose["lado"] == "izquierda" else (255, 150, 30)
            cv2.circle(rgb, (int(px), int(py)), 10, color, 2)

        obj = info["objeto"]
        if obj["visible"] and obj["id_sam3"] is not None:
            en_mano = obj.get("en_mano", True)
            color_obj = (220, 40, 200) if en_mano else (255, 160, 40)
            inst_obj = instancia(objetos, indice, obj["id_sam3"])
            if inst_obj is not None:
                pintar_mascara(rgb, inst_obj["mascara"], color_obj, 0.55)
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            cv2.rectangle(rgb, (x0, y0), (x1, y1), color_obj, 3)
        elif obj["ultima_posicion"] is not None:
            cx, cy = obj["ultima_posicion"]["centro_xy"]
            ix, iy = int(cx), int(cy)
            cv2.drawMarker(
                rgb, (ix, iy), (220, 40, 200), cv2.MARKER_TILTED_CROSS, 28, 2
            )
        if obj.get("estado") == "suelto" and obj.get("ultima_en_mano"):
            cx, cy = obj["ultima_en_mano"]["centro_xy"]
            cv2.drawMarker(
                rgb, (int(cx), int(cy)), (255, 210, 60), cv2.MARKER_CROSS, 22, 2
            )

        roi = info["roi_xyxy"]
        if roi is not None:
            cv2.rectangle(rgb, (roi[0], roi[1]), (roi[2], roi[3]), (255, 220, 0), 2)

        imagen = Image.fromarray(rgb)
        draw = ImageDraw.Draw(imagen)
        if obj.get("estado") == "en_mano":
            estado_txt = f"Objeto EN MANO  {obj['mano'] or '?'}"
            color_txt = (255, 80, 220)
        elif obj.get("estado") == "suelto":
            estado_txt = f"Objeto SUELTO  última mano {obj['mano'] or '?'}"
            color_txt = (255, 170, 40)
        elif obj["ultima_posicion"] is not None:
            ult = obj["ultima_posicion"]
            estado_txt = (
                f"Objeto NO visible  última pos. frame {ult['frame']} "
                f"({ult['centro_xy'][0]:.0f}, {ult['centro_xy'][1]:.0f}) "
                f"mano {ult['mano'] or '?'}"
            )
            color_txt = (255, 210, 60)
        else:
            estado_txt = "Objeto aún no detectado en la mano"
            color_txt = (200, 200, 200)
        draw.rectangle((8, 8, ancho - 8, 78), fill=(0, 0, 0))
        draw.text((16, 12), estado_txt, fill=color_txt, font=fuente)
        draw.text(
            (16, 46),
            f"frame {indice}   t={indice / fps:.2f}s   ROI persona en amarillo",
            fill=(230, 230, 230),
            font=fuente_peq,
        )
        bgr = cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        if carpeta_frames is not None:
            cv2.imwrite(
                str(carpeta_frames / f"frame_{indice:05d}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
        indice += 1

    cap.release()
    writer.release()
    if carpeta_frames is not None:
        print(f"  {indice} frames en {carpeta_frames}")


def quitar_parpadeos(eventos: list[dict], max_hueco: int) -> list[dict]:
    """Un hueco de pocos frames no es una desaparición."""
    if max_hueco <= 0:
        return eventos
    salida = []
    i = 0
    while i < len(eventos):
        evento = eventos[i]
        if (
            evento["tipo"] == "desaparece"
            and i + 1 < len(eventos)
            and eventos[i + 1]["tipo"] in ("reaparece", "reaparece_suelto")
            and eventos[i + 1]["frame"] - evento["frame"] <= max_hueco
        ):
            i += 2
            continue
        salida.append(evento)
        i += 1
    return salida


def describir_eventos(eventos: list[dict], ultima: dict | None) -> list[str]:
    lineas = []
    for ev in eventos:
        if ev["tipo"] in ("aparece", "reaparece", "reaparece_suelto"):
            verbo = {
                "aparece": "Aparece",
                "reaparece": "Reaparece en la mano",
                "reaparece_suelto": "Reaparece suelto",
            }[ev["tipo"]]
            lineas.append(
                f"{verbo} en el frame {ev['frame']} ({ev['tiempo_s']} s), "
                f"mano {ev['mano']}, centro {ev['centro_xy']}"
            )
        elif ev["tipo"] == "desaparece":
            ult = ev["ultima_posicion"]
            frase = (
                f"Deja de verse en el frame {ev['frame']} ({ev['tiempo_s']} s). "
                f"Última posición: frame {ult['frame']} "
                f"centro {ult['centro_xy']} bbox {ult['bbox_xyxy']} "
                f"mano {ult['mano']}"
            )
            en_mano = ev.get("ultima_en_mano")
            if en_mano and en_mano.get("frame") != ult.get("frame"):
                frase += (
                    f". Última vez en la mano: frame {en_mano['frame']} "
                    f"centro {en_mano['centro_xy']} bbox {en_mano['bbox_xyxy']}"
                )
            lineas.append(frase)
        elif ev["tipo"] == "cambio_de_mano":
            lineas.append(
                f"Pasa de la mano {ev['desde']} a la {ev['hacia']} "
                f"en el frame {ev['frame']} ({ev['tiempo_s']} s), centro {ev['centro_xy']}"
            )
        elif ev["tipo"] == "suelta":
            en_mano = ev.get("ultima_en_mano") or {}
            lineas.append(
                f"Se suelta en el frame {ev['frame']} ({ev['tiempo_s']} s) "
                f"y sigue en escena, centro {ev['centro_xy']}. "
                f"Última vez en la mano: frame {en_mano.get('frame', ev['frame'])} "
                f"centro {en_mano.get('centro_xy', ev['centro_xy'])} "
                f"mano {en_mano.get('mano', ev.get('mano'))}"
            )
        elif ev["tipo"] == "reagarre":
            lineas.append(
                f"Vuelve a la mano {ev['mano']} en el frame {ev['frame']} "
                f"({ev['tiempo_s']} s), centro {ev['centro_xy']}"
            )
    if ultima is not None and (not eventos or eventos[-1]["tipo"] != "desaparece"):
        if ultima.get("en_mano"):
            lineas.append(
                "Sigue en la mano al final del vídeo. "
                f"Última posición: frame {ultima['frame']} centro {ultima['centro_xy']} "
                f"mano {ultima['mano']}"
            )
        else:
            en_mano = ultima.get("ultima_en_mano") or ultima
            lineas.append(
                "Sigue en escena al final del vídeo, ya no en la mano. "
                f"Última posición: frame {ultima['frame']} centro {ultima['centro_xy']}. "
                f"Última vez en la mano: frame {en_mano.get('frame')} "
                f"centro {en_mano.get('centro_xy')} mano {en_mano.get('mano')}"
            )
    return lineas


def resumen_texto(video: Path, eventos: list[dict], ultima: dict | None, nframes: int) -> str:
    lineas = [f"Vídeo: {video}", f"Frames: {nframes}", ""]
    if not eventos and ultima is None:
        lineas.append(
            "SAM 3 no asoció ningún objeto a las manos dentro del ROI. "
            "Prueba un prompt más concreto, por ejemplo --prompt-objeto bottle"
        )
        return "\n".join(lineas) + "\n"
    lineas.extend(describir_eventos(eventos, ultima))
    lineas.append(
        "\nLa mano sale del esqueleto (YOLO-pose): izquierda y derecha son las de la persona. "
        "Un cambio de mano solo se anota si se mantiene 3 frames. "
        "SAM 3 confirma una pista nueva tras unos frames (hotstart, unos 15)."
    )
    return "\n".join(lineas) + "\n"


def procesar_video(predictor, video: Path, args, salida_dir: Path) -> None:
    ancho, alto, nframes, fps = info_video(video)
    print(f"\n=== {video.name}: {ancho}x{alto}, {nframes} frames, {fps:.3f} fps ===")
    destino = carpeta_de(salida_dir, "sam3", video)
    medidor = Medidor("sam3")
    try:
        _procesar_sam3(predictor, video, args, destino, medidor, ancho, alto, nframes, fps)
    except Exception as exc:
        publicar_comparativa(
            salida_dir, "sam3", video.stem, "SAM 3", destino, medidor.cerrar(nframes),
            0, 0, 0, ok=False, codigo=1, error=str(exc),
        )
        raise


def _procesar_sam3(predictor, video, args, destino, medidor, ancho, alto, nframes, fps) -> None:
    with medidor.etapa("segmentacion"):
        personas = seguir_concepto(predictor, video, args.prompt_persona, args.umbral)
        persona_id = elegir_persona(personas)
        if persona_id is None:
            print("  No se segmentó ninguna persona. Se omite este vídeo.")
            metricas = medidor.cerrar(nframes)
            (destino / "metricas.json").write_text(
                json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            publicar_comparativa(
                args.salida, "sam3", video.stem, "SAM 3", destino, metricas,
                0, 0, 0, ok=False, codigo=0, error="sin persona",
            )
            return
        print(f"  Persona principal: id SAM 3 {persona_id}")
        manos = seguir_concepto(predictor, video, args.prompt_mano, args.umbral)
        palma = detecciones_palma(video, nframes)
        if palma is not None:
            por_id_palma, objetos = palma
        else:
            por_id_palma = None
            objetos = seguir_concepto(predictor, video, args.prompt_objeto, args.umbral)
        pose = None if args.sin_pose else analizar_pose(video, nframes, args.modelo_pose)
        estado, por_id = asociar_objetos(
            nframes=nframes,
            personas=personas,
            persona_id=persona_id,
            manos=manos,
            objetos=objetos,
            ancho=ancho,
            alto=alto,
            padding=args.padding,
            dilatacion=args.dilatacion_mano,
            contacto_min=args.contacto_min,
            roi_min=args.roi_min,
            de_frente=not args.de_espaldas,
            manos_pose=pose,
            umbral_agarre=args.umbral_agarre,
        )
    if por_id_palma is not None:
        por_id = por_id_palma
    cadena = encadenar(por_id, args.hueco_max, args.distancia_max)
    frames_json, eventos, ultima = linea_temporal(
        nframes, fps, estado, cadena, frames_traspaso=args.frames_traspaso
    )
    eventos = quitar_parpadeos(eventos, 3)
    metricas = medidor.cerrar(nframes)

    anotado = destino / "anotado.mp4"
    print(f"  Escribiendo {anotado}")
    anotar_video(
        video, anotado, personas, persona_id, objetos, estado, frames_json, fps,
        carpeta_frames=destino / "frames",
    )

    informe = {
        "video": str(video),
        "ancho": ancho,
        "alto": alto,
        "frames": nframes,
        "fps": fps,
        "prompts": {
            "persona": args.prompt_persona,
            "mano": args.prompt_mano,
            "objeto": args.prompt_objeto,
        },
        "padding_roi": args.padding,
        "persona_de_frente": not args.de_espaldas,
        "persona_id_sam3": persona_id,
        "ids_objeto_sam3": sorted({d.id_sam3 for d in cadena.detecciones})
        if cadena
        else [],
        "eventos": eventos,
        "linea_temporal": frames_json,
    }
    (destino / "seguimiento.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    texto = resumen_texto(video, eventos, ultima, nframes)
    (destino / "resumen.txt").write_text(texto, encoding="utf-8")
    (destino / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frames_con_objeto = sum(1 for frame in frames_json if frame["objeto"]["visible"])
    objetos_n = len({d.id_sam3 for d in cadena.detecciones}) if cadena else 0
    desapariciones = sum(1 for ev in eventos if ev["tipo"] == "desaparece")
    ruta = publicar_comparativa(
        args.salida, "sam3", video.stem, "SAM 3", destino, metricas,
        objetos_n, desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 3: ROI de la persona y seguimiento del objeto en la mano."
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=Path("videos"),
        help="Vídeo o carpeta de vídeos (por defecto: videos/).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("salidas"),
        help="Carpeta de resultados (por defecto: salidas/).",
    )
    parser.add_argument("--prompt-persona", default="person")
    parser.add_argument("--prompt-mano", default="hand")
    parser.add_argument(
        "--prompt-objeto",
        default="object held in the hand",
        help="Concepto en inglés del objeto. Si falla, usa un nombre concreto (bottle, cup, phone).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.35,
        help="Padding del ROI como fracción del ancho y alto de la persona.",
    )
    parser.add_argument("--umbral", type=float, default=0.5, help="Probabilidad mínima de SAM 3.")
    parser.add_argument("--dilatacion-mano", type=int, default=28)
    parser.add_argument("--contacto-min", type=float, default=0.08)
    parser.add_argument(
        "--umbral-agarre",
        type=float,
        default=0.35,
        help="Puntuación mínima de palma y dedos para dar el objeto por cogido.",
    )
    parser.add_argument(
        "--frames-traspaso",
        type=int,
        default=3,
        help="Frames seguidos en la otra mano antes de anotar el cambio.",
    )
    parser.add_argument(
        "--modelo-pose",
        type=Path,
        default=ruta_pose_por_defecto(),
        help="YOLO-pose para la muñeca izquierda y derecha de la persona.",
    )
    parser.add_argument(
        "--sin-pose",
        action="store_true",
        help="No usa el esqueleto. El lado de la mano sale de la imagen.",
    )
    parser.add_argument("--roi-min", type=float, default=0.5)
    parser.add_argument("--hueco-max", type=int, default=45)
    parser.add_argument("--distancia-max", type=float, default=250.0)
    parser.add_argument(
        "--de-espaldas",
        action="store_true",
        help="La persona no mira a la cámara. Por defecto se asume que está de frente, "
        "así que su mano izquierda cae a la derecha de la imagen.",
    )
    parser.add_argument(
        "--version",
        choices=("sam3.1", "sam3"),
        default="sam3.1",
        help="sam3.1 es el checkpoint actual (Object Multiplex). sam3 es el de noviembre 2025.",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=4,
        help="Frames que SAM 3 procesa a la vez. 16 no cabe en 32 GiB si hay otros procesos en la GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    videos = listar_videos(args.videos)
    args.salida.mkdir(parents=True, exist_ok=True)
    if args.lote < 1:
        raise SystemExit("--lote tiene que ser al menos 1.")
    predictor = construir_predictor(args.version, args.umbral, args.lote)
    try:
        for video in videos:
            procesar_video(predictor, video, args, args.salida)
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()
    print(f"\nListo. Resultados en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
