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
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from detector_palma.agarre import anotar_agarre, es_voluminoso, puntuar_agarre
from detector_palma.geometria import caja_desde_mascara, iou_cajas
from detector_palma.personas import dibujar_pose
from detector_palma.tipos import Instancia
from salida_comun import Medidor, carpeta_de, publicar_comparativa

RAIZ = Path(__file__).resolve().parent
PESOS = RAIZ / "pesos"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
YAML_IDENTIFICADORES = RAIZ / "identificadores.yaml"
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


def cargar_catalogo(ruta: Path) -> list[Identificador]:
    """Lee identificadores.yaml. ``activo: false`` apaga esa detección inmediata."""
    if not ruta.is_file():
        raise SystemExit(f"No existe el YAML de identificadores {ruta}")
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    bloque = datos.get("identificadores")
    if not isinstance(bloque, dict) or not bloque:
        raise SystemExit(f"{ruta} no tiene el mapa ``identificadores``.")
    catalogo: list[Identificador] = []
    for clave, raw in bloque.items():
        if not isinstance(raw, dict):
            raise SystemExit(f"La entrada «{clave}» de {ruta} no es un mapa.")
        uid = raw.get("ultralytics_id")
        prompts = [str(p).strip() for p in (raw.get("prompts") or []) if str(p).strip()]
        nombre = raw.get("nombre")
        if nombre is not None:
            nombre = str(nombre).strip() or None
        catalogo.append(
            Identificador(
                clave=str(clave),
                activo=bool(raw.get("activo", True)),
                ultralytics_id=int(uid) if uid is not None and uid != "" else None,
                nombre=nombre,
                prompts=prompts,
                destino=bool(raw.get("destino", False)),
                accesorio=bool(raw.get("accesorio", False)),
            )
        )
    return catalogo


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


def offsets_muestreo(fps_video: float, fps_proceso: int | None) -> tuple[set[int] | None, int]:
    """Offsets dentro de cada segundo de vídeo. ``None`` = todos los frames."""
    rate = max(1, int(round(float(fps_video) or 25.0)))
    if fps_proceso is None:
        return None, rate
    n = int(fps_proceso)
    if n < 1:
        raise SystemExit("--fps debe ser un entero mayor o igual que 1")
    if n >= rate:
        return None, rate
    return {int(i * rate / n) for i in range(n)}, rate


def describir_muestreo(fps_proceso, fps_video: float, n_origen: int, n_procesados: int) -> dict:
    """Texto y metadatos de si --fps se usó o el análisis fue frame a frame."""
    offsets, rate = offsets_muestreo(fps_video, fps_proceso)
    base = {
        "fps_video": round(float(fps_video), 4),
        "rate_entero": rate,
        "frames_origen": n_origen,
        "frames_procesados": n_procesados,
    }
    if fps_proceso is None:
        return {
            **base,
            "especificado": False,
            "fps_parametro": None,
            "descripcion": (
                "No se ha especificado --fps. El procesamiento ha sido de frame a frame."
            ),
        }
    if offsets is None:
        return {
            **base,
            "especificado": True,
            "fps_parametro": int(fps_proceso),
            "descripcion": (
                f"Se ha establecido --fps {int(fps_proceso)}, igual o mayor que el rate "
                f"del vídeo ({rate} fps). El procesamiento ha sido de frame a frame."
            ),
        }
    ejemplo = ", ".join(str(i + 1) for i in sorted(offsets))
    descripcion = (
        f"Se ha establecido --fps {int(fps_proceso)}. El vídeo va a {fps_video:.3f} fps "
        f"(rate {rate}): de cada segundo se procesan los frames {ejemplo} "
        f"({n_procesados} de {n_origen} frames)."
    )
    return {
        "especificado": True,
        "fps_parametro": int(fps_proceso),
        "fps_video": round(float(fps_video), 4),
        "rate_entero": rate,
        "frames_origen": n_origen,
        "frames_procesados": n_procesados,
        "offsets_en_segundo": sorted(offsets),
        "descripcion": descripcion,
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
    cap.release()
    if not frames:
        raise SystemExit(f"{video} no tiene frames")
    return frames, fps, origen


def resolver_videos(args) -> list[Path]:
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
    catalogo = cargar_catalogo(args.identificadores)
    args.catalogo = catalogo
    activos = [item for item in catalogo if item.activo]
    escena = AnalizadorEscena(
        estrategia="pose",
        modelo_pose=str(args.modelo_pose),
        dispositivo=dispositivo,
        padding_persona=args.padding,
        radio_mano=args.radio_mano,
        conf=args.conf_pose,
        imgsz=args.imgsz,
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
    print(f"  Detectores en {dispositivo}: pose + seg + {'MobileSAM' if sam else 'sin SAM'}")
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
    x, y = centro
    x1, y1, x2, y2 = caja
    return x1 - margen <= x <= x2 + margen and y1 - margen <= y <= y2 + margen


def _radio_palma_objeto(mano, args) -> float:
    """Radio de aceptación: el objeto tiene que estar en la palma, no solo en el recorte."""
    return max(24.0, float(mano.radio) * args.factor_palma)


def _pegado_a_palma(inst: Instancia, mano, args) -> bool:
    palma = mano.palma or mano.punto
    dist = float(np.hypot(inst.centro[0] - palma[0], inst.centro[1] - palma[1]))
    return dist <= _radio_palma_objeto(mano, args)


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
    for mano in contexto.manos:
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
            if es_voluminoso(inst, contexto):
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
    en_mano = bool(lado) and score >= args.umbral_agarre and contacto >= args.contacto_min
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
            from ultralytics import YOLOE

            if not Path(pesos_yoloe).is_file():
                print(f"  Sin YOLOE: falta {pesos_yoloe}. Solo clases COCO del YAML.")
            else:
                self.yoloe = YOLOE(str(pesos_yoloe))
                self.yoloe.set_classes(textos, self.yoloe.get_text_pe(textos))

    def detectar(self, frame) -> list[dict]:
        salida: list[dict] = []
        if self.yolo is not None and self.ids_yolo:
            for inst in self.yolo.detectar_clases(frame, self.ids_yolo, self.conf):
                ident = self.por_id.get(int(inst.extra.get("clase_id", -1)))
                if ident is None:
                    ident = self.por_texto.get((inst.etiqueta or "").lower())
                if ident is None:
                    continue
                salida.append(_det_identificador(ident, inst.etiqueta, inst.score, inst.caja))
        if self.yoloe is not None:
            salida.extend(self._detectar_yoloe(frame))
        return _fusionar_contenedores(salida)

    def _detectar_yoloe(self, frame) -> list[dict]:
        alto, ancho = frame.shape[:2]
        resultados = self.yoloe.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.dispositivo,
            retina_masks=True,
            verbose=False,
        )
        if not resultados or resultados[0].boxes is None:
            return []
        resultado = resultados[0]
        nombres = resultado.names or self.yoloe.names
        salida: list[dict] = []
        for caja in resultado.boxes:
            texto = str(nombres[int(caja.cls)]).lower()
            ident = self.por_texto.get(texto)
            if ident is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho, x2), min(alto, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            salida.append(
                _det_identificador(ident, texto, float(caja.conf), (x1, y1, x2, y2))
            )
        return salida


def _det_identificador(ident: Identificador, texto: str, score: float, caja) -> dict:
    return {
        "categoria": ident.clave,
        "texto": texto,
        "score": float(score),
        "caja": caja,
        "destino": ident.destino,
        "accesorio": ident.accesorio,
    }


DetectorContenedores = DetectorIdentificadores


def _fusionar_contenedores(detecciones: list[dict], iou_max: float = 0.55) -> list[dict]:
    """Varios prompts ven el mismo objeto (cart y trolley): se queda el de más score."""
    salida: list[dict] = []
    for det in sorted(detecciones, key=lambda d: -d["score"]):
        if any(
            iou_cajas(det["caja"], ya["caja"]) > iou_max and ya["categoria"] == det["categoria"]
            for ya in salida
        ):
            continue
        salida.append(det)
    return salida


def _centro_caja(caja) -> tuple[float, float]:
    x1, y1, x2, y2 = caja
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diagonal(caja) -> float:
    x1, y1, x2, y2 = caja
    return float(np.hypot(max(1, x2 - x1), max(1, y2 - y1)))


def _misma_pista(caja_a, caja_b, iou_min: float, factor_centro: float) -> bool:
    """El contenedor se mueve con la persona: el IoU solo no basta para reconocerlo."""
    if iou_cajas(caja_a, caja_b) >= iou_min:
        return True
    ca, cb = _centro_caja(caja_a), _centro_caja(caja_b)
    distancia = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    return distancia <= factor_centro * 0.5 * (_diagonal(caja_a) + _diagonal(caja_b))


def seguir_contenedores(frames, detector, cada: int, iou_min: float = 0.3,
                        factor_centro: float = 0.8) -> list[dict]:
    """Rastrea los contenedores por IoU y cercanía para tener un id estable y su último ROI."""
    if detector is None:
        return []
    pistas: list[dict] = []
    for indice in range(0, len(frames), max(1, cada)):
        for det in detector.detectar(frames[indice]):
            mejor, mejor_iou = None, -1.0
            for pista in pistas:
                if pista["categoria"] != det["categoria"]:
                    continue
                if indice == pista["ultimo_frame"]:
                    continue
                if not _misma_pista(det["caja"], pista["caja"], iou_min, factor_centro):
                    continue
                iou = iou_cajas(det["caja"], pista["caja"])
                if iou > mejor_iou:
                    mejor, mejor_iou = pista, iou
            if mejor is not None:
                mejor["caja"] = det["caja"]
                mejor["ultimo_frame"] = indice
                mejor["scores"].append(det["score"])
                mejor["cajas"][indice] = det["caja"]
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
                    }
                )
    pistas = _unir_contenedores(pistas)
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


def _unir_contenedores(pistas: list[dict], iou_min: float = 0.4) -> list[dict]:
    """El mismo mueble puede salir como carrito y como cesto: gana el de más score."""
    pistas = sorted(pistas, key=lambda p: -max(p["scores"]))
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
        destino["primer_frame"] = min(destino["primer_frame"], pista["primer_frame"])
        if pista["ultimo_frame"] > destino["ultimo_frame"]:
            destino["ultimo_frame"] = pista["ultimo_frame"]
            destino["caja"] = pista["caja"]
    return unidas


def _compatibles_para_unir(a: dict, b: dict) -> bool:
    """Carrito y cesto pueden ser el mismo mueble. Un celular no se fusiona con el cesto."""
    if a["categoria"] == b["categoria"]:
        return True
    return {a["categoria"], b["categoria"]} <= {"carrito", "cesto"}


def caja_contenedor(pista: dict, frame: int) -> tuple[int, int, int, int]:
    """ROI del contenedor en el frame más cercano al pedido."""
    cercano = min(pista["cajas"], key=lambda f: abs(f - frame))
    return pista["cajas"][cercano]


def _fraccion_caja_dentro(caja, contenedor) -> float:
    x1, y1, x2, y2 = caja
    area = float(max(1, x2 - x1) * max(1, y2 - y1))
    cx1, cy1, cx2, cy2 = contenedor
    ix1, iy1 = max(x1, cx1), max(y1, cy1)
    ix2, iy2 = min(x2, cx2), min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1)) / area


def contenedor_final(vistas: list[Vista], contenedores: list[dict], args) -> dict | None:
    """¿El objeto acabó dentro del ROI de un carrito, cesto, bolso, mochila o bolsa?"""
    if not vistas or not contenedores:
        return None
    ultimas = vistas[-args.frames_fin :]
    mejor = None
    for pista in contenedores:
        if not pista.get("destino", False):
            continue
        dentro = []
        for vista in ultimas:
            caja = caja_contenedor(pista, vista.frame)
            area_cont = float(
                max(1, caja[2] - caja[0]) * max(1, caja[3] - caja[1])
            )
            area_obj = float(
                max(1, vista.caja[2] - vista.caja[0]) * max(1, vista.caja[3] - vista.caja[1])
            )
            # Un contenedor tiene que ser claramente más grande que lo que guarda:
            # así el propio objeto no se cuenta como su propio contenedor.
            if area_cont < args.factor_contenedor * area_obj:
                continue
            dentro.append(_fraccion_caja_dentro(vista.caja, caja))
        if not dentro:
            continue
        fraccion = float(np.mean(dentro))
        votos = sum(1 for f in dentro if f >= args.dentro_min) / len(dentro)
        if votos < 0.5:
            continue
        candidato = {
            "id": pista["id"],
            "categoria": pista["categoria"],
            "roi": list(caja_contenedor(pista, ultimas[-1].frame)),
            "fraccion_dentro": round(fraccion, 4),
            "frames_evaluados": len(dentro),
            "score": pista["score_max"],
        }
        if mejor is None or candidato["fraccion_dentro"] > mejor["fraccion_dentro"]:
            mejor = candidato
    return mejor


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
    return elegidos


def detectar_objetos(frames, escena, detector, sam, args):
    """Recorre el vídeo y agrupa en pistas los objetos vistos en las manos."""
    contextos: list = []
    pistas: list[dict] = []
    total = 0
    for indice, frame in enumerate(frames):
        contexto = escena.analizar(frame)
        contextos.append(contexto)
        for inst, agarre in objetos_del_frame(frame, contexto, detector, sam, args):
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
                    and max(area_prev, inst.area) > 4.0 * min(area_prev, inst.area)
                )
            else:
                cambio_lejano = False
                salto_area = False
            if mejor is not None and mejor_d <= args.distancia_max and not cambio_lejano and not salto_area:
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
    pistas = [p for p in pistas if len(p["mascaras"]) >= args.min_detecciones]
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
            if max(area_ya, area_nueva) > 4.0 * min(area_ya, area_nueva):
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
        es_objeto = False
        for objeto in objetos:
            coincidencias = 0
            comparados = 0
            for frame, caja in pista["cajas"].items():
                vista = objeto.vistas.get(frame)
                if vista is None:
                    continue
                comparados += 1
                area_cont = float(max(1, caja[2] - caja[0]) * max(1, caja[3] - caja[1]))
                area_obj = float(
                    max(1, vista.caja[2] - vista.caja[0]) * max(1, vista.caja[3] - vista.caja[1])
                )
                if (
                    iou_cajas(caja, vista.caja) >= 0.35
                    and area_cont < args.factor_contenedor * area_obj
                ):
                    coincidencias += 1
            if comparados and coincidencias / comparados >= 0.5:
                es_objeto = True
                break
        if es_objeto:
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
        votos = 0
        comparados = 0
        for frame, vista in objeto.vistas.items():
            for accesorio in accesorios:
                caja = caja_contenedor(accesorio, frame)
                comparados += 1
                if (
                    iou_cajas(vista.caja, caja) >= 0.25
                    or _fraccion_caja_dentro(vista.caja, caja) >= 0.45
                ):
                    votos += 1
                    break
        if comparados and votos / comparados >= 0.35:
            print(f"  Objeto {objeto.id} se descarta: es un bolso o mochila, no va en la palma")
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


def fusionar_objetos(objetos: list[Objeto], comunes_min: int = 3, solape_min: float = 0.3):
    """Dos propagaciones sobre la misma cosa dejan una sola: la de más frames en mano."""
    def en_mano(objeto: Objeto) -> int:
        return sum(1 for v in objeto.vistas.values() if v.mano)

    supervivientes: list[Objeto] = []
    for objeto in sorted(objetos, key=lambda o: (-en_mano(o), -len(o.vistas))):
        duplicado = False
        for ya in supervivientes:
            comunes = set(objeto.mascaras) & set(ya.mascaras)
            if len(comunes) < comunes_min:
                continue
            solapes = [
                1.0 if _solapan(objeto.mascaras[f], ya.mascaras[f], 0.3) else 0.0
                for f in comunes
            ]
            if float(np.mean(solapes)) >= solape_min:
                duplicado = True
                print(f"  Objeto {objeto.id} se une al {ya.id}: misma máscara")
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
        mascara = objeto.mascaras[idx]
        caja = caja_desde_mascara(mascara)
        if caja is None:
            continue
        agarre = puntuar_manos(mascara, caja, contextos[idx], args)
        ys, xs = np.nonzero(mascara)
        objeto.vistas[idx] = Vista(
            frame=idx,
            caja=caja,
            centro=(float(xs.mean()), float(ys.mean())),
            area=int(mascara.sum()),
            mano=agarre["mano"],
            score=round(agarre["score"], 4),
            scores=agarre["scores"],
            origen="deteccion" if idx in pista["mascaras"] else "propagacion",
        )


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


def estado_final(objeto: Objeto, nframes: int, contenedor: dict | None, args) -> str:
    vistas = [objeto.vistas[i] for i in sorted(objeto.vistas)]
    if not vistas:
        return "sin_datos"
    ultima = vistas[-1]
    if contenedor is not None:
        return "dentro_de_contenedor"
    if nframes - 1 - ultima.frame > args.hueco_parpadeo:
        return "desaparecido"
    if ultima.mano is None:
        return "suelto"
    return "en_mano"


# --------------------------------------------------------------------------- #
# Vídeo anotado
# --------------------------------------------------------------------------- #


def _etiqueta_mano(mano: str | None) -> str:
    return {"izquierda": "IZQ", "derecha": "DER", "ambas": "AMBAS"}.get(mano or "", "SUELTO")


def anotar(frames, destino: Path, objetos: list[Objeto], contenedores, fps_salida: float,
           fps_video: float, indices_origen: list[int] | None,
           carpeta_frames: Path | None, alfa: float,
           contextos=None, dibujar_esqueleto: bool = True) -> int:
    """Máscaras de objeto, ROI de identificadores y pose YOLO de la cintura hacia arriba."""
    if not frames:
        return 0
    alto, ancho = frames[0].shape[:2]
    escritor = cv2.VideoWriter(
        str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps_salida or 25.0, (ancho, alto)
    )
    if carpeta_frames is not None:
        carpeta_frames.mkdir(parents=True, exist_ok=True)
    for indice, original in enumerate(frames):
        frame = original.copy()
        orig = frame_origen(indice, indices_origen)
        for pista in contenedores:
            x1, y1, x2, y2 = caja_contenedor(pista, indice)
            color = COLOR_CONTENEDOR if pista.get("destino") else COLOR_IDENTIFICADOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, pista["id"], (x1 + 4, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )
        for objeto in objetos:
            vista = objeto.vistas.get(indice)
            if vista is None:
                continue
            color = COLOR_OBJETO[(objeto.id - 1) % len(COLOR_OBJETO)]
            mascara = objeto.mascaras[indice]
            frame[mascara] = (
                (1.0 - alfa) * frame[mascara] + alfa * np.array(color, dtype=np.float32)
            ).astype(np.uint8)
            x1, y1, x2, y2 = vista.caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"obj {objeto.id} {_etiqueta_mano(vista.mano)} {vista.score:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
        if dibujar_esqueleto and contextos is not None and indice < len(contextos):
            dibujar_pose(frame, contextos[indice])
        cv2.rectangle(frame, (0, 0), (ancho, 34), (0, 0, 0), -1)
        activos = sum(1 for o in objetos if indice in o.vistas)
        t = orig / fps_video if fps_video else 0.0
        cv2.putText(
            frame,
            f"frame {orig}  t={t:.2f}s  objetos en mano: {activos}",
            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )
        escritor.write(frame)
        if carpeta_frames is not None:
            cv2.imwrite(
                str(carpeta_frames / f"frame_{orig:05d}.jpg"),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
    escritor.release()
    return len(frames)


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #


def informe_objeto(objeto: Objeto, fps: float, nframes: int, contenedores, args,
                   indices_origen: list[int] | None = None) -> dict:
    vistas = [objeto.vistas[i] for i in sorted(objeto.vistas)]
    contenedor = contenedor_final(vistas, contenedores, args)
    manos = {"izquierda": 0, "derecha": 0, "ambas": 0, "sin_mano": 0}
    for vista in vistas:
        manos[vista.mano or "sin_mano"] += 1

    def orig(idx: int) -> int:
        return frame_origen(idx, indices_origen)

    linea = [
        {
            "frame": orig(v.frame),
            "frame_procesado": v.frame,
            "t": round(orig(v.frame) / fps, 4) if fps else None,
            "mano": v.mano,
            "roi": list(v.caja),
            "centro": [round(v.centro[0], 2), round(v.centro[1], 2)],
            "area": v.area,
            "score_agarre": v.score,
            "scores_mano": v.scores,
            "origen": v.origen,
        }
        for v in vistas
    ]
    en_mano = [v for v in vistas if v.mano]
    eventos = eventos_de(objeto, fps, args)
    for evento in eventos:
        interno = evento["frame"]
        evento["frame_procesado"] = interno
        evento["frame"] = orig(interno)
        evento["t"] = round(orig(interno) / fps, 4) if fps else None
    return {
        "id": objeto.id,
        "frames_visible": len(vistas),
        "frames_en_mano": len(en_mano),
        "primer_frame": orig(vistas[0].frame) if vistas else None,
        "ultimo_frame": orig(vistas[-1].frame) if vistas else None,
        "resiembras": objeto.resiembras,
        "manos": manos,
        "mano_predominante": max(manos, key=manos.get) if vistas else None,
        "ultimo_roi": list(vistas[-1].caja) if vistas else None,
        "ultimo_centro": (
            [round(vistas[-1].centro[0], 2), round(vistas[-1].centro[1], 2)] if vistas else None
        ),
        "ultimo_frame_visible": orig(vistas[-1].frame) if vistas else None,
        "estado_final": estado_final(objeto, nframes, contenedor, args),
        "acaba_en_contenedor": contenedor is not None,
        "contenedor_final": contenedor,
        "eventos": eventos,
        "linea_temporal": linea,
    }


def resumen(video: Path, informes: list[dict], contenedores, nframes: int, modelo: str,
            yaml_identificadores: Path | None = None, muestreo: dict | None = None) -> str:
    lineas = [
        f"Vídeo: {video}",
        f"Frames procesados: {nframes}",
        f"Modelo: {modelo}. Las muñecas salen de YOLO26n-pose, el objeto del recorte de "
        "la muñeca y los identificadores de escena del YAML "
        f"({yaml_identificadores or YAML_IDENTIFICADORES}).",
        (muestreo or {}).get(
            "descripcion",
            "No se ha especificado --fps. El procesamiento ha sido de frame a frame.",
        ),
        "",
    ]
    if contenedores:
        lineas.append("Identificadores de escena:")
        for pista in contenedores:
            extra = []
            if pista.get("destino"):
                extra.append("destino")
            if pista.get("accesorio"):
                extra.append("accesorio")
            sufijo = f" [{', '.join(extra)}]" if extra else ""
            lineas.append(
                f"  {pista['id']}: ROI {list(pista['caja'])}, score máx {pista['score_max']:.2f}, "
                f"visto en {pista['frames_visto']} muestreos{sufijo}"
            )
    else:
        lineas.append("No se detectó ningún identificador activo del YAML.")
    lineas.append("")
    if not informes:
        lineas.append("Ningún objeto sostenido en las manos.")
        return "\n".join(lineas) + "\n"
    for info in informes:
        lineas.append(
            f"Objeto {info['id']}: visible en {info['frames_visible']} frames, "
            f"en mano en {info['frames_en_mano']} "
            f"(izquierda {info['manos']['izquierda']}, derecha {info['manos']['derecha']}, "
            f"ambas {info['manos']['ambas']})"
        )
        for evento in info["eventos"]:
            detalle = ""
            if evento["tipo"] == "cambio_de_mano":
                detalle = f" de {evento['desde']} a {evento['hasta']}"
            lineas.append(
                f"  frame {evento['frame']} ({evento['t']} s): {evento['tipo']}{detalle}"
            )
        lineas.append(f"  último ROI: {info['ultimo_roi']} en el frame {info['ultimo_frame']}")
        if info["contenedor_final"]:
            cont = info["contenedor_final"]
            lineas.append(
                f"  acaba dentro de {cont['id']} ({cont['categoria']}), ROI {cont['roi']}, "
                f"{100 * cont['fraccion_dentro']:.0f} % del objeto dentro"
            )
        else:
            lineas.append("  no acaba dentro de ningún contenedor")
        lineas.append(f"  estado final: {info['estado_final']}")
    lineas.append(
        "\nIzquierda y derecha son las de la persona. «ambas» significa que las dos "
        "manos superan el umbral de agarre en el mismo frame."
    )
    return "\n".join(lineas) + "\n"


# --------------------------------------------------------------------------- #
# Proceso compartido
# --------------------------------------------------------------------------- #


def anadir_argumentos_comunes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--videos", type=Path, default=Path("videos"), help="Carpeta de vídeos.")
    parser.add_argument("--video", type=Path, default=None, help="Un único vídeo. Si se indica, se ignora --videos.")
    parser.add_argument("--salida", type=Path, default=Path("salidas"))
    parser.add_argument("--modelo-pose", type=Path, default=PESOS / "yolo26n-pose.pt")
    parser.add_argument("--modelo-objeto", type=Path, default=PESOS / "yolo26n-seg.pt")
    parser.add_argument("--modelo-contenedores", type=Path, default=PESOS / "yoloe-26m-seg.pt")
    parser.add_argument(
        "--identificadores", type=Path, default=YAML_IDENTIFICADORES,
        help="YAML para activar o desactivar carrito, bolso, mochila, celular, etc.",
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
        "--fps", type=int, default=None,
        help="Frames a procesar por cada segundo de vídeo. Con un vídeo a 15 fps, "
             "--fps 1 toma el primero de cada segundo; --fps 2 los frames 1 y 8; "
             "--fps 3 los frames 1, 6 y 11. Si se omite, se procesa frame a frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None)


def quedarse_con_agarrados(objetos: list[Objeto], args) -> list[Objeto]:
    """Se queda con el objeto principal y, si existe, otro realmente sostenido."""
    def en_mano(objeto: Objeto) -> int:
        return sum(1 for vista in objeto.vistas.values() if vista.mano)

    candidatos = []
    for objeto in objetos:
        n = en_mano(objeto)
        if not objeto.vistas or n < args.min_detecciones:
            continue
        if n / len(objeto.vistas) < 0.5:
            continue
        candidatos.append(objeto)
    if not candidatos:
        return []
    candidatos.sort(key=lambda o: -en_mano(o))
    principal = candidatos[0]
    umbral = max(args.min_detecciones * 3, int(0.55 * en_mano(principal)))
    vivos = [principal]
    for objeto in candidatos[1:]:
        if en_mano(objeto) >= umbral:
            vivos.append(objeto)
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
    muestreo = describir_muestreo(args.fps, fps, nframes_origen, nframes)
    informes = [
        informe_objeto(o, fps, nframes, contenedores, args, indices_origen) for o in objetos
    ]
    offsets, _rate = offsets_muestreo(fps, args.fps)
    fps_salida = float(args.fps) if offsets is not None else fps
    anotado = destino / "anotado.mp4"
    print(f"  Escribiendo {anotado}")
    escritos = anotar(
        frames, anotado, objetos, contenedores, fps_salida, fps, indices_origen,
        destino / "frames" if not args.sin_frames else None, args.alfa,
        contextos, dibujar_esqueleto=not args.sin_pose,
    )
    if not args.sin_frames:
        print(f"  {escritos} frames en {destino / 'frames'}")

    def orig(idx: int) -> int:
        return frame_origen(idx, indices_origen)

    traza = {
        "video": str(video),
        "modelo": modelo,
        "ancho": ancho,
        "alto": alto,
        "frames": nframes,
        "frames_origen": nframes_origen,
        "fps": fps,
        "fps_salida": fps_salida,
        "muestreo": muestreo,
        "parametros": {
            "padding_muneca": args.padding_muneca,
            "crop_min": args.crop_min,
            "factor_palma": args.factor_palma,
            "umbral_agarre": args.umbral_agarre,
            "contacto_min": args.contacto_min,
            "conf_contenedor": args.conf_contenedor,
            "cada_contenedor": args.cada_contenedor,
            "dentro_min": args.dentro_min,
            "factor_contenedor": args.factor_contenedor,
            "identificadores_yaml": str(args.identificadores),
            "fps": args.fps,
        },
        "identificadores_yaml": [
            {
                "clave": item.clave,
                "activo": item.activo,
                "ultralytics_id": item.ultralytics_id,
                "nombre": item.nombre,
                "prompts": item.prompts,
                "destino": item.destino,
                "accesorio": item.accesorio,
            }
            for item in getattr(args, "catalogo", [])
        ],
        "contenedores": [
            {
                "id": p["id"],
                "categoria": p["categoria"],
                "prompt": p["texto"],
                "score_medio": p["score_medio"],
                "score_max": p["score_max"],
                "primer_frame": orig(p["primer_frame"]),
                "ultimo_frame": orig(p["ultimo_frame"]),
                "frames_visto": p["frames_visto"],
                "destino": bool(p.get("destino", False)),
                "accesorio": bool(p.get("accesorio", False)),
                "roi_ultimo": list(p["caja"]),
                "rois": {str(orig(f)): list(c) for f, c in sorted(p["cajas"].items())},
            }
            for p in contenedores
        ],
        "objetos": informes,
    }
    if extra:
        traza.update(extra)
    (destino / "seguimiento.json").write_text(
        json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    texto = resumen(
        video, informes, contenedores, nframes, modelo, args.identificadores, muestreo,
    )
    (destino / "resumen.txt").write_text(texto, encoding="utf-8")
    (destino / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frames_con_objeto = len({i for o in objetos for i in o.vistas})
    desapariciones = sum(
        1 for info in informes for e in info["eventos"] if e["tipo"] == "desaparece"
    )
    ruta = publicar_comparativa(
        args.salida, tecnologia, video.stem, backend, destino, metricas,
        len(objetos), desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def procesar_video(
    detectores,
    video: Path,
    args,
    tecnologia: str,
    backend: str,
    modelo: str,
    seguir,
    extra: dict | None = None,
) -> None:
    ancho, alto, nframes_archivo, fps_archivo = info_video(video)
    frames, fps, indices_origen = leer_frames(video, args.max_frames, args.fps)
    fps = fps_archivo or fps
    nframes_origen = nframes_archivo
    if args.max_frames is not None:
        nframes_origen = min(nframes_origen, args.max_frames)
    muestreo = describir_muestreo(args.fps, fps, nframes_origen, len(frames))
    print(f"\n=== {video.name}: {ancho}x{alto}, {len(frames)} frames procesados, {fps:.3f} fps ===")
    print(f"  {muestreo['descripcion']}")
    destino = carpeta_de(args.salida, tecnologia, video)
    medidor = Medidor(tecnologia)
    try:
        escena, detector, sam, det_contenedores = detectores
        with medidor.etapa("segmentacion"):
            contextos, pistas = detectar_objetos(frames, escena, detector, sam, args)
            objetos = seguir(frames, pistas, contextos, args)
            contenedores = seguir_contenedores(frames, det_contenedores, args.cada_contenedor)
            objetos = descartar_accesorios(objetos, contenedores)
            objetos = quedarse_con_agarrados(objetos, args)
            contenedores = descartar_contenedores_en_mano(contenedores, objetos, args)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        escribir_salidas(
            video, args, destino, objetos, contenedores, contextos, frames, indices_origen,
            medidor.cerrar(len(frames)),
            fps, len(frames), nframes_origen, ancho, alto, tecnologia, backend, modelo, extra,
        )
    except Exception as exc:
        publicar_comparativa(
            args.salida, tecnologia, video.stem, backend, destino,
            medidor.cerrar(len(frames)), 0, 0, 0, ok=False, codigo=1, error=str(exc),
        )
        raise

