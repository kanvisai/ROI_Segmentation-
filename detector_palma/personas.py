"""Deteccion de personas y localizacion de manos.

Es el modulo que da el contexto comun a todos los backends: ROI de la persona
con padding y puntos de mano sobre los que buscar el objeto sostenido.

Estrategias disponibles:
  - 'pose'      : modelo YOLO-pose. Da persona + munecas en una sola pasada.
  - 'mediapipe' : MediaPipe Pose (esqueleto) + Hands (dedos / visibilidad).
  - 'ninguno'   : sin filtro de mano, se rastrea todo el frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathlib import Path

from .geometria import Caja, expandir_caja
from .utils import importar, importar_opcional

import atexit

AYUDA_ULTRALYTICS = "pip install ultralytics"
PESOS_RAIZ = Path(__file__).resolve().parent.parent / "pesos"
HOLISTIC_NOMBRE = "holistic_landmarker.task"
HOLISTIC_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
LADO_HOLISTIC = 512


def _asegurar_holistic() -> Path:
    """Downloads the Holistic landmarker into pesos/ on first use."""
    destino = PESOS_RAIZ / HOLISTIC_NOMBRE
    if destino.is_file() and destino.stat().st_size > 1_000_000:
        return destino
    PESOS_RAIZ.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading MediaPipe Holistic → {destino}")
    try:
        import urllib.request

        urllib.request.urlretrieve(HOLISTIC_URL, destino)
    except Exception as exc:
        if destino.is_file():
            destino.unlink()
        raise RuntimeError(
            f"Could not download {HOLISTIC_URL}. "
            f"Place {HOLISTIC_NOMBRE} in {PESOS_RAIZ}."
        ) from exc
    if not destino.is_file() or destino.stat().st_size < 1_000_000:
        raise RuntimeError(f"MediaPipe Holistic download looks empty: {destino}")
    return destino


def _marco_cuadrado(rgb: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Letterbox to a square so MediaPipe landmark projection matches the frame."""
    alto, ancho = rgb.shape[:2]
    lado = max(alto, ancho)
    lienzo = np.zeros((lado, lado, 3), dtype=rgb.dtype)
    x0 = (lado - ancho) // 2
    y0 = (lado - alto) // 2
    lienzo[y0 : y0 + alto, x0 : x0 + ancho] = rgb
    return lienzo, x0, y0, lado


def _lm_xy(lm, x0: int, y0: int, lado: int, ancho: int, alto: int) -> tuple[float, float]:
    del ancho, alto
    x = float(lm.x) * lado - x0
    y = float(lm.y) * lado - y0
    return x, y


def _lm_xy_recorte(
    lm, x1: int, y1: int, sx: int, sy: int, lado: int
) -> tuple[float, float]:
    """Landmark normalizado del recorte letterbox → píxeles del frame completo."""
    x = float(lm.x) * lado - sx + x1
    y = float(lm.y) * lado - sy + y1
    return x, y


def _lm_score(lm) -> float:
    vis = float(getattr(lm, "visibility", 0.0) or 0.0)
    pres = float(getattr(lm, "presence", 0.0) or 0.0)
    score = max(vis, pres)
    return score if score > 0.0 else 1.0


def _lista_landmarks(valor) -> list:
    """HolisticLandmarker puede devolver List[NormalizedLandmark] o List[List[...]]."""
    if valor is None:
        return []
    if hasattr(valor, "x") and hasattr(valor, "y"):
        return [valor]
    try:
        seq = list(valor)
    except TypeError:
        return []
    if not seq:
        return []
    first = seq[0]
    if hasattr(first, "x") and hasattr(first, "y"):
        return seq
    return _lista_landmarks(first)


def _tiene_cabeza(persona: Persona, umbral: float) -> bool:
    """Descarta falsos positivos tipo cesto: YOLO-pose sin nariz ni hombros."""
    kp = persona.keypoints
    if kp is None or len(kp) < 7:
        return False
    umbral_cabeza = min(float(umbral), 0.15)
    nariz = float(kp[0][2])
    hombro = max(float(kp[5][2]), float(kp[6][2]))
    return nariz >= umbral_cabeza or hombro >= umbral_cabeza


def _esqueleto_en_caja(kp: np.ndarray, caja: Caja, umbral: float = 0.15) -> bool:
    """True si el centroide de los keypoints visibles cae dentro (con margen) de la caja."""
    x1, y1, x2, y2 = caja
    xs, ys = [], []
    for fila in kp:
        if float(fila[2]) < umbral:
            continue
        xs.append(float(fila[0]))
        ys.append(float(fila[1]))
    if not xs:
        return False
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    margen = 0.15 * max(x2 - x1, y2 - y1, 1)
    return (x1 - margen) <= cx <= (x2 + margen) and (y1 - margen) <= cy <= (y2 + margen)

# Indices COCO-17 de las articulaciones que nos interesan.
MUNECA_IZQ, MUNECA_DER = 9, 10
CODO_IZQ, CODO_DER = 7, 8
HOMBRO_IZQ, HOMBRO_DER = 5, 6
CADERA_IZQ, CADERA_DER = 11, 12

# Esqueleto de la cintura hacia arriba (COCO-17). Las piernas (13-16) no se dibujan.
POSE_NOMBRES = {
    0: "nariz",
    5: "hombro IZQ",
    6: "hombro DER",
    7: "codo IZQ",
    8: "codo DER",
    9: "muñeca IZQ",
    10: "muñeca DER",
    11: "cadera IZQ",
    12: "cadera DER",
}
POSE_HUESOS = (
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (0, 5),
    (0, 6),
)
COLOR_POSE_IZQ = (40, 140, 255)
COLOR_POSE_DER = (255, 150, 30)
COLOR_POSE_TORSO = (0, 220, 180)
COLOR_PALMA = (80, 255, 80)
COLOR_MP_MANO = (80, 255, 160)

# MediaPipe Pose → COCO-17 (cintura hacia arriba) para dibujar el mismo esqueleto.
MP_POSE_A_COCO = {
    0: 0,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
}
MP_MUNECA_IZQ, MP_MUNECA_DER = 15, 16
MP_CODO_IZQ, MP_CODO_DER = 13, 14
MP_HAND_CONEXIONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class Mano:
    punto: tuple[float, float]     # muneca (o centro MediaPipe)
    lado: str                      # "izquierda" | "derecha" | "desconocida" | "ambas"
    confianza: float = 1.0
    persona_id: int = -1
    radio: float = 60.0            # radio de busqueda en pixeles
    palma: tuple[float, float] | None = None
    radio_palma: float = 24.0
    direccion: tuple[float, float] = (0.0, 1.0)
    puntos_agarre: list[tuple[float, float]] = field(default_factory=list)
    conexiones: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    origen: str = "pose"

    def __post_init__(self) -> None:
        if self.palma is None:
            self.palma = self.punto


@dataclass
class Persona:
    caja: Caja                     # bounding box original
    caja_padded: Caja              # bounding box con el margen aplicado
    confianza: float = 1.0
    keypoints: np.ndarray | None = None

    @property
    def altura(self) -> float:
        return float(self.caja[3] - self.caja[1])


@dataclass
class ContextoEscena:
    """Lo que sabemos de la escena en un frame concreto."""
    personas: list[Persona] = field(default_factory=list)
    manos: list[Mano] = field(default_factory=list)
    ancho: int = 0
    alto: int = 0

    conf_dibujo: float = 0.20

    @property
    def hay_filtro_mano(self) -> bool:
        return bool(self.manos)

    def region_valida(self) -> Caja:
        """Union de los ROI de persona; si no hay personas, el frame completo."""
        if not self.personas:
            return (0, 0, self.ancho, self.alto)
        x1 = min(p.caja_padded[0] for p in self.personas)
        y1 = min(p.caja_padded[1] for p in self.personas)
        x2 = max(p.caja_padded[2] for p in self.personas)
        y2 = max(p.caja_padded[3] for p in self.personas)
        return (x1, y1, x2, y2)


def _xy_keypoint(kp: np.ndarray, indice: int, umbral: float):
    """Devuelve (x, y) entero si el punto existe y supera el umbral de confianza."""
    if indice >= len(kp):
        return None
    fila = kp[indice]
    if fila.shape[0] < 2:
        return None
    x, y = float(fila[0]), float(fila[1])
    conf = float(fila[2]) if fila.shape[0] >= 3 else 1.0
    if conf < umbral:
        return None
    return int(round(x)), int(round(y))


def _color_hueso(a: int, b: int):
    izq = {HOMBRO_IZQ, CODO_IZQ, MUNECA_IZQ, CADERA_IZQ}
    der = {HOMBRO_DER, CODO_DER, MUNECA_DER, CADERA_DER}
    if a in izq or b in izq:
        return COLOR_POSE_IZQ
    if a in der or b in der:
        return COLOR_POSE_DER
    return COLOR_POSE_TORSO


def dibujar_pose(frame: np.ndarray, contexto: ContextoEscena, umbral: float | None = None) -> None:
    """Pinta el esqueleto de la cintura hacia arriba, con brazos, muñecas y palmas."""
    import cv2

    if umbral is None:
        umbral = float(getattr(contexto, "conf_dibujo", 0.20) or 0.20)
    for persona in contexto.personas:
        kp = persona.keypoints
        if kp is None or len(kp) == 0:
            continue
        for a, b in POSE_HUESOS:
            pa, pb = _xy_keypoint(kp, a, umbral), _xy_keypoint(kp, b, umbral)
            if pa is None or pb is None:
                continue
            cv2.line(frame, pa, pb, _color_hueso(a, b), 2, cv2.LINE_AA)
        for indice in POSE_NOMBRES:
            punto = _xy_keypoint(kp, indice, umbral)
            if punto is None:
                continue
            cv2.circle(frame, punto, 5, _color_hueso(indice, indice), -1, cv2.LINE_AA)
    for mano in contexto.manos:
        palma = mano.palma or mano.punto
        px, py = int(round(palma[0])), int(round(palma[1]))
        wx, wy = int(round(mano.punto[0])), int(round(mano.punto[1]))
        color = COLOR_MP_MANO if mano.origen == "mediapipe" else COLOR_PALMA
        if mano.conexiones:
            for a, b in mano.conexiones:
                pa = (int(round(a[0])), int(round(a[1])))
                pb = (int(round(b[0])), int(round(b[1])))
                cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
            for a, _b in mano.conexiones:
                cv2.circle(frame, (int(round(a[0])), int(round(a[1]))), 3, color, -1, cv2.LINE_AA)
        else:
            cv2.line(frame, (wx, wy), (px, py), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, color, -1, cv2.LINE_AA)


class AnalizadorEscena:
    """Produce el ContextoEscena de cada frame con la estrategia elegida."""

    def __init__(
        self,
        estrategia: str = "auto",
        modelo_pose: str = "yolo26n-pose.pt",
        dispositivo: str = "cpu",
        padding_persona: float = 0.25,
        radio_mano: float = 0.22,
        conf: float = 0.25,
        imgsz: int = 640,
        conf_mano: float = 0.20,
    ):
        self.padding_persona = padding_persona
        self.radio_mano = radio_mano
        self.conf = conf
        self.imgsz = imgsz
        self.dispositivo = dispositivo
        self.conf_mano = float(conf_mano)
        self._pose = None
        self._holistic = None
        self._atexit_mp = False

        if estrategia == "auto":
            estrategia = "pose"
        self.estrategia = estrategia

        if estrategia == "pose":
            YOLO = importar("ultralytics", AYUDA_ULTRALYTICS).YOLO
            self._pose = YOLO(modelo_pose)
        elif estrategia == "mediapipe":
            self._inicializar_mediapipe()
            # Holistic falla a menudo a distancia; YOLO-pose cubre muñecas y el traspaso.
            YOLO = importar("ultralytics", AYUDA_ULTRALYTICS).YOLO
            self._pose = YOLO(modelo_pose)
        elif estrategia != "ninguno":
            raise ValueError(f"Estrategia de manos desconocida: {estrategia}")

    def _inicializar_mediapipe(self) -> None:
        mp = importar_opcional("mediapipe")
        if mp is None:
            raise RuntimeError(
                "MediaPipe is not installed.\n"
                "  Install: uv pip install mediapipe\n"
                "  Or omit --mediapipe to keep YOLO-pose."
            )
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        umbral = max(0.05, min(0.95, self.conf_mano))
        ruta = _asegurar_holistic()
        opciones = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ruta)),
            running_mode=VisionTaskRunningMode.IMAGE,
            min_pose_detection_confidence=umbral,
            min_pose_landmarks_confidence=umbral,
            min_hand_landmarks_confidence=umbral,
        )
        print(f"  MediaPipe Holistic ({ruta.name}), min_confidence={umbral:.2f}")
        self._holistic = HolisticLandmarker.create_from_options(opciones)
        if not self._atexit_mp:
            atexit.register(self.cerrar)
            self._atexit_mp = True

    def analizar(self, frame: np.ndarray) -> ContextoEscena:
        alto, ancho = frame.shape[:2]
        contexto = ContextoEscena(ancho=ancho, alto=alto, conf_dibujo=self.conf_mano)
        if self.estrategia == "pose":
            self._analizar_pose(frame, contexto)
        elif self.estrategia == "mediapipe":
            self._analizar_mediapipe(frame, contexto)
        return contexto

    # -- Estrategia YOLO-pose -------------------------------------------------

    def _analizar_pose(self, frame: np.ndarray, contexto: ContextoEscena) -> None:
        imgsz = max(self.imgsz, 960) if self.estrategia == "mediapipe" else self.imgsz
        resultados = self._pose.predict(
            frame, conf=self.conf, imgsz=imgsz, device=self.dispositivo, verbose=False
        )
        if not resultados:
            return
        resultado = resultados[0]
        cajas = resultado.boxes
        if cajas is None or len(cajas) == 0:
            return

        kps = None
        if resultado.keypoints is not None and resultado.keypoints.data is not None:
            kps = resultado.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for i in range(len(cajas)):
            x1, y1, x2, y2 = cajas.xyxy[i].cpu().numpy().tolist()
            caja = (int(x1), int(y1), int(x2), int(y2))
            persona = Persona(
                caja=caja,
                caja_padded=expandir_caja(
                    caja, self.padding_persona, contexto.ancho, contexto.alto
                ),
                confianza=float(cajas.conf[i]),
                keypoints=kps[i] if kps is not None and i < len(kps) else None,
            )
            if not _tiene_cabeza(persona, self.conf_mano):
                continue
            contexto.personas.append(persona)
            if persona.keypoints is not None:
                radio = max(20.0, self.radio_mano * persona.altura)
                contexto.manos.extend(
                    self._manos_desde_keypoints(
                        persona.keypoints, radio, len(contexto.personas) - 1,
                        self.conf_mano, origen="pose",
                    )
                )

    @staticmethod
    def _manos_desde_keypoints(
        kp: np.ndarray,
        radio: float,
        persona_id: int,
        umbral: float = 0.20,
        origen: str = "pose",
        muneca_izq: int = MUNECA_IZQ,
        muneca_der: int = MUNECA_DER,
        codo_izq: int = CODO_IZQ,
        codo_der: int = CODO_DER,
    ) -> list[Mano]:
        """Estima palma y un abanico de puntos de agarre a partir de codo y muneca.

        YOLO-pose no ve los dedos: se proyecta el antebrazo mas alla de la muneca
        (zona de palma) y se generan 5 puntos en abanico que simulan la pinza.
        El lado izquierda/derecha es el de la persona, no el del encuadre.
        """
        manos: list[Mano] = []
        for muneca, codo, lado in (
            (muneca_izq, codo_izq, "izquierda"),
            (muneca_der, codo_der, "derecha"),
        ):
            if muneca >= len(kp):
                continue
            wx, wy, wc = kp[muneca][0], kp[muneca][1], kp[muneca][2]
            if wc < umbral:
                continue
            muneca_xy = (float(wx), float(wy))
            dx, dy = 0.0, 1.0
            if codo < len(kp) and float(kp[codo][2]) >= umbral:
                ex, ey = float(kp[codo][0]), float(kp[codo][1])
                vx, vy = muneca_xy[0] - ex, muneca_xy[1] - ey
                norma = float(np.hypot(vx, vy))
                if norma > 1.0:
                    dx, dy = vx / norma, vy / norma
            palma = (muneca_xy[0] + 0.28 * radio * dx, muneca_xy[1] + 0.28 * radio * dy)
            radio_palma = max(10.0, radio * 0.38)
            perp_x, perp_y = -dy, dx
            puntos_agarre = [palma]
            for k in (-0.55, -0.28, 0.0, 0.28, 0.55):
                puntos_agarre.append(
                    (
                        palma[0] + 0.55 * radio * dx + k * radio_palma * perp_x,
                        palma[1] + 0.55 * radio * dy + k * radio_palma * perp_y,
                    )
                )
            manos.append(
                Mano(
                    punto=muneca_xy,
                    lado=lado,
                    confianza=float(wc),
                    persona_id=persona_id,
                    radio=radio,
                    palma=palma,
                    radio_palma=radio_palma,
                    direccion=(dx, dy),
                    puntos_agarre=puntos_agarre,
                    origen=origen,
                )
            )
        return manos

    # -- Estrategia MediaPipe -------------------------------------------------

    def _analizar_mediapipe(self, frame: np.ndarray, contexto: ContextoEscena) -> None:
        """YOLO localiza a la persona; Holistic solo corre en su recorte cuadrado.

        En un supermercado Holistic a frame completo se ancla al cesto del primer
        plano y el esqueleto queda desfasado respecto a la persona.
        """
        self._analizar_pose(frame, contexto)
        if self._holistic is None or not contexto.personas:
            return
        for pid, persona in enumerate(contexto.personas):
            self._holistic_en_persona(frame, persona, contexto, pid)

    def _holistic_en_persona(
        self, frame: np.ndarray, persona: Persona, contexto: ContextoEscena, pid: int
    ) -> None:
        import cv2
        import mediapipe as mp

        caja_h = expandir_caja(
            persona.caja, max(0.65, self.padding_persona * 2.0),
            contexto.ancho, contexto.alto,
        )
        x1, y1, x2, y2 = caja_h
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(contexto.ancho, int(x2)), min(contexto.alto, int(y2))
        if x2 - x1 < 40 or y2 - y1 < 40:
            return
        recorte = np.ascontiguousarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        lienzo, sx, sy, lado = _marco_cuadrado(recorte)
        if lado != LADO_HOLISTIC:
            lienzo = cv2.resize(
                lienzo, (LADO_HOLISTIC, LADO_HOLISTIC), interpolation=cv2.INTER_LINEAR
            )
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=lienzo)
        try:
            resultado = self._holistic.detect(imagen)
        except Exception:
            return
        kp = self._keypoints_holistic(resultado, contexto, x1, y1, sx, sy, lado)
        if kp is None or not _esqueleto_en_caja(kp, caja_h):
            return
        persona.keypoints = kp
        munecas = {
            "izquierda": float(kp[9][2]),
            "derecha": float(kp[10][2]),
        }
        radio = max(20.0, self.radio_mano * persona.altura)
        contexto.manos = [m for m in contexto.manos if m.persona_id != pid]
        pose_manos = self._manos_desde_keypoints(
            kp, radio, pid, self.conf_mano, origen="mediapipe"
        )
        mp_manos = self._manos_holistic_recorte(
            resultado, contexto, munecas, x1, y1, sx, sy, lado, pid
        )
        por_lado = {m.lado: m for m in pose_manos}
        por_lado.update({m.lado: m for m in mp_manos})
        contexto.manos.extend(por_lado.values())

    def _keypoints_holistic(
        self, resultado, contexto: ContextoEscena, x1: int, y1: int,
        sx: int, sy: int, lado: int,
    ) -> np.ndarray | None:
        lms = _lista_landmarks(getattr(resultado, "pose_landmarks", None))
        if not lms:
            return None
        kp = np.zeros((17, 3), dtype=np.float32)
        utiles = 0
        for mp_i, coco_i in MP_POSE_A_COCO.items():
            if mp_i >= len(lms):
                continue
            lm = lms[mp_i]
            x, y = _lm_xy_recorte(lm, x1, y1, sx, sy, lado)
            score = _lm_score(lm)
            kp[coco_i] = (x, y, score)
            if score >= self.conf_mano:
                utiles += 1
        if utiles < 3:
            return None
        return kp

    def _manos_holistic_recorte(
        self, resultado, contexto, munecas_pose, x1, y1, sx, sy, lado, pid: int
    ) -> list[Mano]:
        manos: list[Mano] = []
        for nombre, attr in (
            ("izquierda", "left_hand_landmarks"),
            ("derecha", "right_hand_landmarks"),
        ):
            lms = _lista_landmarks(getattr(resultado, attr, None))
            mano = self._mano_desde_landmarks_recorte(
                lms, nombre, contexto, munecas_pose.get(nombre, 0.0),
                x1, y1, sx, sy, lado, pid,
            )
            if mano is not None:
                manos.append(mano)
        return manos

    def _mano_desde_landmarks_recorte(
        self, lms, lado: str, contexto: ContextoEscena, confianza: float,
        x1: int, y1: int, sx: int, sy: int, lado_img: int, pid: int = -1,
    ) -> Mano | None:
        if not lms:
            return None
        puntos = np.array(
            [_lm_xy_recorte(lm, x1, y1, sx, sy, lado_img) for lm in lms]
        )
        if puntos.size == 0:
            return None
        scores = [_lm_score(lm) for lm in lms]
        if max(scores) < self.conf_mano and confianza < self.conf_mano:
            return None
        span = float(np.max(puntos.max(axis=0) - puntos.min(axis=0)))
        radio = max(24.0, span * 1.15)
        yemas = [
            (float(puntos[i][0]), float(puntos[i][1]))
            for i in (4, 8, 12, 16, 20, 9) if i < len(puntos)
        ]
        palma = (
            (float(puntos[9][0]), float(puntos[9][1]))
            if len(puntos) > 9
            else (float(puntos.mean(axis=0)[0]), float(puntos.mean(axis=0)[1]))
        )
        muneca = (float(puntos[0][0]), float(puntos[0][1])) if len(puntos) else palma
        conexiones = [
            (
                (float(puntos[a][0]), float(puntos[a][1])),
                (float(puntos[b][0]), float(puntos[b][1])),
            )
            for a, b in MP_HAND_CONEXIONES
            if a < len(puntos) and b < len(puntos)
        ]
        return Mano(
            punto=muneca,
            lado=lado,
            confianza=max(confianza, max(scores) if scores else 0.0),
            persona_id=pid,
            radio=radio,
            palma=palma,
            radio_palma=max(12.0, radio * 0.35),
            puntos_agarre=yemas,
            conexiones=conexiones,
            origen="mediapipe",
        )

    def cerrar(self) -> None:
        h = self._holistic
        if h is None:
            return
        self._holistic = None
        try:
            h.close()
        except Exception:
            pass
        try:
            h.close = lambda *a, **k: None
        except Exception:
            pass

