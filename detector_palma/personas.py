"""Deteccion de personas y localizacion de manos.

Es el modulo que da el contexto comun a todos los backends: ROI de la persona
con padding y puntos de mano sobre los que buscar el objeto sostenido.

Estrategias disponibles:
  - 'pose'      : modelo YOLO-pose. Da persona + munecas en una sola pasada.
  - 'mediapipe' : MediaPipe Hands. Muy ligero en CPU y funciona en video egocentrico.
  - 'ninguno'   : sin filtro de mano, se rastrea todo el frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometria import Caja, expandir_caja
from .utils import importar, importar_opcional

AYUDA_ULTRALYTICS = "pip install ultralytics"

# Indices COCO-17 de las articulaciones que nos interesan.
MUNECA_IZQ, MUNECA_DER = 9, 10
CODO_IZQ, CODO_DER = 7, 8
HOMBRO_IZQ, HOMBRO_DER = 5, 6


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
    ):
        self.padding_persona = padding_persona
        self.radio_mano = radio_mano
        self.conf = conf
        self.imgsz = imgsz
        self.dispositivo = dispositivo
        self._pose = None
        self._mediapipe = None

        if estrategia == "auto":
            estrategia = "pose"
        self.estrategia = estrategia

        if estrategia == "pose":
            YOLO = importar("ultralytics", AYUDA_ULTRALYTICS).YOLO
            self._pose = YOLO(modelo_pose)
        elif estrategia == "mediapipe":
            self._inicializar_mediapipe()
        elif estrategia != "ninguno":
            raise ValueError(f"Estrategia de manos desconocida: {estrategia}")

    def _inicializar_mediapipe(self) -> None:
        mp = importar_opcional("mediapipe")
        if mp is None:
            raise RuntimeError(
                "MediaPipe no esta instalado o no soporta esta version de Python.\n"
                "  Solucion: pip install mediapipe (requiere Python 3.9-3.12)\n"
                "  Alternativa: usa --detector-manos pose"
            )
        self._mediapipe = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analizar(self, frame: np.ndarray) -> ContextoEscena:
        alto, ancho = frame.shape[:2]
        contexto = ContextoEscena(ancho=ancho, alto=alto)
        if self.estrategia == "pose":
            self._analizar_pose(frame, contexto)
        elif self.estrategia == "mediapipe":
            self._analizar_mediapipe(frame, contexto)
        return contexto

    # -- Estrategia YOLO-pose -------------------------------------------------

    def _analizar_pose(self, frame: np.ndarray, contexto: ContextoEscena) -> None:
        resultados = self._pose.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.dispositivo, verbose=False
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
            contexto.personas.append(persona)
            if persona.keypoints is not None:
                radio = max(20.0, self.radio_mano * persona.altura)
                contexto.manos.extend(self._manos_desde_keypoints(persona.keypoints, radio, i))

    @staticmethod
    def _manos_desde_keypoints(kp: np.ndarray, radio: float, persona_id: int) -> list[Mano]:
        """Estima palma y un abanico de puntos de agarre a partir de codo y muneca.

        YOLO-pose no ve los dedos: se proyecta el antebrazo mas alla de la muneca
        (zona de palma) y se generan 5 puntos en abanico que simulan la pinza.
        El lado izquierda/derecha es el de la persona, no el del encuadre.
        """
        manos: list[Mano] = []
        for muneca, codo, lado in (
            (MUNECA_IZQ, CODO_IZQ, "izquierda"),
            (MUNECA_DER, CODO_DER, "derecha"),
        ):
            if muneca >= len(kp):
                continue
            wx, wy, wc = kp[muneca]
            if wc < 0.3:
                continue
            muneca_xy = (float(wx), float(wy))
            dx, dy = 0.0, 1.0
            if codo < len(kp) and kp[codo][2] >= 0.3:
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
                )
            )
        return manos

    # -- Estrategia MediaPipe -------------------------------------------------

    def _analizar_mediapipe(self, frame: np.ndarray, contexto: ContextoEscena) -> None:
        import cv2

        resultado = self._mediapipe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not resultado.multi_hand_landmarks:
            return
        diagonal = float(np.hypot(contexto.ancho, contexto.alto))
        radio = max(30.0, self.radio_mano * diagonal * 0.5)
        lateralidad = resultado.multi_handedness or []

        for i, landmarks in enumerate(resultado.multi_hand_landmarks):
            puntos = np.array(
                [[lm.x * contexto.ancho, lm.y * contexto.alto] for lm in landmarks.landmark]
            )
            centro = puntos.mean(axis=0)
            etiqueta = "desconocida"
            confianza = 1.0
            if i < len(lateralidad) and lateralidad[i].classification:
                clasificacion = lateralidad[i].classification[0]
                etiqueta = "izquierda" if clasificacion.label.lower().startswith("l") else "derecha"
                confianza = float(clasificacion.score)
            # Landmarks 4, 8, 12, 16, 20 = yemas; 0 = muneca; 9 = base del medio.
            yemas = []
            for idx_lm in (4, 8, 12, 16, 20, 9):
                if idx_lm < len(puntos):
                    yemas.append((float(puntos[idx_lm][0]), float(puntos[idx_lm][1])))
            palma = (
                (float(puntos[9][0]), float(puntos[9][1]))
                if len(puntos) > 9
                else (float(centro[0]), float(centro[1]))
            )
            contexto.manos.append(
                Mano(
                    punto=(float(centro[0]), float(centro[1])),
                    lado=etiqueta,
                    confianza=confianza,
                    persona_id=i,
                    radio=radio,
                    palma=palma,
                    radio_palma=max(12.0, radio * 0.35),
                    puntos_agarre=yemas,
                )
            )
            # En video egocentrico no hay cuerpo visible: la propia mano hace de ROI.
            x1, y1 = puntos.min(axis=0)
            x2, y2 = puntos.max(axis=0)
            caja = (int(x1), int(y1), int(x2), int(y2))
            contexto.personas.append(
                Persona(
                    caja=caja,
                    caja_padded=expandir_caja(
                        caja, max(1.5, self.padding_persona * 6), contexto.ancho, contexto.alto
                    ),
                    confianza=confianza,
                )
            )

    def cerrar(self) -> None:
        if self._mediapipe is not None:
            self._mediapipe.close()
            self._mediapipe = None
