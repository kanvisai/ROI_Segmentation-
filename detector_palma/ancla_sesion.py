"""Ancla de sesión: una sola persona en frames con varios sujetos.

Sin poses C1, se siembra el cliente de la sesión (alcance a la balda + tamaño)
y se le sigue por IoU / muñecas / histograma. Si se pierde, no se salta a un
transeúnte: ese frame queda sin persona para SAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometria import iou_cajas

# COCO-pose (mismos índices que detector_palma.personas).
HOMBRO_IZQ, HOMBRO_DER = 5, 6
MUNECA_IZQ, MUNECA_DER = 9, 10
CADERA_IZQ, CADERA_DER = 11, 12

WRIST_MATCH_MAX_PX = 140.0
YOLO_WRIST_CONF = 0.20
IOU_LOCK = 0.28
IOU_RECOVER = 0.12
MAX_CENTRO_FRAC = 0.55
MAX_PERDIDOS = 12
HIST_MIN = 0.22
HIST_FUERTE = 0.40
ANCHO_RIVAL_FRAC = 1.55


def _centro(caja) -> tuple[float, float]:
    x1, y1, x2, y2 = caja
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _munecas_px(persona, umbral: float = YOLO_WRIST_CONF) -> list[tuple[float, float]]:
    kp = getattr(persona, "keypoints", None)
    if kp is None or len(kp) <= MUNECA_DER:
        return []
    pts = []
    for indice in (MUNECA_IZQ, MUNECA_DER):
        fila = kp[indice]
        if float(fila[2]) >= umbral:
            pts.append((float(fila[0]), float(fila[1])))
    return pts


def _punto_kp(kp, indice: int, umbral: float):
    if kp is None or indice >= len(kp):
        return None
    fila = kp[indice]
    if float(fila[2]) < umbral:
        return None
    return float(fila[0]), float(fila[1])


def _centro_torso(persona) -> tuple[float, float]:
    kp = getattr(persona, "keypoints", None)
    pts = []
    for indice in (HOMBRO_IZQ, HOMBRO_DER, CADERA_IZQ, CADERA_DER):
        p = _punto_kp(kp, indice, 0.15)
        if p is not None:
            pts.append(p)
    if pts:
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return _centro(persona.caja)


def _histograma_torso(frame, caja) -> np.ndarray | None:
    if frame is None or caja is None:
        return None
    import cv2

    x1, y1, x2, y2 = (int(v) for v in caja)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    dy = int(0.12 * (y2 - y1))
    recorte = frame[y1 + dy : y2, x1:x2]
    if recorte.size == 0:
        return None
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _corr_hist(a, b) -> float:
    if a is None or b is None:
        return 0.0
    import cv2

    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def _orden_izquierda_derecha(personas) -> list[int]:
    return sorted(range(len(personas)), key=lambda i: _centro(personas[i].caja)[0])


def dibujar_candidatos_ancla(frame, personas, indice_elegido: int):
    """Copia del frame con un número 0,1,2… de izquierda a derecha sobre cada caja."""
    import cv2

    vis = frame.copy()
    orden = _orden_izquierda_derecha(personas)
    for rank, i in enumerate(orden):
        x1, y1, x2, y2 = (int(v) for v in personas[i].caja)
        elegido = i == indice_elegido
        color = (0, 255, 255) if elegido else (255, 160, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if elegido else 2)
        etiqueta = f"{rank}" + (" *" if elegido else "")
        escala = 1.6 if len(orden) <= 4 else 1.2
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, 3)
        tx = max(0, x1)
        ty = max(th + 10, y1 - 6)
        cv2.rectangle(vis, (tx, ty - th - 10), (tx + tw + 14, ty + 8), color, -1)
        cv2.putText(
            vis, etiqueta, (tx + 6, ty),
            cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA,
        )
    leyenda = "0 = izquierda   --ancla-persona N   * = elegido"
    cv2.putText(vis, leyenda, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, leyenda, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def lista_candidatos_ancla(personas, indice_elegido: int | None) -> list[dict]:
    orden = _orden_izquierda_derecha(personas)
    filas = []
    for rank, i in enumerate(orden):
        munecas = [(round(x, 1), round(y, 1)) for x, y in _munecas_px(personas[i])]
        filas.append({
            "index": rank,
            "box": [int(v) for v in personas[i].caja],
            "wrists": munecas,
            "chosen": indice_elegido is not None and i == indice_elegido,
        })
    return filas


def score_munecas_sesion(persona, session_pts, max_dist_px: float = WRIST_MATCH_MAX_PX) -> float:
    """C1: suma -d si alguna muñeca YOLO cae a ≤ max_dist_px de una muñeca de sesión."""
    if not session_pts:
        return float("-inf")
    yolo = _munecas_px(persona)
    if not yolo:
        return float("-inf")
    score = 0.0
    hay = False
    for sx, sy in session_pts:
        d = min(float(np.hypot(wx - float(sx), wy - float(sy))) for wx, wy in yolo)
        if d <= max_dist_px:
            score += -d
            hay = True
    return score if hay else float("-inf")


def elegir_indice_semilla(
    personas, ancho: int, alto: int, criterio: str = "auto",
    indice: int | None = None, punto=None, munecas=None,
    max_dist_px: float = WRIST_MATCH_MAX_PX,
) -> int | None:
    """Elige el sujeto en el primer frame con gente. Índices 0,1,… van de izquierda a derecha."""
    n = len(personas)
    if n == 0:
        return None
    clave = (criterio or "auto").lower()
    session = [tuple(map(float, p)) for p in (munecas or []) if p is not None]
    if clave in ("munecas", "wrists") and session:
        scores = [score_munecas_sesion(p, session, max_dist_px) for p in personas]
        mejor = int(max(range(n), key=lambda i: scores[i]))
        if scores[mejor] == float("-inf"):
            return None
        return mejor
    if n <= 1:
        return 0
    orden = _orden_izquierda_derecha(personas)
    if clave in ("indice", "index") and indice is not None:
        if indice < 0 or indice >= n:
            raise SystemExit(
                f"--ancla-persona {indice} está fuera de rango: hay {n} personas "
                f"(0 = más a la izquierda, {n - 1} = más a la derecha)."
            )
        return orden[indice]
    if clave in ("izquierda", "left"):
        return orden[0]
    if clave in ("derecha", "right"):
        return orden[-1]
    if clave in ("centro", "center"):
        cx, cy = 0.5 * float(ancho), 0.5 * float(alto)
        return int(min(
            range(n),
            key=lambda i: float(np.hypot(_centro(personas[i].caja)[0] - cx,
                                         _centro(personas[i].caja)[1] - cy)),
        ))
    if clave in ("altura", "tallest"):
        return int(max(range(n), key=lambda i: personas[i].altura))
    if clave in ("punto", "point") and punto is not None:
        x, y = float(punto[0]), float(punto[1])
        for i, persona in enumerate(personas):
            x1, y1, x2, y2 = persona.caja
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return int(min(
            range(n),
            key=lambda i: float(np.hypot(_centro(personas[i].caja)[0] - x,
                                         _centro(personas[i].caja)[1] - y)),
        ))
    return int(max(range(n), key=lambda i: score_semilla(personas[i], ancho, alto)))


def score_semilla(persona, ancho: int, alto: int) -> float:
    """Cliente de la sesión: alcanza la balda y está cerca de la cámara."""
    x1, y1, x2, y2 = persona.caja
    area = max(1.0, float((x2 - x1) * (y2 - y1))) / float(max(1, ancho * alto))
    altura = float(max(persona.altura, 1.0)) / float(max(alto, 1))
    munecas = _munecas_px(persona)
    torso = _centro_torso(persona)
    h = max(float(persona.altura), 1.0)
    alcance = 0.0
    for wx, wy in munecas:
        alcance = max(alcance, float(np.hypot(wx - torso[0], wy - torso[1])) / h)
    conf_m = 0.0
    kp = getattr(persona, "keypoints", None)
    if kp is not None and len(kp) > MUNECA_DER:
        conf_m = float(kp[MUNECA_IZQ][2]) + float(kp[MUNECA_DER][2])
    return (
        2.6 * alcance
        + 1.5 * area
        + 0.7 * altura
        + 0.25 * conf_m
        + 0.2 * float(getattr(persona, "confianza", 1.0))
    )


def score_seguimiento(
    persona,
    caja_prev,
    munecas_prev: list[tuple[float, float]],
    hist_prev,
    frame,
    max_dist_px: float,
) -> tuple[float, float]:
    iou = float(iou_cajas(persona.caja, caja_prev))
    c0 = _centro(caja_prev)
    c1 = _centro(persona.caja)
    alto_prev = max(float(caja_prev[3] - caja_prev[1]), 1.0)
    dc = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / alto_prev
    dw = float("inf")
    for sw in munecas_prev:
        for pw in _munecas_px(persona):
            dw = min(dw, float(np.hypot(pw[0] - sw[0], pw[1] - sw[1])))
    hist = _histograma_torso(frame, persona.caja)
    hs = _corr_hist(hist_prev, hist)
    score = 2.6 * iou + 2.4 * max(0.0, hs)
    if dw < max_dist_px:
        score += 1.6 * (1.0 - dw / max_dist_px)
    if dc < MAX_CENTRO_FRAC:
        score += 0.6 * (1.0 - dc / MAX_CENTRO_FRAC)
    return score, iou


def _ancho(caja) -> float:
    return float(caja[2] - caja[0])


def _iou_mejor_rival(caja, rivales) -> float:
    if not rivales:
        return 0.0
    return max(float(iou_cajas(caja, r)) for r in rivales)


@dataclass
class AnclaPersonaSesion:
    activo: bool = True
    max_dist_px: float = WRIST_MATCH_MAX_PX
    caja: tuple | None = None
    munecas: list = field(default_factory=list)
    hist: np.ndarray | None = None
    velocidad: tuple[float, float] = (0.0, 0.0)
    perdidos: int = 0
    frame_semilla: int | None = None
    frame_idx: int = 0
    frames_anclados: int = 0
    frames_perdidos: int = 0
    max_candidatos: int = 0
    metodo: str | None = None
    criterio: str = "auto"
    indice_pedido: int | None = None
    punto: tuple[float, float] | None = None
    munecas_sesion: list = field(default_factory=list)
    indice_semilla: int | None = None
    preview: np.ndarray | None = None
    candidatos: list = field(default_factory=list)
    rivales: list = field(default_factory=list)
    ancho: float | None = None
    altura_ref: float | None = None

    def reset(self) -> None:
        self.caja = None
        self.munecas = []
        self.hist = None
        self.velocidad = (0.0, 0.0)
        self.perdidos = 0
        self.frame_semilla = None
        self.frame_idx = 0
        self.frames_anclados = 0
        self.frames_perdidos = 0
        self.max_candidatos = 0
        self.metodo = None
        self.indice_semilla = None
        self.preview = None
        self.candidatos = []
        self.rivales = []
        self.ancho = None
        self.altura_ref = None

    def resumen(self) -> dict:
        return {
            "session_person_anchor": "temporal_iou" if self.activo else "off",
            "wrist_match_max_px": self.max_dist_px,
            "seed_frame": self.frame_semilla,
            "frames_locked": self.frames_anclados,
            "frames_lost": self.frames_perdidos,
            "max_candidates": self.max_candidatos,
            "seed_criterion": self.criterio,
            "seed_index": self.indice_semilla,
            "seed_wrists": [list(p) for p in self.munecas_sesion],
            "seed_candidates": list(self.candidatos),
        }

    def filtrar(self, contexto, frame=None) -> None:
        personas = list(getattr(contexto, "personas", []) or [])
        n = len(personas)
        self.max_candidatos = max(self.max_candidatos, n)
        contexto.n_personas_detectadas = n
        contexto.ancla_metodo = None
        if not self.activo:
            self.frame_idx += 1
            return
        if not personas:
            self.perdidos += 1
            self.frames_perdidos += 1
            self.frame_idx += 1
            contexto.ancla_metodo = "empty"
            return
        if self.caja is None:
            indice = elegir_indice_semilla(
                personas, contexto.ancho, contexto.alto,
                criterio=self.criterio, indice=self.indice_pedido, punto=self.punto,
                munecas=self.munecas_sesion, max_dist_px=self.max_dist_px,
            )
            self.metodo = "seed"
            self.frame_semilla = self.frame_idx
            orden = _orden_izquierda_derecha(personas)
            self.indice_semilla = (
                orden.index(indice) if indice is not None and indice in orden else None
            )
            self.candidatos = lista_candidatos_ancla(personas, indice)
            if frame is not None and indice is not None:
                self.preview = dibujar_candidatos_ancla(frame, personas, indice)
            elif frame is not None:
                self.preview = dibujar_candidatos_ancla(frame, personas, -1)
            print("    ancla candidatos (muñecas YOLO del primer frame con gente):")
            for fila in self.candidatos:
                mun = " ".join(f"{x},{y}" for x, y in fila.get("wrists") or [])
                marca = "  <- elegido" if fila["chosen"] else ""
                print(f"      {fila['index']} caja={fila['box']} munecas={mun or '—'}{marca}")
            if self.munecas_sesion:
                pts = " ".join(f"{p[0]},{p[1]}" for p in self.munecas_sesion)
                print(f"    ancla: muñecas sesión {pts} criterio={self.criterio}")
            if self.indice_semilla is not None:
                print(
                    f"    ancla: persona {self.indice_semilla}/{max(n - 1, 0)} "
                    f"(0 = izquierda) criterio={self.criterio}"
                )
            else:
                print("    ancla: ninguna persona a ≤ --ancla-distancia de las muñecas")
        else:
            indice = self._seguir(personas, frame)
        if indice is None:
            contexto.personas = []
            contexto.manos = []
            self.perdidos += 1
            self.frames_perdidos += 1
            self.metodo = "lost"
            contexto.ancla_metodo = "lost"
            # Quien queda a la vista es el otro sujeto: no se le hereda el ancla.
            self.rivales = [tuple(int(v) for v in p.caja) for p in personas]
            self.frame_idx += 1
            return
        persona = personas[indice]
        contexto.personas = [persona]
        contexto.manos = [m for m in contexto.manos if getattr(m, "persona_id", -1) == indice]
        for mano in contexto.manos:
            mano.persona_id = 0
        self.rivales = [
            tuple(int(v) for v in p.caja)
            for i, p in enumerate(personas) if i != indice
        ]
        self._actualizar(persona, frame)
        self.frames_anclados += 1
        self.perdidos = 0
        contexto.ancla_metodo = self.metodo
        self.frame_idx += 1

    def _caja_predicha(self):
        if self.caja is None:
            return None
        dx, dy = self.velocidad
        x1, y1, x2, y2 = self.caja
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _es_el_rival(self, caja, iou_track: float) -> bool:
        """True si la detección encaja mejor con un transeúnte que con el sujeto."""
        iou_rival = _iou_mejor_rival(caja, self.rivales)
        if iou_rival >= IOU_LOCK and iou_rival >= iou_track:
            return True
        if iou_rival > iou_track + 0.08:
            return True
        ancho = _ancho(caja)
        if (
            self.ancho
            and ancho > ANCHO_RIVAL_FRAC * self.ancho
            and iou_rival >= 0.20
        ):
            return True
        return False

    def _seguir(self, personas, frame) -> int | None:
        pred = self._caja_predicha() or self.caja
        filas = []
        for i, persona in enumerate(personas):
            s, iou = score_seguimiento(
                persona, pred, self.munecas, self.hist, frame, self.max_dist_px
            )
            hs = _corr_hist(self.hist, _histograma_torso(frame, persona.caja))
            rival = self._es_el_rival(persona.caja, iou)
            filas.append((i, s, iou, hs, rival))
        validas = [f for f in filas if not f[4]]
        if not validas:
            self.metodo = "rival"
            return None
        con_color = [f for f in validas if f[3] >= HIST_FUERTE]
        if con_color:
            pool = con_color
        else:
            pool = [f for f in validas if f[3] >= HIST_MIN or f[2] >= IOU_LOCK]
            if not pool:
                pool = validas
        mejor_i, mejor_s, mejor_iou, mejor_hs, _rival = max(
            pool, key=lambda f: f[1] + 2.2 * max(0.0, f[3])
        )
        if self.hist is not None and mejor_hs < HIST_MIN and mejor_iou < 0.45:
            return None
        if mejor_iou >= IOU_LOCK:
            self.metodo = "iou"
            return mejor_i
        if mejor_s > 1.15 and mejor_iou >= IOU_RECOVER:
            self.metodo = "recover"
            return mejor_i
        if self.perdidos < MAX_PERDIDOS and mejor_iou >= IOU_RECOVER:
            self.metodo = "hold"
            return mejor_i
        return None

    def _actualizar(self, persona, frame) -> None:
        c1 = _centro(persona.caja)
        if self.caja is not None:
            c0 = _centro(self.caja)
            self.velocidad = (c1[0] - c0[0], c1[1] - c0[1])
        self.caja = tuple(int(v) for v in persona.caja)
        self.ancho = _ancho(self.caja)
        self.altura_ref = float(persona.altura)
        self.munecas = _munecas_px(persona)
        ambiguo = bool(self.rivales) and _iou_mejor_rival(self.caja, self.rivales) >= 0.12
        hist = _histograma_torso(frame, persona.caja)
        if hist is not None and not ambiguo:
            self.hist = hist
