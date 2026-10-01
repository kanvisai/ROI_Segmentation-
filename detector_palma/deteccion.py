"""Detectores de instancias reutilizables.

`DetectorUltralytics` cubre YOLO26/YOLO11-seg, FastSAM, MobileSAM y SAM via el
mismo paquete `ultralytics`, y es tambien el generador de mascaras iniciales
para los backends de propagacion (SAM 2, Cutie, Track-Anything...).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometria import caja_desde_mascara, iou_cajas, redimensionar_mascara
from .tipos import Instancia
from .utils import importar

AYUDA_ULTRALYTICS = "pip install ultralytics"


class DetectorUltralytics:
    """Envuelve un modelo de Ultralytics y devuelve `Instancia` normalizadas."""

    def __init__(
        self,
        modelo: str = "yolo26n-seg.pt",
        dispositivo: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        media_precision: bool = False,
        clases_excluidas: tuple[str, ...] = ("person",),
    ):
        YOLO = importar("ultralytics", AYUDA_ULTRALYTICS).YOLO
        self.modelo = YOLO(modelo)
        self.nombre_modelo = modelo
        self.dispositivo = dispositivo
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.media_precision = media_precision
        self.clases_excluidas = {c.strip().lower() for c in clases_excluidas if c.strip()}
        self._tracker_disponible = True
        self._sam: GeneradorMascarasSAM | None = None
        self._sam_intentado = False

    @property
    def nombres_clase(self) -> dict:
        return getattr(self.modelo, "names", {}) or {}

    def detectar(
        self,
        frame: np.ndarray,
        con_tracking: bool = False,
        tracker: str = "botsort.yaml",
        contexto=None,
        recorte_persona: bool = True,
    ) -> list[Instancia]:
        alto, ancho = frame.shape[:2]
        instancias = self._inferir(frame, ancho, alto, con_tracking=con_tracking, tracker=tracker)
        if recorte_persona and contexto is not None and getattr(contexto, "manos", None):
            extras = self._inferir_en_manos(frame, contexto)
            extras.extend(self._sam_en_manos(frame, contexto))
            instancias = self._fusionar(instancias, extras)
        elif recorte_persona and contexto is not None and getattr(contexto, "personas", None):
            extras = self._inferir_en_personas(frame, contexto)
            instancias = self._fusionar(instancias, extras)
        return instancias

    def _comunes(self, conf: float | None = None) -> dict:
        comunes = dict(
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.dispositivo,
            retina_masks=True,
            verbose=False,
        )
        if self.media_precision:
            comunes["half"] = True
        return comunes

    def _inferir(
        self,
        frame: np.ndarray,
        ancho: int,
        alto: int,
        con_tracking: bool = False,
        tracker: str = "botsort.yaml",
        conf: float | None = None,
    ) -> list[Instancia]:
        comunes = self._comunes(conf)
        if con_tracking and self._tracker_disponible:
            try:
                resultados = self.modelo.track(frame, persist=True, tracker=tracker, **comunes)
            except ImportError as exc:
                print(
                    f"[aviso] El tracker de Ultralytics no esta disponible ({exc}).\n"
                    "        Instala 'lap' (pip install lap) para usar BoT-SORT/ByteTrack.\n"
                    "        Mientras tanto se usara solo el rastreador interno del proyecto."
                )
                self._tracker_disponible = False
                resultados = self.modelo.predict(frame, **comunes)
        else:
            resultados = self.modelo.predict(frame, **comunes)
        if not resultados:
            return []
        return self._convertir(resultados[0], ancho, alto)

    def _inferir_en_personas(self, frame: np.ndarray, contexto) -> list[Instancia]:
        """Segunda pasada sobre el ROI de cada persona para objetos pequenos (tarro, bote)."""
        extras: list[Instancia] = []
        for persona in contexto.personas:
            extras.extend(
                self._inferir_recorte(frame, persona.caja_padded, min(self.conf, 0.15))
            )
        return extras

    def _inferir_en_manos(self, frame: np.ndarray, contexto) -> list[Instancia]:
        """Tercera pasada, recorte centrado en cada palma (el tarro ocupa pocos pixeles)."""
        extras: list[Instancia] = []
        for mano in getattr(contexto, "manos", []) or []:
            palma = mano.palma or mano.punto
            radio = max(64, int(round(max(mano.radio, mano.radio_palma * 3.5))))
            cx, cy = int(round(palma[0])), int(round(palma[1]))
            caja = (cx - radio, cy - radio, cx + radio, cy + radio)
            locales = self._inferir_recorte(frame, caja, min(self.conf, 0.18))
            for inst in locales:
                if self._es_objeto_en_palma(inst, mano):
                    extras.append(inst)
        return extras

    def detectar_clases(
        self, frame: np.ndarray, clases: list[int], conf: float | None = None
    ) -> list[Instancia]:
        """Una pasada de frame completo limitada a las clases COCO pedidas."""
        if not clases:
            return []
        alto, ancho = frame.shape[:2]
        comunes = self._comunes(conf)
        comunes["classes"] = [int(c) for c in clases]
        resultados = self.modelo.predict(frame, **comunes)
        if not resultados:
            return []
        return self._convertir(resultados[0], ancho, alto)

    def detectar_en_caja(
        self, frame: np.ndarray, caja, conf: float | None = None
    ) -> list[Instancia]:
        """Una pasada solo dentro de la caja, con las mascaras ya reproyectadas al frame."""
        return self._inferir_recorte(frame, caja, self.conf if conf is None else conf)

    def reproyectar_desde_recorte(
        self, inst: Instancia, caja, ancho: int, alto: int
    ) -> Instancia | None:
        """Lleva una mascara obtenida en un recorte a coordenadas del frame completo."""
        x1, y1, x2, y2 = caja
        return self._reproyectar(inst, x1, y1, x2, y2, ancho, alto)

    def _inferir_recorte(
        self, frame: np.ndarray, caja, conf: float
    ) -> list[Instancia]:
        alto, ancho = frame.shape[:2]
        x1, y1, x2, y2 = caja
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(ancho, int(x2)), min(alto, int(y2))
        if (x2 - x1) < 48 or (y2 - y1) < 48:
            return []
        recorte = frame[y1:y2, x1:x2]
        locales = self._inferir(
            recorte, recorte.shape[1], recorte.shape[0], con_tracking=False, conf=conf
        )
        extras: list[Instancia] = []
        for inst in locales:
            if inst.es_persona:
                continue
            proyectada = self._reproyectar(inst, x1, y1, x2, y2, ancho, alto)
            if proyectada is not None:
                extras.append(proyectada)
        return extras

    @staticmethod
    def _reproyectar(
        inst: Instancia, x1: int, y1: int, x2: int, y2: int, ancho: int, alto: int
    ) -> Instancia | None:
        mascara = np.zeros((alto, ancho), dtype=bool)
        mh, mw = inst.mascara.shape[:2]
        h_dest = min(mh, max(0, alto - y1), max(0, y2 - y1))
        w_dest = min(mw, max(0, ancho - x1), max(0, x2 - x1))
        if h_dest <= 0 or w_dest <= 0:
            return None
        mascara[y1:y1 + h_dest, x1:x1 + w_dest] = inst.mascara[:h_dest, :w_dest]
        caja = caja_desde_mascara(mascara)
        if caja is None:
            return None
        inst.mascara = mascara
        inst.caja = caja
        inst.id_backend = None
        return inst

    @staticmethod
    def _es_objeto_en_palma(inst: Instancia, mano) -> bool:
        """Se queda con mascaras compactas pegadas a la palma (tarro), no brazos ni estanterias."""
        palma = mano.palma or mano.punto
        area = inst.area
        if area < 80 or area > 4000:
            return False
        x1, y1, x2, y2 = inst.caja
        ancho_c, alto_c = max(1, x2 - x1), max(1, y2 - y1)
        if max(ancho_c, alto_c) / min(ancho_c, alto_c) > 3.2:
            return False
        if inst.area / float(ancho_c * alto_c) < 0.28:
            return False
        ox, oy = inst.centro
        if float(np.hypot(ox - palma[0], oy - palma[1])) > max(24.0, mano.radio * 0.85):
            return False
        return True

    def _asegurar_sam(self) -> GeneradorMascarasSAM | None:
        if self._sam_intentado:
            return self._sam
        self._sam_intentado = True
        candidatos = [
            Path(__file__).resolve().parent.parent / "pesos" / "mobile_sam.pt",
            Path("pesos") / "mobile_sam.pt",
        ]
        for ruta in candidatos:
            if ruta.exists():
                try:
                    self._sam = GeneradorMascarasSAM(str(ruta), self.dispositivo, self.imgsz, 0.2)
                    print(f"[deteccion] SAM en palma: {ruta}")
                    break
                except Exception as exc:
                    print(f"[aviso] No se pudo cargar MobileSAM ({exc}).")
                    self._sam = None
        return self._sam

    def _sam_en_manos(self, frame: np.ndarray, contexto) -> list[Instancia]:
        """Segmenta el objeto compacto mas cercano a cada palma, sin depender de la clase COCO."""
        sam = self._asegurar_sam()
        if sam is None:
            return []
        alto, ancho = frame.shape[:2]
        extras: list[Instancia] = []
        for mano in getattr(contexto, "manos", []) or []:
            palma = mano.palma or mano.punto
            radio = max(48, int(round(max(mano.radio * 1.1, mano.radio_palma * 4))))
            cx, cy = int(round(palma[0])), int(round(palma[1]))
            x1, y1 = max(0, cx - radio), max(0, cy - radio)
            x2, y2 = min(ancho, cx + radio), min(alto, cy + radio)
            if (x2 - x1) < 32 or (y2 - y1) < 32:
                continue
            recorte = frame[y1:y2, x1:x2]
            puntos = [(palma[0] - x1, palma[1] - y1)]
            for p in list(mano.puntos_agarre or [])[:3]:
                puntos.append((p[0] - x1, p[1] - y1))
            locales = sam.segmentar(recorte, puntos=puntos)
            area_max = max(120.0, float(np.pi * (mano.radio_palma * 2.6) ** 2))
            mejor = None
            mejor_dist = 1e9
            for inst in locales:
                proyectada = self._reproyectar(inst, x1, y1, x2, y2, ancho, alto)
                if proyectada is None:
                    continue
                if not (40 <= proyectada.area <= area_max):
                    continue
                ox, oy = proyectada.centro
                dist = float(np.hypot(ox - palma[0], oy - palma[1]))
                if dist > max(24.0, mano.radio * 0.8):
                    continue
                if not self._es_objeto_en_palma(proyectada, mano):
                    continue
                if dist < mejor_dist:
                    mejor, mejor_dist = proyectada, dist
            if mejor is not None:
                mejor.etiqueta = "objeto"
                mejor.id_backend = None
                extras.append(mejor)
        return extras

    @staticmethod
    def _fusionar(base: list[Instancia], extras: list[Instancia], iou_fusion: float = 0.45) -> list[Instancia]:
        """Une extras con el frame completo. Conserva un objeto pequeno dentro de uno grande (tarro vs bolso)."""
        if not extras:
            return base
        fusion = list(base)
        for extra in extras:
            mejor = 0.0
            rival = None
            for ya in fusion:
                iou = iou_cajas(extra.caja, ya.caja)
                if iou > mejor:
                    mejor, rival = iou, ya
            if mejor < iou_fusion or (rival is not None and extra.area < 0.55 * max(1, rival.area)):
                fusion.append(extra)
        return fusion

    def _convertir(self, resultado, ancho: int, alto: int) -> list[Instancia]:
        cajas = resultado.boxes
        mascaras = resultado.masks
        if cajas is None or len(cajas) == 0:
            return []

        nombres = getattr(resultado, "names", self.nombres_clase) or {}
        datos_mascara = None
        if mascaras is not None and mascaras.data is not None:
            datos_mascara = mascaras.data.cpu().numpy()

        instancias: list[Instancia] = []
        for i in range(len(cajas)):
            x1, y1, x2, y2 = (int(v) for v in cajas.xyxy[i].cpu().numpy())
            clase_id = int(cajas.cls[i]) if cajas.cls is not None else -1
            etiqueta = str(nombres.get(clase_id, clase_id)).lower()
            score = float(cajas.conf[i]) if cajas.conf is not None else 1.0
            id_backend = int(cajas.id[i]) if getattr(cajas, "id", None) is not None else None

            if datos_mascara is not None and i < len(datos_mascara):
                mascara = redimensionar_mascara(datos_mascara[i] > 0.5, ancho, alto)
            else:
                # Modelos solo-deteccion: la caja hace de mascara rectangular.
                mascara = np.zeros((alto, ancho), dtype=bool)
                mascara[max(0, y1):min(alto, y2), max(0, x1):min(ancho, x2)] = True

            caja = caja_desde_mascara(mascara)
            if caja is None:
                continue
            instancias.append(
                Instancia(
                    mascara=mascara,
                    caja=caja,
                    etiqueta=etiqueta,
                    score=score,
                    id_backend=id_backend,
                    es_persona=etiqueta in self.clases_excluidas,
                    extra={"clase_id": clase_id},
                )
            )
        return instancias


class GeneradorMascarasSAM:
    """Segmentacion sin clases (FastSAM / MobileSAM / SAM) guiada por puntos o cajas."""

    def __init__(
        self,
        modelo: str = "mobile_sam.pt",
        dispositivo: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
    ):
        modulo = importar("ultralytics", AYUDA_ULTRALYTICS)
        nombre = str(modelo).lower()
        if "fastsam" in nombre:
            self.modelo = modulo.FastSAM(modelo)
        else:
            self.modelo = modulo.SAM(modelo)
        self.nombre_modelo = modelo
        self.dispositivo = dispositivo
        self.imgsz = imgsz
        self.conf = conf

    def segmentar(
        self,
        frame: np.ndarray,
        puntos: list[tuple[float, float]] | None = None,
        cajas: list[tuple[int, int, int, int]] | None = None,
    ) -> list[Instancia]:
        alto, ancho = frame.shape[:2]
        argumentos = dict(
            device=self.dispositivo, imgsz=self.imgsz, conf=self.conf, retina_masks=True, verbose=False
        )
        if puntos:
            argumentos["points"] = [[float(x), float(y)] for x, y in puntos]
            argumentos["labels"] = [1] * len(puntos)
        if cajas:
            argumentos["bboxes"] = [list(map(int, c)) for c in cajas]

        resultados = self.modelo(frame, **argumentos)
        if not resultados:
            return []
        resultado = resultados[0]
        if resultado.masks is None or resultado.masks.data is None:
            return []

        instancias: list[Instancia] = []
        datos = resultado.masks.data.cpu().numpy()
        for i, bruta in enumerate(datos):
            mascara = redimensionar_mascara(bruta > 0.5, ancho, alto)
            caja = caja_desde_mascara(mascara)
            if caja is None:
                continue
            instancias.append(
                Instancia(mascara=mascara, caja=caja, etiqueta="objeto", score=1.0 - i * 1e-3)
            )
        return instancias
