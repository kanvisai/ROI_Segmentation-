"""Tipos comunes intercambiados entre backends, filtro de agarre y seguimiento."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometria import Caja, caja_desde_mascara, centroide


@dataclass
class Instancia:
    """Una mascara de objeto detectada en un frame, antes de asociarse a un track."""

    mascara: np.ndarray                      # booleana, del tamano del frame
    caja: Caja
    etiqueta: str = "objeto"
    score: float = 1.0
    id_backend: int | None = None            # id nativo si el backend ya rastrea
    es_persona: bool = False
    score_agarre: float = 0.0
    mano_asociada: str = ""
    en_mano: bool = False
    scores_mano: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mascara))

    @property
    def centro(self) -> tuple[float, float]:
        c = centroide(self.mascara)
        if c is not None:
            return c
        x1, y1, x2, y2 = self.caja
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def desde_mascara(cls, mascara: np.ndarray, **kwargs) -> "Instancia | None":
        caja = caja_desde_mascara(mascara)
        if caja is None:
            return None
        return cls(mascara=mascara.astype(bool), caja=caja, **kwargs)
