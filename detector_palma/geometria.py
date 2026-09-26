"""Operaciones basicas sobre mascaras y bounding boxes."""

from __future__ import annotations

import numpy as np

Caja = tuple[int, int, int, int]


def caja_desde_mascara(mascara: np.ndarray) -> Caja | None:
    """Bounding box (x1, y1, x2, y2) que encierra la mascara, o None si esta vacia."""
    filas = np.any(mascara, axis=1)
    columnas = np.any(mascara, axis=0)
    if not filas.any() or not columnas.any():
        return None
    y1, y2 = np.where(filas)[0][[0, -1]]
    x1, x2 = np.where(columnas)[0][[0, -1]]
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def centroide(mascara: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mascara)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def iou_cajas(a: Caja, b: Caja) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    ancho, alto = max(0, ix2 - ix1), max(0, iy2 - iy1)
    interseccion = ancho * alto
    if interseccion == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - interseccion
    return interseccion / union if union > 0 else 0.0


def iou_mascaras(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    interseccion = np.count_nonzero(a & b)
    if interseccion == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return interseccion / union if union else 0.0


def expandir_caja(caja: Caja, padding: float, ancho: int, alto: int) -> Caja:
    """Aplica un margen relativo al tamano de la caja y recorta a los limites del frame."""
    x1, y1, x2, y2 = caja
    mx = (x2 - x1) * padding
    my = (y2 - y1) * padding
    return (
        int(max(0, round(x1 - mx))),
        int(max(0, round(y1 - my))),
        int(min(ancho, round(x2 + mx))),
        int(min(alto, round(y2 + my))),
    )


def punto_en_caja(punto: tuple[float, float], caja: Caja) -> bool:
    x, y = punto
    x1, y1, x2, y2 = caja
    return x1 <= x <= x2 and y1 <= y <= y2


def distancia_punto_a_mascara(punto: tuple[float, float], mascara: np.ndarray, caja: Caja) -> float:
    """Distancia aproximada de un punto al objeto: 0 si cae dentro de la mascara."""
    x, y = punto
    alto, ancho = mascara.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < alto and 0 <= xi < ancho and mascara[yi, xi]:
        return 0.0
    x1, y1, x2, y2 = caja
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def recortar_mascara(mascara: np.ndarray, caja: Caja) -> np.ndarray:
    """Devuelve una copia de la mascara con todo lo exterior a la caja puesto a False."""
    resultado = np.zeros_like(mascara)
    x1, y1, x2, y2 = caja
    resultado[y1:y2, x1:x2] = mascara[y1:y2, x1:x2]
    return resultado


def redimensionar_mascara(mascara: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Reescala una mascara booleana al tamano del frame usando vecino mas cercano."""
    if mascara.shape[0] == alto and mascara.shape[1] == ancho:
        return mascara.astype(bool)
    import cv2

    redimensionada = cv2.resize(
        mascara.astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_NEAREST
    )
    return redimensionada.astype(bool)


def normalizar_mascara(bruta, ancho: int, alto: int) -> np.ndarray | None:
    """Convierte cualquier formato razonable (tensor, float, logits) en una mascara booleana HxW."""
    if bruta is None:
        return None
    if hasattr(bruta, "detach"):
        bruta = bruta.detach().cpu().numpy()
    arreglo = np.asarray(bruta)
    while arreglo.ndim > 2:
        arreglo = arreglo[0] if arreglo.shape[0] == 1 else arreglo.max(axis=0)
    if arreglo.ndim != 2:
        return None
    if arreglo.dtype == bool:
        mascara = arreglo
    elif np.issubdtype(arreglo.dtype, np.floating):
        # Los predictores de tipo SAM devuelven logits; el umbral util es 0.0 o 0.5.
        umbral = 0.0 if arreglo.min() < 0 else 0.5
        mascara = arreglo > umbral
    else:
        mascara = arreglo > 0
    return redimensionar_mascara(mascara, ancho, alto)
