"""Decide que instancias corresponden a un objeto sostenido en la mano.

La puntuacion combina senales de palma y de dedos, todas en [0, 1]:

  proximidad     : distancia de la palma (o muneca proyectada) a la mascara.
  contacto_palma : solape del disco de palma con el objeto (la mano esta ENCIMA).
  contacto_dedos : fraccion de puntos de agarre (abanico de dedos o landmarks)
                   que caen sobre o pegados al objeto.
  encuadre       : el objeto intersecta el ROI de la persona.

Un objeto grande (carrito, banco, estanteria) se descarta: no es mercancía
en la mano. Un objeto pequeno (bote, tarro) puntua alto si queda dentro de
la zona de la palma. No basta con estar cerca del cuerpo: hace falta
contacto palma/dedos.

Si las dos manos tocan un objeto pequeno a la vez, no se marca 'ambas':
se elige la mano ganadora y el rastreador emite un evento de traspaso
cuando el ganador cambia de lado.
"""

from __future__ import annotations

import numpy as np

from .geometria import Caja, distancia_punto_a_mascara, iou_cajas
from .personas import ContextoEscena, Mano
from .tipos import Instancia

PESO_PROXIMIDAD = 0.30
PESO_PALMA = 0.35
PESO_DEDOS = 0.25
PESO_ENCUADRE = 0.10

MARGEN_DEDO_PX = 10
# Contacto minimo de palma o dedos para aceptar "en mano" (evita el bolso al hombro).
MIN_CONTACTO_AGARRE = 0.10
# Por encima de esta fraccion de frame se permite la etiqueta 'ambas'.
FRACCION_AMBAS = 0.012
# Un carrito/banco ronda el 1-3 % del frame y ~30 % de la persona; un tarro ~0.03 %.
FRACCION_MAX_PERSONA = 0.12
ALTURA_MAX_PERSONA = 0.42


def _solape_disco(mascara: np.ndarray, cx: float, cy: float, radio: float) -> float:
    """Fraccion del disco centrado en (cx, cy) cubierta por la mascara."""
    alto, ancho = mascara.shape[:2]
    radio_i = max(4, int(round(radio)))
    cxi, cyi = int(round(cx)), int(round(cy))
    x1, y1 = max(0, cxi - radio_i), max(0, cyi - radio_i)
    x2, y2 = min(ancho, cxi + radio_i + 1), min(alto, cyi + radio_i + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    recorte = mascara[y1:y2, x1:x2]
    ys, xs = np.ogrid[y1:y2, x1:x2]
    disco = (xs - cxi) ** 2 + (ys - cyi) ** 2 <= radio_i * radio_i
    total = int(np.count_nonzero(disco))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(recorte & disco)) / total


def _punto_toca_mascara(
    mascara: np.ndarray, punto: tuple[float, float], margen: int = MARGEN_DEDO_PX
) -> bool:
    alto, ancho = mascara.shape[:2]
    x, y = int(round(punto[0])), int(round(punto[1]))
    if not (0 <= x < ancho and 0 <= y < alto):
        return False
    if mascara[y, x]:
        return True
    x1, y1 = max(0, x - margen), max(0, y - margen)
    x2, y2 = min(ancho, x + margen + 1), min(alto, y + margen + 1)
    return bool(np.any(mascara[y1:y2, x1:x2]))


def _contacto_dedos(mascara: np.ndarray, mano: Mano) -> float:
    puntos = list(mano.puntos_agarre)
    if not puntos:
        return 0.0
    toques = sum(1 for p in puntos if _punto_toca_mascara(mascara, p))
    return toques / len(puntos)


def _encuadre_en_persona(caja: Caja, contexto: ContextoEscena) -> float:
    if not contexto.personas:
        return 1.0
    return max((1.0 if iou_cajas(caja, p.caja_padded) > 0 else 0.0) for p in contexto.personas)


def es_voluminoso(
    instancia: Instancia,
    contexto: ContextoEscena,
    area_maxima: float = 0.02,
) -> bool:
    """True si el objeto es tipo carrito/mueble, no mercancía que quepa en la mano."""
    area_frame = max(1, contexto.ancho * contexto.alto)
    if instancia.area / area_frame > area_maxima:
        return True
    x1, y1, x2, y2 = instancia.caja
    alto_obj = max(1, y2 - y1)
    if not contexto.personas:
        return False
    persona = max(contexto.personas, key=lambda p: p.altura)
    altura = max(1.0, persona.altura)
    if alto_obj > ALTURA_MAX_PERSONA * altura:
        return True
    area_persona = max(
        1.0,
        float(persona.caja[2] - persona.caja[0]) * float(persona.caja[3] - persona.caja[1]),
    )
    return instancia.area > FRACCION_MAX_PERSONA * area_persona


def _puntuar_con_mano(instancia: Instancia, mano: Mano, encuadre: float) -> dict:
    palma = mano.palma or mano.punto
    radio_palma = max(8.0, mano.radio_palma)
    radio_busqueda = max(radio_palma, mano.radio)
    distancia = distancia_punto_a_mascara(palma, instancia.mascara, instancia.caja)
    proximidad = float(np.clip(1.0 - distancia / max(1.0, radio_busqueda), 0.0, 1.0))
    contacto_palma = _solape_disco(instancia.mascara, palma[0], palma[1], radio_palma)
    palma_boost = min(1.0, contacto_palma * 3.0)
    dedos = _contacto_dedos(instancia.mascara, mano)
    score = (
        PESO_PROXIMIDAD * proximidad
        + PESO_PALMA * palma_boost
        + PESO_DEDOS * dedos
        + PESO_ENCUADRE * encuadre
    )
    return {
        "score": score,
        "palma": palma_boost,
        "dedos": dedos,
        "proximidad": proximidad,
        "contacto": max(palma_boost, dedos),
    }


def etiquetar_lado(lado: str) -> str:
    """Etiqueta corta para HUD: IZQ / DER / AMB / IZQ→DER."""
    clave = (lado or "").lower()
    if "→" in clave or "->" in clave:
        return clave.replace("izquierda", "IZQ").replace("derecha", "DER").replace("->", "→").upper()
    if clave.startswith("izq"):
        return "IZQ"
    if clave.startswith("der"):
        return "DER"
    if clave.startswith("amb"):
        return "AMB"
    return lado.upper()[:3] if lado else ""


def _elegir_lado(puntuaciones: list[tuple[float, str, dict]], fraccion_area: float) -> str:
    if not puntuaciones:
        return ""
    puntuaciones = sorted(puntuaciones, reverse=True)
    mejor_score, mejor_lado, _ = puntuaciones[0]
    if len(puntuaciones) < 2:
        return mejor_lado
    segundo_score, segundo_lado, _ = puntuaciones[1]
    if not mejor_lado or not segundo_lado or mejor_lado == segundo_lado:
        return mejor_lado
    ambos_altos = (
        segundo_score >= 0.55 * max(mejor_score, 1e-6) and segundo_score >= 0.22
    )
    if ambos_altos and fraccion_area >= FRACCION_AMBAS:
        return "ambas"
    return mejor_lado


def puntuar_agarre(instancia: Instancia, contexto: ContextoEscena) -> tuple[float, str]:
    """Devuelve (score, lado_de_la_mano) para una instancia candidata."""
    encuadre = _encuadre_en_persona(instancia.caja, contexto)
    if not contexto.manos:
        return (encuadre * 0.45, "")

    area_frame = max(1, contexto.ancho * contexto.alto)
    fraccion = instancia.area / area_frame
    puntuaciones: list[tuple[float, str, dict]] = []
    detalles: dict[str, dict] = {}
    for mano in contexto.manos:
        det = _puntuar_con_mano(instancia, mano, encuadre)
        puntuaciones.append((det["score"], mano.lado, det))
        detalles[mano.lado] = det

    mejor_score = max(p[0] for p in puntuaciones)
    lado = _elegir_lado(puntuaciones, fraccion)
    instancia.scores_mano = {lado_m: round(d["score"], 4) for lado_m, d in detalles.items()}
    instancia.extra["detalle_agarre"] = {
        lado_m: {k: round(float(v), 4) for k, v in d.items()} for lado_m, d in detalles.items()
    }
    palmas = {}
    for mano in contexto.manos:
        palma = mano.palma or mano.punto
        palmas[mano.lado] = (float(palma[0]), float(palma[1]))
    instancia.extra["palmas"] = palmas
    if "izquierda" in palmas and "derecha" in palmas:
        ix, iy = palmas["izquierda"]
        dx, dy = palmas["derecha"]
        instancia.extra["dist_manos"] = float(np.hypot(ix - dx, iy - dy))
    return mejor_score, lado


def anotar_agarre(
    instancias: list[Instancia],
    contexto: ContextoEscena,
    umbral: float = 0.35,
    area_minima: float = 0.0005,
    area_maxima: float = 0.02,
    ignorar_voluminosos: bool = True,
) -> list[Instancia]:
    """Puntua todas las instancias validas y marca `en_mano` sin descartar las sueltas."""
    area_frame = max(1, contexto.ancho * contexto.alto)
    anotadas: list[Instancia] = []
    for instancia in instancias:
        if instancia.es_persona:
            continue
        fraccion = instancia.area / area_frame
        if fraccion < area_minima:
            continue
        if ignorar_voluminosos:
            if es_voluminoso(instancia, contexto, area_maxima):
                continue
        elif fraccion > area_maxima:
            continue
        score, lado = puntuar_agarre(instancia, contexto)
        instancia.score_agarre = score
        instancia.mano_asociada = lado
        detalle = (instancia.extra.get("detalle_agarre") or {}).get(lado) or {}
        if lado == "ambas":
            contacto = max(
                (d.get("contacto") or 0.0)
                for d in (instancia.extra.get("detalle_agarre") or {}).values()
            ) if instancia.extra.get("detalle_agarre") else 0.0
        else:
            contacto = float(detalle.get("contacto") or 0.0)
        instancia.en_mano = bool(
            score >= umbral
            and (lado or not contexto.manos)
            and (contacto >= MIN_CONTACTO_AGARRE or not contexto.manos)
        )
        anotadas.append(instancia)
    anotadas.sort(key=lambda i: i.score_agarre, reverse=True)
    _desempatar_por_mano(anotadas)
    return anotadas


def _desempatar_por_mano(instancias: list[Instancia]) -> None:
    """Si en una mano hay tarro y bolso, el compacto pegado a la palma gana el agarre."""
    por_lado: dict[str, list[Instancia]] = {}
    for inst in instancias:
        lado = inst.mano_asociada
        if not inst.en_mano or lado not in ("izquierda", "derecha"):
            continue
        por_lado.setdefault(lado, []).append(inst)
    for grupo in por_lado.values():
        if len(grupo) < 2:
            continue
        def clave(inst: Instancia) -> tuple:
            palmas = (inst.extra or {}).get("palmas") or {}
            palma = palmas.get(inst.mano_asociada)
            dist = 0.0
            if palma:
                dist = float(np.hypot(inst.centro[0] - palma[0], inst.centro[1] - palma[1]))
            return (inst.area, dist, -inst.score_agarre)
        ganador = min(grupo, key=clave)
        for inst in grupo:
            if inst is ganador:
                continue
            palmas = (inst.extra or {}).get("palmas") or {}
            palma = palmas.get(inst.mano_asociada)
            dist_g = dist_i = 0.0
            if palma:
                dist_g = float(np.hypot(ganador.centro[0] - palma[0], ganador.centro[1] - palma[1]))
                dist_i = float(np.hypot(inst.centro[0] - palma[0], inst.centro[1] - palma[1]))
            # El ganador compacto anula un objeto claramente mas grande en la misma palma.
            if ganador.area < 80:
                continue
            if inst.area >= ganador.area * 1.6 and dist_i > dist_g + 20:
                inst.en_mano = False


def filtrar_objetos_en_mano(
    instancias: list[Instancia],
    contexto: ContextoEscena,
    umbral: float = 0.35,
    area_minima: float = 0.0005,
    area_maxima: float = 0.02,
    exigir_mano: bool = True,
) -> list[Instancia]:
    """Aplica filtros de area y agarre; por defecto solo devuelve objetos en mano."""
    anotadas = anotar_agarre(instancias, contexto, umbral, area_minima, area_maxima)
    if not exigir_mano:
        return anotadas
    return [i for i in anotadas if i.en_mano]
