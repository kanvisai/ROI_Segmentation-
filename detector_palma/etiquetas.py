"""Clase COCO del recorte de muñeca y confianza de detección (no el grip)."""

from __future__ import annotations

from typing import Any

CLASE_A_CATEGORIA = {
    "bottle": "botella",
    "cell phone": "celular",
    "cellphone": "celular",
    "mobile phone": "celular",
    "phone": "celular",
    "cup": "vaso",
    "wine glass": "copa",
    "book": "libro",
    "handbag": "bolso",
    "backpack": "mochila",
    "suitcase": "maleta",
    "bowl": "bol",
    "box": "caja",
}

GENERICAS = frozenset({"", "objeto", "object", "none", "unknown", "n/a", "person"})


def normalizar_etiqueta(etiqueta: Any) -> str:
    texto = " ".join(str(etiqueta or "").strip().lower().replace("_", " ").split())
    return texto or "objeto"


def es_clase_nombrada(etiqueta: Any) -> bool:
    return normalizar_etiqueta(etiqueta) not in GENERICAS


def categoria_de_etiqueta(etiqueta: Any) -> str:
    texto = normalizar_etiqueta(etiqueta)
    if texto in CLASE_A_CATEGORIA:
        return CLASE_A_CATEGORIA[texto]
    if not es_clase_nombrada(texto):
        return "objeto"
    return texto.replace(" ", "_")


def heredar_etiqueta(destino: Any, fuente: Any) -> None:
    """Copia la clase YOLO nombrada al destino si gana o el destino no tiene clase."""
    if destino is None or fuente is None or destino is fuente:
        return
    lab_f = getattr(fuente, "etiqueta", "objeto")
    if not es_clase_nombrada(lab_f):
        return
    score_f = float(getattr(fuente, "score", 0.0) or 0.0)
    lab_d = getattr(destino, "etiqueta", "objeto")
    score_d = float(getattr(destino, "score", 0.0) or 0.0)
    if es_clase_nombrada(lab_d) and score_d >= score_f:
        return
    destino.etiqueta = lab_f
    destino.score = score_f
    extra_d = dict(getattr(destino, "extra", None) or {})
    extra_f = dict(getattr(fuente, "extra", None) or {})
    if extra_f.get("clase_id") is not None:
        extra_d["clase_id"] = extra_f["clase_id"]
    destino.extra = extra_d


def mezclar_votos_clase(
    etiquetas: dict | None,
    scores: dict | None,
    otras_etiquetas: dict | None,
    otros_scores: dict | None,
) -> tuple[dict, dict]:
    etiq = dict(etiquetas or {})
    sc = {f: float(s or 0.0) for f, s in (scores or {}).items()}
    for frame, lab in (otras_etiquetas or {}).items():
        if frame not in etiq or not es_clase_nombrada(etiq.get(frame)):
            etiq[frame] = lab
        elif es_clase_nombrada(lab):
            s_new = float((otros_scores or {}).get(frame, 0.0) or 0.0)
            s_old = float(sc.get(frame, 0.0) or 0.0)
            if s_new > s_old:
                etiq[frame] = lab
    for frame, s in (otros_scores or {}).items():
        val = float(s or 0.0)
        if frame not in sc or val > float(sc.get(frame) or 0.0):
            sc[frame] = val
    return etiq, sc


def _clip_confianza(valor: float | None) -> float | None:
    if valor is None:
        return None
    conf = round(max(0.0, min(1.0, float(valor))), 4)
    return conf if conf > 0.0 else None


def score_deteccion_util(score: Any, etiqueta: Any = None) -> float | None:
    """Descarta el 1.0 de relleno de SAM cuando no hay clase COCO."""
    try:
        s = float(score or 0.0)
    except (TypeError, ValueError):
        return None
    if s <= 0.0:
        return None
    if not es_clase_nombrada(etiqueta) and s >= 0.999:
        return None
    return min(1.0, s)


def resumir_deteccion_objeto(
    etiquetas: dict | None,
    scores: dict | None,
    fallback_scores: list[float] | None = None,
) -> tuple[str, float | None, str]:
    """Devuelve (category, confidence, label crudo).

    Con clase YOLO usa el máximo de esa clase. Sin clasificar, el score real de
    detección (SAM/YOLO) o, si no hay, el de ``fallback_scores`` (agarre).
    """
    votos: dict[str, list[float]] = {}
    for frame, raw in (etiquetas or {}).items():
        if not es_clase_nombrada(raw):
            continue
        s = float((scores or {}).get(frame, 0.0) or 0.0)
        votos.setdefault(normalizar_etiqueta(raw), []).append(s)
    if votos:
        mejor_lab, vals = max(
            votos.items(),
            key=lambda item: (sum(item[1]), max(item[1]) if item[1] else 0.0, len(item[1])),
        )
        conf = _clip_confianza(max(vals) if vals else None)
        return categoria_de_etiqueta(mejor_lab), conf, mejor_lab
    reales: list[float] = []
    for frame, raw in (etiquetas or {}).items():
        s = score_deteccion_util((scores or {}).get(frame), raw)
        if s is not None:
            reales.append(s)
    if not reales:
        for frame, s in (scores or {}).items():
            val = score_deteccion_util(s, (etiquetas or {}).get(frame))
            if val is not None:
                reales.append(val)
    if reales:
        return "objeto", _clip_confianza(max(reales)), "objeto"
    fb = [
        float(s)
        for s in (fallback_scores or [])
        if s is not None and float(s) > 0.0
    ]
    if fb:
        return "objeto", _clip_confianza(max(fb)), "objeto"
    return "objeto", None, "objeto"
