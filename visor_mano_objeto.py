#!/usr/bin/env python3
"""VISOR / EPIC-KITCHENS: mano y objeto en contacto.

El modelo HOS de VISOR (PointRend sobre detectron2) está entrenado con vídeo
egocéntrico de cocina. Predice la mano, si está en contacto, de qué lado es
y el objeto asociado. En un plano en tercera persona puede ir peor que Cutie
o el detector de palma: entra en la comparativa para medirlo.

La persona y el ROI salen de Faster R-CNN. El lado de la mano lo dice el
propio modelo (0 izquierda, 1 derecha).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from hoist_mano_objeto import detectar_personas
from pipeline_manos import (
    describir_muestreo,
    frame_origen,
    leer_frames,
    offsets_muestreo,
    resolver_videos,
)
from salida_comun import Medidor, carpeta_de, publicar_comparativa
from sam3_mano_objeto import (
    Deteccion,
    anotar_video,
    caja_de_mascara,
    centro_de_mascara,
    comprobar_entorno,
    describir_eventos,
    dilatar,
    encadenar,
    expandir_caja,
    fraccion_dentro_de_caja,
    info_video,
    linea_temporal,
    quitar_parpadeos,
)

RAIZ = Path(__file__).resolve().parent
REPO_POR_DEFECTO = RAIZ / "third_party" / "VISOR-HOS"
PESOS_POR_DEFECTO = REPO_POR_DEFECTO / "checkpoints" / "model_final_hos.pth"
CONFIG_POR_DEFECTO = REPO_POR_DEFECTO / "configs" / "hos" / "hos_pointrend_rcnn_R_50_FPN_1x.yaml"
_ESCALA_OFFSET = 1000.0


def cargar_predictor(repo: Path, pesos: Path, config: Path, umbral: float):
    repo = repo.expanduser().resolve()
    pesos = pesos.expanduser().resolve()
    config = config.expanduser().resolve()
    if not (repo / "hos").is_dir():
        raise SystemExit(
            f"No está VISOR-HOS en {repo}.\n"
            "  git clone --depth 1 https://github.com/epic-kitchens/VISOR-HOS.git third_party/VISOR-HOS"
        )
    if not pesos.is_file():
        raise SystemExit(
            f"No están los pesos en {pesos}.\n"
            "  https://www.dropbox.com/s/bfu94fpft2wi5sn/model_final_hos.pth"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import hos  # noqa: F401  registra HOSROIHeads
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects import point_rend

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(str(config))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = umbral
    cfg.MODEL.WEIGHTS = str(pesos)
    cfg.MODEL.DEVICE = "cuda"
    print(f"Cargando VISOR-HOS ({pesos.name})")
    return DefaultPredictor(cfg)


def _vector(instancias, nombre: str, i: int):
    if not instancias.has(nombre):
        return None
    return instancias.get(nombre)[i].detach().cpu().numpy()


def _lado(vector) -> str | None:
    if vector is None:
        return None
    return "derecha" if int(np.argmax(vector)) == 1 else "izquierda"


def _en_contacto(vector) -> bool:
    if vector is None:
        return False
    return int(np.argmax(vector)) == 1


def _centro_caja(caja) -> tuple[float, float]:
    return ((caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2)


def leer_frame(predictor, frame_bgr: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Manos y objetos. El objeto en contacto hereda el lado de su mano."""
    salida = predictor(frame_bgr)
    instancias = salida["instances"].to("cpu")
    if len(instancias) == 0:
        return [], []
    clases = instancias.pred_classes.numpy()
    mascaras = instancias.pred_masks.numpy()
    scores = instancias.scores.numpy()
    cajas = instancias.pred_boxes.tensor.numpy()
    manos = []
    objetos = []
    for i, clase in enumerate(clases):
        mascara = mascaras[i].astype(bool)
        caja_m = caja_de_mascara(mascara)
        centro = centro_de_mascara(mascara)
        if caja_m is None or centro is None:
            continue
        if int(clase) == 0:
            manos.append(
                {
                    "lado": _lado(_vector(instancias, "pred_handsides", i)),
                    "contacto": _en_contacto(_vector(instancias, "pred_contacts", i)),
                    "offset": _vector(instancias, "pred_offsets", i),
                    "caja": cajas[i],
                    "mascara": mascara,
                    "centro": centro,
                    "bbox": caja_m,
                    "prob": float(scores[i]),
                }
            )
        else:
            objetos.append(
                {
                    "mascara": mascara,
                    "centro": centro,
                    "bbox": caja_m,
                    "prob": float(scores[i]),
                    "mano": None,
                    "en_mano": False,
                }
            )
    usados: set[int] = set()
    for mano in manos:
        if not mano["contacto"] or mano["offset"] is None or not objetos:
            continue
        vx, vy, magnitud = (float(v) for v in mano["offset"][:3])
        hx, hy = _centro_caja(mano["caja"])
        destino = np.array([hx + vx * magnitud * _ESCALA_OFFSET, hy + vy * magnitud * _ESCALA_OFFSET])
        orden = sorted(
            range(len(objetos)),
            key=lambda j: float(np.linalg.norm(np.array(objetos[j]["centro"]) - destino)),
        )
        for j in orden:
            if j in usados:
                continue
            usados.add(j)
            objetos[j]["mano"] = mano["lado"]
            objetos[j]["en_mano"] = mano["lado"] in ("izquierda", "derecha")
            break
    return manos, objetos


def seguir(frames, cajas_persona, predictor, padding, roi_min, dilatacion, ancho, alto):
    por_id: dict[int, list[Deteccion]] = {}
    objetos_vis: dict[int, list[dict]] = {}
    estado: dict[int, dict] = {}
    activos: list[tuple[int, tuple[float, float], int]] = []
    siguiente = 1
    for indice, frame in enumerate(frames):
        caja = cajas_persona[indice]
        roi = expandir_caja(caja, padding, ancho, alto) if caja is not None else None
        centro_persona = None
        if caja is not None:
            centro_persona = ((caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2)
        manos, candidatos = leer_frame(predictor, frame)
        manos_estado = []
        for orden, mano in enumerate(manos):
            if roi is not None and fraccion_dentro_de_caja(mano["mascara"], roi) < 0.3:
                dentro = roi[0] <= mano["centro"][0] <= roi[2] and roi[1] <= mano["centro"][1] <= roi[3]
                if not dentro:
                    continue
            manos_estado.append(
                {
                    "id": orden + 1,
                    "lado": mano["lado"] or "derecha",
                    "centro": mano["centro"],
                    "bbox": mano["bbox"],
                    "mascara": mano["mascara"],
                    "dilatada": dilatar(mano["mascara"], dilatacion),
                    "prob": mano["prob"],
                }
            )
        estado[indice] = {
            "roi": roi,
            "centro_persona": centro_persona,
            "persona_visible": caja is not None,
            "manos": manos_estado,
            "pose": [],
        }
        for inst in candidatos:
            if roi is not None and fraccion_dentro_de_caja(inst["mascara"], roi) < roi_min:
                continue
            mejor_i = None
            mejor_d = 1e9
            for i, (oid, centro, frame_prev) in enumerate(activos):
                if indice - frame_prev > 45:
                    continue
                dist = float(np.hypot(inst["centro"][0] - centro[0], inst["centro"][1] - centro[1]))
                if dist < mejor_d:
                    mejor_d, mejor_i = dist, i
            if mejor_i is not None and mejor_d < 180:
                oid = activos[mejor_i][0]
                activos[mejor_i] = (oid, inst["centro"], indice)
            else:
                oid = siguiente
                siguiente += 1
                activos.append((oid, inst["centro"], indice))
            por_id.setdefault(oid, []).append(
                Deteccion(
                    frame=indice,
                    id_sam3=oid,
                    centro=inst["centro"],
                    bbox=inst["bbox"],
                    prob=inst["prob"],
                    area=int(inst["mascara"].sum()),
                    mano=inst["mano"],
                    en_mano=bool(inst["en_mano"]),
                    score_agarre=inst["prob"] if inst["en_mano"] else 0.0,
                )
            )
            objetos_vis.setdefault(indice, []).append(
                {
                    "id": oid,
                    "mascara": inst["mascara"],
                    "prob": inst["prob"],
                    "caja_xywh_norm": (0.0, 0.0, 1.0, 1.0),
                }
            )
        if indice % 20 == 0:
            en_mano = sum(1 for inst in candidatos if inst["en_mano"])
            print(f"    frame {indice}: {len(manos)} manos, {en_mano} objetos en contacto")
    for dets in por_id.values():
        dets.sort(key=lambda d: d.frame)
    return estado, por_id, objetos_vis


def resumen(video: Path, eventos, ultima, nframes: int, muestreo: dict | None = None) -> str:
    lineas = [
        f"Video: {video}",
        f"Processed frames: {nframes}",
        "Model: VISOR-HOS (EPIC-KITCHENS). Hand, contact and associated object.",
        "Trained on egocentric kitchen video.",
        (muestreo or {}).get(
            "description",
            "--fps was not set. Processing ran frame by frame.",
        ),
        "",
    ]
    if not eventos and ultima is None:
        lineas.append("VISOR left no object in contact inside the ROI.")
        return "\n".join(lineas) + "\n"
    lineas.extend(describir_eventos(eventos, ultima))
    lineas.append(
        "\nLeft and right come from the VISOR laterality head. "
        "In contact means the model links that hand with the object."
    )
    return "\n".join(lineas) + "\n"


def _remapear_origen(frames_json, eventos, indices_origen, fps: float) -> None:
    def orig(idx: int) -> int:
        return frame_origen(idx, indices_origen)

    for item in frames_json:
        interno = item["frame"]
        item["frame_procesado"] = interno
        item["frame"] = orig(interno)
        item["tiempo_s"] = round(orig(interno) / fps, 4) if fps else None
        ultima = item.get("objeto", {}).get("ultima_posicion")
        if isinstance(ultima, dict) and "frame" in ultima:
            ultima["frame"] = orig(ultima["frame"])
        ultima_mano = item.get("objeto", {}).get("ultima_en_mano")
        if isinstance(ultima_mano, dict) and "frame" in ultima_mano:
            ultima_mano["frame"] = orig(ultima_mano["frame"])
    for evento in eventos:
        interno = evento["frame"]
        evento["frame_procesado"] = interno
        evento["frame"] = orig(interno)
        if "tiempo_s" in evento:
            evento["tiempo_s"] = round(orig(interno) / fps, 4) if fps else None
        for clave in ("ultima_posicion", "ultima_en_mano"):
            extra = evento.get(clave)
            if isinstance(extra, dict) and "frame" in extra:
                extra["frame"] = orig(extra["frame"])


def procesar(predictor, video: Path, args, salida: Path) -> None:
    ancho, alto, nframes_archivo, fps_archivo = info_video(video)
    frames, fps, indices_origen = leer_frames(video, args.max_frames, args.fps)
    fps = fps_archivo or fps
    nframes_origen = nframes_archivo
    if args.max_frames is not None:
        nframes_origen = min(nframes_origen, args.max_frames)
    muestreo = describir_muestreo(args.fps, fps, nframes_origen, len(frames))
    print(f"\n=== {video.name}: {ancho}x{alto}, {len(frames)} frames procesados, {fps:.3f} fps ===")
    print(f"  {muestreo['description']}")
    destino = carpeta_de(salida, "visor", video)
    medidor = Medidor("visor")
    try:
        _procesar(
            predictor, video, args, destino, medidor, frames, fps, ancho, alto,
            muestreo, indices_origen,
        )
    except Exception as exc:
        publicar_comparativa(
            salida, "visor", video.stem, "VISOR", destino, medidor.cerrar(len(frames)),
            0, 0, 0, ok=False, codigo=1, error=str(exc),
        )
        raise


def _procesar(predictor, video, args, destino, medidor, frames, fps, ancho, alto,
              muestreo, indices_origen) -> None:
    with medidor.etapa("segmentacion"):
        cajas = detectar_personas(frames, args.umbral_persona)
        estado, por_id, objetos = seguir(
            frames, cajas, predictor, args.padding, args.roi_min, args.dilatacion_mano, ancho, alto
        )
    cadena = encadenar(por_id, args.hueco_max, args.distancia_max)
    frames_json, eventos, ultima = linea_temporal(
        len(frames), fps, estado, cadena, frames_traspaso=args.frames_traspaso
    )
    eventos = quitar_parpadeos(eventos, args.hueco_parpadeo)
    _remapear_origen(frames_json, eventos, indices_origen, fps)
    metricas = medidor.cerrar(len(frames))
    offsets, _rate = offsets_muestreo(fps, args.fps)
    fps_salida = float(args.fps) if offsets is not None else fps
    anotado = destino / "anotado.mp4"
    print(f"  Escribiendo {anotado}")
    anotar_video(
        video, anotado, {}, -1, objetos, estado, frames_json, fps_salida,
        carpeta_frames=destino / "frames",
        frames_bgr=frames,
    )
    informe = {
        "video": str(video),
        "model": "VISOR-HOS",
        "width": ancho,
        "height": alto,
        "frames": len(frames),
        "fps": fps,
        "output_fps": fps_salida,
        "sampling": muestreo,
        "source_indices": indices_origen,
        "roi_padding": args.padding,
        "person_facing_camera": not args.de_espaldas,
        "object_ids": sorted({d.id_sam3 for d in cadena.detecciones}) if cadena else [],
        "events": eventos,
        "timeline": frames_json,
    }
    (destino / "tracking.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    texto = resumen(video, eventos, ultima, len(frames), muestreo)
    (destino / "summary.txt").write_text(texto, encoding="utf-8")
    (destino / "metrics.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frames_con_objeto = sum(1 for frame in frames_json if frame["objeto"]["visible"])
    objetos_n = len({d.id_sam3 for d in cadena.detecciones}) if cadena else 0
    desapariciones = sum(1 for evento in eventos if evento["tipo"] == "desaparece")
    ruta = publicar_comparativa(
        args.salida, "visor", video.stem, "VISOR", destino, metricas,
        objetos_n, desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VISOR-HOS: mano y objeto en contacto (EPIC-KITCHENS)."
    )
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--salida", type=Path, default=Path("salidas"))
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    parser.add_argument("--pesos", type=Path, default=PESOS_POR_DEFECTO)
    parser.add_argument("--config", type=Path, default=CONFIG_POR_DEFECTO)
    parser.add_argument("--umbral", type=float, default=0.5, help="Puntuación mínima de VISOR.")
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--umbral-persona", type=float, default=0.6)
    parser.add_argument("--dilatacion-mano", type=int, default=28)
    parser.add_argument("--roi-min", type=float, default=0.5)
    parser.add_argument("--hueco-max", type=int, default=45)
    parser.add_argument("--hueco-parpadeo", type=int, default=3)
    parser.add_argument("--distancia-max", type=float, default=250.0)
    parser.add_argument("--de-espaldas", action="store_true")
    parser.add_argument("--frames-traspaso", type=int, default=3)
    parser.add_argument("--video", type=Path, default=None, help="Un único vídeo. Si se indica, se ignora --videos.")
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Frames a procesar por cada segundo de vídeo. Con un vídeo a 15 fps, "
             "--fps 1 toma el primero de cada segundo; --fps 2 los frames 1 y 8; "
             "--fps 3 los frames 1, 6 y 11. Si se omite, se procesa frame a frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("VISOR en este script necesita una GPU CUDA.")
    videos = resolver_videos(args)
    predictor = cargar_predictor(args.repo, args.pesos, args.config, args.umbral)
    for video in videos:
        procesar(predictor, video, args, args.salida)


if __name__ == "__main__":
    main()
