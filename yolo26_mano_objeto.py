#!/usr/bin/env python3
"""YOLO26-seg: segmenta las instancias y se queda con el objeto en la mano.

[YOLO26](https://docs.ultralytics.com/models/yolo26) segmenta cada frame.
YOLO26-pose marca a la persona y las muñecas. El agarre se puntúa con la
palma y los dedos, sin descartar objetos grandes. Los objetos pequeños que
la clase COCO no nombra se completan con MobileSAM sobre la palma.

La salida es la misma que en el resto de tecnologías.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from detector_palma.deteccion import DetectorUltralytics
from detector_palma.personas import AnalizadorEscena
from hoist_mano_objeto import leer_frames
from salida_comun import Medidor, carpeta_de, publicar_comparativa
from sam3_mano_objeto import (
    PESOS,
    Deteccion,
    anotar_video,
    comprobar_entorno,
    describir_eventos,
    dilatar,
    encadenar,
    evaluar_agarre,
    expandir_caja,
    fraccion_dentro_de_caja,
    info_video,
    linea_temporal,
    listar_videos,
    quitar_parpadeos,
    ruta_pose_por_defecto,
)


def _disco(alto: int, ancho: int, centro: tuple[float, float], radio: float) -> np.ndarray:
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.circle(
        mascara,
        (int(round(centro[0])), int(round(centro[1]))),
        max(1, int(round(radio))),
        1,
        -1,
    )
    return mascara.astype(bool)


def cargar(modelo: Path, pose: Path, dispositivo: str, conf: float, imgsz: int):
    if not modelo.is_file():
        raise SystemExit(f"No está el modelo de segmentación en {modelo}.")
    if not pose.is_file():
        raise SystemExit(f"No está el modelo de pose en {pose}.")
    print(f"Cargando YOLO26-seg ({modelo.name}) y pose ({pose.name})")
    detector = DetectorUltralytics(
        modelo=str(modelo), dispositivo=dispositivo, conf=conf, imgsz=imgsz
    )
    escena = AnalizadorEscena(
        estrategia="pose",
        modelo_pose=str(pose),
        dispositivo=dispositivo,
        padding_persona=0.35,
        conf=conf,
        imgsz=imgsz,
    )
    return detector, escena


def _persona(contexto):
    if not contexto.personas:
        return None
    return max(contexto.personas, key=lambda p: p.altura * max(1, p.caja[2] - p.caja[0]))


def _como_pose(mano) -> dict:
    return {
        "lado": mano.lado,
        "centro": mano.punto,
        "palma": mano.palma or mano.punto,
        "radio": mano.radio,
        "radio_palma": mano.radio_palma,
        "puntos_agarre": list(mano.puntos_agarre or []),
    }


def seguir(frames, detector, escena, args, ancho, alto):
    por_id: dict[int, list[Deteccion]] = {}
    objetos_vis: dict[int, list[dict]] = {}
    estado: dict[int, dict] = {}
    activos: list[tuple[int, tuple[float, float], int]] = []
    siguiente = 1
    for indice, frame in enumerate(frames):
        contexto = escena.analizar(frame)
        persona = _persona(contexto)
        if persona is None:
            roi = None
            centro_persona = None
        else:
            roi = expandir_caja(persona.caja, args.padding, ancho, alto)
            centro_persona = (
                (persona.caja[0] + persona.caja[2]) / 2,
                (persona.caja[1] + persona.caja[3]) / 2,
            )
        pose = [_como_pose(mano) for mano in contexto.manos if mano.lado in ("izquierda", "derecha")]
        manos_estado = []
        for orden, mano in enumerate(pose):
            radio = max(12.0, float(mano["radio_palma"]))
            mascara = _disco(alto, ancho, mano["palma"], radio)
            bbox = (
                int(max(0, mano["palma"][0] - radio)),
                int(max(0, mano["palma"][1] - radio)),
                int(min(ancho - 1, mano["palma"][0] + radio)),
                int(min(alto - 1, mano["palma"][1] + radio)),
            )
            manos_estado.append(
                {
                    "id": orden + 1,
                    "lado": mano["lado"],
                    "centro": mano["centro"],
                    "bbox": bbox,
                    "mascara": mascara,
                    "dilatada": dilatar(mascara, args.dilatacion_mano),
                    "prob": 1.0,
                }
            )
        estado[indice] = {
            "roi": roi,
            "centro_persona": centro_persona,
            "persona_visible": persona is not None,
            "manos": manos_estado,
            "pose": pose,
        }
        if contexto.manos:
            instancias = detector.detectar(frame, contexto=contexto)
        else:
            instancias = detector.detectar(frame, contexto=contexto, recorte_persona=False)
        en_mano_n = 0
        for inst in instancias:
            if inst.es_persona or inst.etiqueta == "person":
                continue
            if roi is not None and fraccion_dentro_de_caja(inst.mascara, roi) < args.roi_min:
                continue
            agarre = evaluar_agarre(
                inst.mascara, inst.caja, pose, manos_estado, args.umbral_agarre
            )
            mejor_i = None
            mejor_d = 1e9
            for i, (oid, centro, frame_prev) in enumerate(activos):
                if indice - frame_prev > args.hueco_max:
                    continue
                dist = float(np.hypot(inst.centro[0] - centro[0], inst.centro[1] - centro[1]))
                if dist < mejor_d:
                    mejor_d, mejor_i = dist, i
            if mejor_i is not None and mejor_d < args.distancia_max:
                oid = activos[mejor_i][0]
                activos[mejor_i] = (oid, inst.centro, indice)
            else:
                oid = siguiente
                siguiente += 1
                activos.append((oid, inst.centro, indice))
            por_id.setdefault(oid, []).append(
                Deteccion(
                    frame=indice,
                    id_sam3=oid,
                    centro=inst.centro,
                    bbox=inst.caja,
                    prob=float(inst.score),
                    area=inst.area,
                    mano=agarre["lado"],
                    en_mano=bool(agarre["en_mano"]),
                    score_agarre=float(agarre["score"]),
                    scores_mano=agarre["scores"],
                )
            )
            objetos_vis.setdefault(indice, []).append(
                {
                    "id": oid,
                    "mascara": inst.mascara,
                    "prob": float(inst.score),
                    "caja_xywh_norm": (0.0, 0.0, 1.0, 1.0),
                }
            )
            if agarre["en_mano"]:
                en_mano_n += 1
        if indice % 20 == 0:
            print(f"    frame {indice}: {en_mano_n} objetos en mano")
    for dets in por_id.values():
        dets.sort(key=lambda d: d.frame)
    return estado, por_id, objetos_vis


def resumen(video: Path, eventos, ultima, nframes: int) -> str:
    lineas = [
        f"Vídeo: {video}",
        f"Frames: {nframes}",
        "Modelo: YOLO26-seg. La pose localiza las muñecas. MobileSAM completa la palma.",
        "",
    ]
    if not eventos and ultima is None:
        lineas.append("YOLO26-seg no dejó ningún objeto dentro del ROI de la persona.")
        return "\n".join(lineas) + "\n"
    lineas.extend(describir_eventos(eventos, ultima))
    lineas.append(
        "\nIzquierda y derecha salen del esqueleto de la persona. "
        "Un cambio de mano solo se anota si se mantiene varios frames. "
        "Suelto significa que el objeto sigue en escena pero ya no está en la mano."
    )
    return "\n".join(lineas) + "\n"


def procesar(detector, escena, video: Path, args, salida: Path) -> None:
    ancho, alto, _, fps_archivo = info_video(video)
    frames, fps = leer_frames(video, args.max_frames)
    if args.max_frames is None:
        fps = fps_archivo or fps
    print(f"\n=== {video.name}: {ancho}x{alto}, {len(frames)} frames, {fps:.3f} fps ===")
    destino = carpeta_de(salida, "yolo26", video)
    medidor = Medidor("yolo26")
    try:
        _procesar(detector, escena, video, args, destino, medidor, frames, fps, ancho, alto)
    except Exception as exc:
        publicar_comparativa(
            salida, "yolo26", video.stem, "YOLO26-seg", destino, medidor.cerrar(len(frames)),
            0, 0, 0, ok=False, codigo=1, error=str(exc),
        )
        raise


def _procesar(detector, escena, video, args, destino, medidor, frames, fps, ancho, alto) -> None:
    with medidor.etapa("segmentacion"):
        estado, por_id, objetos = seguir(frames, detector, escena, args, ancho, alto)
    cadena = encadenar(por_id, args.hueco_max, args.distancia_max)
    frames_json, eventos, ultima = linea_temporal(
        len(frames), fps, estado, cadena, frames_traspaso=args.frames_traspaso
    )
    eventos = quitar_parpadeos(eventos, args.hueco_parpadeo)
    metricas = medidor.cerrar(len(frames))
    anotado = destino / "anotado.mp4"
    print(f"  Escribiendo {anotado}")
    anotar_video(
        video, anotado, {}, -1, objetos, estado, frames_json, fps,
        carpeta_frames=destino / "frames",
    )
    informe = {
        "video": str(video),
        "modelo": "YOLO26-seg",
        "ancho": ancho,
        "alto": alto,
        "frames": len(frames),
        "fps": fps,
        "padding_roi": args.padding,
        "persona_de_frente": not args.de_espaldas,
        "ids_objeto": sorted({d.id_sam3 for d in cadena.detecciones}) if cadena else [],
        "eventos": eventos,
        "linea_temporal": frames_json,
    }
    (destino / "seguimiento.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    texto = resumen(video, eventos, ultima, len(frames))
    (destino / "resumen.txt").write_text(texto, encoding="utf-8")
    (destino / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frames_con_objeto = sum(1 for frame in frames_json if frame["objeto"]["visible"])
    objetos_n = len({d.id_sam3 for d in cadena.detecciones}) if cadena else 0
    desapariciones = sum(1 for evento in eventos if evento["tipo"] == "desaparece")
    ruta = publicar_comparativa(
        args.salida, "yolo26", video.stem, "YOLO26-seg", destino, metricas,
        objetos_n, desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-seg: segmentación del objeto en la mano."
    )
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--salida", type=Path, default=Path("salidas"))
    parser.add_argument("--modelo", type=Path, default=PESOS / "yolo26n-seg.pt")
    parser.add_argument("--modelo-pose", type=Path, default=ruta_pose_por_defecto())
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--umbral-agarre", type=float, default=0.35)
    parser.add_argument("--dilatacion-mano", type=int, default=28)
    parser.add_argument("--roi-min", type=float, default=0.5)
    parser.add_argument("--hueco-max", type=int, default=45)
    parser.add_argument("--hueco-parpadeo", type=int, default=3)
    parser.add_argument("--distancia-max", type=float, default=250.0)
    parser.add_argument("--de-espaldas", action="store_true")
    parser.add_argument("--frames-traspaso", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    videos = listar_videos(args.videos)
    if not videos:
        raise SystemExit(f"No hay vídeos en {args.videos}")
    detector, escena = cargar(args.modelo, args.modelo_pose, dispositivo, args.conf, args.imgsz)
    for video in videos:
        procesar(detector, escena, video, args, args.salida)


if __name__ == "__main__":
    main()
