#!/usr/bin/env python3
"""Cutie: propaga la máscara del objeto en la mano a lo largo del vídeo.

Cutie (CVPR 2023) es un propagador de máscaras. No busca el objeto por su
cuenta: la primera máscara sale del detector de palma (YOLO + MobileSAM) y
Cutie la sigue frame a frame. Si la pierde durante varios frames y el
detector vuelve a ver el objeto, se resiembra y sigue.

La persona y el ROI salen de Faster R-CNN. La mano, del esqueleto YOLO-pose.
La salida es la misma que en SAM 3 y HOIST.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from hoist_mano_objeto import asociar, detectar_personas, leer_frames
from salida_comun import Medidor, carpeta_de, publicar_comparativa
from sam3_mano_objeto import (
    anotar_video,
    comprobar_entorno,
    describir_eventos,
    detecciones_palma,
    encadenar,
    info_video,
    linea_temporal,
    listar_videos,
    analizar_pose,
    quitar_parpadeos,
    ruta_pose_por_defecto,
)

RAIZ = Path(__file__).resolve().parent
CUTIE_RAIZ = RAIZ / "third_party" / "Cutie"


def cargar_red(repo: Path):
    """Carga los pesos oficiales. Hydra solo se inicializa una vez."""
    repo = repo.expanduser().resolve()
    if not (repo / "cutie" / "config" / "eval_config.yaml").is_file():
        raise SystemExit(
            f"No está el código de Cutie en {repo}.\n"
            "  git clone --depth 1 https://github.com/hkchengrex/Cutie.git third_party/Cutie"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    anterior = Path.cwd()
    os.chdir(repo)
    try:
        from cutie.utils.get_default_model import get_default_model

        print("Cargando Cutie (cutie-base-mega)")
        return get_default_model()
    finally:
        os.chdir(anterior)


def nuevo_procesador(red, resolucion: int):
    from cutie.inference.inference_core import InferenceCore

    procesador = InferenceCore(red, cfg=red.cfg)
    procesador.max_internal_size = resolucion
    return procesador


def _mascara_id(por_frame: dict, frame: int, oid: int) -> np.ndarray | None:
    for inst in por_frame.get(frame, []):
        if inst["id"] != oid:
            continue
        mascara = np.asarray(inst["mascara"]).astype(bool)
        if int(mascara.sum()) > 0:
            return mascara
    return None


def _elegir_semilla(por_id, por_frame):
    oid = max(
        por_id,
        key=lambda i: (sum(1 for d in por_id[i] if d.en_mano), len(por_id[i])),
    )
    for det in por_id[oid]:
        mascara = _mascara_id(por_frame, det.frame, oid)
        if mascara is not None:
            return oid, det.frame, mascara
    return None


def _componente_cercana(
    mascara: np.ndarray,
    centro: tuple[float, float] | None,
    area_min: int,
    area_max: int,
) -> np.ndarray | None:
    """Cutie a veces hincha la máscara hasta cubrir a la persona. Se queda la mancha útil."""
    n, etiquetas, stats, centros = cv2.connectedComponentsWithStats(
        mascara.astype(np.uint8), connectivity=8
    )
    mejor = None
    mejor_d = 1e9
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_min or area > area_max:
            continue
        cx, cy = float(centros[i, 0]), float(centros[i, 1])
        dist = 0.0 if centro is None else float(np.hypot(cx - centro[0], cy - centro[1]))
        if dist < mejor_d:
            mejor_d = dist
            mejor = i
    if mejor is None:
        return None
    return etiquetas == mejor


def propagar(
    red,
    frames: list[np.ndarray],
    por_id,
    por_frame: dict,
    resolucion: int,
    area_min: int,
    hueco_resembra: int,
) -> list[dict]:
    """Sigue la máscara sembrada. Si se vacía o se hincha, la vuelve a tomar del detector."""
    semilla = _elegir_semilla(por_id, por_frame)
    if semilla is None:
        return []
    oid, frame0, mascara0 = semilla
    alto, ancho = frames[0].shape[:2]
    dispositivo = next(red.parameters()).device
    procesador = nuevo_procesador(red, resolucion)
    mascaras: dict[int, np.ndarray] = {}
    perdidos = 0
    resiembras = 0
    centro: tuple[float, float] | None = None
    area_ref = max(area_min, int(mascara0.sum()))
    print(f"  Cutie siembra el objeto {oid} en el frame {frame0} (lado interno {resolucion})")

    with torch.inference_mode():
        for idx in range(frame0, len(frames)):
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            tensor = to_tensor(rgb).to(dispositivo).float()
            sembrar = idx == frame0
            if not sembrar and hueco_resembra >= 0 and perdidos >= hueco_resembra:
                nueva = _mascara_id(por_frame, idx, oid)
                if nueva is not None:
                    procesador = nuevo_procesador(red, resolucion)
                    mascara0 = nueva
                    area_ref = max(area_ref, int(nueva.sum()))
                    sembrar = True
                    perdidos = 0
                    resiembras += 1
            if sembrar:
                etiquetado = np.zeros((alto, ancho), dtype=np.int64)
                etiquetado[mascara0] = 1
                mask_t = torch.from_numpy(etiquetado).to(dispositivo)
                prob = procesador.step(tensor, mask_t, objects=[1])
            else:
                prob = procesador.step(tensor)
            etiquetas = procesador.output_prob_to_mask(prob).detach().cpu().numpy()
            area_max = max(4000, area_ref * 4)
            util = _componente_cercana(etiquetas == 1, centro, area_min, area_max)
            if util is not None:
                mascaras[idx] = util
                ys, xs = np.nonzero(util)
                centro = (float(xs.mean()), float(ys.mean()))
                perdidos = 0
                area_vista = int(util.sum())
            else:
                perdidos += 1
                area_vista = int((etiquetas == 1).sum())
            if idx == frame0 or (idx - frame0) % 20 == 0:
                print(f"    frame {idx}: {area_vista} px")

    print(f"  Cutie: {len(mascaras)} frames con máscara, {resiembras} resiembras")
    if not mascaras:
        return []
    return [{"id": oid, "score": 1.0, "mascaras": mascaras}]


def resumen(video: Path, eventos: list[dict], ultima: dict | None, nframes: int) -> str:
    lineas = [
        f"Vídeo: {video}",
        f"Frames: {nframes}",
        "Modelo: Cutie. La máscara inicial sale del detector de palma y Cutie la propaga.",
        "",
    ]
    if not eventos and ultima is None:
        lineas.append("Cutie no dejó ningún objeto dentro del ROI de la persona.")
        return "\n".join(lineas) + "\n"
    lineas.extend(describir_eventos(eventos, ultima))
    lineas.append(
        "\nIzquierda y derecha salen del esqueleto de la persona. "
        "Un cambio de mano solo se anota si se mantiene varios frames. "
        "Suelto significa que el objeto sigue en escena pero ya no está en la mano."
    )
    return "\n".join(lineas) + "\n"


def procesar(red, video: Path, args, salida: Path) -> None:
    ancho, alto, _, fps_archivo = info_video(video)
    frames, fps = leer_frames(video, args.max_frames)
    if args.max_frames is None:
        fps = fps_archivo or fps
    print(f"\n=== {video.name}: {ancho}x{alto}, {len(frames)} frames, {fps:.3f} fps ===")
    destino = carpeta_de(salida, "cutie", video)
    medidor = Medidor("cutie")
    try:
        _procesar_cutie(red, video, args, destino, medidor, frames, fps, ancho, alto)
    except Exception as exc:
        publicar_comparativa(
            salida, "cutie", video.stem, "Cutie", destino, medidor.cerrar(len(frames)),
            0, 0, 0, ok=False, codigo=1, error=str(exc),
        )
        raise


def _procesar_cutie(red, video, args, destino, medidor, frames, fps, ancho, alto) -> None:
    with medidor.etapa("segmentacion"):
        cajas = detectar_personas(frames, args.umbral_persona)
        pose = None if args.sin_pose else analizar_pose(frames, len(frames), args.modelo_pose)
        palma = detecciones_palma(frames, len(frames))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pistas = []
        if palma is not None:
            por_id_palma, por_frame_palma = palma
            pistas = propagar(
                red,
                frames,
                por_id_palma,
                por_frame_palma,
                resolucion=args.resolucion_interna,
                area_min=args.area_min,
                hueco_resembra=args.hueco_resembra,
            )
        estado, por_id, objetos = asociar(
            nframes=len(frames),
            ancho=ancho,
            alto=alto,
            cajas_persona=cajas,
            manos_por_frame={},
            pistas=pistas,
            padding=args.padding,
            dilatacion=args.dilatacion_mano,
            roi_min=args.roi_min,
            contacto_min=args.contacto_min,
            de_frente=not args.de_espaldas,
            manos_pose=pose,
            umbral_agarre=args.umbral_agarre,
        )
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
        "modelo": "Cutie",
        "ancho": ancho,
        "alto": alto,
        "frames": len(frames),
        "fps": fps,
        "padding_roi": args.padding,
        "resolucion_interna": args.resolucion_interna,
        "persona_de_frente": not args.de_espaldas,
        "ids_objeto": sorted({d.id_sam3 for d in cadena.detecciones}) if cadena else [],
        "eventos": eventos,
        "linea_temporal": frames_json,
    }
    (destino / "seguimiento.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
        args.salida, "cutie", video.stem, "Cutie", destino, metricas,
        objetos_n, desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cutie: propaga la máscara del objeto en la mano."
    )
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--salida", type=Path, default=Path("salidas"))
    parser.add_argument("--repo", type=Path, default=CUTIE_RAIZ)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--umbral-persona", type=float, default=0.6)
    parser.add_argument(
        "--resolucion-interna",
        type=int,
        default=720,
        help="Lado menor al que Cutie reduce el frame. 480 es el valor del paper; "
        "-1 conserva la resolución original. Con objetos pequeños, 720 aguanta mejor.",
    )
    parser.add_argument("--area-min", type=int, default=80, help="Píxeles mínimos para dar la máscara por válida.")
    parser.add_argument(
        "--hueco-resembra",
        type=int,
        default=3,
        help="Frames seguidos sin máscara antes de volver a sembrar con el detector. -1 no resiembra.",
    )
    parser.add_argument("--dilatacion-mano", type=int, default=28)
    parser.add_argument("--contacto-min", type=float, default=0.08)
    parser.add_argument("--roi-min", type=float, default=0.5)
    parser.add_argument("--hueco-max", type=int, default=45)
    parser.add_argument("--hueco-parpadeo", type=int, default=3)
    parser.add_argument("--distancia-max", type=float, default=250.0)
    parser.add_argument("--de-espaldas", action="store_true")
    parser.add_argument("--umbral-agarre", type=float, default=0.35)
    parser.add_argument("--frames-traspaso", type=int, default=3)
    parser.add_argument("--modelo-pose", type=Path, default=ruta_pose_por_defecto())
    parser.add_argument("--sin-pose", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    if not torch.cuda.is_available():
        raise SystemExit("Cutie en este script necesita una GPU CUDA.")
    videos = listar_videos(args.videos)
    if not videos:
        raise SystemExit(f"No hay vídeos en {args.videos}")
    red = cargar_red(args.repo)
    for video in videos:
        procesar(red, video, args, args.salida)


if __name__ == "__main__":
    main()
