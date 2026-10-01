#!/usr/bin/env python3
"""Cutie: propaga la máscara del objeto detectado en la muñeca.

La detección (muñecas, objetos, contenedores y el JSON) es la del pipeline
común. Cutie solo sigue cada máscara semilla frame a frame.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from detector_palma.geometria import caja_desde_mascara
from pipeline_manos import (
    Objeto,
    PESOS,
    anadir_argumentos_comunes,
    cargar_detectores,
    comprobar_entorno,
    componente_cercana,
    fusionar_objetos,
    procesar_video,
    puntuar_manos,
    resolver_videos,
    trazar,
)

RAIZ = Path(__file__).resolve().parent
CUTIE_RAIZ = RAIZ / "third_party" / "Cutie"


def cargar_red(repo: Path):
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
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
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


def _propagar_una(red, frames, pista: dict, contextos, args) -> Objeto:
    objeto = Objeto(id=pista["id"], semillas=pista["mascaras"])
    frame0 = pista["primer_frame"]
    mascara0 = pista["mascaras"][frame0]
    alto, ancho = frames[0].shape[:2]
    dispositivo = next(red.parameters()).device
    procesador = nuevo_procesador(red, args.resolucion_interna)
    area_ref = max(args.area_min, int(mascara0.sum()))
    perdidos = 0
    sin_mano = 0
    print(f"  Objeto {objeto.id}: siembra en el frame {frame0} ({int(mascara0.sum())} px)")

    with torch.inference_mode():
        for idx in range(frame0, len(frames)):
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            tensor = to_tensor(rgb).to(dispositivo).float()
            sembrar = idx == frame0
            if not sembrar and args.hueco_resembra >= 0 and perdidos >= args.hueco_resembra:
                nueva = pista["mascaras"].get(idx)
                if nueva is not None:
                    procesador = nuevo_procesador(red, args.resolucion_interna)
                    mascara0 = nueva
                    area_ref = max(area_ref, int(nueva.sum()))
                    sembrar = True
                    perdidos = 0
                    objeto.resiembras += 1
            if sembrar:
                etiquetado = np.zeros((alto, ancho), dtype=np.int64)
                etiquetado[mascara0] = 1
                prob = procesador.step(
                    tensor, torch.from_numpy(etiquetado).to(dispositivo), objects=[1]
                )
            else:
                prob = procesador.step(tensor)
            etiquetas = procesador.output_prob_to_mask(prob).detach().cpu().numpy()
            area_max = max(4000, area_ref * 4)
            util = componente_cercana(
                etiquetas == 1, objeto.centro, args.area_min, area_max, args.salto_max
            )
            if util is None:
                perdidos += 1
                continue
            objeto.mascaras[idx] = util
            ys, xs = np.nonzero(util)
            objeto.centro = (float(xs.mean()), float(ys.mean()))
            objeto.ultimo_frame = idx
            perdidos = 0
            caja = caja_desde_mascara(util)
            en_mano = (
                caja is not None
                and puntuar_manos(util, caja, contextos[idx], args)["mano"] is not None
            )
            sin_mano = 0 if en_mano else sin_mano + 1
            if sin_mano > args.frames_sin_mano:
                print(f"    se corta en el frame {idx}: {sin_mano} frames sin mano")
                break
    print(f"    {len(objeto.mascaras)} frames con máscara, {objeto.resiembras} resiembras")
    trazar(objeto, contextos, pista, args)
    return objeto


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cutie: objetos en la mano con ROI de muñeca y contenedores de compra."
    )
    anadir_argumentos_comunes(parser)
    parser.add_argument("--repo", type=Path, default=CUTIE_RAIZ)
    parser.add_argument("--resolucion-interna", type=int, default=720)
    parser.add_argument("--hueco-resembra", type=int, default=3)
    parser.add_argument("--salto-max", type=float, default=120.0)
    parser.add_argument("--frames-sin-mano", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    if not torch.cuda.is_available():
        raise SystemExit("Cutie en este script necesita una GPU CUDA.")
    red = cargar_red(args.repo)
    detectores = cargar_detectores(args)

    def seguir(frames, pistas, contextos, local_args):
        objetos = []
        for pista in pistas:
            objeto = _propagar_una(red, frames, pista, contextos, local_args)
            if objeto.mascaras:
                objetos.append(objeto)
        return fusionar_objetos(objetos)

    for video in resolver_videos(args):
        procesar_video(
            detectores, video, args, "cutie", "Cutie",
            "Cutie", seguir,
            extra={"resolucion_interna": args.resolucion_interna},
        )


if __name__ == "__main__":
    main()
