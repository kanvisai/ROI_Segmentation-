#!/usr/bin/env python3
"""YOLO26-seg: el objeto en la mano se detecta en el recorte de cada muñeca.

No hay propagador de vídeo: cada frame se redetecta. Los contenedores salen
de YOLOE (YOLO26-m de vocabulario abierto). La salida es la del pipeline común.
"""

from __future__ import annotations

import argparse

import torch

from pipeline_manos import (
    anadir_argumentos_comunes,
    cargar_detectores,
    comprobar_entorno,
    procesar_video,
    resolver_videos,
    seguir_por_deteccion,
)


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-seg: objetos en la mano con ROI de muñeca y contenedores."
    )
    anadir_argumentos_comunes(parser)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    if not torch.cuda.is_available():
        raise SystemExit("Este script necesita una GPU CUDA.")
    detectores = cargar_detectores(args)
    for video in resolver_videos(args):
        procesar_video(
            detectores, video, args, "yolo26", "YOLO26-seg",
            "YOLO26-seg", seguir_por_deteccion,
        )


if __name__ == "__main__":
    main()
