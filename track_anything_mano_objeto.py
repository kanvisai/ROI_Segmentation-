#!/usr/bin/env python3
"""Track-Anything: propaga con XMem la máscara detectada en la muñeca.

La detección (muñecas, objetos, contenedores y el JSON) es la del pipeline
común. XMem solo sigue cada máscara semilla.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from detector_palma.geometria import caja_desde_mascara
from pipeline_manos import (
    Objeto,
    anadir_argumentos_comunes,
    cargar_detectores,
    comprobar_entorno,
    componente_cercana,
    fusionar_objetos,
    procesar_lote,
    puntuar_manos,
    resolver_videos,
    trazar,
)

RAIZ = Path(__file__).resolve().parent
REPO_POR_DEFECTO = RAIZ / "third_party" / "Track-Anything"
PESOS_POR_DEFECTO = REPO_POR_DEFECTO / "checkpoints" / "XMem-s012.pth"


class SeguidorXMem:
    """Un vídeo, un objeto. La memoria larga solo se vacía al cambiar de vídeo."""

    def __init__(self, network, config, transform, device: str):
        from inference.inference_core import InferenceCore
        from tracker.util.mask_mapper import MaskMapper

        self.network = network
        self.config = config
        self.transform = transform
        self.device = device
        self._InferenceCore = InferenceCore
        self._MaskMapper = MaskMapper
        self.core = None
        self.mapper = None
        self.reiniciar()

    def reiniciar(self) -> None:
        self.core = self._InferenceCore(self.network, self.config)
        self.mapper = self._MaskMapper()

    def _indice(self, probs) -> np.ndarray:
        out = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)
        final = np.zeros_like(out)
        for origen, interno in self.mapper.remappings.items():
            final[out == interno] = origen
        return final

    @torch.no_grad()
    def track(self, frame_rgb: np.ndarray, anotacion: np.ndarray | None = None, memorizar: bool = True) -> np.ndarray:
        if anotacion is not None and not self.mapper.labels:
            mask, labels = self.mapper.convert_mask(anotacion.astype(np.uint8))
            mask = mask.to(self.device)
            self.core.set_all_labels(list(self.mapper.remappings.values()))
        elif anotacion is not None:
            return self._fijar(frame_rgb, anotacion > 0)
        else:
            mask = None
            labels = None
        anterior = self.core.mem_every
        if not memorizar:
            self.core.mem_every = 10**9
        try:
            tensor = self.transform(frame_rgb).to(self.device)
            probs, _ = self.core.step(tensor, mask, labels)
        finally:
            self.core.mem_every = anterior
        return self._indice(probs)

    @torch.no_grad()
    def _fijar(self, frame_rgb: np.ndarray, binaria: np.ndarray) -> np.ndarray:
        """Escribe una máscara buena en la memoria sin borrar el histórico."""
        alto, ancho = binaria.shape[:2]
        uno = torch.zeros((1, alto, ancho), device=self.device)
        sel = torch.from_numpy(np.ascontiguousarray(binaria.astype(bool))).to(self.device)
        uno[0][sel] = 1
        tensor = self.transform(frame_rgb).to(self.device)
        labels = list(self.core.all_labels)
        probs, _ = self.core.step(tensor, uno, labels)
        return self._indice(probs)


def cargar_seguidor(repo: Path, checkpoint: Path) -> SeguidorXMem:
    repo = repo.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    if not (repo / "tracker" / "model" / "network.py").is_file():
        raise SystemExit(
            f"No está Track-Anything en {repo}.\n"
            "  git clone --depth 1 https://github.com/gaomingqi/Track-Anything.git "
            "third_party/Track-Anything"
        )
    if not checkpoint.is_file():
        raise SystemExit(
            f"No está el checkpoint de XMem en {checkpoint}.\n"
            "  Descárgalo de https://github.com/hkchengrex/XMem/releases (XMem-s012.pth)"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    tracker = str(repo / "tracker")
    if tracker not in sys.path:
        sys.path.insert(0, tracker)

    import yaml
    from model.network import XMem
    from torchvision import transforms
    from tracker.util.range_transform import im_normalization

    with (repo / "tracker" / "config" / "config.yaml").open(encoding="utf-8") as fichero:
        config = yaml.safe_load(fichero)
    print(f"Cargando XMem desde {checkpoint.name}")
    network = XMem(config, str(checkpoint)).cuda().eval()
    transform = transforms.Compose([transforms.ToTensor(), im_normalization])
    return SeguidorXMem(network, config, transform, "cuda")


def _escalar(frame_rgb: np.ndarray, mascara: np.ndarray | None, lado: int):
    alto, ancho = frame_rgb.shape[:2]
    if lado <= 0 or min(alto, ancho) <= lado:
        return frame_rgb, mascara, (alto, ancho)
    escala = lado / min(alto, ancho)
    nh, nw = int(round(alto * escala)), int(round(ancho * escala))
    frame_p = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_p = None
    if mascara is not None:
        mask_p = cv2.resize(
            mascara.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return frame_p, mask_p, (alto, ancho)


def _propagar_una(seguidor: SeguidorXMem, frames, pista: dict, contextos, args) -> Objeto:
    objeto = Objeto(id=pista["id"], semillas=pista["mascaras"])
    frame0 = pista["primer_frame"]
    mascara0 = pista["mascaras"][frame0]
    seguidor.reiniciar()
    perdidos = 0
    desde_mem = 0
    area_ref = max(args.area_min, int(mascara0.sum()))
    print(f"  XMem siembra el objeto {objeto.id} en el frame {frame0}")

    for idx in range(frame0, len(frames)):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        sembrar = idx == frame0
        if not sembrar and args.hueco_resembra >= 0 and perdidos >= args.hueco_resembra:
            nueva = pista["mascaras"].get(idx)
            if nueva is not None:
                mascara0 = nueva
                area_ref = max(area_ref, int(nueva.sum()))
                sembrar = True
                perdidos = 0
                objeto.resiembras += 1
                desde_mem = 0
        if sembrar:
            frame_p, mask_p, tam_orig = _escalar(rgb, mascara0, args.resolucion_interna)
            anotacion = np.zeros(mask_p.shape[:2], dtype=np.uint8)
            anotacion[mask_p.astype(bool)] = 1
            etiquetas = seguidor.track(frame_p, anotacion, memorizar=True)
        else:
            frame_p, _, tam_orig = _escalar(rgb, None, args.resolucion_interna)
            etiquetas = seguidor.track(frame_p, memorizar=False)
            desde_mem += 1
        if etiquetas.shape[:2] != tam_orig:
            etiquetas = cv2.resize(
                etiquetas, (tam_orig[1], tam_orig[0]), interpolation=cv2.INTER_NEAREST
            )
        area_max = max(2500, area_ref * 3)
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
        if not en_mano:
            pass
        if not sembrar and desde_mem >= 4:
            comp = cv2.resize(
                util.astype(np.uint8),
                (frame_p.shape[1], frame_p.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            anotacion = np.zeros(comp.shape, dtype=np.uint8)
            anotacion[comp.astype(bool)] = 1
            seguidor.track(frame_p, anotacion, memorizar=True)
            desde_mem = 0
    print(f"    {len(objeto.mascaras)} frames con máscara, {objeto.resiembras} resiembras")
    trazar(objeto, contextos, pista, args)
    return objeto



def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track-Anything (XMem): objetos en la mano con ROI de muñeca y contenedores."
    )
    anadir_argumentos_comunes(parser)
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    parser.add_argument("--pesos", type=Path, default=PESOS_POR_DEFECTO)
    parser.add_argument("--resolucion-interna", type=int, default=1080)
    parser.add_argument("--hueco-resembra", type=int, default=8)
    parser.add_argument("--salto-max", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    if not torch.cuda.is_available():
        raise SystemExit("Track-Anything en este script necesita una GPU CUDA.")
    seguidor = cargar_seguidor(args.repo, args.pesos)
    detectores = cargar_detectores(args)

    def seguir(frames, pistas, contextos, local_args):
        objetos = []
        for pista in pistas:
            try:
                objeto = _propagar_una(seguidor, frames, pista, contextos, local_args)
            except Exception as exc:
                print(f"  Objeto {pista.get('id')}: Track-Anything falló ({exc})")
                continue
            if objeto.mascaras:
                objetos.append(objeto)
        return fusionar_objetos(objetos)

    fallos = procesar_lote(
        detectores, resolver_videos(args), args, "track_anything", "Track-Anything",
        "Track-Anything (XMem)", seguir,
        extra={"resolucion_interna": args.resolucion_interna, "propagador": "XMem"},
    )
    if fallos:
        raise SystemExit(fallos)


if __name__ == "__main__":
    main()
