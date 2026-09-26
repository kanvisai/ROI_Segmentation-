#!/usr/bin/env python3
"""Track-Anything: propaga la máscara del objeto en la mano con XMem.

Track-Anything une Segment Anything y XMem. Aquí la máscara inicial no sale
de clics: la pone el detector de palma (YOLO + MobileSAM) y XMem la sigue.
Si la pierde durante varios frames, el detector corrige la máscara sin vaciar
la memoria larga de XMem. Una máscara hinchada no se guarda.

La persona y el ROI salen de Faster R-CNN. La mano, del esqueleto YOLO-pose.
La salida es la misma que en SAM 3, HOIST y Cutie.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from cutie_mano_objeto import _componente_cercana, _elegir_semilla, _mascara_id
from hoist_mano_objeto import asociar, detectar_personas, leer_frames
from salida_comun import Medidor, carpeta_de, publicar_comparativa
from sam3_mano_objeto import (
    analizar_pose,
    anotar_video,
    comprobar_entorno,
    describir_eventos,
    detecciones_palma,
    encadenar,
    info_video,
    linea_temporal,
    listar_videos,
    quitar_parpadeos,
    ruta_pose_por_defecto,
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


def propagar(seguidor: SeguidorXMem, frames, por_id, por_frame, resolucion, area_min, hueco_resembra):
    semilla = _elegir_semilla(por_id, por_frame)
    if semilla is None:
        return []
    oid, frame0, mascara0 = semilla
    seguidor.reiniciar()
    mascaras: dict[int, np.ndarray] = {}
    perdidos = 0
    resiembras = 0
    desde_mem = 0
    centro = None
    area_ref = max(area_min, int(mascara0.sum()))
    print(f"  XMem siembra el objeto {oid} en el frame {frame0} (lado interno {resolucion})")

    for idx in range(frame0, len(frames)):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        sembrar = idx == frame0
        if not sembrar and hueco_resembra >= 0 and perdidos >= hueco_resembra:
            nueva = _mascara_id(por_frame, idx, oid)
            if nueva is not None:
                mascara0 = nueva
                area_ref = max(area_ref, int(nueva.sum()))
                sembrar = True
                perdidos = 0
                resiembras += 1
                desde_mem = 0
        if sembrar:
            frame_p, mask_p, tam_orig = _escalar(rgb, mascara0, resolucion)
            anotacion = np.zeros(mask_p.shape[:2], dtype=np.uint8)
            anotacion[mask_p.astype(bool)] = 1
            etiquetas = seguidor.track(frame_p, anotacion, memorizar=True)
        else:
            frame_p, _, tam_orig = _escalar(rgb, None, resolucion)
            etiquetas = seguidor.track(frame_p, memorizar=False)
            desde_mem += 1
        if etiquetas.shape[:2] != tam_orig:
            etiquetas = cv2.resize(
                etiquetas, (tam_orig[1], tam_orig[0]), interpolation=cv2.INTER_NEAREST
            )
        area_max = max(2500, area_ref * 3)
        util = _componente_cercana(etiquetas == 1, centro, area_min, area_max)
        if util is not None:
            mascaras[idx] = util
            ys, xs = np.nonzero(util)
            centro = (float(xs.mean()), float(ys.mean()))
            perdidos = 0
            area_vista = int(util.sum())
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
        else:
            perdidos += 1
            area_vista = int((etiquetas == 1).sum())
        if idx == frame0 or (idx - frame0) % 20 == 0:
            print(f"    frame {idx}: {area_vista} px")

    print(f"  XMem: {len(mascaras)} frames con máscara, {resiembras} resiembras")
    if not mascaras:
        return []
    return [{"id": oid, "score": 1.0, "mascaras": mascaras}]


def resumen(video: Path, eventos, ultima, nframes: int) -> str:
    lineas = [
        f"Vídeo: {video}",
        f"Frames: {nframes}",
        "Modelo: Track-Anything (XMem). La máscara inicial sale del detector de palma.",
        "",
    ]
    if not eventos and ultima is None:
        lineas.append("XMem no dejó ningún objeto dentro del ROI de la persona.")
        return "\n".join(lineas) + "\n"
    lineas.extend(describir_eventos(eventos, ultima))
    lineas.append(
        "\nIzquierda y derecha salen del esqueleto de la persona. "
        "Un cambio de mano solo se anota si se mantiene varios frames. "
        "Suelto significa que el objeto sigue en escena pero ya no está en la mano."
    )
    return "\n".join(lineas) + "\n"


def procesar(seguidor, video: Path, args, salida: Path) -> None:
    ancho, alto, _, fps_archivo = info_video(video)
    frames, fps = leer_frames(video, args.max_frames)
    if args.max_frames is None:
        fps = fps_archivo or fps
    print(f"\n=== {video.name}: {ancho}x{alto}, {len(frames)} frames, {fps:.3f} fps ===")
    destino = carpeta_de(salida, "track_anything", video)
    medidor = Medidor("track_anything")
    try:
        _procesar(seguidor, video, args, destino, medidor, frames, fps, ancho, alto)
    except Exception as exc:
        publicar_comparativa(
            salida, "track_anything", video.stem, "Track-Anything", destino,
            medidor.cerrar(len(frames)), 0, 0, 0, ok=False, codigo=1, error=str(exc),
        )
        raise


def _procesar(seguidor, video, args, destino, medidor, frames, fps, ancho, alto) -> None:
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
                seguidor, frames, por_id_palma, por_frame_palma,
                args.resolucion_interna, args.area_min, args.hueco_resembra,
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
        "modelo": "Track-Anything",
        "propagador": "XMem",
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
        args.salida, "track_anything", video.stem, "Track-Anything", destino, metricas,
        objetos_n, desapariciones, frames_con_objeto,
    )
    print(texto)
    print(f"  Comparativa: {ruta}")


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track-Anything (XMem): propaga la máscara del objeto en la mano."
    )
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--salida", type=Path, default=Path("salidas"))
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    parser.add_argument("--pesos", type=Path, default=PESOS_POR_DEFECTO)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--umbral-persona", type=float, default=0.6)
    parser.add_argument(
        "--resolucion-interna",
        type=int,
        default=1080,
        help="Lado menor del frame que ve XMem. 1080 conserva el objeto pequeño; 480 es el yaml oficial. -1 deja el frame original.",
    )
    parser.add_argument("--area-min", type=int, default=80)
    parser.add_argument(
        "--hueco-resembra",
        type=int,
        default=8,
        help="Frames seguidos sin máscara válida antes de corregir con el detector, sin vaciar la memoria.",
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
        raise SystemExit("Track-Anything en este script necesita una GPU CUDA.")
    videos = listar_videos(args.videos)
    if not videos:
        raise SystemExit(f"No hay vídeos en {args.videos}")
    seguidor = cargar_seguidor(args.repo, args.pesos)
    for video in videos:
        procesar(seguidor, video, args, args.salida)


if __name__ == "__main__":
    main()
