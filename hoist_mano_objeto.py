#!/usr/bin/env python3
"""HOIST-Former: segmenta y sigue el objeto que una persona lleva en la mano.

HOIST-Former (CVPR 2024) identifica, segmenta y sigue objetos cogidos.
No segmenta el cuerpo, así que el ROI de la persona sale de Faster R-CNN
(COCO, clase persona) y se amplía con padding. La izquierda y la derecha
salen de YOLO-pose. El agarre se puntúa con la palma y un abanico de dedos.

La salida sigue el mismo esquema que ``sam3_mano_objeto.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline_manos import (
    anadir_argumentos_comunes,
    cargar_detectores,
    procesar_video,
    resolver_videos,
    seguir_por_deteccion,
)
from salida_comun import Medidor, carpeta_de, publicar_comparativa
from sam3_mano_objeto import (
    Deteccion,
    alinear_lados,
    analizar_pose,
    anotar_video,
    caja_de_mascara,
    centro_de_mascara,
    comprobar_entorno,
    describir_eventos,
    dilatar,
    detecciones_palma,
    encadenar,
    evaluar_agarre,
    expandir_caja,
    fraccion_dentro_de_caja,
    info_video,
    lado_anatomico,
    linea_temporal,
    listar_videos,
    pose_en_roi,
    ruta_pose_por_defecto,
)

RAIZ = Path(__file__).resolve().parent
HOIST_RAIZ = RAIZ / "third_party" / "HOISTFormer"
PESOS_POR_DEFECTO = HOIST_RAIZ / "pretrained_models" / "trained_model.pth"


def asegurar_fallback_cuda(raiz: Path) -> None:
    """Si el operador CUDA no está compilado, HOIST usa la versión PyTorch."""
    ruta = raiz / "mask2former/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py"
    if not ruta.is_file():
        return
    texto = ruta.read_text(encoding="utf-8")
    if "MSDA = None" in texto:
        return
    antiguo = (
        "except ModuleNotFoundError as e:\n"
        "    info_string = (\n"
        '        "\\n\\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\\n"\n'
        '        "\\t`cd mask2former/modeling/pixel_decoder/ops`\\n"\n'
        '        "\\t`sh make.sh`\\n"\n'
        "    )\n"
        "    raise ModuleNotFoundError(info_string)\n"
    )
    nuevo = (
        "except ModuleNotFoundError:\n"
        "    # El operador CUDA es opcional: MSDeformAttn cae a la versión PyTorch.\n"
        "    MSDA = None\n"
    )
    if antiguo not in texto:
        return
    texto = texto.replace(antiguo, nuevo, 1)
    ancla = "    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):\n"
    extra = (
        ancla
        + "        if MSDA is None:\n"
        + '            raise RuntimeError("MultiScaleDeformableAttention CUDA op no está compilado")\n'
    )
    if ancla in texto and "MSDA is None" not in texto:
        texto = texto.replace(ancla, extra, 1)
    ruta.write_text(texto, encoding="utf-8")


def preparar_ruta() -> None:
    if not HOIST_RAIZ.is_dir():
        raise SystemExit(
            "No está el código de HOIST-Former.\n"
            "Desde este directorio: "
            "git clone --depth 1 https://github.com/SupreethN/HOISTFormer.git "
            "third_party/HOISTFormer"
        )
    asegurar_fallback_cuda(HOIST_RAIZ)
    ruta = str(HOIST_RAIZ)
    if ruta not in sys.path:
        sys.path.insert(0, ruta)


def cargar_predictor(pesos: Path, lado_corto: int, lado_largo: int):
    preparar_ruta()
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from demo_video.predictor import VideoPredictor
    from mask2former import add_maskformer2_config
    from mask2former_video import add_maskformer2_video_config

    # Registra el nombre de dataset que pide la config, sin descargar HOIST.
    import mask2former_video.data_video.datasets.hoist  # noqa: F401

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(str(HOIST_RAIZ / "configs" / "hoist" / "hoistformer.yaml"))
    cfg.MODEL.WEIGHTS = str(pesos)
    cfg.MODEL.DEVICE = "cuda"
    # El yaml oficial prueba a 360 px de lado corto. En 1080p el objeto
    # de la mano queda demasiado pequeño y la puntuación se hunde.
    cfg.INPUT.MIN_SIZE_TEST = lado_corto
    cfg.INPUT.MAX_SIZE_TEST = lado_largo
    cfg.freeze()
    print(f"Cargando HOIST-Former desde {pesos}")
    predictor = VideoPredictor(cfg)
    _instalar_salida_manos(predictor.model)
    return predictor


def _instalar_salida_manos(modelo) -> None:
    """La demo oficial tira las máscaras de mano. Aquí se conservan las más grandes."""
    from detectron2.structures import ImageList

    def forward(batched_inputs):
        imagenes = []
        for video in batched_inputs:
            for frame in video["image"]:
                imagenes.append(frame.to(modelo.device))
        imagenes = [(x - modelo.pixel_mean) / modelo.pixel_std for x in imagenes]
        imagenes = ImageList.from_tensors(imagenes, modelo.size_divisibility)
        features = modelo.backbone(imagenes.tensor)
        salidas = modelo.sem_seg_head(features)

        logits = salidas["pred_logits"][0]
        mascaras = salidas["pred_masks"][0]
        manos = salidas["hand_pred_masks"][0]
        del salidas

        mascaras = F.interpolate(
            mascaras,
            size=(imagenes.tensor.shape[-2], imagenes.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        entrada = batched_inputs[0]
        tam_sin_pad = imagenes.image_sizes[0]
        alto = entrada.get("height", tam_sin_pad[0])
        ancho = entrada.get("width", tam_sin_pad[1])
        resultado = modelo.inference_video(logits, mascaras, tam_sin_pad, alto, ancho)
        resultado["hand_masks"] = _manos_a_resolucion(
            manos,
            tam_padded=(imagenes.tensor.shape[-2], imagenes.tensor.shape[-1]),
            tam_sin_pad=tam_sin_pad,
            alto=alto,
            ancho=ancho,
            como_maximo=4,
        )
        return resultado

    modelo.forward = forward


def _manos_a_resolucion(logits, tam_padded, tam_sin_pad, alto, ancho, como_maximo: int):
    """logits: [consultas, frames, H, W] en la resolución del pixel decoder."""
    if logits.numel() == 0:
        return []
    area = (logits > 0).flatten(2).sum(-1).sum(-1)
    activas = torch.nonzero(area > 0, as_tuple=False).flatten()
    if activas.numel() == 0:
        return []
    orden = torch.argsort(area[activas], descending=True)[:como_maximo]
    elegidas = logits[activas[orden]]
    elegidas = F.interpolate(
        elegidas, size=tam_padded, mode="bilinear", align_corners=False
    )
    elegidas = elegidas[:, :, : tam_sin_pad[0], : tam_sin_pad[1]]
    elegidas = F.interpolate(
        elegidas, size=(alto, ancho), mode="bilinear", align_corners=False
    )
    return [(elegidas[i] > 0).detach().cpu() for i in range(elegidas.shape[0])]


def leer_frames(video: Path, max_frames: int | None, fps_proceso: int | None = None):
    from pipeline_manos import leer_frames as _leer

    frames, fps, _origen = _leer(video, max_frames, fps_proceso)
    return frames, fps


def detectar_personas(frames: list[np.ndarray], umbral: float) -> list[tuple[int, int, int, int] | None]:
    """Caja de la persona más grande en cada frame. Etiqueta COCO 1."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    print("  Detector de persona (Faster R-CNN)")
    pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos).cuda().eval()
    cajas: list[tuple[int, int, int, int] | None] = []
    lote = 2
    with torch.inference_mode():
        for inicio in range(0, len(frames), lote):
            tensores = []
            for frame in frames[inicio : inicio + lote]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
                tensores.append(tensor.float().div(255.0).cuda())
            for salida in modelo(tensores):
                cajas.append(_mejor_persona(salida, umbral))
    del modelo
    torch.cuda.empty_cache()
    print(f"    persona visible en {sum(c is not None for c in cajas)} / {len(cajas)} frames")
    return cajas


def _mejor_persona(salida, umbral: float):
    etiquetas = salida["labels"].detach().cpu()
    scores = salida["scores"].detach().cpu()
    boxes = salida["boxes"].detach().cpu()
    mejor = None
    mejor_area = 0.0
    for etiqueta, score, caja in zip(etiquetas, scores, boxes):
        if int(etiqueta) != 1 or float(score) < umbral:
            continue
        x0, y0, x1, y1 = (float(v) for v in caja)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area > mejor_area:
            mejor_area = area
            mejor = (int(x0), int(y0), int(x1), int(y1))
    return mejor


def _a_numpy_mascara(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(bool)
    return np.asarray(tensor).astype(bool)


def pistas_de_ventana(prediccion: dict, inicio: int, umbral: float, area_min: int):
    pistas = []
    scores = prediccion.get("pred_scores") or []
    mascaras = prediccion.get("pred_masks") or []
    for consulta, score in enumerate(scores):
        if float(score) < umbral or consulta >= len(mascaras):
            continue
        clip = _a_numpy_mascara(mascaras[consulta])
        if clip.ndim != 3:
            continue
        por_frame = {}
        for t in range(clip.shape[0]):
            mascara = clip[t]
            if int(mascara.sum()) >= area_min:
                por_frame[inicio + t] = mascara
        if por_frame:
            pistas.append({"score": float(score), "mascaras": por_frame})

    manos: dict[int, list[np.ndarray]] = {}
    for consulta in prediccion.get("hand_masks") or []:
        clip = _a_numpy_mascara(consulta)
        if clip.ndim != 3:
            continue
        for t in range(clip.shape[0]):
            mascara = clip[t]
            if int(mascara.sum()) < area_min:
                continue
            manos.setdefault(inicio + t, []).append(mascara)
    return pistas, manos


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    if union <= 0:
        return 0.0
    return inter / union


def coser(globales: list[dict], nuevas: list[dict], iou_min: float, siguiente_id: int) -> int:
    usados_g: set[int] = set()
    usados_n: set[int] = set()
    candidatos = []
    for ig, pista in enumerate(globales):
        for inn, nueva in enumerate(nuevas):
            comunes = set(pista["mascaras"]) & set(nueva["mascaras"])
            if not comunes:
                continue
            media = float(
                np.mean([iou(pista["mascaras"][f], nueva["mascaras"][f]) for f in comunes])
            )
            candidatos.append((media, ig, inn))
    candidatos.sort(reverse=True)
    for media, ig, inn in candidatos:
        if media < iou_min:
            break
        if ig in usados_g or inn in usados_n:
            continue
        usados_g.add(ig)
        usados_n.add(inn)
        for frame, mascara in nuevas[inn]["mascaras"].items():
            globales[ig]["mascaras"].setdefault(frame, mascara)
        globales[ig]["score"] = max(globales[ig]["score"], nuevas[inn]["score"])
    for inn, nueva in enumerate(nuevas):
        if inn in usados_n:
            continue
        globales.append(
            {
                "id": siguiente_id,
                "score": nueva["score"],
                "mascaras": dict(nueva["mascaras"]),
            }
        )
        siguiente_id += 1
    return siguiente_id


def inferir(predictor, frames: list[np.ndarray], ventana: int, solape: int, umbral: float, area_min: int, iou_min: float):
    nframes = len(frames)
    paso = max(1, ventana - solape)
    globales: list[dict] = []
    manos_por_frame: dict[int, list[np.ndarray]] = {}
    siguiente_id = 1
    inicio = 0
    n_ventanas = 0
    while inicio < nframes:
        fin = min(nframes, inicio + ventana)
        print(f"  HOIST-Former frames {inicio}-{fin - 1}")
        try:
            prediccion = predictor(frames[inicio:fin])
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise SystemExit(
                "La GPU se ha quedado sin memoria en HOIST-Former. "
                "Prueba con --ventana 4."
            ) from exc
        pistas, manos = pistas_de_ventana(prediccion, inicio, umbral, area_min)
        for frame, lista in manos.items():
            manos_por_frame.setdefault(frame, lista)
        siguiente_id = coser(globales, pistas, iou_min, siguiente_id)
        n_ventanas += 1
        del prediccion
        torch.cuda.empty_cache()
        if fin == nframes:
            break
        inicio += paso
    print(f"    {n_ventanas} ventanas, {len(globales)} pistas")
    return globales, manos_por_frame


def asociar(
    nframes: int,
    ancho: int,
    alto: int,
    cajas_persona,
    manos_por_frame,
    pistas,
    padding: float,
    dilatacion: int,
    roi_min: float,
    contacto_min: float,
    de_frente: bool,
    manos_pose=None,
    umbral_agarre: float = 0.35,
):
    estado = {}
    por_id: dict[int, list[Deteccion]] = {}
    objetos_vis = {}

    for frame in range(nframes):
        caja = cajas_persona[frame]
        roi = expandir_caja(caja, padding, ancho, alto) if caja is not None else None
        centro_persona = None
        if caja is not None:
            centro_persona = ((caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2)
        cx = centro_persona[0] if centro_persona is not None else None
        candidatas = []
        for mascara in manos_por_frame.get(frame, []):
            centro = centro_de_mascara(mascara)
            if centro is None:
                continue
            if roi is not None and fraccion_dentro_de_caja(mascara, roi) < 0.3:
                dentro = roi[0] <= centro[0] <= roi[2] and roi[1] <= centro[1] <= roi[3]
                if not dentro:
                    continue
            candidatas.append((centro[0], mascara, centro))
        candidatas.sort(key=lambda t: t[0])
        manos = []
        for orden, (_, mascara, centro) in enumerate(candidatas):
            bbox = caja_de_mascara(mascara)
            if bbox is None:
                continue
            manos.append(
                {
                    "id": orden + 1,
                    "lado": lado_anatomico(orden, len(candidatas), centro[0], cx, de_frente),
                    "centro": centro,
                    "bbox": bbox,
                    "mascara": mascara,
                    "dilatada": dilatar(mascara, dilatacion),
                    "prob": 1.0,
                }
            )
        pose_frame = pose_en_roi(
            manos_pose[frame] if manos_pose is not None and frame < len(manos_pose) else None,
            roi,
        )
        alinear_lados(manos, pose_frame)
        estado[frame] = {
            "roi": roi,
            "centro_persona": centro_persona,
            "persona_visible": caja is not None,
            "manos": manos,
            "pose": pose_frame,
        }

    for pista in pistas:
        for frame, mascara in sorted(pista["mascaras"].items()):
            roi = estado[frame]["roi"]
            if roi is not None and fraccion_dentro_de_caja(mascara, roi) < roi_min:
                continue
            centro = centro_de_mascara(mascara)
            bbox = caja_de_mascara(mascara)
            if centro is None or bbox is None:
                continue
            agarre = evaluar_agarre(
                mascara,
                bbox,
                estado[frame]["pose"],
                estado[frame]["manos"],
                umbral_agarre,
            )
            det = Deteccion(
                frame=frame,
                id_sam3=pista["id"],
                centro=centro,
                bbox=bbox,
                prob=pista["score"],
                area=int(mascara.sum()),
                mano=agarre["lado"],
                en_mano=agarre["en_mano"],
                score_agarre=agarre["score"],
                scores_mano=agarre["scores"],
            )
            por_id.setdefault(pista["id"], []).append(det)
            objetos_vis.setdefault(frame, []).append(
                {
                    "id": pista["id"],
                    "mascara": mascara,
                    "prob": pista["score"],
                    "caja_xywh_norm": (0.0, 0.0, 1.0, 1.0),
                }
            )
    for dets in por_id.values():
        dets.sort(key=lambda d: d.frame)
    return estado, por_id, objetos_vis


def quitar_parpadeos(eventos: list[dict], max_hueco: int) -> list[dict]:
    """Un hueco de pocos frames no es una desaparición: el objeto sigue ahí."""
    if max_hueco <= 0:
        return eventos
    salida = []
    i = 0
    while i < len(eventos):
        evento = eventos[i]
        if (
            evento["tipo"] == "desaparece"
            and i + 1 < len(eventos)
            and eventos[i + 1]["tipo"] == "reaparece"
            and eventos[i + 1]["frame"] - evento["frame"] <= max_hueco
        ):
            i += 2
            continue
        salida.append(evento)
        i += 1
    return salida


def resumen(video: Path, eventos: list[dict], ultima: dict | None, nframes: int) -> str:
    lineas = [
        f"Vídeo: {video}",
        f"Frames: {nframes}",
        "Modelo: HOIST-Former. El ROI es la caja de la persona ampliada.",
        "",
    ]
    if not eventos and ultima is None:
        lineas.append(
            "HOIST-Former no dejó ningún objeto dentro del ROI de la persona."
        )
        return "\n".join(lineas) + "\n"
    lineas.extend(describir_eventos(eventos, ultima))
    lineas.append(
        "\nIzquierda y derecha salen del esqueleto de la persona. "
        "Un cambio de mano solo se anota si se mantiene varios frames. "
        "Suelto significa que el objeto sigue en escena pero ya no está en la mano."
    )
    return "\n".join(lineas) + "\n"



def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HOIST-Former: objetos en la mano con ROI de muñeca y contenedores."
    )
    anadir_argumentos_comunes(parser)
    parser.add_argument("--repo", type=Path, default=HOIST_RAIZ)
    parser.add_argument("--pesos", type=Path, default=PESOS_POR_DEFECTO)
    return parser.parse_args()


def main() -> None:
    args = parsear_args()
    comprobar_entorno()
    detectores = cargar_detectores(args)
    for video in resolver_videos(args):
        procesar_video(
            detectores, video, args, "hoist", "HOIST-Former",
            "HOIST-Former", seguir_por_deteccion,
        )
    print(f"\nListo. Resultados en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
