"""Run the existing SAM3 ROI pipeline on a local session layout or in-memory frames."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, load_settings
from .contracts import (
    rpc_status_from_tracking,
    slim_tracking_payload_for_storage,
    summarize_tracking_for_c3,
)
from .session_anchor import (
    PersonFilterInstall,
    SessionRoiAnchor,
    evaluate_sam3_wrist_gate,
    load_frames_meta,
    load_poses,
    pose_row_index,
)

IMAGEN_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class SegmentationResult:
    status: str
    tracking: dict[str, Any]
    review: dict[str, Any]
    error_message: str = ""
    tracking_ref: str = ""


def _fps_from_meta(chunk_dir: Path, default: int = 12) -> int:
    meta = chunk_dir / "meta.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            for key in ("fps", "video_fps", "sample_fps"):
                if data.get(key):
                    return max(1, int(float(data[key])))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return default


def load_chunk_frames(chunk_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    from pipeline_manos import cargar_carpeta_frames

    frames_dir = chunk_dir / "frames"
    if frames_dir.is_dir():
        return cargar_carpeta_frames(frames_dir)
    return cargar_carpeta_frames(chunk_dir)


def load_local_bundle(
    session_root: Path,
    chunk_stems: tuple[str, ...],
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    frames: list[np.ndarray] = []
    origin: list[int] = []
    chunk_names: list[str] = []
    metas: list[dict] = []
    poses_by_chunk: dict[str, np.ndarray] = {}
    fps = 12
    width = height = 0
    present = []
    for stem in chunk_stems:
        chunk_dir = session_root / stem
        if not chunk_dir.is_dir():
            continue
        present.append(stem)
        fr, orig = load_chunk_frames(chunk_dir)
        if not fr:
            continue
        fps = _fps_from_meta(chunk_dir, fps)
        meta_rows = load_frames_meta(chunk_dir)
        poses = load_poses(chunk_dir)
        if poses is not None:
            poses_by_chunk[stem] = poses
        for i, img in enumerate(fr):
            frames.append(img)
            meta_row = meta_rows[i] if i < len(meta_rows) else {}
            origin.append(pose_row_index(int(orig[i]) if i < len(orig) else i, meta_row))
            chunk_names.append(stem)
            metas.append(meta_row)
        height, width = fr[0].shape[:2]
    if not frames:
        raise FileNotFoundError(f"no frames in {session_root} chunks={chunk_stems}")
    return {
        "frames": frames,
        "frames_origin": origin,
        "frame_chunk_names": chunk_names,
        "frames_meta": metas,
        "poses_by_chunk": poses_by_chunk,
        "fps": fps,
        "width": width,
        "height": height,
        "chunks": present,
        "wrist_match_max_px": settings.wrist_match_max_px,
    }


def skipped_payload(*, reason: str, hands_visible: bool = False) -> dict[str, Any]:
    return {
        "final_state": "no_hands",
        "hands_visible": hands_visible,
        "session_person_anchor": "wrist_c1",
        "skip_reason": reason,
        "objects": [],
        "containers": [],
    }


def _make_args(settings: Settings, salida: Path) -> argparse.Namespace:
    from pipeline_manos import anadir_argumentos_comunes

    parser = argparse.ArgumentParser(add_help=False)
    anadir_argumentos_comunes(parser)
    parser.set_defaults(
        min_detecciones=3,
        umbral_agarre=0.35,
        contacto_min=0.12,
        area_max_px=5000,
        area_max_persona=0.02,
        conf_objeto_crop=0.15,
        factor_palma=0.85,
        distancia_max=280.0,
        relacion_max=4.5,
        summary=bool(settings.sam3_summary),
        sin_frames=True,
        mediapipe=bool(settings.mediapipe),
        identificadores=settings.identifiers_yaml,
        salida=salida,
        ancla_distancia=float(settings.wrist_match_max_px),
        ancla_persona="auto",
    )
    return parser.parse_args([])


def run_sam3_on_bundle(
    bundle: dict[str, Any],
    settings: Settings | None = None,
    work_dir: Path | None = None,
    detectores=None,
) -> SegmentationResult:
    settings = settings or load_settings()
    work_dir = Path(work_dir or settings.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    salida = work_dir / "out"
    if salida.exists():
        shutil.rmtree(salida, ignore_errors=True)
    salida.mkdir(parents=True, exist_ok=True)

    metas = bundle.get("frames_meta") or []
    ok_gate, visibles, total = evaluate_sam3_wrist_gate(
        metas,
        min_frames=settings.min_wrist_visible_frames,
        min_fraction=settings.min_wrist_visible_fraction,
        umbral=settings.min_wrist_conf,
    )
    if metas and not ok_gate:
        tracking = skipped_payload(
            reason=f"wrist_gate visibles={visibles}/{total}",
            hands_visible=False,
        )
        tracking = slim_tracking_payload_for_storage(tracking)
        return SegmentationResult(
            status="skipped",
            tracking=tracking,
            review=summarize_tracking_for_c3(tracking),
            error_message=tracking["skip_reason"],
        )

    from pipeline_manos import cargar_detectores, procesar_video, seguir_por_deteccion

    args = _make_args(settings, salida)
    args.fps = int(bundle.get("fps") or 12)
    args.width = int(bundle.get("width") or 0) or None
    args.height = int(bundle.get("height") or 0) or None

    own_detectors = detectores is None
    if detectores is None:
        detectores = cargar_detectores(args)
    escena, _detector, _sam, _idents = detectores
    anchor = SessionRoiAnchor(
        poses_by_chunk=bundle.get("poses_by_chunk") or {},
        frames_origin=list(bundle.get("frames_origin") or []),
        frame_chunk_names=list(bundle.get("frame_chunk_names") or []),
        width=int(bundle.get("width") or 0),
        height=int(bundle.get("height") or 0),
        wrist_match_max_px=float(bundle.get("wrist_match_max_px") or settings.wrist_match_max_px),
        frames_meta=list(metas),
    )
    hook = PersonFilterInstall(escena, anchor)
    if anchor.poses_by_chunk:
        hook.install()
    try:
        ok = procesar_video(
            detectores,
            Path("session"),
            args,
            "sam3",
            "SAM 3",
            "SAM 3",
            seguir_por_deteccion,
            frames=bundle["frames"],
            frames_origin=bundle["frames_origin"],
            fps=int(bundle.get("fps") or 12),
            width=bundle.get("width"),
            height=bundle.get("height"),
        )
    finally:
        hook.uninstall()

    if not ok:
        return SegmentationResult(
            status="error",
            tracking={},
            review=summarize_tracking_for_c3({}),
            error_message="procesar_video failed",
        )
    tracking_path = _find_tracking(salida)
    if tracking_path is None:
        return SegmentationResult(
            status="error",
            tracking={},
            review=summarize_tracking_for_c3({}),
            error_message="tracking.json was not written",
        )
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    tracking["session_person_anchor"] = "wrist_c1" if anchor.poses_by_chunk else tracking.get(
        "session_person_anchor"
    )
    tracking = slim_tracking_payload_for_storage(tracking)
    status = rpc_status_from_tracking(tracking)
    return SegmentationResult(
        status=status,
        tracking=tracking,
        review=summarize_tracking_for_c3(tracking),
    )


def _find_tracking(salida: Path) -> Path | None:
    hits = list(salida.rglob("tracking.json"))
    return hits[0] if hits else None


def run_local_session(
    session_root: Path,
    chunk_stem: str,
    span_spec: str = "single",
    settings: Settings | None = None,
    work_dir: Path | None = None,
    detectores=None,
) -> SegmentationResult:
    from .refs import parse_span_spec

    settings = settings or load_settings()
    stems = parse_span_spec(span_spec, chunk_stem)
    bundle = load_local_bundle(Path(session_root), stems, settings)
    return run_sam3_on_bundle(bundle, settings, work_dir=work_dir, detectores=detectores)
