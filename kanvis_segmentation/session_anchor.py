"""C1 session-subject filter for multi-person SAM3 frames.

See SAM3_MULTI_PERSON_SESSION_ANCHOR.md (wrists at poses_full KP 4–5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

MUNECA_IZQ, MUNECA_DER = 9, 10
POSE_WRIST_SLOTS = (4, 5)
YOLO_WRIST_CONF = 0.20


@dataclass
class SessionRoiAnchor:
    poses_by_chunk: dict[str, np.ndarray]
    frames_origin: list[int]
    frame_chunk_names: list[str]
    width: int
    height: int
    wrist_match_max_px: float = 140.0
    frame_idx: int = 0
    frames_meta: list[dict] = field(default_factory=list)

    def session_wrists_px(self, indice: int) -> list[tuple[float, float]]:
        if indice < 0 or indice >= len(self.frames_origin):
            return []
        chunk = self.frame_chunk_names[indice] if indice < len(self.frame_chunk_names) else ""
        poses = self.poses_by_chunk.get(chunk)
        if poses is None:
            return []
        origen = int(self.frames_origin[indice])
        if origen < 0 or origen >= len(poses):
            return []
        fila = poses[origen]
        puntos = []
        for slot in POSE_WRIST_SLOTS:
            if slot >= len(fila):
                continue
            x, y = float(fila[slot][0]), float(fila[slot][1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x == 0.0 and y == 0.0:
                continue
            if max(abs(x), abs(y)) <= 1.5:
                x, y = x * self.width, y * self.height
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            puntos.append((x, y))
        return puntos


def pose_row_index(origin_from_name: int, meta: dict | None) -> int:
    """Map a SAM3 JPEG to its row in C1 poses_full (12 fps vs 3 fps export)."""
    if isinstance(meta, dict):
        src = meta.get("source_index")
        if src is not None and str(src).strip() != "":
            return int(src)
    return int(origin_from_name)


def _munecas_yolo(persona, umbral: float = YOLO_WRIST_CONF) -> list[tuple[float, float]]:
    kp = getattr(persona, "keypoints", None)
    if kp is None or len(kp) <= MUNECA_DER:
        return []
    pts = []
    for indice in (MUNECA_IZQ, MUNECA_DER):
        fila = kp[indice]
        if float(fila[2]) >= umbral:
            x, y = float(fila[0]), float(fila[1])
            if np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))
    return pts


def _puntos_persona(persona, manos, persona_id: int) -> list[tuple[float, float]]:
    pts = _munecas_yolo(persona)
    if pts:
        return pts
    extra: list[tuple[float, float]] = []
    for mano in manos or []:
        if getattr(mano, "persona_id", -1) != persona_id:
            continue
        for attr in ("punto", "palma"):
            p = getattr(mano, attr, None)
            if p is None or len(p) < 2:
                continue
            x, y = float(p[0]), float(p[1])
            if np.isfinite(x) and np.isfinite(y):
                extra.append((x, y))
                break
    return extra


def _persona_wrist_score(
    persona,
    session_pts: list[tuple[float, float]],
    max_px: float,
    puntos: list[tuple[float, float]] | None = None,
) -> float:
    yolo = list(puntos) if puntos is not None else _munecas_yolo(persona)
    if not session_pts:
        return sum(float(kp[2]) for kp in (getattr(persona, "keypoints", None) or [])[9:11]) if getattr(persona, "keypoints", None) is not None and len(persona.keypoints) > 10 else float("-inf")
    if not yolo:
        return float("-inf")
    total = 0.0
    matched = False
    for sx, sy in session_pts:
        dist = min(((px - sx) ** 2 + (py - sy) ** 2) ** 0.5 for px, py in yolo)
        if np.isfinite(dist) and dist <= max_px:
            total += -dist
            matched = True
    return total if matched else float("-inf")


def filter_contexto_to_session_person(contexto, anchor: SessionRoiAnchor) -> None:
    idx = int(anchor.frame_idx)
    session_pts = anchor.session_wrists_px(idx)
    personas = list(getattr(contexto, "personas", None) or [])
    manos = list(getattr(contexto, "manos", None) or [])
    meta = anchor.frames_meta[idx] if 0 <= idx < len(anchor.frames_meta) else {}
    visible = frame_has_wrist_visibility(meta)
    if not personas:
        return
    if not session_pts and not visible:
        contexto.personas = []
        contexto.manos = []
        return
    mejor_i, mejor = None, float("-inf")
    for i, persona in enumerate(personas):
        score = _persona_wrist_score(
            persona,
            session_pts,
            anchor.wrist_match_max_px,
            puntos=_puntos_persona(persona, manos, i),
        )
        if score > mejor:
            mejor, mejor_i = score, i
    if mejor_i is None or mejor == float("-inf"):
        # C1 wrists and ROI pose can disagree (3 fps JPEGs vs 12 fps poses,
        # Holistic vs YOLO). A single remaining subject is the local ancla
        # choice — keep it instead of inventing no_hands.
        if len(personas) <= 1:
            return
        contexto.personas = []
        contexto.manos = []
        return
    contexto.personas = [personas[mejor_i]]
    contexto.manos = [
        m for m in manos
        if getattr(m, "persona_id", -1) == mejor_i
    ]
    for mano in contexto.manos:
        mano.persona_id = 0


def frame_has_wrist_visibility(meta: dict[str, Any], umbral: float = 0.35) -> bool:
    if not meta:
        return False
    if meta.get("left_wrist_visible") or meta.get("right_wrist_visible"):
        return True
    left = float(meta.get("left_wrist_conf") or meta.get("left_wrist_confidence") or 0.0)
    right = float(meta.get("right_wrist_conf") or meta.get("right_wrist_confidence") or 0.0)
    return max(left, right) >= umbral


def evaluate_sam3_wrist_gate(
    frames_meta: list[dict],
    *,
    min_frames: int = 3,
    min_fraction: float = 0.10,
    umbral: float = 0.35,
) -> tuple[bool, int, int]:
    total = max(1, len(frames_meta) if frames_meta else 0)
    visibles = sum(1 for row in frames_meta if frame_has_wrist_visibility(row, umbral))
    if not frames_meta:
        return True, 0, 0
    ok = visibles >= min_frames and (visibles / total) >= min_fraction
    return ok, visibles, total


def load_poses(path: Path) -> np.ndarray | None:
    for name in ("poses_full.npy", "poses_compat.npy", "poses.npy"):
        ruta = path / name
        if ruta.is_file():
            arr = np.load(str(ruta), allow_pickle=False)
            return arr
    return None


def load_frames_meta(chunk_dir: Path) -> list[dict]:
    ruta = chunk_dir / "frames_meta.json"
    if not ruta.is_file():
        return []
    import json

    data = json.loads(ruta.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row if isinstance(row, dict) else {} for row in data]
    if isinstance(data, dict):
        frames = data.get("frames") or data.get("items") or []
        if isinstance(frames, list):
            return [row if isinstance(row, dict) else {} for row in frames]
    return []


class PersonFilterInstall:
    def __init__(self, escena, anchor: SessionRoiAnchor):
        self.escena = escena
        self.anchor = anchor
        self._original: Callable | None = None

    def install(self) -> None:
        self._original = self.escena.analizar

        def _wrapped(frame, *args, **kwargs):
            ctx = self._original(frame, *args, **kwargs)
            filter_contexto_to_session_person(ctx, self.anchor)
            self.anchor.frame_idx += 1
            return ctx

        self.escena.analizar = _wrapped  # type: ignore[method-assign]

    def uninstall(self) -> None:
        if self._original is not None:
            self.escena.analizar = self._original  # type: ignore[method-assign]
            self._original = None
