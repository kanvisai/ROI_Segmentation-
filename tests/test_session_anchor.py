from types import SimpleNamespace

import numpy as np

from kanvis_segmentation.session_anchor import (
    SessionRoiAnchor,
    evaluate_sam3_wrist_gate,
    filter_contexto_to_session_person,
    pose_row_index,
)


def _persona(x, y, pid):
    kp = np.zeros((17, 3), dtype=float)
    kp[9] = (x, y, 0.9)
    kp[10] = (x + 8, y, 0.9)
    return SimpleNamespace(keypoints=kp, caja=(x - 20, y - 40, x + 20, y + 40), id=pid)


def test_filter_picks_closest_wrist():
    target = _persona(100, 200, 0)
    other = _persona(400, 200, 1)
    ctx = SimpleNamespace(
        personas=[other, target],
        manos=[
            SimpleNamespace(persona_id=0, lado="izquierda"),
            SimpleNamespace(persona_id=1, lado="derecha"),
        ],
    )
    poses = np.zeros((1, 8, 2), dtype=float)
    poses[0, 4] = (0.10, 0.20)  # normalized → 100, 200 at 1000x1000
    poses[0, 5] = (0.108, 0.20)
    anchor = SessionRoiAnchor(
        poses_by_chunk={"chunk_001": poses},
        frames_origin=[0],
        frame_chunk_names=["chunk_001"],
        width=1000,
        height=1000,
        wrist_match_max_px=140.0,
        frames_meta=[{"left_wrist_visible": True, "right_wrist_visible": True}],
    )
    filter_contexto_to_session_person(ctx, anchor)
    assert len(ctx.personas) == 1
    assert ctx.personas[0] is target
    assert all(m.persona_id == 0 for m in ctx.manos)


def test_wrist_gate():
    meta = [{"left_wrist_visible": True}] * 2 + [{"left_wrist_visible": False}] * 8
    ok, vis, total = evaluate_sam3_wrist_gate(meta, min_frames=3, min_fraction=0.10)
    assert vis == 2
    assert total == 10
    assert ok is False
    ok2, vis2, _ = evaluate_sam3_wrist_gate(
        [{"left_wrist_visible": True}] * 4, min_frames=3, min_fraction=0.10
    )
    assert ok2 is True and vis2 == 4


def test_pose_row_index_uses_source_index():
    assert pose_row_index(7, {"source_index": 28}) == 28
    assert pose_row_index(7, {"file": "frame_007.jpg"}) == 7
    assert pose_row_index(3, None) == 3


def test_nan_c1_wrist_is_skipped():
    poses = np.zeros((1, 8, 2), dtype=float)
    poses[0, 4] = (np.nan, np.nan)
    poses[0, 5] = (0.50, 0.40)
    anchor = SessionRoiAnchor(
        poses_by_chunk={"chunk_002": poses},
        frames_origin=[0],
        frame_chunk_names=["chunk_002"],
        width=1000,
        height=1000,
    )
    pts = anchor.session_wrists_px(0)
    assert len(pts) == 1
    assert pts[0] == (500.0, 400.0)


def test_keeps_single_person_when_c1_mismatch():
    subject = _persona(100, 200, 0)
    ctx = SimpleNamespace(personas=[subject], manos=[SimpleNamespace(persona_id=0, lado="derecha")])
    poses = np.zeros((1, 8, 2), dtype=float)
    poses[0, 4] = (0.80, 0.80)
    poses[0, 5] = (0.82, 0.80)
    anchor = SessionRoiAnchor(
        poses_by_chunk={"chunk_002": poses},
        frames_origin=[0],
        frame_chunk_names=["chunk_002"],
        width=1000,
        height=1000,
        wrist_match_max_px=140.0,
        frames_meta=[{"left_wrist_visible": True, "right_wrist_visible": True}],
    )
    filter_contexto_to_session_person(ctx, anchor)
    assert len(ctx.personas) == 1
    assert ctx.personas[0] is subject
    assert ctx.manos


def test_clears_multi_person_when_c1_mismatch():
    a = _persona(100, 200, 0)
    b = _persona(400, 200, 1)
    ctx = SimpleNamespace(
        personas=[a, b],
        manos=[
            SimpleNamespace(persona_id=0, lado="izquierda"),
            SimpleNamespace(persona_id=1, lado="derecha"),
        ],
    )
    poses = np.zeros((1, 8, 2), dtype=float)
    poses[0, 4] = (0.80, 0.80)
    poses[0, 5] = (0.82, 0.80)
    anchor = SessionRoiAnchor(
        poses_by_chunk={"chunk_002": poses},
        frames_origin=[0],
        frame_chunk_names=["chunk_002"],
        width=1000,
        height=1000,
        wrist_match_max_px=140.0,
        frames_meta=[{"left_wrist_visible": True, "right_wrist_visible": True}],
    )
    filter_contexto_to_session_person(ctx, anchor)
    assert ctx.personas == []
    assert ctx.manos == []
