import numpy as np
import pytest

from kanvis_segmentation.pipeline_runner import load_local_bundle
from kanvis_segmentation.refs import (
    job_from_fields,
    overlap_half_counts,
    overlap_half_indices,
    parse_overlap_half,
)


def test_parse_overlap_half_default_and_ratio():
    left, right, ratio = parse_overlap_half(
        "overlap_half:chunk_001:chunk_002:ratio=0.5000"
    )
    assert (left, right, ratio) == ("chunk_001", "chunk_002", 0.5)
    left, right, ratio = parse_overlap_half("overlap_half:chunk_003:chunk_004")
    assert (left, right, ratio) == ("chunk_003", "chunk_004", 0.5)
    assert parse_overlap_half("with_next") is None
    with pytest.raises(ValueError):
        parse_overlap_half("overlap_half:chunk_001")
    with pytest.raises(ValueError):
        parse_overlap_half("overlap_half:chunk_001:chunk_002:ratio=1.5")


def test_overlap_counts_24_frames_are_12_plus_12():
    k_left, k_right = overlap_half_counts(24, 24, 0.5)
    assert (k_left, k_right) == (12, 12)
    left_idx, right_idx = overlap_half_indices(24, 24, 0.5)
    assert left_idx == list(range(12, 24))
    assert right_idx == list(range(0, 12))


def test_overlap_job_uses_physical_chunks_and_p1_stem():
    job = job_from_fields(
        source_ref="s3://kanvis-chunks/sessions/cliente-demo/store/cam/sess/",
        pose_chunk_ref="chunk_001/poses_full.npy",
        image_chunk_ref="chunk_001/frames/",
        chunk_stem="chunk_001_002",
        span_spec="overlap_half:chunk_001:chunk_002:ratio=0.5000",
        dry_run=True,
        persist_to_minio=True,
    )
    assert job.chunk_stems == ("chunk_001", "chunk_002")
    assert job.anchor_chunk == "chunk_001"
    assert job.extra["p1_motion_stem"] == "chunk_001_002"
    assert job.extra["span_mode"] == "overlap_half"
    assert job.extra["overlap_ratio"] == 0.5
    assert job.persist is False
    assert job.chunk_prefix(job.anchor_chunk).endswith("chunk_001/")


def test_relative_pose_ref_with_source():
    job = job_from_fields(
        source_ref="s3://kanvis-chunks/sessions/r/s/c/sid/",
        pose_chunk_ref="chunk_003/poses_full.npy",
        chunk_stem="chunk_003",
        span_spec="single",
    )
    assert job.anchor_chunk == "chunk_003"
    assert job.session_prefix.endswith("sid/")


def test_with_next_and_single_still_work():
    nxt = job_from_fields(
        source_ref="s3://kanvis-chunks/sessions/r/s/c/sid/",
        chunk_stem="chunk_003",
        span_spec="with_next",
    )
    assert nxt.chunk_stems == ("chunk_003", "chunk_004")
    one = job_from_fields(
        source_ref="s3://kanvis-chunks/sessions/r/s/c/sid/",
        chunk_stem="chunk_003",
        span_spec="single",
        persist_to_minio=True,
        dry_run=False,
    )
    assert one.chunk_stems == ("chunk_003",)
    assert one.persist is True


def test_load_local_bundle_overlap_half(tmp_path):
    def _write_chunk(stem: str, start: int):
        frames = tmp_path / stem / "frames"
        frames.mkdir(parents=True)
        import cv2

        for i in range(4):
            cv2.imwrite(str(frames / f"frame_{i:03d}.jpg"), np.full((8, 8, 3), start + i, np.uint8))
        poses = np.zeros((16, 8, 2), dtype=np.float32)
        poses[:, 4] = (0.2, 0.3)
        poses[:, 5] = (0.4, 0.5)
        np.save(str(tmp_path / stem / "poses_full.npy"), poses)
        meta = {
            "frames": [
                {
                    "file": f"frame_{i:03d}.jpg",
                    "source_index": i * 4,
                    "left_wrist_visible": True,
                    "right_wrist_visible": True,
                }
                for i in range(4)
            ]
        }
        (tmp_path / stem / "frames_meta.json").write_text(
            __import__("json").dumps(meta), encoding="utf-8"
        )

    _write_chunk("chunk_001", 10)
    _write_chunk("chunk_002", 50)
    bundle = load_local_bundle(
        tmp_path,
        ("chunk_001", "chunk_002"),
        span={
            "span_mode": "overlap_half",
            "overlap_left": "chunk_001",
            "overlap_right": "chunk_002",
            "overlap_ratio": 0.5,
            "span_spec": "overlap_half:chunk_001:chunk_002:ratio=0.5000",
            "p1_motion_stem": "chunk_001_002",
        },
    )
    assert len(bundle["frames"]) == 4
    assert bundle["frame_chunk_names"] == [
        "chunk_001",
        "chunk_001",
        "chunk_002",
        "chunk_002",
    ]
    assert bundle["frames_used"] == {"left": [2, 3], "right": [0, 1]}
    assert bundle["frames_origin"] == [8, 12, 0, 4]
    assert bundle["p1_motion_stem"] == "chunk_001_002"
    assert len(bundle["poses_by_chunk"]["chunk_001"]) == 16
    assert len(bundle["frames_meta"]) == 4
    assert all(row.get("left_wrist_visible") for row in bundle["frames_meta"])
