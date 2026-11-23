import pytest

from kanvis_segmentation.refs import job_from_fields, parse_span_spec


def test_legacy_pose_ref_derives_chunk_and_session():
    job = job_from_fields(
        source_ref="s3://kanvis-chunks/sessions/cliente-demo/store/cam/sess/",
        pose_chunk_ref="s3://kanvis-chunks/sessions/cliente-demo/store/cam/sess/chunk_003/poses_full.npy",
        session_id="sess",
        dry_run=True,
        persist_to_minio=True,
    )
    assert job.bucket == "kanvis-chunks"
    assert job.anchor_chunk == "chunk_003"
    assert job.session_prefix.endswith("sess/")
    assert job.persist is False
    assert job.dry_run is True


def test_structured_path():
    job = job_from_fields(
        bucket="kanvis-chunks",
        retail_id="cliente-demo",
        store_id="store-1",
        camera_id="cam-1",
        session_id="abc",
        chunk_stem="chunk_001",
        dry_run=False,
        persist_to_minio=True,
    )
    assert job.session_uri == "s3://kanvis-chunks/sessions/cliente-demo/store-1/cam-1/abc/"
    assert job.chunk_stems == ("chunk_001",)
    assert job.persist is True


def test_span_with_prev_next():
    assert parse_span_spec("with_next", "chunk_001") == ("chunk_001", "chunk_002")
    assert parse_span_spec("with_prev", "chunk_002") == ("chunk_001", "chunk_002")
    assert parse_span_spec("next", "chunk_001") == ("chunk_002",)
    assert parse_span_spec("chunk_002,chunk_003", "chunk_003") == ("chunk_002", "chunk_003")
    assert parse_span_spec(
        "overlap_half:chunk_001:chunk_002:ratio=0.5000", "chunk_001"
    ) == ("chunk_001", "chunk_002")


def test_missing_chunk_raises():
    with pytest.raises(ValueError):
        job_from_fields(bucket="b", source_ref="s3://b/sessions/x/")
