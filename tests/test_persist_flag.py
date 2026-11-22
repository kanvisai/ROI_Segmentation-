from kanvis_segmentation.refs import job_from_fields


def test_dry_run_disables_persist():
    job = job_from_fields(
        bucket="kanvis-chunks",
        retail_id="r",
        store_id="s",
        camera_id="c",
        session_id="id",
        chunk_stem="1",
        persist_to_minio=True,
        dry_run=True,
    )
    assert job.dry_run is True
    assert job.persist is False
