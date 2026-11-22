"""MinIO / S3 helpers for session chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .refs import SegmentationJob

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def s3_client(settings: Settings):
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = settings.s3_endpoint_url or None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.s3_access_key_id or None,
        aws_secret_access_key=settings.s3_secret_access_key or None,
        region_name=settings.s3_region,
        use_ssl=settings.s3_secure if endpoint else None,
        config=BotoConfig(s3={"addressing_style": "path"}),
    )


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for item in resp.get("Contents") or []:
            keys.append(item["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def download_file(client, bucket: str, key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return dest


def upload_bytes(client, bucket: str, key: str, body: bytes, content_type: str = "application/json") -> None:
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def object_exists(client, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def _pick_pose_key(keys: Iterable[str], prefix: str) -> str | None:
    wanted = (
        f"{prefix}poses_full.npy",
        f"{prefix}poses_compat.npy",
        f"{prefix}poses.npy",
    )
    have = set(keys)
    for key in wanted:
        if key in have:
            return key
    return None


def materialize_job(
    client,
    job: SegmentationJob,
    work_dir: Path,
) -> Path:
    """Download frames, meta and poses for every chunk in the span. Returns session root."""
    session_root = work_dir / "session"
    session_root.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for stem in job.chunk_stems:
        prefix = job.chunk_prefix(stem)
        dest = session_root / stem
        dest.mkdir(parents=True, exist_ok=True)
        keys = list_keys(client, job.bucket, prefix)
        frame_keys = sorted(
            k for k in keys
            if "/frames/" in k and Path(k).suffix.lower() in IMAGE_EXTS
        )
        if not frame_keys:
            missing.append(stem)
            continue
        for key in frame_keys:
            rel = key[len(prefix):] if key.startswith(prefix) else Path(key).name
            download_file(client, job.bucket, key, dest / rel)
        for name in ("frames_meta.json", "meta.json"):
            key = f"{prefix}{name}"
            if key in keys:
                download_file(client, job.bucket, key, dest / name)
        pose_key = _pick_pose_key(keys, prefix)
        if pose_key:
            download_file(client, job.bucket, pose_key, dest / Path(pose_key).name)
    if missing == list(job.chunk_stems):
        raise FileNotFoundError(f"no frames under {job.session_uri} chunks={job.chunk_stems}")
    return session_root


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
