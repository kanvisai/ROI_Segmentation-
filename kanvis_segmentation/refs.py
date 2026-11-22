"""Resolve structured tenant fields or legacy C2 MinIO refs into a job."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

_S3 = re.compile(r"^s3://([^/]+)/(.+)$")
_CHUNK = re.compile(r"(chunk_\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SegmentationJob:
    bucket: str
    session_prefix: str  # key prefix inside bucket, trailing slash
    chunk_stems: tuple[str, ...]
    session_id: str = ""
    retail_id: str = ""
    store_id: str = ""
    camera_id: str = ""
    persist: bool = False
    dry_run: bool = True
    request_id: str = ""
    trace_id: str = ""
    source_ref: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def session_uri(self) -> str:
        prefix = self.session_prefix if self.session_prefix.endswith("/") else f"{self.session_prefix}/"
        return f"s3://{self.bucket}/{prefix}"

    @property
    def anchor_chunk(self) -> str:
        return self.chunk_stems[0] if self.chunk_stems else "chunk_001"

    def chunk_prefix(self, stem: str) -> str:
        base = self.session_prefix if self.session_prefix.endswith("/") else f"{self.session_prefix}/"
        return f"{base}{stem}/"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    texto = (uri or "").strip()
    if not texto:
        raise ValueError("empty S3 URI")
    if texto.startswith("s3://"):
        hit = _S3.match(texto)
        if not hit:
            raise ValueError(f"invalid s3 uri: {uri}")
        bucket, key = hit.group(1), hit.group(2)
        return bucket, key
    parsed = urlparse(texto)
    if parsed.scheme in {"http", "https"} and parsed.path:
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"invalid http s3 path: {uri}")
        return parts[0], "/".join(parts[1:])
    raise ValueError(f"unsupported uri: {uri}")


def _ensure_slash(key: str) -> str:
    return key if key.endswith("/") else f"{key}/"


def chunk_stem_from_ref(ref: str) -> str | None:
    hit = _CHUNK.search(ref or "")
    return hit.group(1).lower().replace("chunk_", "chunk_") if hit else None


def normalize_chunk_stem(stem: str) -> str:
    texto = (stem or "").strip().rstrip("/")
    if not texto:
        raise ValueError("chunk_stem is required")
    if texto.lower().startswith("chunk_"):
        resto = texto.split("_", 1)[1]
        return f"chunk_{int(resto):03d}" if resto.isdigit() else f"chunk_{resto}"
    if texto.isdigit():
        return f"chunk_{int(texto):03d}"
    return f"chunk_{texto}"


def parse_span_spec(span_spec: str, anchor: str) -> tuple[str, ...]:
    raw = (span_spec or "").strip().lower()
    anchor = normalize_chunk_stem(anchor)
    numero = int(anchor.split("_", 1)[1])
    if raw in {"", "single", "chunk_only"}:
        return (anchor,)
    if raw == "with_prev":
        prev = f"chunk_{numero - 1:03d}" if numero > 1 else None
        return (prev, anchor) if prev else (anchor,)
    if raw == "with_next":
        return (anchor, f"chunk_{numero + 1:03d}")
    stems = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            stems.append(normalize_chunk_stem(piece))
    return tuple(stems or (anchor,))


def session_prefix_from_structured(
    retail_id: str, store_id: str, camera_id: str, session_id: str
) -> str:
    parts = [retail_id, store_id, camera_id, session_id]
    if not all(p.strip() for p in parts):
        raise ValueError("retail_id, store_id, camera_id and session_id are required")
    return _ensure_slash("sessions/" + "/".join(p.strip() for p in parts))


def job_from_fields(
    *,
    request_id: str = "",
    trace_id: str = "",
    session_id: str = "",
    bucket: str = "",
    retail_id: str = "",
    store_id: str = "",
    camera_id: str = "",
    chunk_stem: str = "",
    source_ref: str = "",
    pose_chunk_ref: str = "",
    image_chunk_ref: str = "",
    span_spec: str = "single",
    persist_to_minio: bool = False,
    dry_run: bool = True,
) -> SegmentationJob:
    """Accept structured tenant fields or legacy source_ref / pose_chunk_ref."""
    derived_stem = ""
    derived_bucket = (bucket or "").strip()
    derived_prefix = ""

    pose = (pose_chunk_ref or "").strip()
    source = (source_ref or "").strip()
    images = (image_chunk_ref or "").strip()

    if pose:
        derived_stem = chunk_stem_from_ref(pose) or derived_stem
        bkt, key = parse_s3_uri(pose)
        derived_bucket = derived_bucket or bkt
        # …/chunk_003/poses_full.npy → session directory
        if "/chunk_" in key:
            derived_prefix = _ensure_slash(key[: key.lower().rfind("/chunk_")])
        else:
            derived_prefix = _ensure_slash(key.rsplit("/", 1)[0])

    if images and not derived_prefix:
        bkt, key = parse_s3_uri(images)
        derived_bucket = derived_bucket or bkt
        derived_stem = derived_stem or chunk_stem_from_ref(key) or derived_stem
        if "/chunk_" in key:
            derived_prefix = _ensure_slash(key[: key.lower().rfind("/chunk_")])
        else:
            derived_prefix = _ensure_slash(key)

    if source:
        bkt, key = parse_s3_uri(source)
        derived_bucket = derived_bucket or bkt
        derived_prefix = _ensure_slash(key)

    sid = (session_id or "").strip()
    if sid and retail_id and store_id and camera_id:
        derived_prefix = derived_prefix or session_prefix_from_structured(
            retail_id, store_id, camera_id, sid
        )

    stem = (chunk_stem or derived_stem or "").strip()
    if not stem:
        raise ValueError("chunk_stem or pose_chunk_ref is required")
    stem = normalize_chunk_stem(stem)
    if not derived_bucket:
        raise ValueError("bucket or s3 source_ref is required")
    if not derived_prefix:
        raise ValueError("could not resolve session prefix")

    stems = parse_span_spec(span_spec, stem)
    uri = f"s3://{derived_bucket}/{derived_prefix}"
    return SegmentationJob(
        bucket=derived_bucket,
        session_prefix=derived_prefix,
        chunk_stems=stems,
        session_id=(session_id or "").strip(),
        retail_id=(retail_id or "").strip(),
        store_id=(store_id or "").strip(),
        camera_id=(camera_id or "").strip(),
        persist=bool(persist_to_minio) and not bool(dry_run),
        dry_run=bool(dry_run),
        request_id=request_id or "",
        trace_id=trace_id or "",
        source_ref=uri,
    )
