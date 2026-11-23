"""Resolve structured tenant fields or legacy C2 MinIO refs into a job."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

_S3 = re.compile(r"^s3://([^/]+)/(.+)$")
_CHUNK = re.compile(r"(chunk_\d+)", re.IGNORECASE)
_OVERLAP_HALF = re.compile(
    r"^overlap_half:(chunk_[0-9]+):(chunk_[0-9]+)(?::ratio=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))?$",
    re.IGNORECASE,
)


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


def chunk_number(stem: str) -> int | None:
    hit = re.fullmatch(r"chunk_(\d+)", (stem or "").strip(), re.IGNORECASE)
    return int(hit.group(1)) if hit else None


def parse_overlap_half(span_spec: str) -> tuple[str, str, float] | None:
    """Parse C3 `overlap_half:{left}:{right}:ratio={ratio}`. None if not that mode."""
    raw = (span_spec or "").strip()
    if not raw.lower().startswith("overlap_half:"):
        return None
    hit = _OVERLAP_HALF.match(raw)
    if not hit:
        raise ValueError(f"invalid overlap_half span_spec: {span_spec}")
    left = normalize_chunk_stem(hit.group(1))
    right = normalize_chunk_stem(hit.group(2))
    ratio = float(hit.group(3)) if hit.group(3) is not None else 0.5
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"overlap ratio out of range [0, 1]: {ratio}")
    return left, right, ratio


def overlap_half_counts(n_left: int, n_right: int, ratio: float) -> tuple[int, int]:
    """k_left / k_right with C3 `max(1, round(n * ratio))`, clipped to n."""

    def _k(n: int) -> int:
        if n <= 0:
            return 0
        return min(int(n), max(1, int(round(float(n) * float(ratio)))))

    return _k(n_left), _k(n_right)


def overlap_half_indices(n_left: int, n_right: int, ratio: float) -> tuple[list[int], list[int]]:
    k_left, k_right = overlap_half_counts(n_left, n_right, ratio)
    left_idx = list(range(n_left - k_left, n_left)) if n_left and k_left else []
    right_idx = list(range(0, k_right)) if n_right and k_right else []
    return left_idx, right_idx


def p1_overlap_stem(left: str, right: str) -> str:
    a, b = chunk_number(left), chunk_number(right)
    if a is not None and b is not None:
        return f"chunk_{a:03d}_{b:03d}"
    return f"{left}_{right}"


def parse_span_spec(span_spec: str, anchor: str) -> tuple[str, ...]:
    overlap = parse_overlap_half(span_spec)
    if overlap:
        left, right, _ratio = overlap
        return (left, right)
    raw = (span_spec or "").strip().lower()
    numero = chunk_number(anchor)
    if numero is None:
        anchor_n = normalize_chunk_stem(anchor) if (anchor or "").strip() else ""
        numero = chunk_number(anchor_n)
        if numero is None:
            if raw in {"", "single", "chunk_only"} and anchor_n:
                return (anchor_n,)
            raise ValueError(f"cannot apply span_spec={span_spec!r} to stem {anchor!r}")
    else:
        anchor_n = f"chunk_{numero:03d}"
    if raw in {"", "single", "chunk_only"}:
        return (anchor_n,)
    if raw == "with_prev":
        prev = f"chunk_{numero - 1:03d}" if numero > 1 else None
        return (prev, anchor_n) if prev else (anchor_n,)
    if raw == "with_next":
        return (anchor_n, f"chunk_{numero + 1:03d}")
    if raw == "next":
        return (f"chunk_{numero + 1:03d}",)
    stems = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            stems.append(normalize_chunk_stem(piece))
    return tuple(stems or (anchor_n,))


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

    if source:
        bkt, key = parse_s3_uri(source)
        derived_bucket = derived_bucket or bkt
        derived_prefix = _ensure_slash(key)

    def _absolute(ref: str) -> str:
        texto = (ref or "").strip()
        if not texto:
            return ""
        if texto.startswith("s3://") or "://" in texto.split("/", 1)[0]:
            return texto
        if derived_bucket and derived_prefix:
            return f"s3://{derived_bucket}/{derived_prefix}{texto.lstrip('/')}"
        return texto

    pose = _absolute(pose)
    images = _absolute(images)

    if pose:
        derived_stem = chunk_stem_from_ref(pose) or derived_stem
        bkt, key = parse_s3_uri(pose)
        derived_bucket = derived_bucket or bkt
        # …/chunk_003/poses_full.npy → session directory
        if "/chunk_" in key:
            derived_prefix = derived_prefix or _ensure_slash(key[: key.lower().rfind("/chunk_")])
        else:
            derived_prefix = derived_prefix or _ensure_slash(key.rsplit("/", 1)[0])

    if images and not derived_prefix:
        bkt, key = parse_s3_uri(images)
        derived_bucket = derived_bucket or bkt
        derived_stem = derived_stem or chunk_stem_from_ref(key) or derived_stem
        if "/chunk_" in key:
            derived_prefix = _ensure_slash(key[: key.lower().rfind("/chunk_")])
        else:
            derived_prefix = _ensure_slash(key)

    sid = (session_id or "").strip()
    if sid and retail_id and store_id and camera_id:
        derived_prefix = derived_prefix or session_prefix_from_structured(
            retail_id, store_id, camera_id, sid
        )

    if not derived_bucket:
        raise ValueError("bucket or s3 source_ref is required")
    if not derived_prefix:
        raise ValueError("could not resolve session prefix")

    overlap = parse_overlap_half(span_spec)
    extra: dict = {"span_spec": (span_spec or "").strip() or "single"}
    if overlap:
        left, right, ratio = overlap
        stems = (left, right)
        extra.update(
            {
                "span_mode": "overlap_half",
                "overlap_left": left,
                "overlap_right": right,
                "overlap_ratio": ratio,
                "p1_motion_stem": (chunk_stem or "").strip() or p1_overlap_stem(left, right),
            }
        )
    else:
        stem = (chunk_stem or derived_stem or "").strip()
        if not stem:
            raise ValueError("chunk_stem or pose_chunk_ref is required")
        physical = stem if chunk_number(stem) is not None else (derived_stem or stem)
        if chunk_number(physical) is None:
            physical = normalize_chunk_stem(physical)
        else:
            physical = f"chunk_{chunk_number(physical):03d}"
        stems = parse_span_spec(span_spec, physical)
        extra["span_mode"] = (span_spec or "single").strip().lower() or "single"
        extra["p1_motion_stem"] = (chunk_stem or physical).strip()

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
        extra=extra,
    )
