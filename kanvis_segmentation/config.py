"""Environment for the kanvis-segmentation container."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    listen: str = "0.0.0.0:50100"
    metrics_port: int = 9109
    s3_endpoint_url: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "us-east-1"
    s3_secure: bool = False
    sam3_summary: bool = True
    min_wrist_visible_frames: int = 3
    min_wrist_visible_fraction: float = 0.10
    min_wrist_conf: float = 0.35
    wrist_match_max_px: float = 140.0
    identifiers_yaml: Path = ROOT / "identifiers.yaml"
    work_dir: Path = Path("/tmp/kanvis-segmentation")
    mediapipe: bool = True

    @property
    def host(self) -> str:
        host, _, _port = self.listen.rpartition(":")
        return host or "0.0.0.0"

    @property
    def port(self) -> int:
        _host, _, port = self.listen.rpartition(":")
        return int(port or "50100")


def load_settings() -> Settings:
    yaml_env = os.environ.get("SEGMENTATION_IDENTIFIERS_YAML", "").strip()
    work = os.environ.get("SEGMENTATION_WORK_DIR", "").strip()
    return Settings(
        listen=os.environ.get("KANVIS_SEGMENTATION_LISTEN", "0.0.0.0:50100"),
        metrics_port=_int("KANVIS_SEGMENTATION_METRICS_PORT", 9109),
        s3_endpoint_url=os.environ.get("SEGMENTATION_S3_ENDPOINT_URL", "").strip(),
        s3_access_key_id=os.environ.get("SEGMENTATION_S3_ACCESS_KEY_ID", "").strip(),
        s3_secret_access_key=os.environ.get("SEGMENTATION_S3_SECRET_ACCESS_KEY", "").strip(),
        s3_region=os.environ.get("SEGMENTATION_S3_REGION", "us-east-1"),
        s3_secure=_bool("SEGMENTATION_S3_SECURE", False),
        sam3_summary=_bool("SEGMENTATION_SAM3_SUMMARY", True),
        min_wrist_visible_frames=_int("SEGMENTATION_SAM3_MIN_WRIST_VISIBLE_FRAMES", 3),
        min_wrist_visible_fraction=_float("SEGMENTATION_SAM3_MIN_WRIST_VISIBLE_FRACTION", 0.10),
        min_wrist_conf=_float("SEGMENTATION_SAM3_MIN_WRIST_CONF", 0.35),
        wrist_match_max_px=_float("SEGMENTATION_SAM3_WRIST_MATCH_MAX_PX", 140.0),
        identifiers_yaml=Path(yaml_env) if yaml_env else ROOT / "identifiers.yaml",
        work_dir=Path(work) if work else Path("/tmp/kanvis-segmentation"),
        mediapipe=_bool("SEGMENTATION_MEDIAPIPE", True),
    )
