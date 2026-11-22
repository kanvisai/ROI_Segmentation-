"""Slim tracking payload and C3 roi_fp_review (c2_roi_fp_review_v1)."""

from __future__ import annotations

from typing import Any

ROI_FP_REVIEW_SCHEMA = "c2_roi_fp_review_v1"
ROI_FP_REVIEW_KEYS = frozenset(
    {
        "schema_version",
        "final_state",
        "hands_visible",
        "cell_phone_confidence",
        "inside_container_confidence",
        "container",
    }
)

SKIP_STATES = frozenset({"no_hands", "skipped"})


def slim_tracking_payload_for_storage(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Drop the YAML catalog; keep the path in parameters if present."""
    src = dict(payload or {})
    yaml_path = None
    params = src.get("parameters")
    if isinstance(params, dict):
        yaml_path = params.get("identifiers_yaml")
        src["parameters"] = dict(params)
    src.pop("identifiers_yaml", None)
    if yaml_path and isinstance(src.get("parameters"), dict):
        src["parameters"]["identifiers_yaml"] = yaml_path
    return src


def _container_from_tracking(payload: dict[str, Any]) -> str | None:
    for obj in payload.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        dest = obj.get("final_container")
        if isinstance(dest, dict) and dest.get("category"):
            return str(dest["category"])
        if isinstance(dest, str) and dest:
            return dest
        if obj.get("final_state") == "inside_container":
            cat = obj.get("container") or obj.get("destination")
            if cat:
                return str(cat)
    if payload.get("final_state") == "inside_container":
        for cont in payload.get("containers") or []:
            if isinstance(cont, dict) and cont.get("destination"):
                return str(cont.get("category") or cont.get("id") or "")
    return None


def summarize_tracking_for_c3(payload: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(payload or {})
    review: dict[str, Any] = {
        "schema_version": ROI_FP_REVIEW_SCHEMA,
        "final_state": str(src.get("final_state") or "no_data"),
        "hands_visible": bool(src.get("hands_visible")),
    }
    phone = src.get("cell_phone_confidence")
    if phone is not None:
        review["cell_phone_confidence"] = float(phone)
    inside = src.get("inside_container_confidence")
    if inside is not None:
        review["inside_container_confidence"] = float(inside)
    container = _container_from_tracking(src)
    if container:
        review["container"] = container
    return whitelist_roi_fp_review(review)


def whitelist_roi_fp_review(payload: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(payload or {})
    return {key: src[key] for key in ROI_FP_REVIEW_KEYS if key in src}


def rpc_status_from_tracking(payload: dict[str, Any] | None) -> str:
    src = dict(payload or {})
    state = str(src.get("final_state") or "")
    if state in SKIP_STATES or (
        src.get("hands_visible") is False and state in {"no_hands", "no_data", ""}
    ):
        return "skipped"
    if state:
        return "ok"
    return "error"
