from kanvis_segmentation.contracts import (
    ROI_FP_REVIEW_KEYS,
    rpc_status_from_tracking,
    slim_tracking_payload_for_storage,
    summarize_tracking_for_c3,
    whitelist_roi_fp_review,
)


def test_slim_drops_identifiers_catalog():
    payload = {
        "final_state": "in_hand",
        "hands_visible": True,
        "identifiers_yaml": [{"key": "botella"}],
        "parameters": {"identifiers_yaml": "identifiers.yaml", "fps": 12},
        "objects": [{"id": 1, "timeline": [1, 2]}],
    }
    slim = slim_tracking_payload_for_storage(payload)
    assert "identifiers_yaml" not in slim
    assert slim["parameters"]["identifiers_yaml"] == "identifiers.yaml"
    assert slim["final_state"] == "in_hand"


def test_roi_fp_review_whitelist():
    review = summarize_tracking_for_c3(
        {
            "final_state": "inside_container",
            "hands_visible": True,
            "cell_phone_confidence": 0.8,
            "inside_container_confidence": 0.7,
            "objects": [{"final_container": {"category": "cesto"}}],
            "extra_key": "drop-me",
        }
    )
    assert set(review) <= ROI_FP_REVIEW_KEYS
    assert review["schema_version"] == "c2_roi_fp_review_v1"
    assert review["container"] == "cesto"
    assert "extra_key" not in review
    stripped = whitelist_roi_fp_review({**review, "noise": 1})
    assert "noise" not in stripped


def test_rpc_status_skipped_no_hands():
    assert rpc_status_from_tracking({"final_state": "no_hands", "hands_visible": False}) == "skipped"
    assert rpc_status_from_tracking({"final_state": "in_hand", "hands_visible": True}) == "ok"
