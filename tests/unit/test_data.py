"""Unit tests for COCO parsing/validation and YOLO conversion."""
from __future__ import annotations

import json

from helmet_safety.data.coco import class_counts, load_coco, verify_images
from helmet_safety.data.convert import _to_yolo_line
from helmet_safety.data.split import describe_strata, stratified_indices
from helmet_safety.schemas import BBox

RAW = {
    "licenses": [{"id": 1, "name": "CC BY 4.0"}],
    "categories": [{"id": 0, "name": "hardhat"}, {"id": 1, "name": "no-hardhat"}],
    "images": [
        {"id": 1, "file_name": "a.jpg", "width": 640, "height": 640},
        {"id": 2, "file_name": "b.jpg", "width": 640, "height": 640},
        {"id": 1, "file_name": "dup.jpg", "width": 640, "height": 640},
    ],
    "annotations": [
        {"id": 10, "image_id": 1, "category_id": 0, "bbox": [100, 100, 50, 50]},
        {"id": 11, "image_id": 1, "category_id": 1, "bbox": [200, 200, 20, 20]},
        {"id": 12, "image_id": 2, "category_id": 0, "bbox": [630, 630, 30, 30]},  # clipped
        {"id": 13, "image_id": 99, "category_id": 0, "bbox": [0, 0, 10, 10]},  # orphan
        {"id": 14, "image_id": 2, "category_id": 1, "bbox": [0, 0, 0, 10]},  # invalid
        {"id": 10, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},  # duplicate id
    ],
}


def test_coco_validation_records_all_issue_types(tmp_path):
    p = tmp_path / "_annotations.coco.json"
    p.write_text(json.dumps(RAW), encoding="utf-8")
    ds = load_coco(p, "test")
    assert len(ds.images) == 2
    assert ds.issues["duplicate_image_id"] == 1
    assert ds.issues["orphan_annotation"] == 1
    assert ds.issues["non_positive_box"] == 1
    assert ds.issues["duplicate_annotation_id"] == 1
    assert ds.issues["box_clipped_to_image"] == 1
    assert class_counts(ds) == {"hardhat": 2, "no-hardhat": 1}
    usable, missing = verify_images(ds)
    assert usable == [] and len(missing) == 2  # files intentionally absent


def test_yolo_line_is_normalised_and_clamped():
    line = _to_yolo_line(1, BBox(160, 160, 320, 320), 640, 640)
    assert line.startswith("1 ")
    assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])
    over = _to_yolo_line(0, BBox(-100, -100, 900, 900), 640, 640)
    assert all(0.0 <= float(v) <= 1.0 for v in over.split()[1:])


def test_stratified_sampling_is_deterministic_and_preserves_rare_bucket():
    items = [{"viol": i < 5, "n": 1} for i in range(100)]
    def key(i, x):
        return "viol" if x["viol"] else "clean"
    a = stratified_indices(items, key, seed=7, n=20)
    b = stratified_indices(items, key, seed=7, n=20)
    assert a == b
    assert len(a) == 20
    assert sum(1 for i in a if key(i, items[i]) == "viol") >= 1
    strata = describe_strata(items, a, key)
    assert strata["viol"]["full"] == 5
    assert strata["viol"]["subset"] == 1
    assert stratified_indices(items, key, seed=7, n=None) == list(range(100))
