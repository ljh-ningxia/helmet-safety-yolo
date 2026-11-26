"""COCO parsing and validation for the Hard Hats dataset.

The upstream export stores one ``_annotations.coco.json`` per split with all
images flattened next to it. We deliberately validate rather than trust:
out-of-image boxes, zero-area boxes, duplicate ids and orphan annotations are
all recorded and reported in the dataset card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from ..schemas import BBox, CocoAnnotation


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoDataset:
    split: str
    root: Path
    images: Dict[int, CocoImage] = field(default_factory=dict)
    annotations: List[CocoAnnotation] = field(default_factory=list)
    categories: Dict[int, str] = field(default_factory=dict)
    licenses: List[dict] = field(default_factory=list)
    issues: Dict[str, int] = field(default_factory=dict)

    @property
    def by_image(self) -> Dict[int, List[CocoAnnotation]]:
        out: Dict[int, List[CocoAnnotation]] = {i: [] for i in self.images}
        for a in self.annotations:
            out.setdefault(a.image_id, []).append(a)
        return out


def load_coco(json_path: Path, split: str) -> CocoDataset:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    ds = CocoDataset(split=split, root=json_path.parent)
    ds.licenses = raw.get("licenses", [])
    ds.categories = {int(c["id"]): str(c["name"]) for c in raw.get("categories", [])}

    seen_images: set = set()
    for im in raw.get("images", []):
        iid = int(im["id"])
        if iid in seen_images:
            ds.issues["duplicate_image_id"] = ds.issues.get("duplicate_image_id", 0) + 1
            continue
        seen_images.add(iid)
        ds.images[iid] = CocoImage(iid, str(im["file_name"]),
                                   int(im["width"]), int(im["height"]))

    seen_ann: set = set()
    for a in raw.get("annotations", []):
        aid = int(a["id"])
        if aid in seen_ann:
            ds.issues["duplicate_annotation_id"] = ds.issues.get("duplicate_annotation_id", 0) + 1
            continue
        seen_ann.add(aid)
        iid = int(a["image_id"])
        if iid not in ds.images:
            ds.issues["orphan_annotation"] = ds.issues.get("orphan_annotation", 0) + 1
            continue
        x, y, w, h = (float(v) for v in a["bbox"][:4])
        if w <= 0 or h <= 0:
            ds.issues["non_positive_box"] = ds.issues.get("non_positive_box", 0) + 1
            continue
        im = ds.images[iid]
        box = BBox(x, y, x + w, y + h)
        clipped = box.clip(im.width, im.height)
        if clipped.area < box.area * 0.999:
            ds.issues["box_clipped_to_image"] = ds.issues.get("box_clipped_to_image", 0) + 1
        if clipped.area <= 0:
            ds.issues["box_outside_image"] = ds.issues.get("box_outside_image", 0) + 1
            continue
        ds.annotations.append(CocoAnnotation(iid, int(a["category_id"]), clipped, aid,
                                             int(a.get("iscrowd", 0))))
    return ds


def verify_images(ds: CocoDataset) -> Tuple[List[CocoImage], List[str]]:
    """Return (usable images, list of missing/unreadable file names)."""
    usable, missing = [], []
    for im in ds.images.values():
        if not (ds.root / im.file_name).is_file():
            missing.append(im.file_name)
        else:
            usable.append(im)
    if missing:
        ds.issues["missing_image_file"] = len(missing)
    return usable, missing


def class_counts(ds: CocoDataset) -> Dict[str, int]:
    out = {name: 0 for name in ds.categories.values()}
    for a in ds.annotations:
        name = ds.categories.get(a.category_id, f"unknown_{a.category_id}")
        out[name] = out.get(name, 0) + 1
    return out
