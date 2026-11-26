"""COCO -> YOLO layout conversion with geometry validation.

Output layout (Ultralytics-compatible)::

    data/processed/
      images/{train,valid,test}/<stem>.jpg
      labels/{train,valid,test}/<stem>.txt
      data.yaml

Every converted label line is ``cls cx cy w h`` with values in [0, 1]. We keep
a line-level audit so any conversion loss is visible rather than silent.
"""
from __future__ import annotations

import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from ..logging import get_logger
from ..schemas import BBox
from .coco import CocoDataset, load_coco, verify_images
from .split import _box_bin, describe_strata, stratified_indices

log = get_logger(__name__)


@dataclass
class ConversionReport:
    split: str
    images_written: int = 0
    labels_written: int = 0
    boxes_written: int = 0
    boxes_skipped: Dict[str, int] = field(default_factory=dict)
    strata: Dict[str, Dict[str, int]] = field(default_factory=dict)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    source_images: int = 0

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "source_images": self.source_images,
            "images_written": self.images_written,
            "labels_written": self.labels_written,
            "boxes_written": self.boxes_written,
            "boxes_skipped": self.boxes_skipped,
            "strata": self.strata,
            "class_distribution": self.class_distribution,
        }


def _to_yolo_line(cls: int, box: BBox, w: int, h: int) -> str:
    cx = (box.x1 + box.x2) / 2.0 / w
    cy = (box.y1 + box.y2) / 2.0 / h
    bw = box.width / w
    bh = box.height / h
    # Numerical guard: upstream boxes are already clipped, float noise aside.
    vals = [min(max(v, 0.0), 1.0) for v in (cx, cy, bw, bh)]
    return f"{cls} {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f}"


def convert_split(split: str, raw_dir: Path, out_root: Path, classes: tuple,
                  subset: int | None = None, seed: int = 20261126,
                  copy_mode: str = "copy") -> ConversionReport:
    src = raw_dir / split
    coco_json = src / "_annotations.coco.json"
    if not coco_json.is_file():
        raise FileNotFoundError(f"{coco_json} missing — run 01_download_dataset.py first")

    ds: CocoDataset = load_coco(coco_json, split)
    usable, missing = verify_images(ds)
    log.info("parsed COCO", split=split, images=len(ds.images),
             annotations=len(ds.annotations), usable=len(usable), missing=len(missing),
             issues=ds.issues)

    by_image = ds.by_image
    items = [(im, by_image.get(im.id, [])) for im in usable]

    def group_key(i: int, item) -> str:
        im, anns = item
        has_violation = any(a.category_id == 1 for a in anns)
        return f"{_box_bin(len(anns))}|{'viol' if has_violation else 'clean'}"

    # zlib.crc32 is stable across processes; builtin hash() is randomized.
    chosen = stratified_indices(items, group_key,
                                seed=seed + zlib.crc32(split.encode()) % 10000, n=subset)
    rep = ConversionReport(split=split, source_images=len(usable))
    rep.strata = describe_strata(items, chosen, group_key)
    rep.boxes_skipped = dict(ds.issues)

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    # Idempotency: a re-run with a different subset size must not leave stale
    # files mixed into the new dataset. We ARCHIVE instead of deleting so the
    # previous run stays recoverable and no bulk-delete is ever required.
    stamp = int(time.time() * 1000)
    for d, kind in ((img_dir, "images"), (lbl_dir, "labels")):
        if d.exists() and any(d.iterdir()):
            archive = out_root.parent / "interim" / f"stale_{split}_{kind}_{stamp}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            d.rename(archive)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # category ids in this dataset are already 0-based and contiguous
    cat_ids = sorted(ds.categories)
    remap = {cid: i for i, cid in enumerate(cat_ids)}

    counter: Dict[str, int] = {name: 0 for name in classes}
    for idx in chosen:
        im, anns = items[idx]
        stem = Path(im.file_name).stem
        dest_img = img_dir / f"{stem}.jpg"
        src_img = src / im.file_name
        if copy_mode == "copy":
            shutil.copy2(src_img, dest_img)
        else:
            if not dest_img.exists():
                dest_img.symlink_to(src_img.resolve())
        lines: List[str] = []
        for a in anns:
            if a.category_id not in remap:
                rep.boxes_skipped["unknown_category"] = rep.boxes_skipped.get("unknown_category", 0) + 1
                continue
            line = _to_yolo_line(remap[a.category_id], a.bbox, im.width, im.height)
            lines.append(line)
            name = ds.categories.get(a.category_id, "unknown")
            counter[name] = counter.get(name, 0) + 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        rep.images_written += 1
        rep.labels_written += 1
        rep.boxes_written += len(lines)

    rep.class_distribution = counter
    log.info("converted", split=split, images=rep.images_written,
             boxes=rep.boxes_written, subset_of=len(usable))
    return rep


def write_data_yaml(out_root: Path, classes: tuple, path: Path | None = None) -> Path:
    payload = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }
    target = path or (out_root / "data.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return target
