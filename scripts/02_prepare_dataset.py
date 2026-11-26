#!/usr/bin/env python3
"""Prepare a reproducible YOLO dataset from the downloaded COCO archives.

Pipeline: verify checksums -> unzip -> parse/validate COCO -> stratified
subset -> COCO->YOLO conversion -> data.yaml -> EDA audit + evidence sheet.

    python scripts/02_prepare_dataset.py                 # all splits
    python scripts/02_prepare_dataset.py --splits test  # partial (smoke test)
    python scripts/02_prepare_dataset.py --subset-train 3000 --subset-valid 800 --subset-test 600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.config import load_config
from helmet_safety.data.audit import audit_yolo  # noqa: E402
from helmet_safety.data.coco import class_counts, load_coco  # noqa: E402
from helmet_safety.data.convert import convert_split, write_data_yaml  # noqa: E402
from helmet_safety.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def ensure_unzipped(split: str, raw: Path, expected_sha: str = "") -> Path:
    import hashlib
    arc = raw / f"{split}.zip"
    out = raw / split
    marker = out / ".unpacked"
    if marker.exists():
        return out
    if not arc.is_file():
        raise FileNotFoundError(f"{arc} missing — run 01_download_dataset.py first")
    if expected_sha:
        h = hashlib.sha256()
        with arc.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != expected_sha:
            raise ValueError(f"{arc.name}: checksum mismatch, re-download it")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(arc) as z:
        names = z.namelist()
        z.extractall(out)
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
    log.info("unzipped", split=split, files=len(names), seconds=round(time.time() - t0, 1))
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--subset-train", type=int, default=None)
    p.add_argument("--subset-valid", type=int, default=None)
    p.add_argument("--subset-test", type=int, default=None)
    p.add_argument("--no-audit", action="store_true")
    p.add_argument("--config-env", default=None)
    a = p.parse_args(argv)

    cfg = load_config(env=a.config_env)
    raw: Path = cfg.resolve(cfg.data.raw_dir)
    processed: Path = cfg.resolve(cfg.data.processed_dir)
    results = cfg.resolve("experiments/results")
    results.mkdir(parents=True, exist_ok=True)

    manifest_path = raw / "manifest.json"
    checksums: Dict[str, str] = {}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = {r["split"]: r["sha256"] for r in m["files"]}
        log.info("manifest", dataset=m["dataset_id"], license=m["license"], sha=m["upstream_sha"])

    subsets = {
        "train": a.subset_train if a.subset_train is not None else cfg.data.subset_train,
        "valid": a.subset_valid if a.subset_valid is not None else cfg.data.subset_valid,
        "test": a.subset_test if a.subset_test is not None else cfg.data.subset_test,
    }

    reports = []
    for split in a.splits:
        src = ensure_unzipped(split, raw, checksums.get(split, ""))
        # Early structural validation of the upstream COCO file, before conversion.
        cj = src / "_annotations.coco.json"
        if cj.is_file():
            ds = load_coco(cj, split)
            log.info("coco summary", split=split, images=len(ds.images),
                     annotations=len(ds.annotations), classes=class_counts(ds),
                     issues=ds.issues)
        rep = convert_split(split, raw, processed, cfg.data.classes,
                            subset=subsets.get(split), seed=cfg.data.seed)
        reports.append(rep.as_dict())

    yaml_path = write_data_yaml(processed, cfg.data.classes)
    log.info("wrote", data_yaml=str(yaml_path))

    audit = {}
    if not a.no_audit:
        audit = audit_yolo(processed, cfg.data.classes, results,
                           contact_sheet_n=8, seed=cfg.data.seed)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": cfg.data.seed,
        "subsets_requested": subsets,
        "processed_dir": str(processed.resolve()).replace("\\", "/"),
        "data_yaml": str(yaml_path.resolve()).replace("\\", "/"),
        "splits": reports,
        "audit": audit,
        "evidence_level": "MEASURED",
        "note": "Statistics are computed from the actual downloaded images/labels.",
    }
    (results / "dataset_preparation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("preparation complete", report=str(results / "dataset_preparation.json"))
    print(json.dumps({"splits": reports,
                      "total_images": audit.get("total_images"),
                      "total_boxes": audit.get("total_boxes"),
                      "class_counts": audit.get("class_counts"),
                      "duplicates": audit.get("exact_duplicate_groups")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
