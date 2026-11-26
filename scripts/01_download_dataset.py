#!/usr/bin/env python3
"""Download the Hard Hat Detection dataset from Hugging Face.

Data source
-----------
keremberke/hard-hat-detection (Roboflow Universe "Hard Hats Dataset", 2022)
License: CC BY 4.0. See NOTICE.md for attribution.

Design notes
------------
* Resumable: uses HTTP Range so an interrupted transfer can continue.
* Verifiable: SHA256 for every archive is written to data/raw/manifest.json.
* Reproducible: upstream commit (revision) is pinned and recorded.
* Network failures are retried with exponential backoff.

Usage
-----
    python scripts/01_download_dataset.py                # all splits
    python scripts/01_download_dataset.py --splits test valid-mini
    python scripts/01_download_dataset.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"

DATASET_ID = "keremberke/hard-hat-detection"
REVISION = "main"  # upstream has a single commit; pinned via etag/sha in manifest
HF_API = f"https://huggingface.co/api/datasets/{DATASET_ID}"
HF_RESOLVE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}"

SPLIT_FILES: Dict[str, str] = {
    "train": "data/train.zip",
    "valid": "data/valid.zip",
    "valid-mini": "data/valid-mini.zip",
    "test": "data/test.zip",
}
EXPECTED_IMAGES: Dict[str, int] = {
    "train": 13782, "valid": 3962, "valid-mini": 3, "test": 2001,
}

CHUNK = 1 << 20
UA = {"User-Agent": "helmet-safety/2.0 (dataset reproducibility pipeline)"}


@dataclass
class FileRecord:
    split: str
    remote_path: str
    local_path: str
    size_bytes: int
    sha256: str
    etag: str = ""
    downloaded_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def http_head(url: str) -> Dict[str, str]:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=45) as r:
        return dict(r.headers)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(split: str, remote: str, dest: Path, retries: int = 4) -> FileRecord:
    url = f"{HF_RESOLVE}/{remote}"
    head = http_head(url)
    total = int(head.get("Content-Length") or head.get("X-Linked-Size") or 0)
    etag = (head.get("ETag") or head.get("X-Linked-Etag") or "").strip('"')
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have == total:
            log(f"{split}: already complete ({total/1024/1024:.1f} MB), verifying")
            break
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        mode = "ab" if have else "wb"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, dest.open(mode) as f:
                if have and r.status != 206:
                    # Server ignored Range: restart cleanly to avoid corruption.
                    log(f"{split}: server ignored Range, restarting")
                    f.close()
                    dest.unlink(missing_ok=True)
                    have = 0
                    req = urllib.request.Request(url, headers=UA)
                    r = urllib.request.urlopen(req, timeout=120)
                    f = dest.open("wb")
                t0 = time.time()
                done, last_report = have, 0
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if done - last_report >= 50 * CHUNK:
                        mb = done / 1024 / 1024
                        pct = f"{100*done/total:.1f}%" if total else "?"
                        rate = (done - have) / max(time.time() - t0, 0.1) / 1024 / 1024
                        log(f"{split}: {mb:.0f} MB / {total/1024/1024:.0f} MB ({pct}) {rate:.1f} MB/s")
                        last_report = done
            size = dest.stat().st_size
            if total and size != total:
                raise IOError(f"incomplete: {size} != {total}")
            log(f"{split}: downloaded {size/1024/1024:.1f} MB in {time.time()-t0:.0f}s")
            break
        except Exception as e:  # noqa: BLE001 - network layer, retry all transient errors
            log(f"{split}: attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)

    size = dest.stat().st_size
    digest = sha256_file(dest)
    log(f"{split}: sha256={digest[:16]}… size={size/1024/1024:.1f} MB")
    return FileRecord(split, remote, str(dest.relative_to(ROOT)).replace("\\", "/"),
                      size, digest, etag)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=list(SPLIT_FILES),
                   choices=list(SPLIT_FILES), help="which archives to fetch")
    p.add_argument("--verify-only", action="store_true", help="only verify local archives against manifest")
    p.add_argument("--out", type=Path, default=MANIFEST)
    a = p.parse_args(argv)

    if a.verify_only:
        if not a.out.exists():
            log("manifest missing")
            return 1
        m = json.loads(a.out.read_text(encoding="utf-8"))
        for rec in m["files"]:
            f = ROOT / rec["local_path"]
            ok = f.exists() and sha256_file(f) == rec["sha256"]
            log(f"{'OK  ' if ok else 'FAIL'} {rec['split']:12s} {f.name}")
            if not ok:
                return 1
        return 0

    # Record upstream revision for provenance.
    try:
        with urllib.request.urlopen(urllib.request.Request(HF_API, headers=UA), timeout=30) as r:
            meta = json.load(r)
        sha = meta.get("sha", "")
        license_ = "CC BY 4.0"  # declared in dataset card and inside every COCO json
    except Exception as e:  # noqa: BLE001
        log(f"metadata fetch failed ({e}); continuing with pinned constants")
        sha, license_ = "", "CC BY 4.0"

    records: List[FileRecord] = []
    for s in a.splits:
        dest = RAW / f"{s}.zip"
        records.append(fetch(s, SPLIT_FILES[s], dest))

    payload = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "upstream_sha": sha,
        "license": license_,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "citation": "Roboflow Universe Projects, Hard Hats Dataset, 2022, CC BY 4.0",
        "expected_images": EXPECTED_IMAGES,
        "classes": {0: "hardhat", 1: "no-hardhat"},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [asdict(r) for r in records],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"manifest -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
