#!/usr/bin/env python3
"""Fetch and pin the YOLOv8n pretrained weights.

Ultralytics auto-downloads weights through ``requests``/certifi, which fails on
networks with a TLS-terminating proxy. We fetch through the system urllib
channel instead, record the SHA256 and point the config at a local file so
training is fully offline and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "yolov8n.pt"
MANIFEST = ROOT / "models" / "weights_manifest.json"

SOURCES = [
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt",
]
MIN_BYTES = 1_000_000


def log(m: str) -> None:
    print(f"[weights] {m}", flush=True)


def download() -> dict:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size >= MIN_BYTES:
        log(f"already present: {OUT} ({OUT.stat().st_size} bytes)")
    else:
        last = None
        for url in SOURCES:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                log(f"fetching {url}")
                with urllib.request.urlopen(req, timeout=180) as r, OUT.open("wb") as f:
                    total = 0
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        total += len(buf)
                if total < MIN_BYTES:
                    raise IOError(f"too small: {total} bytes")
                log(f"downloaded {total} bytes from {url}")
                break
            except Exception as e:  # noqa: BLE001
                last = e
                OUT.unlink(missing_ok=True)
                log(f"failed: {type(e).__name__}: {e}")
        else:
            raise RuntimeError(f"all sources failed: {last}")

    h = hashlib.sha256()
    with OUT.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    rec = {
        "file": "models/yolov8n.pt",
        "size_bytes": OUT.stat().st_size,
        "sha256": h.hexdigest(),
        "source": "ultralytics/assets release v8.4.0",
        "license": "AGPL-3.0 (Ultralytics pretrained weights) — see NOTICE.md",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    MANIFEST.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"sha256={rec['sha256']}")
    return rec


if __name__ == "__main__":
    print(json.dumps(download(), ensure_ascii=False, indent=2))
    sys.exit(0)
