"""Exploratory data analysis and honest dataset auditing.

Produces (a) a machine-readable audit JSON, (b) diagnostic charts and (c) a
contact sheet of *real* images with ground-truth boxes. Everything here is
computed from the downloaded pixels and labels — no hard-coded statistics.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
from PIL import Image, ImageDraw, ImageFont

from ..logging import get_logger

log = get_logger(__name__)

FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
GREEN = (15, 155, 108)
RED = (200, 60, 50)
INK = (28, 43, 38)


def _font(sz: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    p = FONT_BOLD if bold and Path(FONT_BOLD).exists() else FONT
    try:
        return ImageFont.truetype(p, sz)
    except OSError:
        return ImageFont.load_default()


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def audit_yolo(root: Path, classes: tuple, out_dir: Path,
               contact_sheet_n: int = 8, seed: int = 20261126) -> dict:
    """Audit a YOLO-layout dataset rooted at ``root``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, dict] = {}
    all_areas: List[float] = []
    all_ratios: List[float] = []
    all_per_image: List[int] = []
    brightness: List[float] = []
    class_counter: Counter = Counter()
    small_medium_large = Counter()
    duplicates: Dict[str, List[str]] = {}
    digest_seen: Dict[str, str] = {}

    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        n_boxes, n_empty = 0, 0
        per_image: List[int] = []
        split_class: Counter = Counter()
        for p in imgs:
            lbl = lbl_dir / f"{p.stem}.txt"
            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()] if lbl.exists() else []
            per_image.append(len(lines))
            if not lines:
                n_empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    continue
                c = int(float(parts[0]))
                w, h = float(parts[3]), float(parts[4])
                name = classes[c] if c < len(classes) else f"unknown_{c}"
                class_counter[name] += 1
                split_class[name] += 1
                # COCO-style scale buckets, measured on 640px images
                pixel_area = w * h * 640 * 640
                all_areas.append(pixel_area)
                all_per_image.append(1)
                all_ratios.append(w / h if h > 0 else 0.0)
                if pixel_area < 32 ** 2:
                    small_medium_large["small"] += 1
                elif pixel_area < 96 ** 2:
                    small_medium_large["medium"] += 1
                else:
                    small_medium_large["large"] += 1
                n_boxes += 1
            # light statistics for scene slicing
            bgr = cv2.imread(str(p))
            if bgr is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean()))
            # exact-duplicate detection by content hash
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in digest_seen:
                duplicates.setdefault(digest, [digest_seen[digest]]).append(p.name)
            else:
                digest_seen[digest] = p.name

        all_per_image.extend(per_image)
        splits[split] = {
            "images": len(imgs),
            "boxes": n_boxes,
            "empty_images": n_empty,
            "boxes_per_image_mean": round(sum(per_image) / max(len(per_image), 1), 3),
            "boxes_per_image_p50": round(_quantile(per_image, 0.5), 2),
            "boxes_per_image_p95": round(_quantile(per_image, 0.95), 2),
            "boxes_per_image_max": max(per_image, default=0),
            "class_counts": dict(split_class),
        }

    dup_groups = {k: v for k, v in duplicates.items() if len(v) > 1}
    total_boxes = sum(class_counter.values())
    audit = {
        "root": str(root.resolve()).replace("\\", "/"),
        "classes": list(classes),
        "splits": splits,
        "total_images": sum(s["images"] for s in splits.values()),
        "total_boxes": total_boxes,
        "class_counts": dict(class_counter),
        "class_ratio": {k: round(v / max(total_boxes, 1), 4) for k, v in class_counter.items()},
        "imbalance_ratio": round(max(class_counter.values(), default=1) /
                                 max(min(class_counter.values(), default=1), 1), 2),
        "scale_buckets_px": dict(small_medium_large),
        "bbox_area_px2": {
            "p10": round(_quantile(all_areas, 0.10), 1),
            "p50": round(_quantile(all_areas, 0.50), 1),
            "p90": round(_quantile(all_areas, 0.90), 1),
        },
        "bbox_aspect_ratio": {
            "p10": round(_quantile(all_ratios, 0.10), 3),
            "p50": round(_quantile(all_ratios, 0.50), 3),
            "p90": round(_quantile(all_ratios, 0.90), 3),
        },
        "image_brightness_mean": {
            "p10": round(_quantile(brightness, 0.10), 1),
            "p50": round(_quantile(brightness, 0.50), 1),
            "p90": round(_quantile(brightness, 0.90), 1),
            "dark_lt_60": sum(1 for b in brightness if b < 60),
            "bright_gt_180": sum(1 for b in brightness if b > 180),
        },
        "exact_duplicate_groups": len(dup_groups),
        "exact_duplicate_examples": list(dup_groups.values())[:5],
        "notes": [
            "Upstream applied auto-orientation and a stretched resize to 640x640; aspect ratio is NOT preserved.",
            "No augmentation was applied by the upstream exporter (declared in the dataset card).",
            "Scale buckets use COCO thresholds on 640x640 pixel area.",
            "Brightness is a crude luminance proxy, not a labelled day/night tag.",
        ],
    }
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    _make_charts(audit, out_dir)
    sheet = _make_contact_sheet(root, classes, out_dir / "dataset_samples.png",
                                n=contact_sheet_n, seed=seed)
    audit["contact_sheet"] = str(sheet).replace("\\", "/") if sheet else ""
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("audit complete", images=audit["total_images"], boxes=total_boxes,
             duplicates=len(dup_groups))
    return audit


def _make_charts(audit: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:  # noqa: BLE001
        log.warning("matplotlib unavailable, charts skipped", error=str(e))
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.patch.set_facecolor("white")

    ax = axes[0][0]
    names = list(audit["class_counts"])
    vals = [audit["class_counts"][n] for n in names]
    ax.bar(names, vals, color=["#0F9B6C", "#C83C32"])
    ax.set_title("Class distribution (labelled boxes)")
    ax.set_ylabel("boxes")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    ax = axes[0][1]
    splits = list(audit["splits"])
    ax.bar(splits, [audit["splits"][s]["images"] for s in splits], color="#0F9B6C")
    ax.set_title("Images per split")
    for i, s in enumerate(splits):
        ax.text(i, audit["splits"][s]["images"], f"{audit['splits'][s]['images']:,}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1][0]
    scale = audit["scale_buckets_px"]
    ax.bar(list(scale), list(scale.values()), color=["#D2A429", "#3F8FD2", "#0F9B6C"])
    ax.set_title("Object scale (COCO buckets)")

    ax = axes[1][1]
    b = audit["image_brightness_mean"]
    ax.barh(["p10", "p50", "p90"], [b["p10"], b["p50"], b["p90"]], color="#5B7F72")
    ax.set_title("Image luminance percentiles")
    ax.set_xlim(0, 255)

    fig.suptitle("Hard Hat Detection — dataset audit (computed from real data)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_audit_charts.png", dpi=150)
    plt.close(fig)


def _make_contact_sheet(root: Path, classes: tuple, out_path: Path,
                        n: int = 8, seed: int = 20261126) -> Path | None:
    """Render real images with their ground-truth boxes — evidence, not mockups."""
    imgs = sorted((root / "images" / "test").glob("*.jpg"))
    if not imgs:
        imgs = sorted((root / "images" / "train").glob("*.jpg"))
    if not imgs:
        return None
    rng = random.Random(seed)
    # prefer images that contain at least one violation, but keep variety
    with_violation, other = [], []
    for p in imgs:
        lbl = root / "labels" / p.parent.name / f"{p.stem}.txt"
        txt = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        (with_violation if any(ln.startswith("1 ") for ln in txt.splitlines()) else other).append(p)
    rng.shuffle(with_violation)
    rng.shuffle(other)
    picked = (with_violation[: max(1, n // 2)] + other)[:n]
    rng.shuffle(picked)

    cell_w, cell_h, label_h = 420, 300, 30
    cols = 4
    rows = math.ceil(len(picked) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h) + 64), "white")
    d = ImageDraw.Draw(sheet)
    d.rectangle([0, 0, sheet.width, 52], fill=(245, 248, 246))
    d.text((18, 12), "真实数据集样本 · Ground truth 标注（Hard Hats Dataset, CC BY 4.0）",
           font=_font(21, True), fill=INK)
    for i, p in enumerate(picked):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell_w - 12, cell_h - 12))
        x0 = (i % cols) * cell_w + 6
        y0 = (i // cols) * (cell_h + label_h) + 60
        canvas = Image.new("RGB", (cell_w - 12, cell_h - 12), (238, 242, 240))
        canvas.paste(im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2))
        cd = ImageDraw.Draw(canvas)
        lbl = root / "labels" / p.parent.name / f"{p.stem}.txt"
        if lbl.exists():
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                c, cx, cy, w, h = (float(v) for v in ln.split()[:5])
                W, H = im.size
                ox, oy = (canvas.width - W) // 2, (canvas.height - H) // 2
                x1, y1 = ox + (cx - w / 2) * W, oy + (cy - h / 2) * H
                x2, y2 = ox + (cx + w / 2) * W, oy + (cy + h / 2) * H
                color = GREEN if int(c) == 0 else RED
                cd.rectangle([x1, y1, x2, y2], outline=color, width=3)
                cd.rectangle([x1, y1 - 17, x1 + 74, y1], fill=color)
                cd.text((x1 + 4, y1 - 16), classes[int(c)] if int(c) < len(classes) else str(c),
                        font=_font(11), fill="white")
        sheet.paste(canvas, (x0, y0))
        d.text((x0, y0 + cell_h - 8), f"{i+1:02d}  {p.stem[:34]}", font=_font(12), fill=(107, 125, 118))
    sheet.save(out_path)
    return out_path
