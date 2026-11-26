#!/usr/bin/env python3
"""Train YOLOv8 on the prepared Hard Hat dataset with full run provenance.

Every run writes ``experiments/results/train_run.json`` containing the exact
command, resolved hyper-parameters, environment fingerprint, dataset checksum
reference and the metrics returned by Ultralytics. Nothing is estimated.

    python scripts/03_train.py --smoke                 # 1-epoch speed probe
    python scripts/03_train.py --epochs 20 --imgsz 320
    python scripts/03_train.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.config import load_config  # noqa: E402
from helmet_safety.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def environment_fingerprint() -> Dict[str, Any]:
    import cv2
    import numpy
    import torch
    import yaml
    try:
        import ultralytics
        uver = ultralytics.__version__
    except ImportError:
        uver = "NOT_INSTALLED"
    try:
        rev = os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip()
    except Exception:  # noqa: BLE001
        rev = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu" if not torch.cuda.is_available() else "cuda",
        "ultralytics": uver,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "pyyaml": yaml.__version__,
        "git_head": rev,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)  # some ops have no CPU deterministic impl
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--model", type=str)
    p.add_argument("--name", type=str)
    p.add_argument("--smoke", action="store_true", help="tiny 1-epoch run to measure CPU throughput")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str)
    p.add_argument("--workers", type=int)
    p.add_argument("--data", type=str, default=None, help="override data.yaml (for speed probes)")
    a = p.parse_args(argv)

    overrides: Dict[str, Dict[str, Any]] = {}
    cfg = load_config(overrides=overrides or None)
    t = cfg.train
    epochs = a.epochs if a.epochs is not None else (1 if a.smoke else t.epochs)
    imgsz = a.imgsz if a.imgsz is not None else (320 if a.smoke else t.imgsz)
    batch = a.batch if a.batch is not None else (8 if a.smoke else t.batch)
    model = a.model or t.model
    name = a.name or (f"smoke-{int(time.time())}" if a.smoke else t.name)
    device = a.device or t.device
    workers = a.workers if a.workers is not None else (0 if a.smoke else t.workers)

    if a.data:
        data_yaml = Path(a.data)
        if not data_yaml.is_absolute():
            data_yaml = cfg.resolve(data_yaml)
    else:
        data_yaml = cfg.resolve(cfg.data.processed_dir) / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"{data_yaml} missing — run 02_prepare_dataset.py first")

    set_global_seed(t.seed, t.deterministic)
    env = environment_fingerprint()
    if env["ultralytics"] == "NOT_INSTALLED":
        raise RuntimeError("ultralytics is not installed; run: pip install ultralytics")
    log.info("environment", **env)

    from ultralytics import YOLO  # imported late so --help works without the dep

    # Resolve a project-relative weight path; never let Ultralytics trigger a
    # network download behind our back (proxy TLS breaks it and it hurts reproducibility).
    model_path = Path(model)
    if not model_path.is_absolute() and not model_path.is_file():
        model_path = cfg.resolve(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"pretrained weights not found: {model}. "
                                "Run scripts/00_fetch_weights.py first.")
    model = str(model_path)

    project = cfg.resolve(t.project)
    project.mkdir(parents=True, exist_ok=True)
    log.info("starting training", model=model, epochs=epochs, imgsz=imgsz,
             batch=batch, device=device, workers=workers, run_name=name, data=str(data_yaml))

    yolo = YOLO(model)
    t0 = time.time()
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        seed=t.seed,
        deterministic=t.deterministic,
        lr0=t.lr0, lrf=t.lrf, weight_decay=t.weight_decay,
        warmup_epochs=t.warmup_epochs, patience=t.patience,
        amp=t.amp, mosaic=t.mosaic, hsv_h=t.hsv_h, hsv_s=t.hsv_s, hsv_v=t.hsv_v,
        fliplr=t.fliplr,
        project=str(project), name=name, exist_ok=t.exist_ok,
        resume=a.resume,
        verbose=not a.smoke,
        plots=not a.smoke,
    )
    elapsed = time.time() - t0

    run_dir = Path(str(results.save_dir)) if getattr(results, "save_dir", None) else project / name
    weights = run_dir / "weights" / "best.pt"
    metrics: Dict[str, Any] = {}
    try:
        rd = getattr(results, "results_dict", None) or {}
        metrics = {k: float(v) for k, v in rd.items() if isinstance(v, (int, float))}
    except Exception as e:  # noqa: BLE001
        log.warning("metric extraction failed", error=str(e))

    # Per-epoch curve from Ultralytics results.csv (real log, not a re-derivation)
    csv_path = run_dir / "results.csv"
    epochs_logged = 0
    final_row: Dict[str, str] = {}
    if csv_path.is_file():
        lines = [ln.strip() for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        epochs_logged = max(0, len(lines) - 1)
        if len(lines) >= 2:
            final_row = dict(zip([c.strip() for c in lines[0].split(",")],
                                 [c.strip() for c in lines[-1].split(",")]))

    speed = getattr(results, "speed", {}) or {}
    payload = {
        "evidence_level": "MEASURED",
        "run_type": "smoke_speed_probe" if a.smoke else "training",
        "command": " ".join([Path(sys.executable).name, "scripts/03_train.py", *(argv or [])]),
        "config": {
            "model": model, "epochs": epochs, "imgsz": imgsz, "batch": batch,
            "device": device, "workers": workers, "seed": t.seed,
            "deterministic": t.deterministic, "lr0": t.lr0, "lrf": t.lrf,
            "weight_decay": t.weight_decay, "patience": t.patience, "amp": t.amp,
            "mosaic": t.mosaic, "data_yaml": str(data_yaml.resolve()).replace("\\", "/"),
        },
        "environment": env,
        "elapsed_seconds": round(elapsed, 2),
        "epochs_logged": epochs_logged,
        "final_epoch_row": final_row,
        "metrics": metrics,
        "ultralytics_speed": {k: float(v) for k, v in speed.items() if isinstance(v, (int, float))},
        "run_dir": str(run_dir.resolve()).replace("\\", "/"),
        "best_weights": str(weights.resolve()).replace("\\", "/") if weights.is_file() else "",
        "artifacts": sorted(str(p.relative_to(run_dir).as_posix())
                            for p in run_dir.rglob("*") if p.is_file())[:60],
        "notes": [
            "Metrics come from the actual Ultralytics validation run on the prepared dataset.",
            "CPU-only training; throughput is not comparable to a GPU baseline.",
            "A smoke run is a throughput probe, NOT a reportable accuracy result.",
        ],
    }
    out = cfg.resolve("experiments/results") / ("train_smoke.json" if a.smoke else "train_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("training complete", elapsed_s=round(elapsed, 1), run_dir=str(run_dir),
             best_weights=str(weights), report=str(out))
    print(json.dumps({"elapsed_seconds": payload["elapsed_seconds"],
                      "epochs_logged": epochs_logged,
                      "metrics": metrics,
                      "best_weights": payload["best_weights"],
                      "report": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run())
