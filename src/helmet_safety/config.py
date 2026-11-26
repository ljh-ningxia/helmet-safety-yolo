"""Typed, validated configuration.

Every runtime knob lives in ``configs/*.yaml``. Nothing is read from a bare
environment variable deep inside business logic: configuration is resolved
once, validated, and passed down explicitly. This makes experiments replayable.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or has invalid values."""


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset_yaml: str = "configs/data/hardhat.yaml"
    classes: tuple = ("hardhat", "no-hardhat")
    image_size: int = 640
    splits: tuple = ("train", "valid", "test")
    seed: int = 20261126
    subset_train: Optional[int] = None
    subset_valid: Optional[int] = None
    subset_test: Optional[int] = None
    val_fraction: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    model: str = "yolov8n.pt"
    epochs: int = 30
    imgsz: int = 640
    batch: int = 16
    workers: int = 4
    device: str = "cpu"
    patience: int = 10
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    seed: int = 20261126
    deterministic: bool = True
    project: str = "experiments/runs"
    name: str = "yolov8n-hardhat"
    exist_ok: bool = True
    amp: bool = False
    mosaic: float = 1.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5


@dataclass(frozen=True)
class EvalConfig:
    weights: str = "experiments/runs/yolov8n-hardhat/weights/best.pt"
    data: str = "configs/data/hardhat.yaml"
    split: str = "test"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.6
    device: str = "cpu"
    batch: int = 8
    benchmark_warmup: int = 5
    benchmark_repeats: int = 30
    seed: int = 20261126
    out_dir: str = "experiments/results"


@dataclass(frozen=True)
class InferenceConfig:
    weights: str = "experiments/runs/yolov8n-hardhat/weights/best.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str = "cpu"
    enable_tiling: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.25
    enable_temporal: bool = True
    temporal_window: int = 5
    temporal_min_votes: int = 3
    temporal_iou: float = 0.45


@dataclass(frozen=True)
class AlertConfig:
    cooldown_frames: int = 30
    enable_dedup: bool = True
    level_2_count: int = 2
    level_3_count: int = 4
    snapshot_dir: str = "experiments/results/snapshots"
    sinks: tuple = ("jsonl",)


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    env: str = "dev"

    def to_dict(self) -> Dict[str, Any]:
        def conv(v: Any) -> Any:
            if is_dataclass(v):
                return {f.name: conv(getattr(v, f.name)) for f in fields(v)}
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, Path):
                return str(v)
            return copy.deepcopy(v)
        return conv(self)

    def resolve(self, p: str | Path) -> Path:
        """Resolve a project-relative path to an absolute one."""
        q = Path(p)
        return q if q.is_absolute() else (ROOT / q)


_SECTIONS = {"data": DataConfig, "train": TrainConfig, "eval": EvalConfig,
             "inference": InferenceConfig, "alert": AlertConfig}


def _build(cls, values: Dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    coerced = {}
    for k, v in values.items():
        f = next(f for f in fields(cls) if f.name == k)
        annotation = f.type
        if isinstance(annotation, str) and "tuple" in annotation and isinstance(v, list):
            v = tuple(v)
        coerced[k] = v
    return cls(**coerced)


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                env: Optional[str] = None) -> AppConfig:
    """Load all section configs, apply an optional env overlay and explicit overrides.

    Override precedence: explicit ``overrides`` > env overlay > section defaults.
    """
    sections: Dict[str, Dict[str, Any]] = {}
    for name in _SECTIONS:
        fp = CONFIG_DIR / name / "default.yaml"
        raw: Dict[str, Any] = {}
        if fp.exists():
            with fp.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, dict):
                raise ConfigError(f"{fp}: expected a mapping")
        sections[name] = raw

    target_env = env or os.environ.get("HELMET_ENV", "dev")
    for name in _SECTIONS:
        env_fp = CONFIG_DIR / name / f"{target_env}.yaml"
        if env_fp.exists():
            with env_fp.open("r", encoding="utf-8") as f:
                overlay = yaml.safe_load(f) or {}
            if not isinstance(overlay, dict):
                raise ConfigError(f"{env_fp}: expected a mapping")
            sections[name].update(overlay)

    if overrides:
        for name, vals in overrides.items():
            if name not in _SECTIONS:
                raise ConfigError(f"unknown config section: {name}")
            sections[name].update(vals)

    built = {name: _build(cls, sections[name]) for name, cls in _SECTIONS.items()}
    cfg = AppConfig(**built, env=target_env)
    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    if cfg.data.image_size <= 0 or cfg.train.imgsz <= 0:
        raise ConfigError("image sizes must be positive")
    if not 0.0 <= cfg.inference.conf <= 1.0:
        raise ConfigError("inference.conf must be in [0, 1]")
    if not 0.0 <= cfg.inference.tile_overlap < 1.0:
        raise ConfigError("inference.tile_overlap must be in [0, 1)")
    if cfg.inference.temporal_min_votes > cfg.inference.temporal_window:
        raise ConfigError("temporal_min_votes cannot exceed temporal_window")
    if not 0.0 <= cfg.data.val_fraction < 1.0:
        raise ConfigError("data.val_fraction must be in [0, 1)")
    for key in ("subset_train", "subset_valid", "subset_test"):
        v = getattr(cfg.data, key)
        if v is not None and v <= 0:
            raise ConfigError(f"data.{key} must be positive when set")
