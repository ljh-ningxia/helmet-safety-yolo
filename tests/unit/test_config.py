"""Unit tests for configuration loading and validation."""
from __future__ import annotations

import pytest

from helmet_safety.config import ROOT, ConfigError, load_config


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg.data.classes == ("hardhat", "no-hardhat")
    assert cfg.data.seed == 20261126
    assert cfg.resolve("README.md") == ROOT / "README.md"
    assert cfg.to_dict()["train"]["epochs"] == cfg.train.epochs


def test_override_precedence():
    cfg = load_config(overrides={"train": {"epochs": 3}})
    assert cfg.train.epochs == 3
    assert cfg.train.imgsz == 320


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"train": {"not_a_real_option": 1}})
    with pytest.raises(ConfigError):
        load_config(overrides={"nope": {}})


def test_invalid_values_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"inference": {"conf": 2.0}})
    with pytest.raises(ConfigError):
        load_config(overrides={"inference": {"temporal_min_votes": 99}})
    with pytest.raises(ConfigError):
        load_config(overrides={"data": {"subset_train": -1}})


def test_list_coerced_to_tuple():
    cfg = load_config(overrides={"data": {"classes": ["a", "b"]}})
    assert cfg.data.classes == ("a", "b")
