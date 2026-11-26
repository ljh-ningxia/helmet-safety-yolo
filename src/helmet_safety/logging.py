"""Structured logging with a machine-readable JSON mode.

Production services emit JSON lines so logs can be indexed; local runs get a
compact human-readable format. Set ``HELMET_LOG_JSON=1`` to force JSON.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict

_CONFIGURED = False
RESERVED = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "getMessage", "taskName",
            # logging.makeRecord() also blacklists these two explicitly
            "message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in RESERVED and not k.startswith("_"):
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = " ".join(
            f"{k}={v}" for k, v in record.__dict__.items()
            if k not in RESERVED and not k.startswith("_")
        )
        return f"{base} | {extra}" if extra else base


def configure(level: str | None = None, json_mode: bool | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("HELMET_LOG_LEVEL", "INFO")).upper()
    use_json = json_mode if json_mode is not None else os.environ.get("HELMET_LOG_JSON") == "1"
    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
                                           "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, lvl, logging.INFO))
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    _CONFIGURED = True


class ContextAdapter(logging.LoggerAdapter):
    """Allows ``log.info("msg", split="train", n=10)`` structured fields.

    ``LoggerAdapter`` only merges ``self.extra`` by default and forwards any
    other keyword arguments straight to ``Logger._log``, which rejects them.
    Here every non-reserved keyword becomes a structured log field.
    """

    _RESERVED = {"exc_info", "stack_info", "stacklevel", "extra"}

    @staticmethod
    def _safe(key: str) -> str:
        # LogRecord owns these attributes; overwriting them raises KeyError.
        return key if key not in RESERVED else f"ctx_{key}"

    def process(self, msg, kwargs):
        fields = {self._safe(k): v for k, v in (self.extra or {}).items()}
        for key in list(kwargs):
            if key not in self._RESERVED:
                fields[self._safe(key)] = kwargs.pop(key)
        provided = kwargs.pop("extra", None) or {}
        kwargs["extra"] = {**fields, **provided}
        return msg, kwargs


def get_logger(name: str, **ctx: Any) -> logging.LoggerAdapter:
    configure()
    return ContextAdapter(logging.getLogger(name), ctx)
