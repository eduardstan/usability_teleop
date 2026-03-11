"""Project-owned logger with compact, pretty console output."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

_RESET = "\033[0m"
_COLORS = {
    "DEBUG": "\033[36m",   # cyan
    "INFO": "\033[32m",    # green
    "WARNING": "\033[33m", # yellow
    "ERROR": "\033[31m",   # red
    "CRITICAL": "\033[35m",# magenta
}
_SYMBOLS = {
    "DEBUG": "·",
    "INFO": "✓",
    "WARNING": "!",
    "ERROR": "x",
    "CRITICAL": "X",
}


@dataclass(frozen=True)
class LoggerConfig:
    """Runtime settings for pretty console logging."""

    level: int = logging.INFO
    use_color: bool = True


class PrettyFormatter(logging.Formatter):
    """Formatter that renders compact colored log lines."""

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        symbol = _SYMBOLS.get(level, "·")
        prefix = f"[{symbol} {record.name}]"
        message = record.getMessage()

        if self.use_color:
            color = _COLORS.get(level, "")
            return f"{color}{prefix}{_RESET} {message}"
        return f"{prefix} {message}"


def _should_use_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def get_logger(name: str, config: LoggerConfig | None = None) -> logging.Logger:
    """Get a configured project logger instance."""
    cfg = config or LoggerConfig(use_color=_should_use_color())

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(cfg.level)
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(PrettyFormatter(use_color=cfg.use_color))

    logger.setLevel(cfg.level)
    logger.propagate = False
    logger.addHandler(handler)
    return logger
