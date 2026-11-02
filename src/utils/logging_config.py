"""Centralized logging configuration for FINRISK NEXUS."""

import logging
import os
from pathlib import Path


def configure_logging(
    level: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure console logging and optional file logging."""
    log_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, None)

    if not isinstance(log_level, int):
        raise ValueError(f"Invalid logging level: {log_level_name}")

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(message)s",
        force=True,
    )

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
