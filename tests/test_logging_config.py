import logging

import pytest
from src.utils.logging_config import configure_logging


def test_configure_logging_sets_level() -> None:
    configure_logging(level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Invalid logging level"):
        configure_logging(level="INVALID")
