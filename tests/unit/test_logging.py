from __future__ import annotations

import logging

from kevlargrid.utils.logging import ColoredFormatter, get_logger


def test_get_logger() -> None:
    """Verify that get_logger returns a properly scoped logger child."""
    logger = get_logger("test_module")
    assert logger.name == "kevlargrid.test_module"
    assert isinstance(logger, logging.Logger)


def test_colored_formatter() -> None:
    """Verify that ColoredFormatter applies colors correctly to records."""
    formatter = ColoredFormatter("%(levelname)s: %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "\033[32mINFO\033[0m" in formatted
    assert "Test message" in formatted
    # Check that it restored the original levelname so non-colored handlers are unaffected
    assert record.levelname == "INFO"
