"""Logging setup shared by the command-line entry points."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Send log records to stdout as bare messages, so command output reads as before.

    Warnings and errors go to stderr instead. Safe to call more than once: it replaces any
    handlers already on the root logger.

    Args:
        level: Lowest severity to show.
    """
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda record: record.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    logging.basicConfig(level=level, format="%(message)s", handlers=[out, err], force=True)
