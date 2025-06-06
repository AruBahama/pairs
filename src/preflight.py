from __future__ import annotations

import sys
from pathlib import Path

from .config import TICKER_FILE


def run_checks() -> None:
    """Ensure environment is ready before running scripts."""
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or higher is required")

    if not Path(TICKER_FILE).exists():
        sys.exit(f"Ticker file {TICKER_FILE} not found")
