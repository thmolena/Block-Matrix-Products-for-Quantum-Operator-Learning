"""Run the authenticated study and regenerate the result directory."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from bmqol.experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
