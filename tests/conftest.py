import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402

from wildfire_mas.config import load_settings  # noqa: E402


@pytest.fixture
def settings():
    return load_settings(ROOT)


@pytest.fixture
def fixtures():
    return ROOT / "tests" / "fixtures"
