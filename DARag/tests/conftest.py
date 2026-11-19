import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import synthetic  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import Database  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, groq_api_key="test", _env_file=None)


@pytest.fixture
def db(settings):
    d = Database(settings.db_path)
    synthetic.seed_database(d, n_employees=80)
    return d
