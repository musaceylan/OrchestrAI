"""Keep configuration publication isolated between tests."""

import pytest

from orchestrai.config import settings as config


@pytest.fixture(autouse=True)
def isolated_settings_cache(monkeypatch):
    # Reload now publishes the singleton, including when candidate loading is
    # mocked. Restore it after each test so later tests never inherit a mock.
    monkeypatch.setattr(config, "_settings", None)
    # Tests must never read host dotenv values. Dotenv behavior is tested using
    # a stubbed source in test_config.py instead.
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
