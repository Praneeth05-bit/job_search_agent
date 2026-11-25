"""Tests for application settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from job_search_agent.config import PROJECT_ROOT, Settings

SETTING_NAMES = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "MODEL_TIMEOUT_SECONDS",
    "MODEL_MAX_RETRIES",
    "USE_MOCK_MODEL",
    "MAX_TOOL_CALLS",
    "DATA_DIR",
    "TRACKER_PATH",
    "LOG_LEVEL",
    "LOG_JSON",
    "API_BASE_URL",
    "UI_REQUEST_TIMEOUT_SECONDS",
    "PRICE_INPUT_PER_MILLION",
    "PRICE_OUTPUT_PER_MILLION",
]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test with no real settings leaking in.

    Removes any SpartanScout variables set in the terminal and switches to an
    empty folder so a developer's real .env file is never read.
    """
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_are_safe() -> None:
    """With nothing configured, the app gets sensible defaults."""
    settings = Settings()

    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-3.6-flash"
    assert settings.model_timeout_seconds == 30.0
    assert settings.model_max_retries == 2
    assert settings.max_tool_calls == 5
    assert settings.use_mock_model is False
    assert settings.log_level == "INFO"
    assert settings.log_json is False
    assert settings.api_base_url == "http://127.0.0.1:8000"
    assert settings.ui_request_timeout_seconds == 120.0
    assert settings.price_input_per_million == 1.50
    assert settings.price_output_per_million == 7.50


def test_default_data_dir_is_absolute_and_points_to_demo_data() -> None:
    """The default path works no matter which folder the app starts from."""
    settings = Settings()

    assert settings.data_dir.is_absolute()
    assert settings.data_dir == PROJECT_ROOT / "data" / "demo" / "opportunities"


def test_default_tracker_path_is_absolute() -> None:
    """Approved shortlists are saved inside the project's data folder."""
    settings = Settings()

    assert settings.tracker_path.is_absolute()
    assert settings.tracker_path == PROJECT_ROOT / "data" / "tracker" / "shortlist.json"


def test_environment_variable_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values come from the environment and are converted to the right type."""
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("USE_MOCK_MODEL", "true")

    settings = Settings()

    assert settings.model_timeout_seconds == 12.5
    assert settings.use_mock_model is True


def test_env_file_is_read(tmp_path: Path) -> None:
    """A .env file in the working folder is loaded."""
    (tmp_path / ".env").write_text("MAX_TOOL_CALLS=3\n", encoding="utf-8")

    assert Settings().max_tool_calls == 3


def test_api_key_is_hidden_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key never appears in printed or logged output."""
    monkeypatch.setenv("GEMINI_API_KEY", "real-secret-value")

    settings = Settings()

    assert settings.gemini_api_key is not None
    assert "real-secret-value" not in str(settings.gemini_api_key)
    assert "real-secret-value" not in repr(settings)
    assert settings.gemini_api_key.get_secret_value() == "real-secret-value"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MODEL_TIMEOUT_SECONDS", "0"),
        ("MODEL_MAX_RETRIES", "6"),
        ("MAX_TOOL_CALLS", "0"),
        ("USE_MOCK_MODEL", "maybe"),
        ("LOG_LEVEL", "LOUD"),
        ("UI_REQUEST_TIMEOUT_SECONDS", "0"),
        ("PRICE_INPUT_PER_MILLION", "-1"),
    ],
)
def test_invalid_values_fail_at_startup(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Bad configuration is rejected immediately with a clear error."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings()
