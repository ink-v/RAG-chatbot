import pytest
from unittest.mock import MagicMock, patch
from ..ui.ui import DefaultElement, LocalChatbotUI


def make_ui() -> LocalChatbotUI:
    """Create a LocalChatbotUI with mocked dependencies (no real pipeline/logger/Gradio)."""
    pipeline = MagicMock()
    pipeline.get_system_prompt.return_value = "test prompt"
    logger = MagicMock()
    with patch("os.makedirs"), patch("os.path.exists", return_value=True):
        return LocalChatbotUI(pipeline=pipeline, logger=logger)


# ── DefaultElement constants ───────────────────────────────────────────────────

class TestDefaultElementConstants:
    def test_no_credit_english(self):
        assert DefaultElement.OPENAI_NO_CREDIT_STATUS_EN == "No credits"

    def test_no_credit_spanish(self):
        assert DefaultElement.OPENAI_NO_CREDIT_STATUS_ES == "No creditos"

    def test_model_access_status_is_string(self):
        assert isinstance(DefaultElement.OPENAI_MODEL_ACCESS_STATUS, str)
        assert len(DefaultElement.OPENAI_MODEL_ACCESS_STATUS) > 0

    def test_api_key_status_is_string(self):
        assert isinstance(DefaultElement.OPENAI_API_KEY_STATUS, str)

    def test_generic_status_is_string(self):
        assert isinstance(DefaultElement.OPENAI_GENERIC_STATUS, str)

    def test_completed_status(self):
        assert DefaultElement.COMPLETED_STATUS == "Completed!"

    def test_answering_status(self):
        assert DefaultElement.ANSWERING_STATUS == "Answering!"

    def test_default_history_is_empty_list(self):
        assert DefaultElement.DEFAULT_HISTORY == []

    def test_default_document_is_empty_list(self):
        assert DefaultElement.DEFAULT_DOCUMENT == []


# ── _map_openai_error_status ───────────────────────────────────────────────────

class TestMapOpenAIErrorStatus:
    @pytest.fixture
    def ui(self):
        return make_ui()

    # Quota / credit errors → eng
    def test_quota_exceeded_eng_returns_no_credits(self, ui):
        result = ui._map_openai_error_status(
            "You exceeded your current quota, please check your plan", "eng"
        )
        assert result == "No credits"

    def test_insufficient_quota_eng(self, ui):
        result = ui._map_openai_error_status("insufficient_quota error", "eng")
        assert result == "No credits"

    def test_rate_limit_eng(self, ui):
        result = ui._map_openai_error_status("rate_limit reached", "eng")
        assert result == "No credits"

    def test_billing_eng(self, ui):
        result = ui._map_openai_error_status("check your billing details", "eng")
        assert result == "No credits"

    # Quota / credit errors → span
    def test_quota_exceeded_span_returns_no_creditos(self, ui):
        result = ui._map_openai_error_status(
            "You exceeded your current quota, please check your plan", "span"
        )
        assert result == "No creditos"

    def test_insufficient_quota_span(self, ui):
        result = ui._map_openai_error_status("insufficient_quota error", "span")
        assert result == "No creditos"

    def test_rate_limit_span(self, ui):
        result = ui._map_openai_error_status("rate_limit reached", "span")
        assert result == "No creditos"

    # Model not found
    def test_model_not_found(self, ui):
        result = ui._map_openai_error_status(
            "The model `gpt-4` does not exist or you do not have access to it.", "eng"
        )
        assert result == DefaultElement.OPENAI_MODEL_ACCESS_STATUS

    def test_model_not_found_code(self, ui):
        result = ui._map_openai_error_status("model_not_found error", "span")
        assert result == DefaultElement.OPENAI_MODEL_ACCESS_STATUS

    # Invalid API key
    def test_invalid_api_key(self, ui):
        result = ui._map_openai_error_status("invalid_api_key provided", "eng")
        assert result == DefaultElement.OPENAI_API_KEY_STATUS

    def test_incorrect_api_key(self, ui):
        result = ui._map_openai_error_status("Incorrect API key provided", "eng")
        assert result == DefaultElement.OPENAI_API_KEY_STATUS

    # Generic fallback
    def test_unknown_error_returns_generic(self, ui):
        result = ui._map_openai_error_status("some unexpected server error", "eng")
        assert result == DefaultElement.OPENAI_GENERIC_STATUS

    def test_unknown_error_span_returns_generic(self, ui):
        result = ui._map_openai_error_status("timeout error occurred", "span")
        assert result == DefaultElement.OPENAI_GENERIC_STATUS

    # Case-insensitive matching
    def test_case_insensitive_quota(self, ui):
        result = ui._map_openai_error_status("EXCEEDED YOUR CURRENT QUOTA", "eng")
        assert result == "No credits"
