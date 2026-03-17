import pytest
from ..setting.setting import RAGSettings, OllamaSettings, OpenAISettings


def test_rag_settings_creation():
    """Test that RAGSettings can be instantiated with defaults."""
    settings = RAGSettings()
    assert settings.ollama.temperature == 0.1
    assert settings.openai.temperature == 0.7
    assert isinstance(settings.ollama, OllamaSettings)
    assert isinstance(settings.openai, OpenAISettings)


def test_ollama_settings_defaults():
    """Test OllamaSettings default values."""
    ollama = OllamaSettings()
    assert ollama.llm == "llama3:8b-instruct-q8_0"
    assert ollama.port == 11434
    assert ollama.temperature == 0.1


def test_openai_settings_defaults():
    """Test OpenAISettings default values."""
    openai = OpenAISettings()
    assert openai.temperature == 0.7
    assert openai.max_tokens == 1000
    assert openai.api_key == ""
