import pytest
from ..core.prompt.qa_prompt import (
    get_system_prompt,
    get_context_prompt,
    get_condensed_context_prompt,
    SYSTEM_PROMPT_ES,
    SYSTEM_PROMPT_RAG_ES,
    CONTEXT_PROMPT_ES,
    CONDENSED_CONTEXT_PROMPT_ES,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_RAG_EN,
    CONTEXT_PROMPT_EN,
    CONDENSED_CONTEXT_PROMPT_EN,
)
from ..core.prompt.query_gen_prompt import get_query_gen_prompt
from ..core.prompt.select_prompt import get_single_select_prompt


# ── get_system_prompt ──────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    def test_eng_rag_returns_english(self):
        result = get_system_prompt("eng", is_rag_prompt=True)
        assert result == SYSTEM_PROMPT_RAG_EN

    def test_eng_no_rag_returns_english(self):
        result = get_system_prompt("eng", is_rag_prompt=False)
        assert result == SYSTEM_PROMPT_EN

    def test_span_rag_returns_spanish(self):
        result = get_system_prompt("span", is_rag_prompt=True)
        assert result == SYSTEM_PROMPT_RAG_ES

    def test_span_no_rag_returns_spanish(self):
        result = get_system_prompt("span", is_rag_prompt=False)
        assert result == SYSTEM_PROMPT_ES

    def test_span_prompt_contains_mandatory_spanish_rule(self):
        for prompt in (SYSTEM_PROMPT_ES, SYSTEM_PROMPT_RAG_ES):
            assert "español" in prompt.lower()

    def test_eng_prompt_does_not_contain_spanish(self):
        for prompt in (SYSTEM_PROMPT_EN, SYSTEM_PROMPT_RAG_EN):
            assert "español" not in prompt.lower()

    def test_unknown_language_defaults_to_english(self):
        result = get_system_prompt("fr", is_rag_prompt=True)
        assert result == SYSTEM_PROMPT_RAG_EN

    def test_default_is_rag_prompt_true(self):
        assert get_system_prompt("eng") == get_system_prompt("eng", is_rag_prompt=True)


# ── get_context_prompt ─────────────────────────────────────────────────────────

class TestGetContextPrompt:
    def test_eng_returns_english(self):
        assert get_context_prompt("eng") == CONTEXT_PROMPT_EN

    def test_span_returns_spanish(self):
        assert get_context_prompt("span") == CONTEXT_PROMPT_ES

    def test_spanish_context_contains_placeholder(self):
        assert "{context_str}" in CONTEXT_PROMPT_ES

    def test_english_context_contains_placeholder(self):
        assert "{context_str}" in CONTEXT_PROMPT_EN

    def test_spanish_context_instructs_in_spanish(self):
        assert "español" in CONTEXT_PROMPT_ES.lower() or "instrucción" in CONTEXT_PROMPT_ES.lower()

    def test_unknown_language_defaults_to_english(self):
        assert get_context_prompt("de") == CONTEXT_PROMPT_EN


# ── get_condensed_context_prompt ───────────────────────────────────────────────

class TestGetCondensedContextPrompt:
    def test_eng_returns_english(self):
        assert get_condensed_context_prompt("eng") == CONDENSED_CONTEXT_PROMPT_EN

    def test_span_returns_spanish(self):
        assert get_condensed_context_prompt("span") == CONDENSED_CONTEXT_PROMPT_ES

    def test_spanish_condensed_contains_placeholders(self):
        assert "{chat_history}" in CONDENSED_CONTEXT_PROMPT_ES
        assert "{question}" in CONDENSED_CONTEXT_PROMPT_ES

    def test_english_condensed_contains_placeholders(self):
        assert "{chat_history}" in CONDENSED_CONTEXT_PROMPT_EN
        assert "{question}" in CONDENSED_CONTEXT_PROMPT_EN

    def test_unknown_language_defaults_to_english(self):
        assert get_condensed_context_prompt("ja") == CONDENSED_CONTEXT_PROMPT_EN


# ── get_query_gen_prompt ───────────────────────────────────────────────────────

class TestGetQueryGenPrompt:
    def test_eng_returns_prompt_template(self):
        prompt = get_query_gen_prompt("eng")
        assert prompt is not None
        assert "{num_queries}" in prompt.template
        assert "{query}" in prompt.template

    def test_span_returns_prompt_template(self):
        prompt = get_query_gen_prompt("span")
        assert prompt is not None
        assert "{num_queries}" in prompt.template
        assert "{query}" in prompt.template

    def test_span_template_contains_mandatory_spanish_rule(self):
        prompt = get_query_gen_prompt("span")
        assert "español" in prompt.template.lower()

    def test_span_and_eng_are_different_templates(self):
        assert get_query_gen_prompt("span").template != get_query_gen_prompt("eng").template

    def test_unknown_language_defaults_to_english(self):
        assert get_query_gen_prompt("zz").template == get_query_gen_prompt("eng").template


# ── get_single_select_prompt ───────────────────────────────────────────────────

class TestGetSingleSelectPrompt:
    def test_eng_returns_english_string(self):
        prompt = get_single_select_prompt("eng")
        assert isinstance(prompt, str)
        assert "{num_choices}" in prompt
        assert "{query_str}" in prompt

    def test_span_returns_spanish_string(self):
        prompt = get_single_select_prompt("span")
        assert isinstance(prompt, str)
        assert "{num_choices}" in prompt
        assert "{query_str}" in prompt

    def test_span_contains_mandatory_spanish_rule(self):
        prompt = get_single_select_prompt("span")
        assert "español" in prompt.lower()

    def test_span_and_eng_are_different(self):
        assert get_single_select_prompt("span") != get_single_select_prompt("eng")

    def test_unknown_language_defaults_to_english(self):
        assert get_single_select_prompt("xx") == get_single_select_prompt("eng")
