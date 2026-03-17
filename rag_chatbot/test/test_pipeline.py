from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .. import pipeline as pipeline_module


@pytest.fixture
def pipeline_with_mocks(monkeypatch):
    engine = MagicMock()
    ingestion = MagicMock()
    vector_store = MagicMock()

    # Defaults used by constructor/setters.
    ingestion.check_nodes_exist.return_value = False
    ingestion.get_ingested_nodes.return_value = ["n1", "n2"]

    model_initial = SimpleNamespace(name="model-initial")
    model_global = SimpleNamespace(name="model-global")
    embedding_default = SimpleNamespace(name="embed-default")

    model_set_mock = MagicMock(side_effect=[model_initial, model_global])
    embedding_set_mock = MagicMock(return_value=embedding_default)

    get_system_prompt_mock = MagicMock(return_value="system-prompt")

    monkeypatch.setattr(pipeline_module, "LocalChatEngine", MagicMock(return_value=engine))
    monkeypatch.setattr(
        pipeline_module, "LocalDataIngestion", MagicMock(return_value=ingestion)
    )
    monkeypatch.setattr(
        pipeline_module, "LocalVectorStore", MagicMock(return_value=vector_store)
    )
    monkeypatch.setattr(
        pipeline_module, "LocalRAGModel", SimpleNamespace(set=model_set_mock)
    )
    monkeypatch.setattr(
        pipeline_module, "LocalEmbedding", SimpleNamespace(set=embedding_set_mock)
    )
    monkeypatch.setattr(pipeline_module, "get_system_prompt", get_system_prompt_mock)

    pipe = pipeline_module.LocalRAGPipeline(host="test-host")

    return {
        "pipe": pipe,
        "engine": engine,
        "ingestion": ingestion,
        "vector_store": vector_store,
        "model_set_mock": model_set_mock,
        "embedding_set_mock": embedding_set_mock,
        "get_system_prompt_mock": get_system_prompt_mock,
    }


class TestPipelineInitialization:
    def test_initial_state(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        assert pipe.get_language() == "eng"
        assert pipe.get_model_name() == ""
        assert pipe.get_system_prompt() == "system-prompt"

    def test_constructor_calls_dependencies(self, pipeline_with_mocks):
        model_set_mock = pipeline_with_mocks["model_set_mock"]
        embedding_set_mock = pipeline_with_mocks["embedding_set_mock"]
        get_system_prompt_mock = pipeline_with_mocks["get_system_prompt_mock"]

        assert model_set_mock.call_count == 2
        assert embedding_set_mock.call_count == 1
        get_system_prompt_mock.assert_called_once_with("eng", is_rag_prompt=False)


class TestPipelineSetters:
    def test_set_language_updates_state(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        pipe.set_language("span")
        assert pipe.get_language() == "span"

    def test_set_model_name_updates_state(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        pipe.set_model_name("mistral:7b-instruct-q4_0")
        assert pipe.get_model_name() == "mistral:7b-instruct-q4_0"

    def test_set_system_prompt_uses_custom_value(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        get_system_prompt_mock = pipeline_with_mocks["get_system_prompt_mock"]

        get_system_prompt_mock.reset_mock()
        pipe.set_system_prompt("manual prompt")

        assert pipe.get_system_prompt() == "manual prompt"
        get_system_prompt_mock.assert_not_called()

    def test_set_system_prompt_uses_language_and_ingestion_when_none(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        ingestion = pipeline_with_mocks["ingestion"]
        get_system_prompt_mock = pipeline_with_mocks["get_system_prompt_mock"]

        ingestion.check_nodes_exist.return_value = True
        pipe.set_language("span")
        pipe.set_system_prompt(None)

        get_system_prompt_mock.assert_called_with(language="span", is_rag_prompt=True)


class TestPipelineEngineAndModel:
    def test_set_model_calls_model_factory(self, pipeline_with_mocks, monkeypatch):
        pipe = pipeline_with_mocks["pipe"]
        model_set_mock = MagicMock(return_value=SimpleNamespace(name="new-model"))
        monkeypatch.setattr(
            pipeline_module, "LocalRAGModel", SimpleNamespace(set=model_set_mock)
        )

        pipe.set_model_name("llama3.1:8b-instruct-q8_0")
        pipe.set_system_prompt("my prompt")
        pipe.set_model()

        model_set_mock.assert_called_once_with(
            model_name="llama3.1:8b-instruct-q8_0",
            system_prompt="my prompt",
            host="test-host",
        )

    def test_set_engine_uses_nodes_and_language(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        engine = pipeline_with_mocks["engine"]
        ingestion = pipeline_with_mocks["ingestion"]

        engine.set_engine.return_value = "query-engine"
        pipe.set_language("span")
        pipe.set_engine()

        engine.set_engine.assert_called_once_with(
            llm=pipe._default_model,
            nodes=ingestion.get_ingested_nodes.return_value,
            language="span",
        )
        assert pipe._query_engine == "query-engine"

    def test_reset_engine_uses_empty_nodes(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        engine = pipeline_with_mocks["engine"]

        pipe.reset_engine()

        engine.set_engine.assert_called_once_with(
            llm=pipe._default_model,
            nodes=[],
            language="eng",
        )

    def test_set_chat_mode_orchestrates_core_steps(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        pipe.set_language = MagicMock()
        pipe.set_system_prompt = MagicMock()
        pipe.set_model = MagicMock()
        pipe.set_engine = MagicMock()

        pipe.set_chat_mode("override-prompt")

        pipe.set_language.assert_called_once_with("eng")
        pipe.set_system_prompt.assert_called_once_with("override-prompt")
        pipe.set_model.assert_called_once()
        pipe.set_engine.assert_called_once()


class TestPipelineQueryFlow:
    def test_get_history_transforms_pairs(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]

        history = pipe.get_history([["hola", "respuesta"], [None, "skip"]])

        assert len(history) == 2
        assert history[0].content == "hola"
        assert history[1].content == "respuesta"

    def test_query_chat_uses_history(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        query_engine = MagicMock()
        query_engine.stream_chat.return_value = "chat-response"
        pipe._query_engine = query_engine

        result = pipe.query("chat", "hola", [["q1", "a1"]])

        assert result == "chat-response"
        assert query_engine.reset.call_count == 0
        query_engine.stream_chat.assert_called_once()
        args = query_engine.stream_chat.call_args.args
        assert args[0] == "hola"
        assert len(args[1]) == 2

    def test_query_qa_resets_before_answer(self, pipeline_with_mocks):
        pipe = pipeline_with_mocks["pipe"]
        query_engine = MagicMock()
        query_engine.stream_chat.return_value = "qa-response"
        pipe._query_engine = query_engine

        result = pipe.query("QA", "pregunta", [])

        assert result == "qa-response"
        query_engine.reset.assert_called_once()
        query_engine.stream_chat.assert_called_once_with("pregunta")
