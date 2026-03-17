from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..core.ingestion import ingestion as ingestion_module


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self, _mode: str):
        return self._text


class FakeDocumentObj:
    def __init__(self, pages):
        self._pages = pages

    def __iter__(self):
        return iter(self._pages)


class FakeLlamaDocument:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


@pytest.fixture
def ingestion_with_mocks(monkeypatch):
    setting = SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=100,
            chunk_overlap=10,
            paragraph_sep="\n\n",
            chunking_regex=r"[^,.;]+",
        )
    )

    splitter_mock = MagicMock(return_value=["node-1", "node-2"])
    splitter_factory_mock = MagicMock(return_value=splitter_mock)
    monkeypatch.setattr(
        ingestion_module.SentenceSplitter,
        "from_defaults",
        splitter_factory_mock,
    )

    fitz_open_mock = MagicMock(
        return_value=FakeDocumentObj([FakePage("Hola   mundo\n\nPDF")])
    )
    monkeypatch.setattr(ingestion_module.fitz, "open", fitz_open_mock)

    monkeypatch.setattr(ingestion_module, "Document", FakeLlamaDocument)

    embed_model_mock = MagicMock(side_effect=lambda nodes, show_progress=True: [f"emb-{n}" for n in nodes])
    monkeypatch.setattr(ingestion_module.Settings, "embed_model", embed_model_mock)

    monkeypatch.setattr(ingestion_module, "tqdm", lambda items, desc=None: items)

    ing = ingestion_module.LocalDataIngestion(setting=setting)
    return {
        "ing": ing,
        "splitter_mock": splitter_mock,
        "splitter_factory_mock": splitter_factory_mock,
        "fitz_open_mock": fitz_open_mock,
        "embed_model_mock": embed_model_mock,
    }


class TestFilterText:
    def test_filter_text_normalizes_spaces(self):
        ing = ingestion_module.LocalDataIngestion(setting=SimpleNamespace(ingestion=SimpleNamespace()))
        result = ing._filter_text("Hola\n\n   mundo\t123")
        assert result == "Hola mundo 123"


class TestStoreNodes:
    def test_store_nodes_empty_returns_empty(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]

        result = ing.store_nodes([])

        assert result == []
        assert ing._ingested_file == []

    def test_store_nodes_builds_nodes_and_embeds(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        splitter_mock = ingestion_with_mocks["splitter_mock"]
        fitz_open_mock = ingestion_with_mocks["fitz_open_mock"]

        result = ing.store_nodes(["/tmp/file_a.pdf"], embed_nodes=True)

        assert result == ["emb-node-1", "emb-node-2"]
        fitz_open_mock.assert_called_once_with("/tmp/file_a.pdf")
        splitter_mock.assert_called_once()
        assert ing._ingested_file == ["file_a.pdf"]
        assert ing._node_store["file_a.pdf"] == ["emb-node-1", "emb-node-2"]

    def test_store_nodes_uses_cache_without_reopening_file(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        fitz_open_mock = ingestion_with_mocks["fitz_open_mock"]

        first = ing.store_nodes(["/tmp/file_cached.pdf"], embed_nodes=False)
        second = ing.store_nodes(["/tmp/file_cached.pdf"], embed_nodes=False)

        assert first == ["node-1", "node-2"]
        assert second == ["node-1", "node-2"]
        assert fitz_open_mock.call_count == 1

    def test_store_nodes_without_embedding_keeps_raw_nodes(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        embed_model_mock = ingestion_with_mocks["embed_model_mock"]

        result = ing.store_nodes(["/tmp/file_raw.pdf"], embed_nodes=False)

        assert result == ["node-1", "node-2"]
        assert embed_model_mock.call_count == 0

    def test_store_nodes_accepts_custom_embed_model(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        custom_embed = MagicMock(return_value=["custom-1", "custom-2"])

        result = ing.store_nodes(
            ["/tmp/file_custom.pdf"],
            embed_nodes=True,
            embed_model=custom_embed,
        )

        assert result == ["custom-1", "custom-2"]
        custom_embed.assert_called_once_with(["node-1", "node-2"], show_progress=True)


class TestNodeStoreUtilities:
    def test_check_nodes_exist_false_then_true(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]

        assert ing.check_nodes_exist() is False
        ing._node_store = {"a.pdf": ["n1"]}
        assert ing.check_nodes_exist() is True

    def test_get_all_nodes_flattens_store(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        ing._node_store = {
            "a.pdf": ["a1", "a2"],
            "b.pdf": ["b1"],
        }

        result = ing.get_all_nodes()

        assert result == ["a1", "a2", "b1"]

    def test_get_ingested_nodes_returns_only_current_file_order(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        ing._node_store = {
            "a.pdf": ["a1"],
            "b.pdf": ["b1", "b2"],
        }
        ing._ingested_file = ["b.pdf", "a.pdf"]

        result = ing.get_ingested_nodes()

        assert result == ["b1", "b2", "a1"]

    def test_reset_clears_store_and_ingested_files(self, ingestion_with_mocks):
        ing = ingestion_with_mocks["ing"]
        ing._node_store = {"a.pdf": ["a1"]}
        ing._ingested_file = ["a.pdf"]

        ing.reset()

        assert ing._node_store == {}
        assert ing._ingested_file == []
