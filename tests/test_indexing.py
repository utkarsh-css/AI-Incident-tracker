from types import SimpleNamespace

import chromadb
import pytest

from incident_assistant.embeddings import EmbeddingError
from incident_assistant.indexing import (
    IndexErrorSafe,
    index_status,
    open_active_collection,
    rebuild_index,
)


class FakeEmbeddingClient:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, *, model, input):
        self.calls.append((model, list(input)))
        if self.fail:
            raise RuntimeError("private test error")
        vectors = []
        for text in input:
            # Deterministic, nonzero vectors for persistence and dimensions tests.
            vectors.append(SimpleNamespace(index=len(vectors), embedding=[float(len(text)), 1.0]))
        return SimpleNamespace(data=vectors)


def create_index(tmp_path, client=None, model="text-embedding-3-small"):
    kb = tmp_path / "knowledge"
    kb.mkdir(exist_ok=True)
    if not list(kb.glob("*.txt")) and not list(kb.glob("*.md")):
        (kb / "guide.txt").write_text("Default test passage.", encoding="utf-8")
    persist = tmp_path / "data" / "chroma"
    client = client or FakeEmbeddingClient()
    stats = rebuild_index(
        knowledge_dir=kb,
        persist_dir=persist,
        client=client,
        embedding_model=model,
    )
    return kb, persist, client, stats


def test_initial_index_persists_and_reloads(tmp_path):
    kb = tmp_path / "knowledge"
    kb.mkdir()
    (kb / "guide.md").write_text("Database timeout troubleshooting.", encoding="utf-8")
    persist = tmp_path / "data" / "chroma"
    embedding_client = FakeEmbeddingClient()
    stats = rebuild_index(knowledge_dir=kb, persist_dir=persist, client=embedding_client)
    assert stats.total_chunks == 1
    assert stats.added_chunks == 1
    assert len(embedding_client.calls) == 1

    chroma_client, collection, manifest = open_active_collection(persist)
    assert manifest["embedding_model"] == "text-embedding-3-small"
    assert collection.count() == 1
    assert collection.get(include=["documents"])["documents"] == ["Database timeout troubleshooting."]
    status = index_status(knowledge_dir=kb, persist_dir=persist)
    assert status["indexed"] and not status["stale"]


def test_unchanged_reindex_skips_embedding_calls(tmp_path):
    kb, persist, initial_client, _ = create_index(tmp_path, FakeEmbeddingClient())
    initial_stats = rebuild_index(knowledge_dir=kb, persist_dir=persist, client=initial_client)
    assert initial_stats.unchanged
    second_client = FakeEmbeddingClient()
    stats = rebuild_index(knowledge_dir=kb, persist_dir=persist, client=second_client)
    assert stats.unchanged
    assert stats.skipped_chunks == 1
    assert second_client.calls == []


def test_reindex_updates_changed_adds_new_and_removes_stale_chunks(tmp_path):
    kb = tmp_path / "knowledge"
    kb.mkdir()
    first = kb / "a.txt"
    stale = kb / "b.md"
    first.write_text("First source content.", encoding="utf-8")
    stale.write_text("Second source content.", encoding="utf-8")
    persist = tmp_path / "data" / "chroma"
    rebuild_index(knowledge_dir=kb, persist_dir=persist, client=FakeEmbeddingClient())

    first.write_text("Changed source content with updated troubleshooting.", encoding="utf-8")
    stale.unlink()
    (kb / "c.md").write_text("A newly added source.", encoding="utf-8")
    update_client = FakeEmbeddingClient()
    stats = rebuild_index(knowledge_dir=kb, persist_dir=persist, client=update_client)
    assert stats.updated_chunks == 1
    assert stats.added_chunks == 1
    assert stats.stale_chunks == 1
    assert len(update_client.calls[0][1]) == 2

    _, collection, _ = open_active_collection(persist)
    data = collection.get(include=["metadatas"])
    assert {metadata["source_path"] for metadata in data["metadatas"]} == {"a.txt", "c.md"}


def test_embedding_failure_keeps_existing_active_index(tmp_path):
    kb = tmp_path / "knowledge"
    kb.mkdir()
    document = kb / "guide.txt"
    document.write_text("Original source.", encoding="utf-8")
    persist = tmp_path / "data" / "chroma"
    rebuild_index(knowledge_dir=kb, persist_dir=persist, client=FakeEmbeddingClient())
    manifest_path = persist.parent / "chroma_active_index.json"
    old_manifest = manifest_path.read_text(encoding="utf-8")
    old_collection_name = __import__("json").loads(old_manifest)["collection_name"]

    document.write_text("Changed source requiring an embedding.", encoding="utf-8")
    with pytest.raises(EmbeddingError):
        rebuild_index(
            knowledge_dir=kb,
            persist_dir=persist,
            client=FakeEmbeddingClient(fail=True),
        )
    assert manifest_path.read_text(encoding="utf-8") == old_manifest
    chroma_client = chromadb.PersistentClient(path=str(persist))
    active = chroma_client.get_collection(old_collection_name)
    assert active.get(include=["documents"])["documents"] == ["Original source."]


def test_embedding_model_change_reembeds_every_chunk(tmp_path):
    kb, persist, _, _ = create_index(tmp_path)
    document = kb / "guide.txt"
    document.write_text("Model-specific vector content.", encoding="utf-8")
    rebuild_index(knowledge_dir=kb, persist_dir=persist, client=FakeEmbeddingClient())
    new_model_client = FakeEmbeddingClient()
    stats = rebuild_index(
        knowledge_dir=kb,
        persist_dir=persist,
        client=new_model_client,
        embedding_model="text-embedding-3-large",
    )
    assert stats.added_chunks == 1
    assert stats.skipped_chunks == 0
    assert len(new_model_client.calls) == 1
    _, collection, manifest = open_active_collection(persist, embedding_model="text-embedding-3-large")
    assert collection.count() == 1
    assert manifest["embedding_model"] == "text-embedding-3-large"


def test_empty_corpus_does_not_clear_existing_index(tmp_path):
    kb, persist, _, _ = create_index(tmp_path)
    (kb / "guide.txt").write_text("Existing passage", encoding="utf-8")
    rebuild_index(knowledge_dir=kb, persist_dir=persist, client=FakeEmbeddingClient())
    (kb / "guide.txt").unlink()
    with pytest.raises(IndexErrorSafe, match="existing index was kept"):
        rebuild_index(knowledge_dir=kb, persist_dir=persist, client=FakeEmbeddingClient())
    _, collection, _ = open_active_collection(persist)
    assert collection.count() == 1
