from types import SimpleNamespace

import pytest

from incident_assistant.embeddings import EmbeddingError
from incident_assistant.indexing import load_chunks
from incident_assistant.retrieval import hybrid_search, keyword_search, retrieve_for_incident


def test_load_chunks_reads_text_and_markdown_with_stable_ids(tmp_path):
    (tmp_path / "z.txt").write_text("Last paragraph.", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Heading\n\nFirst paragraph.", encoding="utf-8")
    first = load_chunks(tmp_path)
    second = load_chunks(tmp_path)
    assert [(chunk.source_path, chunk.chunk_index) for chunk in first] == [
        ("a.md", 0),
        ("z.txt", 0),
    ]
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunker_enforces_limit_and_overlap():
    from incident_assistant.indexing import chunk_text

    text = "word " * 1600
    chunks = chunk_text(text, max_chars=400, overlap_chars=60)
    assert len(chunks) > 2
    assert all(len(chunk) <= 400 for chunk in chunks)
    assert any(chunks[index][-20:] in chunks[index + 1] for index in range(len(chunks) - 1))


def test_keyword_search_finds_all_demo_topics():
    cases = [
        ("billing database connection timeout pool saturation", "database_connectivity.txt"),
        ("HTTP API intermittent 503 gateway upstream latency", "web_api_errors.txt"),
        ("authentication identity provider token expired HTTP 401", "authentication.txt"),
    ]
    for query, expected in cases:
        matches = keyword_search(query, "knowledge_base")
        assert matches
        assert matches[0].source_path == expected


def test_keyword_unrelated_query_returns_none():
    assert keyword_search("purple elephants enjoy violin practice", "knowledge_base") == []


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.last_n_results = None

    def count(self):
        return len(self.rows)

    def query(self, *, query_embeddings, n_results, include):
        self.last_n_results = n_results
        selected = self.rows[:n_results]
        return {
            "ids": [[row[0] for row in selected]],
            "metadatas": [[row[1] for row in selected]],
            "distances": [[row[2] for row in selected]],
        }


def test_hybrid_ranking_fuses_dense_and_bm25_and_limits_results(tmp_path):
    (tmp_path / "a.txt").write_text(
        "A database connection timeout can occur when a pool is saturated.\n\n"
        "Check database connection limits and network latency.\n\n"
        "Compare application deployment timing with database health.",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    rows = [
        (chunk.chunk_id, chunk.metadata(), 0.2 + rank * 0.01)
        for rank, chunk in enumerate(chunks)
    ]
    collection = FakeCollection(rows)
    matches = hybrid_search(
        query="database connection timeout and pool saturation",
        knowledge_dir=tmp_path,
        collection=collection,
        query_embedding=[0.1, 0.2],
    )
    assert matches
    assert len(matches) <= 3
    assert matches[0].bm25_score > 0
    assert matches[0].semantic_similarity == pytest.approx(0.8)
    assert matches[0].fused_score >= 0.016
    assert collection.last_n_results == min(5, len(chunks))


def test_hybrid_search_filters_stale_dense_records(tmp_path):
    (tmp_path / "doc.txt").write_text("Current database timeout guidance.", encoding="utf-8")
    chunk = load_chunks(tmp_path)[0]
    old_metadata = chunk.metadata() | {"content_hash": "stale-hash"}
    collection = FakeCollection([(chunk.chunk_id, old_metadata, 0.1)])
    matches = hybrid_search(
        query="database timeout guidance",
        knowledge_dir=tmp_path,
        collection=collection,
        query_embedding=[0.2, 0.3],
    )
    assert all(result.semantic_similarity is None for result in matches)


def test_dense_similarity_cutoff_drops_weak_semantic_only_result(tmp_path):
    (tmp_path / "doc.txt").write_text("Unrelated local troubleshooting passage.", encoding="utf-8")
    chunk = load_chunks(tmp_path)[0]
    collection = FakeCollection([(chunk.chunk_id, chunk.metadata(), 0.8)])
    assert hybrid_search(
        query="rare query terms",
        knowledge_dir=tmp_path,
        collection=collection,
        query_embedding=[0.2, 0.3],
    ) == []


def test_keyword_candidate_can_qualify_through_rrf_without_dense_match(tmp_path):
    (tmp_path / "doc.txt").write_text("database connection timeout", encoding="utf-8")
    collection = FakeCollection([])
    results = hybrid_search(
        query="database connection timeout",
        knowledge_dir=tmp_path,
        collection=collection,
        query_embedding=[0.2, 0.3],
    )
    assert results
    assert results[0].semantic_similarity is None


def test_embedding_failure_falls_back_to_bm25(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("Database connection timeout guidance.", encoding="utf-8")
    monkeypatch.setattr(
        "incident_assistant.retrieval.open_active_collection",
        lambda *args, **kwargs: (None, FakeCollection([]), {"collection_name": "present"}),
    )
    monkeypatch.setattr(
        "incident_assistant.retrieval.embed_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(EmbeddingError("safe failure")),
    )
    outcome = retrieve_for_incident(
        query="database timeout",
        lexical_query="database timeout error in uploaded log",
        knowledge_dir=tmp_path,
        persist_dir=tmp_path / "index",
        client=object(),
        embedding_model="text-embedding-3-small",
    )
    assert outcome.mode == "bm25"
    assert outcome.warning == "safe failure"
    assert outcome.passages and outcome.passages[0].source_path == "doc.txt"


def test_missing_index_uses_bm25_without_embedding_call(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("Database connection timeout guidance.", encoding="utf-8")
    monkeypatch.setattr(
        "incident_assistant.retrieval.open_active_collection",
        lambda *args, **kwargs: (None, None, None),
    )
    monkeypatch.setattr(
        "incident_assistant.retrieval.embed_query",
        lambda *args, **kwargs: pytest.fail("embedding must not be called without an active index"),
    )
    outcome = retrieve_for_incident(
        query="database timeout",
        lexical_query="database timeout error in uploaded log",
        knowledge_dir=tmp_path,
        persist_dir=tmp_path / "index",
        client=object(),
        embedding_model="text-embedding-3-small",
    )
    assert outcome.mode == "bm25"
    assert outcome.passages

