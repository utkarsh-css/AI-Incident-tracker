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


@pytest.mark.parametrize(
    "query",
    [
        # Each shares exactly one word ("error" / "failed") with web_api_errors.txt.
        "The office printer shows a paper jam error",
        "Disk usage on the file server reached 100 percent and backups failed",
    ],
)
def test_off_topic_incident_sharing_one_generic_term_returns_none(query):
    assert keyword_search(query, "knowledge_base") == []


def test_bm25plus_lower_bound_does_not_admit_chunks_without_query_terms(tmp_path):
    # BM25Plus gives every document a positive score when a query term exists anywhere in the
    # corpus; only the chunk that actually contains the terms may be returned.
    (tmp_path / "db.txt").write_text("Database connection timeout and pool saturation.", encoding="utf-8")
    (tmp_path / "web.txt").write_text("Gateway returned HTTP 503 from upstream.", encoding="utf-8")
    (tmp_path / "auth.txt").write_text("Identity provider rejected expired tokens.", encoding="utf-8")
    matches = keyword_search("database connection timeout", tmp_path)
    assert [match.source_path for match in matches] == ["db.txt"]


def test_single_shared_term_is_not_enough_for_a_keyword_match(tmp_path):
    (tmp_path / "db.txt").write_text("Database connection timeout guidance.", encoding="utf-8")
    assert keyword_search("printer timeout", tmp_path) == []
    assert keyword_search("database timeout", tmp_path)


def test_tokenizer_ignores_trailing_sentence_punctuation():
    from incident_assistant.retrieval import tokenize

    assert tokenize("Timeouts. identity-policy /api/v1/") == ["timeouts", "identity-policy", "api/v1"]


def test_hybrid_off_topic_query_without_dense_match_returns_none():
    matches = hybrid_search(
        query="The office printer shows a paper jam error",
        knowledge_dir="knowledge_base",
        collection=FakeCollection([]),
        query_embedding=[0.1, 0.2],
    )
    assert matches == []


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



def test_log_noise_cannot_qualify_a_passage_without_incident_terms():
    noise = "INFO request completed request id latency 12ms " * 20
    matches = keyword_search(
        "printer jam " + noise, "knowledge_base", relevance_query="printer jam"
    )
    assert matches == []


def test_log_terms_still_rerank_passages_that_qualify_on_incident_terms(tmp_path):
    (tmp_path / "a.txt").write_text("Database connection timeout. Check pool saturation.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Database connection timeout. Check DNS resolution.", encoding="utf-8")
    matches = keyword_search(
        "database connection timeout dns resolution dns",
        tmp_path,
        relevance_query="database connection timeout",
    )
    assert [match.source_path for match in matches] == ["b.txt", "a.txt"]


def test_required_term_matches_scales_with_description_length_and_is_capped():
    from incident_assistant.retrieval import required_term_matches

    assert [required_term_matches(n) for n in (1, 5, 20, 25, 31, 500)] == [2, 2, 2, 3, 4, 4]


def test_long_relevant_description_still_matches_but_long_off_topic_does_not():
    padding = " ".join(f"word{i}" for i in range(200))
    relevant = "Billing database connection timeouts affect two application instances. " + padding
    off_topic = "The printer on floor three reports an error and the request failed. " + padding
    assert [m.source_path for m in keyword_search(relevant, "knowledge_base")] == ["database_connectivity.txt"]
    assert keyword_search(off_topic, "knowledge_base") == []


def _queries_for(incident, log_text):
    from incident_assistant.retrieval import build_retrieval_queries
    from incident_assistant.validation import ValidatedUpload, make_log_evidence

    return build_retrieval_queries(incident, make_log_evidence(ValidatedUpload("t.log", log_text), incident))


def test_error_log_lines_count_toward_relevance_but_info_lines_do_not():
    queries = _queries_for(
        "Billing is down, see attached log.",
        "INFO billing worker started\nERROR database connection timeout after 30000ms\n",
    )
    assert "ERROR database connection timeout" in queries.relevance
    assert "INFO billing worker started" not in queries.relevance
    assert "INFO billing worker started" in queries.lexical


def test_short_description_with_informative_error_log_retrieves_guidance():
    queries = _queries_for(
        "Billing is down, see attached log.",
        "INFO billing worker started\n"
        "ERROR database connection timeout after 30000ms pool=20/20\n"
        "ERROR connection pool exhausted; waiting for available connection\n",
    )
    matches = keyword_search(queries.lexical, "knowledge_base", relevance_query=queries.relevance)
    assert [match.source_path for match in matches][:1] == ["database_connectivity.txt"]


def test_off_topic_description_with_routine_log_lines_retrieves_nothing():
    queries = _queries_for(
        "The office printer shows a paper jam",
        "INFO request completed request id latency 12ms\n" * 20,
    )
    assert keyword_search(queries.lexical, "knowledge_base", relevance_query=queries.relevance) == []


def test_retrieve_for_incident_embeds_the_relevance_query(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("Database connection timeout guidance.", encoding="utf-8")
    embedded = []
    monkeypatch.setattr(
        "incident_assistant.retrieval.open_active_collection",
        lambda *args, **kwargs: (None, FakeCollection([]), {"collection_name": "present"}),
    )
    monkeypatch.setattr(
        "incident_assistant.retrieval.embed_query",
        lambda client, text, **kwargs: embedded.append(text) or [0.1, 0.2],
    )
    queries = _queries_for("Billing is down.", "ERROR database connection timeout\nINFO noise line\n")
    retrieve_for_incident(
        query=queries.relevance,
        lexical_query=queries.lexical,
        knowledge_dir=tmp_path,
        persist_dir=tmp_path / "index",
        client=object(),
        embedding_model="text-embedding-3-small",
    )
    assert embedded == [queries.relevance]
    assert "INFO noise line" not in embedded[0]
