"""BM25 and dense Chroma retrieval with rank-fusion and deterministic evidence IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus

from .indexing import load_chunks
from .embeddings import EmbeddingError, embed_query
from .indexing import IndexErrorSafe, open_active_collection
from .schemas import EvidenceRecord, RetrievalResult


BM25_CANDIDATES = 5
DENSE_CANDIDATES = 5
MAX_RESULTS = 3
RRF_K = 60
MIN_SEMANTIC_SIMILARITY = 0.35
MIN_FUSED_SCORE = 0.016
_TOKEN = re.compile(r"[a-z0-9_./-]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "in",
    "is", "it", "its", "of", "on", "or", "our", "should", "that", "the", "their",
    "then", "there", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "you",
}


class RetrievalError(RuntimeError):
    """Safe retrieval failure suitable for user-facing feedback."""


@dataclass(frozen=True)
class RetrievalOutcome:
    passages: list[RetrievalResult]
    mode: str
    warning: str | None = None


def retrieve_for_incident(
    *,
    query: str,
    lexical_query: str,
    knowledge_dir: str | Path,
    persist_dir: str | Path,
    client,
    embedding_model: str,
) -> RetrievalOutcome:
    keyword_results = keyword_search(lexical_query, knowledge_dir)
    try:
        _, collection, _ = open_active_collection(
            persist_dir, embedding_model=embedding_model
        )
    except IndexErrorSafe as exc:
        return RetrievalOutcome(keyword_results, "bm25", str(exc))
    if collection is None:
        return RetrievalOutcome(
            keyword_results,
            "bm25",
            "The semantic index has not been built; using BM25 keyword retrieval.",
        )
    try:
        vector = embed_query(client, query, model=embedding_model)
        passages = hybrid_search(
            query=lexical_query,
            knowledge_dir=knowledge_dir,
            collection=collection,
            query_embedding=vector,
        )
    except (EmbeddingError, RetrievalError) as exc:
        return RetrievalOutcome(keyword_results, "bm25", str(exc))
    return RetrievalOutcome(passages, "hybrid")


def knowledge_evidence(passages: list[RetrievalResult]) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            evidence_id=passage.evidence_id,
            kind="knowledge_base",
            source_name=passage.source_path,
            excerpt=passage.text,
            passage_id=passage.chunk_id,
        )
        for passage in passages
    ]


def tokenize(text: str) -> list[str]:
    return [term for term in _TOKEN.findall(text.lower()) if term not in _STOP_WORDS]


def keyword_search(
    query: str,
    knowledge_dir: str | Path,
    *,
    max_results: int = MAX_RESULTS,
) -> list[RetrievalResult]:
    chunks = load_chunks(knowledge_dir)
    query_terms = tokenize(query)
    if not chunks or not query_terms:
        return []
    tokenized_docs = [tokenize(chunk.text) for chunk in chunks]
    scores = BM25Plus(tokenized_docs).get_scores(query_terms)
    ranked = sorted(
        ((float(score), chunk) for score, chunk in zip(scores, chunks, strict=True) if score > 0),
        key=lambda item: (-item[0], item[1].source_path.lower(), item[1].chunk_index),
    )[:max_results]
    return [
        RetrievalResult(
            evidence_id=f"K{rank}",
            source_path=chunk.source_path,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            bm25_score=score,
            semantic_similarity=None,
            fused_score=1 / (RRF_K + rank),
        )
        for rank, (score, chunk) in enumerate(ranked, start=1)
    ]


def hybrid_search(
    *,
    query: str,
    knowledge_dir: str | Path,
    collection: Any,
    query_embedding: list[float],
    max_results: int = MAX_RESULTS,
) -> list[RetrievalResult]:
    chunks = load_chunks(knowledge_dir)
    if not chunks:
        return []
    current_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    keyword_candidates = keyword_search(query, knowledge_dir, max_results=BM25_CANDIDATES)
    bm25_rank = {item.chunk_id: rank for rank, item in enumerate(keyword_candidates, start=1)}
    bm25_by_id = {item.chunk_id: item for item in keyword_candidates}

    try:
        count = collection.count()
        if count == 0:
            dense_rows = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        else:
            dense_rows = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(DENSE_CANDIDATES, count),
                include=["metadatas", "distances"],
            )
    except Exception as exc:
        raise RetrievalError("The local semantic index could not be searched. Rebuild it or use keyword search.") from exc

    dense_candidates: list[tuple[str, float]] = []
    ids = dense_rows.get("ids", [[]])[0] or []
    metadatas = dense_rows.get("metadatas", [[]])[0] or []
    distances = dense_rows.get("distances", [[]])[0] or []
    for chunk_id, metadata, distance in zip(ids, metadatas, distances, strict=False):
        current = current_by_id.get(chunk_id)
        if current is None or metadata.get("content_hash") != current.content_hash:
            continue  # An explicit reindex is required before changed sources are trusted.
        similarity = 1.0 - float(distance)
        if similarity >= MIN_SEMANTIC_SIMILARITY:
            dense_candidates.append((chunk_id, similarity))
    dense_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_candidates, start=1)}
    dense_similarity = dict(dense_candidates)

    candidate_ids = set(bm25_rank) | set(dense_rank)
    ranked = []
    for chunk_id in candidate_ids:
        fused_score = (1 / (RRF_K + bm25_rank[chunk_id]) if chunk_id in bm25_rank else 0.0)
        fused_score += 1 / (RRF_K + dense_rank[chunk_id]) if chunk_id in dense_rank else 0.0
        if fused_score >= MIN_FUSED_SCORE:
            ranked.append((fused_score, chunk_id))
    ranked.sort(
        key=lambda item: (
            -item[0],
            current_by_id[item[1]].source_path.lower(),
            current_by_id[item[1]].chunk_index,
        )
    )

    results = []
    for evidence_number, (fused_score, chunk_id) in enumerate(ranked[:max_results], start=1):
        chunk = current_by_id[chunk_id]
        results.append(
            RetrievalResult(
                evidence_id=f"K{evidence_number}",
                source_path=chunk.source_path,
                chunk_id=chunk_id,
                text=chunk.text,
                bm25_score=bm25_by_id[chunk_id].bm25_score if chunk_id in bm25_by_id else 0.0,
                semantic_similarity=dense_similarity.get(chunk_id),
                fused_score=fused_score,
            )
        )
    return results
