"""Manually invoked, live retrieval evaluation; never runs during app startup or tests."""

from __future__ import annotations

import argparse
import json
import sys

import openai

from .analysis import create_client
from .config import PROJECT_ROOT, load_settings
from .embeddings import EmbeddingError, embed_texts
from .indexing import IndexErrorSafe, open_active_collection
from .retrieval import RetrievalError, RetrievalQueries, build_retrieval_queries, hybrid_search
from .validation import make_log_evidence, redact_sensitive_text, validate_uploaded_log


def case_retrieval_queries(case: dict) -> RetrievalQueries:
    """Build the same queries the app builds, including an optional attached ``log``."""
    query = redact_sensitive_text(case["query"])
    log_text = case.get("log")
    if not log_text:
        return build_retrieval_queries(query, [])
    upload = validate_uploaded_log(f"{case['id']}.log", log_text.encode("utf-8"))
    return build_retrieval_queries(query, make_log_evidence(upload, query))


def calculate_metrics(cases: list[dict], retrieved_sources: dict[str, list[str]]) -> dict[str, float]:
    """Source-level metrics.

    With a corpus of only a few chunks, recall@3 can approach "returned anything at all", so
    recall@1 and MRR are the more informative ranking numbers. Multi-source cases also report
    whether every expected source was retrieved, not just one of them.
    """
    relevant = [case for case in cases if case["expected_sources"]]
    multi_source = [case for case in relevant if len(case["expected_sources"]) > 1]
    no_evidence = [case for case in cases if not case["expected_sources"]]
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    for case in relevant:
        expected = set(case["expected_sources"])
        sources = retrieved_sources.get(case["id"], [])[:3]
        rank = next((i for i, source in enumerate(sources, start=1) if source in expected), None)
        if rank is not None:
            hits_at_3 += 1
            hits_at_1 += rank == 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    full_coverage = sum(
        set(case["expected_sources"]) <= set(retrieved_sources.get(case["id"], [])[:3])
        for case in multi_source
    )
    false_positives = sum(bool(retrieved_sources.get(case["id"], [])) for case in no_evidence)
    return {
        "recall_at_1": hits_at_1 / len(relevant) if relevant else 0.0,
        "recall_at_3": hits_at_3 / len(relevant) if relevant else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "multi_source_full_coverage_at_3": full_coverage / len(multi_source) if multi_source else 0.0,
        "false_positive_rate": false_positives / len(no_evidence) if no_evidence else 0.0,
    }


def acceptance_failures(cases: list[dict], retrieved_sources: dict[str, list[str]]) -> list[str]:
    failures = []
    for case in cases:
        if not case.get("acceptance_required", False):
            continue
        sources = retrieved_sources.get(case["id"], [])
        expected = case["expected_sources"]
        if expected and not (set(sources[:3]) & set(expected)):
            failures.append(case["id"])
        elif not expected and sources:
            failures.append(case["id"])
    return failures


def run_live_evaluation() -> int:
    settings = load_settings()
    api_key, embedding_model, persist_dir = settings.api_key, settings.embedding_model, settings.persist_dir
    if not api_key:
        print("Set OPENAI_API_KEY in .env before running the live evaluation.", file=sys.stderr)
        return 2
    cases_path = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
    try:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    except (OSError, ValueError, KeyError, TypeError):
        print(f"Could not read evaluation cases from {cases_path}.", file=sys.stderr)
        return 2
    try:
        chroma_client, collection, manifest = open_active_collection(
            persist_dir, embedding_model=embedding_model
        )
        if collection is None or manifest is None:
            print("Build the knowledge index in the app before evaluating retrieval.", file=sys.stderr)
            return 2
        case_queries = [case_retrieval_queries(case) for case in cases]
        with create_client(api_key) as client:
            query_vectors = embed_texts(
                client, [queries.relevance for queries in case_queries], model=embedding_model
            )
        results = {}
        knowledge_dir = PROJECT_ROOT / "knowledge_base"
        for case, queries, vector in zip(cases, case_queries, query_vectors, strict=True):
            matches = hybrid_search(
                query=queries.lexical,
                knowledge_dir=knowledge_dir,
                collection=collection,
                query_embedding=vector,
                relevance_query=queries.relevance,
            )
            results[case["id"]] = [match.source_path for match in matches]
        metrics = calculate_metrics(cases, results)
    except (EmbeddingError, IndexErrorSafe, RetrievalError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except openai.OpenAIError:
        print("The evaluation embedding request failed. Check API status, model access, and rate limits.", file=sys.stderr)
        return 1
    failures = acceptance_failures(cases, results)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "metrics": metrics,
                "acceptance_pass": not failures,
                "acceptance_failures": failures,
                "results": results,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paid, live hybrid retrieval evaluation.")
    parser.add_argument("--live", action="store_true", help="Explicitly send evaluation queries to the OpenAI Embeddings API.")
    args = parser.parse_args()
    if not args.live:
        parser.error("No API calls were made. Pass --live to run the evaluation and incur embedding usage.")
    raise SystemExit(run_live_evaluation())


if __name__ == "__main__":
    main()
