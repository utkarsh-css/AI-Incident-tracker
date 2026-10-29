"""Manually invoked, live retrieval evaluation; never runs during app startup or tests."""

from __future__ import annotations

import argparse
import json
import sys
import openai

from .config import PROJECT_ROOT, load_settings
from .embeddings import EmbeddingError, embed_texts
from .indexing import IndexErrorSafe, open_active_collection
from .retrieval import RetrievalError, hybrid_search


def calculate_metrics(cases: list[dict], retrieved_sources: dict[str, list[str]]) -> dict[str, float]:
    relevant = [case for case in cases if case["expected_sources"]]
    no_evidence = [case for case in cases if not case["expected_sources"]]
    hits = 0
    reciprocal_ranks = []
    for case in relevant:
        expected = set(case["expected_sources"])
        sources = retrieved_sources.get(case["id"], [])
        rank = next((i for i, source in enumerate(sources, start=1) if source in expected), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    false_positives = sum(bool(retrieved_sources.get(case["id"], [])) for case in no_evidence)
    return {
        "recall_at_3": hits / len(relevant) if relevant else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
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
    api_key, _, embedding_model, persist_dir = load_settings()
    if not api_key:
        print("Set OPENAI_API_KEY in .env before running the live evaluation.", file=sys.stderr)
        return 2
    cases_path = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    try:
        chroma_client, collection, manifest = open_active_collection(
            persist_dir, embedding_model=embedding_model
        )
        if collection is None or manifest is None:
            print("Build the knowledge index in the app before evaluating retrieval.", file=sys.stderr)
            return 2
        with openai.OpenAI(api_key=api_key, max_retries=0, timeout=45.0) as client:
            query_vectors = embed_texts(client, [case["query"] for case in cases], model=embedding_model)
        results = {}
        knowledge_dir = PROJECT_ROOT / "knowledge_base"
        for case, vector in zip(cases, query_vectors, strict=True):
            matches = hybrid_search(
                query=case["query"],
                knowledge_dir=knowledge_dir,
                collection=collection,
                query_embedding=vector,
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
