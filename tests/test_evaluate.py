from incident_assistant.evaluate import acceptance_failures, calculate_metrics


def test_retrieval_evaluation_reports_recall_mrr_and_false_positive_rate():
    cases = [
        {"id": "r1", "expected_sources": ["db.txt"]},
        {"id": "r2", "expected_sources": ["auth.txt"]},
        {"id": "n1", "expected_sources": []},
        {"id": "n2", "expected_sources": []},
    ]
    results = {
        "r1": ["other.txt", "db.txt"],
        "r2": [],
        "n1": ["auth.txt"],
        "n2": [],
    }
    metrics = calculate_metrics(cases, results)
    assert metrics == {
        "recall_at_1": 0.0,
        "recall_at_3": 0.5,
        "mrr": 0.25,
        "multi_source_full_coverage_at_3": 0.0,
        "false_positive_rate": 0.5,
    }


def test_evaluation_acceptance_requires_topic_hits_and_no_false_positives():
    cases = [
        {"id": "semantic", "expected_sources": ["database.txt"], "acceptance_required": True},
        {"id": "negative", "expected_sources": [], "acceptance_required": True},
        {"id": "mixed", "expected_sources": ["a.txt", "b.txt"], "acceptance_required": False},
    ]
    assert acceptance_failures(cases, {"semantic": ["database.txt"], "negative": [], "mixed": []}) == []
    assert acceptance_failures(cases, {"semantic": ["other.txt"], "negative": ["auth.txt"]}) == ["semantic", "negative"]


def test_multi_source_cases_require_every_expected_source_for_full_coverage():
    cases = [
        {"id": "m1", "expected_sources": ["a.txt", "b.txt"]},
        {"id": "m2", "expected_sources": ["a.txt", "b.txt"]},
        {"id": "s1", "expected_sources": ["a.txt"]},
    ]
    results = {"m1": ["b.txt", "a.txt"], "m2": ["a.txt"], "s1": ["c.txt", "d.txt", "e.txt", "a.txt"]}
    metrics = calculate_metrics(cases, results)
    assert metrics["multi_source_full_coverage_at_3"] == 0.5
    assert metrics["recall_at_1"] == 2 / 3
    # A hit beyond rank 3 does not count.
    assert metrics["recall_at_3"] == 2 / 3


def test_evaluation_log_cases_use_the_app_query_building_offline():
    import json
    from pathlib import Path

    from incident_assistant.evaluate import case_retrieval_queries
    from incident_assistant.retrieval import keyword_search

    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evaluation" / "retrieval_cases.json").read_text(encoding="utf-8"))["cases"]
    log_cases = {case["id"]: case for case in cases if case.get("log")}
    assert set(log_cases) == {"log-01", "log-02"}
    for case_id, case in log_cases.items():
        queries = case_retrieval_queries(case)
        sources = [
            match.source_path
            for match in keyword_search(queries.lexical, root / "knowledge_base", relevance_query=queries.relevance)
        ]
        if case["expected_sources"]:
            assert set(sources) & set(case["expected_sources"]), case_id
        else:
            assert sources == [], case_id
