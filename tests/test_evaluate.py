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
    assert metrics == {"recall_at_3": 0.5, "mrr": 0.25, "false_positive_rate": 0.5}


def test_evaluation_acceptance_requires_topic_hits_and_no_false_positives():
    cases = [
        {"id": "semantic", "expected_sources": ["database.txt"], "acceptance_required": True},
        {"id": "negative", "expected_sources": [], "acceptance_required": True},
        {"id": "mixed", "expected_sources": ["a.txt", "b.txt"], "acceptance_required": False},
    ]
    assert acceptance_failures(cases, {"semantic": ["database.txt"], "negative": [], "mixed": []}) == []
    assert acceptance_failures(cases, {"semantic": ["other.txt"], "negative": ["auth.txt"]}) == ["semantic", "negative"]
