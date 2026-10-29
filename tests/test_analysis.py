import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import ValidationError

from incident_assistant.analysis import (
    AnalysisError,
    InvalidModelResponseError,
    MissingAPIKeyError,
    ModelRefusalError,
    analyze_incident,
)
from incident_assistant.schemas import (
    DiagnosticStep,
    EvidenceRecord,
    Finding,
    IncidentAnalysis,
    RetrievalResult,
)


def valid_analysis(refs=("I1", "L1", "K1")):
    return IncidentAnalysis(
        summary="Database connection timeouts are affecting two instances.",
        category="Database",
        severity="Medium",
        observable_facts=[Finding(text="The incident report describes timeouts.", evidence_ids=[refs[0]])],
        potential_causes=[Finding(text="A saturated pool is one possibility.", evidence_ids=[refs[2]])],
        diagnostic_steps=[DiagnosticStep(text="Check pool saturation and database health.", evidence_ids=[refs[2]])],
        suggested_resolution="If saturation is confirmed, follow the team's approved process.",
        evidence_limitations=[],
    )


class FakeResponses:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, result=None, error=None):
        self.responses = FakeResponses(result, error)


def evidence_inputs():
    logs = [
        EvidenceRecord(
            evidence_id="L1", kind="uploaded_log", source_name="service.log",
            excerpt="Database connection timeout", line_start=4, line_end=4,
        )
    ]
    passages = [
        RetrievalResult(
            evidence_id="K1", source_path="database.txt", chunk_id="database-01",
            text="Check connection pool saturation.", bm25_score=0.2,
            semantic_similarity=0.8, fused_score=0.03,
        )
    ]
    return logs, passages


def call(client, **overrides):
    logs, passages = evidence_inputs()
    kwargs = dict(
        incident_text="Database connection timed out.",
        log_evidence=logs,
        passages=passages,
        api_key="test-key",
        model="gpt-6-luna",
        client=client,
    )
    kwargs.update(overrides)
    return analyze_incident(**kwargs)


def test_valid_result_uses_real_application_evidence_records():
    client = FakeClient(SimpleNamespace(output_parsed=valid_analysis(), output=[]))
    result = call(client)
    assert result.analysis.category == "Database"
    assert {record.evidence_id for record in result.evidence} == {"I1", "L1", "K1"}
    assert client.responses.calls[0]["text_format"] is IncidentAnalysis
    assert client.responses.calls[0]["max_output_tokens"] == 2000
    payload = json.loads(client.responses.calls[0]["input"][1]["content"])
    assert payload["incident_description_evidence"]["evidence_id"] == "I1"
    assert [item["evidence_id"] for item in payload["uploaded_log_evidence"]] == ["L1"]
    assert [item["evidence_id"] for item in payload["knowledge_passages"]] == ["K1"]


def test_unknown_model_evidence_reference_is_rejected():
    client = FakeClient(SimpleNamespace(output_parsed=valid_analysis(("I1", "L1", "invented")), output=[]))
    with pytest.raises(InvalidModelResponseError, match="invalid evidence"):
        call(client)


def test_knowledge_guidance_cannot_be_cited_as_an_observed_fact():
    analysis = valid_analysis()
    analysis.observable_facts = [Finding(text="A pool issue occurred.", evidence_ids=["K1"])]
    client = FakeClient(SimpleNamespace(output_parsed=analysis, output=[]))
    with pytest.raises(InvalidModelResponseError, match="troubleshooting guidance"):
        call(client)


def test_observable_fact_without_evidence_is_rejected():
    analysis = valid_analysis()
    analysis.observable_facts = [Finding(text="A pool issue occurred.", evidence_ids=[])]
    client = FakeClient(SimpleNamespace(output_parsed=analysis, output=[]))
    with pytest.raises(InvalidModelResponseError, match="without direct incident evidence"):
        call(client)


def test_refusal_is_reported_without_analysis():
    refusal = SimpleNamespace(type="refusal", refusal="not allowed")
    response = SimpleNamespace(output=[SimpleNamespace(content=[refusal])], output_parsed=None)
    with pytest.raises(ModelRefusalError, match="declined"):
        call(FakeClient(response))


def test_incomplete_model_response_is_rejected():
    with pytest.raises(InvalidModelResponseError, match="incomplete"):
        call(FakeClient(SimpleNamespace(output_parsed=None, output=[])))


def test_schema_validation_error_is_rejected_safely():
    with pytest.raises(ValidationError) as validation_error:
        IncidentAnalysis.model_validate({"category": "Invalid category"})
    client = FakeClient(error=validation_error.value)
    with pytest.raises(InvalidModelResponseError, match="required format"):
        call(client)


def test_missing_key_fails_before_client_call():
    with pytest.raises(MissingAPIKeyError):
        call(FakeClient(), api_key=None)


def test_openai_client_initialization_failure_is_safe(monkeypatch):
    def fail_init(**kwargs):
        raise RuntimeError("internal proxy detail")

    monkeypatch.setattr("incident_assistant.analysis.openai.OpenAI", fail_init)
    logs, passages = evidence_inputs()
    with pytest.raises(AnalysisError, match="unexpected error") as exc_info:
        analyze_incident(
            incident_text="Database timeout",
            log_evidence=logs,
            passages=passages,
            api_key="test-key",
        )
    assert "internal proxy detail" not in str(exc_info.value)


@pytest.mark.parametrize(
    "error, expected",
    [
        (openai.APITimeoutError(request=None), "timed out"),
        (openai.APIStatusError("rate limit", response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")), body=None), "rate limited"),
        (openai.APIStatusError("server failure", response=httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com")), body=None), "could not complete"),
        (openai.OpenAIError("private internal detail"), "returned an error"),
    ],
)
def test_api_failures_map_to_safe_messages(error, expected):
    with pytest.raises(AnalysisError) as exc_info:
        call(FakeClient(error=error))
    assert expected in str(exc_info.value)
    assert "private internal detail" not in str(exc_info.value)
