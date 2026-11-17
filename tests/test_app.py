from types import SimpleNamespace

from app import SAMPLES, get_sample_incident
from app import KNOWLEDGE_DIR
from incident_assistant.retrieval import keyword_search
from streamlit.testing.v1 import AppTest


def test_sample_incidents_are_selectable_and_cover_all_knowledge_topics():
    assert len(SAMPLES) == 3
    assert "database" in get_sample_incident("Database connection timeouts").lower()
    assert "503" in get_sample_incident("Web API 503 responses")
    assert "401" in get_sample_incident("Authentication failures")
    assert get_sample_incident("unknown") is None


def test_each_sample_retrieves_its_matching_knowledge_document():
    expected_sources = {
        "Database connection timeouts": "database_connectivity.txt",
        "Web API 503 responses": "web_api_errors.txt",
        "Authentication failures": "authentication.txt",
    }
    for name, source in expected_sources.items():
        results = keyword_search(get_sample_incident(name), KNOWLEDGE_DIR)
        assert results
        assert results[0].source_name == source


def test_streamlit_load_sample_updates_incident_and_missing_key_disables_analysis(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "replace-with-your-own-key")
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    analyze_button = next(button for button in app.button if button.label == "Analyze incident")
    rebuild_button = next(button for button in app.button if button.label == "Rebuild knowledge index")
    assert analyze_button.disabled
    assert rebuild_button.disabled

    app.selectbox[0].select("Database connection timeouts")
    app.button[0].click().run()
    assert "billing service" in app.text_area[0].value


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        from incident_assistant.schemas import DiagnosticStep, Finding, IncidentAnalysis

        self.calls.append(kwargs)
        analysis = IncidentAnalysis(
            summary="Database connection timeouts on two instances.",
            category="Database",
            severity="Medium",
            observable_facts=[Finding(text="Two of four instances time out.", evidence_ids=["I1"])],
            potential_causes=[Finding(text="Pool saturation is possible.", evidence_ids=["K1"])],
            diagnostic_steps=[DiagnosticStep(text="Compare pool metrics.", evidence_ids=[])],
            suggested_resolution="If saturation is confirmed, follow the approved change process.",
            evidence_limitations=["Semantic search was not used."],
        )
        return SimpleNamespace(output_parsed=analysis, output=[], status="completed")


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_streamlit_analyze_flow_renders_cited_evidence_and_fallback_mode(monkeypatch, tmp_path):
    fake = _FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("incident_assistant.analysis.create_client", lambda api_key: fake)
    app = AppTest.from_file("app.py").run()
    app.text_area[0].input(SAMPLES["Database connection timeouts"]).run()
    next(button for button in app.button if button.label == "Analyze incident").click().run()

    assert not app.exception
    assert len(fake.responses.calls) == 1
    markdown = [element.value for element in app.markdown]
    assert "#### Cited evidence" in markdown
    assert "#### Other evidence supplied to the model (not cited)" in markdown
    assert any("BM25 keyword fallback" in element.value for element in app.warning)
