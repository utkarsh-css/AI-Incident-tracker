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
