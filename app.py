"""Streamlit UI for a local, evidence-grounded IT incident analysis demo."""

from __future__ import annotations

import hashlib
from pathlib import Path

import openai
import streamlit as st

from incident_assistant.analysis import (
    AnalysisError,
    analyze_incident,
)
from incident_assistant.config import load_settings
from incident_assistant.embeddings import EmbeddingError
from incident_assistant.indexing import IndexErrorSafe, index_status, rebuild_index
from incident_assistant.retrieval import keyword_search, retrieve_for_incident
from incident_assistant.validation import (
    InputValidationError,
    MAX_INCIDENT_CHARS,
    ValidatedUpload,
    make_log_evidence,
    redact_sensitive_text,
    validate_incident_description,
    validate_uploaded_log,
)


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"

SAMPLES = {
    "Database connection timeouts": "Since 10:15 UTC, the billing service has intermittently timed out while opening database connections. The issue affects two of four application instances. No recent deployment is confirmed yet.",
    "Web API 503 responses": "The customer API returned intermittent HTTP 503 responses beginning after a traffic increase. The load balancer still reports healthy instances, but request latency is elevated. No cause has been confirmed.",
    "Authentication failures": "Several users are experiencing repeated authentication failures (HTTP 401) when signing into the application. Identity provider availability is unknown, and no account or identity-policy changes have been confirmed.",
}


def get_sample_incident(sample_name: str) -> str | None:
    return SAMPLES.get(sample_name)


def _set_sample() -> None:
    sample_name = st.session_state.get("sample_choice")
    sample_text = get_sample_incident(sample_name)
    if sample_text is not None:
        st.session_state["incident_input"] = sample_text
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_signature", None)


def _render_analysis(result) -> None:
    analysis = result.analysis
    st.subheader("Incident analysis")
    st.markdown("**Summary**")
    st.text(analysis.summary)
    col1, col2 = st.columns(2)
    col1.metric("Category", analysis.category)
    col2.metric("Advisory severity", analysis.severity)

    st.markdown("#### Observable facts")
    if analysis.observable_facts:
        for finding in analysis.observable_facts:
            refs = ", ".join(finding.evidence_ids) or "No direct evidence reference"
            st.text(finding.text)
            st.caption(f"Evidence: {refs}")
    else:
        st.info("The analysis did not identify directly supported observable facts.")

    st.markdown("#### Potential causes (unconfirmed)")
    if analysis.potential_causes:
        for finding in analysis.potential_causes:
            refs = ", ".join(finding.evidence_ids) or "No direct evidence reference"
            st.text(finding.text)
            st.caption(f"Evidence: {refs}")
    else:
        st.info("No potential causes were returned.")

    st.markdown("#### Recommended diagnostic steps")
    if analysis.diagnostic_steps:
        for step in analysis.diagnostic_steps:
            refs = ", ".join(step.evidence_ids) or "General check"
            st.text(step.text)
            st.caption(f"Evidence: {refs}")
    else:
        st.info("No diagnostic steps were returned.")

    st.markdown("#### Suggested resolution")
    st.text(analysis.suggested_resolution)
    st.markdown("#### Evidence limitations and uncertainty")
    if analysis.evidence_limitations:
        for limitation in analysis.evidence_limitations:
            st.text(limitation)
    else:
        st.write("No additional limitations were returned.")


def _render_evidence(records) -> None:
    st.markdown("#### Supporting evidence")
    if not records:
        st.info("No supporting evidence was selected.")
        return
    for record in records:
        location = ""
        if record.line_start is not None:
            location = f" · line {record.line_start}"
        elif record.passage_id:
            location = f" · {record.passage_id}"
        label = f"{record.evidence_id} · {record.source_name}{location}"
        with st.expander(label):
            st.code(record.excerpt, language="text")
            if record.truncated:
                st.caption("This log line is truncated to its first 500 characters.")
            if record.kind == "knowledge_base":
                st.caption("Troubleshooting guidance; this is not proof of the incident's root cause.")
            elif record.kind == "uploaded_log":
                st.caption("Sanitized excerpt from the uploaded log; verify against the original operational record.")
            else:
                st.caption("Sanitized incident description supplied by the user.")


def main() -> None:
    st.set_page_config(page_title="Incident Analysis Assistant", page_icon="🛠️", layout="wide")
    st.title("AI-Powered IT Incident Analysis Assistant")
    st.caption("Local demo · hybrid retrieval with source evidence · human-reviewed recommendations")
    st.info(
        "This assistant provides advisory analysis. It does not execute commands or change systems. "
        "Use synthetic or appropriately sanitized information; secret detection is best-effort."
    )

    api_key, model, embedding_model, persist_dir = load_settings()
    if not api_key:
        st.warning(
            "Analysis is disabled because OPENAI_API_KEY is not configured. "
            "Copy .env.example to .env, add your own key, and restart Streamlit. "
            "Local validation and evidence retrieval remain available."
        )

    if "incident_input" not in st.session_state:
        st.session_state["incident_input"] = ""

    st.subheader("Describe the incident")
    sample_col, button_col = st.columns([3, 1])
    sample_col.selectbox("Load a sample incident", ["Choose a sample…", *SAMPLES], key="sample_choice")
    button_col.button("Load sample", on_click=_set_sample, use_container_width=True)

    incident_raw = st.text_area(
        "Incident description",
        key="incident_input",
        height=150,
        max_chars=MAX_INCIDENT_CHARS,
        placeholder="Describe impact, start time, affected services, and observed symptoms…",
    )
    upload = st.file_uploader(
        "Optional application log (.log or .txt, maximum 1 MiB)",
        type=["log", "txt"],
        max_upload_size=1,
        help="The file is processed in memory. Common secrets and email addresses are masked before use.",
    )

    validation_error = None
    incident_text = ""
    validated_upload: ValidatedUpload | None = None
    try:
        incident_text = redact_sensitive_text(validate_incident_description(incident_raw))
        if upload is not None:
            validated_upload = validate_uploaded_log(upload.name, upload.getvalue())
    except InputValidationError as exc:
        validation_error = str(exc)
        st.error(validation_error)

    log_evidence = []
    passages = []
    if incident_text and validation_error is None:
        log_evidence = make_log_evidence(validated_upload, incident_text)
        retrieval_query = " ".join(
            [incident_text, *(record.excerpt for record in log_evidence)]
        )
        passages = keyword_search(retrieval_query, KNOWLEDGE_DIR)

    st.subheader("Retrieved troubleshooting knowledge")
    if validation_error:
        st.caption("Correct the input above to retrieve local guidance.")
    elif not incident_text:
        st.caption("Enter an incident description to search the local knowledge base.")
    elif not passages:
        st.info("No knowledge passage met the relevance threshold. The analysis will be asked to state this limitation.")
    else:
        for passage in passages:
            with st.expander(f"{passage.evidence_id} · {passage.source_name} · {passage.passage_id}"):
                st.text(passage.text)
                st.caption(f"BM25 score: {passage.bm25_score:.3f} · local keyword preview")

    try:
        current_index_status = index_status(
            knowledge_dir=KNOWLEDGE_DIR,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )
        if not current_index_status["indexed"]:
            st.warning("No semantic index exists yet. Build it to enable embedding search; keyword retrieval remains available.")
        elif current_index_status["stale"]:
            st.warning("The knowledge documents or embedding settings changed. Rebuild the index before semantic search.")
        else:
            st.caption(
                f"Semantic index ready · {current_index_status['indexed_chunks']} chunks · "
                f"{current_index_status['embedding_model']}"
            )
    except IndexErrorSafe as exc:
        current_index_status = {"indexed": False, "stale": True}
        st.warning(str(exc))

    index_clicked = st.button(
        "Rebuild knowledge index",
        disabled=not bool(api_key),
        help="This sends changed knowledge passages to the OpenAI Embeddings API and may incur usage charges.",
    )
    if index_clicked:
        try:
            with st.spinner("Updating the local semantic index…"):
                with openai.OpenAI(api_key=api_key, max_retries=0, timeout=45.0) as embedding_client:
                    stats = rebuild_index(
                        knowledge_dir=KNOWLEDGE_DIR,
                        persist_dir=persist_dir,
                        client=embedding_client,
                        embedding_model=embedding_model,
                    )
            if stats.unchanged:
                st.success(f"Index is already current: {stats.total_chunks} chunks; no embedding calls were needed.")
            else:
                st.success(
                    f"Index updated: {stats.total_chunks} chunks; {stats.added_chunks} added, "
                    f"{stats.updated_chunks} updated, {stats.skipped_chunks} unchanged, "
                    f"{stats.stale_chunks} stale removed."
                )
            st.rerun()
        except (IndexErrorSafe, EmbeddingError) as exc:
            st.error(str(exc))
        except openai.OpenAIError:
            st.error("The indexing API request failed. Check the API status, model access, and rate limits.")
        except Exception:
            st.error("The local knowledge index could not be updated. The active index remains available.")

    can_analyze = bool(api_key and incident_text and validation_error is None)
    analyze_clicked = st.button(
        "Analyze incident",
        type="primary",
        disabled=not can_analyze,
        help="Configure an API key and provide valid input to enable analysis.",
    )

    signature = hashlib.sha256(
        (
            incident_text
            + "\0"
            + (validated_upload.text if validated_upload else "")
            + "\0"
            + model
            + "\0"
            + str(current_index_status.get("indexed_chunks", 0))
            + "\0"
            + str(current_index_status.get("stale", True))
            + "\0"
            + str(current_index_status.get("corpus_hash", ""))
        ).encode("utf-8")
    ).hexdigest()
    if analyze_clicked:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_signature", None)
        try:
            with st.spinner("Analyzing the incident…"):
                with openai.OpenAI(api_key=api_key, max_retries=0, timeout=45.0) as analysis_client:
                    retrieval = retrieve_for_incident(
                        query=incident_text,
                        lexical_query=retrieval_query,
                        knowledge_dir=KNOWLEDGE_DIR,
                        persist_dir=persist_dir,
                        client=analysis_client,
                        embedding_model=embedding_model,
                    )
                    selected_passages = retrieval.passages
                    if retrieval.warning:
                        st.warning(retrieval.warning)
                    if not selected_passages:
                        st.info("No relevant knowledge evidence was retrieved for this incident.")
                    result = analyze_incident(
                        incident_text=incident_text,
                        log_evidence=log_evidence,
                        passages=selected_passages,
                        api_key=api_key,
                        model=model,
                        client=analysis_client,
                    )
                st.session_state["retrieved_passages"] = selected_passages
                st.session_state["retrieval_mode"] = retrieval.mode
            st.session_state["analysis_result"] = result
            st.session_state["analysis_signature"] = signature
        except AnalysisError as exc:
            st.error(str(exc))
        except openai.OpenAIError:
            st.error("Could not initialize or use the OpenAI API client. Check the API configuration and connection.")

    result = st.session_state.get("analysis_result")
    result_signature = st.session_state.get("analysis_signature")
    if result is not None and result_signature == signature:
        _render_analysis(result)
        records = [*result.evidence]
        _render_evidence(records)
        retrieved = st.session_state.get("retrieved_passages", [])
        if retrieved:
            st.markdown("#### Hybrid retrieval scores")
            for passage in retrieved:
                similarity = (
                    f"{passage.semantic_similarity:.3f}"
                    if passage.semantic_similarity is not None
                    else "BM25 fallback"
                )
                st.caption(
                    f"{passage.source_path} · BM25 {passage.bm25_score:.3f} · "
                    f"semantic {similarity} · RRF {passage.fused_score:.4f}"
                )
    elif result is not None:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_signature", None)

    if validated_upload and log_evidence:
        with st.expander(f"Sanitized log excerpts used as evidence ({len(log_evidence)})"):
            for record in log_evidence:
                st.caption(f"{record.evidence_id} · {record.source_name} · line {record.line_start}")
                st.code(record.excerpt, language="text")


if __name__ == "__main__":
    main()
