# AI-Powered IT Incident Analysis Assistant

A local portfolio demo that combines BM25 keyword retrieval, OpenAI embeddings, a persistent Chroma index, and an OpenAI structured analysis. The assistant is advisory: it does not execute commands, modify external systems, or save submitted incidents.

## Requirements

- Python 3.10 or newer (the test suite has only been run on Python 3.14.5)
- An OpenAI API key for live analysis (the app can show local retrieval without one)

## Install and run (Windows PowerShell)

```powershell
cd "F:\AI incident tracker"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
notepad .env
python -m streamlit run app.py
```

`.streamlit/config.toml` binds the server to `127.0.0.1`, turns off Streamlit usage statistics, and hides tracebacks for unexpected errors. `requirements-dev.txt` adds `pytest` to the runtime dependencies in `requirements.txt`.

In `.env`, replace the placeholder `OPENAI_API_KEY` value with your own key. `OPENAI_MODEL` defaults to `gpt-6-luna`; `OPENAI_REASONING_EFFORT` defaults to `low` (set it empty for models that do not accept a reasoning effort); `OPENAI_EMBEDDING_MODEL` defaults to `text-embedding-3-small`; and `CHROMA_PERSIST_DIR` defaults to `data/chroma`. Never commit `.env`; it is excluded by `.gitignore`. Do not paste production logs or sensitive personal information into the demo. Redaction targets common secrets and email addresses but is not exhaustive.

To run the automated checks:

```powershell
python -m pytest -q
```

The tests use fake API responses and make no OpenAI calls. Chroma's anonymized telemetry is disabled for every client the app opens. To run the retrieval evaluation against the hosted embedding model after building the index, explicitly run:

```powershell
python -m incident_assistant.evaluate --live
```

That command sends the 20 evaluation queries (two include an attached log) to the Embeddings API and may incur usage charges. Live analysis is sent to OpenAI only after the user submits a valid incident using **Analyze incident**.

## What the demo does

1. Validates a required incident description up to 10,000 characters and an optional UTF-8 `.log` or `.txt` upload up to 1 MiB.
2. Masks common credential patterns before the app displays or submits incident text or log excerpts: `key=value` secrets including prefixed names such as `DB_PASSWORD`, `access_token` or Azure `AccountKey`; `Authorization` header values for any scheme; command-line flags such as `--password X` and MySQL `-pX`; cookie headers; credentials in URLs; private-key blocks; JWT-like values; common provider key formats (OpenAI, GitHub, AWS, Slack, Stripe, Google, GitLab, npm); and email addresses. Values that carry diagnostic meaning rather than a credential are kept, such as `auth_token_expired=true`, `max_tokens=2000` or `pwd=/home/app`. Masking keeps line numbers aligned with the uploaded file.
3. Uses BM25 over current local chunks for a no-key preview; after explicit analysis submission, it adds dense semantic search from the persistent Chroma index and fuses both rankings with reciprocal rank fusion.
4. Lets the user explicitly rebuild the index from `.txt` and `.md` documents in `knowledge_base/`. Reindexing unchanged documents makes no embedding calls; changed content is embedded and staged before the active index switches.
5. Requests structured incident analysis and validates model evidence IDs against application-created records.
6. Shows facts, unconfirmed potential causes, diagnostic steps, a conditional resolution, evidence limitations, the evidence the analysis cited (separately from evidence it was given but did not cite), the retrieval mode (hybrid or BM25 fallback), and retrieval scores.

The knowledge base contains synthetic guidance about database connectivity, web/API errors, and authentication. Embedding vectors and the active index manifest live under the ignored `data/` directory. A troubleshooting passage is guidance, not proof of an incident cause. The local Chroma index is intended for development and portfolio demonstration, not multi-user production hosting.

## Security and limitations

- The app has no command execution or remediation features.
- API keys are loaded from `.env`; errors shown in the UI do not include raw SDK details.
- Uploaded data stays in memory in this app and is not persisted locally. Sanitized incident/log content is sent to the configured OpenAI model when the user explicitly analyzes it.
- Redaction is pattern-based, so inspect and sanitize inputs before submission. It does not mask IP addresses, hostnames, or usernames.
- Severity and causes are model-generated advisory assessments and require human verification.
- Keyword retrieval may miss relevant documents or return weak matches. A keyword candidate must share at least two distinct non-stop-word terms with a passage, counting the incident description plus log lines that report a failure (ERROR, FATAL, timeout, refused, 5xx). The bar rises to 10% of those distinct terms, at most four, for long input. Routine log lines (such as INFO) can rerank passages but cannot make one qualify. Semantic search embeds the same description-plus-failure-lines text, and a dense candidate must reach cosine similarity 0.35. These thresholds are uncalibrated demo defaults. When no passage qualifies, the model is told there is no retrieved knowledge evidence.
- Index building and hybrid search use the configured OpenAI embedding model. Indexing and live evaluation require explicit user action and can incur API usage.
- Rebuilds are serialized within one Streamlit process. Two separate processes using the same index directory are not coordinated.
- Evidence IDs are validated, but a valid citation does not make a statement correct: any fact may cite the whole incident description (`I1`).

## Evaluation limitations

- The knowledge base has three short synthetic documents, each a single chunk. Up to three passages is therefore the whole corpus, so recall@3 is a weak signal; read recall@1 and MRR first. Multi-source cases also report whether every expected source was retrieved.
- The 20 cases were written alongside the thresholds, so the results do not demonstrate general retrieval quality.
- The evaluation is source-level and covers retrieval only. It does not assess whether the generated analysis is correct or useful.
