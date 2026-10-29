# AI-Powered IT Incident Analysis Assistant

A local portfolio demo that combines BM25 keyword retrieval, OpenAI embeddings, a persistent Chroma index, and an OpenAI structured analysis. The assistant is advisory: it does not execute commands, modify external systems, or save submitted incidents.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for live analysis (the app can show local retrieval without one)

## Install and run (Windows PowerShell)

```powershell
cd "F:\AI incident tracker"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python -m streamlit run app.py --server.address 127.0.0.1
```

In `.env`, replace the placeholder `OPENAI_API_KEY` value with your own key. `OPENAI_MODEL` defaults to `gpt-6-luna`; `OPENAI_EMBEDDING_MODEL` defaults to `text-embedding-3-small`; and `CHROMA_PERSIST_DIR` defaults to `data/chroma`. Never commit `.env`; it is excluded by `.gitignore`. Do not paste production logs or sensitive personal information into the demo. Redaction targets common secrets and email addresses but is not exhaustive.

To run the automated checks:

```powershell
python -m pytest -q
```

The tests use fake API responses and make no network calls. To run the retrieval evaluation against the hosted embedding model after building the index, explicitly run:

```powershell
python -m incident_assistant.evaluate --live
```

That command sends the 15 evaluation queries to the Embeddings API and may incur usage charges. Live analysis is sent to OpenAI only after the user submits a valid incident using **Analyze incident**.

## What the demo does

1. Validates a required incident description up to 10,000 characters and an optional UTF-8 `.log` or `.txt` upload up to 1 MiB.
2. Masks common credential patterns, JWT-like values, and email addresses before the app displays or submits log excerpts.
3. Uses BM25 over current local chunks for a no-key preview; after explicit analysis submission, it adds dense semantic search from the persistent Chroma index and fuses both rankings with reciprocal rank fusion.
4. Lets the user explicitly rebuild the index from `.txt` and `.md` documents in `knowledge_base/`. Reindexing unchanged documents makes no embedding calls; changed content is embedded and staged before the active index switches.
5. Requests structured incident analysis and validates model evidence IDs against application-created records.
6. Shows facts, unconfirmed potential causes, diagnostic steps, a conditional resolution, evidence limitations, and source excerpts with retrieval scores.

The knowledge base contains synthetic guidance about database connectivity, web/API errors, and authentication. Embedding vectors and the active index manifest live under the ignored `data/` directory. A troubleshooting passage is guidance, not proof of an incident cause. The local Chroma index is intended for development and portfolio demonstration, not multi-user production hosting.

## Security and limitations

- The app has no command execution or remediation features.
- API keys are loaded from `.env`; errors shown in the UI do not include raw SDK details.
- Uploaded data stays in memory in this app and is not persisted locally. Sanitized incident/log content is sent to the configured OpenAI model when the user explicitly analyzes it.
- Redaction is pattern-based, so inspect and sanitize inputs before submission.
- Severity and causes are model-generated advisory assessments and require human verification.
- Keyword retrieval may miss relevant documents or return weak matches. When no passage meets the threshold, the model is told there is no retrieved knowledge evidence.
- Index building and hybrid search use the configured OpenAI embedding model. Indexing and live evaluation require explicit user action and can incur API usage.
