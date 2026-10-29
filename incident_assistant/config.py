"""Environment-backed settings shared by the UI and CLI tools."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .analysis import DEFAULT_MODEL
from .embeddings import DEFAULT_EMBEDDING_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> tuple[str | None, str, str, Path]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key.lower() in {"", "replace-with-your-own-key", "your-api-key-here", "changeme"}:
        api_key = ""
    chat_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
    persist_setting = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    persist_dir = persist_setting if persist_setting.is_absolute() else PROJECT_ROOT / persist_setting
    return api_key or None, chat_model, embedding_model, persist_dir
