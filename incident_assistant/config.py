"""Environment-backed settings shared by the UI and CLI tools."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .analysis import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from .embeddings import DEFAULT_EMBEDDING_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDER_KEYS = {"", "replace-with-your-own-key", "your-api-key-here", "changeme"}


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    chat_model: str
    embedding_model: str
    persist_dir: Path
    # None omits the reasoning parameter, for models that do not accept it.
    reasoning_effort: str | None


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key.lower() in _PLACEHOLDER_KEYS:
        api_key = ""
    chat_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip() or None
    persist_setting = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    persist_dir = persist_setting if persist_setting.is_absolute() else PROJECT_ROOT / persist_setting
    return Settings(
        api_key=api_key or None,
        chat_model=chat_model,
        embedding_model=embedding_model,
        persist_dir=persist_dir,
        reasoning_effort=reasoning_effort,
    )
