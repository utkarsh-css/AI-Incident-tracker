"""Explicit OpenAI embedding calls; Chroma never calls a provider implicitly."""

from __future__ import annotations

import openai


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_TIMEOUT_SECONDS = 45.0


class EmbeddingError(RuntimeError):
    """Safe embedding failure intended for display without provider details."""


def embed_texts(
    client,
    texts: list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise EmbeddingError("The embedding service returned an incomplete batch.")
            vectors.extend([list(item.embedding) for item in data])
    except EmbeddingError:
        raise
    except openai.APITimeoutError as exc:
        raise EmbeddingError("The embedding request timed out. Try again.") from exc
    except openai.RateLimitError as exc:
        raise EmbeddingError("The embedding API is rate limited. Wait briefly, then try again.") from exc
    except openai.APIConnectionError as exc:
        raise EmbeddingError("Could not connect to the embedding API. Check your connection and try again.") from exc
    except openai.OpenAIError as exc:
        raise EmbeddingError("The embedding API could not complete the request. Check the model and API configuration.") from exc
    except Exception as exc:
        raise EmbeddingError("The embedding request could not be completed.") from exc
    return vectors


def embed_query(client, text: str, *, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    vectors = embed_texts(client, [text], model=model)
    return vectors[0]
