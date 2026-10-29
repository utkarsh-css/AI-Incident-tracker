from types import SimpleNamespace

import pytest

from incident_assistant.embeddings import embed_query, embed_texts


class FakeEmbeddingAPI:
    def __init__(self):
        self.calls = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, *, model, input):
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i, _ in enumerate(input)]
        )


def test_embeddings_batch_and_preserve_input_order():
    client = FakeEmbeddingAPI()
    vectors = embed_texts(client, ["a", "b", "c"], batch_size=2)
    assert len(client.calls) == 2
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]


def test_empty_batch_makes_no_api_call():
    client = FakeEmbeddingAPI()
    assert embed_texts(client, []) == []
    assert client.calls == []


def test_query_embedding_returns_single_vector():
    client = FakeEmbeddingAPI()
    assert embed_query(client, "network timeout") == [0.0, 1.0]
