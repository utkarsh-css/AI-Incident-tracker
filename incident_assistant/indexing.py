"""Chunk local knowledge documents and maintain a persistent Chroma index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from .embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts


INDEX_VERSION = "1"
COLLECTION_PREFIX = "incident_kb"
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250


class IndexErrorSafe(RuntimeError):
    """Safe, user-displayable indexing error."""


class IndexCompatibilityError(IndexErrorSafe):
    pass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    chunk_index: int
    text: str
    content_hash: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
            "index_version": INDEX_VERSION,
        }


@dataclass(frozen=True)
class IndexStats:
    collection_name: str | None
    total_chunks: int
    skipped_chunks: int
    added_chunks: int
    updated_chunks: int
    stale_chunks: int
    unchanged: bool = False


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_long_piece(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_end = max(text.rfind(mark, start, end) for mark in (". ", "? ", "! ", "; "))
            if sentence_end >= start + max_chars // 2:
                end = sentence_end + 1
            else:
                whitespace = text.rfind(" ", start + max_chars // 2, end)
                if whitespace > start:
                    end = whitespace
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start
    return pieces


def chunk_text(text: str, *, max_chars: int = MAX_CHUNK_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    return _split_long_piece(normalized, max_chars, overlap_chars)


def load_chunks(knowledge_dir: str | Path) -> list[DocumentChunk]:
    root = Path(knowledge_dir)
    if not root.exists():
        return []
    result: list[DocumentChunk] = []
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".md"}),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    for path in paths:
        try:
            source_path = path.relative_to(root).as_posix()
            contents = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            raise IndexErrorSafe("A knowledge document could not be read as UTF-8 text.") from exc
        for chunk_index, text in enumerate(chunk_text(contents), start=0):
            chunk_id = _sha256(f"{source_path}\0{chunk_index}")
            result.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    text=text,
                    content_hash=_sha256(text),
                )
            )
    return result


def _manifest_path(persist_dir: str | Path) -> Path:
    path = Path(persist_dir)
    return path.parent / f"{path.name}_active_index.json"


def _read_manifest(persist_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = _manifest_path(persist_dir)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexErrorSafe("The local knowledge index manifest is unreadable. Rebuild the index.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("collection_name"), str):
        raise IndexErrorSafe("The local knowledge index manifest is invalid. Rebuild the index.")
    return data


def _write_manifest(persist_dir: str | Path, manifest: dict[str, Any]) -> None:
    destination = _manifest_path(persist_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def open_active_collection(
    persist_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[chromadb.ClientAPI, Any | None, dict[str, Any] | None]:
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
    except Exception as exc:
        raise IndexErrorSafe("The local semantic index could not be opened. Check the index directory or rebuild it.") from exc
    manifest = _read_manifest(persist_dir)
    if manifest is None:
        return client, None, None
    if manifest.get("embedding_model") != embedding_model or manifest.get("index_version") != INDEX_VERSION:
        raise IndexCompatibilityError("The local index uses a different embedding model or index format. Rebuild it.")
    try:
        collection = client.get_collection(name=manifest["collection_name"])
    except Exception as exc:
        raise IndexErrorSafe("The active local knowledge index is unavailable. Rebuild it.") from exc
    return client, collection, manifest


def _corpus_fingerprint(chunks: list[DocumentChunk], embedding_model: str) -> str:
    serialized = "\n".join(
        f"{chunk.chunk_id}:{chunk.content_hash}" for chunk in chunks
    )
    return _sha256(f"{INDEX_VERSION}\0{embedding_model}\0{serialized}")


def rebuild_index(
    *,
    knowledge_dir: str | Path,
    persist_dir: str | Path,
    client,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> IndexStats:
    chunks = load_chunks(knowledge_dir)
    if not chunks:
        raise IndexErrorSafe("No non-empty .txt or .md knowledge documents were found; the existing index was kept.")

    try:
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    except Exception as exc:
        raise IndexErrorSafe("The local semantic index could not be opened. Check the index directory.") from exc
    manifest = _read_manifest(persist_dir)
    old_collection = None
    compatible_old = False
    old_rows: dict[str, dict[str, Any]] = {}
    if manifest:
        try:
            old_collection = chroma_client.get_collection(name=manifest["collection_name"])
            compatible_old = (
                manifest.get("embedding_model") == embedding_model
                and manifest.get("index_version") == INDEX_VERSION
            )
            if compatible_old:
                old_data = old_collection.get(include=["embeddings", "metadatas", "documents"])
                for index, chunk_id in enumerate(old_data["ids"]):
                    old_rows[chunk_id] = {
                        "embedding": old_data["embeddings"][index],
                        "metadata": old_data["metadatas"][index],
                        "document": old_data["documents"][index],
                    }
        except Exception:
            # A missing/corrupt old collection can be recovered by staging a complete replacement.
            old_collection = None
            compatible_old = False
            old_rows = {}

    current_ids = {chunk.chunk_id for chunk in chunks}
    skipped = 0
    added = 0
    updated = 0
    changed_chunks: list[DocumentChunk] = []
    for chunk in chunks:
        old = old_rows.get(chunk.chunk_id)
        unchanged = (
            old is not None
            and old["metadata"].get("content_hash") == chunk.content_hash
            and old["metadata"].get("index_version") == INDEX_VERSION
        )
        if unchanged:
            skipped += 1
        else:
            changed_chunks.append(chunk)
            if old is None or not compatible_old:
                added += 1
            else:
                updated += 1
    stale = len(set(old_rows) - current_ids) if compatible_old else 0
    corpus_hash = _corpus_fingerprint(chunks, embedding_model)
    if manifest and compatible_old and skipped == len(chunks) and stale == 0:
        return IndexStats(manifest["collection_name"], len(chunks), skipped, 0, 0, 0, unchanged=True)

    # Complete all remote embedding calls before creating or changing any active index.
    new_vectors = embed_texts(client, [chunk.text for chunk in changed_chunks], model=embedding_model)
    changed_vectors = dict(zip((chunk.chunk_id for chunk in changed_chunks), new_vectors, strict=True))

    staged_name = f"{COLLECTION_PREFIX}_{uuid.uuid4().hex[:24]}"
    try:
        staged = chroma_client.create_collection(
            name=staged_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
                "index_version": INDEX_VERSION,
                "corpus_hash": corpus_hash,
            },
        )
        vectors = []
        for chunk in chunks:
            if chunk.chunk_id in changed_vectors:
                vectors.append(changed_vectors[chunk.chunk_id])
            else:
                old_vector = old_rows[chunk.chunk_id]["embedding"]
                vectors.append(old_vector.tolist() if hasattr(old_vector, "tolist") else list(old_vector))
        staged.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() | {"embedding_model": embedding_model} for chunk in chunks],
        )
        new_manifest = {
            "collection_name": staged_name,
            "embedding_model": embedding_model,
            "index_version": INDEX_VERSION,
            "corpus_hash": corpus_hash,
            "chunk_count": len(chunks),
        }
        _write_manifest(persist_dir, new_manifest)
    except Exception as exc:
        try:
            chroma_client.delete_collection(name=staged_name)
        except Exception:
            pass
        if isinstance(exc, IndexErrorSafe):
            raise
        raise IndexErrorSafe("The new knowledge index could not be saved; the previous active index remains available.") from exc

    if manifest and manifest.get("collection_name") != staged_name:
        try:
            chroma_client.delete_collection(name=manifest["collection_name"])
        except Exception:
            # The active manifest already points at the completed staged collection.
            pass
    return IndexStats(staged_name, len(chunks), skipped, added, updated, stale)


def index_status(
    *, knowledge_dir: str | Path, persist_dir: str | Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> dict[str, Any]:
    chunks = load_chunks(knowledge_dir)
    manifest = _read_manifest(persist_dir)
    if manifest is None:
        return {"indexed": False, "source_chunks": len(chunks), "indexed_chunks": 0, "stale": True}
    compatible = (
        manifest.get("embedding_model") == embedding_model
        and manifest.get("index_version") == INDEX_VERSION
    )
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_collection(name=manifest["collection_name"])
        indexed = collection.get(include=["metadatas"])
        indexed_by_id = {chunk_id: metadata for chunk_id, metadata in zip(indexed["ids"], indexed["metadatas"], strict=True)}
    except Exception as exc:
        raise IndexErrorSafe("The local knowledge index is unavailable. Rebuild it.") from exc
    stale = not compatible or len(chunks) != len(indexed_by_id)
    if not stale:
        stale = any(
            chunk.chunk_id not in indexed_by_id
            or indexed_by_id[chunk.chunk_id].get("content_hash") != chunk.content_hash
            for chunk in chunks
        )
    return {
        "indexed": True,
        "source_chunks": len(chunks),
        "indexed_chunks": len(indexed_by_id),
        "stale": stale,
        "embedding_model": manifest.get("embedding_model"),
        "corpus_hash": manifest.get("corpus_hash"),
    }
