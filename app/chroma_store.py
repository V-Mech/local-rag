from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import chromadb

class ChromaStore:
    """Persistence boundary for documents, chunks, and conversation memory."""

    _chunk_record_type = "chunk"
    _document_record_type = "document_meta"
    _memory_record_type = "memory"
    _uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    def __init__(self, data_directory: Path) -> None:
        data_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(data_directory / "chroma"))
        self._documents = client.get_or_create_collection(name="rag_collection")
        self._memory = client.get_or_create_collection(name="rag_memory")

    @staticmethod
    def new_document_id() -> str:
        return str(uuid.uuid4())

    @classmethod
    def is_document_id(cls, value: str | None) -> bool:
        return bool(value and cls._uuid_pattern.match(value))

    def add_chunks(
        self,
        document_id: str,
        source: str,
        filetype: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")

        self._documents.upsert(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {
                    "record_type": self._chunk_record_type,
                    "document_id": document_id,
                    "source": source,
                    "type": filetype,
                    "chunk_index": index,
                }
                for index in range(len(chunks))
            ],
        )

    def save_document_metadata(
        self,
        document_id: str,
        source: str,
        filetype: str,
        chunk_count: int,
        embedding: list[float],
        summary: str = "",
    ) -> None:

        self._documents.upsert(
            ids=[f"doc_meta_{document_id}"],
            documents=[summary or source],
            embeddings=[embedding],
            metadatas=[
                {
                    "record_type": self._document_record_type,
                    "document_id": document_id,
                    "source": source,
                    "type": filetype,
                    "chunks": chunk_count,
                    "summary": summary,
                }
            ],
        )

    def delete_document(self, document_id: str) -> None:
        self._documents.delete(where={"document_id": document_id})

    def document_ids_for_source(self, source: str) -> list[str]:
        return [
            document["document_id"]
            for document in self.list_documents()
            if document["source"] == source
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        results = self._documents.get(where={"record_type": self._document_record_type})
        documents = [
            {
                "document_id": metadata["document_id"],
                "source": metadata["source"],
                "type": metadata.get("type", ""),
                "chunks": metadata.get("chunks", 0),
                "summary": metadata.get("summary", ""),
            }
            for metadata in results["metadatas"]
            if metadata
        ]
        return sorted(documents, key=lambda document: document["source"].lower())

    def query_chunks(
        self,
        embedding: list[float],
        document_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, str]] = [{"record_type": self._chunk_record_type}]
        if document_id and self.is_document_id(document_id):
            filters.append({"document_id": document_id})

        where: dict[str, Any] = filters[0] if len(filters) == 1 else {"$and": filters}
        results = self._documents.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
        )

        return [
            {"text": text, "metadata": metadata or {}, "distance": distance}
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_document_chunks(self, document_id: str, limit: int = 5) -> list[str]:
        results = self._documents.get(
            where={"$and": [{"document_id": document_id}, {"record_type": self._chunk_record_type}]},
            limit=limit,
            include=["documents"],
        )
        return results["documents"]

    def add_memory(self, session_id: str, role: str, content: str, created_at: int) -> None:
        self._memory.add(
            ids=[str(uuid.uuid4())],
            documents=[content],
            embeddings=[[0.0] * 384],
            metadatas=[
                {
                    "record_type": self._memory_record_type,
                    "session_id": session_id,
                    "role": role,
                    "created_at": created_at,
                }
            ],
        )

    def get_memory(self, session_id: str) -> list[dict[str, Any]]:
        results = self._memory.get(
            where={"$and": [{"session_id": session_id}, {"record_type": self._memory_record_type}]},
            include=["documents", "metadatas"],
        )
        messages = [
            {"id": identifier, "content": content, "metadata": metadata or {}}
            for identifier, content, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
        return sorted(messages, key=lambda message: message["metadata"].get("created_at", 0))

    def delete_memory(self, ids: list[str]) -> None:
        if ids:
            self._memory.delete(ids=ids)
