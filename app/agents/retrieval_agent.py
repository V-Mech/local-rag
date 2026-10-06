from __future__ import annotations

from typing import Any

from app.chroma_store import ChromaStore
from app.embeddings import EmbeddingService

class RetrieverAgent:
    """Retrieves and distance-ranks chunk records from ChromaDB."""

    def __init__(self, store: ChromaStore, embeddings: EmbeddingService, top_k: int = 12) -> None:
        self._store = store
        self._embeddings = embeddings
        self._top_k = top_k

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state["plan"]["retrieve"]:
            state["retrieved_chunks"] = []
            return state

        embedding = self._embeddings.embed([state["question"]])[0]
        document_ids = state["plan"].get("target_document_ids", [])
        if state.get("selected_document"):
            document_ids = [state["selected_document"]]

        chunks: list[dict[str, Any]] = []
        if document_ids:
            per_document_limit = max(3, self._top_k // len(document_ids))
            for document_id in document_ids:
                chunks.extend(
                    self._store.query_chunks(
                        embedding=embedding,
                        document_id=document_id,
                        limit=per_document_limit,
                    )
                )
        else:
            chunks = self._store.query_chunks(
                embedding=embedding,
                document_id=None,
                limit=self._top_k,
            )
        state["retrieved_chunks"] = sorted(chunks, key=lambda chunk: chunk["distance"])
        state["sources"] = sorted(
            {chunk["metadata"].get("source", "Unknown") for chunk in chunks}
        )
        return state
