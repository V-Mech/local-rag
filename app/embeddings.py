from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """Owns the local embedding model used by ingestion and retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self._get_model().encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text on paragraph and sentence boundaries with bounded overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        while len(paragraph) > chunk_size:
            split_at = paragraph.rfind(" ", 0, chunk_size)
            split_at = split_at if split_at > chunk_size // 2 else chunk_size
            chunks.append(paragraph[:split_at].strip())
            paragraph = paragraph[max(0, split_at - overlap):].strip()

        current = paragraph

    if current:
        chunks.append(current)

    return chunks
