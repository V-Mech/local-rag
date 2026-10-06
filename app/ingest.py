from __future__ import annotations

import logging
from pathlib import Path

from app.chroma_store import ChromaStore
from app.document_loader import load_docx, load_pdf, load_pptx, load_txt, load_xml
from app.embeddings import EmbeddingService, chunk_text
from app.scraper import scrape_website

class DocumentIngestionService:
    """Extracts, chunks, embeds, and persists one document atomically by ID."""

    def __init__(self, store: ChromaStore, embeddings: EmbeddingService) -> None:
        self._store = store
        self._embeddings = embeddings

    def ingest_document(self, file_path: Path, source_name: str | None = None) -> dict[str, str | int]:
        extension = file_path.suffix.lower().lstrip(".")
        loaders = {
            "pdf": load_pdf,
            "docx": load_docx,
            "pptx": load_pptx,
            "xml": load_xml,
            "txt": load_txt,
        }
        loader = loaders.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported document type: {extension or 'unknown'}")

        text = loader(file_path)
        if not text.strip():
            raise ValueError("The document contains no readable text")

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("The document did not produce any searchable chunks")

        document_id = self._store.new_document_id()
        source = source_name or file_path.name
        embeddings = self._embeddings.embed(chunks)
        previous_document_ids = self._store.document_ids_for_source(source)
        try:
            self._store.add_chunks(
                document_id=document_id,
                source=source,
                filetype=extension,
                chunks=chunks,
                embeddings=embeddings,
            )
            self._store.save_document_metadata(
                document_id=document_id,
                source=source,
                filetype=extension,
                chunk_count=len(chunks),
                embedding=embeddings[0],
            )
        except Exception:
            self._store.delete_document(document_id)
            raise

        for previous_document_id in previous_document_ids:
            self._store.delete_document(previous_document_id)
        logging.getLogger(__name__).info(
            "Indexed document id=%s source=%s chunks=%s", document_id, source, len(chunks)
        )
        return {"document_id": document_id, "source": source, "chunks": len(chunks)}

    def ingest_website(self, url: str) -> dict[str, str | int]:
        text = scrape_website(url)
        if not text.strip():
            raise ValueError("The website did not contain readable text")

        chunks = chunk_text(text)
        document_id = self._store.new_document_id()
        embeddings = self._embeddings.embed(chunks)
        previous_document_ids = self._store.document_ids_for_source(url)
        try:
            self._store.add_chunks(
                document_id=document_id,
                source=url,
                filetype="website",
                chunks=chunks,
                embeddings=embeddings,
            )
            self._store.save_document_metadata(
                document_id=document_id,
                source=url,
                filetype="website",
                chunk_count=len(chunks),
                embedding=embeddings[0],
            )
        except Exception:
            self._store.delete_document(document_id)
            raise

        for previous_document_id in previous_document_ids:
            self._store.delete_document(previous_document_id)
        return {"document_id": document_id, "source": url, "chunks": len(chunks)}
