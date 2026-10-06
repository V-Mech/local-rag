from pathlib import Path
from typing import Any

from app.chroma_store import ChromaStore

class DocumentCatalog:
    """Read-only adapter for the legacy JSON manifest during migration to Chroma metadata."""

    def __init__(self, store: ChromaStore, legacy_manifest: Path) -> None:
        self._store = store
        self._legacy_manifest = legacy_manifest

    def list_documents(self) -> list[dict[str, Any]]:

        return self._store.list_documents()
