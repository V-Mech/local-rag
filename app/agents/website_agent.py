from typing import Any

from app.ingest import DocumentIngestionService

class WebsiteAgent:
    """Optional ingestion agent for a validated public website URL."""

    def __init__(self, ingestion: DocumentIngestionService) -> None:
        self._ingestion = ingestion

    def ingest(self, url: str) -> dict[str, Any]:
        return self._ingestion.ingest_website(url)
