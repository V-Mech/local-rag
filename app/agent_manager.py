from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from app.agents.memory_agent import MemoryAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.response_agent import ResponseAgent
from app.agents.retrieval_agent import RetrieverAgent
from app.agents.verification_agent import VerificationAgent
from app.chroma_store import ChromaStore
from app.embeddings import EmbeddingService
from app.ingest import DocumentIngestionService
from app.ollama_client import OllamaClient

class AgentManager:
    """Application service that owns dependencies and runs the agent pipeline."""

    def __init__(self, project_directory: Path) -> None:
        data_directory = project_directory / "data"
        model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:latest")
        endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

        store = ChromaStore(data_directory)
        embeddings = EmbeddingService(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        ollama = OllamaClient(model_name, endpoint, timeout_seconds)

        self._store = store
        self._ingestion = DocumentIngestionService(store, embeddings)
        self._planner = PlannerAgent()
        self._retriever = RetrieverAgent(store, embeddings)
        self._memory = MemoryAgent(store, ollama)
        self._verification = VerificationAgent()
        self._response = ResponseAgent(ollama)

    def ask(
        self,
        question: str,
        session_id: str,
        selected_document: str | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "question": question,
            "session_id": session_id,
            "selected_document": selected_document,
            "documents": self._store.list_documents(),
            "retrieved_chunks": [],
            "history": [],
            "context": "",
            "confidence": 0.0,
            "sources": [],
            "answer": "",
        }

        try:
            for agent in (
                self._planner,
                self._retriever,
                self._memory,
                self._verification,
                self._response,
            ):
                state = agent.run(state)
        except (RuntimeError, ValueError) as error:
            logging.getLogger(__name__).exception("Agent pipeline failed")
            state["answer"] = str(error)
            state.setdefault("plan", {"steps": []})
            state.setdefault("verification", {"hallucination_risk": "high"})

        try:
            self._memory.save_turn(state)
        except Exception:
            logging.getLogger(__name__).exception("Failed to persist conversation memory")

        return state

    def ingest_document(self, file_path: Path, source_name: str) -> dict[str, str | int]:
        return self._ingestion.ingest_document(file_path, source_name)

    def ingest_website(self, url: str) -> dict[str, str | int]:
        return self._ingestion.ingest_website(url)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._store.list_documents()
