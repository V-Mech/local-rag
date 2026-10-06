from __future__ import annotations

import re
from typing import Any

class PlannerAgent:
    """Determines the minimal, grounded execution plan for a question."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        question = state["question"].strip()
        documents = state.get("documents", [])
        question_lower = question.lower()
        selected_document = state.get("selected_document")

        memory_markers = ("previous", "earlier", "before", "that", "it", "they", "continue")
        comparison_requested = "compare" in question_lower or "difference" in question_lower
        retrieval_required = bool(documents) and (
            bool(selected_document) or len(question.split()) > 0
        )
        use_memory = any(marker in question_lower.split() for marker in memory_markers)
        target_document_ids = self._matching_document_ids(question_lower, documents)

        if selected_document:
            target_document_ids = [selected_document]

        state["plan"] = {
            "intent": (
                "cross_document_comparison"
                if comparison_requested and len(target_document_ids) >= 2
                else "document_question"
                if retrieval_required
                else "no_documents_available"
            ),
            "retrieve": retrieval_required,
            "use_memory": use_memory,
            "comparison_requested": comparison_requested,
            "target_document_ids": target_document_ids,
            "steps": [
                "Plan the request",
                "Retrieve relevant document chunks" if retrieval_required else "Skip retrieval: no documents available",
                "Load relevant conversation memory" if use_memory else "Skip conversational memory",
                "Verify evidence quality",
                "Generate a cited Markdown response",
            ],
        }
        return state

    @staticmethod
    def _matching_document_ids(question: str, documents: list[dict[str, Any]]) -> list[str]:
        question_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", question)
            if len(term) >= 3
        }
        ignored_terms = {
            "compare", "with", "the", "and", "document", "documents", "what", "does",
            "about", "from", "this", "that", "have", "were", "which",
        }
        question_terms -= ignored_terms
        matches: list[tuple[int, str]] = []

        for document in documents:
            searchable = " ".join(
                [document.get("source", ""), document.get("summary", "")]
            ).lower()
            document_terms = set(re.findall(r"[a-z0-9]+", searchable))
            score = len(question_terms & document_terms)
            if score:
                matches.append((score, document["document_id"]))

        return [document_id for _, document_id in sorted(matches, reverse=True)]
