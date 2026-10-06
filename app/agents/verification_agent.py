from __future__ import annotations

from typing import Any

class VerificationAgent:
    """Filters weak evidence and exposes a transparent grounding confidence score."""

    def __init__(self, minimum_relevance: float = 0.15, maximum_chunks: int = 6) -> None:
        self._minimum_relevance = minimum_relevance
        self._maximum_chunks = maximum_chunks

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        verified_chunks: list[dict[str, Any]] = []
        for chunk in state.get("retrieved_chunks", []):
            relevance = max(0.0, min(1.0, 1.0 - (float(chunk["distance"]) / 2.0)))
            if relevance >= self._minimum_relevance:
                verified_chunk = dict(chunk)
                verified_chunk["relevance"] = relevance
                verified_chunks.append(verified_chunk)

        verified_chunks = verified_chunks[: self._maximum_chunks]
        used_fallback = False
        if (
            not verified_chunks
            and state.get("plan", {}).get("comparison_requested")
            and state.get("retrieved_chunks")
        ):

            seen_sources: set[str] = set()
            for chunk in state["retrieved_chunks"]:
                source = chunk["metadata"].get("source", "Unknown")
                if source not in seen_sources:
                    fallback_chunk = dict(chunk)
                    fallback_chunk["relevance"] = max(
                        0.01, 1.0 - (float(chunk["distance"]) / 2.0)
                    )
                    verified_chunks.append(fallback_chunk)
                    seen_sources.add(source)
                if len(verified_chunks) == self._maximum_chunks:
                    break
            used_fallback = bool(verified_chunks)
        confidence = (
            round(sum(chunk["relevance"] for chunk in verified_chunks) / len(verified_chunks), 2)
            if verified_chunks
            else 0.0
        )
        state["verified_chunks"] = verified_chunks
        state["sources"] = sorted(
            {chunk["metadata"].get("source", "Unknown") for chunk in verified_chunks}
        )
        state["context"] = "\n\n".join(chunk["text"] for chunk in verified_chunks)
        state["confidence"] = confidence
        state["verification"] = {
            "context_sufficient": bool(verified_chunks),
            "hallucination_risk": "low" if confidence >= 0.7 else "medium" if confidence >= 0.45 else "high",
            "reason": (
                "Only weakly matched chunks were available; the response is constrained to them."
                if used_fallback
                else "Relevant source chunks were found."
                if verified_chunks
                else "No sufficiently relevant source chunks were found; unsupported claims must be avoided."
            ),
        }
        return state
