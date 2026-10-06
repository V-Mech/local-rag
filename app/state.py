from dataclasses import dataclass

@dataclass(frozen=True)
class RequestState:
    """Per-request UI state; no state is shared between users."""

    selected_document: str = ""
