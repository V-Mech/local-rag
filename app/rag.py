from app.agent_manager import AgentManager

def ask_rag(
    query: str,
    manager: AgentManager,
    session_id: str,
    document_id: str | None = None,
) -> str:
    """Compatibility helper for programmatic callers of the agent pipeline."""
    return manager.ask(query, session_id, document_id)["answer"]
