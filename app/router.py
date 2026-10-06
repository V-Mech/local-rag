from typing import Any

from app.agents.planner_agent import PlannerAgent

def choose_tool(query: str, documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compatibility adapter; PlannerAgent is the only routing authority."""
    state = PlannerAgent().run({"question": query, "documents": documents or []})
    return state["plan"]
