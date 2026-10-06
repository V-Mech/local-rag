from pathlib import Path
from uuid import uuid4

from app.agent_manager import AgentManager

manager = AgentManager(Path(__file__).resolve().parent)
question = input("Ask a question: ").strip()
result = manager.ask(question, session_id=str(uuid4()))

print("\nANSWER:\n")
print(result["answer"])
