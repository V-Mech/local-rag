from pathlib import Path

from app.agent_manager import AgentManager

manager = AgentManager(Path(__file__).resolve().parent)
result = manager.ingest_website("https://en.wikipedia.org/wiki/Artificial_intelligence")
print(f"Stored {result['chunks']} chunks for document {result['document_id']}")
