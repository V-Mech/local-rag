import unittest

from app.agents.memory_agent import MemoryAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.response_agent import ResponseAgent
from app.agents.retrieval_agent import RetrieverAgent
from app.agents.verification_agent import VerificationAgent
from app.embeddings import chunk_text

class FakeEmbeddings:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]

class FakeStore:
    def query_chunks(self, embedding, document_id, limit):
        return [
            {
                "text": "The approved budget is 100 dollars.",
                "metadata": {"source": "budget.pdf", "document_id": "document-1"},
                "distance": 0.2,
            }
        ]

    def get_memory(self, session_id):
        return []

    def add_memory(self, session_id, role, content, created_at):
        return None

class FakeOllama:
    def generate(self, prompt):
        self.prompt = prompt
        return "The approved budget is 100 dollars. [Source: budget.pdf]"

class AgentPipelineTests(unittest.TestCase):
    def test_agents_share_and_enrich_one_state_dictionary(self):
        state = {
            "question": "What is the approved budget?",
            "session_id": "session-1",
            "selected_document": "document-1",
            "documents": [{"document_id": "document-1", "source": "budget.pdf"}],
            "history": [],
            "context": "",
            "confidence": 0,
            "answer": "",
        }
        store = FakeStore()
        ollama = FakeOllama()

        for agent in (
            PlannerAgent(),
            RetrieverAgent(store, FakeEmbeddings()),
            MemoryAgent(store, ollama),
            VerificationAgent(),
            ResponseAgent(ollama),
        ):
            state = agent.run(state)

        self.assertTrue(state["plan"]["retrieve"])
        self.assertEqual(state["sources"], ["budget.pdf"])
        self.assertGreater(state["confidence"], 0.8)
        self.assertIn("[Source: budget.pdf]", state["answer"])
        self.assertIn("### Sources", state["answer"])

    def test_verification_blocks_low_relevance_context(self):
        state = {"retrieved_chunks": [{"text": "noise", "metadata": {}, "distance": 1.9}]}
        result = VerificationAgent().run(state)
        self.assertEqual(result["confidence"], 0.0)
        self.assertFalse(result["verification"]["context_sufficient"])

    def test_planner_targets_both_named_documents_for_comparison(self):
        state = PlannerAgent().run(
            {
                "question": "Compare the NISM workbook with the analyst report.",
                "selected_document": None,
                "documents": [
                    {"document_id": "nism", "source": "NISM Currency Derivatives.pdf", "summary": "A workbook."},
                    {"document_id": "analyst", "source": "Dohful Analyst Report.pdf", "summary": "Business analysis."},
                ],
            }
        )
        self.assertEqual(state["plan"]["intent"], "cross_document_comparison")
        self.assertEqual(set(state["plan"]["target_document_ids"]), {"nism", "analyst"})

    def test_chunking_keeps_overlap_and_rejects_invalid_settings(self):
        chunks = chunk_text("a" * 1000, chunk_size=500, overlap=100)
        self.assertGreaterEqual(len(chunks), 2)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=100, overlap=100)

if __name__ == "__main__":
    unittest.main()
