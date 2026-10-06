# Local Multi-Agent RAG

A framework-free, local RAG application built with FastAPI, Ollama, ChromaDB, and SentenceTransformers.

## Pipeline

`PlannerAgent → RetrieverAgent → MemoryAgent → VerificationAgent → ResponseAgent`

Each agent receives and returns the same Python dictionary. `AgentManager` owns all dependencies and runs the pipeline. No agent framework or mutable module-level agent/conversation state is used.

## Run

1. Use Python 3.10 or newer and install dependencies: `python -m pip install -r requirements.txt`.
2. Start Ollama and make the configured model available (default: `gpt-oss:latest`).
3. Run: `uvicorn app.main:app --host 127.0.0.1 --port 8000`.

Optional environment variables: `OLLAMA_MODEL`, `OLLAMA_URL`, `OLLAMA_TIMEOUT_SECONDS`, `EMBEDDING_MODEL`, `LOG_LEVEL`, and `COOKIE_SECURE`.

## Security model

The app is designed for a trusted local deployment. It applies upload bounds/type checks, CSRF protection, safe XML parsing, public-only website ingestion, and per-browser-session conversation memory. Add authentication and a tenant boundary before exposing it to multiple users or the public internet.

## Tests

Run `python -m unittest discover -s tests -v`. The tests use fakes for the embedding and language models, so they do not require Ollama or a model download.
