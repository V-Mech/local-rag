from __future__ import annotations

import logging

import requests

class OllamaClient:
    """Small, framework-free client for Ollama generation requests."""

    def __init__(self, model: str, endpoint: str, timeout_seconds: int) -> None:
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self._endpoint,
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            if not answer:
                raise ValueError("Ollama returned an empty response")
            return answer
        except (requests.RequestException, ValueError) as error:
            logging.getLogger(__name__).warning("Ollama generation failed: %s", error)
            raise RuntimeError("The local language model is unavailable. Please try again.") from error
