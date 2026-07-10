"""Ollama client for embeddings using bge-m3 (local, GPU via RTX 4080)."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger("alambique.ollama")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024


class OllamaClient:
    """Async client for Ollama embeddings and chat."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health(self) -> bool:
        """Check if ollama is reachable."""
        try:
            r = await self.client.get(f"{self.base_url}/api/tags")
            return r.is_success
        except Exception:
            return False

    async def ensure_model(self, model: str = EMBEDDING_MODEL) -> bool:
        """Ensure the embedding model is pulled."""
        try:
            r = await self.client.get(f"{self.base_url}/api/tags")
            if r.is_success:
                models = [m["name"] for m in r.json().get("models", [])]
                if model in models or any(m.startswith(f"{model}:") for m in models):
                    return True
            # Pull it
            logger.info("Descargando modelo %s...", model)
            r = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=httpx.Timeout(300.0),
            )
            return r.is_success
        except Exception as e:
            logger.error("Error asegurando modelo %s: %s", model, e)
            return False

    async def embed(self, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
        """Generate embedding for a text string."""
        r = await self.client.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
        )
        r.raise_for_status()
        data = r.json()
        return data["embeddings"][0]

    async def embed_batch(self, texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        if not texts:
            return []
        r = await self.client.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
        return data["embeddings"]


