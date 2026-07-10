"""Tests for OllamaClient — embedding and health endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from alambique.ollama_client import OllamaClient, OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM


@pytest.fixture
def client():
    return OllamaClient()


@pytest.fixture
async def aclient():
    c = OllamaClient()
    yield c
    await c.close()


class TestOllamaClientInit:
    def test_default_base_url(self):
        c = OllamaClient()
        assert c.base_url == OLLAMA_BASE_URL

    def test_custom_base_url(self):
        c = OllamaClient(base_url="http://localhost:12345")
        assert c.base_url == "http://localhost:12345"

    def test_client_lazy_init(self):
        c = OllamaClient()
        assert c._client is None


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self):
        c = OllamaClient()
        mock_response = MagicMock()
        mock_response.is_success = True
        c._client = AsyncMock()
        c._client.get = AsyncMock(return_value=mock_response)

        result = await c.health()
        assert result is True
        c._client.get.assert_called_once_with("http://localhost:11434/api/tags")

    @pytest.mark.asyncio
    async def test_health_down(self):
        c = OllamaClient()
        c._client = AsyncMock()
        c._client.get = AsyncMock(side_effect=Exception("Connection refused"))

        result = await c.health()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_http_error(self):
        c = OllamaClient()
        mock_response = MagicMock()
        mock_response.is_success = False
        c._client = AsyncMock()
        c._client.get = AsyncMock(return_value=mock_response)

        result = await c.health()
        assert result is False


class TestEnsureModel:
    @pytest.mark.asyncio
    async def test_model_already_present(self):
        c = OllamaClient()
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"models": [{"name": "bge-m3:latest"}]}
        c._client = AsyncMock()
        c._client.get = AsyncMock(return_value=mock_response)

        result = await c.ensure_model()
        assert result is True
        c._client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_missing_pulls(self):
        c = OllamaClient()
        mock_get = MagicMock()
        mock_get.is_success = True
        mock_get.json.return_value = {"models": [{"name": "llama3"}]}

        mock_post = MagicMock()
        mock_post.is_success = True

        c._client = AsyncMock()
        c._client.get = AsyncMock(return_value=mock_get)
        c._client.post = AsyncMock(return_value=mock_post)

        result = await c.ensure_model()
        assert result is True
        c._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_pull_fails(self):
        c = OllamaClient()
        mock_get = MagicMock()
        mock_get.is_success = True
        mock_get.json.return_value = {"models": []}

        c._client = AsyncMock()
        c._client.get = AsyncMock(return_value=mock_get)
        c._client.post = AsyncMock(side_effect=Exception("Network error"))

        result = await c.ensure_model()
        assert result is False


class TestEmbed:
    @pytest.mark.asyncio
    async def test_single_embed(self):
        c = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * EMBEDDING_DIM]
        }
        c._client = AsyncMock()
        c._client.post = AsyncMock(return_value=mock_response)

        result = await c.embed("hola mundo")
        assert len(result) == EMBEDDING_DIM
        c._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_uses_bge_m3(self):
        c = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.0] * EMBEDDING_DIM]}
        c._client = AsyncMock()
        c._client.post = AsyncMock(return_value=mock_response)

        await c.embed("test")
        call_args = c._client.post.call_args
        assert call_args[0][0] == f"{OLLAMA_BASE_URL}/api/embed"
        assert call_args[1]["json"]["model"] == EMBEDDING_MODEL
        assert call_args[1]["json"]["input"] == "test"

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        c = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1] * EMBEDDING_DIM,
                [0.2] * EMBEDDING_DIM,
                [0.3] * EMBEDDING_DIM,
            ]
        }
        c._client = AsyncMock()
        c._client.post = AsyncMock(return_value=mock_response)

        texts = ["uno", "dos", "tres"]
        result = await c.embed_batch(texts)
        assert len(result) == 3
        assert all(len(emb) == EMBEDDING_DIM for emb in result)
        # Verify it was a single API call
        c._client.post.assert_called_once()





class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        c = OllamaClient()
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        c._client = mock_client

        await c.close()
        mock_client.aclose.assert_called_once()
        assert c._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        c = OllamaClient()
        await c.close()  # _client is None, should not crash
        assert c._client is None
