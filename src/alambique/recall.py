"""Recall engine — semantic memory retrieval via MiMo-V2.5.

Also handles personality composition for session_start.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

import httpx

from alambique.llm_http import LlmOutcomeCallback, post_opencode_message
from alambique.models import Message

logger = logging.getLogger("alambique.recall")

RECALL_MODEL = "mimo-v2.5"

RECALL_PROMPT = """Eres el motor de búsqueda de memoria de "Alambique".

Agente que pregunta: {agent_name}
El usuario pregunta: "{query}"

═══ HILOS, CÁPSULAS Y SESIONES RELACIONADAS ═══
{top_sessions}

Redacta un resumen CONCISO (máximo 4 frases) que responda a la pregunta
basándote exclusivamente en los hilos, cápsulas y sesiones proporcionados.
No inventes. No especules. Si no hay información suficiente, dilo.

Responde SOLO con el texto del resumen, sin formato JSON ni metadatos."""

PERSONALITY_PROMPT = """Eres el compositor de personalidad de "Alambique".

Vas a generar la personalidad de un asistente para inyectar en su prompt de sistema.

El asistente es Lucy, la memoria personal de Víctor.
- Si en los rasgos de personalidad se especifica un nombre preferido o título descriptivo (ej. "Lucy, la copiloto espacial", "Lucy, la narradora de la Costa de la Espada"), úsalo para definir su identidad.
- Si no se especifica ningún título personalizado en los rasgos, refiérete al asistente simplemente como "Lucy".

═══ RASGOS DE PERSONALIDAD ═══
{traits}

═══ ESTADOS TEMPORALES ═══
{moods}

Redacta un texto de personalidad en español, en segunda persona ("Eres...").
- Prioriza los rasgos con mayor confidence.
- Los estados temporales van al final, entre paréntesis ("hoy: ...").
- Si no hay rasgos suficientes, devuelve null.
- Máximo 4 frases. Sé conciso pero con carácter.

Responde SOLO con el texto de personalidad o la palabra "null".
Sin JSON, sin metadatos."""


class RecallClient:
    """Calls opencode go API for recall and personality composition."""

    def __init__(
        self,
        api_key: str,
        on_outcome: LlmOutcomeCallback | None = None,
    ) -> None:
        self.api_key = api_key
        self._on_outcome = on_outcome
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def compose_summary(
        self,
        query: str,
        agent_name: str,
        context_items: list[dict],
    ) -> str:
        """Compose a summary from recall results (sessions, threads, capsules)."""
        context_text = _format_session_list(context_items)  # reuse, it handles id/key/snippet

        prompt = RECALL_PROMPT.format(
            query=query,
            agent_name=agent_name,
            top_sessions=context_text,
        )

        r = await self._call_llm(prompt)
        return (r or "").strip()

    async def compose_personality(
        self,
        capsule_text: str = "",
    ) -> str | None:
        """Compose a personality prompt for Lucy from relationship capsule."""
        if not (capsule_text or "").strip():
            return None

        prompt = PERSONALITY_PROMPT.format(
            traits=capsule_text,
            moods="(sin estados temporales)",
        )

        r = await self._call_llm(prompt)
        r = (r or "").strip()
        if r.lower() == "null" or not r:
            return None
        return r

    async def _call_llm(self, prompt: str) -> str:
        return await post_opencode_message(
            self.client,
            self.api_key,
            model=RECALL_MODEL,
            prompt=prompt,
            max_tokens=512,
            on_outcome=self._on_outcome,
            log_prefix="Recall LLM",
        )


def _format_session_list(sessions: list[dict]) -> str:
    if not sessions:
        return "(sin sesiones relacionadas)"
    lines = []
    for s in sessions:
        text = s.get("summary") or s.get("snippet", "?")
        sid = s.get("id", "")
        prefix = f"[{sid}] " if sid else ""
        lines.append(f"- {prefix}{text}")
    return "\n".join(lines)