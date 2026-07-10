"""Recall engine — semantic memory retrieval via Qwen3.7 Plus.

Also handles personality composition for session_start.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from alambique.models import Fact, FactCategory

logger = logging.getLogger("alambique.recall")

RECALL_MODEL = "qwen3.7-plus"

RECALL_PROMPT = """Eres el motor de búsqueda de memoria de "Alambique".

Agente que pregunta: {agent_name}
El usuario pregunta: "{query}"

═══ HECHOS RELEVANTES ═══
{top_facts}

═══ SESIONES RELACIONADAS ═══
{top_sessions}

Redacta un resumen CONCISO (máximo 4 frases) que responda a la pregunta
basándote exclusivamente en los hechos y sesiones proporcionados.
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

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
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
        facts: list[dict],
        sessions: list[dict],
    ) -> str:
        """Compose a summary from recall results."""
        facts_text = _format_fact_list(facts)
        sessions_text = _format_session_list(sessions)

        prompt = RECALL_PROMPT.format(
            query=query,
            agent_name=agent_name,
            top_facts=facts_text,
            top_sessions=sessions_text,
        )

        r = await self._call_llm(prompt)
        return r.strip()

    async def compose_personality(
        self,
        traits: list[Fact],
        states: list[Fact],
    ) -> str | None:
        """Compose a personality prompt for Lucy."""
        traits_text = _format_facts_for_personality(traits)
        states_text = _format_facts_for_personality(states)

        if not traits_text.strip() and not states_text.strip():
            return None

        prompt = PERSONALITY_PROMPT.format(
            traits=traits_text or "(sin rasgos definidos aún)",
            moods=states_text or "(sin estados temporales)",
        )

        r = await self._call_llm(prompt)
        r = r.strip()
        if r.lower() == "null" or not r:
            return None
        return r

    async def _call_llm(self, prompt: str) -> str:
        r = await self.client.post(
            "https://opencode.ai/zen/go/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": RECALL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "thinking": {"type": "disabled"},
            },
        )
        r.raise_for_status()
        data = r.json()
        # Anthropic-compatible format: content is an array of blocks
        for block in data["content"]:
            if block.get("type") == "text":
                return block["text"]
        return ""


def _format_fact_list(facts: list[dict]) -> str:
    if not facts:
        return "(sin hechos relevantes)"
    lines = []
    for f in facts:
        lines.append(f"- [{f.get('category', '?')}] {f.get('key', '?')}: {f.get('value', '?')}")
    return "\n".join(lines)


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


def _format_facts_for_personality(facts: list[Fact]) -> str:
    lines = []
    for f in facts:
        lines.append(f"  [{f.confidence:.1f}] {f.key}: {f.value}")
    return "\n".join(lines)
