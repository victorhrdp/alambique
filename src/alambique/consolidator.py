"""Consolidation agent — extracts facts from conversations using Qwen3.7 Plus."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

import httpx

from alambique.memory_config import STATE_DEFAULT_TTL
from alambique.models import (
    ConsolidationAction,
    ConsolidationFactItem,
    ConsolidationResponse,
    Fact,
    FactCategory,
    Message,
)

logger = logging.getLogger("alambique.consolidator")

CONSOLIDATION_PROMPT = """Eres un agente de consolidación de memoria para "Alambique",
un sistema que destila conversaciones en hechos almacenados.

═══ METADATOS ═══
Agente: {agent_name}
Fecha: {date}
═══ HECHOS EXISTENTES ═══
{existing_facts}
═══ CONVERSACIÓN ═══
{conversation}

Tu tarea es extraer hechos objetivos y decidir cómo integrarlos
con los hechos ya existentes.

═══════════════════════════════════════
CATEGORÍAS DE HECHOS (menú cerrado — elige UNA)
═══════════════════════════════════════
- personal:     Datos ESTABLES sobre el usuario: nombre, nacimiento, familia,
                profesión, rasgos de fondo ("es gracioso", "es pesado").
                NO decaen. SÍ se actualizan si el usuario corrige algo.
- personality:  Rasgos del ASISTENTE definidos por el usuario ("Lucy es sarcástica").
                SOLO de mensajes role="user". NO decaen.
- preference:   Gustos y elecciones que PUEDEN cambiar: OS, herramientas, juegos,
                estaciones ("prefiere Linux", "le gustan los gatos").
                NO uses para datos biográficos ni posesiones.
- possessions:  Cosas que el usuario POSEE: teclado, GPU, casa, coche.
                NO uses para software/OS (eso es preference).
- state:        Estados TEMPORALES: ánimo, dolor físico, cansancio, "voy a descansar",
                "ya estoy bien". SIEMPRE ttl={state_default_ttl} (24h). Caducan por TTL, no por tiempo.

═══════════════════════════════════════
REGLAS DE EXTRACCIÓN
═══════════════════════════════════════
1. SOLO extrae hechos con sustancia. Ignora saludos, despedidas,
   agradecimientos, confirmaciones vacías ("ok", "vale", "gracias").
2. NO consolidar anécdotas pasajeras sin relevancia futura
   (ej. "hoy vi un pato rosa") — quedan en la conversación; usa action "discard".
3. Categoría "personality": SOLO mensajes role="user". El asistente NUNCA
   define su personalidad.
4. Categoría "state": SIEMPRE ttl={state_default_ttl}. Si un state anterior queda obsoleto
   (ej. dolor resuelto), usa "update" o contradice el hecho previo — no dejes
   states contradictorios activos.
5. Clasificación — pregúntate en orden:
   a) ¿Seguirá siendo verdad dentro de un mes?
      - No → "state" + ttl (ánimo, dolor, pausa momentánea)
      - Sí, pero puede cambiar con una decisión del usuario → "preference"
        (OS, editor, juego favorito, herramienta)
      - Sí, es un objeto que posee → "possessions"
      - Sí, es dato estable sobre el usuario → "personal"
      - Es sobre el asistente → "personality"
   b) ¿Cambiaría si el usuario toma una decisión un martes cualquiera?
      - Sí → NO es "personal"; usa "preference" o "possessions".
6. PROHIBIDO:
   - Dolor físico o salud temporal en "personal"
   - Gustos ("le gusta X") en "personal" — usa "preference"
   - OS/software en "possessions" — usa "preference"
   - Teclado/GPU/casa en "preference" — usa "possessions"
7. Sé preciso y factual. Un hecho = una idea atómica.
8. SIEMPRE responde en español.

═══════════════════════════════════════
ACCIONES DISPONIBLES
═══════════════════════════════════════
- create:     El hecho es nuevo, no existe nada similar
- update:     El hecho reemplaza a uno existente (info más reciente o corregida)
- merge:      El hecho complementa a uno existente (fusiona valores)
- contradict: El hecho contradice a uno existente. Ambos se conservan.
- discard:    El hecho es redundante, irrelevante o ya está cubierto

Para update/merge/contradict, indica related_fact_id (el ID del hecho existente).
Para create/discard, usa null.

═══════════════════════════════════════
CONFIDENCE
═══════════════════════════════════════
- 1.0: El usuario lo afirma explícitamente
- 0.7-0.9: Fuertemente implicado por la conversación
- 0.4-0.6: Inferencia razonable pero no explícita
- <0.4: No lo uses, mejor discart

═══════════════════════════════════════
SESSION SUMMARY
═══════════════════════════════════════
El resumen de sesión debe ser TEMÁTICO, no cronológico.
Describe los temas tratados, no la secuencia de mensajes.
Debe ser denso y útil para búsqueda semántica futura.
Ejemplo malo:  "Primero hablaron de X, luego de Y..."
Ejemplo bueno: "Discusión sobre diseño de memoria para agentes.
               Se decidió usar SQLite+vec, bge-m3 para embeddings..."

═══════════════════════════════════════
FORMATO DE RESPUESTA (JSON estricto)
═══════════════════════════════════════
Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto adicional:

{{
  "facts": [
    {{
      "action": "create",
      "key": "nombre_del_hecho",
      "value": "descripción precisa y factual en español",
      "category": "personal",
      "confidence": 1.0,
      "ttl": null,
      "related_fact_id": null,
      "reason": "El usuario afirma explícitamente que..."
    }},
    {{
      "action": "update",
      "key": "os_primary",
      "value": "Víctor usa CachyOS como sistema principal",
      "category": "preference",
      "confidence": 1.0,
      "ttl": null,
      "related_fact_id": 42,
      "reason": "Preferencia de SO actualizada por el usuario"
    }},
    {{
      "action": "create",
      "key": "health_neck_pain",
      "value": "Víctor tiene dolor de cuello que le impide trabajar ahora",
      "category": "state",
      "confidence": 1.0,
      "ttl": {state_default_ttl},
      "related_fact_id": null,
      "reason": "Estado físico temporal explícito"
    }},
    {{
      "action": "discard",
      "key": "anecdote_pink_duck",
      "value": "Víctor vio un pato rosa",
      "category": "state",
      "confidence": 0.0,
      "ttl": null,
      "related_fact_id": null,
      "reason": "Anécdota pasajera sin relevancia futura; queda en la conversación"
    }}
  ],
  "session_summary": "Resumen temático en español, 2-4 frases, denso y buscable."
}}
"""


CONSOLIDATION_MODEL = "qwen3.7-plus"


class ConsolidatorClient:
    """Calls opencode go API with Qwen3.7 Plus for consolidation."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(45.0))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def consolidate(
        self,
        agent_name: str,
        messages: list[Message],
        existing_facts: list[Fact],
    ) -> ConsolidationResponse:
        """Send conversation to the consolidator LLM and parse the response."""

        conversation = _format_conversation(messages)
        facts_text = _format_existing_facts(existing_facts)
        date = str(messages[0].timestamp) if messages else "desconocida"

        prompt = CONSOLIDATION_PROMPT.format(
            agent_name=agent_name,
            date=date,
            existing_facts=facts_text,
            conversation=conversation,
            state_default_ttl=STATE_DEFAULT_TTL,
        )

        logger.info(
            "Enviando consolidación: %d mensajes, %d facts existentes, agente=%s",
            len(messages),
            len(existing_facts),
            agent_name,
        )

        # opencode-go API (OpenAI-compatible)
        response = await self.client.post(
            "https://opencode.ai/zen/go/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": CONSOLIDATION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
            },
        )
        response.raise_for_status()
        data = response.json()
        # Anthropic-compatible format: content is an array of blocks
        content = ""
        for block in data["content"]:
            if block.get("type") == "text":
                content += block["text"]

        if not content:
            logger.error("Consolidator devolvió contenido vacío (posible max_tokens insuficiente)")
            raise ValueError("Empty consolidation response")

        # Extract JSON from response (may be wrapped in ```json)
        parsed = _parse_llm_json(content)
        return ConsolidationResponse(**parsed)


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        lines.append(f"[{m.role}]: {m.content}")
    return "\n".join(lines)


def _format_existing_facts(facts: list[Fact]) -> str:
    if not facts:
        return "(no hay hechos previos)"
    lines = []
    for f in facts:
        lines.append(f"[id={f.id}] ({f.category.value}) {f.key}: {f.value}")
    return "\n".join(lines)


def _parse_llm_json(content: str) -> dict:
    """Extract JSON from LLM output, handling markdown code fences and common issues."""
    content = content.strip()
    # Remove markdown code fences
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    # Find the outermost JSON object
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start:end + 1]

    return json.loads(content)


def get_api_key() -> str | None:
    """Get the opencode go API key from environment or `pass`."""
    import os
    # Check environment variables first
    for var in ("ALAMBIQUE_API_KEY", "OPENCODE_API_KEY"):
        if val := os.environ.get(var):
            logger.info("API key cargada desde variable de entorno (%s)", var)
            return val.strip()

    try:
        result = subprocess.run(
            ["pass", "show", "apikeys/alambique"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning("pass show apikeys/alambique falló: %s", result.stderr.strip())
    except FileNotFoundError:
        logger.warning("pass no está instalado.")
    except Exception as e:
        logger.warning("Error al obtener API key: %s", e)
    return None
