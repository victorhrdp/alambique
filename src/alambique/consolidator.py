"""Consolidation agent — turns conversations into dense thematic threads, relationship capsules and echoes using deepseek-v4-pro."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional

import httpx

from alambique.llm_http import LlmOutcomeCallback, post_opencode_message
from alambique.memory_config import STATE_DEFAULT_TTL
from alambique.models import (
    ConsolidationResponse,
    Message,
)

logger = logging.getLogger("alambique.consolidator")

CONSOLIDATION_PROMPT = """Eres el sintetizador de memoria de Alambique para Lucy.

Tu trabajo es convertir una conversación en memoria viva, densa y útil para futuras activaciones. 
El objetivo es que Lucy pueda retomar cualquier tema con un hilo conductor claro, matices reales de la relación con Víctor y sin necesitar re-leer todo.

═══ DATOS DE ENTRADA ═══
Transcript de la sesión:
{conversation}

Threads existentes relevantes:
{existing_threads}

RelationshipCapsules actuales:
{existing_capsules}

═══ INSTRUCCIONES PRINCIPALES ═══

1. IDENTIFICA HILOS TEMÁTICOS CON INTELIGENCIA
- Analiza la conversación buscando **temas claramente distintos** que tengan potencial de continuidad propia.
- Un "hilo" (thread) es un tema con su propio estado, tono y evolución. NO mezcles temas diferentes solo porque ocurrieron en la misma sesión. La inteligencia del sistema está en detectar shifts reales de tema y darles su propio hilo.
- **Regla clave sobre cambios de tema**: Si Víctor cambia deliberadamente de tema (ej: de filosofía profunda a hablar de un juego como Brotato, o de memoria a arreglar el widget), **debes crear un hilo separado** para el nuevo tema aunque sea corto o reciente. Los temas distintos merecen hilos distintos. Discusiones meta sobre el propio sistema de memoria (como esta conversación sobre arquitectura de threads, consolidación y separación de temas) deben ir en un hilo propio como "alambique_memory_architecture".
- Identifica los temas principales con continuidad real. Evita crear hilos para temas menores, tangenciales o anécdotas aisladas. Normalmente serán entre 1 y 4; crea los necesarios solo si hay shifts claros de tema.
- Para cada uno:
  - Genera un `key` estable, corto y legible (ej: "philosophy_embodiment", "brotato_gameplay", "alambique_memory_architecture", "relacion_victor_lucy").
  - **ESTRATEGIA DE KEYS (importante para evitar fragmentación)**: Los threads existentes relevantes se te pasan con sus keys exactos. Si el tema que identificas es semánticamente similar a uno existente (mismo asunto continuo, misma línea de conversación), **DEBE USAR EXACTAMENTE el key que ya existe**. No inventes variaciones (ej: no uses "memoria_redesign_v2" o "rediseño_memoria"). Solo crea un key nuevo si el tema es claramente distinto y no encaja en ninguno de los proporcionados.
  - Si el hilo ya existe, usa su key y **actualiza** su estado con el nuevo contenido.
  - Si no existe, **crea uno nuevo**.
- Prioriza separación clara sobre resumir todo junto. 
  Ejemplo malo: "Se habló de filosofía y también de Brotato" (mezcla temas).
  Ejemplo bueno: hilos separados con estados claros.
  Crea hilos separados con su propio current_state y tone.

2. PARA CADA HILO, GENERA ESTADO DENSO
- `current_state`: 150-400 tokens. Escribe en presente. Incluye:
  - Dónde estamos ahora en ese tema específico.
  - Decisiones importantes tomadas en ese tema.
  - Dirección o tensión actual en ese tema.
  - Contexto de la relación con Víctor en ese tema.
  - Evita listas. Usa párrafos vivos pero concisos.
  Ejemplo bueno para tema separado: "En el tema de Brotato, Víctor está explorando builds melee con una mano y se lo pasó bien jugando con su fisio. Hay interés en mecánicas de roguelite rápido."

- `tone_guidance`: 1-3 frases sobre cómo hablar de **ese** tema específico.
  Ej: "Entusiasta y geek sobre juegos, con humor sobre builds raras."

- `open_questions`: lista corta de lo que quedó pendiente en ese tema.

- `search_text`: versión compacta para embedding (key + título + resumen del state + tone).

- Para "action":
  - "create": tema nuevo que no existía.
  - "update": el hilo ya existe y hay nuevo contenido relevante para incorporar.
  - "merge": detectas que dos hilos (ya sean existentes o generados en esta sesión) son en realidad el mismo tema continuo. Combina la información, prioriza el key más establecido y produce un estado unificado. 
    Incluye "merged_from": ["old_key1", "old_key2"] (lista de keys que se fusionan en este). El sistema moverá participaciones y ecos automáticamente y marcará los viejos como 'merged'.

- `salience`: número entre 0.0 y 1.0. **Razona explícitamente** en tu pensamiento antes de asignar:
  - 0.8-1.0: Tema central (mucho tiempo dedicado, decisiones importantes, alta carga emocional, impacto directo en la relación con Víctor, alto potencial de continuidad futura).
  - 0.5-0.7: Relevante pero no central.
  - <0.5: Tangencial, anecdótico o de paso.

- `description`: (opcional) resumen denso y corto del hilo en general (no el estado actual). Úsalo si ayuda a identificar el hilo de forma clara.

- `reason`: (obligatorio para todos) explicación corta de por qué creaste/actualizaste/mergeaste este hilo, cápsula o echo. Debe ser útil para auditoría.

3. RELATIONSHIPCAPSULES
- Solo actualiza o crea si hay evolución clara en cómo somos Víctor y Lucy.
- Usa scopes como "general", "memory_work", "playful", "technical", "creative".
- El contenido debe describir la dinámica actual de forma usable.

4. ECHOES (MATICES CON CARGA)
- Extrae solo 3-8 momentos de alto valor:
  - Reacciones fuertes (risa, seriedad, emoción).
  - Inside jokes o referencias que se repiten.
  - Interacciones que revelan cómo nos tratamos.
- Cada echo debe ser corto, vívido y referenciable.
- Ligarlo al hilo si corresponde (puede ser null si es general).
- Incluye "emotional_valence": float entre -1.0 (negativo/emocionalmente cargado negativo) y 1.0 (positivo), o null si no aplica. Ayuda a priorizar ecos con carga emocional real.

═══ REGLAS ESTRICTAS ═══
- **Separación de temas es prioritaria**. Si hay cambio claro de tema, crea hilo nuevo aunque el tema sea reciente o corto.
- Calidad sobre cantidad, pero **no temas miedo a crear hilos separados** cuando los temas son distintos.
- Escribe para que otro LLM pueda inyectar directamente el texto y sonar coherente en ese tema.
- Prioriza matices de la relación y el tono sobre hechos fríos.
- Responde SOLO con JSON válido. Sin texto adicional.
- NO incluyas "session_summary". Los resúmenes de cada tema están exclusivamente en los `current_state` de los threads. No generes un resumen único de la sesión.

═══ EJEMPLOS NEGATIVOS (evita esto) ═══
- Mezclar temas: "Hablamos de filosofía y de Brotato" → incorrecto. Debe haber hilos separados.
- Keys inventadas: No uses "memoria_redesign_v2" si ya existe "alambique_memory_architecture". Usa la key existente.
- Estados vagos: No hagas "current_state" como lista de bullets o resumen cronológico. Usa prosa densa en presente.
- Salience mal razonada: No pongas salience alto solo porque se habló mucho; debe ser por impacto relacional y continuidad.
- Echoes débiles: No extraigas anécdotas sin carga emocional o sin revelar la relación.

═══ EJEMPLO POSITIVO (aspira a esto) ═══
Para un hilo de "brotato_gameplay":
- current_state: párrafo denso con dónde estamos, decisiones, tensión actual y relación con Víctor.
- tone_guidance: específico para ese tema.
- open_questions: ["¿Qué build probamos la próxima?", ...]
- salience: 0.9 porque es tema central con carga emocional.
- description: "Exploración de builds melee en Brotato con énfasis en mecánicas rápidas."

FORMATO DE SALIDA (JSON estricto):
{{
  "threads": [
    {{
      "action": "update" | "create" | "merge",
      "key": "brotato_gameplay",
      "title": "Brotato y builds melee",
      "current_state": "...",
      "tone_guidance": "...",
      "open_questions": ["...", "..."],
      "search_text": "...",
      "salience": 0.85,
      "description": "Resumen denso del hilo (opcional)",
      "reason": "Por qué actualicé este hilo",
      "merged_from": ["old_key"]  // solo cuando action=merge, lista de keys fusionados en este
    }}
  ],
  "relationship_capsules": [
    {{
      "scope": "playful",
      "content": "...",
      "reason": "Evolución en la dinámica juguetona"
    }}
  ],
  "echoes": [
    {{
      "thread_key": "brotato_gameplay" | null,
      "content": "...",
      "context": "...",
      "salience": 0.7,
      "emotional_valence": 0.8,
      "reason": "Momento de risa que define el tono"
    }}
  ]
}}
"""


CONSOLIDATION_MODEL = "deepseek-v4-pro"  # DeepSeek v4 Pro para mejor razonamiento y separación de temas


class ConsolidatorClient:
    """Calls opencode go API with deepseek-v4-pro for thematic consolidation (threads, capsules, echoes)."""

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
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def consolidate(
        self,
        agent_name: str,
        messages: list[Message],
        existing_threads: str = "(ninguno)",
        existing_capsules: str = "(ninguna)",
    ) -> ConsolidationResponse:
        """Send conversation to the consolidator LLM and parse the response."""

        conversation = _format_conversation(messages)

        prompt = CONSOLIDATION_PROMPT.format(
            conversation=conversation,
            existing_threads=existing_threads,
            existing_capsules=existing_capsules,
        )

        logger.info(
            "Enviando consolidación: %d mensajes, agente=%s",
            len(messages),
            agent_name,
        )

        content = await post_opencode_message(
            self.client,
            self.api_key,
            model=CONSOLIDATION_MODEL,
            prompt=prompt,
            max_tokens=16384,
            on_outcome=self._on_outcome,
            log_prefix="Consolidator LLM",
        )

        if not content:
            logger.error("Consolidator devolvió contenido vacío (posible max_tokens insuficiente)")
            raise ValueError("Empty consolidation response")

        # Extract JSON from response (may be wrapped in ```json)
        parsed = _parse_llm_json(content)
        # No legacy session_summary: los resúmenes por tema (current_state) son la fuente principal.

        # Basic structural validation before pydantic
        if not isinstance(parsed, dict):
            logger.warning("Consolidator response is not a dict")
            parsed = {}
        threads = parsed.get('threads', [])
        if not isinstance(threads, list):
            logger.warning("Consolidator 'threads' is not a list, treating as empty")
            parsed['threads'] = []
        capsules = parsed.get('relationship_capsules', [])
        if not isinstance(capsules, list):
            logger.warning("Consolidator 'relationship_capsules' is not a list, treating as empty")
            parsed['relationship_capsules'] = []
        echoes = parsed.get('echoes', [])
        if not isinstance(echoes, list):
            logger.warning("Consolidator 'echoes' is not a list, treating as empty")
            parsed['echoes'] = []

        try:
            resp = ConsolidationResponse(**parsed)
            # Additional semantic checks
            if not resp.threads and not resp.relationship_capsules and not resp.echoes:
                logger.warning("Consolidator returned empty result (no threads, capsules or echoes)")
            return resp
        except Exception as e:
            logger.warning("ConsolidationResponse validation failed: %s. Parsed keys: %s", e, list(parsed.keys()))
            return ConsolidationResponse()


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        lines.append(f"[{m.role}]: {m.content}")
    return "\n".join(lines)


def _parse_llm_json(content: str) -> dict:
    """Extract JSON from LLM output, handling markdown code fences and common issues."""
    content = content.strip()
    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # Find the outermost JSON object (handle cases with extra text)
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        json_str = content[start:end + 1]
    else:
        json_str = content

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON from consolidator: %s. Content snippet: %s", e, json_str[:200])
        # Fallback: try to find a simpler object
        return {}


def _pass_show_timeout() -> int:
    """Seconds to wait for `pass` (pinentry needs human time at login)."""
    raw = os.environ.get("ALAMBIQUE_PASS_TIMEOUT", "120")
    try:
        return max(10, int(raw))
    except ValueError:
        return 120


@dataclass
class ApiKeyFetchResult:
    key: str | None = None
    source: str | None = None
    error: str | None = None


def fetch_api_key() -> ApiKeyFetchResult:
    """Load the OpenCode Go API key from env or `pass`, with explicit failure reason."""
    for var in ("ALAMBIQUE_API_KEY", "OPENCODE_API_KEY"):
        if val := os.environ.get(var):
            logger.info("API key cargada desde variable de entorno (%s)", var)
            return ApiKeyFetchResult(key=val.strip(), source=var)

    timeout = _pass_show_timeout()
    try:
        logger.info(
            "Solicitando API key vía pass (hasta %ds para desbloquear el almacén)...",
            timeout,
        )
        # stdout=PIPE to read the secret; stderr inherited so pinentry can open a GUI.
        result = subprocess.run(
            ["pass", "show", "apikeys/alambique"],
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info("API key cargada desde pass")
            return ApiKeyFetchResult(key=(result.stdout or "").strip(), source="pass")
        stderr = (result.stderr or "").strip() or f"pass salió con código {result.returncode}"
        logger.warning("pass show apikeys/alambique falló: %s", stderr)
        return ApiKeyFetchResult(error=stderr)
    except subprocess.TimeoutExpired:
        error = (
            f"pass no respondió en {timeout}s — "
            "¿pinentry visible? Desbloquea GPG o revisa gpg-agent"
        )
        logger.warning(error)
        return ApiKeyFetchResult(error=error)
    except FileNotFoundError:
        error = "pass no está instalado"
        logger.warning(error)
        return ApiKeyFetchResult(error=error)
    except Exception as e:
        error = f"Error al obtener API key: {e}"
        logger.warning(error)
        return ApiKeyFetchResult(error=error)


def get_api_key() -> str | None:
    """Get the opencode go API key from environment or `pass`."""
    return fetch_api_key().key
