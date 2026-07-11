# Alambique — Memoria para Lucy

Memoria semántica y episódica para el asistente virtual **Lucy** (v0.2.1 — Grok + Antigravity CLI). Destila conversaciones en hechos atómicos y mantiene la continuidad de su personalidad usando LLMs y búsqueda vectorial local.

## Arquitectura

| Capa | Tecnología | Rol |
|---|---|---|
| Persistencia | SQLite WAL + `sqlite-vec` | Sesiones, mensajes, hechos, embeddings |
| Embeddings | Ollama `bge-m3` (1024d) | Búsqueda semántica local |
| Razonamiento | OpenCode Go `qwen3.7-plus` | Consolidación y recall narrativo |
| Transcripts | `GrokCliProvider`, `AntigravityCliProvider` | Importación batch al cerrar sesión |
| Daemon | MCP SSE en `:9042` | 12 herramientas + watchdog + consolidación async |

Ámbito exclusivo Lucy — sin namespaces ni multi-agente.

---

## Requisitos

* Python ≥ 3.11
* Ollama en `localhost:11434` con modelo `bge-m3`
* API Key OpenCode Go (`pass show apikeys/alambique`)

```bash
pip install -e .
```

---

## Servidor MCP

### Daemon SSE (producción)

```bash
systemctl --user start alambique.service
systemctl --user status alambique.service
```

URL: `http://localhost:9042/sse`

Tras reiniciar el servicio o desplegar código nuevo, **recarga Grok** — el MCP de la sesión activa queda roto.

### Stdio (desarrollo)

```json
"mcp": {
  "alambique": {
    "type": "local",
    "command": ["/home/victor/Work/Agents/alambique/.venv/bin/python", "-m", "alambique"],
    "enabled": true
  }
}
```

---

## Flujo Grok CLI

Grok persiste el diálogo en `~/.grok/sessions/<cwd-encoded>/<conversation-id>/chat_history.jsonl`. **No existe `message_append`.**

```mermaid
flowchart LR
    A[session_start] -->|binding client+workspace| B[(sessions)]
    C[Grok escribe chat_history.jsonl] --> D[session_end / watchdog]
    D -->|GrokCliProvider| E[(messages)]
    E --> F[Consolidación LLM]
    F --> G[(facts + summary)]
```

1. **`session_start`** — `client="grok"`, `workspace=<cwd absoluto>`. Resuelve `conversation_id` vía `active_sessions.json` (normaliza rutas, desambigua pestañas por `opened_at`). Puede enlazar antes de que exista el fichero de transcript (`grok_transcript_pending`).
2. **Conversación** — Grok escribe el transcript en disco.
3. **`session_end`** — solo `session_id` de Alambique. Sincroniza mensajes user/assistant (sin tool_results), cierra y encola consolidación.

**Binding fallido** (`status: "error"`, `session_id: null`): no se crea sesión huérfana. Revisa `warnings` y reintenta.

**Reuso** — misma conversación Grok → reutiliza sesión open (`session_reused`).

**Red de seguridad** — watchdog (30 min inactividad) y shutdown del daemon sincronizan y cierran sesiones abiertas con binding.

### Antigravity CLI

Transcript en `~/.gemini/antigravity-cli/brain/<conversation-id>/.system_generated/logs/transcript_full.jsonl`.

1. **`session_start`** — `client="antigravity_cli"`, `workspace=<cwd absoluto>`. Resuelve `conversation_id` vía `history.jsonl` (última entrada del workspace por `timestamp`).
2. **Conversación** — Antigravity escribe el transcript en disco.
3. **`session_end`** — solo `session_id` de Alambique.

MCP en `~/.gemini/antigravity-cli/mcp_config.json` → `http://localhost:9042/sse`.

### Respuestas clave

| Tool | Campos útiles |
|---|---|
| `session_start` | `session_id`, `persona`, `client`, `conversation_id`, `session_reused`, `warnings`, `degraded` |
| `session_end` | `queued`, `pending_consolidation` |

---

## Herramientas MCP (12)

| Tool | Uso |
|---|---|
| `session_start` | Abre sesión, binding transcript, persona |
| `session_end` | Sync transcript, cierra, encola consolidación |
| `memory_recall` | Búsqueda semántica + resumen LLM |
| `memory_search` | FTS5 en mensajes |
| `memory_context` | Mensajes literales paginados (+ `client`/`conversation_id`) |
| `memory_status` | Estadísticas |
| `memory_health` | Diagnóstico (Ollama, API, cola, embeddings) |
| `memory_reembed` | Repara embeddings huérfanos |
| `memory_deduplicate` | Fusiona hechos duplicados (`dry_run` por defecto) |
| `session_list` | Sesiones recientes (+ binding) |
| `memory_forget` | Soft-delete de hecho |
| `memory_export` | Export JSON facts + sesiones |

---

## Operativa

* DB: `~/.local/share/alambique/alambique.db`
* No editar la DB con el daemon en marcha — parar el servicio antes.
* Consolidación corre en background; `memory_health` muestra la cola pendiente.

---

## Pruebas

**272** pruebas unitarias (DB, Ollama, consolidación, Grok CLI, herramientas MCP).

```bash
.venv/bin/pytest -v
```