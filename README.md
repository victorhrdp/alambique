# Alambique — Memoria para Lucy

Memoria semántica y episódica para el asistente virtual **Lucy**. Destila conversaciones en hechos atómicos y mantiene la continuidad de su personalidad usando LLMs y búsqueda vectorial local.

## 📌 Arquitectura Simplificada

* **Base de Datos**: SQLite en modo WAL con la extensión `sqlite-vec` para búsqueda semántica local.
* **Embeddings**: Generados localmente con `ollama` y el modelo `bge-m3` (1024 dimensiones).
* **Razonamiento (LLM)**: API de OpenCode Go con el modelo `qwen3.7-plus`.
* **Ámbito**: Exclusivo para Lucy. No hay namespaces compartidos ni crosstalk con otros agentes; toda la base de datos pertenece a la relación entre Víctor y Lucy.

---

## 🛠️ Requisitos

* Python ≥ 3.11
* Ollama corriendo en `localhost:11434` con el modelo `bge-m3`.
* API Key de OpenCode Go configurada en el sistema (leída automáticamente vía `pass show apikeys/alambique`).

```bash
# Instalación de dependencias
pip install -e .
```

---

## 🚀 Uso como Servidor MCP

### Daemon SSE (producción)

Alambique corre como servicio systemd y expone MCP en `http://localhost:9042/sse`:

```bash
systemctl --user start alambique.service
systemctl --user status alambique.service
```

Grok CLI y otros clientes se conectan a esa URL. Tras reiniciar el servicio, abre una conversación nueva en Grok (el MCP de la sesión activa queda roto).

### Stdio (desarrollo)

Para clientes que arrancan el servidor como subproceso:

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

## 🧪 Pruebas Unitarias

El proyecto cuenta con una batería de **253 pruebas unitarias** que cubren el ciclo de vida completo de la base de datos, el cliente Ollama, la consolidación por LLM y las herramientas MCP.

```bash
# Ejecutar la suite completa
.venv/bin/pytest -v
```
