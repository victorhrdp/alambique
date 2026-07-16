"""Alambique MCP server — entry point and lifecycle.

Supports both stdio and SSE transports.
- stdio: for MCP clients that spawn the server (OpenCode, Antigravity)
- SSE: for persistent daemon mode (systemd --user)

Logging goes to stderr, never stdout (MCP protocol requirement).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    TextContent,
    Tool,
)
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from alambique.consolidator import fetch_api_key
from alambique.database import Database
from alambique.ollama_client import OllamaClient
from alambique.tools import ToolHandler
from alambique import __version__

logger = logging.getLogger("alambique")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

DB_PATH = Path.home() / ".local" / "share" / "alambique" / "alambique.db"
DEFAULT_SSE_PORT = 9042


class _StreamableHTTPASGIApp:
    """ASGI app for Streamable HTTP — Starlette must mount this, not a request handler."""

    def __init__(self, session_manager) -> None:
        self._session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:
        await self._session_manager.handle_request(scope, receive, send)


def build_sse_app(
    mcp_server: Server,
    handler_tools: ToolHandler,
    port: int,
):
    """Build the SSE ASGI app (CORS + routes). Returns (app, session_manager)."""
    import contextlib
    from collections.abc import AsyncIterator

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Router

    session_manager = StreamableHTTPSessionManager(mcp_server)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    async def handle_well_known(request):
        return Response(
            content=json.dumps({
                "resource": f"http://127.0.0.1:{port}/sse",
                "authorization_servers": [],
                "scopes_supported": [],
            }),
            media_type="application/json",
            status_code=200,
        )

    async def handle_status(request):
        status = await handler_tools.daemon_status(port=port)
        return Response(
            content=json.dumps(
                status.model_dump(mode="json"),
                default=str,
                ensure_ascii=False,
            ),
            media_type="application/json",
            status_code=200,
        )

    async def handle_close(request):
        """Simple HTTP endpoint for widget to manually close a session."""
        try:
            body = await request.json()
            session_id = body.get("session_id") if isinstance(body, dict) else None
            if not session_id:
                return Response(
                    content=json.dumps({"status": "error", "message": "session_id required"}),
                    media_type="application/json",
                    status_code=400,
                )
            result = await handler_tools.close_session(session_id=session_id)
            return Response(
                content=json.dumps(result, ensure_ascii=False),
                media_type="application/json",
                status_code=200,
            )
        except Exception as e:
            logger.warning("handle_close error: %s", e)
            return Response(
                content=json.dumps({"status": "error", "message": str(e)}),
                media_type="application/json",
                status_code=500,
            )

    async def handle_consolidate(request):
        """HTTP endpoint for the widget (or scripts) to manually trigger consolidation.
        POST body: {"session_id": "...", "force": false, "light": true}
        light=true uses lighter mode (skips similar-threads embed) to avoid CPU 100%.
        This allows a button to consolidate pending sessions without loops.
        """
        try:
            body = await request.json() if request.method != "GET" else {}
            if isinstance(body, dict) is False:
                body = {}
            session_id = body.get("session_id")
            force = bool(body.get("force", False))
            light = bool(body.get("light", False))
            if not session_id:
                return Response(
                    content=json.dumps({"status": "error", "message": "session_id required"}),
                    media_type="application/json",
                    status_code=400,
                )
            result = await handler_tools.consolidate_session(session_id=session_id, force=force, light=light)
            return Response(
                content=json.dumps(result, ensure_ascii=False),
                media_type="application/json",
                status_code=200,
            )
        except Exception as e:
            logger.warning("handle_consolidate error: %s", e)
            return Response(
                content=json.dumps({"status": "error", "message": str(e)}),
                media_type="application/json",
                status_code=500,
            )

    mcp_sse_app = _StreamableHTTPASGIApp(session_manager)

    router = Router(
        routes=[
            Route("/status", endpoint=handle_status, methods=["GET", "OPTIONS"]),
            Route("/close-session", endpoint=handle_close, methods=["POST", "OPTIONS"]),
            Route("/consolidate", endpoint=handle_consolidate, methods=["POST", "OPTIONS"]),
            Route(
                "/sse",
                endpoint=mcp_sse_app,
                methods=["GET", "POST", "DELETE", "OPTIONS"],
            ),
            Route(
                "/sse/",
                endpoint=mcp_sse_app,
                methods=["GET", "POST", "DELETE", "OPTIONS"],
            ),
            Route(
                "/.well-known/oauth-protected-resource",
                endpoint=handle_well_known,
                methods=["GET", "OPTIONS"],
            ),
            Route(
                "/.well-known/oauth-protected-resource/sse",
                endpoint=handle_well_known,
                methods=["GET", "OPTIONS"],
            ),
        ],
        redirect_slashes=False,
        lifespan=lifespan,
    )

    app = CORSMiddleware(
        router,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app, session_manager


async def _serve_sse(
    mcp_server: Server,
    handler_tools: ToolHandler,
    port: int,
) -> None:
    """Run uvicorn until SIGINT/SIGTERM; caller runs ``_shutdown`` in ``finally``."""
    import uvicorn

    app, _ = build_sse_app(mcp_server, handler_tools, port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    srv = uvicorn.Server(config)
    logger.info("Servidor MCP listo (SSE en http://127.0.0.1:%d)", port)
    await srv.serve()


TOOL_DEFINITIONS = [
    Tool(
        name="session_start",
        description=(
            "Inicia una sesión de memoria de Lucy. Llámala al inicio de cada conversación. "
            "En Grok: client='grok' y workspace=<cwd>. "
            "En Antigravity CLI: client='antigravity_cli' y workspace=<cwd>. "
            "En OpenCode: client='opencode' y workspace=<cwd>."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "persona_seed": {
                    "type": "string",
                    "description": "Personalidad inicial si aún no hay rasgos guardados. Se persiste como hecho personality.",
                },
                "client": {
                    "type": "string",
                    "description": "Cliente emisor (ej: grok, antigravity_cli, opencode). Enlaza el transcript externo al abrir.",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "ID externo de la conversación. Opcional si el servidor puede auto-detectarlo.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Ruta absoluta del workspace. Ayuda a resolver la sesión Grok activa.",
                },
            },
        },
    ),

    Tool(
        name="session_end",
        description=(
            "Finaliza la sesión, sincroniza el transcript externo si hay binding guardado. "
            "Activa automáticamente la consolidación en background (sin loops). "
            "Devuelve queued y pending_consolidation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "truncated": {"type": "boolean", "default": False},
                "conversation_id": {
                    "type": "string",
                    "description": "ID de la conversación (opcional, se auto-detecta si no se provee)",
                },
                "client": {
                    "type": "string",
                    "description": "Identificador del cliente o sistema emisor (ej: antigravity_cli, opencode, grok)",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="close_session",
        description=(
            "Cierra manualmente una sesión abierta (útil desde widget o para limpiar sesiones colgadas). "
            "Sincroniza transcript si hay binding y marca como cerrada."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID de la sesión a cerrar"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="consolidate_session",
        description=(
            "Fuerza la consolidación (threads, cápsulas, ecos) de una sesión específica. "
            "Widget: POST to http://localhost:9042/consolidate with {\"session_id\": \"...\", \"light\": true} for manual button. "
            "light=true salta la búsqueda de threads similares (menos CPU en Ollama bge-m3). "
            "Útil si session_end falló o para pendientes. Usa force=true para re-consolidar."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID de la sesión a consolidar"},
                "force": {
                    "type": "boolean",
                    "description": "Si true, resetea el flag consolidated y re-ejecuta aunque ya estuviera consolidada",
                    "default": False,
                },
                "light": {
                    "type": "boolean",
                    "description": "Modo ligero: evita embed + vector search de threads similares para reducir CPU 100%. Aún usa threads recientes.",
                    "default": False,
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="session_update",
        description="Actualiza la expresión y el estado de ánimo de Lucy de la sesión activa en la base de datos.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID de la sesión activa de Alambique"},
                "expression": {"type": "string", "description": "Expresión facial de Lucy"},
                "mood_text": {"type": "string", "description": "Descripción de su estado de ánimo"},
            },
            "required": ["session_id", "expression", "mood_text"],
        },
    ),
    Tool(
        name="memory_recall",
        description="Busca en la memoria semántica hechos y sesiones relacionadas con la consulta.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Pregunta o tema a buscar en la memoria."},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_search",
        description="Búsqueda textual forense en mensajes de conversaciones pasadas.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Texto a buscar."},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_status",
        description="Muestra estadísticas de la memoria.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="memory_health",
        description="Diagnóstico de salud del sistema de memoria: Ollama, API key, consolidación pendiente y embeddings huérfanos.",
        inputSchema={"type": "object", "properties": {}},
    ),

    Tool(
        name="memory_rebuild_vectors",
        description=(
            "Reconstruye vec0 (principalmente sessions; threads/caps/echoes en consolidación). "
            "Limpia huérfanos y regenera. dry_run=true por defecto."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Si true, solo reporta conteos sin modificar vec0.",
                },
                "facts_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "Ignorado (legacy facts system removed).",
                },
                "sessions_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "Reconstruir solo vec0_sessions.",
                },
            },
        },
    ),
    Tool(
        name="session_list",
        description="Lista sesiones ordenadas por fecha de creación, con filtro opcional por status.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Número máximo de sesiones a devolver (default 15)"},
                "status": {"type": "string", "description": "Filtrar por estado de la sesión ('open', 'closed', 'truncated')"},
            },
        },
    ),
    Tool(
        name="memory_context",
        description="Recupera los mensajes de una sesión pasada, paginados. Úsalo cuando memory_recall no tenga suficiente detalle y necesites leer la conversación literal.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID de la sesión a recuperar"},
                "offset": {"type": "integer", "description": "Desde qué mensaje empezar (default 0)"},
                "limit": {"type": "integer", "description": "Cuántos mensajes devolver (default 15, máx 30)"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="memory_expand_thread",
        description=(
            "Expande un hilo temático específico (por thread_key) devolviendo el estado completo, "
            "ecos adicionales y participaciones recientes. Úsalo on-demand cuando necesites más profundidad "
            "en un hilo activo sin inflar el contexto inicial. "
            "Opcional already_sent_echo_ids para evitar repetir ecos ya enviados."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "thread_key": {
                    "type": "string",
                    "description": "Clave del hilo (ej: philosophy_memory_continuity, philosophy_embodiment_robots).",
                },
                "already_sent_echo_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "IDs de ecos ya enviados en esta sesión para no repetirlos.",
                },
            },
            "required": ["thread_key"],
        },
    ),
]


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Alambique MCP server")
    p.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transporte MCP (default: stdio)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SSE_PORT,
        help=f"Puerto SSE (default: {DEFAULT_SSE_PORT})",
    )
    args = p.parse_args()
    asyncio.run(run(transport=args.transport, port=args.port))


async def run(transport: str = "stdio", port: int = DEFAULT_SSE_PORT) -> None:
    """Start Alambique MCP server."""
    logger.info("Alambique v%s arrancando (transport=%s)...", __version__, transport)

    # Init database
    db = Database(DB_PATH)
    db.connect()
    logger.info("Base de datos: %s", DB_PATH)

    # Init ollama
    ollama = OllamaClient()
    if await ollama.health():
        logger.info("Ollama conectado")
        await ollama.ensure_model()
    else:
        logger.warning("Ollama no disponible — búsqueda vectorial degradada")

    # Init tool handler (to_thread: pass may wait for pinentry at login)
    bootstrap = await asyncio.to_thread(fetch_api_key)
    handler_tools = ToolHandler(db, ollama)
    handler_tools.note_api_key_attempt(bootstrap)
    if handler_tools.online:
        logger.info("API key cargada (modo online)")
    else:
        logger.warning(
            "API key no disponible — modo offline (%s)",
            handler_tools._api_key_last_error or "sin detalle",
        )
    mcp_server = Server("alambique")
    _register_handlers(mcp_server, handler_tools)

    try:
        await handler_tools.start_background_tasks()

        if transport == "stdio":
            async with stdio_server() as (reader, writer):
                logger.info("Servidor MCP listo (stdio)")
                await mcp_server.run(
                    reader,
                    writer,
                    mcp_server.create_initialization_options(),
                )
        else:
            await _serve_sse(mcp_server, handler_tools, port)
    finally:
        logger.info("Cerrando Alambique...")
        await _shutdown(ollama, handler_tools, db)


def _register_handlers(server: Server, tools: ToolHandler) -> None:
    @server.list_tools()
    async def list_tools(request: ListToolsRequest) -> list[Tool]:
        return TOOL_DEFINITIONS

    dispatch = {
        "session_start": lambda args: tools.session_start(
            persona_seed=args.get("persona_seed"),
            client=args.get("client"),
            conversation_id=args.get("conversation_id"),
            workspace=args.get("workspace"),
        ),

        "session_end": lambda args: tools.session_end(
            session_id=args["session_id"],
            truncated=args.get("truncated", False),
            conversation_id=args.get("conversation_id"),
            client=args.get("client"),
        ),
        "close_session": lambda args: tools.close_session(
            session_id=args["session_id"],
        ),
        "consolidate_session": lambda args: tools.consolidate_session(
            session_id=args["session_id"],
            force=args.get("force", False),
            light=args.get("light", False),
        ),
        "session_update": lambda args: tools.session_update(
            session_id=args["session_id"],
            expression=args["expression"],
            mood_text=args["mood_text"],
        ),
        "memory_recall": lambda args: tools.memory_recall(
            query=args["query"],
        ),
        "memory_search": lambda args: tools.memory_search(
            args["query"],
        ),
        "memory_status": lambda args: tools.memory_status(),
        "memory_health": lambda args: tools.memory_health(),

        "memory_rebuild_vectors": lambda args: tools.memory_rebuild_vectors(
            dry_run=args.get("dry_run", True),
            facts_only=args.get("facts_only", False),
            sessions_only=args.get("sessions_only", False),
        ),
        "session_list": lambda args: tools.session_list(
            limit=args.get("limit", 15),
            status=args.get("status"),
        ),
        "memory_context": lambda args: tools.memory_context(
            session_id=args["session_id"],
            offset=args.get("offset", 0),
            limit=args.get("limit", 15),
        ),
        "memory_expand_thread": lambda args: tools.memory_expand_thread(
            thread_key=args["thread_key"],
            already_sent_echo_ids=args.get("already_sent_echo_ids"),
        ),
    }

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            handler = dispatch.get(name)
            if handler is None:
                return [TextContent(type="text", text=f"Tool desconocida: {name}")]
            result = await handler(arguments)
            return [TextContent(
                type="text",
                text=json.dumps(
                    result.model_dump() if hasattr(result, "model_dump") else result,
                    default=str,
                    ensure_ascii=False,
                ),
            )]
        except ValueError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        except Exception as e:
            logger.exception("Error en tool %s", name)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _shutdown(ollama: OllamaClient, tools: ToolHandler, db: Database) -> None:
    await tools.shutdown_open_sessions()
    await tools.stop_background_tasks()
    await ollama.close()
    db.close()


if __name__ == "__main__":
    main()
