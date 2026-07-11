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
import os
import signal
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
from starlette.responses import Response
from starlette.routing import Route

from alambique.consolidator import get_api_key
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


TOOL_DEFINITIONS = [
    Tool(
        name="session_start",
        description=(
            "Inicia una sesión de memoria de Lucy. Llámala al inicio de cada conversación. "
            "En Grok: client='grok' y workspace=<cwd>. "
            "En Antigravity CLI: client='antigravity_cli' y workspace=<cwd>."
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
            "Finaliza la sesión, sincroniza el transcript externo si hay binding guardado "
            "y encola consolidación. Devuelve queued y pending_consolidation."
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
        name="memory_forget",
        description="Olvida un hecho (soft-delete).",
        inputSchema={
            "type": "object",
            "properties": {
                "fact_id": {"type": "integer"},
                "key": {"type": "string"},
            },
        },
    ),
    Tool(
        name="memory_export",
        description="Exporta facts y sesiones en formato JSON.",
        inputSchema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "default": "json"},
            },
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
        name="memory_reembed",
        description="Genera embeddings faltantes en vec0_facts para hechos activos (confidence>0). Úsalo cuando memory_health reporte hechos sin embedding.",
        inputSchema={
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Si true, solo lista hechos sin embedding sin generar vectores.",
                },
                "fact_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Re-embedar solo estos IDs (deben carecer de embedding).",
                },
            },
        },
    ),
    Tool(
        name="memory_deduplicate",
        description="Busca hechos semánticamente duplicados (similitud >= 0.85) y opcionalmente los fusiona.",
        inputSchema={
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Si true, solo reporta pares sin fusionar.",
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

    # Init tool handler
    api_key = get_api_key()
    if api_key:
        logger.info("API key cargada (modo online)")
    else:
        logger.warning("API key no disponible — modo offline")

    handler_tools = ToolHandler(db, ollama, api_key=api_key)
    await handler_tools.start_background_tasks()

    # Create MCP server
    server = Server("alambique")
    _register_handlers(server, handler_tools)

    # Shutdown handler
    def shutdown(signum=None, frame=None):
        logger.info("Señal recibida, cerrando...")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_shutdown(ollama, handler_tools, db))
        except RuntimeError:
            logger.warning("Sin event loop activo; shutdown diferido al cierre del proceso")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if transport == "stdio":
        async with stdio_server() as (reader, writer):
            logger.info("Servidor MCP listo (stdio)")
            await server.run(reader, writer, server.create_initialization_options())
    else:
        import contextlib
        from collections.abc import AsyncIterator
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.middleware.cors import CORSMiddleware
        from starlette.routing import Router

        session_manager = StreamableHTTPSessionManager(server)

        @contextlib.asynccontextmanager
        async def lifespan(app_starlette: Starlette) -> AsyncIterator[None]:
            async with session_manager.run():
                yield

        async def handle_well_known(request):
            return Response(
                content=json.dumps({
                    "resource": f"http://127.0.0.1:{port}/sse",
                    "authorization_servers": [],
                    "scopes_supported": []
                }),
                media_type="application/json",
                status_code=200
            )

        # Grok MCP client calls POST /sse (no trailing slash). Mount alone 307→/sse/ breaks clients.
        mcp_sse_app = _StreamableHTTPASGIApp(session_manager)

        router = Router(
            routes=[
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
                Route("/.well-known/oauth-protected-resource", endpoint=handle_well_known, methods=["GET", "OPTIONS"]),
                Route("/.well-known/oauth-protected-resource/sse", endpoint=handle_well_known, methods=["GET", "OPTIONS"]),
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

        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
        srv = uvicorn.Server(config)
        logger.info(f"Servidor MCP listo (SSE en http://127.0.0.1:{port})")
        await srv.serve()


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
        "memory_recall": lambda args: tools.memory_recall(
            query=args["query"],
        ),
        "memory_search": lambda args: tools.memory_search(
            args["query"],
        ),
        "memory_forget": lambda args: tools.memory_forget(
            fact_id=args.get("fact_id"),
            key=args.get("key"),
        ),
        "memory_export": lambda args: tools.memory_export(args.get("format", "json")),
        "memory_status": lambda args: tools.memory_status(),
        "memory_health": lambda args: tools.memory_health(),

        "memory_reembed": lambda args: tools.memory_reembed(
            dry_run=args.get("dry_run", False),
            fact_ids=args.get("fact_ids"),
        ),
        "memory_deduplicate": lambda args: tools.memory_deduplicate(
            dry_run=args.get("dry_run", True),
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
