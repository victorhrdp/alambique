"""Human-readable labels for internal warning codes (widget / daemon status)."""

from __future__ import annotations

# Mirrors PREFERENCE_NOISE_PREFIXES in memory_maintenance.py
_CONSOLIDATION_PREFIX_LABELS: dict[str, str] = {
    "architecture_": "arquitectura del código",
    "task_": "tareas de desarrollo",
    "integration_": "integraciones",
    "testing_": "pruebas",
    "system_": "sistema",
    "protocol_": "protocolos",
    "feature_": "funcionalidades",
    "infrastructure_": "infraestructura",
    "comparison_": "comparativas",
    "workflow_memory_": "flujos de memoria",
    "widget_": "widget",
}

_WARNING_LABELS: dict[str, str] = {
    "ollama_unavailable": "Ollama no responde",
    "offline_mode": "Sin API key (modo limitado)",
    "pending_consolidation": "Hay sesiones pendientes de consolidar",
    "embeddings_orphaned": "Faltan embeddings en algunos hechos",
    "stale_embeddings": "Hay vectores de hechos obsoletos",
    "orphan_session_embeddings": "Hay vectores de sesión huérfanos",
    "sessions_missing_embeddings": "Sesiones con resumen sin vector",
    "consolidation_filtered": "Se descartó un hecho ruidoso al consolidar",
    "consolidation_empty_result": "Consolidación vacía o no-JSON (no marcada como hecha)",
    "persona_offline_fallback": "Personalidad en modo offline",
    "persona_llm_failed": "No pude recomponer la personalidad",
    "vector_search_failed": "Búsqueda vectorial falló",

    "no_candidates_after_filter": "Sin candidatos tras filtrar",
    "summary_llm_failed": "Resumen de recall falló",
    "summary_offline": "Resumen de recall en modo offline",
    "summary_fallback_generic": "Resumen de recall genérico",
}


def _is_consolidation_filter(code: str) -> bool:
    return (
        code == "consolidation_filtered"
        or code.startswith(
            (
                "consolidation_filtered_prefix:",
                "consolidation_filtered_substring:",
                "consolidation_thread_unanchored:",
            )
        )
    )


def _consolidation_filter_detail(code: str) -> str:
    if code.startswith("consolidation_filtered_prefix:"):
        prefix = code.split(":", 1)[1]
        topic = _CONSOLIDATION_PREFIX_LABELS.get(prefix, "desarrollo")
        return f"nota de {topic}"
    if code.startswith("consolidation_filtered_substring:"):
        return "nota efímera de sesión"
    return "nota técnica de sesión"


def humanize_warning(code: str) -> str:
    """Map one internal warning code to a short Spanish phrase."""
    if code in _WARNING_LABELS:
        return _WARNING_LABELS[code]
    if code.startswith("consolidation_thread_unanchored:"):
        key = code.split(":", 1)[1]
        return f"hilo «{key}» no anclado a la sesión (update omitido)"
    if _is_consolidation_filter(code):
        return f"{_consolidation_filter_detail(code)} descartada al consolidar"
    if code.startswith("classification_warning:"):
        return "Posible clasificación dudosa en un hecho"
    if code.startswith("reembed_batch_failed:"):
        return "Error re-embediendo un lote de hechos"
    if code.startswith("reembed_failed:"):
        return "Error re-embediendo un hecho"
    return code.replace("_", " ")


def is_benign_consolidation_only(codes: list[str]) -> bool:
    """True when warnings are only post-consolidation noise filters (system still healthy)."""
    if not codes:
        return False
    unique = list(dict.fromkeys(codes))
    return all(_is_consolidation_filter(c) for c in unique)


def warning_message_level(codes: list[str]) -> str:
    """Widget severity: info for benign consolidation filters, warning otherwise."""
    if is_benign_consolidation_only(codes):
        return "info"
    return "warning"


def format_warnings_for_humans(codes: list[str]) -> str:
    """Build a widget-friendly status sentence from warning codes."""
    if not codes:
        return ""

    unique = list(dict.fromkeys(codes))
    consolidation = [c for c in unique if _is_consolidation_filter(c)]
    other = [c for c in unique if not _is_consolidation_filter(c)]

    if consolidation and not other:
        if len(consolidation) == 1:
            detail = _consolidation_filter_detail(consolidation[0])
            return (
                "Todo funciona. Al consolidar la última charla, descarté una "
                f"{detail} — ruido de sesión, no un recuerdo importante."
            )
        return (
            f"Todo funciona. Al consolidar, descarté {len(consolidation)} notas "
            "técnicas de sesión que no merecían quedarse en memoria."
        )

    parts = [humanize_warning(c) for c in unique]
    return f"Funcionando con avisos: {'; '.join(parts)}."