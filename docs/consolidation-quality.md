# Calidad de consolidación — Qwen 3.7 Plus

Documento de referencia (2026-07-12). Diagnóstico y mitigaciones para la extracción
de hechos en Alambique.

## Diagnóstico

Qwen (`qwen3.7-plus` vía OpenCode Go) cumple bien en conversaciones **identitarias y
filosóficas**: categorías cerradas, `discard`, resúmenes temáticos, actualizaciones de
`personality` cuando el usuario corrige explícitamente.

Degrada en sesiones **de desarrollo**:

1. **Changelog como preferencia** — `architecture_*`, `task_*`, `integration_*` con
   confianza 0.95+.
2. **Confianza inflada** — casi todo sale ≥0.95; el prompt pide <0.4 para inferencias
   débiles pero no se usa.
3. **`update` agresivo** — fusiona sesiones recientes en hechos existentes no
   relacionados (ej. fact 77 contaminado con Grok Build vs Composer).
4. **Bleed de categorías** — avatar/teclado repartido entre personality, preference,
   possessions.
5. **Validación post-consolidación mínima** — solo heurística health→personal.

## Mitigaciones implementadas (v1)

Post-filtro en `memory_maintenance.filter_consolidation_fact()` antes de aplicar hechos:

- `personality`, `personal`, `state` → pasan (state sigue con TTL).
- `preference` con prefijos ruidosos → se descartan (`task_`, `architecture_`, …).
- Allowlist: `alambique_transcript_sync_architecture`.
- Tope de confianza 0.85 en preference con marcadores técnicos (`grok_`, `daemon_`, …).

## Pendiente (v2+)

- `update`/`merge` solo si `related_fact_id` existe y similitud semántica alta.
- Modo sesión dev: solo `session_summary`, sin facts.
- Revisión humana periódica o curado (`scripts/curate_memory.py`).

## Criterio de éxito

Tras N sesiones de desarrollo, el ratio `preference` ruidosas / total facts debe bajar
sin pérdida de los 9 hechos `personality` ni datos `personal`.