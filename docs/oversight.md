# Explainability & Oversight

## Purpose
Provide human-readable context for each decision so operators can audit why the
system traded, skipped, or resized orders.

## Decision Traces
Decision traces are written to JSONL and include:
- Strategy signals and orchestrator selection.
- Risk and sizing context (haircuts, exposure).
- Active model pointer metadata when RL strategies are used.

Key fields:
- `signals[]` with `name`, `action`, optional `features`.
- `orchestrator_selected`, `orchestrator_weights`, `orchestrator_mode`.
- `haircuts` with stress/liquidity details when applied.
- `model_active` from `learning.registry.active_path`.

## Audit & Compliance
- Audit logs capture full decision payloads; enable with `monitoring.audit.enabled`.
- Compliance exports write JSONL/CSV plus `.sha256` digests; enable with `monitoring.compliance.enabled`.

## Runbook
1. Check `model_active` to confirm the policy version in use.
2. Inspect `signals` and `orchestrator_selected` for strategy rationale.
3. Look at `haircuts` to see sizing reductions or liquidity caps.
4. Verify compliance digest (`.sha256`) for export integrity.
