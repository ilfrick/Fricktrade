<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Fricktrade Documentation

**Trading retired 2026-08-20.** The repository is retained as engineering infrastructure and a
research platform. See `AGENTS.md` for the operational state and the decision record.

## Living documents

| Document | What it is |
|---|---|
| [`../AGENTS.md`](../AGENTS.md) | **Authoritative operational document** — current config values, strategy logic, risk parameters, LLM flow, runbook, pitfalls |
| [`PLAN-2026-Q4.md`](PLAN-2026-Q4.md) | **The development plan.** 8 phases, ~55 tasks with acceptance criteria, dependencies, milestones |
| [`audit/2026-11-14-sistema.md`](audit/2026-11-14-sistema.md) | System documentation as-built: what actually runs vs what config declares, the 37-reason decision funnel, 10 documentation↔code divergences |
| [`audit/2026-11-14-bug.md`](audit/2026-11-14-bug.md) | Bug and inefficiency inventory — ~60 verified findings with severity, liveness, file:line and fix |
| [`audit/2026-11-14-letteratura.md`](audit/2026-11-14-letteratura.md) | Applicable literature: validation statistics, sizing, execution, crypto factors, reference architectures |

## Archived

[`archive/`](archive/) holds the pre-v3.0 planning material (37 documents, Dec 2025 – Mar 2026:
architecture, deployment, strategies, brokers, execution, risk, learning, backtesting, ops).
It predates the strategic reset and describes features that are disabled or removed. It is kept
for historical reference only — **do not use it to reason about current behaviour**.

## Where to start

- **"What does this system actually do?"** → `audit/2026-11-14-sistema.md`
- **"What is broken?"** → `audit/2026-11-14-bug.md`, section *Priorità*
- **"What should I work on?"** → `PLAN-2026-Q4.md`, section *Ordine e dipendenze*
- **"How do I run it?"** → `AGENTS.md`, section *Operational Runbook*
