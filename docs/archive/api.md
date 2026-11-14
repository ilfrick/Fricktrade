<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# API

## Purpose
Expose health/config endpoints and the web UI.

## Implementation
- FastAPI server: `app/api/server.py`.
- Entrypoint: `app/main.py` with `api` subcommand.

## Endpoints
- `/health`
- `/config`
- `/config/raw`
- `/config/schema`
- `/config/update`
- `/restart`
- `/`
- `/ui`

## Run
```bash
docker compose run --rm api
```

## Configuration
`config/config.yaml`:
- `api.*` (if present)
- `logging.*` (API logs)
