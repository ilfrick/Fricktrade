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
- `/config/update`
- `/restart`
- `/ui`

## Configuration
`config/config.yaml`:
- `api.*` (if present)
- `logging.*` (API logs)
