# Session Notes — Fricktrade (2026-01-07)

## Trigger
Market scheduler kept most services down because `healthwatch.market_shutdown` required every configured venue to be open (`market.open_mode: all`).

## Diagnosis
- Verified via `app.utils.market.is_market_open` that NYSE, Nasdaq, and Borsa Italiana report open status at the current time (UTC `2026-01-07T16:33:12`).
- Healthwatch respects `market.open_mode: all`, so the stack only woke when all venues were simultaneously open, which never happened yet.

## Remediation
- Switched `market.open_mode` to `any` in `config/config.yaml` so the stack wakes when at least one venue is live (`config/config.yaml:28`).
- Rebuilt and redeployed the services with `docker compose up -d --build` to ensure every container restarted under the new gate.

## Follow-up
- Watch the scheduler on the next open market window to confirm it no longer stops the stack while any venue is trading.
