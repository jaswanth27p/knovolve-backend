# Backend (Knovolve)

FastAPI + SQLAlchemy 2.x + Alembic + Postgres + Celery + redis + LangChain. This directory (`backend/`) is its OWN nested git repo — the top-level `Knovolve` repo gitignores it entirely. Run all git commands from inside `backend/`, not the top-level repo.

## Required per plan implementation

Every implementation plan (via `superpowers:subagent-driven-development` or `superpowers:executing-plans`) MUST run the type checker at least once before the plan is considered done — in its own final/hygiene task if the plan has one, otherwise as a last step before wrapping up:

```
cd backend && .venv/bin/pyright
```

Do this even if the plan's own text doesn't mention it. Pytest passing is not sufficient evidence of type correctness — this codebase has shipped test files with real `reportOptionalMemberAccess`/`reportReturnType` errors that pytest never catches (e.g. `db.get(Model, id)` returns `Model | None`, accessed without narrowing). Before trusting pyright's output, confirm `[tool.pyright]` in `pyproject.toml` has `venvPath`/`venv` set — without it pyright silently fails to resolve any installed dependency and reports hundreds of `reportMissingImports` false positives that bury the real signal.

When pyright surfaces errors: fix ones in files your plan created or touched. Pre-existing errors in files you didn't touch are out of scope for the plan — note them, don't silently ignore them, but don't scope-creep into fixing unrelated debt either.

## Commands

- Tests: `cd backend && OPENCODE_API_KEY=x OPENROUTER_API_KEY=x JWT_SECRET=x .venv/bin/pytest -q` (add `CELERY_TASK_ALWAYS_EAGER=1` for tasks that dispatch Celery work)
- Type check: `cd backend && .venv/bin/pyright`
- Migrations: `cd backend && .venv/bin/alembic revision --autogenerate -m "..."` / `.venv/bin/alembic upgrade head`
- Infra: `docker compose up -d` (Postgres + Redis + MinIO)

## Conventions

- Business logic lives in `app/services/<domain>.py` — routes (`app/routes/*.py`) are thin HTTP adapters: parse params, auth, call a service function, map HTTP status, done. No `select()`/`db.query`/redis/`.delay()`/agent imports directly in a route body.
- Services raise `HTTPException` directly (same pattern routes used to). Services never touch FastAPI's `Response` object — if a route needs to override a status code based on the service's result (e.g. `200` vs the decorator's `202` default), the route reads the result's own `status` field and sets `response.status_code` itself.
- Celery task objects are called directly in tests (`some_task(args)`, not `.delay(...)`) — the `@celery_app.task` decorator leaves the plain function callable; `.delay()` is only for real dispatch. This is also why `.delay()` call sites carry `# pyright: ignore[reportFunctionMemberAccess]` — pyright sees the plain function signature, not the Task object the decorator swaps in at runtime.
- No real LLM/embedding/Redis/S3 calls in the test suite — every external call is mocked. `unittest.mock.patch` targets the symbol where it's *used* (the owning service module), not where it's originally defined.
- Alembic autogenerate has a recurring false positive: it proposes dropping the LangGraph checkpoint tables (`checkpoint_writes`, `checkpoint_blobs`, `checkpoints`, `checkpoint_migrations`) because those are created by `PostgresSaver.setup()`, not SQLAlchemy. Remove those `op.drop_table(...)` calls by hand before applying a new migration.

## Working tree hazards

This repo is sometimes worked on by more than one agent/session at once, directly on `main`, without a worktree (by explicit user choice). If `git status` shows files you don't recognize as part of your own task:
- Investigate before touching anything — it's very likely someone else's in-progress, uncommitted work, not stray cruft.
- Stage and commit only the exact files your task names. Never `git add -A` or `git add .`.
- If a required import points at a module that doesn't exist yet on disk, don't assume it's broken — check whether it's mid-rename by someone else before "fixing" it.
