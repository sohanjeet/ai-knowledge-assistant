# Project Overview

AI Knowledge Assistant: a full-stack RAG app. Users will upload documents and ask questions answered only from those documents, with source citations. Currently only the foundation exists (no document, auth, or AI features yet).

# Repository Structure

- `frontend/` – React + TypeScript + Vite app (`src/App.tsx`, `src/main.tsx`; still near Vite template state).
- `backend/` – Django project.
  - `backend/config/` – settings, root URLs (`/admin/`, `/api/`), wsgi/asgi.
  - `backend/apps/core/` – only app so far; `GET /api/health/` returns `{"status": "ok"}`.
  - `backend/requirements.txt`, `backend/.env.example`.
- `README.md` at root. No Docker, CI, or root-level tooling files yet.

# Current Stack

- Frontend: React 19, TypeScript, Vite, ESLint. Scripts: `dev`, `build`, `lint` (no test runner).
- Backend: Python, Django 6.1, Django REST Framework (JSON renderer/parser only), django-cors-headers, python-dotenv.
- Database: PostgreSQL is configured in settings via env vars (`psycopg`), but no models or migrations exist yet.
- Tests: Django test runner (`apps/core/tests.py`).

# Planned Stack (NOT implemented)

JWT auth, PDF/document processing, chunking, embeddings, pgvector, manual RAG, Redis, Celery, Docker, CI/CD, and possibly LangChain/LangGraph (only after manual RAG works).

# Architecture Rules

- Keep frontend and backend separate; they communicate only via the `/api/` HTTP API.
- Follow standard Django/DRF conventions; organize backend by domain under `backend/apps/<domain>/`.
- Keep views thin; put substantial business logic in service modules (`services.py`).
- Avoid unnecessary abstractions; reuse existing patterns before adding new ones.
- Preserve existing API contracts unless the task explicitly changes them.
- Avoid circular dependencies between apps/modules.

# Django Rules

- Use the ORM, not raw SQL, unless justified.
- Schema changes go through migrations; don't hand-edit generated migrations unless necessary and understood.
- Use DRF serializers for validation/serialization and DRF permissions for authorization.
- Secrets/config come from environment variables.
- Prevent N+1 queries (`select_related`/`prefetch_related`).
- Use `transaction.atomic` when operations must be atomic.

# Frontend Rules

- Use TypeScript; reuse existing components/patterns.
- Keep API access centralized (single API module/client), not scattered in components.
- Handle loading, error, and empty states.
- Avoid unnecessary global state.
- Don't duplicate backend business rules.

# Security Rules

- Never commit secrets or `.env`; never log credentials/tokens.
- Validate all user input and uploaded files (type, size, content).
- Enforce authentication where required and object ownership on every document/conversation access.
- Prevent cross-user access; never trust client-provided user/owner IDs.
- Don't leak internal errors or stack traces to clients.

# RAG Rules (future constraints)

Ingestion: Document → extraction → cleaning → chunking → embeddings → pgvector.
Query: Question → query embedding → vector retrieval → relevant chunks → context → LLM → answer + source citations.

- Implement and verify manual RAG before introducing LangChain/LangGraph.
- Preserve document/page/chunk metadata.
- Test retrieval independently before relying on LLM generation.
- Answers must be grounded in retrieved context and return sources.
- If evidence is insufficient, say so; never invent an answer.
- Isolate external AI calls behind service abstractions; never expose provider API keys.

# Agent Workflow

1. Read AGENTS.md, then inspect relevant existing code before editing.
2. Understand existing patterns before introducing new ones.
3. Implement only the requested task/phase, with the smallest coherent change.
4. Don't build future roadmap features early or refactor unrelated working code.
5. Don't silently change API contracts.
6. Add/update tests for changed behavior.
7. Run relevant validation; report failures truthfully.
8. Never hide failing tests with workarounds or by disabling them.

# Validation

Run from `backend/`: `python manage.py check`, `python manage.py test` (needs the configured PostgreSQL available).
Run from `frontend/`: `npm run lint`, `npm run build` (`npm test` doesn't exist yet).
Never claim a command passed unless it ran successfully. If required infrastructure (e.g. PostgreSQL) is unavailable, say so explicitly.

# Dependency Rules

- Don't add packages unnecessarily; prefer existing dependencies; justify significant new ones.
- No LangChain/LangGraph, Redis/Celery, or pgvector before their phase.
- Keep `requirements.txt` / `package.json` updated when dependencies legitimately change.

# Environment Rules

- `.env` is never committed; maintain `backend/.env.example` with placeholders only.
- Avoid environment-specific hardcoding.

# Scope Control

"Phase X only" means only that phase. If a prerequisite change is required, make the minimum one and report it.

# Response Format

After implementation tasks, reply concisely:

FILES_CHANGED:
- ...

IMPLEMENTED:
- ...

VALIDATION:
- command: PASS/FAIL

ISSUES:
- None / ...

NEXT_PHASE:
- one concise recommendation

Don't output large unchanged code blocks unless asked.

# Learning Consideration

The owner knows MERN/Node.js and is learning Python/Django. Prefer straightforward, conventional Django over clever abstractions. When first introducing an important Django-specific pattern, keep the code readable and briefly explain the concept in the summary.
