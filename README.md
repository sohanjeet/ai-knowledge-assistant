# AI Knowledge Assistant

React + TypeScript (Vite) frontend, Django + DRF backend, PostgreSQL database.

## Backend setup (Windows)

1. Create the database in PostgreSQL: `CREATE DATABASE ai_knowledge_assistant;`
2. Install the [pgvector](https://github.com/pgvector/pgvector) extension into the PostgreSQL server
   (once per machine; migrations then run `CREATE EXTENSION vector` and need a DB user allowed to do so).
   On Windows, build it from source in an **Administrator** "x64 Native Tools Command Prompt for VS"
   (needs Visual Studio's C++ build tools; use a pgvector release that supports your PostgreSQL version):
   ```bat
   set "PGROOT=C:\Program Files\PostgreSQL\18"
   cd %TEMP%
   git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /F Makefile.win
   nmake /F Makefile.win install
   ```
   Check: `SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';` returns a row.
3. Set up the environment:
   ```powershell
   cd backend
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env   # then edit .env with your DB password
   python manage.py migrate
   python manage.py runserver
   ```
4. Check it works: http://127.0.0.1:8000/api/health/ returns `{"status": "ok"}`.

Run tests: `python manage.py test` (needs pgvector installed; tests never call the embedding API).

Embeddings are generated only on request, never during upload, so `OPENAI_API_KEY` can stay empty
unless you embed documents (see `embed_document_chunks` in `backend/apps/documents/services.py`).

Semantic search: `POST /api/retrieval/search/` with `{"query": "...", "limit": 5, "document_ids": [1, 2]}`
(`limit` and `document_ids` are optional) returns the caller's own most relevant embedded chunks.
It embeds the query, so it needs `OPENAI_API_KEY`; no answer is generated yet.

## Frontend setup

```powershell
cd frontend
npm install
npm run dev   # http://localhost:5173
```
