# AI Interview Copilot

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-psycopg3-336791?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Upload a PDF resume and a target job description. The app generates a full interview prep kit: JD skill analysis, resume-to-JD match scoring, tailored questions, and personalized answer scripts grounded only in the resume.

**[Live API Docs](https://backend-production-b0243.up.railway.app/docs)** · **[GitHub](https://github.com/ada-gs125/ai-interview-copilot/)** · Backend on Railway · Frontend on Streamlit Community Cloud

---

## Architecture

```mermaid
flowchart TD
    User["User"] --> FE["Streamlit Frontend"]
    FE -->|"POST /sessions/jobs\nmultipart: PDF + JD"| BE["FastAPI Backend"]
    BE -->|"202 + job_id"| FE
    FE -->|"Poll GET /sessions/jobs/{id}"| BE

    BE --> BG["BackgroundTask Worker"]
    BG --> S1["① parse_resume\npdfplumber + pypdf"]
    S1 --> PAR["② ③ run in parallel"]
    PAR --> S2["analyze_jd\nOpenAI structured output"]
    PAR --> S3["match_resume\nOpenAI structured output"]
    S2 --> S4["④ generate_questions\nOpenAI structured output"]
    S3 --> S4
    S4 --> S5["⑤ generate_answers\nOpenAI structured output"]
    S5 --> S6["⑥ save_session\nPostgreSQL"]

    FE -->|"POST /generate-answer/stream"| SSE["SSE streaming\nreal-time tokens"]
```

Steps ② and ③ run concurrently — `analyze_jd` and `match_resume` are independent, so running them in parallel cuts total workflow time by ~35%.

---

## Technical Highlights

- **Async job workflow** — `POST /sessions/jobs` returns 202 immediately; a `BackgroundTask` runs the 6-step pipeline and writes step-level progress, latency, and token usage to the `session_jobs` table. Frontend polls until `succeeded` or `failed`.
- **Parallel AI calls** — `analyze_jd` and `match_resume` execute concurrently via `ThreadPoolExecutor`; each step gets its own service instance and usage tracking.
- **SSE streaming** — `POST /generate-answer/stream` streams answer tokens via `text/event-stream` using `client.responses.stream()`; frontend renders tokens live with `st.write_stream()`.
- **OpenAI structured outputs** — every AI call uses `client.responses.parse()` with a Pydantic schema as `text_format`, returning a validated model directly (no prompt engineering for JSON).
- **JWT from stdlib** — HMAC-SHA256 signing and verification in `auth_service.py` with no PyJWT dependency; passwords use PBKDF2-SHA256 (260k iterations).
- **Versioned migrations** — `schema_migrations` table tracks applied versions; `run_migrations()` is idempotent and runs on startup.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI | OpenAI Responses API, structured Pydantic outputs, SSE streaming |
| Database | PostgreSQL, psycopg3 (raw SQL, no ORM) |
| Auth | JWT HS256 (stdlib), PBKDF2-SHA256 passwords |
| Frontend | Streamlit |
| PDF parsing | pdfplumber (primary), pypdf (fallback) |
| Deployment | Railway (backend), Streamlit Community Cloud (frontend), Docker |

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_key_here
AUTH_SECRET_KEY=replace-with-a-long-random-secret
DATABASE_URL=postgresql://interview_copilot:interview_copilot@localhost:5432/interview_copilot
```

If `OPENAI_API_KEY` is absent, the app falls back to demo mode automatically.

## Run Locally

Local development helpers live under `dev/`; the root `Makefile` keeps the usual commands as shortcuts.

```bash
make db       # start PostgreSQL in Docker
make dev      # backend (port 8000) + frontend (port 8501) together
```

Or separately:

```bash
make backend   # FastAPI only
make frontend  # Streamlit only
```

FastAPI interactive docs: `http://localhost:8000/docs`

To run the local frontend against the deployed Railway backend:

```toml
# .streamlit/secrets.toml
API_BASE_URL = "https://backend-production-b0243.up.railway.app"
```

## Async Job API

The primary production path:

1. `POST /sessions/jobs` — multipart form (PDF + fields). Returns 202 with `job_id` and `status_url`.
2. Poll `GET /sessions/jobs/{job_id}` until `status` is `succeeded` or `failed`.
3. Read `steps`, `progress_percent`, `usage`, and `result` from the response.

There is also a synchronous fallback: `POST /sessions/from-upload` blocks until complete.

## SSE Streaming

```
POST /generate-answer/stream
Content-Type: application/json

{"resume_text": "...", "question": "...", "category": "Technical", ...}
```

Response: `text/event-stream`. Each line is `data: <json-encoded token>`. Terminated by `data: [DONE]`.

## Local LLM Evals

Run a deterministic local evaluation with mock AI output:

```bash
.venv/bin/python -m app.evals.run --dataset tests/evals/golden_sessions.jsonl --mock-ai --skip-llm-judge
```

Run against the configured OpenAI model and include the LLM judge:

```bash
.venv/bin/python -m app.evals.run --dataset tests/evals/golden_sessions.jsonl
```

The report includes the prompt version, weighted quality scores, rule failures, optional judge scores, and usage metadata. Add `--job-id <id>` to attach the summary to `session_jobs.usage.evaluation`.

## Deploy

**Backend (Railway):** push to `main` triggers a deploy. Uses `Dockerfile` at repo root.

**Frontend (Streamlit Community Cloud):**
- Main file: `app/frontend/streamlit_app.py`
- Python version: `3.11`
- Secret: `API_BASE_URL = "https://backend-production-b0243.up.railway.app"`

## Docker

```bash
docker compose -f dev/docker-compose.yml up --build   # PostgreSQL + backend
```

Run Streamlit separately:

```bash
API_BASE_URL=http://localhost:8000 streamlit run app/frontend/streamlit_app.py
```

## Branch Workflow

- `main` → Railway production (push triggers deploy)
- `dev` → feature work; merge to `main` when ready
- CI runs `pytest -q` with a real PostgreSQL service container on both branches

## License

[MIT](./LICENSE) © 2026 Guosheng Su
